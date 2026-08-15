"""
One-time (re-run whenever the catalogue changes) indexing script.

    python src/build_index.py [--limit N] [--workers 8]

For every row in data/images.csv:
  1. Download the product image (skip / reuse if already cached on disk).
  2. Compute the fused embedding (CLIP + colour histogram + texture).
  3. Upsert into the persistent Chroma collection with product metadata
     (name, sku, price, image_url, product page link) attached.

Runs on CPU in a few minutes for ~1000 images; use --workers to raise the
download concurrency (embedding itself is done single-threaded since the
CLIP forward pass is the real bottleneck and batching it isn't worth the
added complexity at this dataset size).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

import config
import embeddings
import vector_store


def _cache_path_for(url: str) -> Path:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    ext = Path(url).suffix or ".jpg"
    return config.IMAGE_CACHE_DIR / f"{h}{ext}"


def _download(url: str) -> Path | None:
    dest = _cache_path_for(url)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        resp = requests.get(url, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] failed to download {url}: {exc}")
        return None


def load_catalogue(limit: int | None = None) -> list[dict]:
    with open(config.IMAGES_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only index first N rows (debugging)")
    parser.add_argument("--workers", type=int, default=8, help="parallel image downloads")
    args = parser.parse_args()

    rows = load_catalogue(args.limit)
    print(f"Loaded {len(rows)} catalogue rows from {config.IMAGES_CSV}")

    # Download images concurrently (I/O bound, so threads are fine here).
    print("Downloading images...")
    url_to_path: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download, row["image_url"]): row["image_url"] for row in rows}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            url = futures[fut]
            path = fut.result()
            if path:
                url_to_path[url] = path

    collection = vector_store.get_collection()

    print("Embedding + indexing...")
    batch_ids, batch_embeds, batch_meta = [], [], []
    BATCH = 32

    def flush():
        if batch_ids:
            vector_store.upsert(collection, batch_ids, batch_embeds, batch_meta)
            batch_ids.clear()
            batch_embeds.clear()
            batch_meta.clear()

    skipped = 0
    for row_index, row in enumerate(tqdm(rows)):
        url = row["image_url"]
        path = url_to_path.get(url)
        if path is None:
            skipped += 1
            continue
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] unreadable image {path}: {exc}")
            skipped += 1
            continue

        vec = embeddings.fused_embedding(image)

        sku = row.get("SKU", "").strip()
        unique_id = f"{sku}_{row_index}" if sku else f"item_{row_index}"
        batch_ids.append(unique_id)
        batch_embeds.append(vec.tolist())
        batch_meta.append(
            {
                "name": row.get("Name", ""),
                "sku": row.get("SKU", ""),
                "retail_price": row.get("Retail Price", ""),
                "discounted_price": row.get("Discounted Price", ""),
                "image_url": url,
                "product_url": row.get("Website Link", ""),
                "local_image_path": str(path),
            }
        )

        if len(batch_ids) >= BATCH:
            flush()
    flush()

    print(f"Done. Indexed {vector_store.count(collection)} images. Skipped {skipped} rows.")


if __name__ == "__main__":
    main()
