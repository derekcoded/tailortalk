"""
Search pipeline: embed query image -> over-fetch candidates from Chroma ->
re-rank with a perceptual-hash signal -> return top_k matches.

This module has no LangChain / Streamlit dependency so it can be unit
tested and reused directly.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import requests
from PIL import Image

import config
import embeddings
import vector_store


@dataclass
class Match:
    name: str
    sku: str
    score: float
    image_url: str
    product_url: str
    retail_price: str = ""
    discounted_price: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sku": self.sku,
            "score": round(self.score, 4),
            "image_url": self.image_url,
            "product_url": self.product_url,
            "retail_price": self.retail_price,
            "discounted_price": self.discounted_price,
        }


def load_query_image(image_input: str) -> Image.Image:
    """
    image_input may be:
      - a local file path (Streamlit upload saved to disk)
      - a raw http(s) URL
    """
    if image_input.startswith("http://") or image_input.startswith("https://"):
        resp = requests.get(image_input, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        return embeddings.load_image_bytes(resp.content)
    with open(image_input, "rb") as f:
        return embeddings.load_image_bytes(f.read())


def _phash_similarity(query_img: Image.Image, candidate_local_path: str) -> float:
    """Returns a similarity in [0, 1], higher = more visually alike."""
    try:
        q_hash = embeddings.perceptual_hash(query_img)
        c_hash = embeddings.perceptual_hash(Image.open(candidate_local_path))
        # phash is a 64-bit hash; max hamming distance is 64.
        distance = q_hash - c_hash
        return 1.0 - (distance / 64.0)
    except Exception:  # noqa: BLE001
        return 0.5  # neutral fallback if a candidate image is unreadable


def search_similar(
    image_input: str,
    top_k: int = config.DEFAULT_TOP_K,
    candidate_pool_k: int = config.CANDIDATE_POOL_K,
) -> list[Match]:
    query_img = load_query_image(image_input)
    query_vec = embeddings.fused_embedding(query_img)

    collection = vector_store.get_collection()
    raw = vector_store.query(collection, query_vec.tolist(), top_k=candidate_pool_k)

    if not raw["ids"] or not raw["ids"][0]:
        return []

    ids = raw["ids"][0]
    metas = raw["metadatas"][0]
    distances = raw["distances"][0]  # cosine distance, lower = closer

    reranked = []
    for _id, meta, dist in zip(ids, metas, distances):
        fused_sim = 1.0 - dist  # cosine distance -> similarity
        phash_sim = _phash_similarity(query_img, meta.get("local_image_path", ""))
        blended = (
            config.RERANK_FUSED_WEIGHT * fused_sim
            + config.RERANK_PHASH_WEIGHT * phash_sim
        )
        reranked.append((blended, meta))

    reranked.sort(key=lambda x: x[0], reverse=True)
    top = reranked[:top_k]

    return [
        Match(
            name=meta.get("name", ""),
            sku=meta.get("sku", ""),
            score=score,
            image_url=meta.get("image_url", ""),
            product_url=meta.get("product_url", ""),
            retail_price=meta.get("retail_price", ""),
            discounted_price=meta.get("discounted_price", ""),
        )
        for score, meta in top
    ]
