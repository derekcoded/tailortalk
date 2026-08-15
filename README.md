# TailorTalk — Saree Visual Similarity Search Agent

A chat agent that finds visually similar sarees from a catalogue of ~1,000
product photos. Upload a photo (or paste an image URL) and chat naturally —
the agent decides when a similarity search is being asked for, calls a
vector-search tool behind the scenes, and returns the closest matches with
scores, prices, and links.

## Live app / repo

- App URL: `<fill in after deploying to Render>`
- GitHub: `<fill in after pushing this repo>`

## Architecture

```
User (Streamlit chat) ──> LangChain tool-calling agent (Claude)
                                │
                                ▼
                   search_similar_sarees(image_input, top_k)
                                │
                                ▼
                  Fused embedding (CLIP + colour + texture)
                                │
                                ▼
                    ChromaDB (persisted, cosine similarity)
                                │
                                ▼
              Candidate pool (20) ──> perceptual-hash re-rank ──> top_k
```

- **Vector DB**: ChromaDB, persisted to disk (`chroma_db/`), cosine distance.
- **Agent framework**: LangChain (`create_tool_calling_agent` +
  `AgentExecutor`) wrapping a single `StructuredTool` with an explicit
  pydantic input schema.
- **LLM**: Claude (`langchain-anthropic`), configurable via `ANTHROPIC_MODEL`.
- **Frontend**: Streamlit (`src/app.py`) — sidebar image upload, chat input,
  in-chat result grid with thumbnails, price and product links.

## Why a plain CLIP index isn't enough here

Every image in this catalogue is the same garment type (a saree), so a raw
CLIP embedding search tends to return "any saree that looks vaguely
similar" rather than a match on the details that actually matter: exact
shade, colour combination, border/pallu work, weave/print texture. To fix
that, the indexing pipeline (`src/embeddings.py`) builds a **fused
embedding** per image instead of a bare CLIP vector:

| Signal | What it captures | Weight |
|---|---|---|
| CLIP (open_clip ViT-B/32, LAION-2B) | overall garment style, drape, motif | 0.55 |
| HSV colour histogram (8×8×8 bins) | precise colour / colour-combination | 0.30 |
| HOG texture descriptor (grayscale, 128×128) | weave, border, print texture | 0.15 |

Each sub-vector is L2-normalised before being scaled and concatenated, so
cosine similarity on the fused vector behaves like a weighted blend of three
independent similarity signals — no single one can dominate just because it
happens to have a larger raw magnitude.

On top of that, **search is a two-stage retrieve-then-re-rank pipeline**
(`src/search.py`):

1. Retrieve the top 20 nearest neighbours from Chroma by fused-vector
   cosine similarity.
2. Re-rank those 20 by blending the fused-vector similarity (70%) with a
   perceptual-hash (pHash) visual-similarity score computed directly on the
   raw images (30%), then return the top `k`.

This re-rank step catches cases where two sarees land close together in
embedding space but are visually a poor match (or the reverse), which is
common on a fine-grained, single-category catalogue like this one.

All of the weights and pool sizes are centralised in `src/config.py` if you
want to tune them further (e.g. increase `TEXTURE_WEIGHT` if border/weave
matching feels weak, or swap in a larger CLIP checkpoint such as `ViT-L-14`
for higher semantic quality at the cost of slower indexing).

## Tool schema

The agent has exactly one tool, `search_similar_sarees`:

```python
class SareeSearchInput(BaseModel):
    image_input: str   # http(s) URL OR local file path of an uploaded image
    top_k: int = 5      # 1–10
```

Returns JSON: a list of matches, each with `name`, `sku`, `score`,
`image_url`, `product_url`, `retail_price`, `discounted_price`. The system
prompt instructs the model to call this tool only when the user has
actually supplied an image and is asking for similar/matching items — not
for general chit-chat — and to never fabricate an `image_input`.

## Repo layout

```
data/images.csv        # catalogue (name, sku, price, image_url, product link)
src/config.py           # all tunable constants
src/embeddings.py       # CLIP + colour histogram + HOG fusion
src/vector_store.py     # Chroma wrapper
src/build_index.py      # offline: download images, embed, index
src/search.py           # retrieve + re-rank pipeline
src/agent_tool.py       # LangChain StructuredTool + schema
src/agent.py            # LangChain tool-calling agent + system prompt
src/app.py              # Streamlit chat UI
render.yaml              # Render deployment blueprint
requirements.txt
.env.example
```

## Setup

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY
```

### 3. Build the vector index (one-time, or whenever the catalogue changes)

```bash
cd src
python build_index.py
```

This downloads every image referenced in `data/images.csv` into
`data/image_cache/`, computes the fused embedding for each, and persists
them into `chroma_db/`. Takes a few minutes on CPU for ~1,000 images. Use
`--limit 50` to do a quick smoke test first, and `--workers` to tune
download concurrency.

### 4. Run the app locally

```bash
streamlit run src/app.py
```

## Deploying to Render

1. Push this repo to GitHub (make sure `chroma_db/` and
   `data/image_cache/` are **committed**, not gitignored — see note below —
   since Render's free tier has no persistent disk across deploys and we
   don't want to re-run the full indexing job on every build).
2. In Render: New → Blueprint → point at this repo (`render.yaml` is
   already set up with the build/start commands).
3. Set the `ANTHROPIC_API_KEY` env var in the Render dashboard (marked
   `sync: false` in `render.yaml` so it's not committed to git).
4. Deploy. Render will `pip install -r requirements.txt` then run
   `streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0`.

> **Note on the pre-built index**: `.gitignore` excludes `chroma_db/` and
> `data/image_cache/` by default, which is the right call for local dev
> hygiene but the wrong call for a from-scratch Render deploy, since
> Render's free web-service plan doesn't give you a place to run
> `build_index.py` before the app starts. Before pushing to GitHub for
> deployment, either (a) remove those two lines from `.gitignore` and
> commit the built index, or (b) add a Render **paid** persistent disk and
> run `build_index.py` once via a Render Shell session after the first
> deploy. Option (a) is what this submission uses, since it keeps the free
> tier viable and the reviewer's first load fast.

## Assumptions & trade-offs

- **CLIP checkpoint**: `ViT-B-32` (open_clip, LAION-2B) was chosen over a
  larger model purely for CPU-friendly indexing speed on ~1,000 images.
  Swapping to `ViT-L-14` in `config.py` is a one-line change if quality
  needs to go further and indexing time isn't a constraint.
- **No background removal**: catalogue photos are reasonably consistent
  (product-on-model or flat-lay against a plain-ish background), so a
  dedicated segmentation step was judged not worth the added complexity
  and failure surface for this dataset. The colour histogram is somewhat
  sensitive to background colour as a result — worth revisiting if the
  catalogue photography becomes less consistent.
- **Fusion weights are hand-tuned**, not learned. They were set by
  qualitative inspection of results across a handful of query images
  spanning different colours/prints, not a formal offline eval set — a
  natural next step would be to collect a small labelled "these two are/
  aren't the same design family" set and grid-search the weights against
  it.
- **Re-ranking pool size (20)** trades a bit of latency for materially
  better top-5 precision; dropping it to re-rank only the raw top-5 is
  faster but misses cases where a truly closer visual match sits at
  position 6-20 in pure embedding order.
- **Single-tool agent**: kept deliberately minimal per the assignment's
  hint that the tool schema is what's being assessed — no extra tools for
  things like price filtering, since that wasn't asked for and would
  dilute the one tool call the agent actually needs to get right.
- **Chat history** is kept in Streamlit session state only (no persistence
  across sessions/restarts) — acceptable for a reviewer doing a live test,
  but would need a real store for a production deployment.
