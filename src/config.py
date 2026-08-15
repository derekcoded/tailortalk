"""
Central configuration for TailorTalk.

Every tunable knob (paths, model names, fusion weights, top_k defaults)
lives here so the rest of the codebase never hardcodes magic numbers.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IMAGES_CSV = DATA_DIR / "images.csv"
IMAGE_CACHE_DIR = DATA_DIR / "image_cache"          # downloaded catalogue images
CHROMA_DIR = ROOT_DIR / "chroma_db"                  # persisted vector store
COLLECTION_NAME = "sarees"

IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# open_clip checkpoint. ViT-B-32 is a good speed/quality tradeoff for ~1000
# images on CPU. Swap to "ViT-L-14" / "openai" pretrained for higher quality
# if you have a GPU or don't mind a slower one-off indexing run.
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# ---------------------------------------------------------------------------
# Fine-grained similarity fusion
# ---------------------------------------------------------------------------
# A plain CLIP embedding captures "this is a saree" very well but is weak on
# the fine-grained cues that actually distinguish one saree from another
# (exact colour, border/pallu pattern, weave texture). We fuse three signals
# into a single vector before it goes into Chroma:
#
#   1. CLIP semantic embedding      -> overall garment style / drape / motif
#   2. HSV colour histogram         -> precise colour & colour-combination match
#   3. Edge/texture histogram (HOG) -> weave, border, print texture
#
# Each sub-vector is L2-normalised on its own, then scaled by the weights
# below and concatenated. Because every sub-vector is unit length, the
# weights directly control how much each signal contributes to cosine
# similarity in the fused space.
CLIP_WEIGHT = 0.55
COLOR_WEIGHT = 0.30
TEXTURE_WEIGHT = 0.15

COLOR_HIST_BINS = (8, 8, 8)   # H, S, V bins -> 512-dim histogram
HOG_RESIZE = (128, 128)       # image resized before HOG for a fixed-length descriptor

# ---------------------------------------------------------------------------
# Search / re-ranking
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 5
CANDIDATE_POOL_K = 20   # over-fetch this many nearest neighbours, then re-rank

# Re-ranking blends the fused-vector cosine score with a perceptual-hash
# distance computed on the *raw* query/candidate images. This catches cases
# where two sarees are semantically similar but visually a poor match (or
# vice versa) in a way pure vector search sometimes misses.
RERANK_FUSED_WEIGHT = 0.7
RERANK_PHASH_WEIGHT = 0.3

# ---------------------------------------------------------------------------
# LLM (agent) config
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

REQUEST_TIMEOUT = 20  # seconds, for downloading images from URLs
