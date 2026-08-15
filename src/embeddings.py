"""
Turns a PIL image into the fused embedding vector used for similarity search.

Fusion = [ w_clip * CLIP(img), w_color * ColorHist(img), w_texture * HOG(img) ]

All three sub-vectors are L2-normalised before scaling, so cosine similarity
on the concatenated vector behaves like a weighted sum of three independent
cosine similarities. See config.py for the weights and the reasoning.
"""

from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from PIL import Image

import config

# Heavy imports (torch / open_clip / skimage / cv2) are done lazily inside
# the functions that need them so that lightweight parts of the codebase
# (e.g. the Streamlit UI importing this module) don't pay the import cost
# until an embedding is actually requested.


@lru_cache(maxsize=1)
def _load_clip():
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, preprocess, device


def clip_embedding(image: Image.Image) -> np.ndarray:
    """Semantic embedding capturing overall garment style/drape/motif."""
    import torch

    model, preprocess, device = _load_clip()
    image_t = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(image_t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy().astype("float32")


def color_histogram(image: Image.Image) -> np.ndarray:
    """
    HSV colour histogram. This is what actually pins down "same shade of
    maroon with a gold border" in a way a general-purpose CLIP embedding
    tends to blur together.
    """
    import cv2

    arr = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1, 2],
        mask=None,
        histSize=list(config.COLOR_HIST_BINS),
        ranges=[0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten()
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm
    return hist.astype("float32")


def texture_descriptor(image: Image.Image) -> np.ndarray:
    """
    HOG descriptor on a downsized grayscale image. Captures weave/border/
    print texture — the "fine detail" the assignment calls out — without
    being thrown off by exact pixel colour.
    """
    from skimage.feature import hog

    gray = np.array(image.convert("L").resize(config.HOG_RESIZE))
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat.astype("float32")


def fused_embedding(image: Image.Image) -> np.ndarray:
    """The single vector that gets stored in / queried against Chroma."""
    clip_vec = clip_embedding(image) * config.CLIP_WEIGHT
    color_vec = color_histogram(image) * config.COLOR_WEIGHT
    texture_vec = texture_descriptor(image) * config.TEXTURE_WEIGHT
    return np.concatenate([clip_vec, color_vec, texture_vec]).astype("float32")


def perceptual_hash(image: Image.Image):
    """Used only at re-rank time, not stored in the index."""
    import imagehash

    return imagehash.phash(image.convert("RGB"))


def load_image_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")
