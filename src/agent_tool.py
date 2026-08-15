"""
The single tool the agent is allowed to call. Keeping the schema tight and
explicit is what the assignment hints at ("the tool schema the LLM calls...
decide the outcome here") — the LLM should only ever need to pass an image
reference and, optionally, how many results it wants.
"""

from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

import config
import search


class SareeSearchInput(BaseModel):
    image_input: str = Field(
        description=(
            "Either an http(s) URL pointing directly to an image, or the "
            "local file path of an image the user just uploaded. Do not "
            "pass free text here — only a real image reference."
        )
    )
    top_k: int = Field(
        default=config.DEFAULT_TOP_K,
        description="How many similar sarees to return, between 1 and 10.",
        ge=1,
        le=10,
    )


def _run_search(image_input: str, top_k: int = config.DEFAULT_TOP_K) -> str:
    """
    Returns a JSON string (not a Python object) because tool outputs get
    fed back into the LLM's context as text. The Streamlit layer separately
    re-runs / caches the same search to render actual image thumbnails, so
    the LLM only needs the structured facts (name, price, score, link) to
    talk about the results — it never has to describe the pixels itself.
    """
    matches = search.search_similar(image_input, top_k=top_k)
    if not matches:
        return json.dumps({"matches": [], "message": "No similar sarees were found."})
    return json.dumps({"matches": [m.to_dict() for m in matches]})


saree_similarity_tool = StructuredTool.from_function(
    func=_run_search,
    name="search_similar_sarees",
    description=(
        "Search the saree catalogue vector index for items that look "
        "visually similar to a given query image (matching on fabric, "
        "weave, print, colour combination, and border/pallu work). "
        "Call this whenever the user provides or references an image "
        "(an uploaded file path or an image URL) and asks to find similar, "
        "matching, or related sarees. Returns each match's name, SKU, "
        "similarity score, price, product page link and image URL."
    ),
    args_schema=SareeSearchInput,
)
