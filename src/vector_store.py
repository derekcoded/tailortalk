"""
Thin wrapper around a persistent ChromaDB collection.

We store our own pre-computed fused vectors (see embeddings.py) rather than
letting Chroma compute embeddings itself, so we set embedding_function=None
and always pass `embeddings=` explicitly.
"""

from __future__ import annotations

from typing import Any

import chromadb

import config


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(client: chromadb.PersistentClient | None = None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert(
    collection,
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
):
    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)


def query(collection, embedding: list[float], top_k: int) -> dict:
    return collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["metadatas", "distances"],
    )


def count(collection) -> int:
    return collection.count()
