"""Embed chunks.json with all-MiniLM-L6-v2 and write to ChromaDB.

Spec (planning.md, Milestone 4):
  - Embedding model: sentence-transformers/all-MiniLM-L6-v2 (local, no API)
  - Vector store: ChromaDB (persistent, local)
  - Collection: neu_reviews
  - Per-chunk metadata: source, professor, course, chunk_index

Re-running this script wipes and rebuilds the collection — embedding all 27
chunks is cheap and idempotent reads are easier to debug than partial updates.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
CHUNKS_PATH = ROOT / "chunks.json"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "neu_reviews"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found — run `python ingest.py` first"
        )
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def to_metadata(chunk: dict) -> dict:
    # ChromaDB metadata values must be str/int/float/bool — convert None → ""
    return {
        "source": chunk["source"],
        "professor": chunk.get("professor") or "",
        "course": chunk.get("course") or "",
        "chunk_index": chunk["chunk_index"],
    }


def build_collection() -> None:
    chunks = load_chunks()
    print(f"loaded {len(chunks)} chunks from {CHUNKS_PATH.name}")

    print(f"loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print(f"embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Wipe-and-rebuild: simpler than upsert reasoning for a small corpus
    try:
        client.delete_collection(COLLECTION_NAME)
    except (ValueError, Exception):
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c['source']}::{c['chunk_index']}" for c in chunks]
    metadatas = [to_metadata(c) for c in chunks]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    print(f"wrote {collection.count()} embeddings → {CHROMA_DIR}/ (collection={COLLECTION_NAME})")


if __name__ == "__main__":
    build_collection()
