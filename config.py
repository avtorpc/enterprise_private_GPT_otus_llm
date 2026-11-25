from functools import lru_cache
from os import getenv
from pathlib import Path

import httpx

QDRANT_URL = getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = getenv("OLLAMA_URL", "http://ollama:11434")
INGEST_URL = getenv("INGEST_URL", "http://ingest:8000")
OLLAMA_MODEL = getenv("OLLAMA_MODEL", "qwen3:4b")
EMBEDDING_PATH = getenv("EMBEDDING_PATH", "/opt/models/multilingual-e5-small")
DOCUMENT_ROOT = Path(getenv("DOCUMENT_ROOT", "/documents"))
COLLECTION_ALIAS = "knowledge_active"
VECTOR_SIZE = 384


@lru_cache(maxsize=1)
def embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_PATH, device="cpu", local_files_only=True)


def embeddings(texts, prefix):
    return embedding_model().encode(
        [f"{prefix}: {text}" for text in texts],
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=16,
    ).tolist()


def active_collection(client: httpx.Client):
    response = client.get("/aliases")
    response.raise_for_status()
    for alias in response.json()["result"]["aliases"]:
        if alias["alias_name"] == COLLECTION_ALIAS:
            return alias["collection_name"]
    return None
