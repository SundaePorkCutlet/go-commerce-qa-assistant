from __future__ import annotations

from typing import Any
import os
import re

import chromadb
from chromadb.api.models.Collection import Collection

from pqa.config import Settings
from pqa.models import Chunk


def _prepare_local_cache(settings: Settings) -> None:
    cache_root = settings.data_dir / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    # Chroma ONNX embedding function downloads model files under cache directories.
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))


class LocalHashEmbeddingFunction:
    """Offline embedding for local/dev usage.

    This avoids model download/network dependency and keeps vectors deterministic.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*|[가-힣]{2,}")

    def name(self) -> str:
        return "local_hash_embedding_v1"

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in input:
            vec = [0.0] * self.dim
            tokens = [t.lower() for t in self.token_re.findall(text)]
            if not tokens:
                vectors.append(vec)
                continue
            for tok in tokens:
                idx = hash(tok) % self.dim
                vec[idx] += 1.0
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)


def _get_collection(settings: Settings) -> Collection:
    _prepare_local_cache(settings)
    if settings.chroma_mode == "http":
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    else:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(settings.chroma_path))

    return client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=LocalHashEmbeddingFunction(),
    )


def _to_metadata(chunk: Chunk) -> dict[str, Any]:
    metadata: dict[str, Any] = {"path": chunk.path}
    if chunk.service:
        metadata["service"] = chunk.service
    if chunk.symbol_hint:
        metadata["symbol_hint"] = chunk.symbol_hint
    if chunk.start_line is not None:
        metadata["start_line"] = int(chunk.start_line)
    if chunk.end_line is not None:
        metadata["end_line"] = int(chunk.end_line)
    return metadata


def upsert_chunks(settings: Settings, chunks: list[Chunk], batch_size: int = 200) -> None:
    collection = _get_collection(settings)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[_to_metadata(c) for c in batch],
        )


def query_chunks(settings: Settings, question: str, top_k: int = 8) -> list[Chunk]:
    collection = _get_collection(settings)
    result = collection.query(query_texts=[question], n_results=top_k)

    docs = result.get("documents", [[]])[0]
    ids = result.get("ids", [[]])[0]
    metas = result.get("metadatas", [[]])[0]

    chunks: list[Chunk] = []
    for cid, doc, meta in zip(ids, docs, metas):
        meta = meta or {}
        chunks.append(
            Chunk(
                id=cid,
                path=str(meta.get("path", "unknown")),
                text=doc,
                service=meta.get("service"),
                symbol_hint=meta.get("symbol_hint"),
                start_line=meta.get("start_line"),
                end_line=meta.get("end_line"),
            )
        )
    return chunks

