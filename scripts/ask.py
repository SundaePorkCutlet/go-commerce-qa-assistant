from __future__ import annotations

import typer
from rich import print

from pqa.answerer import build_answer
from pqa.config import Settings
from pqa.retriever import extract_symbol_queries
from pqa.retriever import search
from pqa.vector_store import list_chunks
from pqa.vector_store import query_chunks

app = typer.Typer(add_completion=False)


def _symbol_priority_matches(
    chunks: list,
    symbols: set[str],
    service: str | None,
    path_prefix: str | None,
    limit: int = 5,
) -> list:
    if not symbols:
        return []
    svc = service.upper() if service else None
    normalized_prefix = path_prefix.replace("\\", "/") if path_prefix else None

    exact: list = []
    partial: list = []
    for c in chunks:
        if svc and c.service != svc:
            continue
        if normalized_prefix and not c.path.startswith(normalized_prefix):
            continue
        if not c.symbol_hint:
            continue
        hint = c.symbol_hint.lower()
        if any(sym == hint or sym in hint for sym in symbols):
            if any(sym == hint for sym in symbols):
                exact.append(c)
            else:
                partial.append(c)

    ordered = exact + partial
    deduped = []
    seen_ids: set[str] = set()
    for c in ordered:
        if c.id in seen_ids:
            continue
        seen_ids.add(c.id)
        deduped.append(c)
        if len(deduped) >= limit:
            break
    return deduped


@app.command()
def main(
    question: str,
    service: str | None = typer.Option(
        default=None, help="서비스 필터 (ORDERFC/PAYMENTFC/PRODUCTFC/USERFC)"
    ),
    path_prefix: str | None = typer.Option(
        default=None, help="경로 prefix 필터 (예: ORDERFC/cmd/order)"
    ),
) -> None:
    settings = Settings.load()
    if not settings.use_chroma:
        raise RuntimeError(
            "Chroma-only mode: USE_CHROMA must be true. "
            "JSONL fallback is removed for production-style deployment."
        )

    candidates = query_chunks(settings, question, top_k=60)
    symbol_queries = extract_symbol_queries(question)
    for symbol in symbol_queries:
        candidates.extend(query_chunks(settings, symbol, top_k=30))

    deduped: list = []
    seen_ids: set[str] = set()
    for c in candidates:
        if c.id in seen_ids:
            continue
        seen_ids.add(c.id)
        deduped.append(c)
    candidates = deduped
    evidence = search(
        question,
        candidates,
        top_k=5,
        service_filter=service,
        path_prefix=path_prefix,
    )
    full_chunks = None
    if symbol_queries:
        # Force symbol-first evidence when explicit code symbols are in question.
        full_chunks = list_chunks(settings)
        symbol_hits = _symbol_priority_matches(
            full_chunks, symbol_queries, service=service, path_prefix=path_prefix, limit=5
        )
        if symbol_hits:
            evidence = symbol_hits

    if not evidence and symbol_queries:
        # Symbol query fallback: run lexical rerank against full Chroma corpus.
        if full_chunks is None:
            full_chunks = list_chunks(settings)
        evidence = search(
            question,
            full_chunks,
            top_k=5,
            service_filter=service,
            path_prefix=path_prefix,
        )
    if not evidence:
        raw = candidates
        if service:
            svc = service.upper()
            raw = [c for c in raw if c.service == svc]
        if path_prefix:
            normalized_prefix = path_prefix.replace("\\", "/")
            raw = [c for c in raw if c.path.startswith(normalized_prefix)]
        evidence = raw[:5]
    answer = build_answer(question, evidence, settings)
    print(answer)


if __name__ == "__main__":
    app()

