from __future__ import annotations

import typer
from rich import print

from pqa.answerer import build_answer
from pqa.config import Settings
from pqa.intent_router import route_intent
from pqa.query_rewriter import expand_query
from pqa.retriever import definition_first_candidates
from pqa.retriever import extract_symbol_queries
from pqa.retriever import search
from pqa.vector_store import list_chunks
from pqa.vector_store import query_chunks

app = typer.Typer(add_completion=False)


def _is_noisy_path(path: str) -> bool:
    lower = path.lower()
    return (
        "/pb/" in lower
        or lower.endswith("_pb.go")
        or "/docs/" in lower
        or lower.endswith(".yaml")
        or lower.endswith(".yml")
        or lower.endswith(".json")
    )


def _needs_second_pass(evidence: list, definition_mode: bool) -> bool:
    if definition_mode:
        return False
    if not evidence:
        return True
    noisy_count = sum(1 for c in evidence if _is_noisy_path(c.path))
    return noisy_count >= max(1, len(evidence) // 2)


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


def _expand_core_logic_evidence(
    evidence: list,
    full_chunks: list,
    service: str | None,
    path_prefix: str | None,
    question: str,
    limit: int = 5,
) -> list:
    svc = service.upper() if service else None
    normalized_prefix = path_prefix.replace("\\", "/") if path_prefix else None
    expanded: list = []
    seen: set[str] = set()

    if svc:
        scoped = [c for c in full_chunks if c.service == svc]
    else:
        scoped = full_chunks
    if normalized_prefix:
        scoped = [c for c in scoped if c.path.startswith(normalized_prefix)]

    bridge_candidates = search(
        f"{question} handler usecase service",
        scoped,
        top_k=20,
        service_filter=service,
        path_prefix=path_prefix,
    )
    handler_pool = [c for c in bridge_candidates if "/handler/" in c.path.lower()]
    core_pool = [
        c
        for c in bridge_candidates
        if ("/usecase/" in c.path.lower()) or ("/service/" in c.path.lower())
    ]

    for pool in (handler_pool, core_pool):
        for c in pool:
            if c.id in seen:
                continue
            expanded.append(c)
            seen.add(c.id)
            break

    for c in bridge_candidates:
        if c.id in seen:
            continue
        if "/handler/" in c.path.lower() or "/usecase/" in c.path.lower() or "/service/" in c.path.lower():
            expanded.append(c)
            seen.add(c.id)
        if len(expanded) >= limit:
            break

    for c in evidence:
        if c.id in seen:
            continue
        expanded.append(c)
        seen.add(c.id)
        if len(expanded) >= limit:
            break

    has_core = any(("/usecase/" in c.path.lower()) or ("/service/" in c.path.lower()) for c in expanded)
    if not has_core:
        for c in scoped:
            if c.id in seen:
                continue
            path = c.path.lower()
            if "/usecase/" not in path and "/service/" not in path:
                continue
            if (c.kind or "").lower() not in {"func_def", "method_def"}:
                continue
            expanded.append(c)
            seen.add(c.id)
            break

    for c in scoped:
        if c.id in seen:
            continue
        if svc and c.service != svc:
            continue
        if normalized_prefix and not c.path.startswith(normalized_prefix):
            continue
        path = c.path.lower()
        if "/handler/" not in path and "/usecase/" not in path and "/service/" not in path:
            continue
        if (c.kind or "").lower() not in {"func_def", "method_def"}:
            continue
        expanded.append(c)
        seen.add(c.id)
        if len(expanded) >= limit:
            break
    return expanded[:limit]


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

    intent = route_intent(question, settings)
    definition_mode = intent.mode == "definition"
    core_logic_mode = intent.mode == "core-logic"
    architecture_mode = intent.mode == "architecture"
    rewritten_query, expanded_terms = expand_query(question, settings, service=service)
    candidates = query_chunks(settings, rewritten_query, top_k=60)
    symbol_queries = extract_symbol_queries(question) | extract_symbol_queries(rewritten_query)
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
    evidence = []
    full_chunks = None
    if definition_mode and symbol_queries:
        full_chunks = list_chunks(settings)
        evidence = definition_first_candidates(
            rewritten_query,
            full_chunks,
            service_filter=service,
            path_prefix=path_prefix,
            top_k=5,
        )
    if not evidence:
        evidence = search(
            rewritten_query,
            candidates,
            top_k=5,
            service_filter=service,
            path_prefix=path_prefix,
        )
    if symbol_queries:
        # Force symbol-first evidence when explicit code symbols are in question.
        if full_chunks is None:
            full_chunks = list_chunks(settings)
        symbol_hits = _symbol_priority_matches(
            full_chunks, symbol_queries, service=service, path_prefix=path_prefix, limit=5
        )
        if symbol_hits and not definition_mode:
            evidence = symbol_hits

    if not evidence and symbol_queries:
        # Symbol query fallback: run lexical rerank against full Chroma corpus.
        if full_chunks is None:
            full_chunks = list_chunks(settings)
        evidence = search(
            rewritten_query,
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
    if _needs_second_pass(evidence, definition_mode=definition_mode):
        if full_chunks is None:
            full_chunks = list_chunks(settings)
        filtered_chunks = [c for c in full_chunks if not _is_noisy_path(c.path)]
        second_pass = search(
            rewritten_query,
            filtered_chunks,
            top_k=8,
            service_filter=service,
            path_prefix=path_prefix,
        )
        if second_pass:
            evidence = second_pass
    if core_logic_mode:
        if full_chunks is None:
            full_chunks = list_chunks(settings)
        evidence = _expand_core_logic_evidence(
            evidence,
            full_chunks,
            service=service,
            path_prefix=path_prefix,
            question=question,
            limit=5,
        )
    primary_symbol = next(iter(symbol_queries), None)
    answer = build_answer(
        question,
        evidence,
        settings,
        definition_mode=definition_mode,
        core_logic_mode=core_logic_mode,
        architecture_mode=architecture_mode,
        target_symbol=primary_symbol,
        confidence="medium" if evidence else "none",
    )
    print(f"(query-expanded: {' '.join(expanded_terms[:12])})")
    print(answer)


if __name__ == "__main__":
    app()

