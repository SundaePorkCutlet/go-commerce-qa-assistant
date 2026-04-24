from __future__ import annotations

from pqa.answerer import build_answer
from pqa.config import Settings
from pqa.query_rewriter import expand_query
from pqa.retriever import definition_first_candidates
from pqa.retriever import extract_symbol_queries
from pqa.retriever import is_core_logic_query
from pqa.retriever import is_definition_lookup_query
from pqa.retriever import search
from pqa.vector_store import list_chunks
from pqa.vector_store import query_chunks


KNOWN_SERVICES = ("ORDERFC", "PAYMENTFC", "PRODUCTFC", "USERFC")


def _infer_service_from_query(question: str, rewritten_query: str) -> str | None:
    q = f"{question} {rewritten_query}".lower()
    if any(k in q for k in ("login", "signin", "signup", "auth", "jwt", "user", "유저", "회원", "로그인", "인증")):
        return "USERFC"
    if any(k in q for k in ("checkout", "order", "주문", "장바구니", "idempotency", "멱등")):
        return "ORDERFC"
    if any(k in q for k in ("payment", "결제", "invoice", "paid", "refund", "환불")):
        return "PAYMENTFC"
    if any(k in q for k in ("product", "catalog", "inventory", "stock", "상품", "재고")):
        return "PRODUCTFC"
    return None


def _merge_dedup_chunks(chunks: list, limit: int = 5) -> list:
    merged: list = []
    seen_ids: set[str] = set()
    for c in chunks:
        if c.id in seen_ids:
            continue
        seen_ids.add(c.id)
        merged.append(c)
        if len(merged) >= limit:
            break
    return merged


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


def _symbol_priority_matches(chunks: list, symbols: set[str], service: str | None, path_prefix: str | None, limit: int = 5) -> list:
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


def ask_question(question: str, service: str | None = None, path_prefix: str | None = None) -> tuple[str, str]:
    settings = Settings.load()
    if not settings.use_chroma:
        raise RuntimeError("USE_CHROMA must be true for API mode.")

    definition_mode = is_definition_lookup_query(question)
    core_logic_mode = (not definition_mode) and is_core_logic_query(question)
    rewritten_query, _expanded_terms = expand_query(question, settings, service=service)
    effective_service = service or _infer_service_from_query(question, rewritten_query)
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
            service_filter=effective_service,
            path_prefix=path_prefix,
            top_k=5,
        )
    if not evidence:
        evidence = search(
            rewritten_query,
            candidates,
            top_k=5,
            service_filter=effective_service,
            path_prefix=path_prefix,
        )
    if symbol_queries:
        if full_chunks is None:
            full_chunks = list_chunks(settings)
        symbol_hits = _symbol_priority_matches(
            full_chunks, symbol_queries, service=effective_service, path_prefix=path_prefix, limit=5
        )
        if symbol_hits and not definition_mode:
            evidence = symbol_hits
    if not evidence:
        raw = candidates
        if effective_service:
            svc = effective_service.upper()
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
            service_filter=effective_service,
            path_prefix=path_prefix,
        )
        if second_pass:
            evidence = second_pass

    # In ALL mode, broaden recovery path when first retrieval misses.
    if not evidence and service is None:
        if full_chunks is None:
            full_chunks = list_chunks(settings)
        sweep_hits: list = []
        for svc in KNOWN_SERVICES:
            sweep_hits.extend(
                search(
                    rewritten_query,
                    full_chunks,
                    top_k=2,
                    service_filter=svc,
                    path_prefix=path_prefix,
                )
            )
        evidence = _merge_dedup_chunks(sweep_hits, limit=5)

    primary_symbol = next(iter(symbol_queries), None)
    answer = build_answer(
        question,
        evidence,
        settings,
        definition_mode=definition_mode,
        core_logic_mode=core_logic_mode,
        target_symbol=primary_symbol,
    )
    mode = "definition" if definition_mode else "core-logic" if core_logic_mode else "general"
    return answer, mode
