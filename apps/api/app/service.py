from __future__ import annotations

from pathlib import Path

from pqa.answerer import build_answer
from pqa.config import Settings
from pqa.intent_router import route_intent
from pqa.query_rewriter import expand_query
from pqa.retriever import definition_first_candidates
from pqa.retriever import extract_symbol_queries
from pqa.retriever import search
from pqa.vector_store import list_chunks
from pqa.vector_store import query_chunks


KNOWN_SERVICES = ("ORDERFC", "PAYMENTFC", "PRODUCTFC", "USERFC")


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


def _chunk_role_candidates(chunk) -> set[str]:
    path = chunk.path.lower()
    text = chunk.text.lower()
    roles: set[str] = set()

    if "/handler/" in path or "router" in text or "http." in text:
        roles.add("entry")
    if "/usecase/" in path or "/service/" in path:
        roles.add("core")
    if "/repository/" in path or "/repo/" in path or "gorm" in text or "mongo" in text:
        roles.add("repository")
    if "/kafka/" in path or "publish" in text or "consumer" in text or "event" in text:
        roles.add("message")
    if "/grpc/" in path or "/client/" in path or "httpclient" in text or "external" in text:
        roles.add("external_api")
    if "redis" in text or "/cache/" in path:
        roles.add("cache")
    if "validate" in text or "policy" in text or "rbac" in text:
        roles.add("validation")
    if "tracing" in path or "prometheus" in path or "logger" in text or "otel" in text:
        roles.add("observability")

    if not roles:
        roles.add("other")
    return roles


def _select_flexible_evidence(evidence: list, limit: int = 5) -> list:
    """Context expansion by role candidates.

    Do not depend only on top-k similarity.
    1) Secure entrypoint/core first according to question intent (core preferred).
    2) Expand optionally with related roles:
       repository/persistence, message/event, external API, cache,
       validation/policy, observability.
    3) Fill remaining slots by original ranking to keep recall.
    """
    if not evidence:
        return []

    selected: list = []
    seen_ids: set[str] = set()

    def push(chunk) -> None:
        if chunk.id in seen_ids:
            return
        seen_ids.add(chunk.id)
        selected.append(chunk)

    role_map: dict[str, list] = {
        "entry": [],
        "core": [],
        "repository": [],
        "message": [],
        "external_api": [],
        "cache": [],
        "validation": [],
        "observability": [],
        "other": [],
    }
    for c in evidence:
        for role in _chunk_role_candidates(c):
            role_map.setdefault(role, []).append(c)

    # required: core 우선 확보, entry 보강
    if role_map["core"]:
        push(role_map["core"][0])
    if role_map["entry"]:
        push(role_map["entry"][0])
    # core가 전혀 없으면 entry라도 확보
    if not selected and role_map["entry"]:
        push(role_map["entry"][0])
    # entry/core 둘 다 없으면 기존 상위 결과 유지
    if not selected and evidence:
        push(evidence[0])

    # optional roles
    for opt_role in ("repository", "message", "external_api", "cache", "validation", "observability"):
        if len(selected) >= limit:
            break
        if role_map[opt_role]:
            push(role_map[opt_role][0])

    # fill remainder by original ranking
    for c in evidence:
        if len(selected) >= limit:
            break
        push(c)

    return selected[:limit]


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


def _needs_second_pass(evidence: list, definition_mode: bool, architecture_mode: bool) -> bool:
    if definition_mode:
        return False
    if architecture_mode:
        return False
    if not evidence:
        return True
    noisy_count = sum(1 for c in evidence if _is_noisy_path(c.path))
    return noisy_count >= max(1, len(evidence) // 2)


def _has_core_logic_pair(evidence: list) -> bool:
    has_handler = any("/handler/" in c.path.lower() for c in evidence)
    has_core = any(("/usecase/" in c.path.lower()) or ("/service/" in c.path.lower()) for c in evidence)
    has_message = any(
        ("/kafka/" in c.path.lower())
        or ("outbox" in c.path.lower())
        or ("consumer" in c.path.lower())
        or ("publisher" in c.path.lower())
        for c in evidence
    )
    has_repository = any("/repository/" in c.path.lower() or "repository" in c.path.lower() for c in evidence)
    return (has_handler and has_core) or (has_message and (has_core or has_repository))


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


def _normalize_symbol(symbol: str) -> str:
    lower = symbol.lower()
    if "#part" in lower:
        lower = lower.split("#part", 1)[0]
    if "." in lower:
        lower = lower.split(".")[-1]
    return lower.strip()


def _estimate_confidence(
    evidence: list,
    *,
    service: str | None,
    symbol_queries: set[str],
    used_second_pass: bool,
    used_all_sweep: bool,
) -> str:
    if not evidence:
        return "none"

    score = 0.0

    count = len(evidence)
    if count >= 1:
        score += 1.0
    if count >= 3:
        score += 1.0

    roles: set[str] = set()
    for c in evidence:
        roles.update(r for r in _chunk_role_candidates(c) if r != "other")
    if len(roles) >= 2:
        score += 1.0
    if len(roles) >= 3:
        score += 0.5

    has_code = any(Path(c.path).suffix.lower() == ".go" for c in evidence)
    if has_code:
        score += 1.0
    else:
        score -= 0.5

    normalized_symbols = {_normalize_symbol(s) for s in symbol_queries}
    has_symbol_exact = any(
        (c.symbol_hint and _normalize_symbol(c.symbol_hint) in normalized_symbols)
        for c in evidence
    )
    if normalized_symbols and has_symbol_exact:
        score += 1.0

    if service:
        svc = service.upper()
        svc_hits = sum(1 for c in evidence if c.service == svc)
        if svc_hits >= max(1, len(evidence) // 2):
            score += 0.5
        else:
            score -= 0.5

    if used_second_pass:
        score -= 0.4
    if used_all_sweep:
        score -= 0.4

    if score >= 3.8:
        return "high"
    if score >= 2.2:
        return "medium"
    return "low"


def ask_question(
    question: str, service: str | None = None, path_prefix: str | None = None
) -> tuple[str, str, str]:
    settings = Settings.load()
    if not settings.use_chroma:
        raise RuntimeError("USE_CHROMA must be true for API mode.")

    intent = route_intent(question, settings)
    definition_mode = intent.mode == "definition"
    core_logic_mode = intent.mode == "core-logic"
    architecture_mode = intent.mode == "architecture"
    rewritten_query, _expanded_terms = expand_query(question, settings, service=service)
    normalized_service = (service or "").strip().upper()
    # In ALL mode, never lock retrieval to a single service.
    effective_service = None if normalized_service in {"", "ALL"} else normalized_service
    retrieval_query = rewritten_query
    if architecture_mode:
        retrieval_query = f"{rewritten_query} architecture overview component flow docs readme"
    candidates = query_chunks(settings, retrieval_query, top_k=60)
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
    used_second_pass = False
    used_all_sweep = False
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

    # Architecture queries should not fail only because service-scoped evidence is sparse.
    if architecture_mode and not evidence:
        if full_chunks is None:
            full_chunks = list_chunks(settings)
        evidence = search(
            retrieval_query,
            full_chunks,
            top_k=8,
            service_filter=None,
            path_prefix=path_prefix,
        )

    if _needs_second_pass(
        evidence, definition_mode=definition_mode, architecture_mode=architecture_mode
    ):
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
            used_second_pass = True

    # In ALL mode, broaden recovery path when first retrieval misses.
    if not evidence and effective_service is None:
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
        used_all_sweep = True

    evidence = _select_flexible_evidence(evidence, limit=5)
    confidence = _estimate_confidence(
        evidence,
        service=effective_service,
        symbol_queries=symbol_queries,
        used_second_pass=used_second_pass,
        used_all_sweep=used_all_sweep,
    )
    # Guardrail: core-logic answers without handler+usecase/service pair must not look "high confidence".
    if core_logic_mode and not _has_core_logic_pair(evidence):
        confidence = "low"

    primary_symbol = next(iter(symbol_queries), None)
    answer = build_answer(
        question,
        evidence,
        settings,
        definition_mode=definition_mode,
        core_logic_mode=core_logic_mode,
        architecture_mode=architecture_mode,
        target_symbol=primary_symbol,
        confidence=confidence,
    )
    mode = intent.mode
    return answer, mode, confidence
