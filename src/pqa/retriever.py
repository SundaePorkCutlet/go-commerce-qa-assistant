from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from pqa.models import Chunk


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*|[가-힣]{2,}")
SYMBOL_QUERY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DEFINITION_QUERY_HINT_RE = re.compile(
    r"(정의|선언|method|symbol|where\s+(defined|declared)|definition|declaration)",
    re.IGNORECASE,
)
CORE_LOGIC_QUERY_HINT_RE = re.compile(
    r"(핵심\s*로직|실제\s*처리|검증|상태\s*변경|흐름|어디서\s*하나요)",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def extract_symbol_queries(query: str) -> set[str]:
    symbols: set[str] = set()
    for token in SYMBOL_QUERY_RE.findall(query):
        # Keep likely code symbols (camel/Pascal/snake with enough length)
        if len(token) < 4:
            continue
        if any(ch.isupper() for ch in token) or "_" in token:
            symbols.add(token.lower())
    return symbols


def is_definition_lookup_query(query: str) -> bool:
    return bool(DEFINITION_QUERY_HINT_RE.search(query))


def is_core_logic_query(query: str) -> bool:
    if CORE_LOGIC_QUERY_HINT_RE.search(query):
        return True
    qtokens = tokenize(query)
    return any(t in {"핵심", "로직", "처리", "검증", "상태", "흐름"} for t in qtokens)


def normalize_symbol_name(symbol: str) -> str:
    base = symbol.lower()
    if "#part" in base:
        base = base.split("#part", 1)[0]
    if "." in base:
        base = base.split(".")[-1]
    return base.strip()


def definition_first_candidates(
    query: str,
    chunks: list[Chunk],
    service_filter: str | None = None,
    path_prefix: str | None = None,
    top_k: int = 5,
) -> list[Chunk]:
    if not is_definition_lookup_query(query):
        return []
    symbol_queries = extract_symbol_queries(query)
    if not symbol_queries:
        return []

    normalized_service_filter = service_filter.upper() if service_filter else None
    normalized_path_prefix = path_prefix.replace("\\", "/") if path_prefix else None
    target_symbols = {normalize_symbol_name(s) for s in symbol_queries}

    exact_defs: list[Chunk] = []
    symbol_refs: list[Chunk] = []
    docs: list[Chunk] = []
    for c in chunks:
        if normalized_service_filter and c.service != normalized_service_filter:
            continue
        if normalized_path_prefix and not c.path.startswith(normalized_path_prefix):
            continue
        symbol_norm = normalize_symbol_name(c.symbol_hint or "")
        kind = (c.kind or "").lower()

        if symbol_norm in target_symbols and kind in {"func_def", "method_def", "type_def"}:
            exact_defs.append(c)
            continue
        if symbol_norm in target_symbols:
            symbol_refs.append(c)
            continue
        docs.append(c)

    if exact_defs:
        return exact_defs[:top_k]
    if symbol_refs:
        return symbol_refs[:top_k]
    return docs[:top_k]


def expand_query_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    if "멱등" in expanded or "멱등성" in expanded:
        expanded.update(
            {
                "idempotency",
                "idempotent",
                "enable.idempotence",
                "acks",
                "dedup",
                "duplicate",
            }
        )
    if "트레이싱" in expanded:
        expanded.update({"tracing", "trace", "context"})
    if "감사로그" in expanded or "감사" in expanded:
        expanded.update({"audit", "log"})
    if "카프카" in expanded:
        expanded.update({"kafka", "producer", "consumer", "partition", "acks"})
    if "msa" in expanded or "마이크로서비스" in expanded or "서비스분리" in expanded:
        expanded.update({"microservice", "architecture", "component", "boundary", "domain"})
    if "msa" in expanded or "마이크로서비스" in expanded or "구조" in expanded:
        expanded.update(
            {
                "microservice",
                "architecture",
                "component",
                "boundary",
                "domain",
                "service",
                "readme",
                "docs",
            }
        )
    if "구성" in expanded or "아키텍처" in expanded:
        expanded.update({"overview", "component", "boundary", "interaction", "service"})
    if "메시지" in expanded or "메시징" in expanded:
        expanded.update({"message", "event", "kafka", "producer", "consumer"})
    return expanded


def tf(tokens: list[str]) -> Counter[str]:
    return Counter(tokens)


def cosine_sim(a: Counter[str], b: Counter[str]) -> float:
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def bm25_rerank(query_tokens: set[str], candidates: list[tuple[float, Chunk]], top_k: int) -> list[Chunk]:
    if not candidates:
        return []

    docs = [tokenize(c.text) for _, c in candidates]
    avgdl = mean(len(d) for d in docs) if docs else 1.0
    k1 = 1.5
    b = 0.75

    # document frequency for query terms only
    df: dict[str, int] = {}
    for term in query_tokens:
        df[term] = 0
    for d in docs:
        uniq = set(d)
        for term in query_tokens:
            if term in uniq:
                df[term] += 1

    n_docs = len(docs)
    reranked: list[tuple[float, Chunk]] = []
    for i, (base_score, chunk) in enumerate(candidates):
        terms = docs[i]
        tf_terms = Counter(terms)
        dl = max(1, len(terms))

        bm25 = 0.0
        for term in query_tokens:
            f = tf_terms.get(term, 0)
            if f == 0:
                continue
            dfi = df.get(term, 0)
            idf = math.log((n_docs - dfi + 0.5) / (dfi + 0.5) + 1.0)
            denom = f + k1 * (1 - b + b * (dl / avgdl))
            bm25 += idf * ((f * (k1 + 1)) / denom)

        # blend vector-like score + bm25 score
        final_score = (base_score * 0.55) + (bm25 * 0.45)
        reranked.append((final_score, chunk))

    reranked.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in reranked[:top_k]]


def search(
    query: str,
    chunks: list[Chunk],
    top_k: int = 5,
    service_filter: str | None = None,
    path_prefix: str | None = None,
) -> list[Chunk]:
    qtokens = expand_query_tokens(set(tokenize(query)))
    symbol_queries = extract_symbol_queries(query)
    qvec = tf(list(qtokens))
    scored: list[tuple[float, Chunk]] = []
    require_idempotency_signal = bool(
        {"idempotency", "idempotent", "enable.idempotence"} & qtokens
    )
    service_hint = None
    implementation_query = any(
        (t.startswith("어디") or t in {"구현", "검증", "where", "implemented"})
        for t in qtokens
    )

    for svc in ("orderfc", "paymentfc", "productfc", "userfc"):
        if svc in qtokens:
            service_hint = svc.upper()
            break
    normalized_service_filter = service_filter.upper() if service_filter else None
    normalized_path_prefix = path_prefix.replace("\\", "/") if path_prefix else None

    for c in chunks:
        if normalized_service_filter and c.service != normalized_service_filter:
            continue
        if normalized_path_prefix and not c.path.startswith(normalized_path_prefix):
            continue

        text_lower = c.text.lower()
        path_lower = c.path.lower()

        if require_idempotency_signal:
            idempotency_hints = (
                "idempotency",
                "idempotent",
                "enable.idempotence",
                "duplicate",
                "dedup",
                "acks=all",
                "acks",
            )
            if not any(h in text_lower or h in path_lower for h in idempotency_hints):
                # Do not hard-drop weakly related chunks; keep them with lower score.
                score_penalty = 0.55
            else:
                score_penalty = 1.0
        else:
            score_penalty = 1.0

        cvec = tf(tokenize(c.text))
        score = cosine_sim(qvec, cvec)
        if score <= 0:
            continue
        score *= score_penalty

        path = c.path
        suffix = Path(path).suffix.lower()
        if implementation_query and suffix != ".go":
            continue

        # Prefer backend source files over docs/config for code Q&A.
        if suffix == ".go":
            score *= 1.35
        elif suffix in {".md", ".yaml", ".yml", ".json"}:
            score *= 0.9

        if "/prometheus/" in path.lower() or "/promtail/" in path.lower():
            score *= 0.5

        # Prefer primary service directories.
        if path.startswith(("ORDERFC/", "PAYMENTFC/", "PRODUCTFC/", "USERFC/")):
            score *= 1.25
        if service_hint and path.startswith(service_hint + "/"):
            score *= 1.5

        # Strong boost when question includes implementation keywords.
        impl_keywords = {"idempotency", "token", "kafka", "tracing", "grpc", "audit", "repository"}
        if qtokens & impl_keywords:
            if suffix == ".go":
                score *= 1.2

        matched_key_terms = 0
        for key in ("idempotency", "repository", "usecase", "handler"):
            if key in qtokens and (key in text_lower or key in path_lower):
                matched_key_terms += 1
        if matched_key_terms > 0:
            score *= 1.0 + (0.35 * matched_key_terms)

        if c.symbol_hint:
            s_hint = c.symbol_hint.lower()
            if any(sym in s_hint for sym in symbol_queries):
                score *= 1.8
            if "idempotency" in qtokens and "idempotency" in s_hint:
                score *= 1.4
        kind = (c.kind or "").lower()
        if kind in {"func_def", "method_def", "type_def"}:
            score *= 1.25
        elif kind == "call_site":
            score *= 0.9
        elif kind in {"doc_section", "comment"}:
            score *= 0.8

        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    candidate_pool = scored[: max(top_k * 6, 20)]
    return bm25_rerank(qtokens, candidate_pool, top_k=top_k)

