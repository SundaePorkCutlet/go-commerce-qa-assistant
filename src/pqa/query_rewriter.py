from __future__ import annotations

import json
import re

from openai import OpenAI

from pqa.config import Settings

RULE_MAP: dict[str, list[str]] = {
    "인증": ["auth", "authenticate", "jwt", "token", "validate", "middleware", "claims"],
    "인가": ["authorization", "authorize", "role", "permission", "bearer", "rbac"],
    "로그인": ["login", "signin", "token", "jwt", "password"],
    "결제": ["payment", "invoice", "paid", "pending", "kafka"],
    "주문": ["order", "checkout", "idempotency", "order_id", "usecase"],
    "재고": ["stock", "inventory", "rollback", "decrease", "update"],
}

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*|[가-힣]{2,}")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _rule_expand(question: str) -> list[str]:
    expanded: list[str] = []
    for key, terms in RULE_MAP.items():
        if key in question:
            expanded.extend(terms)
    return expanded


def _llm_expand(question: str, settings: Settings, service: str | None = None) -> list[str]:
    if not settings.openai_api_key:
        return []
    client = OpenAI(api_key=settings.openai_api_key)
    service_text = service or "ALL"
    prompt = (
        "You rewrite developer questions into code-search keywords.\n"
        "Return strict JSON only: {\"terms\": [\"...\"]}\n"
        "Rules:\n"
        "- 6~12 concise terms\n"
        "- mix Korean + English technical terms\n"
        "- prioritize implementation symbols, middleware, handlers, usecases, services\n"
        "- no prose, no markdown\n\n"
        f"service={service_text}\n"
        f"question={question}\n"
    )
    try:
        res = client.chat.completions.create(
            model=settings.query_rewrite_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = res.choices[0].message.content or ""
        data = json.loads(content)
        terms = data.get("terms", [])
        if not isinstance(terms, list):
            return []
        return [str(t).strip().lower() for t in terms if str(t).strip()]
    except Exception:
        return []


def expand_query(question: str, settings: Settings, service: str | None = None) -> tuple[str, list[str]]:
    base_tokens = _tokenize(question)
    terms: list[str] = []
    terms.extend(base_tokens)
    terms.extend(_rule_expand(question))
    if settings.enable_query_rewrite:
        terms.extend(_llm_expand(question, settings, service=service))

    seen: set[str] = set()
    unique_terms: list[str] = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        unique_terms.append(t)

    rewritten = " ".join(unique_terms[:24]).strip()
    if not rewritten:
        rewritten = question
    return rewritten, unique_terms[:24]
