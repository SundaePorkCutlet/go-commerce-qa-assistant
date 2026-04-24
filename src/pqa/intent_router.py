from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import OpenAI

from pqa.config import Settings

ARCHITECTURE_HINT_RE = re.compile(
    r"(아키텍처|구조|전체\s*흐름|전체\s*구성|구성도|시스템\s*구성|서비스\s*분리|마이크로서비스|msa|architecture|high[- ]level|overview|microservice)",
    re.IGNORECASE,
)
DEFINITION_HINT_RE = re.compile(
    r"(정의|선언|method|symbol|where\s+(defined|declared)|definition|declaration)",
    re.IGNORECASE,
)
CORE_LOGIC_HINT_RE = re.compile(
    r"(핵심\s*로직|실제\s*처리|검증|상태\s*변경|흐름|어디서\s*하나요)",
    re.IGNORECASE,
)


@dataclass
class IntentDecision:
    mode: str
    confidence: float
    source: str


def _rule_route(question: str) -> IntentDecision | None:
    if DEFINITION_HINT_RE.search(question):
        return IntentDecision(mode="definition", confidence=0.98, source="rule")
    if CORE_LOGIC_HINT_RE.search(question):
        return IntentDecision(mode="core-logic", confidence=0.95, source="rule")
    if ARCHITECTURE_HINT_RE.search(question):
        return IntentDecision(mode="architecture", confidence=0.95, source="rule")
    return None


def _llm_route(question: str, settings: Settings) -> IntentDecision | None:
    if not settings.openai_api_key:
        return None
    client = OpenAI(api_key=settings.openai_api_key)
    prompt = (
        "Classify the user question intent for codebase QA.\n"
        "Return strict JSON only: "
        '{"mode":"definition|core-logic|architecture|general","confidence":0.0-1.0,"reason":"..."}\n'
        "Guidelines:\n"
        "- definition: asks where symbol/function/type is defined\n"
        "- core-logic: asks actual processing flow/validation/state changes\n"
        "- architecture: asks high-level structure/components/interactions\n"
        "- general: anything else\n\n"
        f"question={question}\n"
    )
    try:
        res = client.chat.completions.create(
            model=settings.intent_router_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = res.choices[0].message.content or ""
        data = json.loads(content)
        mode = str(data.get("mode", "general")).strip()
        if mode not in {"definition", "core-logic", "architecture", "general"}:
            return None
        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        return IntentDecision(mode=mode, confidence=confidence, source="llm")
    except Exception:
        return None


def route_intent(question: str, settings: Settings) -> IntentDecision:
    rule = _rule_route(question)
    if rule:
        return rule

    if settings.enable_intent_router:
        llm = _llm_route(question, settings)
        if llm and llm.confidence >= settings.intent_router_min_confidence:
            return llm

    return IntentDecision(mode="general", confidence=0.5, source="default")
