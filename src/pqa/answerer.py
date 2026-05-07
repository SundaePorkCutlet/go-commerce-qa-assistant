from __future__ import annotations

from pathlib import Path
import re

from openai import OpenAI

from pqa.config import Settings
from pqa.models import Chunk


def _mode_label(definition_mode: bool, core_logic_mode: bool, architecture_mode: bool) -> str:
    if definition_mode:
        return "definition"
    if core_logic_mode:
        return "core-logic"
    if architecture_mode:
        return "architecture"
    return "general"


def _format_with_mode(
    content: str, definition_mode: bool, core_logic_mode: bool, architecture_mode: bool
) -> str:
    # Mode is rendered as UI meta badge; avoid duplicating it in answer body.
    return content


def _has_core_logic_pair(evidence: list[Chunk]) -> bool:
    has_handler = any("/handler/" in c.path.lower() for c in evidence)
    has_core = any(
        ("/usecase/" in c.path.lower()) or ("/service/" in c.path.lower()) for c in evidence
    )
    has_message = any(
        ("/kafka/" in c.path.lower())
        or ("outbox" in c.path.lower())
        or ("consumer" in c.path.lower())
        or ("publisher" in c.path.lower())
        for c in evidence
    )
    has_repository = any("/repository/" in c.path.lower() or "repository" in c.path.lower() for c in evidence)
    return (has_handler and has_core) or (has_message and (has_core or has_repository))


def _build_core_logic_insufficient_answer() -> str:
    return (
        "답변:\n"
        "핵심 로직 흐름을 확정하기 위한 근거가 부족합니다.\n"
        "현재 근거에서 진입점과 핵심 처리 계층의 조합을 충분히 확보하지 못했습니다.\n\n"
        "근거:\n"
        "- HTTP 흐름은 handler + usecase/service, 이벤트 흐름은 kafka/worker + repository/service 근거가 필요합니다."
    )


def _build_definition_fallback_answer(target_symbol: str | None, evidence: list[Chunk]) -> str:
    symbol = target_symbol or "(unknown)"
    def_chunks = [c for c in evidence if (c.kind or "") in {"func_def", "method_def", "type_def"}]
    if not def_chunks:
        return (
            "답변:\n"
            f"`{symbol}`의 정의 위치를 확정할 수 없습니다.\n\n"
            "근거:\n"
            "- kind=func_def/method_def/type_def 근거가 없음"
        )

    first = def_chunks[0]
    lines = [
        "답변:",
        f"`{symbol}`의 정의 위치는 `{first.path}` 입니다.",
        "",
        "근거:",
    ]
    for c in def_chunks[:5]:
        line_span = f"L{c.start_line}-L{c.end_line}" if c.start_line and c.end_line else "unknown"
        lines.append(f"- path: {c.path}")
        lines.append(f"  symbol: {c.symbol_hint or symbol}")
        lines.append(f"  kind: {c.kind or 'unknown'}")
        lines.append(f"  lines: {line_span}")
    return "\n".join(lines)


def _build_fallback_answer(
    question: str,
    evidence: list[Chunk],
    definition_mode: bool,
    core_logic_mode: bool,
    architecture_mode: bool,
    target_symbol: str | None,
    confidence: str,
) -> str:
    if definition_mode:
        return _build_definition_fallback_answer(target_symbol, evidence)
    if not evidence:
        return (
            "관련 근거를 찾지 못했습니다. 질문 범위를 좁히거나 정확한 서비스/심볼 이름으로 다시 질문해 주세요.\n"
            "근거: 없음"
        )

    if confidence == "low":
        lines = [
            "답변:",
            "제한된 근거 기준으로는 관련 구현이 존재하는 것으로 추정됩니다.",
            "다만 추가 확인이 필요합니다.",
            "",
            "근거:",
        ]
    elif confidence == "medium":
        lines = [
            f"질문: {question}",
            "",
            "중간 신뢰도 답변(근거 기반):",
            "- 현재 근거상 관련 구현 가능성이 높습니다.",
        ]
    else:
        lines = [
            f"질문: {question}",
            "",
            "초기 답변(근거 기반):",
            "- 아래 파일 조각에서 관련 구현 흔적을 찾았습니다.",
        ]
    for i, chunk in enumerate(evidence, start=1):
        preview = " ".join(chunk.text.strip().split())
        preview = (preview[:160] + "...") if len(preview) > 160 else preview
        meta = []
        if chunk.service:
            meta.append(chunk.service)
        if chunk.symbol_hint:
            meta.append(f"symbol={chunk.symbol_hint}")
        if chunk.kind:
            meta.append(f"kind={chunk.kind}")
        if chunk.start_line and chunk.end_line:
            meta.append(f"L{chunk.start_line}-L{chunk.end_line}")
        meta_text = ", ".join(meta) if meta else "unknown"
        lines.append(f"- [{i}] `{chunk.path}` ({meta_text}) -> {preview}")

    lines.append("")
    if architecture_mode:
        lines.append("보강: 아키텍처 질문으로 분류되어 서비스 경계/상호작용 관점으로 근거를 우선 제시합니다.")
    elif core_logic_mode:
        has_handler = any("/handler/" in c.path.lower() for c in evidence)
        has_core = any(("/usecase/" in c.path.lower()) or ("/service/" in c.path.lower()) for c in evidence)
        if has_handler and has_core:
            lines.append("보강: 진입점은 handler이며, 핵심 비즈니스 로직은 usecase/service 계층에서 처리됩니다.")
    lines.append("주의: 현재 MVP는 키워드/유사도 검색 기반이며, 최종 확정 전 원본 파일 확인이 필요합니다.")
    return "\n".join(lines)


def _load_system_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "system.md"
    if not prompt_path.exists():
        return (
            "You are a codebase Q&A assistant.\n"
            "Answer only from provided evidence and include file paths."
        )
    return prompt_path.read_text(encoding="utf-8").strip()


def _build_evidence_context(evidence: list[Chunk]) -> str:
    lines: list[str] = []
    for i, chunk in enumerate(evidence, start=1):
        meta = []
        if chunk.service:
            meta.append(chunk.service)
        if chunk.symbol_hint:
            meta.append(f"symbol={chunk.symbol_hint}")
        if chunk.kind:
            meta.append(f"kind={chunk.kind}")
        if chunk.start_line is not None and chunk.end_line is not None:
            meta.append(f"L{chunk.start_line}-L{chunk.end_line}")
        meta_text = ", ".join(meta) if meta else "unknown"
        lines.append(f"[{i}] path={chunk.path} ({meta_text})")
        lines.append(chunk.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def _append_evidence_files_section(content: str, evidence: list[Chunk]) -> str:
    cleaned = re.sub(r"\n*근거 파일:\n(?:- .*\n?)*\s*$", "", content.strip(), flags=re.MULTILINE)
    dedup: dict[str, str] = {}
    for chunk in evidence:
        line_span = ""
        if chunk.start_line is not None and chunk.end_line is not None:
            line_span = f"L{chunk.start_line}-L{chunk.end_line}"
        dedup.setdefault(chunk.path, line_span)

    lines = ["근거 파일:"]
    for path, line_span in dedup.items():
        if line_span:
            lines.append(f"- `{path}` ({line_span})")
        else:
            lines.append(f"- `{path}`")
    return f"{cleaned}\n\n" + "\n".join(lines)


def _build_llm_answer(
    question: str,
    evidence: list[Chunk],
    settings: Settings,
    definition_mode: bool,
    core_logic_mode: bool,
    architecture_mode: bool,
    target_symbol: str | None,
    confidence: str,
) -> str:
    if core_logic_mode and not _has_core_logic_pair(evidence):
        return _format_with_mode(
            _build_core_logic_insufficient_answer(), definition_mode, core_logic_mode, architecture_mode
        )

    if not settings.openai_api_key:
        return _format_with_mode(
            _build_fallback_answer(
                question,
                evidence,
                definition_mode,
                core_logic_mode,
                architecture_mode,
                target_symbol,
                confidence,
            ),
            definition_mode,
            core_logic_mode,
            architecture_mode,
        )
    if not evidence:
        return _format_with_mode(
            _build_fallback_answer(
                question,
                evidence,
                definition_mode,
                core_logic_mode,
                architecture_mode,
                target_symbol,
                confidence,
            ),
            definition_mode,
            core_logic_mode,
            architecture_mode,
        )

    client = OpenAI(api_key=settings.openai_api_key)
    system_prompt = _load_system_prompt()
    evidence_context = _build_evidence_context(evidence)
    extra_rules = ""
    if definition_mode:
        extra_rules = (
            "- 이 질문은 정의 위치 조회다.\n"
            "- 호출 위치(call_site), 문서(doc_section), 주석(comment)을 구현 위치로 답하지 마라.\n"
            "- kind가 func_def/method_def/type_def 인 근거만으로 답하라.\n"
            "- 출력 형식은 반드시 다음을 지켜라:\n"
            "  답변:\n"
            f"  `{target_symbol or 'symbol'}`의 정의 위치는 XXX입니다.\n\n"
            "  근거:\n"
            "  - path: ...\n"
            "    symbol: ...\n"
            "    kind: func_def|method_def|type_def\n"
            "    lines: Lxx-Lyy\n"
        )
    elif core_logic_mode:
        extra_rules = (
            "- 이 질문은 핵심 처리 흐름 질의다.\n"
            "- HTTP 요청 흐름이면 handler 진입점과 usecase/service 핵심 로직을 구분해서 설명하라.\n"
            "- Kafka/Outbox/worker 흐름이면 저장 repository와 발행 worker/producer를 구분해서 설명하라.\n"
            "- 가능하면 두 계층 이상의 근거 파일을 함께 제시하라.\n"
        )
    elif architecture_mode:
        extra_rules = (
            "- 이 질문은 아키텍처 질의다.\n"
            "- 서비스 경계, 저장소, 메시징, 외부 연동, 관측성 관점으로 구조를 요약하라.\n"
            "- 구현 세부보다 컴포넌트 관계와 데이터 흐름을 우선 설명하라.\n"
        )

    confidence_rules = ""
    if confidence == "high":
        confidence_rules = (
            "- confidence=high: 단정형 문장(~입니다)을 사용해도 좋다.\n"
            "- 핵심 근거를 명확하게 제시하라.\n"
        )
    elif confidence == "medium":
        confidence_rules = (
            "- confidence=medium: 가능성/추정 표현(~로 보입니다, 가능성이 높습니다)을 사용하라.\n"
            "- 과도한 단정은 피하라.\n"
        )
    elif confidence == "low":
        confidence_rules = (
            "- confidence=low: 1~2줄로 짧고 보수적으로 답하라.\n"
            "- '제한된 근거 기준으로는 ...로 추정됩니다. 다만 추가 확인이 필요합니다.' 톤을 유지하라.\n"
            "- 근거가 약한 부분은 명확히 불확실하다고 적어라.\n"
        )

    user_prompt = (
        f"질문:\n{question}\n\n"
        f"근거 청크:\n{evidence_context}\n\n"
        "요구사항:\n"
        "- 한국어로 답변\n"
        "- 근거 기반으로만 답변\n"
        "- 핵심 결론 2~4줄\n"
        "- 마지막에 '근거 파일:' 줄을 만들고 파일 경로를 bullet로 나열\n"
        "- 확실하지 않으면 불확실하다고 명시"
        + ("\n" + extra_rules if extra_rules else "")
        + ("\n" + confidence_rules if confidence_rules else "")
    )

    try:
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        message = response.choices[0].message.content
        if not message:
            return _format_with_mode(
                _build_fallback_answer(
                    question,
                    evidence,
                    definition_mode,
                    core_logic_mode,
                    architecture_mode,
                    target_symbol,
                    confidence,
                ),
                definition_mode,
                core_logic_mode,
                architecture_mode,
            )
        finalized = _append_evidence_files_section(message.strip(), evidence)
        return _format_with_mode(finalized, definition_mode, core_logic_mode, architecture_mode)
    except Exception:
        # Keep the app available even when network/API is temporarily unavailable.
        return _format_with_mode(
            _build_fallback_answer(
                question,
                evidence,
                definition_mode,
                core_logic_mode,
                architecture_mode,
                target_symbol,
                confidence,
            ),
            definition_mode,
            core_logic_mode,
            architecture_mode,
        )


def build_answer(
    question: str,
    evidence: list[Chunk],
    settings: Settings,
    definition_mode: bool = False,
    core_logic_mode: bool = False,
    architecture_mode: bool = False,
    target_symbol: str | None = None,
    confidence: str = "none",
) -> str:
    return _build_llm_answer(
        question,
        evidence,
        settings,
        definition_mode,
        core_logic_mode,
        architecture_mode,
        target_symbol,
        confidence,
    )
