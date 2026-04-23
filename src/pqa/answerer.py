from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from pqa.config import Settings
from pqa.models import Chunk


def _build_fallback_answer(question: str, evidence: list[Chunk]) -> str:
    if not evidence:
        return (
            "관련 근거를 찾지 못했습니다. 질문 범위를 좁히거나 정확한 서비스/심볼 이름으로 다시 질문해 주세요.\n"
            "근거: 없음"
        )

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
        if chunk.start_line and chunk.end_line:
            meta.append(f"L{chunk.start_line}-L{chunk.end_line}")
        meta_text = ", ".join(meta) if meta else "unknown"
        lines.append(f"- [{i}] `{chunk.path}` ({meta_text}) -> {preview}")

    lines.append("")
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
        if chunk.start_line is not None and chunk.end_line is not None:
            meta.append(f"L{chunk.start_line}-L{chunk.end_line}")
        meta_text = ", ".join(meta) if meta else "unknown"
        lines.append(f"[{i}] path={chunk.path} ({meta_text})")
        lines.append(chunk.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def _build_llm_answer(question: str, evidence: list[Chunk], settings: Settings) -> str:
    if not settings.openai_api_key:
        return _build_fallback_answer(question, evidence)
    if not evidence:
        return _build_fallback_answer(question, evidence)

    client = OpenAI(api_key=settings.openai_api_key)
    system_prompt = _load_system_prompt()
    evidence_context = _build_evidence_context(evidence)
    user_prompt = (
        f"질문:\n{question}\n\n"
        f"근거 청크:\n{evidence_context}\n\n"
        "요구사항:\n"
        "- 한국어로 답변\n"
        "- 근거 기반으로만 답변\n"
        "- 핵심 결론 2~4줄\n"
        "- 마지막에 '근거 파일:' 줄을 만들고 파일 경로를 bullet로 나열\n"
        "- 확실하지 않으면 불확실하다고 명시"
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
            return _build_fallback_answer(question, evidence)
        return message.strip()
    except Exception:
        # Keep the app available even when network/API is temporarily unavailable.
        return _build_fallback_answer(question, evidence)


def build_answer(question: str, evidence: list[Chunk], settings: Settings) -> str:
    return _build_llm_answer(question, evidence, settings)

