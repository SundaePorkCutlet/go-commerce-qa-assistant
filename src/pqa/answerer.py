from __future__ import annotations

from pqa.models import Chunk


def build_answer(question: str, evidence: list[Chunk]) -> str:
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

