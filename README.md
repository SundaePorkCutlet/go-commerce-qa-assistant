# Project Q&A Assistant

Code-aware RAG assistant for the `go-commerce` repository.

## Goals

- Answer codebase questions with **file-grounded evidence**
- Separate ingestion/indexing from query-time generation
- Keep private interview materials out of indexing scope

## Project Layout

- `src/pqa/` - application source
- `scripts/` - CLI entrypoints for indexing and asking
- `prompts/` - system and answer templates
- `eval/` - evaluation dataset and reports
- `tests/` - unit tests
- `data/` - local runtime artifacts (`raw/`, `index/`, `cache/`)

## Quick Start

1. Create virtualenv and install:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -e .[dev]`
2. Copy env:
   - `cp .env.example .env`
3. Build index:
   - `python scripts/index_once.py`
4. Ask question:
   - `python scripts/ask.py "ORDERFC 멱등성 토큰은 어디서 검증돼?"`
   - 서비스 범위 고정: `python scripts/ask.py "멱등성 토큰 검증은 어디?" --service ORDERFC`
   - 경로 범위 고정: `python scripts/ask.py "검증 로직 어디?" --path-prefix ORDERFC/cmd/order`

## Safety Rules

- Private files like `INTERVIEW_PREP*` are excluded by default.
- Answers must include evidence paths.
- If confidence is low, assistant should say so explicitly.

## Retrieval Precision Tuning Notes

정확도를 높이기 위해 아래를 적용했다.

1. 인덱싱 제외 강화
- `vendor/`, `node_modules/`, `.gopath/`, `.cursor/`, `tools/project-qa-assistant/` 제외
- 개인 문서(`INTERVIEW_PREP*`) 제외

2. 검색 점수 튜닝
- `.go` 파일 가중치, 문서/설정 파일 페널티
- `ORDERFC` 같은 서비스 힌트가 질문에 있으면 해당 서비스 경로 가중치
- `멱등성` 같은 한국어 키워드를 `idempotency`로 확장

3. 심볼 단위 청킹 (정확도 핵심 개선)
- Go 파일은 고정 길이 대신 `func/type/method` 단위로 청킹
- `symbol_hint`(예: `CheckOutOrder`, `*OrderService.CheckIdempotencyToken`)와 라인 범위를 메타데이터로 저장
- "어디 구현돼?" 같은 질문에서 심볼 매칭 가중치를 적용

4. BM25 재정렬
- 1차 후보(유사도) 이후 BM25로 재정렬해서 질문 단어 일치도를 반영

5. 구현 질의 하드 룰
- "어디/구현/검증" 성격 질문이면 코드 파일 중심으로만 답변

6. 사용자 강제 필터
- `--service`, `--path-prefix` 옵션으로 범위를 강제해 오답 문맥 유입 차단

면접에서는 "고정 길이 chunk 한계를 심볼 단위 청킹과 symbol-aware 재정렬로 개선했다"라고 설명하면 좋다.
