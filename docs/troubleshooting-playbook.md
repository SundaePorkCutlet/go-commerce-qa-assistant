# Troubleshooting Playbook

이 문서는 `go-commerce-qa-assistant`를 만들면서 실제로 겪은 시행착오와 해결 방법을 정리한 운영 플레이북이다.  
형식은 **증상 -> 원인 -> 빠른 진단 -> 해결 -> 재발 방지** 순서로 통일했다.

## 1) `python: command not found` (macOS/zsh)

- 증상
  - `python scripts/ask.py ...` 실행 시 `command not found`.
- 원인
  - macOS 환경에서 기본 명령이 `python`이 아니라 `python3`인 경우.
- 빠른 진단
  - `python3 --version`은 되는데 `python --version`은 실패.
- 해결
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `python3 scripts/ask.py ...`
- 재발 방지
  - 문서와 스크립트 사용 예시는 `python3` 기준으로 통일.

## 2) `zsh: no matches found: .[dev]`

- 증상
  - `pip install -e .[dev]` 실행 시 zsh glob 오류.
- 원인
  - zsh가 `[]`를 패턴으로 해석.
- 빠른 진단
  - 따옴표 없이 extras install 명령 실행 시 즉시 재현.
- 해결
  - `python3 -m pip install -e '.[dev]'`
- 재발 방지
  - README 예시에서 extras는 항상 인용부호 사용.

## 3) Chroma 인덱싱은 되는데 질의가 엉뚱한 파일만 잡힘

- 증상
  - `CheckOutOrder` 질문에서 `main.go`, `redis.go`, 문서성 청크가 상위로 노출.
- 원인
  - "정의 위치 찾기" 질의를 일반 의미 검색으로 처리.
  - 심볼 `정의/호출/문서` 구분 신호가 약함.
- 빠른 진단
  - 같은 질문에서 결과 근거가 선언부(`func/method`)가 아닌 호출부/설정 파일 위주.
- 해결
  - definition intent 분리 (`어디 구현/정의/선언`, `where implemented/defined`).
  - 정의 질의 모드에서 `symbol_hint` + `kind(func_def/method_def/type_def)` 우선.
  - `Chunk.kind` 메타데이터 추가 후 rerank에서 정의 > 호출 > 문서 가중치 적용.
- 재발 방지
  - "구현 위치" 류 질문은 default로 definition mode 적용.
  - 평가 질의셋에 심볼 정의 질의(예: `CheckOutOrder`)를 고정 회귀 테스트로 추가.

## 4) LLM이 근거를 과해석해서 "구현되어 있다"라고 잘못 답함

- 증상
  - 근거가 호출/문서인데 최종 답변이 구현 위치로 단정.
- 원인
  - LLM 프롬프트에서 정의 질의 규칙이 느슨함.
- 빠른 진단
  - 근거 파일이 선언부가 아닌데 답변은 구현 위치로 표현.
- 해결
  - 정의 질의면 프롬프트에 강제 규칙 삽입:
    - call_site/doc_section/comment를 구현 근거로 사용 금지.
    - kind가 `func_def/method_def/type_def`인 근거만 허용.
  - fallback 답변도 같은 규칙 적용(엄격 포맷).
- 재발 방지
  - 답변 후처리에서 `kind` 검증 실패 시 "확정 불가" 반환.

## 5) `.env`를 넣었는데도 API 키를 못 읽음

- 증상
  - `OPENAI_API_KEY`를 설정했는데 `has_key=False`.
- 원인
  - `.env.example`만 수정하거나, 상대 경로가 실행 위치 기준으로 해석됨.
- 빠른 진단
  - `Settings.load()` 출력에서 `has_key=False` 또는 `chroma_path`가 의도와 다름.
- 해결
  - 실제 파일: `tools/project-qa-assistant/.env` 수정.
  - 설정 경로는 프로젝트 루트 기준으로 resolve하도록 `config` 보정.
- 재발 방지
  - 시작 시 진단 로그: `repo_root`, `chroma_path`, `has_key` 확인.

## 6) OpenAI 호출 실패 (네트워크/프록시)

- 증상
  - `ProxyError 403`, `APIConnectionError`.
- 원인
  - 런타임 환경에서 외부 네트워크 제한 또는 프록시 이슈.
- 빠른 진단
  - 같은 입력에서 LLM 호출 stacktrace, 로컬에서는 성공.
- 해결
  - LLM 호출 예외를 잡고 deterministic fallback 답변으로 강등.
- 재발 방지
  - 운영에서는 health check에 outbound API 연결성 점검 포함.
  - 장애 시 degraded mode(검색+템플릿 응답) 유지.

## 7) Chroma-only 전환 후 JSONL 파일 삭제 가능 여부 혼선

- 증상
  - `chunks.jsonl` 삭제 가능 여부가 실행 시점마다 다르게 보임.
- 원인
  - 과도기 코드에서 JSON fallback 경로가 일부 남아있음.
- 빠른 진단
  - `ask.py` 또는 `index_once.py`에 `read_index` 경로 존재 여부 확인.
- 해결
  - Chroma-only로 단일화:
    - `index_once.py`: Chroma upsert만 수행.
    - `ask.py`: Chroma 후보 + 재랭크만 사용.
- 재발 방지
  - README에 "runtime requires vector DB, not JSONL" 명시.

## 8) 운영 체크리스트 (질문 품질 이슈 발생 시)

1. 인덱스 최신화
   - `python3 scripts/index_once.py`
2. definition intent 여부 판단
   - 질의에 "어디 구현/정의/선언" 포함 확인
3. symbol 후보 확인
   - symbol_hint exact/contains 매칭 로그 확인
4. evidence kind 검증
   - 최종 근거에 `func_def/method_def/type_def` 존재 확인
5. LLM 단계 분리 확인
   - retrieval 결과와 generation 결과를 분리 출력
6. 네트워크 상태 확인
   - OpenAI 호출 실패 시 fallback 동작 여부 확인

## 9) 앞으로 추가할 개선

- call graph/AST 기반 `def-use` 연결 (정의와 호출 관계 명시)
- `go/parser` 기반 선언/호출 정밀 분류
- definition 질의 전용 회귀 테스트 케이스 확대
- `--debug` 옵션으로 후보군/점수/kind를 CLI에 출력
