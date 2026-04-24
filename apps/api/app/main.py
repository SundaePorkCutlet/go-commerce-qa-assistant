from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AskRequest
from app.schemas import AskResponse
from app.service import ask_question

app = FastAPI(title="Project QA Assistant API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    answer, mode, confidence = ask_question(
        question=payload.question,
        service=payload.service,
        path_prefix=payload.path_prefix,
    )
    return AskResponse(answer=answer, mode=mode, confidence=confidence)
