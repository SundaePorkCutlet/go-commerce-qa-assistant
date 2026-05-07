from __future__ import annotations

import asyncio

from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AskRequest
from app.schemas import AskResponse
from app.service import ask_question
from pqa.config import Settings
from pqa.rate_limit import rate_limiter

app = FastAPI(title="Project QA Assistant API", version="0.1.0")
settings = Settings.load()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", maxsplit=1)[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def enforce_rate_limit(request: Request) -> None:
    allowed, retry_after = rate_limiter.allow(
        _client_key(request),
        limit=settings.api_rate_limit_per_minute,
    )
    if allowed:
        return
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Too many requests. Please try again later.",
        headers={"Retry-After": str(retry_after)},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/ask", response_model=AskResponse)
async def ask(payload: AskRequest, _: None = Depends(enforce_rate_limit)) -> AskResponse:
    try:
        answer, mode, confidence = await asyncio.wait_for(
            asyncio.to_thread(
                ask_question,
                question=payload.question,
                service=payload.service,
                path_prefix=payload.path_prefix,
            ),
            timeout=settings.api_request_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Question processing timed out. Please try a narrower question.",
        ) from exc
    return AskResponse(answer=answer, mode=mode, confidence=confidence)
