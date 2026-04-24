from __future__ import annotations

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    service: str | None = None
    path_prefix: str | None = None


class AskResponse(BaseModel):
    answer: str
    mode: str
    confidence: str
