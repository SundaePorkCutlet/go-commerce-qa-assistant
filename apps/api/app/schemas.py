from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    service: str | None = Field(default=None, max_length=32)
    path_prefix: str | None = Field(default=None, max_length=200)


class AskResponse(BaseModel):
    answer: str
    mode: str
    confidence: str
