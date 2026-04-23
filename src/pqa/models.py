from __future__ import annotations

from pydantic import BaseModel


class Chunk(BaseModel):
    id: str
    path: str
    text: str
    service: str | None = None
    symbol_hint: str | None = None

