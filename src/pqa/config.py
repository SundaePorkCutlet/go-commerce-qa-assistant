from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field
import os


class Settings(BaseModel):
    repo_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[4])
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    index_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data/index/chunks.jsonl"
    )

    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    @staticmethod
    def load() -> "Settings":
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(dotenv_path=project_root / ".env")
        return Settings(
            repo_root=Path(
                os.getenv("REPO_ROOT", str(Path(__file__).resolve().parents[4]))
            ).resolve(),
            data_dir=Path(
                os.getenv("DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))
            ).resolve(),
            index_path=Path(
                os.getenv(
                    "INDEX_PATH",
                    str(Path(__file__).resolve().parents[2] / "data/index/chunks.jsonl"),
                )
            ).resolve(),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        )

