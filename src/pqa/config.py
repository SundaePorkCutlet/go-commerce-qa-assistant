from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field
import os


class Settings(BaseModel):
    repo_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[4])
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    use_chroma: bool = True
    chroma_mode: str = "persistent"  # persistent | http
    chroma_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data/index/chroma"
    )
    chroma_collection: str = "go-commerce-code-chunks"
    chroma_host: str = "localhost"
    chroma_port: int = 8000

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
            use_chroma=os.getenv("USE_CHROMA", "true").lower() in {"1", "true", "yes", "on"},
            chroma_mode=os.getenv("CHROMA_MODE", "persistent").strip().lower(),
            chroma_path=Path(
                os.getenv(
                    "CHROMA_PATH",
                    str(Path(__file__).resolve().parents[2] / "data/index/chroma"),
                )
            ).resolve(),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "go-commerce-code-chunks"),
            chroma_host=os.getenv("CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        )

