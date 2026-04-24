from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field
import os


def _resolve_env_path(raw: str, *, base_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _safe_parent(path: Path, levels: int) -> Path:
    current = path
    for _ in range(levels):
        if current.parent == current:
            return current
        current = current.parent
    return current


class Settings(BaseModel):
    repo_root: Path = Field(
        default_factory=lambda: _safe_parent(Path(__file__).resolve().parents[2], 1)
    )
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
    enable_query_rewrite: bool = True
    query_rewrite_model: str = "gpt-4o-mini"

    @staticmethod
    def load() -> "Settings":
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(dotenv_path=project_root / ".env")
        default_repo_root = _safe_parent(project_root, 1)
        return Settings(
            repo_root=_resolve_env_path(
                os.getenv("REPO_ROOT", str(default_repo_root)),
                base_dir=project_root,
            ),
            data_dir=_resolve_env_path(
                os.getenv("DATA_DIR", "./data"),
                base_dir=project_root,
            ),
            use_chroma=os.getenv("USE_CHROMA", "true").lower() in {"1", "true", "yes", "on"},
            chroma_mode=os.getenv("CHROMA_MODE", "persistent").strip().lower(),
            chroma_path=_resolve_env_path(
                os.getenv("CHROMA_PATH", "./data/index/chroma"),
                base_dir=project_root,
            ),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "go-commerce-code-chunks"),
            chroma_host=os.getenv("CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            enable_query_rewrite=os.getenv("ENABLE_QUERY_REWRITE", "true").lower()
            in {"1", "true", "yes", "on"},
            query_rewrite_model=os.getenv("QUERY_REWRITE_MODEL", "gpt-4o-mini"),
        )

