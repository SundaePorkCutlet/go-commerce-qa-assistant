from __future__ import annotations

from rich import print

from pqa.config import Settings
from pqa.indexer import build_chunks
from pqa.vector_store import upsert_chunks


def main() -> None:
    settings = Settings.load()
    if not settings.use_chroma:
        raise RuntimeError(
            "Chroma-only mode: USE_CHROMA must be true. "
            "JSONL local index is not used in this deployment model."
        )
    print(f"[bold cyan]Indexing repo:[/bold cyan] {settings.repo_root}")
    chunks = build_chunks(settings.repo_root)
    print(f"[green]Done[/green] - chunks: {len(chunks)}")
    upsert_chunks(settings, chunks)
    print(
        "[green]Chroma rebuild done[/green]: "
        f"{settings.chroma_collection} ({settings.chroma_mode})"
    )


if __name__ == "__main__":
    main()
