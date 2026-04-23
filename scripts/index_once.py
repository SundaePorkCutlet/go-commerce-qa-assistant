from __future__ import annotations

from rich import print

from pqa.config import Settings
from pqa.indexer import build_chunks
from pqa.indexer import write_index


def main() -> None:
    settings = Settings.load()
    print(f"[bold cyan]Indexing repo:[/bold cyan] {settings.repo_root}")
    chunks = build_chunks(settings.repo_root)
    write_index(chunks, settings.index_path)
    print(f"[green]Done[/green] - chunks: {len(chunks)}")
    print(f"[green]Index path[/green]: {settings.index_path}")


if __name__ == "__main__":
    main()

