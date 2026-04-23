from __future__ import annotations

import typer
from rich import print

from pqa.answerer import build_answer
from pqa.config import Settings
from pqa.indexer import read_index
from pqa.retriever import search

app = typer.Typer(add_completion=False)


@app.command()
def main(
    question: str,
    service: str | None = typer.Option(
        default=None, help="서비스 필터 (ORDERFC/PAYMENTFC/PRODUCTFC/USERFC)"
    ),
    path_prefix: str | None = typer.Option(
        default=None, help="경로 prefix 필터 (예: ORDERFC/cmd/order)"
    ),
) -> None:
    settings = Settings.load()
    chunks = read_index(settings.index_path)
    evidence = search(question, chunks, top_k=5, service_filter=service, path_prefix=path_prefix)
    answer = build_answer(question, evidence)
    print(answer)


if __name__ == "__main__":
    app()

