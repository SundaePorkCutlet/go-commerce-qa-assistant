from pqa.models import Chunk
from pqa.retriever import search


def test_search_returns_relevant_chunk() -> None:
    chunks = [
        Chunk(id="1", path="ORDERFC/a.go", text="idempotency token is checked in usecase"),
        Chunk(id="2", path="PAYMENTFC/b.go", text="audit log daily report in mongodb repository"),
    ]
    res = search("idempotency token", chunks, top_k=1)
    assert len(res) == 1
    assert res[0].path == "ORDERFC/a.go"

