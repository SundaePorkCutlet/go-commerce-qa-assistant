from pqa.config import Settings
from pqa.query_rewriter import expand_query


def test_expand_query_adds_order_outbox_terms() -> None:
    settings = Settings(enable_query_rewrite=False)

    rewritten, terms = expand_query(
        "Transactional Outbox는 어디서 저장하고 누가 Kafka로 발행하나요?",
        settings,
    )

    assert "order_outbox_events" in terms
    assert "OrderOutboxPublisher" in terms
    assert "InsertOrderOutboxEventsTx" in terms
    assert "ORDERFC" in terms
    assert "order_outbox_events" in rewritten
