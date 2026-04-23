from pqa.indexer import build_go_symbol_chunks


def test_build_go_symbol_chunks_extracts_symbols_and_lines() -> None:
    go_src = """package demo

// CheckOutOrder creates order.
func CheckOutOrder() error {
    return nil
}

type OrderService struct{}

func (s *OrderService) CheckIdempotencyToken(token string) bool {
    return token != ""
}
"""
    chunks = build_go_symbol_chunks("ORDERFC/cmd/order/usecase/usecase.go", go_src, "ORDERFC")
    assert len(chunks) >= 3

    symbol_hints = [c.symbol_hint or "" for c in chunks]
    assert any("CheckOutOrder" in s for s in symbol_hints)
    assert any("CheckIdempotencyToken" in s for s in symbol_hints)

    checkout = next(c for c in chunks if c.symbol_hint and "CheckOutOrder" in c.symbol_hint)
    assert checkout.start_line is not None
    assert checkout.end_line is not None
    assert checkout.start_line <= checkout.end_line

