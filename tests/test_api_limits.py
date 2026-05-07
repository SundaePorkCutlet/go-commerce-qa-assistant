from pqa.rate_limit import InMemoryRateLimiter


def test_rate_limiter_blocks_after_limit_until_window_expires() -> None:
    limiter = InMemoryRateLimiter()

    assert limiter.allow("client", limit=2, window_seconds=60, now=100.0) == (True, 0)
    assert limiter.allow("client", limit=2, window_seconds=60, now=101.0) == (True, 0)

    allowed, retry_after = limiter.allow("client", limit=2, window_seconds=60, now=102.0)
    assert allowed is False
    assert retry_after == 58

    assert limiter.allow("client", limit=2, window_seconds=60, now=161.0) == (True, 0)
