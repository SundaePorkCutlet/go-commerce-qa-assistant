from __future__ import annotations

from collections import deque
from threading import Lock
import time


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(
        self,
        key: str,
        *,
        limit: int,
        window_seconds: int = 60,
        now: float | None = None,
    ) -> tuple[bool, int]:
        if limit <= 0:
            return True, 0

        current = time.time() if now is None else now
        cutoff = current - window_seconds

        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= limit:
                retry_after = max(1, int(bucket[0] + window_seconds - current))
                return False, retry_after

            bucket.append(current)
            return True, 0


rate_limiter = InMemoryRateLimiter()
