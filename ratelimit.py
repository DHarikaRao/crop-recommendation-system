"""Sliding-window rate limiter kept in process memory (one entry per client key)."""
import math
import threading
import time
from collections import deque


class RateLimiter:
    """Allow at most `limit` calls per `window` seconds for each key. A limit <= 0 disables it."""

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._hits = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Return (allowed, seconds_until_retry)."""
        if self.limit <= 0:
            return True, 0
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            self._drop_old(hits, now)
            if len(hits) >= self.limit:
                return False, max(1, math.ceil(self.window - (now - hits[0])))
            hits.append(now)
            if len(self._hits) > 10_000:  # keep memory bounded
                for stale in [k for k, h in self._hits.items() if not self._drop_old(h, now) and not h]:
                    del self._hits[stale]
            return True, 0

    def _drop_old(self, hits, now):
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        return hits
