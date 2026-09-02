"""HTTP fetching with an on-disk response cache and client-side rate limiting.

Every network call in this project goes through here. Two reasons:

1. A crawl is re-run constantly during development. Without a cache you re-fetch
   thousands of responses you already have, which is slow and rude to the host.
2. Rate limiting lives in one place, so there is no way to accidentally hammer an
   API from some module that forgot about it.

Cached 404s matter as much as cached 200s: most user IDs discovered by the crawl
have no leagues in a given season, and re-asking every run is pure waste.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

# Sleeper documents a ceiling near 1000 calls/min. Sit well under it by default;
# a corpus crawl is not time-critical and being a good citizen is free.
DEFAULT_RATE_PER_SECOND = 5.0

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class RateLimiter:
    """Blocks callers so requests are spaced at least ``1/per_second`` apart."""

    def __init__(self, per_second: float = DEFAULT_RATE_PER_SECOND) -> None:
        if per_second <= 0:
            raise ValueError("per_second must be positive")
        self._min_interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._last_call = float("-inf")

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._last_call = now


@dataclass(frozen=True)
class CachedResponse:
    """A stored response. ``body`` is None for a cached 404."""

    status: int
    body: Any
    fetched_at: float

    @property
    def is_miss(self) -> bool:
        return self.status == 404


class HttpCache:
    """Caching GET client for JSON APIs.

    ``ttl`` semantics: ``None`` means the entry never expires, which is correct for
    immutable data (a completed season's drafts and transactions). Pass a number of
    seconds for anything that still moves, such as current-season rosters.
    """

    def __init__(
        self,
        root: Path | str,
        base_url: str,
        *,
        rate_limiter: RateLimiter | None = None,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.rate_limiter = rate_limiter or RateLimiter()
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.stats = {"hits": 0, "fetches": 0, "misses_cached": 0}

    def _path_for(self, path: str) -> Path:
        digest = hashlib.sha256(f"{self.base_url}/{path}".encode()).hexdigest()
        return self.root / digest[:2] / f"{digest}.json"

    def _read(self, path: str) -> CachedResponse | None:
        f = self._path_for(path)
        if not f.exists():
            return None
        try:
            raw = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("corrupt cache entry, refetching: %s", f)
            return None
        return CachedResponse(raw["status"], raw["body"], raw["fetched_at"])

    def _write(self, path: str, entry: CachedResponse) -> None:
        f = self._path_for(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"path": path, "status": entry.status, "body": entry.body,
                 "fetched_at": entry.fetched_at}
            )
        )
        tmp.replace(f)  # atomic, so an interrupted crawl never leaves half a file

    def get_json(self, path: str, *, ttl: float | None = None) -> Any:
        """Return the decoded body, or None if the resource does not exist (404)."""
        cached = self._read(path)
        if cached is not None and (ttl is None or time.time() - cached.fetched_at < ttl):
            self.stats["hits"] += 1
            return cached.body

        entry = self._fetch(path)
        self._write(path, entry)
        self.stats["fetches"] += 1
        if entry.is_miss:
            self.stats["misses_cached"] += 1
        return entry.body

    def _fetch(self, path: str) -> CachedResponse:
        url = f"{self.base_url}/{path.lstrip('/')}"
        backoff = 1.0
        last_status = 0

        for attempt in range(self.max_retries):
            self.rate_limiter.acquire()
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("request failed (%s), retry %d: %s", exc, attempt + 1, url)
                time.sleep(backoff)
                backoff *= 2
                continue

            last_status = resp.status_code
            if resp.status_code == 404:
                return CachedResponse(404, None, time.time())
            if resp.status_code in RETRY_STATUSES:
                log.warning("status %d, retry %d: %s", resp.status_code, attempt + 1, url)
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            return CachedResponse(resp.status_code, resp.json(), time.time())

        raise RuntimeError(f"giving up after {self.max_retries} attempts (last={last_status}): {url}")
