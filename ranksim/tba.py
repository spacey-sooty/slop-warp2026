"""Minimal Blue Alliance v3 client with an on-disk cache.

The cache is what makes the tool usable at an event with flaky wifi: every fetch
lands in cache/<event>/ and every read falls back to it, so `--offline` (or a
dead network) still produces a full simulation from the last good pull.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://www.thebluealliance.com/api/v3"

# TBA answers 403 to urllib's default "Python-urllib/3.x" User-Agent. Without
# this every live fetch fails and silently serves the cache instead.
USER_AGENT = "ranksim/1.0 (FRC event ranking simulator)"
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"


class TBAError(RuntimeError):
    pass


def load_api_key() -> str | None:
    """TBA_API_KEY from the environment, else from event-sims/.env."""
    key = os.environ.get("TBA_API_KEY")
    if key:
        return key.strip()
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("TBA_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return None


class TBAClient:
    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path = CACHE_DIR,
        offline: bool = False,
        ttl: float = 120.0,
    ):
        self.api_key = api_key if api_key is not None else load_api_key()
        self.cache_dir = Path(cache_dir)
        self.offline = offline
        self.ttl = ttl
        # Whether the most recent _fetch actually reached TBA, and why not if it
        # did not. A refresh button that quietly serves yesterday's cache is
        # worse than one that says it failed.
        self.last_source = "none"
        self.warnings: list[str] = []

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.json"

    def _fetch(self, path: str, name: str, force: bool = False):
        cache_path = self._cache_path(name)
        fresh = (
            cache_path.exists()
            and not force
            and (time.time() - cache_path.stat().st_mtime) < self.ttl
        )
        if self.offline or fresh:
            if cache_path.exists():
                self.last_source = "cache"
                return json.loads(cache_path.read_text())
            if self.offline:
                raise TBAError(f"offline and no cached copy of {name} at {cache_path}")

        if not self.api_key:
            if cache_path.exists():
                self._fell_back(name, "no TBA API key configured")
                return json.loads(cache_path.read_text())
            raise TBAError(
                "No TBA API key. Set TBA_API_KEY in the environment or in event-sims/.env"
            )

        req = urllib.request.Request(
            f"{BASE}{path}",
            headers={"X-TBA-Auth-Key": self.api_key, "User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if cache_path.exists():
                self._fell_back(name, str(exc))
                return json.loads(cache_path.read_text())
            raise TBAError(f"TBA request failed for {path}: {exc}") from exc

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload))
        self.last_source = "tba"
        return payload

    def _fell_back(self, name: str, reason: str) -> None:
        self.last_source = "stale-cache"
        self.warnings.append(f"{name}: served from cache ({reason})")

    def event(self, event_key: str, force: bool = False) -> dict:
        return self._fetch(f"/event/{event_key}", "event", force)

    def matches(self, event_key: str, force: bool = False) -> list[dict]:
        return self._fetch(f"/event/{event_key}/matches", "matches", force)

    def rankings(self, event_key: str, force: bool = False) -> dict:
        return self._fetch(f"/event/{event_key}/rankings", "rankings", force)
