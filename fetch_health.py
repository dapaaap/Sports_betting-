
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FetchStatus(str, Enum):
    FULL_SUCCESS = "FULL_SUCCESS"
    PARTIAL_FETCH = "PARTIAL_FETCH"
    API_DEGRADED = "API_DEGRADED"
    STALE_CACHE = "STALE_CACHE"
    API_FAILURE = "API_FAILURE"


class NoResultReason(str, Enum):
    NO_VALUE_BETS = "NO_VALUE_BETS"
    API_FAILURE = "API_FAILURE"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"


STALE_WARNING_MINUTES = 5.0


@dataclass
class OddsFreshness:

    fetch_time_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _epoch: float = field(default_factory=time.monotonic, repr=False)

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self._epoch

    @property
    def age_minutes(self) -> float:
        return self.age_seconds / 60.0

    @property
    def is_stale(self) -> bool:
        return self.age_minutes > STALE_WARNING_MINUTES

    @property
    def freshness_label(self) -> str:
        mins = self.age_minutes
        if mins < 1:
            return f"{self.age_seconds:.0f}d"
        if mins < STALE_WARNING_MINUTES:
            return f"{mins:.1f}m"
        return f"[yellow]{mins:.1f}m ⚠[/yellow]"

    def annotate_events(self, events: List[Dict]) -> None:
        ts = self.fetch_time_utc.isoformat()
        for ev in events:
            ev["_fetch_timestamp"] = ts
            ev["_fetch_epoch"] = self._epoch


@dataclass
class LeagueFetchResult:
    sport_key: str
    sport_name: str
    events_found: int
    success: bool
    error: Optional[str] = None


@dataclass
class FetchHealthReport:

    status: FetchStatus
    freshness: OddsFreshness
    total_leagues: int = 0
    ok_leagues: int = 0
    failed_leagues: int = 0
    total_events: int = 0
    league_results: List[LeagueFetchResult] = field(default_factory=list)
    cache_hit: bool = False
    no_result_reason: Optional[NoResultReason] = None
    api_quota_remaining: Optional[str] = None

    @property
    def partial_rate(self) -> float:
        if self.total_leagues == 0:
            return 0.0
        return self.failed_leagues / self.total_leagues

    @property
    def status_label(self) -> str:
        colors = {
            FetchStatus.FULL_SUCCESS: "bright_green",
            FetchStatus.PARTIAL_FETCH: "yellow",
            FetchStatus.API_DEGRADED: "red",
            FetchStatus.STALE_CACHE: "yellow",
            FetchStatus.API_FAILURE: "red",
        }
        labels = {
            FetchStatus.FULL_SUCCESS: "✅ FULL SUCCESS",
            FetchStatus.PARTIAL_FETCH: "⚠ PARTIAL FETCH",
            FetchStatus.API_DEGRADED: "⚠ API DEGRADED",
            FetchStatus.STALE_CACHE: "📦 STALE CACHE",
            FetchStatus.API_FAILURE: "❌ API FAILURE",
        }
        c = colors.get(self.status, "white")
        l = labels.get(self.status, str(self.status))
        return f"[bold {c}]{l}[/bold {c}]"

    @property
    def data_age_label(self) -> str:
        return self.freshness.freshness_label

    def classify_no_results(
        self, n_matches: int, n_value_bets: int
    ) -> Optional[NoResultReason]:
        if self.status == FetchStatus.API_FAILURE or self.total_events == 0:
            self.no_result_reason = NoResultReason.API_FAILURE
        elif n_matches > 0 and n_value_bets == 0:
            self.no_result_reason = NoResultReason.NO_VALUE_BETS
        elif n_matches == 0 and self.failed_leagues > 0:
            self.no_result_reason = NoResultReason.PARTIAL_FAILURE
        elif self.total_events == 0:
            self.no_result_reason = NoResultReason.EMPTY_RESPONSE
        return self.no_result_reason

    def log_summary(self) -> None:
        logger.info(
            "Fetch health: %s | leagues %d/%d OK | events %d | age %.1fm | cache=%s",
            self.status.value,
            self.ok_leagues,
            self.total_leagues,
            self.total_events,
            self.freshness.age_minutes,
            self.cache_hit,
        )
        if self.failed_leagues > 0:
            failed = [r.sport_name for r in self.league_results if not r.success]
            logger.warning("Failed leagues: %s", ", ".join(failed))


def build_fetch_health(
    events: List[Dict],
    total_leagues: int,
    league_results: Optional[List[LeagueFetchResult]] = None,
    cache_hit: bool = False,
    freshness: Optional[OddsFreshness] = None,
    api_quota_remaining: Optional[str] = None,
) -> FetchHealthReport:
    if freshness is None:
        freshness = OddsFreshness()

    league_results = league_results or []
    ok_leagues = sum(1 for r in league_results if r.success)
    failed_leagues = total_leagues - ok_leagues

    n_events = len(events)

    if n_events == 0:
        status = FetchStatus.API_FAILURE
    elif cache_hit and freshness.is_stale:
        status = FetchStatus.STALE_CACHE
    elif failed_leagues == 0:
        status = FetchStatus.FULL_SUCCESS
    elif failed_leagues / max(total_leagues, 1) > 0.5:
        status = FetchStatus.API_DEGRADED
    else:
        status = FetchStatus.PARTIAL_FETCH

    report = FetchHealthReport(
        status=status,
        freshness=freshness,
        total_leagues=total_leagues,
        ok_leagues=ok_leagues,
        failed_leagues=failed_leagues,
        total_events=n_events,
        league_results=league_results,
        cache_hit=cache_hit,
        api_quota_remaining=api_quota_remaining,
    )
    report.log_summary()
    return report


class TTLCache:

    def __init__(self, ttl_seconds: float = 300.0):
        self._ttl = ttl_seconds
        self._store: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        exp = self._expiry.get(key)
        if exp is None:
            return None
        if time.monotonic() > exp:
            self._evict(key)
            return None
        return self._store.get(key)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        ttl = ttl if ttl is not None else self._ttl
        self._store[key] = value
        self._expiry[key] = time.monotonic() + ttl

    def age_seconds(self, key: str) -> Optional[float]:
        exp = self._expiry.get(key)
        if exp is None:
            return None
        remaining = exp - time.monotonic()
        if remaining < 0:
            return None
        return self._ttl - remaining

    def is_fresh(self, key: str) -> bool:
        return self.get(key) is not None

    def invalidate(self, key: str) -> None:
        self._evict(key)

    def _evict(self, key: str) -> None:
        self._store.pop(key, None)
        self._expiry.pop(key, None)


_results_cache = TTLCache(ttl_seconds=300.0)

CACHE_KEY_EVENTS = "fetch:events"
CACHE_KEY_HEALTH = "fetch:health"


def cache_events(events: List[Dict], health: FetchHealthReport) -> None:
    _results_cache.set(CACHE_KEY_EVENTS, events)
    _results_cache.set(CACHE_KEY_HEALTH, health)
    logger.debug("TTL cache updated: %d events, TTL=300s", len(events))


def get_cached_events() -> Optional[tuple]:
    events = _results_cache.get(CACHE_KEY_EVENTS)
    health = _results_cache.get(CACHE_KEY_HEALTH)
    if events is None or health is None:
        return None
    age = _results_cache.age_seconds(CACHE_KEY_EVENTS) or 0.0
    health.cache_hit = True
    health.status = FetchStatus.STALE_CACHE
    return events, health, age
