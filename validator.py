"""Two-source verification for upcoming football fixtures."""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from thefuzz import fuzz

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str
    source_fixture: Optional[Dict[str, Any]] = None
    home_score: int = 0
    away_score: int = 0


def _normalise(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class FixtureValidator:
    """Verify an odds event against a configurable statistics API.

    The endpoint contract is intentionally adapter-light: responses may be a list,
    or a mapping containing ``response``, ``fixtures``, ``events`` or ``data``.
    Common API-Football-style nested fields and flat fields are both supported.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str = "",
        timeout_seconds: float = 10.0,
        team_threshold: int = 85,
        league_threshold: int = 70,
        kickoff_tolerance_minutes: int = 120,
        cache_ttl_seconds: int = 300,
        session: Any = requests,
    ):
        if not endpoint:
            raise ValueError("FixtureValidator requires a source-of-truth endpoint")
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.team_threshold = team_threshold
        self.league_threshold = league_threshold
        self.kickoff_tolerance_minutes = kickoff_tolerance_minutes
        self.cache_ttl_seconds = cache_ttl_seconds
        self.session = session
        self._cache: Dict[Tuple[str, str, str], Tuple[float, List[Dict[str, Any]]]] = {}

    def _request(self, fixture: Dict[str, Any]) -> List[Dict[str, Any]]:
        kickoff = _parse_datetime(fixture.get("commence_time"))
        params = {
            "date": kickoff.date().isoformat() if kickoff else "",
            "league": fixture.get("sport_key") or fixture.get("sport_name", ""),
        }
        params = {key: value for key, value in params.items() if value}
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-apisports-key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self.session.get(
            self.endpoint,
            params=params,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        for key in ("response", "fixtures", "events", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []

    @staticmethod
    def _field(item: Dict[str, Any], *paths: str) -> Any:
        for path in paths:
            value: Any = item
            for part in path.split("."):
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(part)
            if value not in (None, ""):
                return value
        return ""

    def _source_details(self, item: Dict[str, Any]) -> Tuple[str, str, str, Optional[datetime]]:
        home = self._field(item, "home_team", "teams.home.name", "home", "teams.home")
        away = self._field(item, "away_team", "teams.away.name", "away", "teams.away")
        league = self._field(item, "league.name", "competition.name", "sport_name", "league")
        kickoff = _parse_datetime(
            self._field(item, "commence_time", "kickoff", "date", "fixture.date")
        )
        return str(home), str(away), str(league), kickoff

    def _matches(self, fixture: Dict[str, Any], source: Dict[str, Any]) -> Tuple[bool, str]:
        source_home, source_away, source_league, source_kickoff = self._source_details(source)
        home_score = fuzz.WRatio(
            _normalise(fixture.get("home_team")), _normalise(source_home)
        )
        away_score = fuzz.WRatio(
            _normalise(fixture.get("away_team")), _normalise(source_away)
        )
        if home_score < self.team_threshold or away_score < self.team_threshold:
            return False, f"team mismatch ({home_score}/{away_score})"

        expected_league = fixture.get("sport_name") or fixture.get("sport_key", "")
        if source_league and expected_league:
            league_score = fuzz.token_set_ratio(_normalise(expected_league), _normalise(source_league))
            if league_score < self.league_threshold:
                return False, f"league mismatch ({league_score})"

        expected_kickoff = _parse_datetime(fixture.get("commence_time"))
        if expected_kickoff and source_kickoff:
            delta_minutes = abs((expected_kickoff - source_kickoff).total_seconds()) / 60
            if delta_minutes > self.kickoff_tolerance_minutes:
                return False, f"kickoff mismatch ({delta_minutes:.0f}m)"
        return True, "verified"

    def validate_fixture(self, fixture: Dict[str, Any]) -> ValidationResult:
        """Return a structured result; network/API failures are invalid by design."""
        key = (
            str(fixture.get("home_team", "")),
            str(fixture.get("away_team", "")),
            str(fixture.get("commence_time", "")),
        )
        try:
            cached = self._cache.get(key)
            if cached and time.monotonic() - cached[0] <= self.cache_ttl_seconds:
                source_fixtures = cached[1]
            else:
                source_fixtures = self._request(fixture)
                self._cache[key] = (time.monotonic(), source_fixtures)
        except (requests.RequestException, TimeoutError, ValueError, TypeError) as exc:
            logger.warning("Fixture verification failed for %s vs %s: %s", key[0], key[1], exc)
            return ValidationResult(False, f"source error: {exc}")

        for source_fixture in source_fixtures:
            valid, reason = self._matches(fixture, source_fixture)
            if valid:
                return ValidationResult(True, reason, source_fixture)
        return ValidationResult(False, "fixture not found in source of truth")

    def verify_fixture(self, fixture: Dict[str, Any]) -> ValidationResult:
        return self.validate_fixture(fixture)

    def filter_fixtures(self, fixtures: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        verified: List[Dict[str, Any]] = []
        for fixture in fixtures:
            result = self.validate_fixture(fixture)
            if result.valid:
                fixture["fixture_verified"] = True
                verified.append(fixture)
            else:
                logger.warning(
                    "Dropping unverified fixture %s vs %s: %s",
                    fixture.get("home_team", "?"), fixture.get("away_team", "?"), result.reason,
                )
        return verified