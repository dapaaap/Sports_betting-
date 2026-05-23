
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from config import (
    ANALYSIS_CONFIG,
    FOOTBALL_SPORTS_MAJOR,
    MARKET_CONFIG,
    ODDS_API_BASE_URL,
    ODDS_API_IO_BASE_URL,
    ODDS_API_IO_BOOKMAKERS,
    ODDS_API_IO_KEY,
    ODDS_API_IO_MAX_EVENTS,
    ODDS_API_MARKETS,
    ODDS_API_ODDS_FORMAT,
    ODDS_API_REGIONS,
    THE_ODDS_API_KEY,
)
from fetch_health import (
    STALE_WARNING_MINUTES,
    FetchHealthReport,
    LeagueFetchResult,
    OddsFreshness,
    build_fetch_health,
    cache_events,
    get_cached_events,
)

logger = logging.getLogger(__name__)

SPORT_NAME_MAP = {
    "soccer_epl": "Premier League",
    "soccer_spain_la_liga": "La Liga",
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_france_ligue_one": "Ligue 1",
    "soccer_uefa_champs_league": "Champions League",
    "soccer_uefa_europa_league": "Europa League",
    "soccer_conmebol_libertadores": "Copa Libertadores",
    "soccer_brazil_campeonato": "Serie A Brasil",
    "soccer_efl_champ": "Championship",
    "soccer_netherlands_eredivisie": "Eredivisie",
    "soccer_portugal_primeira_liga": "Primeira Liga",
    "soccer_turkey_super_league": "Super Lig",
    "soccer_belgium_first_div": "Belgian Pro League",
    "soccer_scotland_premiership": "Scottish Premiership",
    "soccer_austria_football_bundesliga": "Austrian Bundesliga",
    "soccer_denmark_superliga": "Danish Superliga",
    "soccer_norway_eliteserien": "Eliteserien",
    "soccer_sweden_allsvenskan": "Allsvenskan",
    "soccer_swiss_superleague": "Swiss Super League",
    "soccer_greece_super_league": "Greek Super League",
    "soccer_czech_liga": "Czech First League",
    "soccer_russia_premier_league": "Russian Premier League",
    "soccer_ukraine_premier_league": "Ukrainian Premier League",
    "soccer_croatia_football": "Croatian HNL",
    "soccer_romania_liga_1": "Romanian Liga 1",
    "soccer_serbia_superliga": "Serbian SuperLiga",
    "soccer_poland_ekstraklasa": "Polish Ekstraklasa",
    "soccer_usa_mls": "MLS",
    "soccer_argentina_primera_division": "Liga Argentina",
    "soccer_mexico_ligamx": "Liga MX",
    "soccer_chile_primera_division": "Primera Chile",
    "soccer_colombia_primera_a": "Liga BetPlay",
    "soccer_japan_j_league": "J-League",
    "soccer_south_korea_kleague1": "K League 1",
    "soccer_australia_aleague": "A-League",
    "soccer_china_superleague": "CSL China",
}


LOOKAHEAD_HOURS = ANALYSIS_CONFIG.lookahead_hours


class OddsAPIClient:

    def __init__(self, api_key: str = THE_ODDS_API_KEY):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.remaining_requests = None
        self.requests_used = None
        self.fetch_health: Optional[FetchHealthReport] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"Accept": "application/json"},
            timeout=aiohttp.ClientTimeout(total=30),
        )
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    def _update_quota(self, headers: Dict):
        self.remaining_requests = headers.get("x-requests-remaining")
        self.requests_used = headers.get("x-requests-used")

    async def _get(self, endpoint: str, params: Dict = None) -> Optional[Any]:
        if not self.session:
            raise RuntimeError("Session not started. Use async context manager.")

        base_params = {"apiKey": self.api_key}
        if params:
            base_params.update(params)

        url = f"{ODDS_API_BASE_URL}/{endpoint}"

        for attempt in range(3):
            try:
                async with self.session.get(url, params=base_params) as resp:
                    self._update_quota(resp.headers)
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 401:
                        text = await resp.text()
                        if "OUT_OF_USAGE_CREDITS" in text:
                            logger.error(
                                "❌ KUOTA HABIS! Ganti API Key atau kurangi liga di api_keys.py."
                            )
                        else:
                            logger.error("❌ API key salah atau tidak valid.")
                        return None
                    elif resp.status == 429:
                        logger.warning(
                            "⚠️  Rate limit. Menunggu %ds...", 10 * (attempt + 1)
                        )
                        await asyncio.sleep(10 * (attempt + 1))
                        continue
                    else:
                        text = await resp.text()
                        if resp.status == 422 and "btts" in text:
                            return "ERR_BTTS"
                        logger.error("API Error %s: %s", resp.status, text[:200])
                        return None
            except aiohttp.ClientError as e:
                if attempt < 2:
                    logger.warning(
                        "Network error (attempt %s/3), retry 2s... (%s)", attempt + 1, e
                    )
                    await asyncio.sleep(2)
                else:
                    logger.error("Network error: %s", e)
                    return None

    async def get_sports(self) -> List[Dict]:
        data = await self._get("sports", {"all": "false"})
        return data or []

    async def get_events_with_odds(
        self,
        sport: str,
        markets: str = ODDS_API_MARKETS,
        regions: str = ODDS_API_REGIONS,
    ) -> List[Dict]:
        params = {
            "markets": markets,
            "regions": regions,
            "oddsFormat": ODDS_API_ODDS_FORMAT,
            "dateFormat": "iso",
        }
        data = await self._get(f"sports/{sport}/odds", params)

        if data == "ERR_BTTS" or data is None:
            fallback1 = "h2h,totals"
            if fallback1 != markets:
                params["markets"] = fallback1
                data = await self._get(f"sports/{sport}/odds", params)

        if data is None:
            params["markets"] = "h2h"
            data = await self._get(f"sports/{sport}/odds", params)

        return data if isinstance(data, list) else []

    async def fetch_all_football_markets(self) -> List[Dict]:
        freshness = OddsFreshness()

        if MARKET_CONFIG.mode == "broad":
            all_sports_data = await self.get_sports()
            if all_sports_data:
                soccer_sport_keys = [
                    s["key"]
                    for s in all_sports_data
                    if s.get("group", "").lower() in ("soccer", "football")
                    and s.get("active", False)
                ]
                logger.info(
                    "Ditemukan %d liga sepak bola aktif (Broad Mode)",
                    len(soccer_sport_keys),
                )
            else:
                from config import FOOTBALL_SPORTS_BROAD

                soccer_sport_keys = FOOTBALL_SPORTS_BROAD
                logger.warning(
                    "Failed to fetch league list, using fallback list (%d leagues)",
                    len(soccer_sport_keys),
                )
        else:
            soccer_sport_keys = FOOTBALL_SPORTS_MAJOR
            logger.info("Mode MAJOR: %d liga besar", len(soccer_sport_keys))

        sem = asyncio.Semaphore(4)
        league_results: List[LeagueFetchResult] = []

        async def fetch_with_limit(sport):
            async with sem:
                res = await self._fetch_sport_with_meta(sport)
                await asyncio.sleep(0.5)

                sport_name = SPORT_NAME_MAP.get(
                    sport, sport.replace("soccer_", "").replace("_", " ").title()
                )
                lr = LeagueFetchResult(
                    sport_key=sport,
                    sport_name=sport_name,
                    events_found=len(res) if res else 0,
                    success=bool(res),
                    error=None if res else "No data returned",
                )
                league_results.append(lr)
                return res

        tasks = [fetch_with_limit(sport) for sport in soccer_sport_keys]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_events = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                sport = (
                    soccer_sport_keys[i] if i < len(soccer_sport_keys) else "unknown"
                )
                sport_name = SPORT_NAME_MAP.get(sport, sport)

                if not any(r.sport_key == sport for r in league_results):
                    league_results.append(
                        LeagueFetchResult(
                            sport_key=sport,
                            sport_name=sport_name,
                            events_found=0,
                            success=False,
                            error=str(result),
                        )
                    )
                continue
            if result:
                all_events.extend(result)

        now = datetime.now(timezone.utc)
        seen_ids: set = set()
        filtered = []

        for event in all_events:
            try:
                eid = event.get("id", "")
                if eid in seen_ids:
                    continue
                seen_ids.add(eid)

                commence_time = datetime.fromisoformat(
                    event["commence_time"].replace("Z", "+00:00")
                )
                hours_until = (commence_time - now).total_seconds() / 3600

                if 0 < hours_until <= LOOKAHEAD_HOURS:
                    event["hours_until"] = round(hours_until, 2)
                    event["commence_utc"] = commence_time
                    event["is_live"] = False
                    filtered.append(event)

            except (KeyError, ValueError):
                continue

        freshness.annotate_events(filtered)

        filtered.sort(key=lambda x: x.get("hours_until", 99))

        self.fetch_health = build_fetch_health(
            events=filtered,
            total_leagues=len(soccer_sport_keys),
            league_results=league_results,
            freshness=freshness,
            api_quota_remaining=self.remaining_requests,
        )

        cache_events(filtered, self.fetch_health)

        logger.info(
            "Fetch complete: %d events | %s | age %.1fs",
            len(filtered),
            self.fetch_health.status.value,
            freshness.age_seconds,
        )
        return filtered

    async def _fetch_sport_with_meta(self, sport: str) -> List[Dict]:
        events = await self.get_events_with_odds(sport)
        sport_name = SPORT_NAME_MAP.get(
            sport, sport.replace("soccer_", "").replace("_", " ").title()
        )
        for event in events:
            event["sport_key"] = sport
            event["sport_name"] = sport_name
        return events


class OddsAPIIOClient:

    def __init__(
        self,
        api_key: str = ODDS_API_IO_KEY,
        bookmakers: str = ODDS_API_IO_BOOKMAKERS,
    ):
        self.api_key = api_key
        self.bookmakers = bookmakers
        self.session: Optional[aiohttp.ClientSession] = None
        self.remaining_requests = None
        self.requests_used = None
        self.fetch_health: Optional[FetchHealthReport] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"Accept": "application/json"},
            timeout=aiohttp.ClientTimeout(total=45),
        )
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    def _update_quota(self, headers: Dict):
        self.remaining_requests = headers.get("x-ratelimit-remaining") or headers.get(
            "x-requests-remaining"
        )
        self.requests_used = headers.get("x-requests-used")

    async def _get(self, endpoint: str, params: Dict = None) -> Optional[Any]:
        if not self.session:
            raise RuntimeError("Session not started.")

        base_params = {"apiKey": self.api_key}
        if params:
            base_params.update(params)

        url = f"{ODDS_API_IO_BASE_URL}/{endpoint.lstrip('/')}"
        for attempt in range(3):
            try:
                async with self.session.get(url, params=base_params) as resp:
                    self._update_quota(resp.headers)
                    if resp.status == 200:
                        return await resp.json()
                    if resp.status in (401, 403):
                        logger.error("Odds-API.io key invalid atau quota habis.")
                        return None
                    if resp.status == 429:
                        retry_after = resp.headers.get("Retry-After")
                        try:
                            wait = (
                                float(retry_after)
                                if retry_after
                                else 10 * (attempt + 1)
                            )
                        except ValueError:
                            wait = 10 * (attempt + 1)
                        logger.warning(
                            "Odds-API.io rate limit (attempt %s/3). Menunggu %.1fs...",
                            attempt + 1,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    text = await resp.text()
                    logger.warning("Odds-API.io HTTP %s: %s", resp.status, text[:200])
                    return None
            except aiohttp.ClientError as e:
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    logger.error("Odds-API.io network error: %s", e)
                    return None
        return None

    async def fetch_all_football_markets(self) -> List[Dict]:
        from datetime import timedelta

        freshness = OddsFreshness()
        now = datetime.now(timezone.utc)
        from_dt = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        to_dt = (
            (now + timedelta(hours=LOOKAHEAD_HOURS))
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

        events = await self._get(
            "events",
            {
                "sport": "football",
                "status": "pending",
                "from": from_dt,
                "to": to_dt,
                "bookmaker": self.bookmakers.split(",")[0],
                "limit": ODDS_API_IO_MAX_EVENTS,
            },
        )
        if not isinstance(events, list):
            self.fetch_health = build_fetch_health(
                events=[],
                total_leagues=1,
                freshness=freshness,
                api_quota_remaining=self.remaining_requests,
            )
            cache_events([], self.fetch_health)
            return []

        all_events: List[Dict] = []
        for event in events[:ODDS_API_IO_MAX_EVENTS]:
            event_id = event.get("id")
            if not event_id:
                continue
            odds_payload = await self._get(
                "odds",
                {
                    "eventId": event_id,
                    "bookmakers": self.bookmakers,
                },
            )
            normalized = (
                self._normalize_event(odds_payload)
                if isinstance(odds_payload, dict)
                else None
            )
            if normalized and normalized.get("hours_until", -1) > 0:
                all_events.append(normalized)

        freshness.annotate_events(all_events)
        all_events.sort(key=lambda x: x.get("hours_until", 99))

        self.fetch_health = build_fetch_health(
            events=all_events,
            total_leagues=1,
            freshness=freshness,
            api_quota_remaining=self.remaining_requests,
        )
        cache_events(all_events, self.fetch_health)
        return all_events

    def _normalize_event(self, event: Dict) -> Optional[Dict]:
        try:
            commence_time = event.get("date", "")
            commence_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None

        bookmakers = []
        raw_books = event.get("bookmakers") or {}
        if not isinstance(raw_books, dict):
            return None

        for book_name, markets in raw_books.items():
            normalized_markets = self._normalize_markets(
                markets,
                home_team=event.get("home", ""),
                away_team=event.get("away", ""),
            )
            if normalized_markets:
                bookmakers.append(
                    {
                        "key": self._book_key(book_name),
                        "title": book_name,
                        "last_update": commence_time,
                        "markets": normalized_markets,
                    }
                )

        if not bookmakers:
            return None

        now = datetime.now(timezone.utc)
        hours_until = (commence_dt - now).total_seconds() / 3600
        league = event.get("league") or {}
        sport = event.get("sport") or {}
        league_name = league.get("name") or league.get("slug") or "Football"

        return {
            "id": str(event.get("id", "")),
            "sport_key": f"odds_api_io_{league.get('slug', 'football')}",
            "sport_name": league_name,
            "sport_title": league_name,
            "commence_time": commence_dt.isoformat(),
            "hours_until": round(hours_until, 2),
            "commence_utc": commence_dt,
            "is_live": False,
            "home_team": event.get("home", ""),
            "away_team": event.get("away", ""),
            "bookmakers": bookmakers,
            "source_sport": sport.get("name", "Football"),
        }

    def _normalize_markets(
        self, markets: Any, home_team: str, away_team: str
    ) -> List[Dict]:
        if not isinstance(markets, list):
            return []
        normalized = []
        for market in markets:
            market_name = str(market.get("name", "")).lower()
            odds_rows = market.get("odds") or []
            if not odds_rows:
                continue
            if market_name in {
                "ml",
                "moneyline",
                "match winner",
                "match result",
                "1x2",
            }:
                outcomes = self._outcomes_from_ml(odds_rows[0], home_team, away_team)
                if outcomes:
                    normalized.append({"key": "h2h", "outcomes": outcomes})
            elif any(
                k in market_name for k in ("asian handicap", "spread", "handicap")
            ):
                for row in odds_rows:
                    outcomes = self._outcomes_from_spread(row, home_team, away_team)
                    if outcomes:
                        normalized.append({"key": "spreads", "outcomes": outcomes})
            elif any(k in market_name for k in ("over/under", "total")):
                for row in odds_rows:
                    outcomes = self._outcomes_from_totals(row)
                    if outcomes:
                        normalized.append({"key": "totals", "outcomes": outcomes})
            elif any(k in market_name for k in ("both teams", "btts")):
                outcomes = self._outcomes_from_btts(odds_rows[0])
                if outcomes:
                    normalized.append({"key": "btts", "outcomes": outcomes})
        return normalized

    def _outcomes_from_ml(
        self, row: Dict, home_team: str, away_team: str
    ) -> List[Dict]:
        outcomes = []
        for api_key, name in (
            ("home", home_team),
            ("draw", "Draw"),
            ("away", away_team),
        ):
            price = self._parse_price(row.get(api_key))
            if price:
                outcomes.append({"name": name, "price": price})
        return outcomes

    def _outcomes_from_spread(
        self, row: Dict, home_team: str, away_team: str
    ) -> List[Dict]:
        hdp = self._parse_point(row.get("hdp", row.get("handicap")))
        home = self._parse_price(row.get("home"))
        away = self._parse_price(row.get("away"))
        outcomes = []
        if hdp is not None and home:
            outcomes.append({"name": home_team, "price": home, "point": hdp})
        if hdp is not None and away:
            outcomes.append({"name": away_team, "price": away, "point": -hdp})
        return outcomes

    def _outcomes_from_totals(self, row: Dict) -> List[Dict]:
        point = self._parse_point(row.get("max", row.get("total", row.get("hdp"))))
        over = self._parse_price(row.get("over"))
        under = self._parse_price(row.get("under"))
        outcomes = []
        if point is not None and over:
            outcomes.append({"name": "Over", "price": over, "point": point})
        if point is not None and under:
            outcomes.append({"name": "Under", "price": under, "point": point})
        return outcomes

    def _outcomes_from_btts(self, row: Dict) -> List[Dict]:
        yes = self._parse_price(row.get("yes", row.get("Yes")))
        no = self._parse_price(row.get("no", row.get("No")))
        outcomes = []
        if yes:
            outcomes.append({"name": "Yes", "price": yes})
        if no:
            outcomes.append({"name": "No", "price": no})
        return outcomes

    @staticmethod
    def _parse_price(value: Any) -> Optional[float]:
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        return price if price > 1.0 else None

    @staticmethod
    def _parse_point(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _book_key(name: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


class MockAPIClient:

    def __init__(self):
        self.fetch_health: Optional[FetchHealthReport] = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def fetch_all_football_markets(self) -> List[Dict]:
        import random
        from datetime import timedelta

        freshness = OddsFreshness()
        random.seed(42)
        now = datetime.now(timezone.utc)

        sample_matches = [
            ("Manchester City", "Arsenal", "soccer_epl", "Premier League", 1.5),
            ("Real Madrid", "Barcelona", "soccer_spain_la_liga", "La Liga", 3.0),
            (
                "Bayern Munich",
                "Borussia Dortmund",
                "soccer_germany_bundesliga",
                "Bundesliga",
                5.0,
            ),
            ("Juventus", "Inter Milan", "soccer_italy_serie_a", "Serie A", 6.0),
            ("PSG", "Marseille", "soccer_france_ligue_one", "Ligue 1", 7.5),
            ("Liverpool", "Chelsea", "soccer_epl", "Premier League", 8.0),
            ("Atletico Madrid", "Sevilla", "soccer_spain_la_liga", "La Liga", 9.0),
            ("Ajax", "PSV", "soccer_netherlands_eredivisie", "Eredivisie", 10.0),
            (
                "Porto",
                "Benfica",
                "soccer_portugal_primeira_liga",
                "Primeira Liga",
                11.0,
            ),
            (
                "Galatasaray",
                "Fenerbahce",
                "soccer_turkey_super_league",
                "Super Lig",
                11.5,
            ),
        ]

        events = []
        bookmakers_list = [
            "pinnacle",
            "bet365",
            "williamhill",
            "bwin",
            "unibet_eu",
            "betway",
        ]

        for home, away, sport_key, sport_name, hours in sample_matches:
            if hours > LOOKAHEAD_HOURS:
                continue
            commence = now + timedelta(hours=hours)

            home_true = random.uniform(0.30, 0.55)
            draw_true = random.uniform(0.20, 0.28)
            away_true = 1 - home_true - draw_true
            ou_line = random.choice([2.5, 3.5])
            over_true = random.uniform(0.45, 0.55)
            home_spread = random.choice([-1.5, -0.5, 0.5, 1.5])
            spread_home_true = random.uniform(0.40, 0.60)
            btts_yes_true = random.uniform(0.40, 0.60)

            bookmaker_odds_list = []
            for book in bookmakers_list:
                vig = 1.03 if book == "pinnacle" else random.uniform(1.04, 1.10)
                noise = lambda: random.uniform(0.92, 1.08)

                h_odds = round((1 / (home_true * vig)) * noise(), 2)
                d_odds = round((1 / (draw_true * vig)) * noise(), 2)
                a_odds = round((1 / (away_true * vig)) * noise(), 2)
                over_odds = round((1 / (over_true * vig)) * noise(), 2)
                under_odds = round((1 / ((1 - over_true) * vig)) * noise(), 2)
                sh_odds = round((1 / (spread_home_true * vig)) * noise(), 2)
                sa_odds = round((1 / ((1 - spread_home_true) * vig)) * noise(), 2)
                by_odds = round((1 / (btts_yes_true * vig)) * noise(), 2)
                bn_odds = round((1 / ((1 - btts_yes_true) * vig)) * noise(), 2)

                bookmaker_odds_list.append(
                    {
                        "key": book,
                        "title": book.replace("_", " ").title(),
                        "last_update": commence.isoformat(),
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": home, "price": h_odds},
                                    {"name": "Draw", "price": d_odds},
                                    {"name": away, "price": a_odds},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {
                                        "name": "Over",
                                        "price": over_odds,
                                        "point": ou_line,
                                    },
                                    {
                                        "name": "Under",
                                        "price": under_odds,
                                        "point": ou_line,
                                    },
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {
                                        "name": home,
                                        "price": sh_odds,
                                        "point": home_spread,
                                    },
                                    {
                                        "name": away,
                                        "price": sa_odds,
                                        "point": -home_spread,
                                    },
                                ],
                            },
                            {
                                "key": "btts",
                                "outcomes": [
                                    {"name": "Yes", "price": by_odds},
                                    {"name": "No", "price": bn_odds},
                                ],
                            },
                        ],
                    }
                )

            events.append(
                {
                    "id": f"mock_{sport_key}_{home.lower().replace(' ', '_')}",
                    "sport_key": sport_key,
                    "sport_name": sport_name,
                    "sport_title": sport_name,
                    "commence_time": commence.isoformat(),
                    "hours_until": hours,
                    "is_live": False,
                    "home_team": home,
                    "away_team": away,
                    "bookmakers": bookmaker_odds_list,
                }
            )

        freshness.annotate_events(events)

        from fetch_health import LeagueFetchResult

        sport_keys_seen = {}
        for e in events:
            sk = e.get("sport_key", "mock")
            if sk not in sport_keys_seen:
                sport_keys_seen[sk] = e.get("sport_name", sk)
        mock_league_results = [
            LeagueFetchResult(sport_key=sk, sport_name=sn, events_found=1, success=True)
            for sk, sn in sport_keys_seen.items()
        ]
        self.fetch_health = build_fetch_health(
            events=events,
            total_leagues=len(mock_league_results),
            league_results=mock_league_results,
            freshness=freshness,
            api_quota_remaining="MOCK",
        )
        cache_events(events, self.fetch_health)
        return events
