
import difflib
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


FBREF_LEAGUES: Dict[str, Dict[str, Any]] = {
    "soccer_indonesia_liga1": {"comp_id": 37, "name": "Liga-1", "country": "ID"},
    "soccer_japan_j_league": {"comp_id": 25, "name": "J1-League", "country": "JP"},
    "soccer_south_korea_kleague1": {
        "comp_id": 55,
        "name": "K-League-1",
        "country": "KR",
    },
    "soccer_australia_aleague": {
        "comp_id": 48,
        "name": "A-League-Men",
        "country": "AU",
    },
    "soccer_china_superleague": {
        "comp_id": 62,
        "name": "Chinese-Super-League",
        "country": "CN",
    },
    "soccer_saudi_premier_league": {
        "comp_id": 70,
        "name": "Saudi-Pro-League",
        "country": "SA",
    },
    "soccer_usa_mls": {"comp_id": 22, "name": "Major-League-Soccer", "country": "US"},
    "soccer_brazil_campeonato": {"comp_id": 24, "name": "Serie-A", "country": "BR"},
    "soccer_argentina_primera_division": {
        "comp_id": 21,
        "name": "Primera-Division",
        "country": "AR",
    },
    "soccer_mexico_ligamx": {"comp_id": 31, "name": "Liga-MX", "country": "MX"},
    "soccer_chile_primera_division": {
        "comp_id": 35,
        "name": "Primera-Division",
        "country": "CL",
    },
    "soccer_colombia_primera_a": {"comp_id": 41, "name": "Primera-A", "country": "CO"},
    "soccer_epl": {"comp_id": 9, "name": "Premier-League", "country": "EN"},
    "soccer_spain_la_liga": {"comp_id": 12, "name": "La-Liga", "country": "ES"},
    "soccer_germany_bundesliga": {"comp_id": 20, "name": "Bundesliga", "country": "DE"},
    "soccer_italy_serie_a": {"comp_id": 11, "name": "Serie-A", "country": "IT"},
    "soccer_france_ligue_one": {"comp_id": 13, "name": "Ligue-1", "country": "FR"},
    "soccer_russia_premier_league": {
        "comp_id": 30,
        "name": "Russian-Premier-League",
        "country": "RU",
    },
    "soccer_netherlands_eredivisie": {
        "comp_id": 23,
        "name": "Eredivisie",
        "country": "NL",
    },
    "soccer_portugal_primeira_liga": {
        "comp_id": 32,
        "name": "Primeira-Liga",
        "country": "PT",
    },
    "soccer_turkey_super_league": {"comp_id": 26, "name": "Super-Lig", "country": "TR"},
    "soccer_belgium_first_div": {
        "comp_id": 37,
        "name": "Belgian-Pro-League",
        "country": "BE",
    },
    "soccer_efl_champ": {"comp_id": 10, "name": "Championship", "country": "EN"},
    "soccer_scotland_premiership": {
        "comp_id": 40,
        "name": "Scottish-Premiership",
        "country": "SC",
    },
    "soccer_austria_football_bundesliga": {
        "comp_id": 56,
        "name": "Austrian-Bundesliga",
        "country": "AT",
    },
    "soccer_denmark_superliga": {
        "comp_id": 50,
        "name": "Danish-Superliga",
        "country": "DK",
    },
    "soccer_norway_eliteserien": {
        "comp_id": 28,
        "name": "Eliteserien",
        "country": "NO",
    },
    "soccer_sweden_allsvenskan": {
        "comp_id": 29,
        "name": "Allsvenskan",
        "country": "SE",
    },
    "soccer_swiss_superleague": {
        "comp_id": 57,
        "name": "Super-League",
        "country": "CH",
    },
    "soccer_greece_super_league": {
        "comp_id": 27,
        "name": "Super-League",
        "country": "GR",
    },
    "soccer_poland_ekstraklasa": {
        "comp_id": 36,
        "name": "Ekstraklasa",
        "country": "PL",
    },
}


FBREF_NAME_TO_KEY: Dict[str, str] = {}
for _key, _info in FBREF_LEAGUES.items():
    _readable = _info["name"].replace("-", " ").lower()
    FBREF_NAME_TO_KEY[_readable] = _key


FBREF_TEAM_ALIASES: Dict[str, str] = {
    "Persija Jakarta": "Persija Jakarta",
    "Persija": "Persija Jakarta",
    "Persib Bandung": "Persib Bandung",
    "Persib": "Persib Bandung",
    "Bali United": "Bali United",
    "Arema FC": "Arema",
    "Arema": "Arema",
    "PSM Makassar": "PSM Makassar",
    "PSM": "PSM Makassar",
    "PSIS Semarang": "PSIS Semarang",
    "PSIS": "PSIS Semarang",
    "Persebaya Surabaya": "Persebaya Surabaya",
    "Persebaya": "Persebaya Surabaya",
    "Borneo FC": "Borneo",
    "Madura United": "Madura United",
    "Bhayangkara FC": "Bhayangkara",
    "Dewa United": "Dewa United",
    "Man City": "Manchester City",
    "Man Utd": "Manchester Utd",
    "Man United": "Manchester Utd",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Brighton & Hove Albion": "Brighton",
    "Nottm Forest": "Nott'ham Forest",
    "Nottingham Forest": "Nott'ham Forest",
    "West Ham United": "West Ham",
    "Sheffield Utd": "Sheffield Utd",
    "Newcastle United": "Newcastle Utd",
    "Newcastle": "Newcastle Utd",
    "Leicester City": "Leicester City",
    "Ipswich Town": "Ipswich Town",
    "Luton Town": "Luton Town",
}


_fbref_known_teams: Dict[int, List[str]] = {}


def normalize_fbref_team(name: str, comp_id: int = 0) -> str:
    if name in FBREF_TEAM_ALIASES:
        return FBREF_TEAM_ALIASES[name]

    known = _fbref_known_teams.get(comp_id, [])
    all_candidates = (
        list(FBREF_TEAM_ALIASES.keys()) + list(set(FBREF_TEAM_ALIASES.values())) + known
    )

    matches = difflib.get_close_matches(name, all_candidates, n=1, cutoff=0.75)
    if matches:
        best = matches[0]
        mapped = FBREF_TEAM_ALIASES.get(best, best)
        logger.debug(f"FBref fuzzy match: '{name}' -> '{mapped}' (via '{best}')")
        return mapped

    return name


def _poisson(lam: float, k: int) -> float:
    return (lam**k * math.exp(-lam)) / math.factorial(k)


class FBrefEnricher:

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
    REQUEST_DELAY = 1.5

    def __init__(self, cache_ttl: int = 600):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_request_ts: float = 0.0
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self.USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml",
                "Referer": "https://www.google.com/",
            }
        )

    def _get_from_cache(self, key: str) -> Optional[Any]:
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self.cache_ttl:
                return entry["data"]
            else:
                del self._cache[key]
        return None

    def _set_to_cache(self, key: str, data: Any) -> None:
        self._cache[key] = {"data": data, "timestamp": time.time()}

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_ts = time.time()

    def _build_url(self, comp_id: int, league_name: str, season: str) -> str:

        if "-" not in season:
            season = f"{season}-{int(season) + 1}"
        return (
            f"https://fbref.com/en/comps/{comp_id}"
            f"/{season}/schedule/{season}-{league_name}-Scores-and-Fixtures"
        )

    def _fetch_page(self, url: str) -> Optional[str]:
        self._rate_limit()
        try:
            resp = self._session.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 429:
                logger.warning("FBref: Rate limited (429). Backing off 10s...")
                time.sleep(10)
                return None
            else:
                logger.warning(f"FBref: HTTP {resp.status_code} for {url}")
                return None
        except requests.RequestException as e:
            logger.error(f"FBref: Request failed for {url}: {e}")
            return None

    def _parse_fixtures_table(self, html: str, comp_id: int) -> List[Dict[str, Any]]:

        cleaned = html.replace("<!--", "").replace("-->", "")
        soup = BeautifulSoup(cleaned, "html.parser")

        table = soup.find("table", {"id": re.compile(r"sched.*overall")})
        if not table:
            table = soup.find("table", {"class": "stats_table"})
        if not table:
            logger.warning("FBref: Could not find fixtures table in HTML")
            return []

        rows = table.find("tbody")
        if not rows:
            return []

        matches = []
        team_names_seen: List[str] = []

        for tr in rows.find_all("tr"):
            if tr.get("class") and ("spacer" in tr["class"] or "thead" in tr["class"]):
                continue

            cells = tr.find_all(["td", "th"])
            if len(cells) < 5:
                continue

            def _cell(stat: str) -> Optional[str]:
                c = tr.find(["td", "th"], {"data-stat": stat})
                return c.get_text(strip=True) if c else None

            date_str = _cell("date")
            home_team = _cell("home_team") or _cell("squad_a") or _cell("team_a")
            away_team = _cell("away_team") or _cell("squad_b") or _cell("team_b")
            home_xg = _cell("home_xg") or _cell("xg_a")
            away_xg = _cell("away_xg") or _cell("xg_b")
            score = _cell("score")

            if not home_team or not away_team:
                continue

            if home_team not in team_names_seen:
                team_names_seen.append(home_team)
            if away_team not in team_names_seen:
                team_names_seen.append(away_team)

            if not score or score == "":
                continue

            try:
                h_xg = float(home_xg) if home_xg else None
                a_xg = float(away_xg) if away_xg else None
            except (ValueError, TypeError):
                h_xg, a_xg = None, None

            if h_xg is not None and a_xg is not None:
                matches.append(
                    {
                        "date": date_str or "",
                        "home_team": home_team.strip(),
                        "away_team": away_team.strip(),
                        "home_xg": h_xg,
                        "away_xg": a_xg,
                    }
                )

        if team_names_seen:
            _fbref_known_teams[comp_id] = team_names_seen

        return matches

    def get_league_matches(
        self, sport_key: str, season: str
    ) -> Optional[List[Dict[str, Any]]]:
        league_info = FBREF_LEAGUES.get(sport_key)
        if not league_info:
            logger.debug(f"FBref: No mapping for sport_key '{sport_key}'")
            return None

        cache_key = f"fbref_league_{sport_key}_{season}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        comp_id = league_info["comp_id"]
        league_name = league_info["name"]

        url = self._build_url(comp_id, league_name, season)
        logger.info(f"FBref: Fetching {url}")

        html = self._fetch_page(url)
        if not html:
            self._set_to_cache(cache_key, [])
            return None

        matches = self._parse_fixtures_table(html, comp_id)
        logger.info(f"FBref: Parsed {len(matches)} completed matches for {sport_key}")

        self._set_to_cache(cache_key, matches)
        return matches

    def get_team_xg_stats(
        self, team_name: str, sport_key: str, season: str
    ) -> Optional[Dict[str, float]]:
        league_info = FBREF_LEAGUES.get(sport_key)
        comp_id = league_info["comp_id"] if league_info else 0
        normalized = normalize_fbref_team(team_name, comp_id)

        cache_key = f"fbref_team_{normalized}_{sport_key}_{season}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        all_matches = self.get_league_matches(sport_key, season)
        if not all_matches:
            self._set_to_cache(cache_key, None)
            return None

        all_team_names = list(
            {m["home_team"] for m in all_matches}
            | {m["away_team"] for m in all_matches}
        )
        best = difflib.get_close_matches(normalized, all_team_names, n=1, cutoff=0.65)
        fbref_name = best[0] if best else normalized

        team_matches = []
        for m in all_matches:
            if m["home_team"] == fbref_name or m["away_team"] == fbref_name:
                team_matches.append(m)

        if not team_matches:
            logger.warning(
                f"FBref: No matches found for '{team_name}' (normalized: '{normalized}', fbref: '{fbref_name}')"
            )
            self._set_to_cache(cache_key, None)
            return None

        last_5 = team_matches[-5:]

        xg_for = xg_against = 0.0
        for m in last_5:
            if m["home_team"] == fbref_name:
                xg_for += m["home_xg"]
                xg_against += m["away_xg"]
            else:
                xg_for += m["away_xg"]
                xg_against += m["home_xg"]

        count = len(last_5)
        stats = {
            "avg_xg_for": round(xg_for / count, 2),
            "avg_xg_against": round(xg_against / count, 2),
            "avg_goals_for": round(xg_for / count, 2),
            "avg_goals_against": round(xg_against / count, 2),
        }

        self._set_to_cache(cache_key, stats)
        logger.info(f"FBref: xG stats for '{fbref_name}': {stats}")
        return stats

    def get_match_xg_context(
        self, home_team: str, away_team: str, sport_key: str, season: str
    ) -> Optional[Dict[str, Any]]:
        home_stats = self.get_team_xg_stats(home_team, sport_key, season)
        away_stats = self.get_team_xg_stats(away_team, sport_key, season)

        if not home_stats and not away_stats:
            return None

        home_xg_proj = away_xg_proj = 1.0
        if home_stats and away_stats:
            home_xg_proj = (home_stats["avg_xg_for"] + away_stats["avg_xg_against"]) / 2
            away_xg_proj = (away_stats["avg_xg_for"] + home_stats["avg_xg_against"]) / 2
        elif home_stats:
            home_xg_proj = home_stats["avg_xg_for"]
            away_xg_proj = home_stats["avg_xg_against"]
        elif away_stats:
            home_xg_proj = away_stats["avg_xg_against"]
            away_xg_proj = away_stats["avg_xg_for"]

        hw = aw = dr = 0.0
        for h in range(7):
            for a in range(7):
                p = _poisson(home_xg_proj, h) * _poisson(away_xg_proj, a)
                if h > a:
                    hw += p
                elif a > h:
                    aw += p
                else:
                    dr += p

        total = hw + aw + dr
        if total > 0:
            hw /= total
            aw /= total
            dr /= total

        return {
            "home_stats": home_stats,
            "away_stats": away_stats,
            "xg_implied_home_prob": round(hw, 3),
            "xg_implied_away_prob": round(aw, 3),
            "xg_implied_draw_prob": round(dr, 3),
        }

    def is_supported(self, sport_key: str) -> bool:
        return sport_key in FBREF_LEAGUES

    def enrich_matches(self, matches: List[Dict], season: str) -> List[Dict]:
        for match in matches:
            if match.get("xg_context") is not None:
                continue

            home_team = match.get("home_team")
            away_team = match.get("away_team")
            sport_key = match.get("sport_key", "")

            if not home_team or not away_team:
                continue

            if not self.is_supported(sport_key):
                logger.debug(
                    f"FBref: No mapping for '{sport_key}', skipping {home_team} vs {away_team}"
                )
                continue

            logger.info(
                f"FBref: Enriching '{home_team} vs {away_team}' ({sport_key})..."
            )
            try:
                xg_ctx = self.get_match_xg_context(
                    home_team, away_team, sport_key, season
                )
                match["xg_context"] = xg_ctx
                if xg_ctx:
                    logger.info(f"FBref: ✓ Enriched '{home_team} vs {away_team}'")
                else:
                    logger.info(f"FBref: No xG data for '{home_team} vs {away_team}'")
            except Exception as e:
                logger.error(f"FBref: Error enriching {home_team} vs {away_team}: {e}")
                match["xg_context"] = None

        return matches


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    import json

    enricher = FBrefEnricher(cache_ttl=300)

    print("--- Testing FBref: Liga 1 Indonesia ---")
    matches_data = enricher.get_league_matches("soccer_indonesia_liga1", "2025-2026")
    if matches_data:
        print(f"Found {len(matches_data)} completed matches")
        print(json.dumps(matches_data[:3], indent=2))
    else:
        print("No data returned (season may not have started yet)")

    print("\n--- Testing team xG stats ---")
    stats = enricher.get_team_xg_stats(
        "Persija Jakarta", "soccer_indonesia_liga1", "2025-2026"
    )
    print(f"Persija stats: {json.dumps(stats, indent=2) if stats else 'N/A'}")

    print("\n--- Testing match context ---")
    ctx = enricher.get_match_xg_context(
        "Persija Jakarta", "Persib Bandung", "soccer_indonesia_liga1", "2025-2026"
    )
    print(f"Context: {json.dumps(ctx, indent=2) if ctx else 'N/A'}")
