import asyncio
import difflib
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
from understat import Understat

logger = logging.getLogger(__name__)

UNDERSTAT_SUPPORTED_LEAGUES = {
    "EPL",
    "La Liga",
    "La_Liga",
    "Bundesliga",
    "German Bundesliga",
    "Serie A",
    "Italian Serie A",
    "Ligue 1",
    "French Ligue 1",
    "RFPL",
    "Russian Premier League",
}

UNDERSTAT_LEAGUE_ALIASES = {
    "premier league",
    "english premier league",
    "la liga",
    "laliga",
    "bundesliga",
    "german bundesliga",
    "serie a",
    "italian serie a",
    "ligue 1",
    "french ligue 1",
    "russian premier league",
}

UNDERSTAT_SUPPORTED_SPORT_KEYS = {
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_russia_premier_league",
}

UNDERSTAT_SUPPORTED_TEAMS = {
    "Arsenal", "Aston_Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal_Palace", "Everton", "Fulham",
    "Ipswich", "Leicester", "Liverpool", "Manchester_City",
    "Manchester_United", "Newcastle_United", "Nottingham_Forest",
    "Southampton", "Tottenham", "West_Ham", "Wolverhampton_Wanderers",
    "Barcelona", "Atletico_Madrid", "Real_Madrid", "Sevilla", "Valencia",
    "Villarreal", "Athletic_Club", "Real_Sociedad", "Real_Betis", "Girona",
    "Bayern_Munich", "Borussia_Dortmund", "Bayer_Leverkusen", "RasenBallsport_Leipzig",
    "Eintracht_Frankfurt", "Stuttgart", "Wolfsburg", "Freiburg", "Union_Berlin",
    "Inter", "Milan", "Roma", "Juventus", "Napoli", "Lazio", "Atalanta",
    "Fiorentina", "Bologna", "Torino", "Udinese", "Genoa", "Monza",
    "Paris_Saint_Germain", "Marseille", "Lyon", "Monaco", "Lille", "Lens",
    "Nice", "Rennes", "Strasbourg", "Toulouse", "Nantes", "Reims",
}

UNDERSTAT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
    "Referer": "https://understat.com/",
    "Accept": "application/json,text/plain,*/*",
    "X-Requested-With": "XMLHttpRequest",
}


def normalize_team_name(name: str) -> str:
    mapping = {
        # === EPL ===
        "Man City": "Manchester_City",
        "Manchester City": "Manchester_City",
        "Man Utd": "Manchester_United",
        "Man United": "Manchester_United",
        "Manchester United": "Manchester_United",
        "Spurs": "Tottenham",
        "Tottenham Hotspur": "Tottenham",
        "Wolves": "Wolverhampton_Wanderers",
        "Wolverhampton": "Wolverhampton_Wanderers",
        "Brighton": "Brighton",
        "Brighton & Hove Albion": "Brighton",
        "Newcastle": "Newcastle_United",
        "Newcastle United": "Newcastle_United",
        "West Ham": "West_Ham",
        "West Ham United": "West_Ham",
        "Nottm Forest": "Nottingham_Forest",
        "Nottingham Forest": "Nottingham_Forest",
        "Aston Villa": "Aston_Villa",
        "Crystal Palace": "Crystal_Palace",
        "Sheffield Utd": "Sheffield_United",
        "Sheffield United": "Sheffield_United",
        "Luton": "Luton",
        "Luton Town": "Luton",
        "Bournemouth": "Bournemouth",
        "Everton": "Everton",
        "Brentford": "Brentford",
        "Fulham": "Fulham",
        "Burnley": "Burnley",
        "Arsenal": "Arsenal",
        "Liverpool": "Liverpool",
        "Chelsea": "Chelsea",
        "Leicester": "Leicester",
        "Leicester City": "Leicester",
        "Southampton": "Southampton",
        "Ipswich": "Ipswich",
        "Ipswich Town": "Ipswich",
        # === La Liga ===
        "Atletico Madrid": "Atletico_Madrid",
        "Atlético Madrid": "Atletico_Madrid",
        "Atletico de Madrid": "Atletico_Madrid",
        "Athletic Bilbao": "Athletic_Club",
        "Athletic Club": "Athletic_Club",
        "Real Sociedad": "Real_Sociedad",
        "Real Betis": "Real_Betis",
        "Celta Vigo": "Celta_Vigo",
        "Rayo Vallecano": "Rayo_Vallecano",
        "CA Osasuna": "Osasuna",
        "Osasuna": "Osasuna",
        "Deportivo Alavés": "Alaves",
        "Alavés": "Alaves",
        "Cadiz CF": "Cadiz",
        "Cadiz": "Cadiz",
        "UD Almería": "Almeria",
        "Almería": "Almeria",
        "UD Las Palmas": "Las_Palmas",
        "Las Palmas": "Las_Palmas",
        "Real Madrid": "Real_Madrid",
        "Barcelona": "Barcelona",
        "Sevilla": "Sevilla",
        "Villarreal": "Villarreal",
        "Valencia": "Valencia",
        "Mallorca": "Mallorca",
        "Girona": "Girona",
        "Getafe": "Getafe",
        "Espanyol": "Espanyol",
        # === Bundesliga ===
        "Bayern Munich": "Bayern_Munich",
        "Bayern München": "Bayern_Munich",
        "FC Bayern": "Bayern_Munich",
        "Borussia Dortmund": "Borussia_Dortmund",
        "Dortmund": "Borussia_Dortmund",
        "Bayer Leverkusen": "Bayer_Leverkusen",
        "Leverkusen": "Bayer_Leverkusen",
        "RB Leipzig": "RasenBallsport_Leipzig",
        "Leipzig": "RasenBallsport_Leipzig",
        "Eintracht Frankfurt": "Eintracht_Frankfurt",
        "Frankfurt": "Eintracht_Frankfurt",
        "Borussia Monchengladbach": "Borussia_M.Gladbach",
        "Borussia M'Gladbach": "Borussia_M.Gladbach",
        "VfB Stuttgart": "Stuttgart",
        "Stuttgart": "Stuttgart",
        "VfL Wolfsburg": "Wolfsburg",
        "Wolfsburg": "Wolfsburg",
        "SC Freiburg": "Freiburg",
        "Freiburg": "Freiburg",
        "Hoffenheim": "Hoffenheim",
        "TSG Hoffenheim": "Hoffenheim",
        "Werder Bremen": "Werder_Bremen",
        "1. FC Union Berlin": "Union_Berlin",
        "Union Berlin": "Union_Berlin",
        "FC Augsburg": "Augsburg",
        "Augsburg": "Augsburg",
        "1. FC Heidenheim": "Heidenheim",
        "Heidenheim": "Heidenheim",
        "SV Darmstadt 98": "Darmstadt_98",
        "Darmstadt": "Darmstadt_98",
        "1. FC Koln": "FC_Cologne",
        "Koln": "FC_Cologne",
        "FC Cologne": "FC_Cologne",
        # === Serie A ===
        "Inter Milan": "Inter",
        "Internazionale": "Inter",
        "Inter": "Inter",
        "AC Milan": "Milan",
        "Milan": "Milan",
        "AS Roma": "Roma",
        "Roma": "Roma",
        "Juventus": "Juventus",
        "SSC Napoli": "Napoli",
        "Napoli": "Napoli",
        "Lazio": "Lazio",
        "SS Lazio": "Lazio",
        "Atalanta": "Atalanta",
        "ACF Fiorentina": "Fiorentina",
        "Fiorentina": "Fiorentina",
        "Bologna": "Bologna",
        "Torino": "Torino",
        "Udinese": "Udinese",
        "Sassuolo": "Sassuolo",
        "Cagliari": "Cagliari",
        "Genoa": "Genoa",
        "CFC Genoa": "Genoa",
        "Hellas Verona": "Verona",
        "Verona": "Verona",
        "Salernitana": "Salernitana",
        "US Salernitana 1919": "Salernitana",
        "Empoli": "Empoli",
        "Lecce": "Lecce",
        "Frosinone": "Frosinone",
        "Monza": "Monza",
        # === Ligue 1 ===
        "PSG": "Paris_Saint_Germain",
        "Paris Saint Germain": "Paris_Saint_Germain",
        "Paris Saint-Germain": "Paris_Saint_Germain",
        "Marseille": "Marseille",
        "Olympique Marseille": "Marseille",
        "Olympique de Marseille": "Marseille",
        "Lyon": "Lyon",
        "Olympique Lyonnais": "Lyon",
        "Olympique Lyon": "Lyon",
        "AS Monaco": "Monaco",
        "Monaco": "Monaco",
        "Lille": "Lille",
        "Lille OSC": "Lille",
        "RC Lens": "Lens",
        "Lens": "Lens",
        "OGC Nice": "Nice",
        "Nice": "Nice",
        "Rennes": "Rennes",
        "Stade Rennais": "Rennes",
        "Strasbourg": "Strasbourg",
        "RC Strasbourg": "Strasbourg",
        "Toulouse": "Toulouse",
        "Toulouse FC": "Toulouse",
        "Nantes": "Nantes",
        "FC Nantes": "Nantes",
        "Reims": "Reims",
        "Stade de Reims": "Reims",
        "Montpellier": "Montpellier",
        "Montpellier HSC": "Montpellier",
        "Le Havre": "Le_Havre",
        "Le Havre AC": "Le_Havre",
        "Lorient": "Lorient",
        "FC Lorient": "Lorient",
        "Metz": "Metz",
        "FC Metz": "Metz",
        "Clermont": "Clermont_Foot",
        "Clermont Foot": "Clermont_Foot",
    }

    if name in mapping:
        return mapping[name]

    known_names = list(mapping.keys()) + list(set(mapping.values()))
    matches = difflib.get_close_matches(name, known_names, n=1, cutoff=0.8)
    if matches:
        best_match = matches[0]
        mapped_val = mapping.get(best_match, best_match)
        logger.debug(
            f"Understat fuzzy match: '{name}' -> '{mapped_val}' (matched '{best_match}')"
        )
        return mapped_val

    return name.replace(" ", "_")


def poisson_probability(lam: float, k: int) -> float:
    return (lam**k * math.exp(-lam)) / math.factorial(k)


class UnderstatEnricher:

    def __init__(self, cache_ttl: int = 300):

        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._consecutive_failures = 0
        self._circuit_open = False

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

    def _is_supported_league(self, league: str, sport_key: str = "") -> bool:
        if sport_key:
            if sport_key in UNDERSTAT_SUPPORTED_SPORT_KEYS:
                return True
        if not league:
            return False
        league_lower = league.lower().replace("-", " ").replace("_", " ").strip()
        return league_lower in UNDERSTAT_LEAGUE_ALIASES or any(
            supported.lower() in league.lower()
            for supported in UNDERSTAT_SUPPORTED_LEAGUES
        )

    def _is_supported_team(self, team_name: str) -> bool:
        return normalize_team_name(team_name) in UNDERSTAT_SUPPORTED_TEAMS

    async def _diagnose_response(self, normalized_name: str, season: str) -> str:
        url = f"https://understat.com/getTeamData/{normalized_name}/{season}"
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(headers=UNDERSTAT_HEADERS, timeout=timeout) as session:
                async with session.get(url, allow_redirects=False) as response:
                    body = await response.text(errors="replace")
                    return (
                        f"status={response.status} content_type={response.headers.get('Content-Type', '')!r} "
                        f"location={response.headers.get('Location', '')!r} body={body[:240]!r}"
                    )
        except Exception as exc:
            return f"diagnostic_error={type(exc).__name__}: {exc}"

    async def get_team_xg_stats(self, team_name: str, season: str) -> Optional[dict]:
        normalized_name = normalize_team_name(team_name)
        cache_key = f"team_stats_{normalized_name}_{season}"

        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        if self._circuit_open:
            logger.warning("Understat circuit open; skipping '%s' for this run", normalized_name)
            return None

        try:
            results = None
            for attempt in range(3):
                try:
                    # Diagnostic probe confirmed getTeamData/Liverpool/2026 returns
                    # HTTP 200 JSON; the original failure was result processing after
                    # `break`, which skipped valid package results entirely.
                    async with aiohttp.ClientSession(headers=UNDERSTAT_HEADERS) as session:
                        understat = Understat(session)
                        results = await understat.get_team_results(normalized_name, season)
                    self._consecutive_failures = 0
                    break
                except json.JSONDecodeError as exc:
                    self._consecutive_failures += 1
                    diagnostic = await self._diagnose_response(normalized_name, season)
                    logger.warning(
                        "Understat JSON response invalid (%s, attempt %d/3): %s; %s",
                        normalized_name, attempt + 1, exc, diagnostic,
                    )
                    if self._consecutive_failures >= 3:
                        self._circuit_open = True
                        logger.warning(
                            "Understat kemungkinan sedang diblokir/down; skip xG enrichment untuk run ini"
                        )
                        self._set_to_cache(cache_key, None)
                        return None
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    self._consecutive_failures += 1
                    if attempt < 2:
                        logger.warning(f"Understat API error ({normalized_name}): {e}. Retrying in {2**attempt}s...")
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"Understat API failed after 3 attempts for '{normalized_name}': {e}")
                        self._set_to_cache(cache_key, None)
                        return None

            if not results:
                logger.warning(
                    f"Understat: No results found for team '{normalized_name}' (Original: '{team_name}', season: '{season}')"
                )
                self._set_to_cache(cache_key, None)
                return None

            played_matches = [m for m in results if m.get("isResult") is True]
            last_5 = played_matches[-5:]

            if not last_5:
                logger.warning(
                    f"Understat: No played matches found for team '{normalized_name}' (season: '{season}')"
                )
                return None

            total_xg_for = 0.0
            total_xg_against = 0.0
            total_goals_for = 0.0
            total_goals_against = 0.0

            for match in last_5:
                side = match.get("side")
                if side == "h":
                    total_xg_for += float(match["xG"]["h"])
                    total_xg_against += float(match["xG"]["a"])
                    total_goals_for += float(match["goals"]["h"])
                    total_goals_against += float(match["goals"]["a"])
                else:
                    total_xg_for += float(match["xG"]["a"])
                    total_xg_against += float(match["xG"]["h"])
                    total_goals_for += float(match["goals"]["a"])
                    total_goals_against += float(match["goals"]["h"])

            count = len(last_5)
            stats = {
                "avg_xg_for": round(total_xg_for / count, 2),
                "avg_xg_against": round(total_xg_against / count, 2),
                "avg_goals_for": round(total_goals_for / count, 2),
                "avg_goals_against": round(total_goals_against / count, 2),
            }

            self._set_to_cache(cache_key, stats)
            logger.info(
                f"Understat: Successfully retrieved xG stats for '{normalized_name}'"
            )
            return stats

        except Exception as e:
            logger.error(
                f"Understat: Failed to fetch data for '{team_name}' gracefully: {e}"
            )
            return None

    async def get_match_xg_context(
        self, home_team: str, away_team: str, season: str
    ) -> Optional[dict]:
        home_stats = await self.get_team_xg_stats(home_team, season)
        away_stats = await self.get_team_xg_stats(away_team, season)

        if not home_stats and not away_stats:
            logger.warning(
                f"Understat: Could not get xG context for '{home_team} vs {away_team}' (Neither team found)"
            )
            return None

        home_xg_proj = 1.0
        away_xg_proj = 1.0

        if home_stats and away_stats:
            home_xg_proj = (home_stats["avg_xg_for"] + away_stats["avg_xg_against"]) / 2
            away_xg_proj = (away_stats["avg_xg_for"] + home_stats["avg_xg_against"]) / 2
        elif home_stats:
            home_xg_proj = home_stats["avg_xg_for"]
            away_xg_proj = home_stats["avg_xg_against"]
        elif away_stats:
            home_xg_proj = away_stats["avg_xg_against"]
            away_xg_proj = away_stats["avg_xg_for"]

        home_win_prob = 0.0
        away_win_prob = 0.0
        draw_prob = 0.0

        for h in range(7):
            for a in range(7):
                prob = poisson_probability(home_xg_proj, h) * poisson_probability(
                    away_xg_proj, a
                )
                if h > a:
                    home_win_prob += prob
                elif a > h:
                    away_win_prob += prob
                else:
                    draw_prob += prob

        total = home_win_prob + away_win_prob + draw_prob
        if total > 0:
            home_win_prob /= total
            away_win_prob /= total
            draw_prob /= total

        context = {
            "home_stats": home_stats,
            "away_stats": away_stats,
            "xg_implied_home_prob": round(home_win_prob, 3),
            "xg_implied_away_prob": round(away_win_prob, 3),
            "xg_implied_draw_prob": round(draw_prob, 3),
        }

        return context

    async def enrich_matches(self, matches: list[dict], season: str) -> list[dict]:
        for match in matches:
            home_team = match.get("home_team")
            away_team = match.get("away_team")
            league = match.get(
                "sport_title", match.get("sport_name", match.get("league", ""))
            )
            sport_key = match.get("sport_key", "")

            if not home_team or not away_team:
                logger.warning(
                    "Understat: Match dictionary missing 'home_team' or 'away_team'. Skipping."
                )
                match["xg_context"] = None
                continue

            if not self._is_supported_league(league, sport_key):
                logger.debug(
                    f"Understat: Skipping unsupported league '{league}' for match {home_team} vs {away_team}"
                )
                match["xg_context"] = None
                continue

            if not self._is_supported_team(home_team) or not self._is_supported_team(away_team):
                logger.debug(
                    "Understat: Skipping teams outside allowlist: %s vs %s",
                    home_team,
                    away_team,
                )
                match["xg_context"] = None
                continue

            if self._circuit_open:
                match["xg_context"] = None
                continue

            logger.info(
                f"Understat: Enriching match '{home_team} vs {away_team}' with xG data..."
            )
            xg_context = await self.get_match_xg_context(home_team, away_team, season)
            match["xg_context"] = xg_context

            if xg_context is None:
                logger.info(
                    f"Understat: Fallback to None for xG context: '{home_team} vs {away_team}'"
                )
            else:
                logger.info(
                    f"Understat: Successfully enriched match '{home_team} vs {away_team}'"
                )

        return matches


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    async def run_example():
        enricher = UnderstatEnricher(cache_ttl=300)

        matches = [
            {"home_team": "Man City", "away_team": "Arsenal"},
            {"home_team": "Nottm Forest", "away_team": "Wolves"},
            {"home_team": "Fake Team ABC", "away_team": "Another Fake Team"},
        ]

        season_year = "2023"
        print(f"--- Fetching Understat Data for Season {season_year} ---")
        enriched_matches = await enricher.enrich_matches(matches, season=season_year)

        import json

        print("\n--- Output ---")
        print(json.dumps(enriched_matches, indent=2))

    asyncio.run(run_example())
