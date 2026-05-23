import asyncio
import difflib
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


def normalize_team_name(name: str) -> str:
    mapping = {
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

    def _is_supported_league(self, league: str) -> bool:
        if not league:
            return True
        return any(
            supported.lower() in league.lower()
            for supported in UNDERSTAT_SUPPORTED_LEAGUES
        )

    async def get_team_xg_stats(self, team_name: str, season: str) -> Optional[dict]:
        normalized_name = normalize_team_name(team_name)
        cache_key = f"team_stats_{normalized_name}_{season}"

        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            async with aiohttp.ClientSession() as session:
                understat = Understat(session)
                results = await understat.get_team_results(normalized_name, season)

                if not results:
                    logger.warning(
                        f"Understat: No results found for team '{normalized_name}' (Original: '{team_name}')"
                    )
                    self._set_to_cache(cache_key, None)
                    return None

                played_matches = [m for m in results if m.get("isResult") is True]
                last_5 = played_matches[-5:]

                if not last_5:
                    logger.warning(
                        f"Understat: No played matches found for team '{normalized_name}'"
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

            if not home_team or not away_team:
                logger.warning(
                    "Understat: Match dictionary missing 'home_team' or 'away_team'. Skipping."
                )
                match["xg_context"] = None
                continue

            if league and not self._is_supported_league(league):
                logger.debug(
                    f"Understat: Skipping unsupported league '{league}' for match {home_team} vs {away_team}"
                )
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
