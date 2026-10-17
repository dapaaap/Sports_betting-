import argparse
import asyncio
import logging
import os
import random
import sys
from datetime import datetime
from typing import List, Optional

from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer import BettingAnalyzer, MatchAnalysis
from api_client import MockAPIClient, OddsAPIClient, OddsAPIIOClient
from config import (
    ANALYSIS_CONFIG,
    FBREF_ENABLED,
    FIXTURE_VALIDATION_API_KEY,
    FIXTURE_VALIDATION_CACHE_TTL_SECONDS,
    FIXTURE_VALIDATION_ENABLED,
    FIXTURE_VALIDATION_ENDPOINT,
    FIXTURE_VALIDATION_KICKOFF_TOLERANCE_MINUTES,
    FIXTURE_VALIDATION_LEAGUE_THRESHOLD,
    FIXTURE_VALIDATION_TEAM_THRESHOLD,
    FIXTURE_VALIDATION_TIMEOUT_SECONDS,
    GITHUB_AI_CONFIG,
    GOOGLE_AI_CONFIG,
    MARKET_CONFIG,
    ODDS_API_IO_KEY,
    ODDS_PROVIDER,
    REFRESH_INTERVAL_SECONDS,
    THE_ODDS_API_KEY,
)
from display import (
    console,
    print_all_markets,
    print_banner,
    print_llm_insights,
    print_match_detail,
    print_match_analysis_table,
    print_statistics,
    print_top10_upcoming,
)
from fetch_health import FetchHealthReport, get_cached_events
from github_ai import GitHubBatchAnalyzer
from google_ai import GeminiBatchAnalyzer
from validator import FixtureValidator

try:
    from understat_client import UnderstatEnricher
except ImportError:
    UnderstatEnricher = None

try:
    from fbref_client import FBrefEnricher
except ImportError:
    FBrefEnricher = None

XG_ENABLED = True
XG_AVAILABLE = UnderstatEnricher is not None or FBrefEnricher is not None

_log_dir = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(_log_dir, "bot.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logging.getLogger().handlers[-1].setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def generate_mock_completed_matches(sport_key: str) -> List[dict]:
    """Generate reproducible sample results for local form and xG analysis."""
    teams_by_sport = {
        "soccer_epl": ["Manchester City", "Arsenal", "Liverpool", "Chelsea", "Manchester United", "Tottenham", "Aston Villa", "Newcastle"],
        "soccer_spain_la_liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Real Sociedad", "Real Betis", "Villarreal", "Valencia"],
        "soccer_germany_bundesliga": ["Bayern Munich", "Borussia Dortmund", "Bayer Leverkusen", "RB Leipzig", "Eintracht Frankfurt", "VfB Stuttgart", "Freiburg", "Wolfsburg"],
        "soccer_italy_serie_a": ["Juventus", "Inter Milan", "AC Milan", "Napoli", "Roma", "Lazio", "Atalanta", "Fiorentina"],
        "soccer_france_ligue_one": ["PSG", "Marseille", "Monaco", "Lille", "Lyon", "Lens", "Rennes", "Nice"],
        "soccer_netherlands_eredivisie": ["Ajax", "PSV", "Feyenoord", "AZ Alkmaar", "Twente", "Utrecht"],
        "soccer_portugal_primeira_liga": ["Porto", "Benfica", "Sporting CP", "Braga"],
    }
    teams = teams_by_sport.get(sport_key, ["Team A", "Team B", "Team C", "Team D"])
    random.seed(42)
    matches = []
    for day in range(1, 31):
        home, away = random.sample(teams, 2)
        matches.append(
            {
                "date": f"2026-06-{day:02d}",
                "home_team": home,
                "away_team": away,
                "home_goals": random.randint(0, 4),
                "away_goals": random.randint(0, 4),
                "home_xg": round(random.uniform(0.5, 3.0), 2),
                "away_xg": round(random.uniform(0.5, 3.0), 2),
            }
        )
    return matches


async def fetch_and_analyze(
    use_mock: bool = False,
    use_ai: bool = True,
    use_xg: bool = True,
) -> tuple:
    if use_mock:
        client_class = MockAPIClient
    elif ODDS_PROVIDER in ("odds_api_io", "odds-api-io", "oddsapiio"):
        client_class = OddsAPIIOClient
    else:
        client_class = OddsAPIClient

    analyzer = BettingAnalyzer()
    api_remaining = None
    fetch_health: Optional[FetchHealthReport] = None

    with Progress(
        SpinnerColumn(style="bright_cyan"),
        TextColumn("[cyan]{task.description}"),
        BarColumn(style="cyan"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading match data...", total=None)
        async with client_class() as client:
            events = await client.fetch_all_football_markets()
            api_remaining = getattr(client, "remaining_requests", None)
            fetch_health = getattr(client, "fetch_health", None)

        if not events and not use_mock:
            cached = get_cached_events()
            if cached:
                events, fetch_health, cache_age_seconds = cached
                console.print(
                    f"\n[yellow]Using cached match data "
                    f"({cache_age_seconds / 60:.1f} minutes old).[/yellow]"
                )
            else:
                console.print("\n[red]No match data is currently available.[/red]")

        if events and not use_mock and FIXTURE_VALIDATION_ENABLED:
            if not FIXTURE_VALIDATION_ENDPOINT:
                logger.error(
                    "Fixture validation enabled but FIXTURE_VALIDATION_ENDPOINT is empty"
                )
                events = []
            else:
                progress.update(task, description="Validating fixtures...")
                validator = FixtureValidator(
                    endpoint=FIXTURE_VALIDATION_ENDPOINT,
                    api_key=FIXTURE_VALIDATION_API_KEY,
                    timeout_seconds=FIXTURE_VALIDATION_TIMEOUT_SECONDS,
                    team_threshold=FIXTURE_VALIDATION_TEAM_THRESHOLD,
                    league_threshold=FIXTURE_VALIDATION_LEAGUE_THRESHOLD,
                    kickoff_tolerance_minutes=FIXTURE_VALIDATION_KICKOFF_TOLERANCE_MINUTES,
                    cache_ttl_seconds=FIXTURE_VALIDATION_CACHE_TTL_SECONDS,
                )
                events = await asyncio.to_thread(validator.filter_fixtures, events)

        now = datetime.now()
        season_year = str(now.year if now.month >= 8 else now.year - 1)
        fbref = FBrefEnricher() if (
            use_xg and FBrefEnricher is not None and FBREF_ENABLED
        ) else None

        if use_xg and UnderstatEnricher is not None and events:
            progress.update(task, description="Adding expected-goals context...")
            events = await UnderstatEnricher().enrich_matches(
                events, season=season_year
            )
        if fbref is not None and events:
            progress.update(task, description="Adding team statistics...")
            season = f"{season_year}-{int(season_year) + 1}"
            events = fbref.enrich_matches(events, season=season)

        completed_by_sport = {}
        active_sports = {event.get("sport_key") for event in events}
        if use_mock:
            completed_by_sport = {
                sport: generate_mock_completed_matches(sport)
                for sport in active_sports
                if sport
            }
        elif fbref is not None:
            for sport in active_sports:
                if sport and fbref.is_supported(sport):
                    try:
                        season = f"{season_year}-{int(season_year) + 1}"
                        matches_data = fbref.get_league_matches(sport, season)
                        if matches_data:
                            completed_by_sport[sport] = matches_data
                    except Exception:
                        logger.exception("Could not load historical match data for %s", sport)

        progress.update(task, description=f"Analyzing {len(events)} matches...")
        results = analyzer.analyze_all(events, completed_by_sport)

        if use_ai and GOOGLE_AI_CONFIG.enabled and GOOGLE_AI_CONFIG.is_configured:
            progress.update(task, description="Adding contextual match analysis...")
            results = await GeminiBatchAnalyzer().enrich_matches(results)
        elif use_ai and GITHUB_AI_CONFIG.enabled and GITHUB_AI_CONFIG.is_configured:
            progress.update(task, description="Adding contextual match analysis...")
            results = await GitHubBatchAnalyzer().enrich_matches(results)

        if fetch_health:
            fetch_health.classify_no_results(
                n_matches=len(results),
                n_value_bets=len(results),
            )

    return results, api_remaining, fetch_health


def run_dashboard(
    matches: List[MatchAnalysis],
    api_remaining: Optional[str] = None,
    fetch_health: Optional[FetchHealthReport] = None,
):
    console.clear()
    print_banner(api_remaining, fetch_health)
    print_statistics(matches)
    print_match_analysis_table(matches)
    print_llm_insights(matches)


def show_match_detail(matches: List[MatchAnalysis], query: str):
    query_lower = query.casefold()
    found = [
        match
        for match in matches
        if query_lower in match.home_team.casefold()
        or query_lower in match.away_team.casefold()
        or query_lower in match.match_label.casefold()
    ]
    if not found:
        console.print(f"[red]No match found for '{query}'.[/red]")
        return
    for match in found:
        print_match_detail(match)


def export_csv(matches: List[MatchAnalysis]):
    import csv
    import time

    filename = f"match_analysis_{int(time.time())}.csv"
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    fields = [
        "match",
        "league",
        "kickoff_utc",
        "home_probability",
        "draw_probability",
        "away_probability",
        "data_quality_score",
        "xg_home",
        "xg_away",
        "score_estimate",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for match in matches:
            h2h = next(
                (market for market in match.markets if market.market_key == "h2h"),
                None,
            )
            llm = getattr(match, "llm_analysis", None)
            xg_market = next(
                (market for market in match.markets if market.xg_home_avg is not None),
                None,
            )
            writer.writerow(
                {
                    "match": match.match_label,
                    "league": match.sport_name,
                    "kickoff_utc": match.commence_time,
                    "home_probability": h2h.fair_probs.get(match.home_team, "") if h2h else "",
                    "draw_probability": h2h.fair_probs.get("Draw", "") if h2h else "",
                    "away_probability": h2h.fair_probs.get(match.away_team, "") if h2h else "",
                    "data_quality_score": max(
                        (market.data_quality_score for market in match.markets),
                        default="",
                    ),
                    "xg_home": xg_market.xg_home_avg if xg_market else "",
                    "xg_away": xg_market.xg_away_avg if xg_market else "",
                    "score_estimate": llm.score_prediction if llm else "",
                }
            )
    console.print(f"[green]Analysis exported to {filepath}[/green]")


def interactive_menu(matches: List[MatchAnalysis]):
    global XG_ENABLED
    while True:
        console.print(
            "\n[bold bright_cyan]━━━ ANALYSIS MENU ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold bright_cyan]"
        )
        console.print("  [white]1[/white] → Refresh analysis")
        console.print("  [white]2[/white] → View match details")
        console.print("  [white]3[/white] → Filter by league")
        console.print("  [white]4[/white] → Export match analysis to CSV")
        console.print("  [white]5[/white] → Upcoming matches overview")
        mode = "Broad" if MARKET_CONFIG.mode == "broad" else "Major leagues"
        console.print(f"  [white]6[/white] → Switch league coverage (current: {mode})")
        xg_status = "ON" if XG_ENABLED else "OFF"
        console.print(f"  [white]7[/white] → Toggle xG enrichment (currently {xg_status})")
        console.print("  [white]q[/white] → Quit")
        choice = Prompt.ask("[cyan]Select[/cyan]", default="1")

        if choice == "1":
            return "refresh"
        if choice == "2":
            show_match_detail(matches, Prompt.ask("[cyan]Team or match[/cyan]"))
        elif choice == "3":
            leagues = sorted({match.sport_name for match in matches})
            for number, league in enumerate(leagues, 1):
                console.print(f"  {number} → {league}")
            try:
                league = leagues[int(Prompt.ask("[cyan]League number[/cyan]")) - 1]
                filtered = [match for match in matches if match.sport_name == league]
                print_match_analysis_table(filtered)
                print_all_markets(filtered)
            except (IndexError, ValueError):
                console.print("[red]Invalid selection.[/red]")
        elif choice == "4":
            export_csv(matches)
        elif choice == "5":
            print_top10_upcoming(matches, hours=12)
        elif choice == "6":
            MARKET_CONFIG.mode = "major" if MARKET_CONFIG.mode == "broad" else "broad"
            return "refresh"
        elif choice == "7":
            if not XG_AVAILABLE:
                console.print("[yellow]No xG data provider is installed.[/yellow]")
            else:
                XG_ENABLED = not XG_ENABLED
                return "refresh"
        elif choice.casefold() == "q":
            return "quit"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Football match analysis dashboard",
        epilog=(
            "Examples:\n"
            "  python main.py --live\n"
            "  python main.py --detail Arsenal\n"
            "  python main.py --watch\n"
            "  python main.py --lookahead 24"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--live", action="store_true", help="Use live match data")
    parser.add_argument("--watch", action="store_true", help="Refresh automatically")
    parser.add_argument("--detail", type=str, help="Show analysis for a team or match")
    parser.add_argument("--lookahead", type=int, help="Hours to look ahead")
    parser.add_argument("--no-menu", action="store_true", help="Do not open the menu")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI context analysis")
    parser.add_argument("--model", type=str, help="Choose the AI model")
    parser.add_argument("--config", action="store_true", help="Show data configuration")
    return parser.parse_args()


async def main():
    args = parse_args()
    if args.lookahead is not None:
        if args.lookahead <= 0:
            raise SystemExit("--lookahead must be a positive number of hours")
        ANALYSIS_CONFIG.lookahead_hours = args.lookahead

    if ODDS_PROVIDER in ("odds_api_io", "odds-api-io", "oddsapiio"):
        provider, api_key = "Odds-API.io", ODDS_API_IO_KEY
    else:
        provider, api_key = "The Odds API", THE_ODDS_API_KEY
    key_invalid = not api_key or api_key == "YOUR_API_KEY_HERE"
    if provider == "Odds-API.io":
        ANALYSIS_CONFIG.min_bookmakers = min(ANALYSIS_CONFIG.min_bookmakers, 3)

    use_mock = not args.live and key_invalid
    if args.live and key_invalid:
        console.print(
            Panel(
                f"Configure the {provider} data API key to use live data. "
                "Sample data will be used for this run.",
                title="Live data unavailable",
                border_style="yellow",
            )
        )
        use_mock = True
    if args.config:
        console.print(
            Panel(
                f"Data source: {provider}\n"
                f"Live data key: {'configured' if not key_invalid else 'not configured'}\n"
                f"Mode: {'sample data' if use_mock else 'live data'}\n"
                f"AI context: {'enabled' if GOOGLE_AI_CONFIG.is_configured or GITHUB_AI_CONFIG.is_configured else 'not configured'}\n"
                f"xG enrichment: {'enabled' if XG_AVAILABLE and XG_ENABLED else 'disabled'}\n"
                f"Look-ahead: {ANALYSIS_CONFIG.lookahead_hours} hours\n"
                f"League coverage: {MARKET_CONFIG.mode}",
                title="Analysis configuration",
                border_style="cyan",
            )
        )
        return

    if args.model:
        GOOGLE_AI_CONFIG.model = args.model
        GITHUB_AI_CONFIG.model = args.model

    while True:
        matches, api_remaining, fetch_health = await fetch_and_analyze(
            use_mock=use_mock,
            use_ai=not args.no_ai,
            use_xg=XG_ENABLED,
        )
        if args.detail:
            print_banner(api_remaining, fetch_health)
            show_match_detail(matches, args.detail)
        else:
            run_dashboard(matches, api_remaining, fetch_health)

        if args.watch:
            console.print(
                f"\n[dim]Refreshing in {REFRESH_INTERVAL_SECONDS} seconds "
                "(Ctrl+C to stop)[/dim]"
            )
            await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
            continue
        if args.no_menu or args.detail:
            break
        if interactive_menu(matches) == "quit":
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[cyan]Analysis stopped.[/cyan]")
