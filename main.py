
import argparse
import asyncio
import logging
import os
import sys
import time
from typing import List, Optional

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.rule import Rule

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer import BettingAnalyzer, MatchAnalysis
from api_client import MockAPIClient, OddsAPIClient, OddsAPIIOClient
from config import (
    ANALYSIS_CONFIG,
    FOOTBALL_SPORTS_MAJOR,
    GITHUB_AI_CONFIG,
    GOOGLE_AI_CONFIG,
    MARKET_CONFIG,
    MAX_BETS_DISPLAYED,
    ODDS_API_IO_KEY,
    ODDS_PROVIDER,
    OLLAMA_AI_CONFIG,
    REFRESH_INTERVAL_SECONDS,
    SHOW_ALL_MARKETS,
    THE_ODDS_API_KEY,
)
from display import (
    console,
    print_all_markets,
    print_banner,
    print_bet_slip,
    print_llm_insights,
    print_match_detail,
    print_mispricing_report,
    print_statistics,
    print_top10_upcoming,
    print_value_bets_table,
    print_wr_report,
)
from fetch_health import STALE_WARNING_MINUTES, FetchHealthReport, get_cached_events
from github_ai import GitHubBatchAnalyzer
from google_ai import GeminiBatchAnalyzer
from ollama_ai import OllamaBatchAnalyzer
from portfolio import PortfolioManager
from tracker import log_picks

try:
    from understat_client import UnderstatEnricher

    XG_UNDERSTAT_AVAILABLE = True
except ImportError:
    XG_UNDERSTAT_AVAILABLE = False

try:
    from fbref_client import FBrefEnricher

    XG_FBREF_AVAILABLE = True
except ImportError:
    XG_FBREF_AVAILABLE = False

XG_AVAILABLE = XG_UNDERSTAT_AVAILABLE or XG_FBREF_AVAILABLE

XG_ENABLED = True

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

if not XG_UNDERSTAT_AVAILABLE:
    logger.warning("understat not installed — Understat xG disabled")
if not XG_FBREF_AVAILABLE:
    logger.warning("fbref_client deps missing — FBref xG disabled")


async def fetch_and_analyze(
    use_mock: bool = False,
    use_ai: bool = True,
    use_xg: bool = True,
) -> tuple:

    if use_mock:
        ClientClass = MockAPIClient
    elif ODDS_PROVIDER in ("odds_api_io", "odds-api-io", "oddsapiio"):
        ClientClass = OddsAPIIOClient
    else:
        ClientClass = OddsAPIClient
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
        task = progress.add_task("Fetching football markets...", total=None)

        async with ClientClass() as client:
            events = await client.fetch_all_football_markets()
            if hasattr(client, "remaining_requests"):
                api_remaining = client.remaining_requests

            if hasattr(client, "fetch_health") and client.fetch_health:
                fetch_health = client.fetch_health
                fetch_health.classify_no_results(
                    n_matches=len(events),
                    n_value_bets=0,
                )

        if not events and not use_mock:
            cached = get_cached_events()
            if cached:
                events, fetch_health, cache_age_s = cached
                cache_age_m = cache_age_s / 60.0
                console.print(
                    f"\n[bold yellow]⚠ API returned 0 events — using cached odds data "
                    f"({cache_age_m:.1f}m old)[/bold yellow]\n"
                )
                logger.warning(
                    "Live fetch returned 0 events. Fell back to TTL cache (age=%.1fs).",
                    cache_age_s,
                )
            else:
                console.print(
                    "\n[bold red]❌ API returned 0 events and no cache available.[/bold red]\n"
                )

        from datetime import datetime

        now = datetime.now()
        current_season_year = str(now.year if now.month >= 8 else now.year - 1)

        if use_xg and XG_UNDERSTAT_AVAILABLE and events:
            progress.update(
                task, description="Enriching matches with xG data (Understat)..."
            )
            understat_enricher = UnderstatEnricher()
            events = await understat_enricher.enrich_matches(
                events, season=current_season_year
            )

        if use_xg and XG_FBREF_AVAILABLE and events:
            unenriched = sum(1 for e in events if e.get("xg_context") is None)
            if unenriched > 0:
                progress.update(
                    task,
                    description=f"Enriching {unenriched} matches with xG data (FBref)...",
                )
                fbref_enricher = FBrefEnricher()
                fbref_season = f"{current_season_year}-{int(current_season_year) + 1}"
                events = fbref_enricher.enrich_matches(events, season=fbref_season)

        progress.update(task, description=f"Analyzing {len(events)} matches...")
        results = analyzer.analyze_all(events)

        if use_ai and GITHUB_AI_CONFIG.enabled and GITHUB_AI_CONFIG.is_configured:
            value_count = sum(1 for m in results if m.has_value)
            if value_count > 0:
                progress.update(
                    task,
                    description=f"[bright_cyan]GPT-4o (GitHub Models) analyzing {value_count} value bets...",
                )
                ai_enricher = GitHubBatchAnalyzer()
                results = await ai_enricher.enrich_matches(results)

        elif use_ai and GOOGLE_AI_CONFIG.enabled and GOOGLE_AI_CONFIG.is_configured:
            value_count = sum(1 for m in results if m.has_value)
            if value_count > 0:
                progress.update(
                    task,
                    description=f"[bright_cyan]Gemini AI analyzing {value_count} value bets...",
                )
                ai_enricher = GeminiBatchAnalyzer()
                results = await ai_enricher.enrich_matches(results)

        progress.update(task, description="Applying portfolio risk filter...")
        _portfolio = PortfolioManager(
            bankroll=ANALYSIS_CONFIG.bankroll,
            max_per_match_pct=3.0,
            session_guidance_rp=ANALYSIS_CONFIG.max_total_stake_per_session,
        )
        _portfolio.apply(results)

        if fetch_health:
            n_value = sum(1 for m in results if m.has_value)
            fetch_health.classify_no_results(
                n_matches=len(results),
                n_value_bets=n_value,
            )

    if not use_mock:
        saved = log_picks(results)
        if saved > 0:
            logger.info("tracker: %d new picks saved to picks_log.csv", saved)

    return results, api_remaining, fetch_health


def run_dashboard(
    matches: List[MatchAnalysis],
    api_remaining: Optional[str] = None,
    fetch_health: Optional["FetchHealthReport"] = None,
):
    console.clear()
    print_banner(api_remaining, fetch_health)
    print_statistics(matches)
    print_bet_slip(matches)
    print_value_bets_table(matches)
    print_llm_insights(matches)
    print_mispricing_report(matches)
    if SHOW_ALL_MARKETS:
        print_all_markets(matches)


def show_match_detail(matches: List[MatchAnalysis], query: str):
    query_lower = query.lower()
    found = [
        m
        for m in matches
        if query_lower in m.home_team.lower()
        or query_lower in m.away_team.lower()
        or query_lower in m.match_label.lower()
    ]
    if not found:
        console.print(f"[red]No match found matching: '{query}'[/red]")
        return
    for match in found:
        print_match_detail(match)


def interactive_menu(matches: List[MatchAnalysis]):
    global XG_ENABLED
    while True:
        console.print()
        mode_tag = (
            "[bold green]● ACTIVE[/bold green]" if MARKET_CONFIG.mode == "broad" else ""
        )
        major_tag = (
            "[bold green]● ACTIVE[/bold green]" if MARKET_CONFIG.mode == "major" else ""
        )
        console.print(
            "[bold bright_cyan]━━━ MENU ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold bright_cyan]"
        )
        console.print("  [white]1[/white] → Refresh dashboard")
        console.print("  [white]2[/white] → View match detail")
        console.print("  [white]3[/white] → Filter by league")
        console.print("  [white]4[/white] → Adjust min edge threshold")
        console.print("  [white]5[/white] → Export value bets to CSV")
        console.print(
            "  [bold bright_yellow]6[/bold bright_yellow] → [bold bright_yellow]Top 10 Best Bets — Next 12 Hours[/bold bright_yellow]"
        )
        console.print(
            f"  [white]7[/white] → Broad Market (all leagues)                  {mode_tag}"
        )
        console.print(
            f"  [white]8[/white] → Major Leagues Only                          {major_tag}"
        )
        xg_tag = "[bold green]● ON[/bold green]" if XG_ENABLED else "[dim]○ OFF[/dim]"
        console.print(
            f"  [white]9[/white] → Toggle xG Enrichment (Understat)            {xg_tag}"
        )
        console.print(
            "  [dim]Filter: upcoming 12h window | live matches hidden[/dim]"
        )
        console.print("  [white]w[/white] → Weekly Report (win rate & P&L)")
        console.print("  [white]q[/white] → Quit")
        console.print(
            "[bold bright_cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold bright_cyan]"
        )

        choice = Prompt.ask("[cyan]Select[/cyan]", default="1")

        if choice == "1":
            return "refresh"
        elif choice == "2":
            q = Prompt.ask("[cyan]Enter team name or match[/cyan]")
            show_match_detail(matches, q)
        elif choice == "3":
            leagues = sorted(set(m.sport_name for m in matches))
            for i, lg in enumerate(leagues, 1):
                console.print(f"  [white]{i}[/white] → {lg}")
            idx = Prompt.ask("[cyan]Select league number[/cyan]")
            try:
                league = leagues[int(idx) - 1]
                filtered = [m for m in matches if m.sport_name == league]
                console.clear()
                print_banner()
                print_value_bets_table(filtered)
                print_all_markets(filtered)
            except (IndexError, ValueError):
                console.print("[red]Invalid selection[/red]")
        elif choice == "4":
            new_edge = Prompt.ask(
                f"[cyan]Min edge % (current: {ANALYSIS_CONFIG.min_edge_pct})[/cyan]",
                default=str(ANALYSIS_CONFIG.min_edge_pct),
            )
            try:
                ANALYSIS_CONFIG.min_edge_pct = float(new_edge)
                console.print(
                    f"[green]Min edge set to {ANALYSIS_CONFIG.min_edge_pct}%[/green]"
                )
                return "refresh"
            except ValueError:
                console.print("[red]Invalid value[/red]")
        elif choice == "5":
            export_csv(matches)
        elif choice == "6":
            console.clear()
            print_banner()

            upcoming_only = [m for m in matches if m.hours_until > 0]
            print_top10_upcoming(upcoming_only, hours=12)
        elif choice == "7":
            if MARKET_CONFIG.mode == "broad":
                console.print(
                    "[yellow]Already in Broad Market mode. Reloading...[/yellow]"
                )
            else:
                MARKET_CONFIG.mode = "broad"
                console.print(
                    "[green]✅ Switched to Broad Market mode. Reloading...[/green]"
                )
            return "refresh"
        elif choice == "8":
            if MARKET_CONFIG.mode == "major":
                console.print(
                    "[yellow]Already in Major Leagues mode. Reloading...[/yellow]"
                )
            else:
                MARKET_CONFIG.mode = "major"
                console.print(
                    "[green]✅ Switched to Major Leagues mode. Reloading...[/green]"
                )
            return "refresh"
        elif choice == "9":
            if not XG_AVAILABLE:
                console.print(
                    "[red]❌ understat module not installed. Run: pip install understat[/red]"
                )
            else:
                XG_ENABLED = not XG_ENABLED
                status = "ON" if XG_ENABLED else "OFF"
                console.print(f"[green]✅ xG Enrichment: {status}[/green]")
            return "refresh"
        elif choice.lower() == "w":
            from tracker import weekly_report

            wk = Prompt.ask(
                "[cyan]Which week (1=last week, 0=all time)[/cyan]", default="1"
            )
            try:
                weekly_report(weeks_back=int(wk))
            except ValueError:
                weekly_report(weeks_back=1)
        elif choice.lower() == "q":
            return "quit"


def export_csv(matches: List[MatchAnalysis]):
    import csv

    filename = f"value_bets_{int(time.time())}.csv"
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    value_bets = [m for m in matches if m.has_value]
    if not value_bets:
        console.print("[yellow]No value bets to export.[/yellow]")
        return

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "match",
                "league",
                "hours_until",
                "market",
                "outcome",
                "best_odds",
                "bookmaker",
                "fair_prob",
                "ev_pct",
                "kelly_pct",
                "kelly_stake",
            ],
        )
        writer.writeheader()
        for m in value_bets:
            writer.writerow(
                {
                    "match": m.match_label,
                    "league": m.sport_name,
                    "hours_until": m.hours_until,
                    "market": m.top_bet_market,
                    "outcome": m.top_bet_outcome,
                    "best_odds": m.top_bet_odds,
                    "bookmaker": m.top_bet_book,
                    "fair_prob": "",
                    "ev_pct": m.top_bet_ev,
                    "kelly_pct": m.top_bet_kelly,
                    "kelly_stake": round(
                        m.top_bet_kelly / 100 * ANALYSIS_CONFIG.bankroll, 2
                    ),
                }
            )

    console.print(
        f"[green]✅ Exported {len(value_bets)} bets → [bold]{filepath}[/bold][/green]"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="⚽ Sports Betting Intelligence Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    → Dashboard with mock data (no API key needed)
  python main.py --live             → Use real API (set THE_ODDS_API_KEY env var)
  python main.py --watch            → Auto-refresh every 60s
  python main.py --detail "Arsenal"  → Detailed analysis for Arsenal matches
  python main.py --edge 3.0         → Override minimum edge threshold
  python main.py --bankroll 5000    → Set bankroll for Kelly sizing
  python main.py --lookahead 12    → Only show matches in next 12 hours
        """,
    )
    parser.add_argument(
        "--live", action="store_true", help="Use real API (default: mock data)"
    )
    parser.add_argument("--watch", action="store_true", help="Auto-refresh mode")
    parser.add_argument("--detail", type=str, help="Show detail for a specific match")
    parser.add_argument("--edge", type=float, help="Override min edge %%")
    parser.add_argument("--bankroll", type=float, help="Override bankroll amount")
    parser.add_argument(
        "--lookahead", type=int, help="Hours to look ahead (default: 48)"
    )
    parser.add_argument("--no-menu", action="store_true", help="Skip interactive menu")
    parser.add_argument(
        "--no-ai", action="store_true", help="Skip AI enrichment (faster)"
    )
    parser.add_argument("--model", type=str, help="Override AI model")
    parser.add_argument(
        "--config", action="store_true", help="Show current configuration"
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    if args.edge:
        ANALYSIS_CONFIG.min_edge_pct = args.edge
    if args.bankroll:
        ANALYSIS_CONFIG.bankroll = args.bankroll

    ANALYSIS_CONFIG.lookahead_hours = 12

    if ODDS_PROVIDER in ("odds_api_io", "odds-api-io", "oddsapiio"):
        active_provider = "Odds-API.io"
        active_key = ODDS_API_IO_KEY
        active_key_name = "ODDS_API_IO_KEY"
    else:
        active_provider = "The Odds API"
        active_key = THE_ODDS_API_KEY
        active_key_name = "THE_ODDS_API_KEY"

    _key_invalid = not active_key or active_key in ("YOUR_API_KEY_HERE", "")

    if active_provider == "Odds-API.io":
        ANALYSIS_CONFIG.min_bookmakers = min(ANALYSIS_CONFIG.min_bookmakers, 3)

    if args.live and _key_invalid:
        use_mock = True
    elif args.live:
        use_mock = False
    elif _key_invalid:
        use_mock = True
    else:
        use_mock = False

    if use_mock:
        console.print(
            "[yellow]⚠️  Running in MOCK MODE (simulated data).\n"
            f"   -> Set {active_key_name} environment variable for live data.[/yellow]"
        )
    else:
        console.print(
            f"[dim green]Using live data from {active_provider}[/dim green]"
        )

    if args.live and _key_invalid:
        console.print(
            Panel(
                "[red]No Odds API key configured![/red]\n\n"
                f"1. Get a free key for {active_provider}\n"
                f"2. Set environment variable: [bold]{active_key_name}[/bold]\n\n"
                "[yellow]Running in MOCK MODE instead...[/yellow]",
                title="API Key Required",
                border_style="red",
            )
        )
        use_mock = True

    if args.config:
        github_status = (
            f"[green]CONFIGURED ({GITHUB_AI_CONFIG.model})[/green]"
            if GITHUB_AI_CONFIG.is_configured
            else "[yellow]NOT SET — set GITHUB_TOKEN in api_keys.py[/yellow]"
        )
        gemini_status = (
            f"[dim]CONFIGURED ({GOOGLE_AI_CONFIG.model}) — secondary fallback[/dim]"
            if GOOGLE_AI_CONFIG.is_configured
            else "[dim]not set[/dim]"
        )
        console.print(
            Panel(
                f"[cyan]Odds Provider :[/cyan] {active_provider}\n"
                f"[cyan]Odds API Key  :[/cyan] {'SET' if active_key else 'NOT SET (mock mode)'}\n"
                f"[cyan]Mode          :[/cyan] {'MOCK' if use_mock else 'LIVE'}\n"
                f"[cyan]AI Provider   :[/cyan] [bold]GitHub Models (GPT-4o)[/bold]\n"
                f"[cyan]AI Status     :[/cyan] {github_status}\n"
                f"[cyan]AI Base URL   :[/cyan] {GITHUB_AI_CONFIG.base_url}\n"
                f"[cyan]Gemini (bkp)  :[/cyan] {gemini_status}\n"
                f"[cyan]xG Enrichment :[/cyan] {'ENABLED (Understat+FBref)' if XG_AVAILABLE and XG_ENABLED else 'DISABLED'}\n"
                f"[cyan]Min Edge      :[/cyan] {ANALYSIS_CONFIG.min_edge_pct}%\n"
                f"[cyan]Bankroll      :[/cyan] ${ANALYSIS_CONFIG.bankroll:,.0f}\n"
                f"[cyan]Kelly Frac    :[/cyan] {ANALYSIS_CONFIG.kelly_fraction} (half kelly)\n"
                f"[cyan]Lookahead     :[/cyan] {ANALYSIS_CONFIG.lookahead_hours}h\n"
                f"[cyan]Market Mode   :[/cyan] {MARKET_CONFIG.mode.upper()}\n"
                f"[cyan]Min Books     :[/cyan] {ANALYSIS_CONFIG.min_bookmakers}\n"
                f"[cyan]Refresh       :[/cyan] {REFRESH_INTERVAL_SECONDS}s",
                title="[bold]Configuration[/bold]",
                border_style="cyan",
            )
        )
        return

    if hasattr(args, "model") and args.model:
        GITHUB_AI_CONFIG.model = args.model

    use_ai = not getattr(args, "no_ai", False)

    while True:
        matches, api_remaining, fetch_health = await fetch_and_analyze(
            use_mock=use_mock, use_ai=use_ai, use_xg=XG_ENABLED
        )

        if args.detail:
            print_banner(api_remaining, fetch_health)
            show_match_detail(matches, args.detail)
        else:
            run_dashboard(matches, api_remaining, fetch_health)

        if args.watch:
            console.print(
                f"\n[dim]⏱ Auto-refreshing in {REFRESH_INTERVAL_SECONDS}s... (Ctrl+C to stop)[/dim]"
            )
            try:
                await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
                continue
            except asyncio.CancelledError:
                break

        if args.no_menu:
            break

        action = interactive_menu(matches)
        if action == "quit":
            try:
                from tracker import weekly_report

                report = weekly_report(weeks_back=0, print_output=False)
                if report and report.get("settled", 0) > 0:
                    console.print()
                    console.print(
                        Rule(
                            "[bold bright_cyan]SESSION END — PERFORMANCE SUMMARY[/bold bright_cyan]",
                            style="bright_cyan",
                        )
                    )
                    print_wr_report(report, compact=True)
            except Exception:
                pass

            console.print(
                "\n[bold cyan]👋 Goodbye! Good luck betting responsibly.[/bold cyan]\n"
            )
            break
        elif action == "refresh":
            continue


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold cyan]👋 Interrupted. Goodbye![/bold cyan]\n")
