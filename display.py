import io
import sys
from datetime import datetime, timezone
from typing import List, Optional

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from analyzer import MatchAnalysis
from fetch_health import FetchHealthReport, FetchStatus

console = Console(highlight=False, force_terminal=True, emoji=True)
UTC = timezone.utc


def format_hours(hours: float) -> str:
    if hours < -0.1:
        return f"Live ({int(abs(hours) * 60)}m)"
    if hours < 0.016:
        return "Kick-off"
    if hours < 1:
        return f"{int(hours * 60)}m"
    if hours < 24:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def format_kickoff_utc(match: MatchAnalysis) -> str:
    try:
        dt = datetime.fromisoformat(match.commence_time.replace("Z", "+00:00"))
    except ValueError:
        return "Time unavailable"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%a %d %b %H:%M UTC")


def _probability(match: MatchAnalysis, market_key: str, outcome: str) -> Optional[float]:
    market = next((m for m in match.markets if m.market_key == market_key), None)
    if market is None:
        return None
    probability = market.fair_probs.get(outcome)
    return probability if probability is not None and 0 <= probability <= 1 else None


def _format_probability(probability: Optional[float]) -> str:
    return f"{probability:.1%}" if probability is not None else "—"


def _data_quality(match: MatchAnalysis) -> Optional[float]:
    scores = [market.data_quality_score for market in match.markets]
    return max(scores) if scores else None


def print_banner(
    api_remaining: Optional[str] = None,
    fetch_health: Optional[FetchHealthReport] = None,
):
    title = (
        "[bold bright_cyan]FOOTBALL MATCH ANALYSIS[/bold bright_cyan]\n"
        "[dim]Probabilities, team context, and match data quality[/dim]"
    )
    console.print(
        Panel(Align.center(title), border_style="bright_cyan", padding=(0, 2))
    )
    meta = Text()
    meta.append(
        datetime.now(UTC).strftime("  %a %d %b %Y  %H:%M UTC  "),
        style="bold bright_yellow",
    )
    if fetch_health:
        if fetch_health.api_quota_remaining:
            api_remaining = fetch_health.api_quota_remaining
        meta.append(f"Data age: {fetch_health.data_age_label}  ", style="cyan")
        meta.append_text(Text.from_markup(fetch_health.status_label))
    if api_remaining:
        meta.append(f"  |  API requests remaining: {api_remaining}", style="dim")
    console.print(meta)

    if fetch_health and fetch_health.status in (
        FetchStatus.PARTIAL_FETCH,
        FetchStatus.API_DEGRADED,
    ):
        failed = ", ".join(
            result.sport_name
            for result in fetch_health.league_results
            if not result.success
        )
        message = (
            f"{fetch_health.ok_leagues}/{fetch_health.total_leagues} leagues "
            f"loaded successfully."
        )
        if failed:
            message += f"\nUnavailable leagues: {failed}"
        console.print(Panel(message, border_style="yellow", title="Data Warning"))
    console.print()


def print_statistics(matches: List[MatchAnalysis]):
    upcoming = sum(1 for match in matches if match.hours_until >= 0)
    leagues = len({match.sport_name for match in matches})
    with_xg = sum(
        1
        for match in matches
        if any(market.xg_home_avg is not None for market in match.markets)
    )
    summary = Table.grid(expand=True)
    summary.add_column(justify="center")
    summary.add_column(justify="center")
    summary.add_column(justify="center")
    summary.add_row(
        f"[bold cyan]{len(matches)}[/bold cyan]\n[dim]Matches analyzed[/dim]",
        f"[bold cyan]{leagues}[/bold cyan]\n[dim]Leagues[/dim]",
        f"[bold cyan]{with_xg}[/bold cyan]\n[dim]Matches with xG[/dim]",
    )
    console.print(Panel(summary, border_style="cyan"))


def print_match_analysis_table(matches: List[MatchAnalysis]):
    console.print(Rule("[bold bright_cyan]MATCH FORECASTS[/bold bright_cyan]"))
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Match", min_width=24)
    table.add_column("League", min_width=16)
    table.add_column("Kick-off", min_width=15)
    table.add_column("Home", justify="right")
    table.add_column("Draw", justify="right")
    table.add_column("Away", justify="right")
    table.add_column("Data quality", justify="right")

    for match in sorted(matches, key=lambda item: item.hours_until):
        table.add_row(
            f"[bold]{match.home_team}[/bold] vs [bold]{match.away_team}[/bold]",
            match.sport_name,
            format_hours(match.hours_until),
            _format_probability(_probability(match, "h2h", match.home_team)),
            _format_probability(_probability(match, "h2h", "Draw")),
            _format_probability(_probability(match, "h2h", match.away_team)),
            f"{quality:.0f}/100" if (quality := _data_quality(match)) is not None else "—",
        )
    if matches:
        console.print(table)
    else:
        console.print("[yellow]No match analysis is available.[/yellow]")
    console.print()


def print_all_markets(matches: List[MatchAnalysis]):
    console.print(Rule("[bold bright_cyan]PROBABILITY BREAKDOWN[/bold bright_cyan]"))
    for match in sorted(matches, key=lambda item: item.hours_until):
        table = Table(
            title=f"{match.home_team} vs {match.away_team} · {match.sport_name}",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("Analysis")
        table.add_column("Outcome")
        table.add_column("Estimated probability", justify="right")
        has_probabilities = False
        for market in match.markets:
            for outcome, probability in sorted(
                market.fair_probs.items(), key=lambda item: item[0].casefold()
            ):
                if not 0 <= probability <= 1:
                    continue
                has_probabilities = True
                table.add_row(
                    market.market_label,
                    outcome,
                    _format_probability(probability),
                )
        if has_probabilities:
            console.print(table)
    console.print()


def print_match_detail(match: MatchAnalysis):
    console.print(
        Panel(
            f"[bold]{match.home_team} vs {match.away_team}[/bold]\n"
            f"{match.sport_name} · {format_kickoff_utc(match)} · "
            f"{format_hours(match.hours_until)}",
            title="Match analysis",
            border_style="bright_cyan",
        )
    )
    print_all_markets([match])
    for market in match.markets:
        if market.xg_home_avg is not None or market.xg_away_avg is not None:
            console.print(
                f"[cyan]Expected goals context:[/cyan] "
                f"{match.home_team} {market.xg_home_avg if market.xg_home_avg is not None else '—'} · "
                f"{match.away_team} {market.xg_away_avg if market.xg_away_avg is not None else '—'}"
            )
            break
    analysis = getattr(match, "llm_analysis", None)
    if analysis and not analysis.is_fallback:
        if analysis.score_prediction:
            console.print(f"[cyan]AI score estimate:[/cyan] {analysis.score_prediction}")
        if analysis.reasoning:
            console.print(Panel(analysis.reasoning, title="Contextual analysis"))
        if analysis.key_factors:
            console.print("[bold]Factors:[/bold]")
            for factor in analysis.key_factors:
                console.print(f"  • {factor}")
        if analysis.risk_flags:
            console.print("[bold yellow]Uncertainties:[/bold yellow]")
            for flag in analysis.risk_flags:
                console.print(f"  • {flag}")
    console.print()


def print_llm_insights(matches: List[MatchAnalysis]):
    for match in matches:
        analysis = getattr(match, "llm_analysis", None)
        if not analysis or analysis.is_fallback or not analysis.reasoning:
            continue
        console.print(
            Panel(
                f"[bold]{match.home_team} vs {match.away_team}[/bold]\n"
                f"{analysis.reasoning}",
                title="AI match context",
                border_style="blue",
            )
        )


def print_top10_upcoming(matches: List[MatchAnalysis], hours: int = 12):
    upcoming = [match for match in matches if 0 <= match.hours_until <= hours]
    print_match_analysis_table(upcoming)
