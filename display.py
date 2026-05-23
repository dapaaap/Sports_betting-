
import io
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from analyzer import MarketAnalysis, MatchAnalysis
from config import (
    ANALYSIS_CONFIG,
    COLOR_DANGER,
    COLOR_GOOD_VALUE,
    COLOR_HEADER,
    COLOR_INFO,
    COLOR_NO_VALUE,
    COLOR_STRONG_VALUE,
    COLOR_WEAK_VALUE,
    MAX_BETS_DISPLAYED,
)
from fetch_health import FetchHealthReport, FetchStatus, NoResultReason

console = Console(highlight=False, force_terminal=True, emoji=True)


WIB = timezone(timedelta(hours=7))


def ev_color(ev: float) -> str:
    if ev >= 8.0:
        return COLOR_STRONG_VALUE
    elif ev >= 4.0:
        return COLOR_GOOD_VALUE
    elif ev >= ANALYSIS_CONFIG.min_edge_pct:
        return COLOR_WEAK_VALUE
    elif ev < -3.0:
        return COLOR_DANGER
    return COLOR_NO_VALUE


def prob_bar(prob: float, width: int = 12) -> str:
    filled = round(prob * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {prob * 100:.1f}%"


def format_odds(odds: float) -> str:
    if odds <= 0:
        return "—"
    return f"{odds:.2f}"


def format_hours(hours: float) -> str:
    if hours < -0.1:
        mins_ago = int(abs(hours) * 60)
        return f"[bold bright_red blink]\u25cf LIVE[/bold bright_red blink] [dim]{mins_ago}m lalu[/dim]"
    elif hours < 0.016:
        return f"[bold bright_red blink]\u25cf KICK-OFF![/bold bright_red blink]"
    elif hours < 1:
        mins = int(hours * 60)
        return f"[bold bright_red]{mins}m[/bold bright_red]"
    elif hours < 24:
        return f"[bright_yellow]{hours:.1f}j[/bright_yellow]"
    else:
        days = hours / 24
        return f"[white]{days:.1f} hari[/white]"


def format_kickoff_wib(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    wib_dt = dt.astimezone(WIB)
    return wib_dt.strftime("%a %d %b %H:%M WIB")


def signal_style(signal: str) -> str:
    if "STRONG" in signal:
        return f"[bold bright_green]{signal}[/bold bright_green]"
    elif "BUY" in signal:
        return f"[green]{signal}[/green]"
    elif "WATCH" in signal:
        return f"[yellow]{signal}[/yellow]"
    elif "AVOID" in signal:
        return f"[red]{signal}[/red]"
    return f"[white]{signal}[/white]"


def sparkline(values: List[float]) -> str:
    if not values:
        return ""
    bars = "▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn
    if rng == 0:
        return bars[3] * len(values)
    return "".join(bars[min(7, int((v - mn) / rng * 7))] for v in values)


MARKET_EXPLAIN = {
    "Match Result (1X2)": "HASIL PERTANDINGAN",
    "Over/Under Goals": "TOTAL GOL",
    "Both Teams to Score": "KEDUA TIM CETAK GOL",
    "Asian Handicap": "HANDICAP ASIA",
    "H2H_LAY": "PASARAN EXCHANGE (LAY)",
}


def explain_selection(market_label: str, outcome: str, home: str, away: str) -> str:
    mkt = market_label.upper()

    if "1X2" in mkt or "MATCH RESULT" in mkt or "H2H" in mkt:
        if outcome == home:
            return f"Menang {home} (HOME WIN)"
        elif outcome == away:
            return f"Menang {away} (AWAY WIN)"
        elif outcome.lower() in ("draw", "seri"):
            return "Seri / Draw"
        else:
            return f"Menang {outcome}"

    if "OVER" in mkt or "UNDER" in mkt or "TOTAL" in mkt or "GOALS" in mkt:
        line_val = outcome.split()[-1] if len(outcome.split()) > 1 else "garis"
        if "over" in outcome.lower():
            return f"OVER — Total gol pertandingan LEBIH DARI {line_val}"
        elif "under" in outcome.lower():
            return f"UNDER — Total gol pertandingan KURANG DARI {line_val}"
        return outcome

    if "BOTH" in mkt or "BTTS" in mkt or "SCORE" in mkt:
        out_clean = outcome.lower().strip()
        if out_clean in ("yes", "ya"):
            return "YA — Kedua tim sama-sama cetak minimal 1 gol"
        elif out_clean in ("no", "tidak"):
            return "TIDAK — Salah satu tim tidak cetak gol (atau 0-0)"
        return f"{outcome} (Both Teams to Score)"

    if "HANDICAP" in mkt or "SPREAD" in mkt:
        return f"{outcome} (Handicap)"

    if "LAY" in mkt:
        return f"LAY {outcome} (menang jika {outcome} TIDAK menang)"

    return outcome


def explain_market(market_label: str) -> str:
    return MARKET_EXPLAIN.get(market_label, market_label)


def calc_winnings(odds: float, stake: float) -> float:
    return round((odds - 1) * stake, 2)


def print_bet_slip(matches: List[MatchAnalysis]):
    value_bets = [m for m in matches if m.has_value]

    _rpt = next((getattr(m, "_portfolio_report", None) for m in value_bets), None)
    if _rpt:
        lines = []

        for adv in _rpt.correlation_advisories[:6]:
            lines.append(
                f"  [yellow]⚠[/yellow] [bold]{adv.match_label}[/bold] — "
                f"[cyan]{adv.market_label_a}[/cyan] berkorelasi dengan "
                f"[cyan]{adv.market_label_b}[/cyan] "
                f"[dim](corr: {adv.correlation:.0%}) | stake gabungan Rp {adv.combined_stake_rp:,.0f} "
                f"| panduan maks Rp {adv.recommended_max_rp:,.0f}[/dim]"
            )

        for s in _rpt.suppressions[:4]:
            lines.append(
                f"  [red]⛔[/red] [bold]{s.match_label}[/bold] — "
                f"[red]{s.market_label}[/red] dihapus "
                f"[dim](korelasi ekstrem {s.correlation:.0%} dengan {s.correlated_with})[/dim]"
            )
        if lines:
            console.print(
                Panel(
                    "\n".join(lines),
                    title="[bold yellow]Risk Transparency — Correlated Markets[/bold yellow]",
                    border_style="yellow",
                    padding=(0, 2),
                )
            )
            console.print()

    console.print(
        Rule(
            "[bold bright_yellow]=== BET SLIP — REKOMENDASI TARUHAN ===[/bold bright_yellow]",
            style="bright_yellow",
        )
    )
    console.print()

    if not value_bets:
        console.print(
            Panel(
                "[yellow]Tidak ada rekomendasi taruhan saat ini.\n"
                "Coba turunkan min_edge_pct di config.py[/yellow]",
                border_style="yellow",
            )
        )
        console.print()
        return

    for i, match in enumerate(value_bets, 1):
        ev = match.top_bet_ev
        color = ev_color(ev)
        stake = round(match.top_bet_kelly / 100 * ANALYSIS_CONFIG.bankroll, 2)
        win = calc_winnings(match.top_bet_odds, stake)

        mkt_type = explain_market(match.top_bet_market or "")

        selection_explain = explain_selection(
            match.top_bet_market or "",
            match.top_bet_outcome or "",
            match.home_team,
            match.away_team,
        )

        fair_prob = 0.0
        for mkt in match.markets:
            if mkt.market_label == match.top_bet_market:
                fair_prob = mkt.fair_probs.get(
                    match.top_bet_outcome or "", mkt.fair_prob
                )
                break

        cqs_val = getattr(match, "cqs", 0)
        cqs_grd = getattr(match, "cqs_grade", "")
        if "PREMIUM" in cqs_grd:
            tier = "[bold bright_green]PREMIUM ★★★[/bold bright_green]"
            cqs_badge = f"[bold bright_green]CQS {cqs_val:.0f}[/bold bright_green]"
        elif "STANDARD" in cqs_grd:
            tier = "[bold green]STANDARD ★★[/bold green]"
            cqs_badge = f"[bold green]CQS {cqs_val:.0f}[/bold green]"
        elif cqs_grd:
            tier = "[yellow]MARGINAL ★[/yellow]"
            cqs_badge = f"[yellow]CQS {cqs_val:.0f}[/yellow]"
        elif ev >= 8.0:
            tier = "[bold bright_green]SANGAT KUAT[/bold bright_green]"
            cqs_badge = "[dim]-[/dim]"
        elif ev >= 4.0:
            tier = "[bold green]KUAT[/bold green]"
            cqs_badge = "[dim]-[/dim]"
        else:
            tier = "[yellow]MODERAT[/yellow]"
            cqs_badge = "[dim]-[/dim]"

        dt_str = format_kickoff_wib(match.commence_dt)

        xg_note = ""
        best_mkt = None
        for mkt in match.markets:
            if mkt.market_label == match.top_bet_market:
                best_mkt = mkt
                break

        if best_mkt:
            xg_signal = getattr(best_mkt, "xg_signal", "XG_UNAVAILABLE")
            xg_h_avg = getattr(best_mkt, "xg_home_avg", None)
            xg_a_avg = getattr(best_mkt, "xg_away_avg", None)

            if xg_signal == "XG_CONFIRM":
                xg_fmt = "[green]XG_CONFIRM ✓[/green]"
            elif xg_signal == "XG_CONFLICT":
                xg_fmt = "[bold yellow]XG_CONFLICT ⚠[/bold yellow]"
            elif xg_signal == "XG_NEUTRAL":
                xg_fmt = "[dim white]XG_NEUTRAL ~[/dim white]"
            else:
                xg_fmt = "[dim]XG_UNAVAILABLE -[/dim]"

            if xg_h_avg is not None and xg_a_avg is not None:
                xg_note = f"\n  xG Signal  : {xg_fmt}\n  xG Avg     : Home {xg_h_avg:.2f} | Away {xg_a_avg:.2f}"
            elif xg_signal != "XG_UNAVAILABLE":
                xg_note = f"\n  xG Signal  : {xg_fmt}"

        ai_note = ""
        if (
            hasattr(match, "llm_analysis")
            and match.llm_analysis
            and not match.llm_analysis.is_fallback
        ):
            llm = match.llm_analysis
            rs = getattr(llm, "risk_score", None)
            if rs is not None:
                risk_label = f"Risk Score {rs:.0f}/10"
            else:
                risk_label = "AI Analysis"
            reasoning_snippet = (
                (llm.reasoning[:120] + "...")
                if len(llm.reasoning) > 120
                else llm.reasoning
            )
            model_short = llm.model_used.split("/")[-1] if llm.model_used else "AI"
            ai_note = f"\n  AI ({model_short}) : [italic]{risk_label}[/italic] — {reasoning_snippet}"

        advisory_note = ""

        _rpt_m = getattr(match, "_portfolio_report", None)
        if _rpt_m:
            advisory_note += (
                f"\n  [dim]▸ Panduan maks per-pertandingan: "
                f"Rp {_rpt_m.recommended_max_per_match_rp:,.0f} (3% bankroll)[/dim]"
            )

        for cadv in getattr(match, "_portfolio_corr_advisories", []):
            advisory_note += (
                f"\n  [yellow]⚠ Korelasi {cadv.correlation:.0%}:[/yellow] "
                f"[cyan]{cadv.market_label_a}[/cyan] \u2194 "
                f"[cyan]{cadv.market_label_b}[/cyan] "
                f"[dim]| Stake gabungan Rp {cadv.combined_stake_rp:,.0f}[/dim]"
            )

        for adv in getattr(match, "_portfolio_advisories", []):
            if adv.advisory_type == "MATCH_OVEREXPOSURE":
                sev_color = "red" if adv.severity == "HIGH" else "yellow"
                advisory_note += (
                    f"\n  [bold {sev_color}]⚠ RISK:[/bold {sev_color}] "
                    f"[italic]{adv.message}[/italic]"
                )

        for adv in getattr(match, "_portfolio_advisories", []):
            if adv.advisory_type == "SESSION_GUIDANCE":
                advisory_note += (
                    f"\n  [bold red]⚠ PANDUAN SESI:[/bold red] "
                    f"[italic]{adv.message}[/italic]"
                )

        content = (
            f"[bold white]#{i}  {match.home_team}  vs  {match.away_team}[/bold white]\n"
            f"  Liga       : [cyan]{match.sport_name}[/cyan]\n"
            f"  Kick-off   : [white]{dt_str}[/white]  ([yellow]{match.hours_until:.1f}j lagi[/yellow])\n"
            f"\n"
            f"  PASARAN    : [bold bright_blue]{match.top_bet_market}[/bold bright_blue]"
            f"   [dim]({mkt_type})[/dim]\n"
            f"  BET        : [bold white]{match.top_bet_outcome}[/bold white]\n"
            f"  ARTINYA    : [italic]{selection_explain}[/italic]\n"
            f"\n"
            f"  Odds Terbaik : [bold {color}]{format_odds(match.top_bet_odds)}[/bold {color}]"
            f"  @ [dim]{match.top_bet_book}[/dim]\n"
            f"  Prob Fair  : [cyan]{fair_prob * 100:.1f}%[/cyan]"
            f"  |  Edge (EV) : [bold {color}]+{ev:.1f}%[/bold {color}]"
            f"  |  Grade: {tier}  |  {cqs_badge}\n"
            f"\n"
            f"  STAKE      : [bold green]Rp {stake:,.0f}[/bold green]"
            f"  [dim](dari bankroll Rp {ANALYSIS_CONFIG.bankroll:,.0f}  |  Kelly {match.top_bet_kelly:.2f}%)[/dim]\n"
            f"  Potensi Untung: [bold bright_green]+Rp {win:,.0f}[/bold bright_green]"
            f"  [dim](jika menang)[/dim]"
            f"{xg_note}"
            f"{ai_note}"
            f"{advisory_note}"
        )

        console.print(
            Panel(
                content,
                border_style=color,
                padding=(0, 2),
                title=f"[bold]BET #{i}[/bold]  [dim]{match.sport_name}[/dim]",
            )
        )
        console.print()

    console.print(
        Panel(
            "[bold white]PANDUAN PASARAN:[/bold white]\n\n"
            "  [cyan]Match Result (1X2)[/cyan]   : Pilih HOME WIN, DRAW, atau AWAY WIN\n"
            "  [cyan]Over/Under Goals[/cyan]     : Pilih OVER atau UNDER dari garis gol yang ditawarkan\n"
            "                           Contoh: Over 2.5 = pertandingan berakhir 3 gol atau lebih\n"
            "  [cyan]Both Teams to Score[/cyan]  : YES = kedua tim cetak gol, NO = salah satu tidak cetak\n"
            "  [cyan]Asian Handicap[/cyan]       : Tim favorit diberi handicap gol untuk penyeimbang\n\n"
            "[bold white]CARA BACA ODDS DESIMAL:[/bold white]\n\n"
            "  Odds 2.10 x Stake 100 = Return 210  (Untung bersih 110)\n"
            "  Odds 1.75 x Stake 100 = Return 175  (Untung bersih 75)\n\n"
            "[bold white]KELLY CRITERION (Sizing):[/bold white]\n\n"
            "  Stake = (Edge / Odds) x Bankroll x Kelly Fraction\n"
            "  Bot menggunakan Half-Kelly (50%) untuk keseimbangan risiko",
            title="[dim]Panduan[/dim]",
            border_style="dim",
            padding=(0, 2),
        )
    )
    console.print()


def print_top10_upcoming(matches: List[MatchAnalysis], hours: int = 12):

    LIVE_WINDOW = 2.0
    upcoming = [m for m in matches if -LIVE_WINDOW <= m.hours_until <= hours]

    live_count = sum(1 for m in upcoming if m.hours_until < 0)

    title_live = f" [bright_red]({live_count} LIVE)[/bright_red]" if live_count else ""
    console.print(
        Rule(
            f"[bold bright_yellow]TOP 10 BET TERBAIK \u2014 {hours} JAM MENDATANG{title_live}[/bold bright_yellow]",
            style="bright_yellow",
        )
    )
    console.print()

    if not upcoming:
        console.print(
            Panel(
                f"[yellow]Tidak ada pertandingan LIVE atau yang kick-off dalam {hours} jam ke depan.\n"
                "Coba refresh atau tunggu pertandingan berikutnya.[/yellow]",
                border_style="yellow",
            )
        )
        console.print()
        return

    candidates = []
    _global_rpt = next((getattr(m, "_portfolio_report", None) for m in upcoming), None)
    for match in upcoming:
        _suppressed_keys = {
            m.market_key for m in getattr(match, "_portfolio_suppressed", [])
        }
        for mkt in match.markets:
            if mkt.market_key in _suppressed_keys:
                continue
            for outcome in mkt.outcomes:
                fp = mkt.fair_probs.get(outcome.name, 0)
                if fp <= 0:
                    continue
                ev = (fp * outcome.best_odds - 1) * 100
                if ev < ANALYSIS_CONFIG.min_edge_pct:
                    continue
                kelly_raw = max(0.0, (fp - (1 - fp) / (outcome.best_odds - 1)))
                kelly = kelly_raw * ANALYSIS_CONFIG.kelly_fraction
                candidates.append(
                    {
                        "match": match,
                        "mkt": mkt,
                        "outcome": outcome.name,
                        "odds": outcome.best_odds,
                        "book": outcome.bookmaker,
                        "fair_prob": fp,
                        "ev": ev,
                        "kelly_pct": round(kelly * 100, 2),
                        "stake": round(kelly * ANALYSIS_CONFIG.bankroll, 2),
                        "win": round(
                            (outcome.best_odds - 1) * kelly * ANALYSIS_CONFIG.bankroll,
                            2,
                        ),
                        "market_label": mkt.market_label,
                        "hours": match.hours_until,
                    }
                )

    candidates.sort(key=lambda x: -x["ev"])
    seen_matches: set = set()
    top10 = []
    for c in candidates:
        mid = c["match"].match_label
        if mid not in seen_matches:
            seen_matches.add(mid)
            top10.append(c)
        if len(top10) >= 10:
            break

    if not top10:
        console.print(
            Panel(
                f"[yellow]Tidak ada value bet dalam {hours} jam ke depan "
                f"(threshold EV +{ANALYSIS_CONFIG.min_edge_pct}%).\n"
                "Coba turunkan min_edge_pct di config.py[/yellow]",
                border_style="yellow",
            )
        )
        console.print()
        return

    _rpt = next((getattr(m, "_portfolio_report", None) for m in upcoming), None)
    if _rpt:
        for adv in _rpt.session_advisories():
            console.print(
                Panel(
                    f"[bold red]⚠ PANDUAN SESI:[/bold red] {adv.message}",
                    border_style="red",
                    padding=(0, 2),
                )
            )
            console.print()

    total_stake = sum(c["stake"] for c in top10)
    total_win_max = sum(c["win"] for c in top10)
    avg_ev = sum(c["ev"] for c in top10) / len(top10)

    info_cols = [
        Panel(
            f"[bold bright_yellow]{len(upcoming)}[/bold bright_yellow]\n[dim]Pertandingan < {hours}j[/dim]",
            border_style="yellow",
            padding=(0, 2),
        ),
        Panel(
            f"[bold bright_green]{len(top10)}[/bold bright_green]\n[dim]Bet Terpilih[/dim]",
            border_style="green",
            padding=(0, 2),
        ),
        Panel(
            f"[bold cyan]{avg_ev:.1f}%[/bold cyan]\n[dim]Avg EV[/dim]",
            border_style="cyan",
            padding=(0, 2),
        ),
        Panel(
            f"[bold green]Rp {total_stake:,.0f}[/bold green]\n[dim]Total Stake[/dim]",
            border_style="green",
            padding=(0, 2),
        ),
        Panel(
            f"[bold bright_green]+Rp {total_win_max:,.0f}[/bold bright_green]\n[dim]Pot. Untung Maks[/dim]",
            border_style="bright_green",
            padding=(0, 2),
        ),
    ]
    console.print(Columns(info_cols, equal=True))
    console.print()

    table = Table(
        box=box.MINIMAL_HEAVY_HEAD,
        show_header=True,
        header_style="bold bright_yellow",
        border_style="yellow",
        expand=True,
    )
    table.add_column("#", width=3, justify="center")
    table.add_column("Pertandingan", min_width=22)
    table.add_column("Liga", min_width=14)
    table.add_column("Kick-off", min_width=6, justify="center")
    table.add_column("Pasaran", min_width=18)
    table.add_column("Bet", min_width=12)
    table.add_column("Artinya", min_width=22)
    table.add_column("Odds", min_width=6, justify="center")
    table.add_column("EV%", min_width=8, justify="right")
    table.add_column("Stake", min_width=10, justify="right")
    table.add_column("Pot. Untung", min_width=12, justify="right")

    for rank, c in enumerate(top10, 1):
        match = c["match"]
        ec = ev_color(c["ev"])
        expl = explain_selection(
            c["market_label"], c["outcome"], match.home_team, match.away_team
        )
        expl_s = (expl[:26] + "\u2026") if len(expl) > 28 else expl
        table.add_row(
            f"[bold {ec}]{rank}[/bold {ec}]",
            f"[bold]{match.home_team}[/bold]\n[dim]vs {match.away_team}[/dim]",
            f"[dim]{match.sport_name[:14]}[/dim]",
            format_hours(c["hours"]),
            f"[bright_blue]{c['market_label']}[/bright_blue]",
            f"[bold white]{c['outcome']}[/bold white]",
            f"[italic]{expl_s}[/italic]",
            f"[bold {ec}]{format_odds(c['odds'])}[/bold {ec}]",
            f"[bold {ec}]+{c['ev']:.1f}%[/bold {ec}]",
            f"[green]Rp {c['stake']:,.0f}[/green]",
            f"[bright_green]+Rp {c['win']:,.0f}[/bright_green]",
        )
    console.print(table)
    console.print()

    console.print(
        Rule(
            "[bold bright_yellow]DETAIL SETIAP BET[/bold bright_yellow]", style="yellow"
        )
    )
    console.print()

    for rank, c in enumerate(top10, 1):
        match = c["match"]
        ec = ev_color(c["ev"])
        expl = explain_selection(
            c["market_label"], c["outcome"], match.home_team, match.away_team
        )
        if c["ev"] >= 8.0:
            tier = "[bold bright_green]SANGAT KUAT[/bold bright_green]"
        elif c["ev"] >= 4.0:
            tier = "[bold green]KUAT[/bold green]"
        else:
            tier = "[yellow]MODERAT[/yellow]"

        dt_str = format_kickoff_wib(match.commence_dt)
        ai_note = ""
        if (
            hasattr(match, "llm_analysis")
            and match.llm_analysis
            and not match.llm_analysis.is_fallback
        ):
            llm = match.llm_analysis
            rs = getattr(llm, "risk_score", None)
            if rs is not None:
                risk_label = f"Risk Score {rs:.0f}/10"
            else:
                risk_label = "AI Analysis"
            model_short = llm.model_used.split("/")[-1] if llm.model_used else "AI"
            ai_note = (
                f"\n\n  [bold]AI ({model_short})[/bold] : "
                f"[italic]{risk_label}[/italic] \u2014 {llm.reasoning[:120]}"
            )

        content = (
            f"[bold white]\u25b6  #{rank}  {match.home_team}  vs  {match.away_team}[/bold white]\n"
            f"  Liga    : [cyan]{match.sport_name}[/cyan]  |  "
            f"Kick-off : [white]{dt_str}[/white]  ([bright_red]{c['hours']:.1f}j lagi[/bright_red])\n"
            f"\n"
            f"  PASARAN : [bold bright_blue]{c['market_label']}[/bold bright_blue]"
            f"  [dim]({explain_market(c['market_label'])})[/dim]\n"
            f"  BET     : [bold white]{c['outcome']}[/bold white]\n"
            f"  ARTINYA : [italic]{expl}[/italic]\n"
            f"\n"
            f"  Odds    : [bold {ec}]{format_odds(c['odds'])}[/bold {ec}]"
            f"  @ [dim]{c['book']}[/dim]   "
            f"Prob Fair : [cyan]{c['fair_prob'] * 100:.1f}%[/cyan]\n"
            f"  Edge EV : [bold {ec}]+{c['ev']:.1f}%[/bold {ec}]   "
            f"Keyakinan : {tier}\n"
            f"\n"
            f"  STAKE   : [bold green]Rp {c['stake']:,.0f}[/bold green]"
            f"  [dim](Kelly {c['kelly_pct']:.2f}% dari bankroll Rp {ANALYSIS_CONFIG.bankroll:,.0f})[/dim]\n"
            f"  Potensi Untung : [bold bright_green]+Rp {c['win']:,.0f}[/bold bright_green]"
            f"  [dim](jika menang)[/dim]"
            f"{ai_note}"
        )
        console.print(
            Panel(
                content,
                title=f"[bold]RANK #{rank}[/bold]",
                border_style=ec,
                padding=(0, 2),
            )
        )
        console.print()


BANNER = r"""
 [SPORTS BETTING INTELLIGENCE BOT]
 =========================================================
  AI-Powered | Mispricing Detection | Kelly Sizing | Multi-League
"""


def print_banner(
    api_remaining: Optional[str] = None,
    fetch_health: Optional[FetchHealthReport] = None,
):
    console.print()
    panel = Panel(
        Align.center(BANNER.strip()),
        border_style="bright_cyan",
        padding=(0, 2),
    )
    console.print(panel)

    now_wib = datetime.now(WIB)
    now_utc = datetime.now(timezone.utc)
    wib_str = now_wib.strftime("%a %d %b %Y  %H:%M:%S WIB")
    utc_str = now_utc.strftime("%H:%M UTC")

    meta = Text()
    meta.append(f"  \u23f0 {wib_str}  ({utc_str})  ", style="bold bright_yellow")
    meta.append(f"  Bankroll: Rp {ANALYSIS_CONFIG.bankroll:,.0f}  ", style="cyan")
    meta.append(f"  Lookahead: {ANALYSIS_CONFIG.lookahead_hours}j  ", style="cyan")
    meta.append(f"  Min Edge: {ANALYSIS_CONFIG.min_edge_pct}%  ", style="cyan")

    if fetch_health:
        if fetch_health.api_quota_remaining:
            api_remaining = fetch_health.api_quota_remaining

        meta.append("  |  ", style="dim")
        meta.append("Freshness: ")
        meta.append(f"{fetch_health.data_age_label}", style="white")

        meta.append("  Status: ")
        meta.append_text(Text.from_markup(fetch_health.status_label))

    if api_remaining:
        meta.append(f"  |  API Quota: {api_remaining} sisa", style="green")

    console.print(meta)

    if fetch_health and fetch_health.status in (
        FetchStatus.PARTIAL_FETCH,
        FetchStatus.API_DEGRADED,
    ):
        failed_count = fetch_health.failed_leagues
        total_count = fetch_health.total_leagues
        failed_names = ", ".join(
            [r.sport_name for r in fetch_health.league_results if not r.success]
        )

        warning_msg = f"[yellow]Hanya {fetch_health.ok_leagues}/{total_count} liga berhasil di-fetch.[/yellow]\n"
        warning_msg += f"Gagal: [dim]{failed_names}[/dim]"
        console.print(
            Panel(warning_msg, border_style="yellow", title="⚠ Partial Fetch Warning")
        )

    console.print()


def print_value_bets_table(matches: List[MatchAnalysis]):
    value_bets = [m for m in matches if m.has_value][:MAX_BETS_DISPLAYED]

    console.print(
        Rule(
            f"[bold bright_green]>> TOP VALUE BETS  ({len(value_bets)} found)[/bold bright_green]",
            style="bright_green",
        )
    )
    console.print()

    if not value_bets:
        console.print(
            Panel(
                "[yellow]No value bets detected at current threshold "
                f"({ANALYSIS_CONFIG.min_edge_pct}% min edge).\n"
                "Try lowering [bold]min_edge_pct[/bold] in config.py",
                title="[yellow]! No Value Found[/yellow]",
                border_style="yellow",
            )
        )
        console.print()
        return

    table = Table(
        box=box.MINIMAL_HEAVY_HEAD,
        show_header=True,
        header_style=f"bold {COLOR_HEADER}",
        border_style="bright_blue",
        pad_edge=True,
        expand=True,
    )

    table.add_column("Match", min_width=22, no_wrap=True)
    table.add_column("League", min_width=14, no_wrap=True)
    table.add_column("Kick-off", min_width=6, no_wrap=True, justify="center")
    table.add_column("Market", min_width=16, no_wrap=True)
    table.add_column("Bet Selection", min_width=14, no_wrap=True)
    table.add_column("Best Odds", min_width=8, justify="center")
    table.add_column("Book", min_width=10, no_wrap=True)
    table.add_column("Fair Prob", min_width=14, no_wrap=True)
    table.add_column("Edge (EV%)", min_width=10, justify="right")
    table.add_column("CQS", min_width=6, justify="center")
    table.add_column("xG", min_width=4, justify="center")
    table.add_column("AI Signal", min_width=16, no_wrap=True)
    table.add_column("Kelly Stake", min_width=10, justify="right")

    for match in value_bets:
        ev = match.top_bet_ev
        color = ev_color(ev)

        best_mkt = None
        for mkt in match.markets:
            if mkt.market_label == match.top_bet_market:
                best_mkt = mkt
                break

        fair_bar = prob_bar(best_mkt.fair_prob if best_mkt else 0)
        ai_signal = best_mkt.ai_signal if best_mkt else "—"

        xg_icon = "[dim]-[/dim]"
        if best_mkt:
            xg_signal = getattr(best_mkt, "xg_signal", "XG_UNAVAILABLE")
            if xg_signal == "XG_CONFIRM":
                xg_icon = "[green]✓[/green]"
            elif xg_signal == "XG_CONFLICT":
                xg_icon = "[bold yellow]⚠[/bold yellow]"
            elif xg_signal == "XG_NEUTRAL":
                xg_icon = "[dim white]~[/dim white]"

        cqs_val = getattr(match, "cqs", 0)
        cqs_grd = getattr(match, "cqs_grade", "")
        if "PREMIUM" in cqs_grd:
            cqs_display = f"[bold bright_green]{cqs_val:.0f}[/bold bright_green]"
        elif "STANDARD" in cqs_grd:
            cqs_display = f"[green]{cqs_val:.0f}[/green]"
        elif cqs_grd:
            cqs_display = f"[yellow]{cqs_val:.0f}[/yellow]"
        else:
            cqs_display = "[dim]-[/dim]"

        table.add_row(
            f"[bold]{match.home_team}\n[dim]vs[/dim] {match.away_team}[/bold]",
            f"[dim]{match.sport_name}[/dim]",
            format_hours(match.hours_until),
            f"[bright_blue]{match.top_bet_market}[/bright_blue]",
            f"[bold white]{match.top_bet_outcome}[/bold white]",
            f"[bold {color}]{format_odds(match.top_bet_odds)}[/bold {color}]",
            f"[dim]{match.top_bet_book}[/dim]",
            f"[cyan]{fair_bar}[/cyan]",
            f"[bold {color}]+{ev:.1f}%[/bold {color}]",
            cqs_display,
            xg_icon,
            signal_style(ai_signal),
            f"[bold green]${match.top_bet_kelly / 100 * ANALYSIS_CONFIG.bankroll:.2f}[/bold green]"
            if match.top_bet_kelly > 0
            else "[dim]—[/dim]",
        )

    console.print(table)
    console.print()


def print_all_markets(matches: List[MatchAnalysis]):
    console.print(
        Rule(
            f"[bold bright_cyan]ALL UPCOMING MARKETS  ({len(matches)} matches)[/bold bright_cyan]",
            style="bright_cyan",
        )
    )
    console.print()

    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold {COLOR_HEADER}",
        border_style="blue",
        pad_edge=True,
        expand=True,
    )

    table.add_column("Match", min_width=24, no_wrap=False)
    table.add_column("League", min_width=14, no_wrap=True)
    table.add_column("Kick-off", min_width=6, justify="center")
    table.add_column("Books", min_width=4, justify="center")
    table.add_column("1X2 Odds (H/D/A)", min_width=18, justify="left")
    table.add_column("O/U", min_width=12, justify="left")
    table.add_column("BTTS", min_width=10, justify="left")
    table.add_column("Best EV%", min_width=9, justify="right")
    table.add_column("Rating", min_width=18, no_wrap=True)

    for match in matches:
        h2h = next((m for m in match.markets if m.market_key == "h2h"), None)
        tot = next((m for m in match.markets if m.market_key == "totals"), None)
        btts = next((m for m in match.markets if m.market_key == "btts"), None)

        def get_outcome_odds(mkt: Optional[MarketAnalysis], name_filter) -> str:
            if not mkt:
                return "[dim]—[/dim]"
            outcomes = sorted(mkt.outcomes, key=lambda o: o.best_odds)
            filtered = [o for o in outcomes if name_filter(o.name)]
            if not filtered:
                return "[dim]—[/dim]"
            return " / ".join(f"{format_odds(o.best_odds)}" for o in filtered[:3])

        if h2h:
            home_odds = next(
                (o.best_odds for o in h2h.outcomes if o.name == match.home_team), 0
            )
            draw_odds = next((o.best_odds for o in h2h.outcomes if o.name == "Draw"), 0)
            away_odds = next(
                (o.best_odds for o in h2h.outcomes if o.name == match.away_team), 0
            )
            h2h_str = f"[green]{format_odds(home_odds)}[/green] / [dim]{format_odds(draw_odds)}[/dim] / [red]{format_odds(away_odds)}[/red]"
        else:
            h2h_str = "[dim]—[/dim]"

        if tot:
            over = next((o for o in tot.outcomes if "Over" in o.name), None)
            under = next((o for o in tot.outcomes if "Under" in o.name), None)
            line = over.name.split()[-1] if over and len(over.name.split()) > 1 else "?"
            ou_str = f"O[green]{format_odds(over.best_odds if over else 0)}[/green] U[red]{format_odds(under.best_odds if under else 0)}[/red]"
            if line != "?":
                ou_str += f" [dim]({line})[/dim]"
        else:
            ou_str = "[dim]—[/dim]"

        if btts:
            yes = next((o.best_odds for o in btts.outcomes if o.name == "Yes"), 0)
            no_ = next((o.best_odds for o in btts.outcomes if o.name == "No"), 0)
            btts_str = (
                f"Y[green]{format_odds(yes)}[/green] N[red]{format_odds(no_)}[/red]"
            )
        else:
            btts_str = "[dim]—[/dim]"

        ev = match.top_bet_ev
        color = ev_color(ev)
        ev_str = (
            f"[bold {color}]+{ev:.1f}%[/bold {color}]"
            if ev > 0
            else f"[dim]{ev:.1f}%[/dim]"
        )

        rating_str = ""
        for mkt in match.markets:
            if mkt.rating != "—":
                rating_str = f"[{color}]{mkt.rating}[/{color}]"
                break

        table.add_row(
            f"[bold]{match.home_team}[/bold]\n[dim]vs {match.away_team}[/dim]",
            f"[dim]{match.sport_name[:15]}[/dim]",
            format_hours(match.hours_until),
            str(match.num_bookmakers),
            h2h_str,
            ou_str,
            btts_str,
            ev_str,
            rating_str or "[dim]—[/dim]",
        )

    console.print(table)
    console.print()


def print_match_detail(match: MatchAnalysis):
    console.print()
    dt = format_kickoff_wib(match.commence_dt)
    console.print(
        Panel(
            f"[bold white]{match.home_team}[/bold white]  [dim]vs[/dim]  [bold white]{match.away_team}[/bold white]\n"
            f"[dim]{match.sport_name} | {dt} | {match.num_bookmakers} bookmakers[/dim]",
            title=f"[bold bright_cyan]Detailed Analysis[/bold bright_cyan]",
            border_style="bright_cyan",
        )
    )

    for mkt in match.markets:
        color = ev_color(mkt.best_ev_pct)
        mkt_table = Table(
            title=f"[bold]{mkt.market_label}[/bold]  [dim](Overround: {(mkt.overround - 1) * 100:.1f}%)[/dim]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
            expand=False,
        )
        mkt_table.add_column("Outcome", min_width=14)
        mkt_table.add_column("Best Odds", min_width=8, justify="center")
        mkt_table.add_column("Avg Odds", min_width=8, justify="center")
        mkt_table.add_column("Best Book", min_width=12)
        mkt_table.add_column("Odds Range", min_width=10)
        mkt_table.add_column("Fair Prob", min_width=18)
        mkt_table.add_column("EV%", min_width=8, justify="right")
        mkt_table.add_column("AI Signal", min_width=16)

        for o in sorted(mkt.outcomes, key=lambda x: -x.best_odds):
            fp = mkt.fair_probs.get(o.name, 0)
            ev = (fp * o.best_odds - 1) * 100
            o_color = ev_color(ev)
            spark = sparkline(o.all_odds)

            mkt_table.add_row(
                f"[bold]{o.name}[/bold]",
                f"[bold {o_color}]{format_odds(o.best_odds)}[/bold {o_color}]",
                f"[dim]{format_odds(o.avg_odds)}[/dim]",
                f"[dim]{o.bookmaker}[/dim]",
                f"[dim]{spark}[/dim]",
                f"[cyan]{prob_bar(fp, 15)}[/cyan]",
                f"[bold {o_color}]{ev:+.1f}%[/bold {o_color}]",
                signal_style(mkt.ai_signal) if o.name == mkt.best_outcome else "",
            )

        console.print(mkt_table)

    if match.top_bet_ev >= ANALYSIS_CONFIG.min_edge_pct:
        stake = match.top_bet_kelly / 100 * ANALYSIS_CONFIG.bankroll
        console.print(
            Panel(
                f"[bold green]🎯 RECOMMENDED BET[/bold green]\n\n"
                f"  Market   : [white]{match.top_bet_market}[/white]\n"
                f"  Selection: [bold white]{match.top_bet_outcome}[/bold white]\n"
                f"  Best Odds: [bold green]{format_odds(match.top_bet_odds)}[/bold green] @ [dim]{match.top_bet_book}[/dim]\n"
                f"  Edge     : [bold green]+{match.top_bet_ev:.1f}%[/bold green] EV\n"
                f"  Kelly %  : [bold cyan]{match.top_bet_kelly:.2f}%[/bold cyan] of bankroll\n"
                f"  Stake    : [bold green]${stake:.2f}[/bold green] "
                f"[dim](of ${ANALYSIS_CONFIG.bankroll:,.0f} bankroll)[/dim]",
                border_style="green",
            )
        )


def print_statistics(matches: List[MatchAnalysis]):
    total = len(matches)
    value_count = sum(1 for m in matches if m.has_value)
    strong_count = sum(1 for m in matches if m.top_bet_ev >= 8)
    avg_ev = sum(m.top_bet_ev for m in matches if m.has_value) / max(value_count, 1)
    total_kelly = sum(m.top_bet_kelly for m in matches if m.has_value)

    leagues = len(set(m.sport_name for m in matches))
    markets_analyzed = sum(len(m.markets) for m in matches)

    console.print(Rule("[bold white]SESSION STATISTICS[/bold white]", style="white"))
    console.print()

    stats = [
        Panel(
            f"[bold bright_green]{total}[/bold bright_green]\n[dim]Matches Analyzed[/dim]",
            border_style="green",
            padding=(0, 2),
        ),
        Panel(
            f"[bold bright_green]{value_count}[/bold bright_green]\n[dim]Value Bets Found[/dim]",
            border_style="green",
            padding=(0, 2),
        ),
        Panel(
            f"[bold bright_yellow]{strong_count}[/bold bright_yellow]\n[dim]Strong Value (>8%)[/dim]",
            border_style="yellow",
            padding=(0, 2),
        ),
        Panel(
            f"[bold cyan]{avg_ev:.1f}%[/bold cyan]\n[dim]Avg EV on Value Bets[/dim]",
            border_style="cyan",
            padding=(0, 2),
        ),
        Panel(
            f"[bold white]{leagues}[/bold white]\n[dim]Leagues Covered[/dim]",
            border_style="white",
            padding=(0, 2),
        ),
        Panel(
            f"[bold white]{markets_analyzed}[/bold white]\n[dim]Markets Analyzed[/dim]",
            border_style="white",
            padding=(0, 2),
        ),
    ]

    console.print(Columns(stats, equal=True))
    console.print()

    xg_conflict_count = 0
    for m in matches:
        if m.has_value:
            for mkt in m.markets:
                if mkt.market_label == m.top_bet_market:
                    if getattr(mkt, "xg_signal", "") == "XG_CONFLICT":
                        xg_conflict_count += 1
                    break

    if xg_conflict_count > 0:
        console.print(
            f"  [bold yellow]⚠ xG Conflicts: {xg_conflict_count} matches[/bold yellow] [dim](Memiliki prediksi xG yang sangat bertentangan dengan Odds)[/dim]"
        )
        console.print()

    _rpt = next((getattr(m, "_portfolio_report", None) for m in matches), None)
    if _rpt and _rpt.session:
        snap = _rpt.session
        bar = snap.progress_bar(22)
        status_color = {
            "SAFE": "bright_green",
            "WARNING": "yellow",
            "EXCEEDED": "red",
        }.get(snap.status, "white")
        status_label = {
            "SAFE": "AMAN",
            "WARNING": "PERHATIAN",
            "EXCEEDED": "TERLAMPAUI",
        }.get(snap.status, snap.status)
        corr_count = len(_rpt.correlation_advisories)
        sup_count = _rpt.n_suppressed
        parts = [
            f"[{status_color}]{bar}[/{status_color}]"
            f"  Rp {snap.total_stake_rp:,.0f} / Rp {snap.guidance_limit_rp:,.0f}"
            f"  ([bold {status_color}]{snap.pct_used:.0f}% — {status_label}[/bold {status_color}])",
        ]
        if corr_count:
            parts.append(
                f"  [dim]Korelasi terdeteksi: {corr_count} pasangan pasaran "
                f"| Panduan maks per-match: Rp {_rpt.recommended_max_per_match_rp:,.0f}[/dim]"
            )
        if sup_count:
            parts.append(
                f"  [red]Hard suppress: {sup_count} market (korelasi ekstrem >85%)[/red]"
            )
        console.print(
            Panel(
                "\n".join(parts),
                title="[bold]Session Exposure Tracker[/bold]",
                border_style=status_color,
                padding=(0, 2),
            )
        )
        console.print()


def print_mispricing_report(matches: List[MatchAnalysis]):
    console.print(
        Rule(
            "[bold bright_magenta]MISPRICING DETECTOR[/bold bright_magenta]",
            style="bright_magenta",
        )
    )
    console.print()

    mispricings = []
    for match in matches:
        for mkt in match.markets:
            for o in mkt.outcomes:
                fp = mkt.fair_probs.get(o.name, 0)
                if fp <= 0:
                    continue
                fair_odds = 1.0 / fp
                gap = o.best_odds - fair_odds
                gap_pct = (gap / fair_odds) * 100
                if gap_pct >= ANALYSIS_CONFIG.min_edge_pct:
                    ev = expected_value_pct_local(fp, o.best_odds)
                    mispricings.append(
                        {
                            "match": match.match_label,
                            "sport": match.sport_name,
                            "hours": match.hours_until,
                            "market": mkt.market_label,
                            "outcome": o.name,
                            "market_odds": o.best_odds,
                            "fair_odds": fair_odds,
                            "gap_pct": gap_pct,
                            "ev": ev,
                            "book": o.bookmaker,
                        }
                    )

    mispricings.sort(key=lambda x: -x["gap_pct"])

    if not mispricings:
        console.print("[yellow]  No significant mispricings detected.[/yellow]")
        console.print()
        return

    table = Table(
        box=box.MINIMAL,
        show_header=True,
        header_style="bold magenta",
        border_style="magenta",
        expand=True,
    )
    table.add_column("Match", min_width=22)
    table.add_column("Market", min_width=16)
    table.add_column("Outcome", min_width=14)
    table.add_column("Market Odds", min_width=10, justify="center")
    table.add_column("Fair Odds", min_width=10, justify="center")
    table.add_column("Mispricing", min_width=10, justify="right")
    table.add_column("EV%", min_width=8, justify="right")
    table.add_column("Book", min_width=12)
    table.add_column("Kick-off", min_width=6, justify="center")

    for mp in mispricings[:20]:
        color = ev_color(mp["ev"])
        table.add_row(
            f"[white]{mp['match'][:28]}[/white]",
            f"[dim]{mp['market']}[/dim]",
            f"[bold]{mp['outcome']}[/bold]",
            f"[bold {color}]{format_odds(mp['market_odds'])}[/bold {color}]",
            f"[dim]{format_odds(mp['fair_odds'])}[/dim]",
            f"[bold bright_magenta]+{mp['gap_pct']:.1f}%[/bold bright_magenta]",
            f"[bold {color}]+{mp['ev']:.1f}%[/bold {color}]",
            f"[dim]{mp['book']}[/dim]",
            format_hours(mp["hours"]),
        )

    console.print(table)
    console.print()


def expected_value_pct_local(fair_prob: float, decimal_odds: float) -> float:
    return (fair_prob * decimal_odds - 1) * 100


def print_llm_insights(matches: List[MatchAnalysis]):
    enriched = [
        m
        for m in matches
        if m.has_value and hasattr(m, "llm_analysis") and m.llm_analysis
    ]
    if not enriched:
        return

    real = [m for m in enriched if not m.llm_analysis.is_fallback]
    if not real:
        return

    console.print(
        Rule(
            "[bold bright_yellow]AI DEEP ANALYSIS[/bold bright_yellow]",
            style="bright_yellow",
        )
    )
    console.print()

    for match in real[:8]:
        llm = match.llm_analysis
        ev = match.top_bet_ev
        color = ev_color(ev)

        rs = getattr(llm, "risk_score", 5.0)
        if rs <= 2:
            rec_str = f"[bold bright_green]LOW RISK {rs:.0f}/10 ✅[/bold bright_green]"
        elif rs <= 4:
            rec_str = f"[bold green]RISK {rs:.0f}/10 ★[/bold green]"
        elif rs <= 6:
            rec_str = f"[yellow]RISK {rs:.0f}/10 ⚠[/yellow]"
        elif rs <= 7.9:
            rec_str = f"[bold red]HIGH RISK {rs:.0f}/10 ⚠⚠[/bold red]"
        else:
            rec_str = f"[bold bright_red]DANGER {rs:.0f}/10 ✗[/bold bright_red]"

        conf_pct = int(llm.confidence * 10)
        conf_bar = (
            "[bright_green]"
            + "=" * conf_pct
            + "[/bright_green]"
            + "[dim]"
            + "-" * (10 - conf_pct)
            + "[/dim]"
        )

        factors_str = ""
        if llm.key_factors:
            factors_str = "\n".join(
                f"  [green]+ {f}[/green]" for f in llm.key_factors[:3]
            )

        risks_str = ""
        if llm.risk_flags:
            risks_str = "\n".join(f"  [red]! {r}[/red]" for r in llm.risk_flags[:2])

        lines = [
            f"[bold white]{match.home_team} vs {match.away_team}[/bold white]"
            f"  [dim]({match.sport_name}  |  {format_hours(match.hours_until)} to kick-off)[/dim]",
            f"",
            f"  Bet       : [bold white]{match.top_bet_outcome}[/bold white]"
            f" @ [bold {color}]{format_odds(match.top_bet_odds)}[/bold {color}]"
            f"  [dim]({match.top_bet_book})[/dim]",
            f"  EV        : [bold {color}]+{ev:.1f}%[/bold {color}]"
            f"  |  Kelly  : [cyan]{match.top_bet_kelly:.2f}%[/cyan]",
            f"",
            f"  AI Signal : {rec_str}   Confidence: [{conf_bar}] {llm.confidence * 100:.0f}%",
            f"",
            f"  Reasoning : [italic]{llm.reasoning}[/italic]",
        ]

        if factors_str:
            lines += ["", "  Key Factors:", factors_str]
        if risks_str:
            lines += ["", "  Risk Flags:", risks_str]

        if llm.model_used:
            lines += [
                "",
                f"  [dim]Model: {llm.model_used}  |  Tokens: {llm.tokens_used}[/dim]",
            ]

        console.print(
            Panel(
                "\n".join(lines),
                border_style=color,
                padding=(0, 2),
            )
        )
        console.print()


def _wr_color(wr: float) -> str:
    if wr >= 60:
        return "bright_green"
    elif wr >= 40:
        return "yellow"
    return "red"


def _pnl_str(pnl: float) -> str:
    sign = "+" if pnl >= 0 else ""
    color = "bright_green" if pnl >= 0 else "red"
    return f"[bold {color}]{sign}{pnl:,.2f}[/bold {color}]"


def print_wr_report(report: dict, compact: bool = False):
    if not report:
        console.print(
            Panel(
                "[yellow]Belum ada data di picks_log.csv[/yellow]",
                title="[bold]📊 WR Report[/bold]",
                border_style="yellow",
            )
        )
        return

    wr = report["win_rate_pct"]
    wr_c = _wr_color(wr)

    pnl_sign = "+" if report["total_pnl"] >= 0 else ""
    pnl_color = "bright_green" if report["total_pnl"] >= 0 else "red"
    roi_color = "bright_green" if report["roi_pct"] >= 0 else "red"

    header_content = (
        f"  [dim]Period[/dim]       : [white]{report['period']}[/white]\n"
        f"  [dim]Total Picks[/dim]  : [bold white]{report['total_picks']}[/bold white]\n"
        f"  [dim]Settled[/dim]      : [bold white]{report['settled']}[/bold white]"
        f"  [dim](W:[/dim][green]{report['wins']}[/green]"
        f" [dim]/ L:[/dim][red]{report['losses']}[/red]"
        f" [dim]/ VOID:[/dim][white]{report['voids']}[/white][dim])[/dim]\n"
        f"  [dim]Pending[/dim]      : [yellow]{report['pending']}[/yellow]\n"
        f"\n"
        f"  [bold]Win Rate[/bold]     : [bold {wr_c}]{wr:.1f}%[/bold {wr_c}]"
    )

    bar_len = 20
    filled = round(wr / 100 * bar_len)
    bar = f"[{wr_c}]{'█' * filled}[/{wr_c}][dim]{'░' * (bar_len - filled)}[/dim]"
    header_content += f"  {bar}\n"

    header_content += (
        f"  [bold]Profit/Loss[/bold]  : [bold {pnl_color}]{pnl_sign}{report['total_pnl']:,.2f}[/bold {pnl_color}]"
        f"  [dim](Stake: {report['total_stake']:,.2f})[/dim]\n"
        f"  [bold]ROI[/bold]          : [bold {roi_color}]{report['roi_pct']:+.1f}%[/bold {roi_color}]"
    )

    console.print(
        Panel(
            header_content,
            title="[bold bright_cyan]📊  WIN RATE REPORT[/bold bright_cyan]",
            border_style="bright_cyan",
            padding=(1, 2),
        )
    )

    if compact:
        console.print()
        return

    if report.get("by_signal"):
        sig_table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold bright_yellow",
            border_style="yellow",
            title="[bold]Win Rate per AI Signal[/bold]",
            expand=True,
            padding=(0, 1),
        )
        sig_table.add_column("AI Signal", min_width=18, no_wrap=True)
        sig_table.add_column("W", width=5, justify="center", style="green")
        sig_table.add_column("L", width=5, justify="center", style="red")
        sig_table.add_column("WR%", width=8, justify="center")
        sig_table.add_column("P&L", min_width=12, justify="right")

        for sig, s in sorted(report["by_signal"].items()):
            s_wr = s["win_rate"]
            s_c = _wr_color(s_wr)
            sig_table.add_row(
                f"[bold]{sig}[/bold]",
                str(s["wins"]),
                str(s["losses"]),
                f"[bold {s_c}]{s_wr:.1f}%[/bold {s_c}]",
                _pnl_str(s["pnl"]),
            )
        console.print(sig_table)
        console.print()

    if report.get("by_league"):
        lg_table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold bright_blue",
            border_style="blue",
            title="[bold]Win Rate per Liga[/bold]",
            expand=True,
            padding=(0, 1),
        )
        lg_table.add_column("League", min_width=24, no_wrap=True)
        lg_table.add_column("W", width=5, justify="center", style="green")
        lg_table.add_column("L", width=5, justify="center", style="red")
        lg_table.add_column("WR%", width=8, justify="center")
        lg_table.add_column("P&L", min_width=12, justify="right")

        for lg, s in sorted(report["by_league"].items(), key=lambda x: -x[1]["wins"]):
            l_wr = s["win_rate"]
            l_c = _wr_color(l_wr)
            lg_table.add_row(
                f"[white]{lg[:28]}[/white]",
                str(s["wins"]),
                str(s["losses"]),
                f"[bold {l_c}]{l_wr:.1f}%[/bold {l_c}]",
                _pnl_str(s["pnl"]),
            )
        console.print(lg_table)
        console.print()

    if report.get("by_market"):
        mkt_table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold bright_magenta",
            border_style="magenta",
            title="[bold]Win Rate per Market[/bold]",
            expand=True,
            padding=(0, 1),
        )
        mkt_table.add_column("Market", min_width=22, no_wrap=True)
        mkt_table.add_column("W", width=5, justify="center", style="green")
        mkt_table.add_column("L", width=5, justify="center", style="red")
        mkt_table.add_column("WR%", width=8, justify="center")
        mkt_table.add_column("P&L", min_width=12, justify="right")

        for mkt, s in sorted(report["by_market"].items(), key=lambda x: -x[1]["wins"]):
            m_wr = s["win_rate"]
            m_c = _wr_color(m_wr)
            mkt_table.add_row(
                f"[bright_blue]{mkt[:28]}[/bright_blue]",
                str(s["wins"]),
                str(s["losses"]),
                f"[bold {m_c}]{m_wr:.1f}%[/bold {m_c}]",
                _pnl_str(s["pnl"]),
            )
        console.print(mkt_table)
        console.print()

    if report.get("by_ev_tier"):
        ev_table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold bright_green",
            border_style="green",
            title="[bold]Win Rate per EV Tier[/bold]",
            expand=True,
            padding=(0, 1),
        )
        ev_table.add_column("EV Tier", min_width=16, no_wrap=True)
        ev_table.add_column("W", width=5, justify="center", style="green")
        ev_table.add_column("L", width=5, justify="center", style="red")
        ev_table.add_column("WR%", width=8, justify="center")
        ev_table.add_column("Avg EV", width=8, justify="center")
        ev_table.add_column("P&L", min_width=12, justify="right")

        tier_order = ["EV 8%+", "EV 4-8%", "EV 2-4%", "EV <2%"]
        for tier in tier_order:
            if tier in report["by_ev_tier"]:
                s = report["by_ev_tier"][tier]
                t_wr = s["win_rate"]
                t_c = _wr_color(t_wr)
                ev_table.add_row(
                    f"[bold]{tier}[/bold]",
                    str(s["wins"]),
                    str(s["losses"]),
                    f"[bold {t_c}]{t_wr:.1f}%[/bold {t_c}]",
                    f"[cyan]{s.get('avg_ev', 0):.1f}%[/cyan]",
                    _pnl_str(s["pnl"]),
                )
        console.print(ev_table)
        console.print()

    if report.get("streak"):
        sk = report["streak"]
        sk_type = sk.get("type", "")
        sk_len = sk.get("length", 0)
        if sk_type == "W":
            sk_str = f"[bold bright_green]🔥 {sk_len}W STREAK[/bold bright_green]"
        elif sk_type == "L":
            sk_str = f"[bold red]❄ {sk_len}L STREAK[/bold red]"
        else:
            sk_str = "[dim]No active streak[/dim]"

        best = report.get("best_streak", 0)
        worst = report.get("worst_streak", 0)
        console.print(
            Panel(
                f"  Current  : {sk_str}\n"
                f"  Best Win : [green]{best}W[/green]  "
                f"| Worst Loss : [red]{worst}L[/red]",
                title="[bold]Streak Tracker[/bold]",
                border_style="bright_cyan",
                padding=(0, 2),
            )
        )
        console.print()
