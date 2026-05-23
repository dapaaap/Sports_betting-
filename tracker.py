
import asyncio
import csv
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_DIR, "picks_log.csv")

FIELDNAMES = [
    "id",
    "date",
    "kickoff",
    "match",
    "league",
    "market",
    "outcome",
    "ev",
    "ai_signal",
    "odds",
    "stake",
    "cqs",
    "result",
    "profit_loss",
    "notes",
]

RESULT_VALUES = {"WIN", "LOSS", "VOID", "PUSH", ""}


def _ensure_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        logger.info("tracker: CSV dibuat → %s", CSV_PATH)


def _load_rows() -> List[Dict]:
    _ensure_csv()
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_rows(rows: List[Dict]):
    _ensure_csv()
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _make_id(event_id: str, market_key: str) -> str:
    return f"{event_id}::{market_key}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def log_pick(match, notes: str = "") -> bool:
    if not getattr(match, "has_value", False):
        return False

    _ensure_csv()

    best_mkt = None
    for mkt in getattr(match, "markets", []):
        if mkt.market_label == match.top_bet_market:
            best_mkt = mkt
            break

    market_key = best_mkt.market_key if best_mkt else match.top_bet_market or ""
    pick_id = _make_id(match.event_id, market_key)

    rows = _load_rows()
    existing_ids = {r["id"] for r in rows}
    if pick_id in existing_ids:
        logger.debug("tracker: pick sudah ada, skip → %s", pick_id)
        return False

    llm = getattr(match, "llm_analysis", None)
    if llm and not llm.is_fallback:
        rs = getattr(llm, "risk_score", None)
        if rs is not None:
            if rs <= 2:
                ai_signal = f"LOW RISK {rs:.0f}/10"
            elif rs <= 4:
                ai_signal = f"RISK {rs:.0f}/10"
            elif rs <= 6:
                ai_signal = f"RISK {rs:.0f}/10 [!]"
            elif rs <= 7.9:
                ai_signal = f"HIGH RISK {rs:.0f}/10"
            else:
                ai_signal = f"DANGER {rs:.0f}/10"
        else:
            ai_signal = (
                getattr(best_mkt, "ai_signal", "NEUTRAL") if best_mkt else "NEUTRAL"
            )
    elif best_mkt:
        ai_signal = best_mkt.ai_signal
    else:
        ai_signal = "NEUTRAL"

    kickoff_str = ""
    try:
        kickoff_str = match.commence_dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        kickoff_str = match.commence_time or ""

    row = {
        "id": pick_id,
        "date": _today(),
        "kickoff": kickoff_str,
        "match": match.match_label,
        "league": match.sport_name,
        "market": match.top_bet_market or "",
        "outcome": match.top_bet_outcome or "",
        "ev": round(match.top_bet_ev, 2),
        "ai_signal": ai_signal,
        "odds": round(match.top_bet_odds, 2),
        "stake": round(best_mkt.kelly_stake, 2) if best_mkt else 0.0,
        "cqs": round(getattr(match, "cqs", 0), 1),
        "result": "",
        "profit_loss": "",
        "notes": notes,
    }

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)

    logger.info("tracker: pick disimpan → %s", match.match_label)
    return True


def log_picks(matches: list, notes: str = "") -> int:
    saved = 0
    for m in matches:
        if getattr(m, "has_value", False):
            if log_pick(m, notes=notes):
                saved += 1
    logger.info("tracker: %d pick baru disimpan ke CSV.", saved)
    return saved


def update_result(
    match_label: str,
    result: str,
    profit_loss: Optional[float] = None,
    notes: str = "",
    kickoff_date: Optional[str] = None,
) -> int:
    result = result.upper().strip()
    if result not in RESULT_VALUES:
        raise ValueError(f"result harus salah satu dari: {RESULT_VALUES}")

    rows = _load_rows()
    updated = 0

    for row in rows:
        if match_label.lower() not in row["match"].lower():
            continue
        if kickoff_date and not row["kickoff"].startswith(kickoff_date):
            continue
        if row["result"] in ("WIN", "LOSS", "VOID", "PUSH"):
            continue

        row["result"] = result

        if profit_loss is not None:
            row["profit_loss"] = round(profit_loss, 2)
        elif result in ("WIN",) and row.get("stake") and row.get("odds"):
            try:
                stake = float(row["stake"])
                odds = float(row["odds"])
                row["profit_loss"] = round(stake * (odds - 1), 2)
            except (ValueError, TypeError):
                pass
        elif result == "LOSS" and row.get("stake"):
            try:
                row["profit_loss"] = -round(float(row["stake"]), 2)
            except (ValueError, TypeError):
                pass
        elif result in ("VOID", "PUSH"):
            row["profit_loss"] = 0.0

        if notes:
            row["notes"] = notes

        updated += 1

    if updated:
        _save_rows(rows)
        logger.info("tracker: %d baris diupdate → result=%s", updated, result)
    else:
        logger.warning("tracker: tidak ada baris yang cocok untuk '%s'", match_label)

    return updated


async def auto_update_results(api_key: str = "", lookback_days: int = 7) -> int:
    import aiohttp

    api_key = api_key or os.environ.get("FOOTBALL_DATA_KEY", "")
    if not api_key:
        logger.warning(
            "tracker.auto_update_results: FOOTBALL_DATA_KEY tidak diset. "
            "Daftarkan di https://www.football-data.org/ (gratis)."
        )
        return 0

    rows_pending = [r for r in _load_rows() if r["result"] == "" and r["kickoff"]]
    if not rows_pending:
        logger.info("tracker: tidak ada pick pending result.")
        return 0

    from_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
        "%Y-%m-%d"
    )
    to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    url = (
        f"https://api.football-data.org/v4/matches"
        f"?dateFrom={from_date}&dateTo={to_date}&status=FINISHED"
    )
    headers = {"X-Auth-Token": api_key}

    updated_count = 0
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    logger.warning("tracker: football-data.org HTTP %d", resp.status)
                    return 0
                data = await resp.json()

        matches_api = data.get("matches", [])

        result_lookup: Dict[str, dict] = {}
        for m in matches_api:
            home = m.get("homeTeam", {}).get("name", "")
            away = m.get("awayTeam", {}).get("name", "")
            label = f"{home} vs {away}"
            score = m.get("score", {}).get("fullTime", {})
            result_lookup[label.lower()] = {
                "home_score": score.get("home"),
                "away_score": score.get("away"),
                "status": m.get("status", ""),
            }

        for row in rows_pending:
            match_key = row["match"].lower()

            info = result_lookup.get(match_key)
            if not info:
                for key, val in result_lookup.items():
                    parts = match_key.split(" vs ")
                    if len(parts) == 2:
                        home_q = parts[0].strip()
                        away_q = parts[1].strip()
                        if home_q in key and away_q in key:
                            info = val
                            break

            if not info or info["status"] != "FINISHED":
                continue

            home_score = info["home_score"]
            away_score = info["away_score"]
            if home_score is None or away_score is None:
                continue

            outcome = row.get("outcome", "").lower()
            home_team = row["match"].split(" vs ")[0].strip().lower()
            away_team = row["match"].split(" vs ")[-1].strip().lower()

            result_val = _determine_result(
                outcome=outcome,
                home_score=home_score,
                away_score=away_score,
                home_team=home_team,
                away_team=away_team,
            )
            if result_val:
                n = update_result(
                    match_label=row["match"],
                    result=result_val,
                    kickoff_date=row["kickoff"][:10] if row["kickoff"] else None,
                )
                updated_count += n

    except Exception as e:
        logger.error("tracker.auto_update_results error: %s", e)

    logger.info("tracker: auto_update selesai — %d pick diupdate.", updated_count)
    return updated_count


def _determine_result(
    outcome: str,
    home_score: int,
    away_score: int,
    home_team: str,
    away_team: str,
) -> Optional[str]:
    outcome = outcome.lower()

    if home_team in outcome or outcome in ("1", "home"):
        actual = (
            "WIN"
            if home_score > away_score
            else ("PUSH" if home_score == away_score else "LOSS")
        )
        return actual

    if away_team in outcome or outcome in ("2", "away"):
        actual = (
            "WIN"
            if away_score > home_score
            else ("PUSH" if home_score == away_score else "LOSS")
        )
        return actual

    if outcome in ("draw", "x", "tie"):
        return "WIN" if home_score == away_score else "LOSS"

    if "over" in outcome:
        try:
            line = float(outcome.split()[-1])
            total = home_score + away_score
            return "WIN" if total > line else "LOSS"
        except (ValueError, IndexError):
            pass
    if "under" in outcome:
        try:
            line = float(outcome.split()[-1])
            total = home_score + away_score
            return "WIN" if total < line else "LOSS"
        except (ValueError, IndexError):
            pass

    if "both" in outcome or "btts" in outcome:
        btts = home_score > 0 and away_score > 0
        if "yes" in outcome:
            return "WIN" if btts else "LOSS"
        if "no" in outcome:
            return "WIN" if not btts else "LOSS"
    return None


def weekly_report(
    weeks_back: int = 1,
    print_output: bool = True,
    compact: bool = False,
) -> Dict:
    rows = _load_rows()
    if not rows:
        if print_output:
            _print_report({}, compact=compact)
        return {}

    cutoff_start = datetime.now(timezone.utc) - timedelta(weeks=weeks_back)
    cutoff_end = datetime.now(timezone.utc) - timedelta(weeks=weeks_back - 1)

    week_rows = []
    for r in rows:
        try:
            row_date = datetime.strptime(r["date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            if cutoff_start <= row_date < cutoff_end:
                week_rows.append(r)
        except (ValueError, KeyError):
            pass

    if weeks_back == 0:
        week_rows = rows

    if not week_rows:
        if print_output:
            _print_report({}, compact=compact)
        return {}

    total = len(week_rows)
    wins = [r for r in week_rows if r["result"] == "WIN"]
    losses = [r for r in week_rows if r["result"] == "LOSS"]
    voids = [r for r in week_rows if r["result"] in ("VOID", "PUSH")]
    pending = [r for r in week_rows if r["result"] == ""]
    settled = wins + losses

    win_rate = (len(wins) / len(settled) * 100) if settled else 0.0

    def _pnl(rows_list):
        total_pnl = 0.0
        for r in rows_list:
            try:
                total_pnl += float(r["profit_loss"])
            except (ValueError, TypeError):
                pass
        return total_pnl

    def _stake_total(rows_list):
        s = 0.0
        for r in rows_list:
            try:
                s += float(r["stake"])
            except (ValueError, TypeError):
                pass
        return s

    total_pnl = _pnl(week_rows)
    total_stake = _stake_total(settled)
    roi_pct = (total_pnl / total_stake * 100) if total_stake > 0 else 0.0

    signal_stats: Dict[str, Dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "pnl": 0.0}
    )
    for r in week_rows:
        sig = _normalize_signal(r.get("ai_signal", "UNKNOWN"))
        if r["result"] == "WIN":
            signal_stats[sig]["wins"] += 1
            signal_stats[sig]["pnl"] += _safe_float(r["profit_loss"])
        elif r["result"] == "LOSS":
            signal_stats[sig]["losses"] += 1
            signal_stats[sig]["pnl"] += _safe_float(r["profit_loss"])

    signal_report = {}
    for sig, s in signal_stats.items():
        total_sig = s["wins"] + s["losses"]
        signal_report[sig] = {
            "wins": s["wins"],
            "losses": s["losses"],
            "win_rate": round(s["wins"] / total_sig * 100, 1) if total_sig else 0.0,
            "pnl": round(s["pnl"], 2),
        }

    league_stats: Dict[str, Dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "pnl": 0.0}
    )
    for r in week_rows:
        lg = r.get("league", "Unknown")
        if r["result"] == "WIN":
            league_stats[lg]["wins"] += 1
            league_stats[lg]["pnl"] += _safe_float(r["profit_loss"])
        elif r["result"] == "LOSS":
            league_stats[lg]["losses"] += 1
            league_stats[lg]["pnl"] += _safe_float(r["profit_loss"])

    league_report = {}
    for lg, s in league_stats.items():
        total_lg = s["wins"] + s["losses"]
        league_report[lg] = {
            "wins": s["wins"],
            "losses": s["losses"],
            "win_rate": round(s["wins"] / total_lg * 100, 1) if total_lg else 0.0,
            "pnl": round(s["pnl"], 2),
        }

    market_stats: Dict[str, Dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "pnl": 0.0}
    )
    for r in week_rows:
        mkt = r.get("market", "Unknown") or "Unknown"
        if r["result"] == "WIN":
            market_stats[mkt]["wins"] += 1
            market_stats[mkt]["pnl"] += _safe_float(r["profit_loss"])
        elif r["result"] == "LOSS":
            market_stats[mkt]["losses"] += 1
            market_stats[mkt]["pnl"] += _safe_float(r["profit_loss"])

    market_report = {}
    for mkt, s in market_stats.items():
        total_mkt = s["wins"] + s["losses"]
        market_report[mkt] = {
            "wins": s["wins"],
            "losses": s["losses"],
            "win_rate": round(s["wins"] / total_mkt * 100, 1) if total_mkt else 0.0,
            "pnl": round(s["pnl"], 2),
        }

    ev_tiers: Dict[str, Dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "ev_sum": 0.0, "count": 0}
    )
    for r in week_rows:
        ev_val = _safe_float(r.get("ev", 0))
        if ev_val >= 8.0:
            tier = "EV 8%+"
        elif ev_val >= 4.0:
            tier = "EV 4-8%"
        elif ev_val >= 2.0:
            tier = "EV 2-4%"
        else:
            tier = "EV <2%"
        ev_tiers[tier]["ev_sum"] += ev_val
        ev_tiers[tier]["count"] += 1
        if r["result"] == "WIN":
            ev_tiers[tier]["wins"] += 1
            ev_tiers[tier]["pnl"] += _safe_float(r["profit_loss"])
        elif r["result"] == "LOSS":
            ev_tiers[tier]["losses"] += 1
            ev_tiers[tier]["pnl"] += _safe_float(r["profit_loss"])

    ev_tier_report = {}
    for tier, s in ev_tiers.items():
        total_t = s["wins"] + s["losses"]
        ev_tier_report[tier] = {
            "wins": s["wins"],
            "losses": s["losses"],
            "win_rate": round(s["wins"] / total_t * 100, 1) if total_t else 0.0,
            "pnl": round(s["pnl"], 2),
            "avg_ev": round(s["ev_sum"] / s["count"], 1) if s["count"] else 0.0,
        }

    settled_sorted = [r for r in week_rows if r["result"] in ("WIN", "LOSS")]

    settled_sorted.sort(key=lambda r: (r.get("date", ""), r.get("kickoff", "")))

    current_streak = {"type": "", "length": 0}
    best_w_streak = 0
    worst_l_streak = 0
    _run_type = ""
    _run_len = 0

    for r in settled_sorted:
        res = r["result"]
        if res == _run_type:
            _run_len += 1
        else:
            _run_type = res
            _run_len = 1
        if _run_type == "WIN":
            best_w_streak = max(best_w_streak, _run_len)
        elif _run_type == "LOSS":
            worst_l_streak = max(worst_l_streak, _run_len)

    if _run_type and _run_len > 0:
        current_streak = {
            "type": "W" if _run_type == "WIN" else "L",
            "length": _run_len,
        }

    report = {
        "period": f"{cutoff_start.strftime('%Y-%m-%d')} – {cutoff_end.strftime('%Y-%m-%d')}",
        "total_picks": total,
        "wins": len(wins),
        "losses": len(losses),
        "voids": len(voids),
        "pending": len(pending),
        "settled": len(settled),
        "win_rate_pct": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "total_stake": round(total_stake, 2),
        "roi_pct": round(roi_pct, 1),
        "by_signal": signal_report,
        "by_league": league_report,
        "by_market": market_report,
        "by_ev_tier": ev_tier_report,
        "streak": current_streak,
        "best_streak": best_w_streak,
        "worst_streak": worst_l_streak,
    }

    if print_output:
        _print_report(report, compact=compact)

    return report


def _normalize_signal(sig: str) -> str:
    sig_upper = sig.upper()

    if "DANGER" in sig_upper:
        return "DANGER (9-10)"
    if "HIGH RISK" in sig_upper:
        return "HIGH RISK (7-8)"
    if "LOW RISK" in sig_upper:
        return "LOW RISK (0-2)"
    if "RISK" in sig_upper and "/10" in sig_upper:
        try:
            num = float(sig_upper.split("/")[0].split()[-1])
            if num <= 4:
                return "LOW-MED RISK (3-4)"
            return "MED RISK (5-6)"
        except (ValueError, IndexError):
            return "MED RISK"

    if "STRONG BUY" in sig_upper or "STRONG" in sig_upper:
        return "STRONG BUY"
    if "BUY" in sig_upper:
        return "BUY"
    if "AVOID" in sig_upper or "SKIP" in sig_upper:
        return "AVOID/SKIP"
    if "WATCH" in sig_upper:
        return "WATCH"
    return "NEUTRAL"


def _safe_float(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _print_report(report: Dict, compact: bool = False):
    try:
        from display import print_wr_report

        print_wr_report(report, compact=compact)
    except ImportError:
        if not report:
            print("tracker: No data found.")
            return
        print(
            f"\n  📊 WR: {report.get('win_rate_pct', 0):.1f}%"
            f"  W:{report.get('wins', 0)} L:{report.get('losses', 0)}"
            f"  P&L:{report.get('total_pnl', 0):+.2f}\n"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bot Pick Tracker")
    parser.add_argument(
        "--report",
        "-r",
        nargs="?",
        const=1,
        type=int,
        metavar="WEEKS_BACK",
        help="Show weekly report. Example: --report 2 (two weeks ago)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Tampilkan laporan semua data (tidak dibatasi minggu)",
    )
    parser.add_argument(
        "--update",
        nargs=2,
        metavar=("MATCH", "RESULT"),
        help="Update result manual. Contoh: --update 'Arsenal vs Chelsea' WIN",
    )
    parser.add_argument(
        "--pnl",
        type=float,
        default=None,
        help="Nominal profit/loss untuk --update",
    )
    parser.add_argument(
        "--auto-results",
        action="store_true",
        help="Scrape hasil otomatis dari football-data.org",
    )
    args = parser.parse_args()

    if args.update:
        match_q, result_q = args.update
        n = update_result(match_q, result_q, profit_loss=args.pnl)
        print(f"✅ {n} baris diupdate.")

    elif args.auto_results:
        n = asyncio.run(auto_update_results())
        print(f"✅ {n} pick diupdate otomatis.")

    elif args.all:
        weekly_report(weeks_back=0)

    else:
        weeks = args.report if args.report is not None else 1
        weekly_report(weeks_back=weeks)
