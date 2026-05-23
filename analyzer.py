
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from config import (
    ANALYSIS_CONFIG,
    BOOKMAKER_TRUST,
    get_league_min_edge,
    get_league_multiplier,
)

logger = logging.getLogger(__name__)


@dataclass
class OutcomeOdds:
    name: str
    best_odds: float
    avg_odds: float
    bookmaker: str
    all_odds: List[float] = field(default_factory=list)

    @property
    def best_implied_prob(self) -> float:
        return 1 / self.best_odds if self.best_odds > 0 else 0

    @property
    def avg_implied_prob(self) -> float:
        return 1 / self.avg_odds if self.avg_odds > 0 else 0


@dataclass
class MarketAnalysis:
    market_key: str
    market_label: str
    outcomes: List[OutcomeOdds]

    overround: float = 0.0
    fair_probs: Dict[str, float] = field(default_factory=dict)

    best_outcome: Optional[str] = None
    best_odds: float = 0.0
    best_book: str = ""
    best_ev_pct: float = 0.0
    fair_prob: float = 0.0

    ai_prob: float = 0.0
    ai_signal: str = "NEUTRAL"

    kelly_pct: float = 0.0
    kelly_stake: float = 0.0

    rating: str = "—"

    sharpness_score: float = 0.0
    steam_flag: str = "STABLE"
    confidence_score: float = 0.0
    data_confidence: float = 0.0
    edge_confidence: float = 0.0
    num_books_market: int = 0

    risk_score: float = 0.0
    final_score: float = 0.0

    league_multiplier: float = 1.0

    xg_signal: str = "XG_UNAVAILABLE"
    xg_blended_prob: Optional[float] = None
    xg_home_avg: Optional[float] = None
    xg_away_avg: Optional[float] = None

    cqs: float = 0.0
    cqs_grade: str = ""


@dataclass
class MatchAnalysis:
    event_id: str
    sport_key: str
    sport_name: str
    home_team: str
    away_team: str
    commence_time: str
    hours_until: float
    num_bookmakers: int

    markets: List[MarketAnalysis] = field(default_factory=list)

    top_bet_market: Optional[str] = None
    top_bet_outcome: Optional[str] = None
    top_bet_odds: float = 0.0
    top_bet_ev: float = 0.0
    top_bet_kelly: float = 0.0
    top_bet_book: str = ""

    @property
    def match_label(self) -> str:
        return f"{self.home_team} vs {self.away_team}"

    @property
    def commence_dt(self) -> datetime:
        try:
            return datetime.fromisoformat(self.commence_time.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)

    cqs: float = 0.0
    cqs_grade: str = ""

    @property
    def has_value(self) -> bool:
        if self.top_bet_ev < ANALYSIS_CONFIG.min_edge_pct:
            return False

        if self.cqs > 0 and self.cqs < ANALYSIS_CONFIG.cqs_min_display:
            return False
        return True


def decimal_to_prob(odds: float) -> float:
    if odds <= 1.0:
        return 0.0
    return 1.0 / odds


def remove_vig(implied_probs: List[float]) -> List[float]:
    total = sum(implied_probs)
    if total <= 1.0:
        return implied_probs

    low, high = 1.0, 5.0
    k = 1.0
    for _ in range(30):
        k = (low + high) / 2
        s = sum(p**k for p in implied_probs)
        if s > 1.0:
            low = k
        else:
            high = k

    return [p**k for p in implied_probs]


def expected_value_pct(
    fair_prob: float, decimal_odds: float, is_lay: bool = False
) -> float:

    fair_prob = max(0.001, min(0.999, fair_prob))

    if is_lay:
        return (1 - (fair_prob * decimal_odds)) * 100

    return (fair_prob * decimal_odds - 1) * 100


def kelly_fraction(
    fair_prob: float, decimal_odds: float, fraction: float = None, is_lay: bool = False
) -> float:

    fair_prob = max(0.001, min(0.999, fair_prob))
    fraction = fraction or ANALYSIS_CONFIG.kelly_fraction

    if is_lay:
        full_kelly = 1.0 - (fair_prob * decimal_odds)
    else:
        b = decimal_odds - 1.0
        p = fair_prob
        q = 1.0 - p
        if b <= 0 or p <= 0:
            return 0.0
        full_kelly = (b * p - q) / b

    if full_kelly <= 0:
        return 0.0

    fractional = full_kelly * fraction
    return min(fractional, ANALYSIS_CONFIG.max_kelly_pct / 100.0)


def format_outcome_name(outcome: Dict, market_key: str) -> str:
    name = outcome.get("name", "")
    point = outcome.get("point")
    if point is None:
        return name
    if market_key == "spreads":
        sign = "+" if point > 0 else ""
        return f"{name} {sign}{point}"
    return f"{name} {point}"


def weighted_fair_prob(
    outcome_name: str,
    bookmakers: List[Dict],
    market_key: str,
) -> Tuple[float, int]:
    probs = []
    weights = []

    for book in bookmakers:
        trust = BOOKMAKER_TRUST.get(book.get("key", ""), 0.70)
        for market in book.get("markets", []):
            if market["key"] != market_key:
                continue
            outcomes = market.get("outcomes", [])
            mkt_probs = [decimal_to_prob(o["price"]) for o in outcomes]
            fair = remove_vig(mkt_probs)
            for i, outcome in enumerate(outcomes):
                if format_outcome_name(outcome, market_key) == outcome_name:
                    probs.append(fair[i])
                    weights.append(trust)
                    break

    if not probs:
        return 0.0, 0

    total_weight = sum(weights)
    wp = sum(p * w for p, w in zip(probs, weights)) / total_weight
    return wp, len(probs)


def market_sharpness(overround: float, n_books: int) -> float:

    vig_score = max(0.0, 1.0 - (overround - 1.0) * 10.0)

    books_score = min(1.0, n_books / 8.0)

    return round((vig_score * 0.6 + books_score * 0.4), 3)


def steam_detection(all_odds: List[float], sharp_odds: Optional[float] = None) -> str:
    if len(all_odds) < 2:
        return "STABLE"
    spread = max(all_odds) - min(all_odds)
    avg = statistics.mean(all_odds)
    cv = spread / avg if avg > 0 else 0

    if cv > 0.12:
        return "STEAM"
    elif cv > 0.07:
        return "HIGH_SPREAD"
    elif cv > 0.04:
        return "MED_SPREAD"
    return "STABLE"


def data_confidence(sharpness: float, n_books: int, overround: float) -> float:

    books_score = min(1.0, n_books / 6.0)

    sharpness_score = sharpness

    overround_score = max(0.0, min(1.0, (1.12 - overround) / 0.09))

    score = (books_score * 0.40) + (sharpness_score * 0.35) + (overround_score * 0.25)
    return round(min(1.0, max(0.0, score)), 3)


def edge_confidence(ev_pct: float, data_conf: float) -> float:
    if ev_pct <= 0:
        return 0.0

    import math

    ev_scale = 1.0 - math.exp(-ev_pct / 8.0)

    return round(min(data_conf, ev_scale * data_conf + ev_scale * 0.3), 3)


def confidence_score(ev_pct: float, sharpness: float, n_books: int) -> float:
    return data_confidence(sharpness, n_books, 1.05)


def compute_cqs(
    ev_pct: float,
    data_confidence: float,
    league_mult: float,
    risk_score: float = 0.0,
    xg_signal: str = "XG_UNAVAILABLE",
    n_books: int = 3,
    steam_flag: str = "STABLE",
) -> float:

    ev_comp = min(100.0, (1.0 - math.exp(-ev_pct / 10.0)) * 100.0)

    data_comp = min(100.0, data_confidence * 100.0)

    league_comp = max(0.0, min(100.0, (league_mult - 0.40) / 0.60 * 100.0))

    if risk_score > 0:
        risk_comp = max(0.0, (10.0 - risk_score) / 10.0 * 100.0)
    else:
        risk_comp = 50.0

    xg_map = {
        "XG_CONFIRM": 85.0,
        "XG_NEUTRAL": 50.0,
        "XG_CONFLICT": 15.0,
        "XG_UNAVAILABLE": 45.0,
    }
    xg_comp = xg_map.get(xg_signal, 45.0)

    books_comp = min(100.0, max(0.0, (n_books - 2) * 15.0))

    steam_bonus = {
        "STEAM": 10.0,
        "HIGH_SPREAD": 5.0,
        "MED_SPREAD": 0.0,
        "STABLE": 0.0,
    }.get(steam_flag, 0.0)

    cqs = (
        ev_comp * 0.25
        + data_comp * 0.20
        + league_comp * 0.20
        + risk_comp * 0.20
        + xg_comp * 0.10
        + books_comp * 0.05
        + steam_bonus
    )

    return round(min(100.0, max(0.0, cqs)), 1)


def cqs_grade(cqs: float) -> str:
    if cqs >= ANALYSIS_CONFIG.cqs_premium_threshold:
        return "PREMIUM ★★★"
    elif cqs >= ANALYSIS_CONFIG.cqs_standard_threshold:
        return "STANDARD ★★"
    elif cqs >= ANALYSIS_CONFIG.cqs_min_display:
        return "MARGINAL ★"
    else:
        return "SKIP"


class AIBettingModel:

    def analyze_outcome(
        self,
        outcome_name: str,
        best_odds: float,
        fair_prob: float,
        all_odds: List[float],
        market_overround: float,
        n_books: int = 3,
    ) -> Tuple[float, str, float]:
        fair_prob = max(0.001, min(0.999, fair_prob))
        if best_odds <= 1:
            return fair_prob, "NEUTRAL", 0.0

        if best_odds > ANALYSIS_CONFIG.max_odds:
            return fair_prob, "NEUTRAL", 0.0

        ai_prob = fair_prob

        sharp = market_sharpness(market_overround, n_books)
        steam = steam_detection(all_odds)
        d_conf = data_confidence(sharp, n_books, market_overround)

        ev = expected_value_pct(ai_prob, best_odds)

        e_conf = edge_confidence(ev, d_conf)

        if ev >= 8.0 and d_conf >= 0.75 and n_books >= 5:
            signal = "STRONG BUY ★★★"
        elif ev >= 5.0 and d_conf >= 0.65 and n_books >= 4:
            signal = "BUY ★★"
        elif ev >= ANALYSIS_CONFIG.min_edge_pct and d_conf >= 0.55:
            signal = "WATCH ★"
        elif ev <= -3.0:
            signal = "AVOID ✗"
        else:
            signal = "NEUTRAL"

        if steam == "STEAM" and ev >= 5.0 and signal == "WATCH ★" and n_books >= 6:
            signal = "BUY ★★"

        return max(0.001, min(0.999, ai_prob)), signal, d_conf


class BettingAnalyzer:

    def __init__(self):
        self.ai_model = AIBettingModel()
        self.config = ANALYSIS_CONFIG

    def incorporate_xg_signal(
        self,
        fair_prob: float,
        xg_context: Optional[dict],
        weight_bookmaker: float = 0.75,
        weight_xg: float = 0.25,
        outcome_name: str = "",
        home_team: str = "",
        away_team: str = "",
    ) -> Tuple[float, str]:
        if not xg_context:
            return fair_prob, "XG_UNAVAILABLE"

        xg_implied_prob = None
        if outcome_name and outcome_name == home_team:
            xg_implied_prob = xg_context.get("xg_implied_home_prob")
        elif outcome_name and outcome_name == away_team:
            xg_implied_prob = xg_context.get("xg_implied_away_prob")
        elif outcome_name and outcome_name.lower() == "draw":
            xg_implied_prob = xg_context.get("xg_implied_draw_prob")

        if xg_implied_prob is None:
            return fair_prob, "XG_UNAVAILABLE"

        diff = abs(xg_implied_prob - fair_prob)
        blended_prob = (fair_prob * weight_bookmaker) + (xg_implied_prob * weight_xg)

        if diff < 0.03:
            return blended_prob, "XG_CONFIRM"
        elif diff <= 0.08:
            return blended_prob, "XG_NEUTRAL"
        else:
            return blended_prob, "XG_CONFLICT"

    def analyze_event(
        self, event: Dict, xg_context: Optional[dict] = None
    ) -> Optional[MatchAnalysis]:
        bookmakers = event.get("bookmakers", [])
        if xg_context is None:
            xg_context = event.get("xg_context")
        if len(bookmakers) < self.config.min_bookmakers:
            return None

        match = MatchAnalysis(
            event_id=event.get("id", ""),
            sport_key=event.get("sport_key", ""),
            sport_name=event.get("sport_name", event.get("sport_title", "")),
            home_team=event.get("home_team", ""),
            away_team=event.get("away_team", ""),
            commence_time=event.get("commence_time", ""),
            hours_until=event.get("hours_until", 99),
            num_bookmakers=len(bookmakers),
        )

        market_keys: set = set()
        for book in bookmakers:
            for mkt in book.get("markets", []):
                market_keys.add(mkt["key"])

        for mkt_key in market_keys:
            analysis = self._analyze_market(mkt_key, event, xg_context)
            if analysis:
                match.markets.append(analysis)

        self._find_top_bet(match)
        return match

    def _analyze_market(
        self, market_key: str, event: Dict, xg_context: Optional[dict] = None
    ) -> Optional[MarketAnalysis]:
        bookmakers = event.get("bookmakers", [])
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        market_label = {
            "h2h": "Match Result (1X2)",
            "totals": "Over/Under Goals",
            "btts": "Both Teams to Score",
            "spreads": "Asian Handicap",
        }.get(market_key, market_key.upper())

        outcome_odds_map: Dict[str, List[Tuple[float, str]]] = {}
        overrounds = []
        n_books_market = 0

        for book in bookmakers:
            book_key = book.get("key", "unknown")
            for mkt in book.get("markets", []):
                if mkt["key"] != market_key:
                    continue

                outcomes = mkt.get("outcomes", [])
                implied_sum = sum(decimal_to_prob(o["price"]) for o in outcomes)
                if implied_sum > 0:
                    overrounds.append(implied_sum)
                    n_books_market += 1

                for o in outcomes:
                    name = format_outcome_name(o, market_key)
                    price = o["price"]
                    if price < self.config.min_odds or price > self.config.max_odds:
                        continue
                    if name not in outcome_odds_map:
                        outcome_odds_map[name] = []
                    outcome_odds_map[name].append((price, book_key))

        if not outcome_odds_map:
            return None

        avg_overround = statistics.mean(overrounds) if overrounds else 1.10
        sharp = market_sharpness(avg_overround, n_books_market)

        outcome_list = []
        for name, odds_books in outcome_odds_map.items():
            if not odds_books:
                continue
            odds_values = [o for o, _ in odds_books]
            best_price, best_bk = max(odds_books, key=lambda x: x[0])
            avg_price = statistics.mean(odds_values)
            outcome_list.append(
                OutcomeOdds(
                    name=name,
                    best_odds=best_price,
                    avg_odds=avg_price,
                    bookmaker=best_bk,
                    all_odds=odds_values,
                )
            )

        if not outcome_list:
            return None

        fair_probs: Dict[str, float] = {}
        for o in outcome_list:
            wp, _ = weighted_fair_prob(o.name, bookmakers, market_key)
            fair_probs[o.name] = wp

        total_fair = sum(fair_probs.values())
        if total_fair > 0:
            fair_probs = {k: v / total_fair for k, v in fair_probs.items()}

        best_ev, best_out = -999, None
        best_odds_val, best_book_val = 0, ""
        best_fair, best_ai_prob = 0, 0
        best_signal, best_kelly = "NEUTRAL", 0
        best_conf = 0.0
        best_steam = "STABLE"
        best_xg_signal = "XG_UNAVAILABLE"
        best_xg_blended = None

        for o in outcome_list:
            fp = fair_probs.get(o.name, 0)
            if fp < self.config.min_implied_prob or fp > self.config.max_implied_prob:
                continue

            if o.best_odds > self.config.max_odds or o.best_odds < self.config.min_odds:
                continue

            implied = 1.0 / o.best_odds if o.best_odds > 0 else 0
            if abs(implied - fp) > 0.40:
                logger.debug(
                    "Skipping divergent fair_prob=%.3f implied=%.3f for %s/%s",
                    fp,
                    implied,
                    market_key,
                    o.name,
                )
                continue

            ai_prob, signal, conf = self.ai_model.analyze_outcome(
                o.name, o.best_odds, fp, o.all_odds, avg_overround, n_books_market
            )

            blended_prob, xg_signal = self.incorporate_xg_signal(
                fair_prob=ai_prob,
                xg_context=xg_context,
                outcome_name=o.name,
                home_team=home_team,
                away_team=away_team,
            )

            eval_prob = blended_prob if xg_signal != "XG_UNAVAILABLE" else ai_prob

            ev = expected_value_pct(eval_prob, o.best_odds)
            if ev > self.config.max_edge_pct:
                logger.debug(
                    "Skipping suspicious edge %.2f%% for %s/%s", ev, market_key, o.name
                )
                continue

            if ev > best_ev:
                best_ev = ev
                best_out = o.name
                best_odds_val = o.best_odds
                best_book_val = o.bookmaker
                best_fair = fp
                best_ai_prob = ai_prob
                best_signal = signal
                best_kelly = kelly_fraction(eval_prob, o.best_odds)
                best_conf = conf
                best_steam = steam_detection(o.all_odds)
                best_xg_signal = xg_signal
                best_xg_blended = (
                    blended_prob if xg_signal != "XG_UNAVAILABLE" else None
                )

        if best_out is None:
            return None

        sport_key = event.get("sport_key", "")
        league_mult = get_league_multiplier(sport_key)
        adjusted_kelly = best_kelly * league_mult

        final_edge_conf = edge_confidence(best_ev, best_conf)
        if best_xg_signal == "XG_CONFLICT":
            final_edge_conf = max(0.0, final_edge_conf * 0.90)

        rating = "—"
        if best_ev >= 8.0:
            rating = "★★★ STRONG VALUE"
        elif best_ev >= 5.0:
            rating = "★★ GOOD VALUE"
        elif best_ev >= self.config.min_edge_pct:
            rating = "★ WEAK VALUE"

        home_avg, away_avg = None, None
        if xg_context and isinstance(xg_context, dict):
            h_stats = xg_context.get("home_stats")
            a_stats = xg_context.get("away_stats")
            if h_stats:
                home_avg = h_stats.get("avg_xg_for")
            if a_stats:
                away_avg = a_stats.get("avg_xg_for")

        return MarketAnalysis(
            market_key=market_key,
            market_label=market_label,
            outcomes=outcome_list,
            overround=avg_overround,
            fair_probs=fair_probs,
            best_outcome=best_out,
            best_odds=best_odds_val,
            best_book=best_book_val,
            best_ev_pct=round(best_ev, 2),
            fair_prob=round(best_fair, 4),
            ai_prob=round(best_ai_prob, 4),
            ai_signal=best_signal,
            kelly_pct=round(adjusted_kelly * 100, 2),
            kelly_stake=round(adjusted_kelly * ANALYSIS_CONFIG.bankroll, 2),
            league_multiplier=league_mult,
            rating=rating,
            sharpness_score=sharp,
            steam_flag=best_steam,
            confidence_score=best_conf,
            data_confidence=best_conf,
            edge_confidence=final_edge_conf,
            num_books_market=n_books_market,
            xg_signal=best_xg_signal,
            xg_blended_prob=best_xg_blended,
            xg_home_avg=home_avg,
            xg_away_avg=away_avg,
        )

    def _find_top_bet(self, match: MatchAnalysis):

        league_min_edge = get_league_min_edge(match.sport_key)
        effective_min_edge = max(self.config.min_edge_pct, league_min_edge)

        best = None
        for mkt in match.markets:
            if mkt.best_ev_pct > (best.best_ev_pct if best else -999):
                best = mkt
        if best and best.best_ev_pct >= effective_min_edge:
            match.top_bet_market = best.market_label
            match.top_bet_outcome = best.best_outcome
            match.top_bet_odds = best.best_odds
            match.top_bet_ev = best.best_ev_pct
            match.top_bet_kelly = best.kelly_pct
            match.top_bet_book = best.best_book

            cqs = compute_cqs(
                ev_pct=best.best_ev_pct,
                data_confidence=best.data_confidence,
                league_mult=best.league_multiplier,
                risk_score=best.risk_score,
                xg_signal=best.xg_signal,
                n_books=best.num_books_market,
                steam_flag=best.steam_flag,
            )
            grade = cqs_grade(cqs)
            best.cqs = cqs
            best.cqs_grade = grade
            match.cqs = cqs
            match.cqs_grade = grade

    def analyze_all(self, events: List[Dict]) -> List[MatchAnalysis]:
        results = []
        for event in events:
            try:
                analysis = self.analyze_event(event)
                if analysis:
                    results.append(analysis)
            except Exception as e:
                logger.warning("Error analyzing event %s: %s", event.get("id"), e)

        value_all = sorted(
            [r for r in results if r.has_value],
            key=lambda x: (-x.cqs, -x.top_bet_ev),
        )

        max_per_league = self.config.max_picks_per_league
        max_total = self.config.max_picks_per_session
        league_counts: Dict[str, int] = {}
        selected: List[MatchAnalysis] = []

        for m in value_all:
            if len(selected) >= max_total:
                break
            lg = m.sport_name
            if league_counts.get(lg, 0) >= max_per_league:
                logger.debug(
                    "Smart selection: skipping %s (league cap %d reached for %s)",
                    m.match_label,
                    max_per_league,
                    lg,
                )
                continue
            selected.append(m)
            league_counts[lg] = league_counts.get(lg, 0) + 1

        other_value = [m for m in value_all if m not in selected]
        non_value = sorted(
            [r for r in results if not r.has_value], key=lambda x: x.hours_until
        )

        return selected + other_value + non_value
