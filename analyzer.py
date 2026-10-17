import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from config import (
    ANALYSIS_CONFIG,
    BOOKMAKER_TRUST,
    get_league_min_edge,
    get_league_multiplier,
)

from probability_engine import ProbabilityEngine, remove_vig_power
from calibration import CalibrationLayer
from mispricing_detector import MispricingDetector
from risk_engine import RiskEngine

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

    # Pricing Engine fields
    raw_prob: float = 0.0
    calibrated_prob: float = 0.0
    implied_prob: float = 0.0
    model_confidence: float = 0.0
    calibration_confidence: float = 0.0
    market_agreement: str = "NEUTRAL"
    data_quality_score: float = 0.0

    ai_prob: float = 0.0
    ai_signal: str = "NEUTRAL"

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
    top_bet_book: str = ""

    # pricing metrics
    cqs: float = 0.0
    cqs_grade: str = ""

    @property
    def match_label(self) -> str:
        return f"{self.home_team} vs {self.away_team}"

    @property
    def commence_dt(self) -> datetime:
        try:
            return datetime.fromisoformat(self.commence_time.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)

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
    """Helper preserving Shin's de-vig algorithm signature."""
    return remove_vig_power(implied_probs)


def expected_value_pct(
    fair_prob: float, decimal_odds: float, is_lay: bool = False
) -> float:
    fair_prob = max(0.001, min(0.999, fair_prob))
    if is_lay:
        liability = decimal_odds - 1.0
        if liability <= 0:
            return 0.0
        return ((1.0 - fair_prob * decimal_odds) / liability) * 100
    return (fair_prob * decimal_odds - 1) * 100


def format_outcome_name(outcome: Dict, market_key: str) -> str:
    name = outcome.get("name", "")
    point = outcome.get("point")
    if point is None:
        return name
    if market_key == "spreads":
        sign = "+" if point > 0 else ""
        return f"{name} {sign}{point}"
    return f"{name} {point}"


def market_sharpness(overround: float, n_books: int) -> float:
    vig_score = max(0.0, 1.0 - (overround - 1.0) * 10.0)
    books_score = min(1.0, n_books / 8.0)
    return round((vig_score * 0.6 + books_score * 0.4), 3)


def steam_detection(all_odds: List[float]) -> str:
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
    ev_scale = 1.0 - math.exp(-ev_pct / 8.0)
    return round(min(data_conf, ev_scale * data_conf + ev_scale * 0.3), 3)


def time_confidence_factor(hours_until: float) -> float:
    if hours_until < 0.5:
        return 0.60
    if hours_until < 1.0:
        return 0.85
    if hours_until <= 6.0:
        return 1.00
    if hours_until <= 12.0:
        return 0.90
    return 0.80


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
        risk_comp = max(0.0, (100.0 - risk_score) / 100.0 * 100.0)
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


class BettingAnalyzer:

    def __init__(self):
        self.prob_engine = ProbabilityEngine()
        self.calibration_layer = CalibrationLayer()
        self.detector = MispricingDetector(
            min_edge_pct=ANALYSIS_CONFIG.min_edge_pct,
            min_ev_pct=getattr(ANALYSIS_CONFIG, "min_ev_pct", 2.0)
        )
        self.risk_engine = RiskEngine()

    def analyze_outrights(
        self,
        progression_probs: Dict[str, Dict[str, float]],
        outright_odds: Dict[str, float],
        market_type: str = "winner"
    ) -> List[Dict[str, Any]]:
        """
        Analyze outright odds against Monte Carlo progression probabilities.
        Returns a list of value outrights ranked by EV.
        """
        value_outrights = []
        for team, probs in progression_probs.items():
            prob_key = "champion_prob" if market_type == "winner" else "qualify_prob"
            true_prob = probs.get(prob_key, 0.0)
            odds = outright_odds.get(team, 0.0)
            
            if odds <= 1.0 or true_prob <= 0.0:
                continue
                
            calibrated_p = self.calibration_layer.calibrate(true_prob, method="platt")
            has_val, implied_p, edge_pct, ev_pct = self.detector.check_outcome(odds, calibrated_p)
            if has_val:
                value_outrights.append({
                    "team": team,
                    "market": "Outright Winner" if market_type == "winner" else "Group Qualification",
                    "odds": odds,
                    "true_prob": calibrated_p,
                    "raw_prob": true_prob,
                    "implied_prob": implied_p,
                    "edge_pct": edge_pct,
                    "ev_pct": ev_pct
                })
        
        value_outrights.sort(key=lambda x: x["ev_pct"], reverse=True)
        return value_outrights

    def calculate_data_quality_score(
        self, num_books: int, has_xg: bool, hours_until: float
    ) -> float:
        # Bookmaker count (up to 40 pts)
        books_pts = min(8, num_books) * 5.0
        # xG Availability (30 pts)
        xg_pts = 30.0 if has_xg else 0.0
        # Data freshness (30 pts)
        if hours_until < 0:
            freshness_pts = 5.0  # Live match, extremely volatile
        elif hours_until < 0.5:
            freshness_pts = 10.0  # Extremely close
        elif hours_until <= 12.0:
            freshness_pts = 30.0  # Sweet spot
        else:
            freshness_pts = 15.0  # Far away
        return books_pts + xg_pts + freshness_pts

    def calculate_model_confidence(self, probs: Dict[str, float]) -> float:
        # Simple agreement metric: higher peak probability in ensemble suggests stronger model agreement
        if not probs:
            return 0.5
        max_prob = max(probs.values())
        return max_prob

    def analyze_event(
        self,
        event: Dict,
        completed_matches: Optional[List[Dict[str, Any]]] = None,
        xg_context: Optional[dict] = None
    ) -> Optional[MatchAnalysis]:
        bookmakers = event.get("bookmakers", [])
        if xg_context is None:
            xg_context = event.get("xg_context")
            
        if len(bookmakers) < ANALYSIS_CONFIG.min_bookmakers:
            return None

        match = MatchAnalysis(
            event_id=event.get("id", ""),
            sport_key=event.get("sport_key", ""),
            sport_name=event.get("sport_name", event.get("sport_title", "")),
            home_team=event.get("home_team", ""),
            away_team=event.get("away_team", ""),
            commence_time=event.get("commence_time", ""),
            hours_until=event.get("hours_until", 99.0),
            num_bookmakers=len(bookmakers),
        )

        market_keys = set()
        for book in bookmakers:
            for mkt in book.get("markets", []):
                market_keys.add(mkt["key"])

        for mkt_key in market_keys:
            analysis = self._analyze_market(
                mkt_key, event, completed_matches, xg_context
            )
            if analysis:
                match.markets.append(analysis)

        self._find_top_bet(match)
        return match

    def _analyze_market(
        self,
        market_key: str,
        event: Dict,
        completed_matches: Optional[List[Dict[str, Any]]] = None,
        xg_context: Optional[dict] = None
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

        outcome_odds_map = {}
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
                    min_o = ANALYSIS_CONFIG.min_odds
                    max_o = ANALYSIS_CONFIG.max_odds
                    if price < min_o or price > max_o:
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

        # Parse point if market is totals or spreads
        market_point = None
        for o in outcome_list:
            parts = o.name.split()
            if parts:
                try:
                    market_point = float(parts[-1])
                    break
                except ValueError:
                    pass

        # Build raw ensemble probabilities
        outcomes_odds_dicts = [{"name": o.name, "price": o.best_odds} for o in outcome_list]
        ensemble_probs = self.prob_engine.build_ensemble(
            home_team=home_team,
            away_team=away_team,
            sport_key=event.get("sport_key", ""),
            outcomes_odds=outcomes_odds_dicts,
            xg_context=xg_context,
            completed_matches=completed_matches,
            market_key=market_key,
            point=market_point
        )

        if not ensemble_probs:
            return None

        # Get de-vigged market probability for comparison/fallback
        market_probs = {}
        for o in outcome_list:
            # Simple de-vigged market prob
            implied_p = 1.0 / o.best_odds if o.best_odds > 0 else 0.0
            market_probs[o.name] = implied_p
        m_total = sum(market_probs.values())
        if m_total > 0:
            market_probs = {k: v / m_total for k, v in market_probs.items()}

        best_ev, best_out = -999.0, None
        best_odds_val, best_book_val = 0.0, ""
        best_raw_prob, best_calibrated_prob = 0.0, 0.0
        best_implied_prob = 0.0
        best_steam = "STABLE"

        for o in outcome_list:
            raw_p = ensemble_probs.get(o.name)
            if raw_p is None:
                raw_p = market_probs.get(o.name, 1.0 / len(outcome_list))

            # Calibrate raw ensemble probability
            calibrated_p = self.calibration_layer.calibrate(raw_p, method="platt")

            # Check value & calculate EV/edge using MispricingDetector
            has_val, implied_p, edge_pct, ev_pct = self.detector.check_outcome(
                o.best_odds, calibrated_p
            )

            # Avoid extreme statistical outliers
            min_p = ANALYSIS_CONFIG.min_implied_prob
            max_p = ANALYSIS_CONFIG.max_implied_prob
            min_o = ANALYSIS_CONFIG.min_odds
            max_o = ANALYSIS_CONFIG.max_odds
            if calibrated_p < min_p or calibrated_p > max_p:
                continue
            if o.best_odds > max_o or o.best_odds < min_o:
                continue

            if ev_pct > best_ev:
                best_ev = ev_pct
                best_out = o.name
                best_odds_val = o.best_odds
                best_book_val = o.bookmaker
                best_raw_prob = raw_p
                best_calibrated_prob = calibrated_p
                best_implied_prob = implied_p
                best_steam = steam_detection(o.all_odds)

        if best_out is None:
            return None

        # Data quality
        has_xg = xg_context is not None
        dq_score = self.calculate_data_quality_score(
            num_books=n_books_market,
            has_xg=has_xg,
            hours_until=event.get("hours_until", 6.0)
        )

        # Model and calibration confidence
        model_conf = self.calculate_model_confidence(ensemble_probs)
        cal_conf = 1.0 if self.calibration_layer.is_fitted else 0.5

        # Market agreement
        m_prob = market_probs.get(best_out, 0.0)
        diff = abs(best_calibrated_prob - m_prob)
        if diff < 0.03:
            market_agreement = "STRONG"
        elif diff < 0.07:
            market_agreement = "MODERATE"
        else:
            market_agreement = "WEAK"

        sport_key = event.get("sport_key", "")
        league_mult = get_league_multiplier(sport_key)

        # Calculate initial Risk Score before AI
        init_risk = self.risk_engine.calculate_risk_score(
            sport_key=sport_key,
            league_multiplier=league_mult,
            num_books=n_books_market,
            hours_until=event.get("hours_until", 6.0),
            steam_flag=best_steam,
            ai_risk_flags_count=0
        )

        hours_until = event.get("hours_until", 6.0)
        time_factor = time_confidence_factor(hours_until)
        
        # Rating labels
        rating = "—"
        if best_ev >= 8.0:
            rating = "★★★ STRONG VALUE"
        elif best_ev >= 5.0:
            rating = "★★ GOOD VALUE"
        elif best_ev >= self.detector.min_edge_pct:
            rating = "★ WEAK VALUE"

        # xG indicators
        xg_signal = "XG_UNAVAILABLE"
        xg_blended = None
        if xg_context:
            xg_implied = None
            if best_out == home_team:
                xg_implied = xg_context.get("xg_implied_home_prob")
            elif best_out == away_team:
                xg_implied = xg_context.get("xg_implied_away_prob")
            elif best_out.lower() == "draw":
                xg_implied = xg_context.get("xg_implied_draw_prob")
            
            if xg_implied is not None:
                x_diff = abs(xg_implied - best_calibrated_prob)
                if x_diff < 0.03:
                    xg_signal = "XG_CONFIRM"
                elif x_diff <= 0.08:
                    xg_signal = "XG_NEUTRAL"
                else:
                    xg_signal = "XG_CONFLICT"
                xg_blended = (best_calibrated_prob * 0.7) + (xg_implied * 0.3)

        home_avg, away_avg = None, None
        if xg_context:
            h_stats = xg_context.get("home_stats")
            a_stats = xg_context.get("away_stats")
            if h_stats:
                home_avg = h_stats.get("avg_xg_for")
            if a_stats:
                away_avg = a_stats.get("avg_xg_for")

        edge_conf = edge_confidence(best_ev, sharp) * time_factor

        return MarketAnalysis(
            market_key=market_key,
            market_label=market_label,
            outcomes=outcome_list,
            overround=avg_overround,
            fair_probs=ensemble_probs,
            best_outcome=best_out,
            best_odds=best_odds_val,
            best_book=best_book_val,
            best_ev_pct=round(best_ev, 2),
            fair_prob=round(best_raw_prob, 4), # base ensemble
            
            # Pricing Engine
            raw_prob=round(best_raw_prob, 4),
            calibrated_prob=round(best_calibrated_prob, 4),
            implied_prob=round(best_implied_prob, 4),
            model_confidence=round(model_conf, 2),
            calibration_confidence=round(cal_conf, 2),
            market_agreement=market_agreement,
            data_quality_score=round(dq_score, 1),
            
            ai_prob=round(best_calibrated_prob, 4), # starts as calibrated prob, adjusted by LLM later
            ai_signal="NEUTRAL",
            league_multiplier=league_mult,
            rating=rating,
            sharpness_score=sharp,
            steam_flag=best_steam,
            confidence_score=sharp,
            data_confidence=sharp,
            edge_confidence=edge_conf,
            num_books_market=n_books_market,
            risk_score=init_risk,
            xg_signal=xg_signal,
            xg_blended_prob=xg_blended,
            xg_home_avg=home_avg,
            xg_away_avg=away_avg,
        )

    def _find_top_bet(self, match: MatchAnalysis):
        league_min_edge = get_league_min_edge(match.sport_key)
        effective_min_edge = max(ANALYSIS_CONFIG.min_edge_pct, league_min_edge)

        best = None
        for mkt in match.markets:
            if mkt.best_ev_pct > (best.best_ev_pct if best else -999.0):
                best = mkt

        if best and best.best_ev_pct >= effective_min_edge:
            match.top_bet_market = best.market_label
            match.top_bet_outcome = best.best_outcome
            match.top_bet_odds = best.best_odds
            match.top_bet_ev = best.best_ev_pct
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

    def analyze_all(
        self,
        events: List[Dict],
        completed_matches_by_sport: Optional[Dict[str, List[Dict]]] = None
    ) -> List[MatchAnalysis]:
        results = []
        for event in events:
            try:
                sport_key = event.get("sport_key", "")
                completed = completed_matches_by_sport.get(sport_key) if completed_matches_by_sport else None
                analysis = self.analyze_event(event, completed)
                if analysis:
                    results.append(analysis)
            except Exception as e:
                logger.warning("Error analyzing event %s: %s", event.get("id"), e, exc_info=True)

        value_all = sorted(
            [r for r in results if r.has_value],
            key=lambda x: (-x.cqs, -x.top_bet_ev),
        )

        max_per_league = ANALYSIS_CONFIG.max_picks_per_league
        max_total = ANALYSIS_CONFIG.max_picks_per_session
        league_counts = {}
        selected = []

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
