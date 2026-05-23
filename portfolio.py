
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from analyzer import MarketAnalysis

logger = logging.getLogger(__name__)


HARD_SUPPRESS_THRESHOLD = 0.85
ADVISORY_THRESHOLD = 0.50

MARKET_CORRELATION: Dict[Tuple[str, str], float] = {
    ("h2h", "spreads"): 0.85,
    ("spreads", "h2h"): 0.85,
    ("totals", "btts"): 0.60,
    ("btts", "totals"): 0.60,
    ("h2h", "totals"): 0.40,
    ("totals", "h2h"): 0.40,
    ("h2h", "btts"): 0.35,
    ("btts", "h2h"): 0.35,
    ("spreads", "totals"): 0.45,
    ("totals", "spreads"): 0.45,
    ("spreads", "btts"): 0.40,
    ("btts", "spreads"): 0.40,
}


@dataclass
class MarketSuppression:

    match_label: str
    market_key: str
    market_label: str
    ev_pct: float
    correlated_with: str
    correlation: float


@dataclass
class CorrelationAdvisory:

    match_label: str
    market_key_a: str
    market_label_a: str
    ev_pct_a: float
    market_key_b: str
    market_label_b: str
    ev_pct_b: float
    correlation: float
    combined_stake_rp: float
    recommended_max_rp: float


@dataclass
class RiskAdvisory:

    advisory_type: str
    match_label: str
    message: str
    severity: str
    current_rp: float
    limit_rp: float


@dataclass
class SessionExposureSnapshot:

    total_stake_rp: float
    guidance_limit_rp: float

    @property
    def pct_used(self) -> float:
        if self.guidance_limit_rp <= 0:
            return 0.0
        return min(100.0, self.total_stake_rp / self.guidance_limit_rp * 100)

    @property
    def status(self) -> str:
        if self.pct_used >= 100:
            return "EXCEEDED"
        if self.pct_used >= 75:
            return "WARNING"
        return "SAFE"

    def progress_bar(self, width: int = 20) -> str:
        filled = min(width, round(self.pct_used / 100 * width))
        return "█" * filled + "░" * (width - filled)


@dataclass
class PortfolioReport:

    suppressions: List[MarketSuppression] = field(default_factory=list)
    correlation_advisories: List[CorrelationAdvisory] = field(default_factory=list)
    risk_advisories: List[RiskAdvisory] = field(default_factory=list)
    session: Optional[SessionExposureSnapshot] = None
    recommended_max_per_match_rp: float = 0.0

    @property
    def n_suppressed(self) -> int:
        return len(self.suppressions)

    def is_suppressed(self, match_label: str, market_key: str) -> bool:
        return any(
            s.match_label == match_label and s.market_key == market_key
            for s in self.suppressions
        )

    def corr_advisories_for(self, match_label: str) -> List[CorrelationAdvisory]:
        return [a for a in self.correlation_advisories if a.match_label == match_label]

    def risk_advisories_for(self, match_label: str) -> List[RiskAdvisory]:
        return [a for a in self.risk_advisories if a.match_label == match_label]

    def session_advisories(self) -> List[RiskAdvisory]:
        return [
            a for a in self.risk_advisories if a.advisory_type == "SESSION_GUIDANCE"
        ]


class PortfolioManager:

    def __init__(
        self,
        bankroll: float,
        max_per_match_pct: float = 3.0,
        session_guidance_rp: Optional[float] = None,
    ):
        self.bankroll = bankroll
        self.max_per_match_rp = bankroll * max_per_match_pct / 100.0

        self.session_guidance = (
            session_guidance_rp
            if (session_guidance_rp and session_guidance_rp > bankroll * 0.01)
            else bankroll * 0.15
        )

    def apply(self, matches: List) -> PortfolioReport:
        report = PortfolioReport(
            recommended_max_per_match_rp=self.max_per_match_rp,
        )
        session_accum = 0.0

        for match in matches:
            match._portfolio_suppressed = []
            match._portfolio_corr_advisories = []
            match._portfolio_advisories = []
            match._portfolio_report = report

            if not match.has_value:
                continue

            value_markets = [
                m
                for m in match.markets
                if m.best_ev_pct >= 2.0 and m.best_outcome is not None
            ]
            if not value_markets:
                continue

            suppressed, corr_advisories = self._handle_correlation(
                value_markets, match.match_label, report
            )
            match._portfolio_suppressed = suppressed
            match._portfolio_corr_advisories = corr_advisories

            surviving = [m for m in value_markets if m not in suppressed]
            match_advisories = self._check_match_exposure(surviving, match, report)

            match_stake = sum((m.kelly_pct / 100.0) * self.bankroll for m in surviving)
            session_accum += match_stake
            session_adv = self._check_session_guidance(session_accum, match, report)
            if session_adv:
                match_advisories.append(session_adv)

            match._portfolio_advisories = match_advisories

        report.session = SessionExposureSnapshot(
            total_stake_rp=session_accum,
            guidance_limit_rp=self.session_guidance,
        )

        logger.info(
            "Portfolio advisory: %d hard suppressed, %d corr advisories, "
            "%d risk advisories | session $%.0f / guidance $%.0f",
            report.n_suppressed,
            len(report.correlation_advisories),
            len(report.risk_advisories),
            session_accum,
            self.session_guidance,
        )
        return report

    def _handle_correlation(
        self,
        markets: List,
        match_label: str,
        report: PortfolioReport,
    ) -> Tuple[List, List[CorrelationAdvisory]]:
        sorted_mkts = sorted(markets, key=lambda m: -m.best_ev_pct)
        kept: List = []
        suppressed_set: set = set()
        corr_advisories: List[CorrelationAdvisory] = []

        warned_pairs: set = set()

        for i, candidate in enumerate(sorted_mkts):
            if candidate.market_key in suppressed_set:
                continue
            for kept_mkt in kept:
                pair_key = (candidate.market_key, kept_mkt.market_key)
                corr = MARKET_CORRELATION.get(pair_key, 0.0)
                if corr <= ADVISORY_THRESHOLD:
                    continue

                norm_pair = tuple(sorted([candidate.market_key, kept_mkt.market_key]))

                if corr > HARD_SUPPRESS_THRESHOLD:
                    suppressed_set.add(candidate.market_key)
                    report.suppressions.append(
                        MarketSuppression(
                            match_label=match_label,
                            market_key=candidate.market_key,
                            market_label=candidate.market_label,
                            ev_pct=candidate.best_ev_pct,
                            correlated_with=kept_mkt.market_key,
                            correlation=corr,
                        )
                    )
                    logger.debug(
                        "HARD SUPPRESS %s/%s (corr=%.2f >0.85 with %s)",
                        match_label,
                        candidate.market_key,
                        corr,
                        kept_mkt.market_key,
                    )
                    break
                else:
                    if norm_pair not in warned_pairs:
                        warned_pairs.add(norm_pair)
                        combined_stake = (
                            (candidate.kelly_pct + kept_mkt.kelly_pct)
                            / 100.0
                            * self.bankroll
                        )
                        adv = CorrelationAdvisory(
                            match_label=match_label,
                            market_key_a=kept_mkt.market_key,
                            market_label_a=kept_mkt.market_label,
                            ev_pct_a=kept_mkt.best_ev_pct,
                            market_key_b=candidate.market_key,
                            market_label_b=candidate.market_label,
                            ev_pct_b=candidate.best_ev_pct,
                            correlation=corr,
                            combined_stake_rp=combined_stake,
                            recommended_max_rp=self.max_per_match_rp,
                        )
                        corr_advisories.append(adv)
                        report.correlation_advisories.append(adv)
                        logger.debug(
                            "CORR ADVISORY %s: %s ↔ %s (corr=%.2f)",
                            match_label,
                            kept_mkt.market_key,
                            candidate.market_key,
                            corr,
                        )

            if candidate.market_key not in suppressed_set:
                kept.append(candidate)

        suppressed_list = [m for m in markets if m.market_key in suppressed_set]
        return suppressed_list, corr_advisories

    def _check_match_exposure(
        self,
        surviving: List,
        match,
        report: PortfolioReport,
    ) -> List[RiskAdvisory]:
        advisories: List[RiskAdvisory] = []
        total_stake = sum((m.kelly_pct / 100.0) * self.bankroll for m in surviving)
        if total_stake > self.max_per_match_rp:
            excess_pct = (total_stake / self.max_per_match_rp - 1) * 100
            severity = "HIGH" if excess_pct > 50 else "MEDIUM"
            adv = RiskAdvisory(
                advisory_type="MATCH_OVEREXPOSURE",
                match_label=match.match_label,
                message=(
                    f"Total stake {match.match_label}: "
                    f"${total_stake:,.0f} exceeds the "
                    f"${self.max_per_match_rp:,.0f} per-match limit "
                    f"(+{excess_pct:.0f}%)"
                ),
                severity=severity,
                current_rp=total_stake,
                limit_rp=self.max_per_match_rp,
            )
            advisories.append(adv)
            report.risk_advisories.append(adv)
        return advisories

    def _check_session_guidance(
        self,
        session_accum: float,
        match,
        report: PortfolioReport,
    ) -> Optional[RiskAdvisory]:
        already = any(
            a.advisory_type == "SESSION_GUIDANCE" for a in report.risk_advisories
        )
        if not already and session_accum > self.session_guidance:
            adv = RiskAdvisory(
                advisory_type="SESSION_GUIDANCE",
                match_label=match.match_label,
                message=(
                    f"Total session stake ${session_accum:,.0f} exceeds "
                    f"the ${self.session_guidance:,.0f} guidance limit. "
                    f"Consider stopping for today."
                ),
                severity="HIGH",
                current_rp=session_accum,
                limit_rp=self.session_guidance,
            )
            report.risk_advisories.append(adv)
            return adv
        return None
