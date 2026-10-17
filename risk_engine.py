import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class RiskEngine:
    def __init__(self):
        pass

    def calculate_risk_score(
        self,
        sport_key: str,
        league_multiplier: float,
        num_books: int,
        hours_until: float,
        steam_flag: str = "STABLE",
        ai_risk_flags_count: int = 0
    ) -> float:
        """
        Evaluate match characteristics and compute a Risk Score (0-100).
        - Volatility (30%): Lower league quality multiplier = higher volatility.
        - Liquidity (25%): Fewer bookmakers sampled = higher risk.
        - Data Freshness / Time Window (25%): Matches too far off or extremely close to kickoff have higher uncertainty.
        - Injury & AI Flags (20%): Risks identified by contextual news.
        """
        # 1. League Volatility (30%)
        # league_multiplier is in range [0.4, 1.0]. Map 1.0 -> 10 points, 0.4 -> 100 points
        volatility_subscore = (1.0 - (league_multiplier - 0.40) / 0.60) * 90.0 + 10.0
        volatility_subscore = max(10.0, min(100.0, volatility_subscore))

        # 2. Market Liquidity (25%)
        # Standard: 8 bookmakers is full liquidity (0 risk). 1 bookmaker is max risk (100)
        liquidity_subscore = max(0.0, min(100.0, (8 - num_books) * 14.3))

        # 3. Data Freshness / Time (25%)
        # Kickoff window: sweet spot is 1-6 hours (risk 15). Live (<0h) or close kickoff (<0.5h) is risky (80). Far off (>24h) is risky (50).
        if hours_until < 0.0:
            time_subscore = 90.0  # Live match
        elif hours_until < 0.5:
            time_subscore = 75.0  # Immediate kickoff, lineups locking
        elif hours_until <= 6.0:
            time_subscore = 15.0  # Perfect window
        elif hours_until <= 24.0:
            time_subscore = 35.0  # Moderately far
        else:
            time_subscore = 60.0  # Very far, odds likely to drift

        # Steam movement adds risk
        if steam_flag == "STEAM":
            time_subscore = min(100.0, time_subscore + 20.0)
        elif steam_flag == "HIGH_SPREAD":
            time_subscore = min(100.0, time_subscore + 10.0)

        # 4. Injury / AI Context (20%)
        # 0 flags = 20 risk, each flag adds 25 risk
        context_subscore = min(100.0, 20.0 + ai_risk_flags_count * 25.0)

        # Weighted calculation
        risk_score = (
            volatility_subscore * 0.30 +
            liquidity_subscore * 0.25 +
            time_subscore * 0.25 +
            context_subscore * 0.20
        )

        return round(max(0.0, min(100.0, risk_score)), 1)
