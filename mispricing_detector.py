import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class MispricingDetector:
    def __init__(self, min_edge_pct: float = 3.0, min_ev_pct: float = 2.0):
        self.min_edge_pct = min_edge_pct
        self.min_ev_pct = min_ev_pct

    def check_outcome(self, best_odds: float, calibrated_prob: float) -> Tuple[bool, float, float, float]:
        """
        Evaluate if an outcome is mispriced.
        Returns: (has_value, implied_prob, edge_pct, ev_pct)
        """
        if best_odds <= 1.0:
            return False, 0.0, 0.0, 0.0

        implied_prob = 1.0 / best_odds
        edge = calibrated_prob - implied_prob
        edge_pct = edge * 100
        ev_pct = (best_odds * calibrated_prob - 1) * 100

        # High extreme edge values are often data errors, let's bound them in range [min_edge, 100]
        has_value = (edge_pct >= self.min_edge_pct) or (ev_pct >= self.min_ev_pct)
        
        # Prevent extreme longshot noise with low absolute true probability
        if best_odds >= 15.0 and calibrated_prob < 0.04:
            has_value = False
        elif best_odds >= 8.0 and calibrated_prob < 0.06:
            has_value = False
        elif calibrated_prob < 0.02:
            has_value = False
            
        # Avoid extreme data errors (e.g., edge over 60%)
        if edge_pct > 60.0 or ev_pct > 150.0:
            has_value = False

        return has_value, implied_prob, round(edge_pct, 2), round(ev_pct, 2)

    def filter_and_rank(self, match_analyses: List[Any]) -> List[Any]:
        """
        Given a list of match analyses, filters for those with a valid top bet
        that exceeds the mispricing thresholds, and ranks them by EV.
        """
        valid_picks = []
        for m in match_analyses:
            if getattr(m, "top_bet_market", None) and getattr(m, "top_bet_ev", -99.0) >= self.min_ev_pct:
                valid_picks.append(m)
                
        # Rank primarily by EV % in descending order
        valid_picks.sort(key=lambda x: getattr(x, "top_bet_ev", 0.0), reverse=True)
        return valid_picks
