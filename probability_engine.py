import logging
import math
import statistics
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Default weights for the ensemble system (fully configurable)
DEFAULT_ENSEMBLE_WEIGHTS = {
    "poisson": 0.33,
    "elo": 0.28,
    "form": 0.22,
    "market": 0.17,
}

def poisson_probability(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    try:
        return (lam**k * math.exp(-lam)) / math.factorial(k)
    except (ValueError, OverflowError):
        return 0.0

def remove_vig_power(implied_probs: List[float]) -> List[float]:
    """Remove vig using the Implied Probability Power Method.
    Solves for k such that sum(p_i ^ (1/k)) = 1.
    Falls back to multiplicative if it fails to converge.
    """
    total = sum(implied_probs)
    if total <= 1.0 or len(implied_probs) < 2:
        return implied_probs
        
    def sum_power(k_val):
        return sum(math.pow(p, 1.0/k_val) for p in implied_probs if p > 0)
        
    low = 0.01
    high = 1.0
    
    for _ in range(50):
        mid = (low + high) / 2.0
        s = sum_power(mid)
        if abs(s - 1.0) < 1e-6:
            break
        elif s > 1.0:
            high = mid
        else:
            low = mid
            
    k = (low + high) / 2.0
    fair = [math.pow(p, 1.0/k) if p > 0 else 0.0 for p in implied_probs]
    fair_total = sum(fair)
    if fair_total > 0:
        return [f / fair_total for f in fair]
    return [p / total for p in implied_probs]

class EloTracker:
    """Manages ELO ratings for a single league/sport key."""
    def __init__(self, k_factor: float = 30.0, home_advantage: float = 50.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: Dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.setdefault(team, 1500.0)

    def update(self, home_team: str, away_team: str, home_goals: int, away_goals: int):
        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)

        # Expected outcome for Home
        diff = r_home + self.home_advantage - r_away
        expected_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

        # Actual outcome: 1.0 = Home Win, 0.5 = Draw, 0.0 = Away Win
        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals == away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0

        shift = self.k_factor * (actual_home - expected_home)
        self.ratings[home_team] = r_home + shift
        self.ratings[away_team] = r_away - shift


class ProbabilityEngine:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_ENSEMBLE_WEIGHTS

    def get_market_probabilities(
        self,
        outcomes_odds: List[Dict[str, Any]],
        market_key: str,
        home_team: str,
        away_team: str
    ) -> Dict[str, float]:
        """Calculates de-vigged market implied probabilities."""
        implied_probs = []
        names = []
        for o in outcomes_odds:
            price = o.get("price", 0)
            implied_probs.append(1.0 / price if price > 0 else 0.0)
            names.append(o.get("name", ""))

        fair_probs = remove_vig_power(implied_probs)
        return {name: prob for name, prob in zip(names, fair_probs)}

    def calculate_poisson_probs(
        self,
        home_team: str,
        away_team: str,
        xg_context: Optional[Dict[str, Any]],
        market_key: str,
        point: Optional[float] = None
    ) -> Optional[Dict[str, float]]:
        """Calculates probabilities from xG Poisson Goal Model."""
        if not xg_context:
            return None

        h_stats = xg_context.get("home_stats")
        a_stats = xg_context.get("away_stats")
        if not h_stats or not a_stats:
            return None

        try:
            home_proj = (h_stats["avg_xg_for"] + a_stats["avg_xg_against"]) / 2
            away_proj = (a_stats["avg_xg_for"] + h_stats["avg_xg_against"]) / 2
        except KeyError:
            return None

        # Build grid (0 to 8 goals)
        grid_size = 9
        grid = [[0.0] * grid_size for _ in range(grid_size)]
        for h in range(grid_size):
            for a in range(grid_size):
                grid[h][a] = poisson_probability(home_proj, h) * poisson_probability(away_proj, a)

        # Normalize grid
        grid_sum = sum(sum(row) for row in grid)
        if grid_sum > 0:
            grid = [[val / grid_sum for val in row] for row in grid]

        if market_key == "h2h":
            home_p = sum(grid[h][a] for h in range(grid_size) for a in range(grid_size) if h > a)
            draw_p = sum(grid[h][a] for h in range(grid_size) for a in range(grid_size) if h == a)
            away_p = sum(grid[h][a] for h in range(grid_size) for a in range(grid_size) if h < a)
            return {home_team: home_p, "Draw": draw_p, away_team: away_p}

        elif market_key == "btts":
            yes_p = sum(grid[h][a] for h in range(1, grid_size) for a in range(1, grid_size))
            no_p = 1.0 - yes_p
            return {"Yes": yes_p, "No": no_p}

        elif market_key == "totals" and point is not None:
            over_p = sum(grid[h][a] for h in range(grid_size) for a in range(grid_size) if h + a > point)
            under_p = sum(grid[h][a] for h in range(grid_size) for a in range(grid_size) if h + a < point)
            push_p = sum(grid[h][a] for h in range(grid_size) for a in range(grid_size) if h + a == point)
            # Re-normalize if there is a push probability
            total_active = over_p + under_p
            if total_active > 0:
                return {"Over": over_p / total_active, "Under": under_p / total_active}
            return {"Over": 0.5, "Under": 0.5}

        elif market_key == "spreads" and point is not None:
            # Point is for home team. e.g. -1.5, +0.5
            home_win_p = sum(grid[h][a] for h in range(grid_size) for a in range(grid_size) if h - a + point > 0)
            away_win_p = sum(grid[h][a] for h in range(grid_size) for a in range(grid_size) if h - a + point < 0)
            push_p = sum(grid[h][a] for h in range(grid_size) for a in range(grid_size) if h - a + point == 0)
            total_active = home_win_p + away_win_p
            if total_active > 0:
                return {"Home": home_win_p / total_active, "Away": away_win_p / total_active}
            return {"Home": 0.5, "Away": 0.5}

        return None

    def calculate_elo_probs(
        self,
        home_team: str,
        away_team: str,
        sport_key: str,
        completed_matches: Optional[List[Dict[str, Any]]],
        market_key: str,
        point: Optional[float] = None
    ) -> Optional[Dict[str, float]]:
        """Builds season ELO tracker on the fly and estimates probabilities."""
        if not completed_matches or market_key != "h2h":
            return None

        tracker = EloTracker()
        
        # Sort matches chronologically
        try:
            sorted_matches = sorted(completed_matches, key=lambda x: x.get("date", ""))
        except Exception:
            sorted_matches = completed_matches

        for m in sorted_matches:
            h_team = m.get("home_team")
            a_team = m.get("away_team")
            h_goals = m.get("home_goals")
            a_goals = m.get("away_goals")
            if h_team and a_team and h_goals is not None and a_goals is not None:
                tracker.update(h_team, a_team, h_goals, a_goals)

        r_home = tracker.get_rating(home_team)
        r_away = tracker.get_rating(away_team)

        diff = r_home + tracker.home_advantage - r_away
        expected_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

        # Standard soccer draw ELO translation
        draw_prob = 0.26 * (1.0 - abs(expected_home - 0.5))
        home_prob = max(0.01, expected_home - 0.5 * draw_prob)
        away_prob = max(0.01, 1.0 - home_prob - draw_prob)

        # Normalize
        total = home_prob + draw_prob + away_prob
        return {
            home_team: home_prob / total,
            "Draw": draw_prob / total,
            away_team: away_prob / total
        }

    def calculate_form_probs(
        self,
        home_team: str,
        away_team: str,
        completed_matches: Optional[List[Dict[str, Any]]],
        market_key: str,
        point: Optional[float] = None
    ) -> Optional[Dict[str, float]]:
        """Calculates probabilities from recent Form (last 5 matches points)."""
        if not completed_matches:
            return None

        # Gather points for each team in chronological matches
        try:
            sorted_matches = sorted(completed_matches, key=lambda x: x.get("date", ""))
        except Exception:
            sorted_matches = completed_matches

        def get_team_points(team: str) -> List[int]:
            pts = []
            for m in sorted_matches:
                h = m.get("home_team")
                a = m.get("away_team")
                h_goals = m.get("home_goals")
                a_goals = m.get("away_goals")
                if h_goals is None or a_goals is None:
                    continue
                if h == team:
                    if h_goals > a_goals:
                        pts.append(3)
                    elif h_goals == a_goals:
                        pts.append(1)
                    else:
                        pts.append(0)
                elif a == team:
                    if a_goals > h_goals:
                        pts.append(3)
                    elif a_goals == h_goals:
                        pts.append(1)
                    else:
                        pts.append(0)
            return pts[-5:]  # last 5 games

        home_pts_list = get_team_points(home_team)
        away_pts_list = get_team_points(away_team)

        if not home_pts_list or not away_pts_list:
            return None

        home_form = sum(home_pts_list) / 15.0
        away_form = sum(away_pts_list) / 15.0

        if market_key == "h2h":
            home_win = 0.38 + 0.18 * (home_form - away_form)
            away_win = 0.36 + 0.18 * (away_form - home_form)
            draw = max(0.05, 1.0 - home_win - away_win)
            home_win = max(0.05, home_win)
            away_win = max(0.05, away_win)
            total = home_win + draw + away_win
            return {
                home_team: home_win / total,
                "Draw": draw / total,
                away_team: away_win / total
            }
        
        elif market_key == "totals":
            goals_index = (home_form + away_form) / 2
            over_p = 0.40 + 0.20 * goals_index
            under_p = 1.0 - over_p
            return {"Over": over_p, "Under": under_p}

        elif market_key == "btts":
            goals_index = (home_form + away_form) / 2
            yes_p = 0.42 + 0.18 * goals_index
            no_p = 1.0 - yes_p
            return {"Yes": yes_p, "No": no_p}

        return None

    def build_ensemble(
        self,
        home_team: str,
        away_team: str,
        sport_key: str,
        outcomes_odds: List[Dict[str, Any]],
        xg_context: Optional[Dict[str, Any]],
        completed_matches: Optional[List[Dict[str, Any]]],
        market_key: str,
        point: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculates the weighted ensemble probability of all outcomes.
        If a model is unavailable, its weight is redistributed.
        """
        probs_by_model = {}

        # Market model
        if outcomes_odds:
            probs_by_model["market"] = self.get_market_probabilities(outcomes_odds, market_key, home_team, away_team)

        # Poisson Model
        poisson_res = self.calculate_poisson_probs(home_team, away_team, xg_context, market_key, point)
        if poisson_res:
            probs_by_model["poisson"] = poisson_res

        # Elo Model
        elo_res = self.calculate_elo_probs(home_team, away_team, sport_key, completed_matches, market_key, point)
        if elo_res:
            probs_by_model["elo"] = elo_res

        # Form Model
        form_res = self.calculate_form_probs(home_team, away_team, completed_matches, market_key, point)
        if form_res:
            probs_by_model["form"] = form_res

        all_outcomes = []
        if outcomes_odds:
            all_outcomes = [o["name"] for o in outcomes_odds]
        elif poisson_res:
            all_outcomes = list(poisson_res.keys())
        else:
            return {}

        active_weights = {}
        for model in self.weights:
            if model in probs_by_model:
                active_weights[model] = self.weights[model]
        
        non_ai_total_weight = sum(active_weights.values())
        if non_ai_total_weight <= 0:
            if "market" in probs_by_model:
                return probs_by_model["market"]
            return {out: 1.0 / len(all_outcomes) for out in all_outcomes}

        normalized_weights = {m: w / non_ai_total_weight for m, w in active_weights.items()}

        base_probs = {}
        for outcome in all_outcomes:
            prob = 0.0
            for model, norm_w in normalized_weights.items():
                model_probs = probs_by_model[model]
                outcome_prob = model_probs.get(outcome)
                if outcome_prob is None:
                    # Fuzzy match fallback
                    if outcome == home_team and "Home" in model_probs:
                        outcome_prob = model_probs["Home"]
                    elif outcome == away_team and "Away" in model_probs:
                        outcome_prob = model_probs["Away"]
                    elif outcome.lower() in ("home", home_team.lower()) and home_team in model_probs:
                        outcome_prob = model_probs[home_team]
                    elif outcome.lower() in ("away", away_team.lower()) and away_team in model_probs:
                        outcome_prob = model_probs[away_team]
                    else:
                        outcome_prob = 0.0
                prob += norm_w * outcome_prob
            base_probs[outcome] = prob

        total_prob = sum(base_probs.values())
        if total_prob > 0:
            base_probs = {k: v / total_prob for k, v in base_probs.items()}

        return base_probs
