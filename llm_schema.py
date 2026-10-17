from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class LLMBetAnalysis:
    recommendation: str = "NEUTRAL"
    confidence: float = 0.5
    prob_adjustment: float = 0.0
    risk_score: Optional[float] = None
    reasoning: str = ""
    key_factors: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    alternative_markets: List[str] = field(default_factory=list)
    score_prediction: Optional[str] = None   # e.g. "1-0", "2-1", "1-1"
    model_used: str = ""
    tokens_used: int = 0
    is_fallback: bool = False
