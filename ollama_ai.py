
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMBetAnalysis:

    recommendation: str = "NEUTRAL"
    confidence: float = 0.0
    prob_adjustment: float = 0.0
    reasoning: str = ""
    key_factors: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    model_used: str = ""
    tokens_used: int = 0
    is_fallback: bool = False


SYSTEM_PROMPT = """\
You are a football betting RISK ANALYST. Your sole task is to identify concrete, verifiable red flags
or advantages that pure statistical models cannot capture — such as confirmed player injuries,
adverse weather conditions, or significant squad rotation. You are NOT a gatekeeper; you are an objective analyst.

You MUST respond with ONLY a raw JSON object — no markdown, no explanation, no code blocks.

Required schema:
{"recommendation":"<STRONG BUY|BUY|NEUTRAL|AVOID>","confidence":<0.0-1.0>,"prob_adjustment":<-0.08 to 0.08>,"reasoning":"<max 80 words>","key_factors":["<factor1>","<factor2>","<factor3>"],"risk_flags":["<risk1>","<risk2>"]}

Critical rules:
- prob_adjustment STRICTLY -0.08 to +0.08.
- Apply POSITIVE prob_adjustment when you identify a concrete contextual advantage (e.g. key injury to opponent, strong home form run, favorable conditions).
- Apply NEGATIVE prob_adjustment ONLY for confirmed concrete red flags: key player injury, extreme weather, major squad rotation.
- Do NOT apply negative adjustment based on uncertainty alone — default to 0.0 when no concrete red flag exists.
- confidence: 0.5=uncertain, 0.65=fairly confident, 0.80=very confident.
- STRONG BUY: clear contextual advantage AND confidence>=0.70.
- BUY: positive contextual factors AND confidence>=0.60.
- NEUTRAL: when no concrete red flag or advantage is identified.
- AVOID: ONLY when a specific, concrete red flag is confirmed (injury, weather, rotation).
- Do NOT hallucinate specific match results, scores, or player stats.
- OUTPUT ONLY THE JSON OBJECT.
"""


def build_match_prompt(
    home_team: str,
    away_team: str,
    league: str,
    hours_until: float,
    market_label: str,
    outcome_name: str,
    market_odds: float,
    fair_prob: float,
    ev_pct: float,
    overround_pct: float,
    num_bookmakers: int,
    all_outcomes: List[Dict],
    sharpness_score: float = 0.5,
    steam_flag: str = "STABLE",
    confidence_score: float = 0.5,
) -> str:
    outcome_lines = "\n".join(
        f"  - {o['name']}: odds {o['best_odds']:.2f} | fair prob {o['fair_prob'] * 100:.1f}%"
        for o in all_outcomes
    )
    overround_pct_val = (overround_pct - 1) * 100
    market_quality = (
        "SHARP"
        if sharpness_score >= 0.65
        else ("MODERATE" if sharpness_score >= 0.35 else "SOFT")
    )

    steam_note = ""
    if steam_flag in ("STEAM", "HIGH_SPREAD"):
        steam_note = (
            f"\nMARKET MOVEMENT: {steam_flag} — significant odds dispersion detected"
        )

    return f"""\
MATCH: {home_team} vs {away_team}
LEAGUE: {league}
KICK-OFF: in {hours_until:.1f} hours

MARKET: {market_label}
TARGET OUTCOME: {outcome_name} @ {market_odds:.2f} (fair prob {fair_prob * 100:.1f}%)

MARKET CONTEXT:
  Overround: {overround_pct_val:.1f}% | Market Quality: {market_quality}
  Bookmakers sampled: {num_bookmakers}{steam_note}

ALL OUTCOMES:
{outcome_lines}

TASK: Identify concrete red flags or advantages for {outcome_name} in {league}.
1. Look for confirmed red flags: key player injuries, adverse weather, significant squad rotation
2. Look for concrete advantages: opponent key injury, strong recent home/away form streak
3. Apply prob_adjustment only when a specific, verifiable factor exists — not based on uncertainty
4. Default to prob_adjustment=0.0 and NEUTRAL when no concrete factor is identified
"""


class OllamaAI:

    def __init__(
        self, model: str, host: str = "http://localhost:11434", api_key: str = ""
    ):
        self.model = model
        self.host = host
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from ollama import AsyncClient

                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                self._client = AsyncClient(host=self.host, headers=headers)
            except ImportError:
                raise ImportError(
                    "Package 'ollama' belum terinstall.\nJalankan: pip install ollama"
                )
        return self._client

    async def analyze_bet(
        self,
        home_team: str,
        away_team: str,
        league: str,
        hours_until: float,
        market_label: str,
        outcome_name: str,
        market_odds: float,
        fair_prob: float,
        ev_pct: float,
        overround_pct: float,
        num_bookmakers: int,
        all_outcomes: List[Dict],
        sharpness_score: float = 0.5,
        steam_flag: str = "STABLE",
        confidence_score: float = 0.5,
    ) -> LLMBetAnalysis:

        user_msg = build_match_prompt(
            home_team,
            away_team,
            league,
            hours_until,
            market_label,
            outcome_name,
            market_odds,
            fair_prob,
            ev_pct,
            overround_pct,
            num_bookmakers,
            all_outcomes,
            sharpness_score,
            steam_flag,
            confidence_score,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        for attempt in range(3):
            try:
                client = self._get_client()
                response = await client.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": 0.1},
                )
                content = response.message.content or ""
                return self._parse_response(content)

            except ImportError as e:
                logger.error("Ollama import error: %s", e)
                return LLMBetAnalysis(is_fallback=True, reasoning=str(e))
            except Exception as e:
                err_str = str(e).lower()

                if "model" in err_str and ("not found" in err_str or "pull" in err_str):
                    logger.error(
                        "Model '%s' not found in Ollama.\n"
                        "   → Run: ollama pull %s",
                        self.model,
                        self.model,
                    )
                    return LLMBetAnalysis(
                        is_fallback=True,
                        reasoning=f"Model '{self.model}' not found. Run: ollama pull {self.model}",
                    )

                if (
                    "connection" in err_str
                    or "refused" in err_str
                    or "connect" in err_str
                ):
                    logger.error(
                        "❌ Ollama is not running.\n"
                        "   → Download & install: https://ollama.com/\n"
                        "     Start Ollama, then run: ollama pull %s",
                        self.model,
                    )
                    return LLMBetAnalysis(
                        is_fallback=True,
                        reasoning="Ollama is not running. Install from https://ollama.com/",
                    )
                if attempt < 2:
                    wait = 2**attempt * 2
                    logger.warning(
                        "Ollama error (attempt %s/3): %s. Retry %ds...",
                        attempt + 1,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error("Ollama error setelah 3x: %s", e)
                    return LLMBetAnalysis(is_fallback=True, reasoning=str(e))

        return LLMBetAnalysis(is_fallback=True, reasoning="Max retries exhausted.")

    def _parse_response(self, content: str) -> LLMBetAnalysis:
        try:
            content = re.sub(r"```(?:json)?\s*", "", content).strip()
            content = content.rstrip("`").strip()

            if not content.startswith("{"):
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    content = match.group(0)
                else:
                    raise ValueError(f"No JSON in response: {content[:200]}")

            parsed = json.loads(content)

            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))

            prob_adjustment = max(
                -0.08, min(0.08, float(parsed.get("prob_adjustment", 0.0)))
            )
            recommendation = str(parsed.get("recommendation", "NEUTRAL")).upper()

            if "STRONG" in recommendation:
                recommendation = "STRONG BUY"
            elif "BUY" in recommendation:
                recommendation = "BUY"
            elif "AVOID" in recommendation:
                recommendation = "AVOID"
            else:
                recommendation = "NEUTRAL"

            return LLMBetAnalysis(
                recommendation=recommendation,
                confidence=confidence,
                prob_adjustment=prob_adjustment,
                reasoning=str(parsed.get("reasoning", "")),
                key_factors=list(parsed.get("key_factors", [])),
                risk_flags=list(parsed.get("risk_flags", [])),
                model_used=self.model,
                tokens_used=0,
                is_fallback=False,
            )

        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logger.warning("Ollama parse error: %s | content: %s", e, content[:300])
            return LLMBetAnalysis(is_fallback=True, reasoning=f"Parse error: {e}")


class OllamaBatchAnalyzer:

    def __init__(self):
        from config import OLLAMA_AI_CONFIG

        self.cfg = OLLAMA_AI_CONFIG
        self.client = OllamaAI(
            model=self.cfg.model,
            host=self.cfg.host,
            api_key=self.cfg.api_key,
        )

    async def enrich_matches(self, matches: List) -> List:
        if not self.cfg.enabled:
            return matches

        value_matches = [
            m
            for m in matches
            if m.has_value and m.top_bet_ev >= self.cfg.min_ev_for_llm
        ]
        if not value_matches:
            return matches

        sem = asyncio.Semaphore(self.cfg.max_concurrent_requests)

        async def analyze_one(match):
            async with sem:
                await self._enrich_match(match)
                if self.cfg.delay_between_calls > 0:
                    await asyncio.sleep(self.cfg.delay_between_calls)

        await asyncio.gather(
            *[analyze_one(m) for m in value_matches], return_exceptions=True
        )
        return matches

    async def _enrich_match(self, match) -> None:
        from typing import Optional as Opt

        from analyzer import MarketAnalysis
        from config import ANALYSIS_CONFIG

        best_mkt: Opt[MarketAnalysis] = None
        for mkt in match.markets:
            if mkt.market_label == match.top_bet_market:
                best_mkt = mkt
                break

        if not best_mkt:
            return

        all_outcomes = [
            {
                "name": o.name,
                "best_odds": o.best_odds,
                "fair_prob": best_mkt.fair_probs.get(o.name, 0),
            }
            for o in best_mkt.outcomes
        ]

        result = await self.client.analyze_bet(
            home_team=match.home_team,
            away_team=match.away_team,
            league=match.sport_name,
            hours_until=match.hours_until,
            market_label=best_mkt.market_label,
            outcome_name=match.top_bet_outcome or "",
            market_odds=match.top_bet_odds,
            fair_prob=best_mkt.fair_prob,
            ev_pct=best_mkt.best_ev_pct,
            overround_pct=best_mkt.overround,
            num_bookmakers=match.num_bookmakers,
            all_outcomes=all_outcomes,
            sharpness_score=getattr(best_mkt, "sharpness_score", 0.5),
            steam_flag=getattr(best_mkt, "steam_flag", "STABLE"),
            confidence_score=getattr(best_mkt, "confidence_score", 0.5),
        )

        match.llm_analysis = result

        if not result.is_fallback:
            from analyzer import expected_value_pct, kelly_fraction

            LLM_TRUST_FACTOR = 0.5
            raw_adjustment = result.prob_adjustment
            trust_adjusted = raw_adjustment * LLM_TRUST_FACTOR

            if result.confidence < 0.6:
                trust_adjusted *= result.confidence / 0.6

            original_prob = best_mkt.ai_prob
            adjusted_prob = max(0.001, min(0.999, original_prob + trust_adjusted))

            logger.info(
                "AI trust-adj [%s/%s | %s]: "
                "orig_prob=%.4f raw_adj=%.4f trust_factor=%.1f "
                "conf=%.2f effective_adj=%.4f final_prob=%.4f",
                match.home_team,
                match.away_team,
                best_mkt.market_key,
                original_prob,
                raw_adjustment,
                LLM_TRUST_FACTOR,
                result.confidence,
                trust_adjusted,
                adjusted_prob,
            )

            best_mkt.ai_prob = round(adjusted_prob, 4)
            best_mkt.ai_signal = _llm_signal_label(result)

            new_ev = expected_value_pct(adjusted_prob, best_mkt.best_odds)
            best_mkt.best_ev_pct = round(new_ev, 2)
            match.top_bet_ev = best_mkt.best_ev_pct

            new_kelly = kelly_fraction(adjusted_prob, best_mkt.best_odds)
            best_mkt.kelly_pct = round(new_kelly * 100, 2)
            best_mkt.kelly_stake = round(new_kelly * ANALYSIS_CONFIG.bankroll, 2)
            match.top_bet_kelly = best_mkt.kelly_pct


def _llm_signal_label(result: LLMBetAnalysis) -> str:
    conf = result.confidence
    rec = result.recommendation
    if rec == "STRONG BUY":
        return "STRONG BUY [AI] ★★★"
    elif rec == "BUY" and conf >= 0.70:
        return "BUY [AI] ★★"
    elif rec == "BUY":
        return "WATCH [AI] ★"
    elif rec == "AVOID":
        return "AVOID [AI] ✗"
    return "NEUTRAL [AI]"
