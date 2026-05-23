
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

from config import ANALYSIS_CONFIG, GOOGLE_AI_CONFIG

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
    kick_label = f"in {hours_until:.1f} hours"

    overround_pct_val = (overround_pct - 1) * 100
    market_quality = (
        "SHARP"
        if sharpness_score >= 0.65
        else ("MODERATE" if sharpness_score >= 0.35 else "SOFT")
    )

    steam_note = ""
    if steam_flag in ("STEAM", "HIGH_SPREAD"):
        steam_note = f"\nMARKET MOVEMENT: {steam_flag} — significant odds dispersion detected across bookmakers"

    return f"""\
MATCH: {home_team} vs {away_team}
LEAGUE: {league}
KICK-OFF: {kick_label}

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


class GeminiAI:
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self):
        self.cfg = GOOGLE_AI_CONFIG
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=self.cfg.timeout_seconds),
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

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
        if not self.cfg.api_key or self.cfg.api_key in (
            "",
            "YOUR_GOOGLE_AI_STUDIO_KEY_HERE",
        ):
            return LLMBetAnalysis(
                is_fallback=True,
                reasoning="Google AI Studio key belum diisi di api_keys.py.",
            )

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

        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
            "generationConfig": {
                "temperature": self.cfg.temperature,
                "maxOutputTokens": self.cfg.max_tokens,
            },
        }

        url = self.BASE_URL.format(model=self.cfg.model)
        params = {"key": self.cfg.api_key}
        session = await self._get_session()

        for attempt in range(3):
            try:
                async with session.post(url, json=payload, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_response(data)
                    elif resp.status == 429:
                        wait = 2**attempt * 3
                        logger.warning(
                            "Gemini rate limit (attempt %s/3). Menunggu %ds...",
                            attempt + 1,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    elif resp.status in (502, 503, 504):
                        wait = 2**attempt * 2
                        logger.warning(
                            "Gemini HTTP %s (attempt %s/3). Menunggu %ds...",
                            resp.status,
                            attempt + 1,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    elif resp.status == 400:
                        text = await resp.text()
                        if "API_KEY_INVALID" in text or "API key not valid" in text:
                            logger.error(
                                "❌ Google AI Studio API key TIDAK VALID.\n"
                                "   → Isi key yang benar di api_keys.py → GOOGLE_AI_STUDIO_KEY"
                            )
                            self.cfg.enabled = False
                            return LLMBetAnalysis(
                                is_fallback=True,
                                reasoning="Key invalid — isi di api_keys.py.",
                            )
                        logger.error("Gemini 400: %s", text[:300])
                        return LLMBetAnalysis(
                            is_fallback=True, reasoning=f"Bad request: {text[:100]}"
                        )
                    elif resp.status == 403:
                        logger.error("Gemini: API key invalid atau quota habis.")
                        self.cfg.enabled = False
                        return LLMBetAnalysis(
                            is_fallback=True, reasoning="Gemini key habis/invalid."
                        )
                    else:
                        text = await resp.text()
                        logger.warning("Gemini HTTP %s: %s", resp.status, text[:200])
                        return LLMBetAnalysis(
                            is_fallback=True, reasoning=f"API error {resp.status}"
                        )

            except asyncio.TimeoutError:
                logger.warning(
                    "Gemini timeout %s vs %s (attempt %s/3)",
                    home_team,
                    away_team,
                    attempt + 1,
                )
                if attempt == 2:
                    return LLMBetAnalysis(
                        is_fallback=True, reasoning="Request timeout."
                    )
            except aiohttp.ClientError as e:
                logger.warning(
                    "Gemini network error: %s (attempt %s/3)", e, attempt + 1
                )
                if attempt == 2:
                    return LLMBetAnalysis(is_fallback=True, reasoning=str(e))
            except Exception as e:
                logger.error("Gemini unexpected error: %s", e)
                return LLMBetAnalysis(is_fallback=True, reasoning=str(e))

        return LLMBetAnalysis(is_fallback=True, reasoning="Max retries exhausted.")

    def _parse_response(self, data: Dict[str, Any]) -> LLMBetAnalysis:
        try:
            candidate = data["candidates"][0]
            content = candidate["content"]["parts"][0]["text"] or ""
            usage = data.get("usageMetadata", {})
            tokens_used = usage.get("totalTokenCount", 0)

            content = re.sub(r"```(?:json)?\s*", "", content).strip()
            content = content.rstrip("`").strip()

            if not content.startswith("{"):
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    content = match.group(0)

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
                model_used=self.cfg.model,
                tokens_used=tokens_used,
                is_fallback=False,
            )

        except (KeyError, IndexError, json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("Gemini parse error: %s | raw: %s", e, str(data)[:300])
            return LLMBetAnalysis(is_fallback=True, reasoning=f"Parse error: {e}")


class GeminiBatchAnalyzer:
    def __init__(self):
        self.client = GeminiAI()
        self.cfg = GOOGLE_AI_CONFIG

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
        await self.client.close()
        return matches

    async def _enrich_match(self, match) -> None:
        from typing import Optional as Opt

        from analyzer import MarketAnalysis

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

            from analyzer import expected_value_pct, kelly_fraction

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
