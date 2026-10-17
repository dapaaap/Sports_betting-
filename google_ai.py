
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

from config import ANALYSIS_CONFIG, GOOGLE_AI_CONFIG

logger = logging.getLogger(__name__)


from llm_schema import LLMBetAnalysis


SYSTEM_PROMPT = """\
You are an objective football match analyst for sports research and education.
This project is not for gambling. Never give betting, wagering, or financial advice,
and never suggest that a user should take an action involving money.
Discuss only match context, estimated probabilities, team form, expected goals,
confirmed availability, weather, and uncertainty. Do not infer facts that are not supplied.

Respond only with a raw JSON object using this schema:
{"recommendation":"NEUTRAL","confidence":<0.0-1.0>,"prob_adjustment":<-0.08 to 0.08>,"risk_score":<0-100>,"reasoning":"<max 80 words>","key_factors":["<factor1>","<factor2>","<factor3>"],"risk_flags":["<uncertainty1>","<uncertainty2>"],"alternative_markets":[],"score_prediction":"<likely scoreline or null>"}

Rules:
- recommendation must always be "NEUTRAL"; it is a compatibility field, not advice.
- Adjust probabilities only for concrete contextual facts; otherwise use 0.0.
- Keep confidence calibrated and describe important uncertainty.
- Do not provide market suggestions, betting language, or monetary values.
- Do not hallucinate results, scores, player statistics, injuries, or other facts.
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
    external_context: Optional[str] = None,
) -> str:
    outcome_lines = "\n".join(
        f"  - {o['name']}: estimated probability {o['fair_prob'] * 100:.1f}%"
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

ANALYSIS AREA: {market_label}
OUTCOME UNDER REVIEW: {outcome_name} (estimated probability {fair_prob * 100:.1f}%)

DATA CONTEXT:
  Data quality: {market_quality}
  Independent data sources sampled: {num_bookmakers}{steam_note}

ALL OUTCOMES:
{outcome_lines}

TASK: Summarize relevant football context and uncertainty for {outcome_name} in {league}.
Do not provide any wagering recommendation. Apply a probability adjustment only when
a specific, verifiable fact supports it; otherwise use 0.0.
{_external_context_block(external_context)}"""


def _external_context_block(external_context: Optional[str]) -> str:
    """Return the ADDITIONAL CONTEXT prompt section, or empty string if None."""
    if not external_context:
        return ""
    return (
        "\nADDITIONAL CONTEXT (from API-Football, treat as supplementary — verify plausibility, "
        "do not treat as ground truth, ignore if irrelevant to your assessment):\n"
        + external_context
        + "\n"
    )


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
        external_context: Optional[str] = None,
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
            external_context,
        )

        use_search = getattr(self.cfg, "enable_search", True)

        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
            "generationConfig": {
                "temperature": self.cfg.temperature,
                "maxOutputTokens": self.cfg.max_tokens,
            },
        }
        if use_search:
            payload["tools"] = [{"googleSearchRetrieval": {}}]

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
                        text = await resp.text()
                        if "quota" in text.lower() or "resource_exhausted" in text.lower():
                            if "tools" in payload:
                                logger.warning(
                                    "Gemini search grounding quota exceeded. Retrying WITHOUT search tools..."
                                )
                                del payload["tools"]
                                try:
                                    self.cfg.enable_search = False
                                except Exception:
                                    pass
                                continue

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
                        if attempt == 2:
                            return LLMBetAnalysis(
                                is_fallback=True,
                                reasoning=f"Gemini sementara tidak tersedia (HTTP {resp.status}).",
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
            risk_score = max(0.0, min(100.0, float(parsed.get("risk_score", 50.0))))
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
                risk_score=risk_score,
                reasoning=str(parsed.get("reasoning", "")),
                key_factors=list(parsed.get("key_factors", [])),
                risk_flags=list(parsed.get("risk_flags", [])),
                alternative_markets=list(parsed.get("alternative_markets", [])),
                score_prediction=parsed.get("score_prediction") or None,
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

            from analyzer import expected_value_pct

            new_ev = expected_value_pct(adjusted_prob, best_mkt.best_odds)
            best_mkt.best_ev_pct = round(new_ev, 2)
            match.top_bet_ev = best_mkt.best_ev_pct

            from analyzer import compute_cqs, cqs_grade

            new_cqs = compute_cqs(
                ev_pct=best_mkt.best_ev_pct,
                data_confidence=best_mkt.data_confidence,
                league_mult=best_mkt.league_multiplier,
                risk_score=result.risk_score if result.risk_score is not None else getattr(best_mkt, "risk_score", 0.0),
                xg_signal=getattr(best_mkt, "xg_signal", "XG_UNAVAILABLE"),
                n_books=best_mkt.num_books_market,
                steam_flag=best_mkt.steam_flag,
            )
            new_grade = cqs_grade(new_cqs)
            best_mkt.cqs = new_cqs
            best_mkt.cqs_grade = new_grade
            match.cqs = new_cqs
            match.cqs_grade = new_grade


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
