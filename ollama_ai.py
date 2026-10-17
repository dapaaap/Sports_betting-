
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
                    "Package 'ollama' is not installed.\nRun: pip install ollama"
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
        external_context: Optional[str] = None,
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
            external_context,
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
                    logger.error("Ollama error after 3 attempts: %s", e)
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
            from analyzer import expected_value_pct

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
