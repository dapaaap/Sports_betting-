
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

from config import ANALYSIS_CONFIG, GITHUB_AI_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class LLMBetAnalysis:

    risk_score: float = 5.0
    confidence: float = 0.5

    reasoning: str = ""
    key_factors: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)

    model_used: str = ""
    tokens_used: int = 0
    is_fallback: bool = False


SYSTEM_PROMPT = """\
You are a football betting RISK ANALYST. Your task is to assign a risk_score (0–10) to a bet
based on concrete, verifiable contextual factors ONLY.
  0–2 : No red flags. Clean match context.
  3–4 : Minor concerns (travel, fatigue, minor rotation).
  5–6 : Moderate risk (possible rotation, form dip, fixture congestion).
  7–8 : Significant risk (key player doubt, adverse weather, high-pressure context).
  9–10: Confirmed high-impact red flags (star player confirmed out, extreme weather).
Default to 5 when you have no concrete information.
You MUST respond with ONLY a raw JSON object. Do NOT include any text, explanation, or markdown.

Exact schema required:
{"risk_score":<0-10>,"confidence":<0.0-1.0>,"reasoning":"<max 80 words>","key_factors":["<factor1>","<factor2>"],"risk_flags":["<risk1>"]}

Rules:
- risk_score: numeric 0–10. Use the scale above strictly.
- confidence: how certain you are of the risk_score (0.5=uncertain, 0.80=very confident).
- key_factors: list 1–3 contextual factors you identified (positive or negative).
- risk_flags: list ONLY confirmed red flags. Empty list [] if none.
- Do NOT hallucinate player stats, match results, or team news.
- Do NOT output recommendation fields (BUY/AVOID). Only risk_score matters.
- OUTPUT ONLY THE JSON OBJECT, NOTHING ELSE.
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
    pinnacle_odds: Optional[float] = None,
    best_book_name: str = "",
    odds_spread_pct: Optional[float] = None,
    consensus_direction: str = "STABLE",
    books_above_avg: int = 0,
) -> str:

    outcome_lines = "\n".join(
        f"  - {o['name']}: market {o['best_odds']:.2f} | avg {o.get('avg_odds', o['best_odds']):.2f} | fair prob {o['fair_prob'] * 100:.1f}%"
        for o in all_outcomes
    )
    overround_pct_val = (overround_pct - 1) * 100
    market_quality = (
        "SHARP"
        if sharpness_score >= 0.65
        else ("MODERATE" if sharpness_score >= 0.35 else "SOFT")
    )

    steam_lines = []

    steam_desc = {
        "STEAM": "STEAM MOVE — very large cross-book spread (>12%). Likely sharp money or late news.",
        "HIGH_SPREAD": "HIGH SPREAD — significant cross-book dispersion (7–12%). Possible information asymmetry.",
        "MED_SPREAD": "MODERATE SPREAD — some bookmaker disagreement (4–7%). Worth noting.",
        "STABLE": "STABLE — bookmakers broadly agree. No unusual movement detected.",
    }.get(steam_flag, steam_flag)
    steam_lines.append(f"Cross-book spread : {steam_desc}")

    if odds_spread_pct is not None:
        if odds_spread_pct >= 8.0:
            direction_label = (
                "STRONG DRIFT"
                if consensus_direction == "DRIFTING"
                else "STRONG SHORTEN"
            )
        elif odds_spread_pct >= 4.0:
            direction_label = consensus_direction
        else:
            direction_label = "STABLE"
        steam_lines.append(
            f"Odds spread (best vs min) : {odds_spread_pct:.1f}% → {direction_label}"
        )

    if pinnacle_odds and pinnacle_odds > 0:
        diff_pct = (market_odds - pinnacle_odds) / pinnacle_odds * 100
        if abs(diff_pct) < 1.0:
            pin_label = "Pinnacle aligned with market best"
        elif diff_pct > 0:
            pin_label = f"Market best is {diff_pct:.1f}% ABOVE Pinnacle (soft books offering value vs sharp)"
        else:
            pin_label = (
                f"Pinnacle is {abs(diff_pct):.1f}% ABOVE market best (sharp book leads)"
            )
        steam_lines.append(f"Pinnacle vs best  : {pinnacle_odds:.2f} | {pin_label}")
    else:
        steam_lines.append("Pinnacle          : not available in sample")

    if best_book_name:
        book_type = (
            "SHARP"
            if best_book_name in ("pinnacle", "betfair_ex_eu", "matchbook")
            else "SOFT"
        )
        steam_lines.append(f"Best odds source  : {best_book_name} ({book_type})")

    if books_above_avg > 0 and num_bookmakers > 0:
        pct_above = books_above_avg / num_bookmakers * 100
        if pct_above >= 70:
            consensus_note = f"{books_above_avg}/{num_bookmakers} books above avg odds — odds may be SHORTENING broadly"
        elif pct_above <= 30:
            consensus_note = f"{books_above_avg}/{num_bookmakers} books above avg odds — odds may be DRIFTING broadly"
        else:
            consensus_note = f"{books_above_avg}/{num_bookmakers} books above avg odds — mixed consensus"
        steam_lines.append(f"Consensus         : {consensus_note}")

    market_intel_block = "\n".join(f"  {l}" for l in steam_lines)

    return f"""\
MATCH: {home_team} vs {away_team}
LEAGUE: {league}
KICK-OFF: in {hours_until:.1f} hours

MARKET: {market_label}
TARGET OUTCOME: {outcome_name} @ {market_odds:.2f} (fair prob {fair_prob * 100:.1f}% | EV {ev_pct:+.1f}%)

MARKET CONTEXT:
  Overround: {overround_pct_val:.1f}% | Market Quality: {market_quality}
  Bookmakers sampled: {num_bookmakers}

MARKET INTELLIGENCE (odds movement signals):
{market_intel_block}

ALL OUTCOMES IN MARKET:
{outcome_lines}

TASK: Assign a risk_score (0–10) for betting on {outcome_name} in {league}.
1. Check for confirmed red flags: key player injuries, adverse weather, significant squad rotation
2. Consider fixture context: importance of match, fatigue, competition stage
3. Use MARKET INTELLIGENCE above as supporting signal (e.g. STEAM = sharp money = lower risk)
4. Score 5 (neutral) if no specific information is available — do not guess
5. Score higher (7–10) ONLY for confirmed, concrete negative factors
"""


class GitHubAI:

    def __init__(self):
        self.cfg = GITHUB_AI_CONFIG

        self.endpoint = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.cfg.api_key}",
                    "Content-Type": "application/json",
                },
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
        pinnacle_odds: Optional[float] = None,
        best_book_name: str = "",
        odds_spread_pct: Optional[float] = None,
        consensus_direction: str = "STABLE",
        books_above_avg: int = 0,
    ) -> LLMBetAnalysis:
        if not self.cfg.api_key or self.cfg.api_key in ("", "YOUR_GITHUB_TOKEN_HERE"):
            logger.warning(
                "GitHub Models: GITHUB_TOKEN tidak dikonfigurasi — fallback ke no-AI mode."
            )
            return LLMBetAnalysis(
                is_fallback=True, reasoning="GitHub token not configured."
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
            pinnacle_odds=pinnacle_odds,
            best_book_name=best_book_name,
            odds_spread_pct=odds_spread_pct,
            consensus_direction=consensus_direction,
            books_above_avg=books_above_avg,
        )

        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        }

        session = await self._get_session()
        for attempt in range(3):
            try:
                async with session.post(self.endpoint, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_response(data)
                    elif resp.status in (429, 502, 503, 504):
                        wait = 2**attempt * 2
                        logger.warning(
                            "GitHub Models HTTP %d (attempt %d/3). Retry in %ds...",
                            resp.status,
                            attempt + 1,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    elif resp.status == 401:
                        logger.error(
                            "GitHub Models: Invalid/expired token — fallback ke no-AI mode."
                        )
                        return LLMBetAnalysis(
                            is_fallback=True, reasoning="Invalid GitHub token."
                        )
                    elif resp.status == 402:
                        logger.warning(
                            "GitHub Models: Insufficient credits — fallback ke no-AI mode."
                        )
                        return LLMBetAnalysis(
                            is_fallback=True,
                            reasoning="GitHub Models credits exhausted.",
                        )
                    else:
                        text = await resp.text()
                        logger.warning(
                            "GitHub Models HTTP %d: %s", resp.status, text[:200]
                        )
                        return LLMBetAnalysis(
                            is_fallback=True, reasoning=f"API error {resp.status}"
                        )

            except asyncio.TimeoutError:
                logger.warning(
                    "GitHub Models timeout: %s vs %s (attempt %d/3)",
                    home_team,
                    away_team,
                    attempt + 1,
                )
                if attempt == 2:
                    return LLMBetAnalysis(
                        is_fallback=True, reasoning="Request timed out — no-AI mode."
                    )
            except aiohttp.ClientError as e:
                logger.warning(
                    "GitHub Models network error: %s (attempt %d/3)", e, attempt + 1
                )
                if attempt == 2:
                    return LLMBetAnalysis(
                        is_fallback=True, reasoning=f"Network error: {e}"
                    )
            except Exception as e:
                logger.error(
                    "GitHub Models unexpected error: %s — fallback ke no-AI mode.", e
                )
                return LLMBetAnalysis(is_fallback=True, reasoning=str(e))

        return LLMBetAnalysis(
            is_fallback=True, reasoning="Max retries exhausted — no-AI mode."
        )

    def _parse_response(self, data: Dict[str, Any]) -> LLMBetAnalysis:
        try:
            content = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            model = data.get("model", self.cfg.model)

            content = re.sub(r"```(?:json)?\s*", "", content).strip()
            content = content.rstrip("`").strip()

            if not content.startswith("{"):
                m = re.search(r"\{.*\}", content, re.DOTALL)
                if m:
                    content = m.group(0)

            parsed = json.loads(content)

            risk_score = max(0.0, min(10.0, float(parsed.get("risk_score", 5.0))))
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))

            return LLMBetAnalysis(
                risk_score=round(risk_score, 1),
                confidence=confidence,
                reasoning=str(parsed.get("reasoning", "")),
                key_factors=list(parsed.get("key_factors", [])),
                risk_flags=list(parsed.get("risk_flags", [])),
                model_used=model,
                tokens_used=usage.get("total_tokens", 0),
                is_fallback=False,
            )

        except (KeyError, json.JSONDecodeError, ValueError, TypeError) as e:
            raw_preview = str(data)[:300]
            logger.warning("GitHub Models parse error: %s | raw: %s", e, raw_preview)
            return LLMBetAnalysis(is_fallback=True, reasoning=f"Parse error: {e}")


class GitHubBatchAnalyzer:

    def __init__(self):
        self.client = GitHubAI()
        self.cfg = GITHUB_AI_CONFIG

    async def enrich_matches(self, matches: List) -> List:
        if not self.cfg.enabled:
            logger.info("GitHub Models AI disabled — statistical model only.")
            return matches

        if not self.cfg.is_configured:
            logger.warning(
                "GitHub Models: GITHUB_TOKEN tidak diset di api_keys.py — "
                "fallback ke no-AI mode. Set GITHUB_TOKEN untuk mengaktifkan GPT-4o enrichment."
            )
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
        from analyzer import MarketAnalysis

        best_mkt: Optional[MarketAnalysis] = None
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
                "avg_odds": o.avg_odds,
                "fair_prob": best_mkt.fair_probs.get(o.name, 0),
                "all_odds": o.all_odds,
            }
            for o in best_mkt.outcomes
        ]

        target_outcome = match.top_bet_outcome or ""
        pinnacle_odds: Optional[float] = None
        for o in best_mkt.outcomes:
            if o.name == target_outcome and o.bookmaker in (
                "pinnacle",
                "pinnacle_sports",
            ):
                pinnacle_odds = o.best_odds
                break

        if pinnacle_odds is None:
            for o in best_mkt.outcomes:
                if o.name == target_outcome:
                    if o.all_odds:
                        pinnacle_odds = min(o.all_odds)
                    break

        odds_spread_pct: Optional[float] = None
        books_above_avg: int = 0
        consensus_direction: str = "STABLE"
        for o in best_mkt.outcomes:
            if o.name == target_outcome and len(o.all_odds) >= 2:
                o_min = min(o.all_odds)
                o_max = max(o.all_odds)
                o_avg = sum(o.all_odds) / len(o.all_odds)
                if o_min > 0:
                    odds_spread_pct = round((o_max - o_min) / o_min * 100, 1)
                books_above_avg = sum(1 for x in o.all_odds if x > o_avg)

                if o.best_odds > o_avg * 1.03:
                    consensus_direction = "DRIFTING"
                elif o.best_odds < o_avg * 0.97:
                    consensus_direction = "SHORTENING"
                else:
                    consensus_direction = "STABLE"
                break

        result = await self.client.analyze_bet(
            home_team=match.home_team,
            away_team=match.away_team,
            league=match.sport_name,
            hours_until=match.hours_until,
            market_label=best_mkt.market_label,
            outcome_name=target_outcome,
            market_odds=match.top_bet_odds,
            fair_prob=best_mkt.fair_prob,
            ev_pct=best_mkt.best_ev_pct,
            overround_pct=best_mkt.overround,
            num_bookmakers=match.num_bookmakers,
            all_outcomes=all_outcomes,
            sharpness_score=getattr(best_mkt, "sharpness_score", 0.5),
            steam_flag=getattr(best_mkt, "steam_flag", "STABLE"),
            confidence_score=getattr(best_mkt, "confidence_score", 0.5),
            pinnacle_odds=pinnacle_odds,
            best_book_name=getattr(best_mkt, "best_book", ""),
            odds_spread_pct=odds_spread_pct,
            consensus_direction=consensus_direction,
            books_above_avg=books_above_avg,
        )

        match.llm_analysis = result

        if not result.is_fallback:
            risk_score = result.risk_score
            ev_pct = best_mkt.best_ev_pct

            if risk_score >= 7.0 and ev_pct < 7.0:
                best_mkt.ai_signal = f"SKIP — Risk {risk_score:.0f}/10 [AI] ✗"
                best_mkt.risk_score = risk_score

                best_mkt.best_ev_pct = 0.0
                match.top_bet_ev = 0.0
                logger.info(
                    "GPT-4o SKIP [%s/%s]: risk_score=%.1f ev=%.2f%% — bet dilewati.",
                    match.home_team,
                    match.away_team,
                    risk_score,
                    ev_pct,
                )
                return

            final_score = ev_pct * 0.70 - (risk_score / 10.0) * ev_pct * 0.30
            final_score = round(final_score, 2)

            logger.info(
                "GPT-4o risk-score [%s/%s | %s]: "
                "ev=%.2f%% risk_score=%.1f conf=%.2f final_score=%.2f%%",
                match.home_team,
                match.away_team,
                best_mkt.market_key,
                ev_pct,
                risk_score,
                result.confidence,
                final_score,
            )

            best_mkt.risk_score = risk_score
            best_mkt.final_score = final_score
            best_mkt.ai_signal = _llm_signal_label(result)

            best_mkt.best_ev_pct = final_score
            match.top_bet_ev = final_score

            if ev_pct > 0:
                kelly_scale = final_score / ev_pct
                best_mkt.kelly_pct = round(best_mkt.kelly_pct * kelly_scale, 2)
                best_mkt.kelly_stake = round(best_mkt.kelly_stake * kelly_scale, 2)
                match.top_bet_kelly = best_mkt.kelly_pct

            from analyzer import compute_cqs, cqs_grade

            new_cqs = compute_cqs(
                ev_pct=best_mkt.best_ev_pct,
                data_confidence=getattr(best_mkt, "data_confidence", 0.5),
                league_mult=getattr(best_mkt, "league_multiplier", 0.7),
                risk_score=risk_score,
                xg_signal=getattr(best_mkt, "xg_signal", "XG_UNAVAILABLE"),
                n_books=getattr(best_mkt, "num_books_market", 4),
                steam_flag=getattr(best_mkt, "steam_flag", "STABLE"),
            )
            new_grade = cqs_grade(new_cqs)
            best_mkt.cqs = new_cqs
            best_mkt.cqs_grade = new_grade
            match.cqs = new_cqs
            match.cqs_grade = new_grade


def _llm_signal_label(result: LLMBetAnalysis) -> str:
    rs = result.risk_score
    if rs <= 2:
        return f"LOW RISK {rs:.0f}/10 [AI] ✅"
    elif rs <= 4:
        return f"RISK {rs:.0f}/10 [AI] ★"
    elif rs <= 6:
        return f"RISK {rs:.0f}/10 [AI] ⚠"
    elif rs <= 7.9:
        return f"HIGH RISK {rs:.0f}/10 [AI] ⚠⚠"
    else:
        return f"DANGER {rs:.0f}/10 [AI] ✗"
