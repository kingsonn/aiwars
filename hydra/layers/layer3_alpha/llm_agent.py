"""
LLM Market Structure Agent

This agent thinks like a derivatives desk.
It answers:
- Is leverage crowded?
- Who is trapped?
- Which side is paying to stay in?
- Where will forced exits occur?
- Is narrative reinforcing leverage or breaking it?

Outputs:
- Crowding score
- Trap direction (long-trap / short-trap)
- Narrative-leverage alignment score
- Risk flags
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Literal
from loguru import logger

from hydra.core.config import HydraConfig
from hydra.core.types import MarketState, Side, Regime, TrapDirection
from hydra.layers.layer2_statistical import StatisticalResult


@dataclass
class LLMAnalysis:
    """Output from LLM Market Structure Agent."""
    timestamp: datetime
    symbol: str
    
    # Crowding
    crowding_score: float  # -1 (crowded short) to 1 (crowded long)
    crowding_severity: str  # "low", "moderate", "high", "extreme"
    
    # Trap detection
    trap_direction: TrapDirection
    trap_probability: float
    trap_reasoning: str
    
    # Funding analysis
    funding_burden_side: Side  # Who is paying
    funding_sustainability: str  # "sustainable", "stressed", "breaking"
    
    # Narrative
    narrative_direction: Side  # Where narrative pushes price
    narrative_strength: float  # 0 to 1
    narrative_leverage_alignment: float  # -1 (divergent) to 1 (aligned)
    
    # Forced exit prediction
    forced_exit_zone_long: tuple[float, float]  # Price range where longs liquidate
    forced_exit_zone_short: tuple[float, float]  # Price range where shorts liquidate
    cascade_risk: str  # "low", "moderate", "high"
    
    # Overall
    directional_bias: Side
    confidence: float
    risk_flags: list[str] = field(default_factory=list)
    reasoning: str = ""


# Optimized system prompt - reduced tokens while keeping quality
SYSTEM_PROMPT = """Crypto derivatives analyst. Analyze futures market structure for leverage imbalances and trapped traders.

Output JSON only:
{"crowding_score":float(-1to1),"crowding_severity":"low|moderate|high|extreme","trap_direction":"long_trap|short_trap|none","trap_probability":float,"trap_reasoning":"brief","funding_burden_side":"long|short|flat","funding_sustainability":"sustainable|stressed|breaking","narrative_direction":"long|short|flat","narrative_strength":float,"narrative_leverage_alignment":float(-1to1),"forced_exit_zone_long_pct":[min,max],"forced_exit_zone_short_pct":[min,max],"cascade_risk":"low|moderate|high","directional_bias":"long|short|flat","confidence":float,"risk_flags":[],"reasoning":"1-2 sentences"}"""

# Store last prompt/response for debugging
_last_llm_prompt = ""
_last_llm_response = ""

def get_last_llm_interaction() -> tuple[str, str]:
    """Return last LLM prompt and response for debugging."""
    return _last_llm_prompt, _last_llm_response


class LLMMarketStructureAgent:
    """
    LLM-powered market structure analyzer.
    
    Uses Claude or GPT to reason about:
    - Leverage crowding
    - Trapped traders
    - Forced liquidation zones
    - Narrative-positioning alignment
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        self._client = None
        self._provider = config.llm.llm_provider
        self._model = config.llm.llm_model
        self._enabled = False
    
    async def setup(self) -> None:
        """Initialize LLM client."""
        if self._provider == "groq" and self.config.llm.groq_api_key:
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(
                    api_key=self.config.llm.groq_api_key
                )
                self._enabled = True
                logger.info(f"LLM Agent initialized with Groq ({self._model})")
            except ImportError:
                logger.warning("groq package not installed - run: pip install groq")
        
        elif self._provider == "anthropic" and self.config.llm.anthropic_api_key:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(
                    api_key=self.config.llm.anthropic_api_key
                )
                self._enabled = True
                logger.info("LLM Agent initialized with Anthropic")
            except ImportError:
                logger.warning("anthropic package not installed")
                
        elif self._provider == "openai" and self.config.llm.openai_api_key:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    api_key=self.config.llm.openai_api_key
                )
                self._enabled = True
                logger.info("LLM Agent initialized with OpenAI")
            except ImportError:
                logger.warning("openai package not installed")
        
        if not self._enabled:
            logger.warning("LLM Agent disabled - no API key configured")
    
    async def analyze(
        self,
        market_state: MarketState,
        stat_result: StatisticalResult,
        recent_news: list[str] = None,
    ) -> LLMAnalysis:
        """
        Analyze market structure using LLM.
        
        Args:
            market_state: Current market data
            stat_result: Statistical analysis results
            recent_news: Optional list of recent news headlines
            
        Returns:
            LLMAnalysis with crowding, traps, and directional bias
        """
        if not self._enabled:
            return self._fallback_analysis(market_state, stat_result)
        
        # Prepare context for LLM
        context = self._prepare_context(market_state, stat_result, recent_news)
        
        try:
            response = await self._query_llm(context)
            analysis = self._parse_response(response, market_state)
            return analysis
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analysis(market_state, stat_result)
    
    def _prepare_context(
        self,
        market_state: MarketState,
        stat_result: StatisticalResult,
        recent_news: list[str] = None,
    ) -> str:
        """Prepare COMPACT context string for LLM - optimized for token efficiency."""
        global _last_llm_prompt
        price = market_state.price
        
        # Compact funding
        fr = f"{market_state.funding_rate.rate*100:.4f}%" if market_state.funding_rate else "N/A"
        
        # Compact OI
        oi = "N/A"
        if market_state.open_interest:
            oi_usd = market_state.open_interest.open_interest_usd or 0
            oi_delta = market_state.open_interest.delta_pct or 0
            oi = f"${oi_usd/1e6:.0f}M({oi_delta:+.1f}%)"
        
        # Compact orderbook
        ob = f"{market_state.order_book.imbalance:+.2f}" if market_state.order_book else "N/A"
        
        # Compact liquidations
        liq = "0/0"
        if market_state.recent_liquidations:
            long_liq = sum(l.usd_value for l in market_state.recent_liquidations if l.side == Side.LONG)
            short_liq = sum(l.usd_value for l in market_state.recent_liquidations if l.side == Side.SHORT)
            liq = f"L${long_liq/1e3:.0f}K/S${short_liq/1e3:.0f}K"
        
        # Compact news (only first 3, truncated)
        news = ""
        if recent_news:
            news = "|".join(n[:40] for n in recent_news[:3])
        
        # Super compact context
        context = f"""{market_state.symbol}|${price:.2f}|chg:{market_state.price_change_24h*100:+.1f}%|vol:${market_state.volume_24h/1e6:.0f}M
fr:{fr}|oi:{oi}|ob:{ob}|liq:{liq}|basis:{market_state.basis*100:+.3f}%
regime:{stat_result.regime.name}|vol:{stat_result.realized_volatility*100:.0f}%|jump:{stat_result.jump_probability*100:.0f}%|cascade:{stat_result.cascade_probability*100:.0f}%
news:{news if news else 'none'}
Analyze: Who trapped? Forced exits where?"""
        
        _last_llm_prompt = context
        logger.debug(f"LLM Prompt ({len(context)} chars): {context[:200]}...")
        return context
    
    async def _query_llm(self, context: str) -> str:
        """Query the LLM for analysis."""
        global _last_llm_response
        
        if self._provider == "groq":
            # Groq uses OpenAI-compatible API
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=500,  # Reduced from config - we only need JSON output
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            result = response.choices[0].message.content
            _last_llm_response = result
            
            # Log token usage
            usage = response.usage
            logger.info(f"LLM tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
            logger.debug(f"LLM Response: {result[:300]}...")
            return result
        
        elif self._provider == "anthropic":
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self.config.llm.max_tokens_per_request,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
            )
            return response.content[0].text
            
        elif self._provider == "openai":
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self.config.llm.max_tokens_per_request,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        
        raise ValueError(f"Unknown provider: {self._provider}")
    
    def _parse_response(self, response: str, market_state: MarketState) -> LLMAnalysis:
        """Parse LLM response into structured analysis."""
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON: {e}")
            return self._fallback_analysis(market_state, None)
        
        price = market_state.price
        
        # Parse forced exit zones (percentages to prices)
        long_zone_pct = data.get('forced_exit_zone_long_pct', [5, 15])
        short_zone_pct = data.get('forced_exit_zone_short_pct', [5, 15])
        
        forced_exit_zone_long = (
            price * (1 - long_zone_pct[1] / 100),
            price * (1 - long_zone_pct[0] / 100),
        )
        forced_exit_zone_short = (
            price * (1 + short_zone_pct[0] / 100),
            price * (1 + short_zone_pct[1] / 100),
        )
        
        # Map sides
        def parse_side(s: str) -> Side:
            if s == "long":
                return Side.LONG
            elif s == "short":
                return Side.SHORT
            return Side.FLAT
        
        def parse_trap(s: str) -> TrapDirection:
            if s == "long_trap":
                return TrapDirection.LONG_TRAP
            elif s == "short_trap":
                return TrapDirection.SHORT_TRAP
            return TrapDirection.NONE
        
        return LLMAnalysis(
            timestamp=datetime.now(timezone.utc),
            symbol=market_state.symbol,
            crowding_score=float(data.get('crowding_score', 0)),
            crowding_severity=data.get('crowding_severity', 'low'),
            trap_direction=parse_trap(data.get('trap_direction', 'none')),
            trap_probability=float(data.get('trap_probability', 0)),
            trap_reasoning=data.get('trap_reasoning', ''),
            funding_burden_side=parse_side(data.get('funding_burden_side', 'flat')),
            funding_sustainability=data.get('funding_sustainability', 'sustainable'),
            narrative_direction=parse_side(data.get('narrative_direction', 'flat')),
            narrative_strength=float(data.get('narrative_strength', 0.5)),
            narrative_leverage_alignment=float(data.get('narrative_leverage_alignment', 0)),
            forced_exit_zone_long=forced_exit_zone_long,
            forced_exit_zone_short=forced_exit_zone_short,
            cascade_risk=data.get('cascade_risk', 'low'),
            directional_bias=parse_side(data.get('directional_bias', 'flat')),
            confidence=float(data.get('confidence', 0.5)),
            risk_flags=data.get('risk_flags', []),
            reasoning=data.get('reasoning', ''),
        )
    
    def _fallback_analysis(
        self,
        market_state: MarketState,
        stat_result: Optional[StatisticalResult],
    ) -> LLMAnalysis:
        """Generate analysis without LLM using heuristics."""
        price = market_state.price
        
        # Derive crowding from funding
        crowding_score = 0.0
        crowding_severity = "low"
        trap_direction = TrapDirection.NONE
        funding_burden = Side.FLAT
        
        if market_state.funding_rate:
            rate = market_state.funding_rate.rate
            crowding_score = min(1.0, max(-1.0, rate * 1000))
            
            if abs(rate) > 0.001:
                crowding_severity = "extreme"
            elif abs(rate) > 0.0005:
                crowding_severity = "high"
            elif abs(rate) > 0.0002:
                crowding_severity = "moderate"
            
            if rate > 0:
                funding_burden = Side.LONG
                if crowding_severity in ["high", "extreme"]:
                    trap_direction = TrapDirection.LONG_TRAP
            elif rate < 0:
                funding_burden = Side.SHORT
                if crowding_severity in ["high", "extreme"]:
                    trap_direction = TrapDirection.SHORT_TRAP
        
        # Default exit zones (10-20% away)
        forced_exit_zone_long = (price * 0.80, price * 0.90)
        forced_exit_zone_short = (price * 1.10, price * 1.20)
        
        # Directional bias from crowding (fade the crowd)
        if crowding_score > 0.5:
            directional_bias = Side.SHORT
        elif crowding_score < -0.5:
            directional_bias = Side.LONG
        else:
            directional_bias = Side.FLAT
        
        # Cascade risk from stat result
        cascade_risk = "low"
        if stat_result:
            if stat_result.cascade_probability > 0.5:
                cascade_risk = "high"
            elif stat_result.cascade_probability > 0.2:
                cascade_risk = "moderate"
        
        return LLMAnalysis(
            timestamp=datetime.now(timezone.utc),
            symbol=market_state.symbol,
            crowding_score=crowding_score,
            crowding_severity=crowding_severity,
            trap_direction=trap_direction,
            trap_probability=0.3 if trap_direction != TrapDirection.NONE else 0.0,
            trap_reasoning="Derived from funding rate and crowding heuristics",
            funding_burden_side=funding_burden,
            funding_sustainability="sustainable",
            narrative_direction=Side.FLAT,
            narrative_strength=0.0,
            narrative_leverage_alignment=0.0,
            forced_exit_zone_long=forced_exit_zone_long,
            forced_exit_zone_short=forced_exit_zone_short,
            cascade_risk=cascade_risk,
            directional_bias=directional_bias,
            confidence=0.4,  # Lower confidence for heuristic-based
            risk_flags=["LLM_DISABLED"],
            reasoning="Heuristic analysis based on funding and market data.",
        )
