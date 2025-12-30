"""
Layer 3: Alpha & Behavior Engine

Orchestrates all alpha generation components:
- Deep Futures Transformer
- LLM Market Structure Agent
- Opponent & Crowd Model
- Execution RL Agent

This is where HYDRA earns money.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

from hydra.core.config import HydraConfig
from hydra.core.types import MarketState, Signal, Side, Position, Regime
from hydra.layers.layer2_statistical import StatisticalResult
from hydra.layers.layer3_alpha.transformer_model import FuturesTransformer, TransformerOutput
from hydra.layers.layer3_alpha.llm_agent import LLMMarketStructureAgent, LLMAnalysis
from hydra.layers.layer3_alpha.opponent_model import OpponentCrowdModel, CrowdState
from hydra.layers.layer3_alpha.rl_agent import ExecutionRLAgent, RLDecision, Action


@dataclass
class AlphaSignal:
    """Combined alpha signal from all sources."""
    # Direction
    direction: Side
    confidence: float
    
    # Source signals
    transformer_output: Optional[TransformerOutput] = None
    llm_analysis: Optional[LLMAnalysis] = None
    crowd_state: Optional[CrowdState] = None
    rl_decision: Optional[RLDecision] = None
    
    # Combined metrics
    squeeze_opportunity: bool = False
    fade_opportunity: bool = False
    trap_play: bool = False
    
    # Risk
    expected_adverse_excursion: float = 0.0
    expected_holding_period_hours: float = 1.0
    
    # Reasoning
    primary_thesis: str = ""
    risk_flags: list[str] = None
    
    def __post_init__(self):
        if self.risk_flags is None:
            self.risk_flags = []


class AlphaBehaviorEngine:
    """
    Layer 3: Alpha & Behavior Modeling Engine
    
    Combines multiple alpha sources:
    1. Deep Learning (Transformer) - statistical patterns
    2. LLM Agent - market structure reasoning
    3. Crowd Model - behavioral exploitation
    4. RL Agent - execution optimization
    
    Outputs trading signals with confidence and reasoning.
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        
        # Component models
        self.transformer = FuturesTransformer(
            d_model=config.model.transformer_hidden_size,
            nhead=config.model.transformer_num_heads,
            num_layers=config.model.transformer_num_layers,
            max_seq_len=config.model.transformer_sequence_length,
        )
        
        self.llm_agent = LLMMarketStructureAgent(config)
        self.crowd_model = OpponentCrowdModel(config)
        self.rl_agent = ExecutionRLAgent(config)
        
        # Signal combination weights
        self._weights = {
            'transformer': 0.3,
            'llm': 0.25,
            'crowd': 0.25,
            'statistical': 0.2,
        }
        
        logger.info("Alpha & Behavior Engine initialized")
    
    async def setup(self) -> None:
        """Initialize all components."""
        await self.llm_agent.setup()
        await self.crowd_model.setup()
        await self.rl_agent.setup()
        
        # Load pre-trained transformer if available
        # self.transformer.load(f"{self.config.model.models_dir}/transformer.pt")
        
        logger.info("Alpha components initialized")
    
    async def generate_signals(
        self,
        market_state: MarketState,
        stat_result: StatisticalResult,
        current_position: Optional[Position] = None,
        recent_news: list[str] = None,
    ) -> list[Signal]:
        """
        Generate trading signals from all alpha sources.
        
        Args:
            market_state: Current market data from Layer 1
            stat_result: Statistical analysis from Layer 2
            current_position: Current position if any
            recent_news: News headlines from Layer 1 for LLM context
        
        Returns list of signals, sorted by confidence.
        """
        signals = []
        
        try:
            # 1. Deep Learning prediction
            transformer_output = self.transformer.predict(market_state)
            
            # 2. LLM market structure analysis (with news from Layer 1)
            llm_analysis = await self.llm_agent.analyze(market_state, stat_result, recent_news)
            
            # 3. Crowd behavior analysis
            crowd_state = await self.crowd_model.analyze(market_state)
            
            # 4. Combine signals
            alpha_signal = self._combine_signals(
                transformer_output,
                llm_analysis,
                crowd_state,
                stat_result,
            )
            
            # 5. RL execution decision
            rl_decision = self.rl_agent.decide(
                market_state=market_state,
                stat_result=stat_result,
                current_position=current_position,
                proposed_direction=alpha_signal.direction,
                proposed_confidence=alpha_signal.confidence,
            )
            
            alpha_signal.rl_decision = rl_decision
            
            # Convert to Signal object
            if alpha_signal.direction != Side.FLAT and alpha_signal.confidence > 0.3:
                signal = Signal(
                    timestamp=datetime.now(timezone.utc),
                    symbol=market_state.symbol,
                    side=alpha_signal.direction,
                    confidence=alpha_signal.confidence,
                    expected_return=self._estimate_expected_return(alpha_signal),
                    expected_adverse_excursion=alpha_signal.expected_adverse_excursion,
                    holding_period_minutes=int(alpha_signal.expected_holding_period_hours * 60),
                    source="alpha_engine",
                    regime=stat_result.regime,
                    metadata={
                        'thesis': alpha_signal.primary_thesis,
                        'squeeze_opportunity': alpha_signal.squeeze_opportunity,
                        'fade_opportunity': alpha_signal.fade_opportunity,
                        'trap_play': alpha_signal.trap_play,
                        'rl_action': rl_decision.action.name,
                        'rl_size_mult': rl_decision.size_multiplier,
                        'risk_flags': alpha_signal.risk_flags,
                    },
                )
                signals.append(signal)
            
        except Exception as e:
            logger.exception(f"Error generating signals: {e}")
        
        return signals
    
    def _combine_signals(
        self,
        transformer: TransformerOutput,
        llm: LLMAnalysis,
        crowd: CrowdState,
        stat: StatisticalResult,
    ) -> AlphaSignal:
        """
        Combine signals from all sources into unified alpha.
        
        Priority logic:
        1. If squeeze opportunity detected, prioritize that
        2. If trap play available, evaluate fade
        3. Otherwise, weight-combine directional signals
        """
        risk_flags = []
        
        # Check for squeeze opportunity
        squeeze_opportunity = False
        if transformer.squeeze_probability > 0.6 and crowd.squeeze_vulnerability > 0.5:
            squeeze_opportunity = True
            risk_flags.append("SQUEEZE_PLAY")
        
        # Check for trap/fade opportunity  
        fade_opportunity = crowd.fade_opportunity
        trap_play = llm.trap_direction.value != "none" and llm.trap_probability > 0.5
        
        # Aggregate directional signals
        direction_scores = {Side.LONG: 0.0, Side.SHORT: 0.0, Side.FLAT: 0.0}
        
        # Transformer contribution
        if transformer.long_probability > 0.5:
            direction_scores[Side.LONG] += transformer.long_probability * self._weights['transformer']
        elif transformer.short_probability > 0.5:
            direction_scores[Side.SHORT] += transformer.short_probability * self._weights['transformer']
        else:
            direction_scores[Side.FLAT] += transformer.flat_probability * self._weights['transformer']
        
        # LLM contribution
        llm_weight = self._weights['llm'] * llm.confidence
        direction_scores[llm.directional_bias] += llm_weight
        
        # Crowd contribution (fade or follow based on context)
        crowd_weight = self._weights['crowd'] * crowd.archetype_confidence
        if fade_opportunity:
            # Fade the crowd
            direction_scores[crowd.fade_direction] += crowd_weight * crowd.fade_confidence
        else:
            # Follow smart money if detected
            if crowd.dominant_archetype.value == "smart_money":
                direction_scores[crowd.crowd_direction] += crowd_weight
        
        # Statistical contribution (regime-based)
        stat_weight = self._weights['statistical']
        if stat.regime == Regime.TRENDING_UP:
            direction_scores[Side.LONG] += stat_weight * stat.regime_confidence
        elif stat.regime == Regime.TRENDING_DOWN:
            direction_scores[Side.SHORT] += stat_weight * stat.regime_confidence
        elif stat.regime == Regime.CASCADE_RISK:
            direction_scores[Side.FLAT] += stat_weight
            risk_flags.append("CASCADE_RISK")
        
        # Squeeze adjustment
        if squeeze_opportunity:
            # Position to benefit from squeeze
            if transformer.squeeze_direction == Side.SHORT:
                # Shorts will be squeezed, go long
                direction_scores[Side.LONG] += 0.3
            elif transformer.squeeze_direction == Side.LONG:
                # Longs will be squeezed, go short
                direction_scores[Side.SHORT] += 0.3
        
        # Trap adjustment
        if trap_play:
            if llm.trap_direction.value == "long_trap":
                direction_scores[Side.SHORT] += 0.2 * llm.trap_probability
                risk_flags.append("LONG_TRAP_PLAY")
            elif llm.trap_direction.value == "short_trap":
                direction_scores[Side.LONG] += 0.2 * llm.trap_probability
                risk_flags.append("SHORT_TRAP_PLAY")
        
        # Determine final direction
        final_direction = max(direction_scores, key=direction_scores.get)
        final_confidence = direction_scores[final_direction]
        
        # Normalize confidence
        total_score = sum(direction_scores.values())
        if total_score > 0:
            final_confidence = direction_scores[final_direction] / total_score
        
        # Apply penalties
        if stat.is_abnormal:
            final_confidence *= 0.7
            risk_flags.append("ABNORMAL_MOVE")
        
        if stat.regime_break_alert:
            final_confidence *= 0.5
            risk_flags.append("REGIME_BREAK")
        
        # Add LLM risk flags
        risk_flags.extend(llm.risk_flags)
        
        # Build thesis
        thesis = self._build_thesis(
            final_direction, transformer, llm, crowd, 
            squeeze_opportunity, fade_opportunity, trap_play
        )
        
        # Expected adverse excursion
        eae = transformer.expected_adverse_excursion
        if stat.volatility_regime == "high":
            eae *= 1.5
        elif stat.volatility_regime == "extreme":
            eae *= 2.0
        
        # Holding period based on regime
        if stat.regime in [Regime.TRENDING_UP, Regime.TRENDING_DOWN]:
            holding_hours = 4.0
        elif stat.regime == Regime.RANGING:
            holding_hours = 1.0
        else:
            holding_hours = 0.5
        
        return AlphaSignal(
            direction=final_direction,
            confidence=final_confidence,
            transformer_output=transformer,
            llm_analysis=llm,
            crowd_state=crowd,
            squeeze_opportunity=squeeze_opportunity,
            fade_opportunity=fade_opportunity,
            trap_play=trap_play,
            expected_adverse_excursion=eae,
            expected_holding_period_hours=holding_hours,
            primary_thesis=thesis,
            risk_flags=risk_flags,
        )
    
    def _build_thesis(
        self,
        direction: Side,
        transformer: TransformerOutput,
        llm: LLMAnalysis,
        crowd: CrowdState,
        squeeze: bool,
        fade: bool,
        trap: bool,
    ) -> str:
        """Build human-readable thesis for the trade."""
        parts = []
        
        if direction == Side.FLAT:
            return "No clear edge - staying flat"
        
        dir_str = "LONG" if direction == Side.LONG else "SHORT"
        
        if squeeze:
            squeeze_target = "shorts" if transformer.squeeze_direction == Side.SHORT else "longs"
            parts.append(f"Squeeze opportunity: {squeeze_target} vulnerable ({transformer.squeeze_probability:.0%} prob)")
        
        if trap:
            trap_side = "longs" if llm.trap_direction.value == "long_trap" else "shorts"
            parts.append(f"Trap play: {trap_side} trapped ({llm.trap_probability:.0%} prob)")
        
        if fade:
            parts.append(f"Fade crowd: {crowd.crowding_percentile:.0f}th percentile crowding")
        
        if llm.funding_sustainability == "breaking":
            parts.append(f"Funding unsustainable for {llm.funding_burden_side.value}s")
        
        if not parts:
            if transformer.directional_bias == direction:
                parts.append(f"Statistical edge: {transformer.confidence:.0%} confidence")
            else:
                parts.append(f"Combined signal alignment")
        
        thesis = f"{dir_str}: " + "; ".join(parts)
        
        if llm.reasoning:
            thesis += f" | LLM: {llm.reasoning[:100]}"
        
        return thesis
    
    def _estimate_expected_return(self, alpha: AlphaSignal) -> float:
        """Estimate expected return based on alpha signal."""
        base_return = 0.01  # 1% base expectation
        
        # Adjust for squeeze
        if alpha.squeeze_opportunity:
            base_return *= 2.0
        
        # Adjust for confidence
        base_return *= alpha.confidence
        
        # Adjust for adverse excursion (risk-adjusted)
        if alpha.expected_adverse_excursion > 0:
            risk_ratio = base_return / alpha.expected_adverse_excursion
            if risk_ratio < 1.0:
                base_return *= risk_ratio
        
        return base_return
