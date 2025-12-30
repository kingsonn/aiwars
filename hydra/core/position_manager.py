"""
HYDRA Position Manager

Manages the dual-mode operation:
1. Entry Mode - For pairs WITHOUT positions (can LONG, SHORT, or FLAT)
2. Position Management Mode - For pairs WITH positions (can only HOLD or EXIT)

This is the central orchestrator that coordinates all 5 layers for both modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from enum import Enum
from loguru import logger

from hydra.core.config import HydraConfig
from hydra.core.types import (
    MarketState,
    Position,
    Signal,
    Side,
    Regime,
    PositionAction,
    ThesisStatus,
    PositionManagementDecision,
    EntryDecision,
)
from hydra.layers.layer2_statistical import StatisticalResult, TradingDecision


class OperatingMode(Enum):
    """Operating mode for a symbol."""
    ENTRY = "entry"  # No position - evaluating entry
    MANAGEMENT = "management"  # Has position - managing it


@dataclass
class PositionContext:
    """Context for position management decisions."""
    position: Position
    entry_thesis: str = ""
    entry_signal_source: str = ""
    entry_confidence: float = 0.0
    
    # Tracking
    peak_unrealized_pnl: float = 0.0
    peak_unrealized_pnl_pct: float = 0.0
    total_funding_paid: float = 0.0
    
    # Original conditions
    entry_funding_rate: float = 0.0
    entry_oi_delta: float = 0.0
    entry_long_short_ratio: float = 1.0


@dataclass
class LeverageDecision:
    """Decision on what leverage to use."""
    leverage: float
    reasoning: str
    confidence_factor: float  # How much confidence affects leverage
    volatility_factor: float  # How much volatility reduces leverage
    funding_factor: float  # How much funding affects leverage
    drawdown_factor: float  # How much portfolio drawdown reduces leverage


class PositionManager:
    """
    Central position manager for HYDRA.
    
    Coordinates all 5 layers differently based on operating mode:
    
    ENTRY MODE (no position):
        - L1: Gather market data
        - L2: Gate decision (ALLOW/RESTRICT/BLOCK)
        - L3: Generate entry signals
        - L4: Size and risk check
        - L5: Multi-agent vote and execute
        
    MANAGEMENT MODE (has position):
        - L1: Same - gather data
        - L2: Safety check - FORCE EXIT if BLOCK
        - L3: Check if thesis still valid
        - L4: Exit decision (PnL, time, funding, drawdown)
        - L5: Execute exit (reduce-only, limit orders)
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        
        # Track position contexts (entry thesis, etc.)
        self._position_contexts: Dict[str, PositionContext] = {}
        
        # Portfolio state
        self._total_equity: float = 0.0
        self._available_balance: float = 0.0
        
        logger.info("Position Manager initialized")
    
    def get_operating_mode(self, symbol: str, position: Optional[Position]) -> OperatingMode:
        """Determine operating mode for a symbol."""
        if position and position.is_open:
            return OperatingMode.MANAGEMENT
        return OperatingMode.ENTRY
    
    def store_position_context(
        self,
        symbol: str,
        position: Position,
        thesis: str,
        signal_source: str,
        confidence: float,
        funding_rate: float = 0.0,
        oi_delta: float = 0.0,
        long_short_ratio: float = 1.0,
    ) -> None:
        """Store context when opening a position (for later thesis checking)."""
        self._position_contexts[symbol.lower()] = PositionContext(
            position=position,
            entry_thesis=thesis,
            entry_signal_source=signal_source,
            entry_confidence=confidence,
            entry_funding_rate=funding_rate,
            entry_oi_delta=oi_delta,
            entry_long_short_ratio=long_short_ratio,
        )
        logger.debug(f"Stored position context for {symbol}: {thesis[:50]}...")
    
    def get_position_context(self, symbol: str) -> Optional[PositionContext]:
        """Get stored context for a position."""
        return self._position_contexts.get(symbol.lower())
    
    def clear_position_context(self, symbol: str) -> None:
        """Clear context when position is closed."""
        self._position_contexts.pop(symbol.lower(), None)
    
    def update_portfolio_state(self, total_equity: float, available_balance: float) -> None:
        """Update portfolio state for sizing decisions."""
        self._total_equity = total_equity
        self._available_balance = available_balance
    
    # =========================================================================
    # LAYER 2: Safety Gate Check
    # =========================================================================
    
    def check_layer2_safety(
        self,
        stat_result: StatisticalResult,
        position: Optional[Position],
    ) -> tuple[bool, str]:
        """
        Layer 2 safety check for position management.
        
        For positions: If BLOCK → FORCE EXIT immediately
        For no position: If BLOCK → Don't enter
        
        Returns:
            (force_exit, reason)
        """
        if stat_result.trading_decision == TradingDecision.BLOCK:
            if position and position.is_open:
                return True, f"L2 BLOCK: {stat_result.danger_zone_reason} - FORCE EXIT"
            return False, "L2 BLOCK - no entry allowed"
        
        return False, ""
    
    # =========================================================================
    # LAYER 3: Thesis Checking (Position Management Mode)
    # =========================================================================
    
    def check_thesis_validity(
        self,
        position: Position,
        market_state: MarketState,
        stat_result: StatisticalResult,
        context: Optional[PositionContext],
    ) -> ThesisStatus:
        """
        Layer 3: Check if original trade thesis is still valid.
        
        Checks:
        - Funding rate change (if we entered because funding was favorable)
        - OI change (if we entered because of OI signal)
        - Long/short ratio change
        - Regime change
        - Liquidation pressure switch
        """
        if not context:
            # No context - assume thesis weakening
            return ThesisStatus.WEAKENING
        
        reasons_broken = []
        reasons_weakening = []
        
        # 1. Check regime change
        if position.side == Side.LONG:
            if stat_result.regime == Regime.TRENDING_DOWN:
                reasons_broken.append("Regime turned bearish")
            elif stat_result.regime == Regime.CASCADE_RISK:
                reasons_broken.append("Cascade risk detected")
            elif stat_result.regime == Regime.SQUEEZE_LONG:
                reasons_weakening.append("Long squeeze risk")
        else:  # SHORT
            if stat_result.regime == Regime.TRENDING_UP:
                reasons_broken.append("Regime turned bullish")
            elif stat_result.regime == Regime.CASCADE_RISK:
                reasons_broken.append("Cascade risk detected")
            elif stat_result.regime == Regime.SQUEEZE_SHORT:
                reasons_weakening.append("Short squeeze risk")
        
        # 2. Check funding rate flip
        if market_state.funding_rate:
            current_funding = market_state.funding_rate.rate
            entry_funding = context.entry_funding_rate
            
            # Funding flipped significantly against us
            if position.side == Side.LONG:
                if current_funding > 0.0005 and entry_funding <= 0:
                    reasons_weakening.append("Funding turned against longs")
                if current_funding > 0.001:
                    reasons_broken.append("High funding burden for long")
            else:
                if current_funding < -0.0005 and entry_funding >= 0:
                    reasons_weakening.append("Funding turned against shorts")
                if current_funding < -0.001:
                    reasons_broken.append("High funding burden for short")
        
        # 3. Check volatility explosion
        if stat_result.volatility_regime == "extreme":
            reasons_weakening.append("Volatility explosion")
        
        # 4. Check if original squeeze/trap is unwound
        if "trap" in context.entry_thesis.lower() or "squeeze" in context.entry_thesis.lower():
            # Check if the trap condition has resolved
            if stat_result.regime == Regime.RANGING:
                reasons_weakening.append("Trap/squeeze condition may have resolved")
        
        # Determine status
        if reasons_broken:
            logger.info(f"Thesis BROKEN for {position.symbol}: {reasons_broken}")
            return ThesisStatus.BROKEN
        elif reasons_weakening:
            logger.debug(f"Thesis WEAKENING for {position.symbol}: {reasons_weakening}")
            return ThesisStatus.WEAKENING
        
        return ThesisStatus.INTACT
    
    # =========================================================================
    # LAYER 4: Position Exit Decision
    # =========================================================================
    
    def evaluate_exit_decision(
        self,
        position: Position,
        market_state: MarketState,
        stat_result: StatisticalResult,
        thesis_status: ThesisStatus,
        context: Optional[PositionContext],
    ) -> PositionManagementDecision:
        """
        Layer 4: Decide whether to hold or exit position.
        
        Exit triggers:
        1. Thesis broken → EXIT
        2. Stop loss hit → EXIT
        3. Take profit hit → EXIT
        4. Max holding time exceeded → EXIT
        5. Funding cost too high → EXIT
        6. Drawdown from peak → Partial EXIT
        7. Kill switch conditions → FORCE EXIT
        """
        now = datetime.now(timezone.utc)
        
        # Calculate metrics
        unrealized_pnl_pct = position.unrealized_pnl_pct
        time_in_position = 0.0
        if position.entry_time:
            time_in_position = (now - position.entry_time).total_seconds() / 3600
        
        funding_paid = position.funding_paid if hasattr(position, 'funding_paid') else 0.0
        if context:
            funding_paid = context.total_funding_paid
        
        # Update peak tracking
        if context and unrealized_pnl_pct > context.peak_unrealized_pnl_pct:
            context.peak_unrealized_pnl_pct = unrealized_pnl_pct
            context.peak_unrealized_pnl = position.unrealized_pnl
        
        # =====================================================================
        # EXIT CHECKS (in priority order)
        # =====================================================================
        
        # 1. Thesis broken → EXIT
        if thesis_status == ThesisStatus.BROKEN:
            return PositionManagementDecision(
                action=PositionAction.EXIT,
                thesis_status=thesis_status,
                exit_reason="Thesis broken - original trade reason no longer valid",
                exit_pct=1.0,
                unrealized_pnl_pct=unrealized_pnl_pct,
                time_in_position_hours=time_in_position,
                funding_paid=funding_paid,
                confidence=0.9,
                reasoning="Exit: Trade thesis is broken",
            )
        
        # 2. Stop loss hit (force exit at configured loss %)
        force_exit_loss = self.config.risk.force_exit_loss_pct / 100
        if unrealized_pnl_pct < -force_exit_loss:
            return PositionManagementDecision(
                action=PositionAction.EXIT,
                thesis_status=thesis_status,
                exit_reason=f"Stop loss hit: {unrealized_pnl_pct:.1%} < -{force_exit_loss:.1%}",
                exit_pct=1.0,
                unrealized_pnl_pct=unrealized_pnl_pct,
                time_in_position_hours=time_in_position,
                funding_paid=funding_paid,
                confidence=1.0,
                reasoning="Exit: Stop loss triggered",
            )
        
        # 3. Take profit hit
        take_profit = self.config.risk.take_profit_pct / 100
        if unrealized_pnl_pct > take_profit:
            return PositionManagementDecision(
                action=PositionAction.EXIT,
                thesis_status=thesis_status,
                exit_reason=f"Take profit hit: {unrealized_pnl_pct:.1%} > +{take_profit:.1%}",
                exit_pct=1.0,
                unrealized_pnl_pct=unrealized_pnl_pct,
                time_in_position_hours=time_in_position,
                funding_paid=funding_paid,
                confidence=0.85,
                reasoning="Exit: Take profit reached",
            )
        
        # 4. Max holding time exceeded
        max_hold = self.config.risk.max_holding_time_hours
        if time_in_position > max_hold:
            return PositionManagementDecision(
                action=PositionAction.EXIT,
                thesis_status=thesis_status,
                exit_reason=f"Max hold time exceeded: {time_in_position:.1f}h > {max_hold}h",
                exit_pct=1.0,
                unrealized_pnl_pct=unrealized_pnl_pct,
                time_in_position_hours=time_in_position,
                funding_paid=funding_paid,
                confidence=0.7,
                reasoning="Exit: Position held too long",
            )
        
        # 5. High funding cost (eating into profits)
        if funding_paid > position.size_usd * 0.01:  # Funding > 1% of position
            return PositionManagementDecision(
                action=PositionAction.EXIT,
                thesis_status=thesis_status,
                exit_reason=f"High funding cost: ${funding_paid:.2f}",
                exit_pct=1.0,
                unrealized_pnl_pct=unrealized_pnl_pct,
                time_in_position_hours=time_in_position,
                funding_paid=funding_paid,
                confidence=0.75,
                reasoning="Exit: Funding costs too high",
            )
        
        # 6. Drawdown from peak (partial exit to lock in gains)
        if context and context.peak_unrealized_pnl_pct > 0.02:  # Had 2%+ gains
            drawdown_from_peak = context.peak_unrealized_pnl_pct - unrealized_pnl_pct
            if drawdown_from_peak > 0.015:  # Gave back 1.5%
                return PositionManagementDecision(
                    action=PositionAction.EXIT,
                    thesis_status=thesis_status,
                    exit_reason=f"Drawdown from peak: {drawdown_from_peak:.1%}",
                    exit_pct=0.5,  # Partial exit
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    time_in_position_hours=time_in_position,
                    funding_paid=funding_paid,
                    confidence=0.65,
                    reasoning="Partial exit: Protecting gains",
                )
        
        # 7. Thesis weakening + small loss → consider exit
        if thesis_status == ThesisStatus.WEAKENING and unrealized_pnl_pct < -0.01:
            return PositionManagementDecision(
                action=PositionAction.EXIT,
                thesis_status=thesis_status,
                exit_reason="Thesis weakening with unrealized loss",
                exit_pct=1.0,
                unrealized_pnl_pct=unrealized_pnl_pct,
                time_in_position_hours=time_in_position,
                funding_paid=funding_paid,
                confidence=0.6,
                reasoning="Exit: Thesis weakening, cutting losses early",
            )
        
        # =====================================================================
        # HOLD - Thesis intact or weakening with profit
        # =====================================================================
        
        return PositionManagementDecision(
            action=PositionAction.HOLD,
            thesis_status=thesis_status,
            unrealized_pnl_pct=unrealized_pnl_pct,
            time_in_position_hours=time_in_position,
            funding_paid=funding_paid,
            confidence=0.7 if thesis_status == ThesisStatus.INTACT else 0.5,
            reasoning=f"Hold: Thesis {thesis_status.value}, P&L {unrealized_pnl_pct:+.1%}",
        )
    
    # =========================================================================
    # LEVERAGE DECISION
    # =========================================================================
    
    def calculate_leverage(
        self,
        signal_confidence: float,
        stat_result: StatisticalResult,
        market_state: MarketState,
        portfolio_drawdown: float = 0.0,
    ) -> LeverageDecision:
        """
        Calculate appropriate leverage (up to 20x).
        
        Leverage factors:
        1. Signal confidence (higher confidence = higher leverage)
        2. Volatility (higher vol = lower leverage)
        3. Funding rate (high funding = lower leverage)
        4. Portfolio drawdown (in drawdown = lower leverage)
        5. Regime (trending = higher, ranging = lower)
        """
        base = self.config.risk.base_leverage
        max_lev = self.config.risk.max_leverage  # Now 20x
        
        # Start with base leverage
        leverage = base
        
        # 1. Confidence factor (0.3 to 1.5x multiplier)
        conf_factor = 0.3 + (signal_confidence * 1.2)
        leverage *= conf_factor
        
        # 2. Volatility factor
        vol_factor = 1.0
        if stat_result.volatility_regime == "extreme":
            vol_factor = 0.25
        elif stat_result.volatility_regime == "high":
            vol_factor = 0.5
        elif stat_result.volatility_regime == "low":
            vol_factor = 1.5
        leverage *= vol_factor
        
        # 3. Funding factor
        funding_factor = 1.0
        if market_state.funding_rate:
            rate = abs(market_state.funding_rate.rate)
            if rate > 0.001:
                funding_factor = 0.5
            elif rate > 0.0005:
                funding_factor = 0.75
        leverage *= funding_factor
        
        # 4. Drawdown factor
        dd_factor = 1.0
        if portfolio_drawdown > 0.05:
            dd_factor = max(0.3, 1 - (portfolio_drawdown * 5))
        leverage *= dd_factor
        
        # 5. Regime factor
        regime_factor = 1.0
        if stat_result.regime in [Regime.TRENDING_UP, Regime.TRENDING_DOWN]:
            regime_factor = 1.3
        elif stat_result.regime == Regime.CASCADE_RISK:
            regime_factor = 0.3
        elif stat_result.regime in [Regime.SQUEEZE_LONG, Regime.SQUEEZE_SHORT]:
            regime_factor = 0.7
        leverage *= regime_factor
        
        # Apply limits
        leverage = max(1.0, min(leverage, max_lev))
        
        reasoning = (
            f"Base {base}x → "
            f"Conf({conf_factor:.1f}) × Vol({vol_factor:.1f}) × "
            f"Fund({funding_factor:.1f}) × DD({dd_factor:.1f}) × "
            f"Regime({regime_factor:.1f}) = {leverage:.1f}x"
        )
        
        return LeverageDecision(
            leverage=round(leverage, 1),
            reasoning=reasoning,
            confidence_factor=conf_factor,
            volatility_factor=vol_factor,
            funding_factor=funding_factor,
            drawdown_factor=dd_factor,
        )
    
    # =========================================================================
    # CAPITAL ALLOCATION
    # =========================================================================
    
    def calculate_position_size(
        self,
        total_equity: float,
        signal_confidence: float,
        leverage: float,
        stop_loss_pct: float,
        max_positions: int = 5,
        current_positions: int = 0,
    ) -> float:
        """
        Calculate position size in USD.
        
        Uses risk-based sizing:
        - Risk X% of equity per trade
        - Adjusted for confidence and leverage
        - Limited by max position size and available slots
        """
        # Base risk per trade (from config, default 1%)
        risk_pct = self.config.risk.risk_per_trade_pct / 100
        
        # Adjust risk by confidence
        adjusted_risk = risk_pct * signal_confidence
        
        # Risk amount in USD
        risk_amount = total_equity * adjusted_risk
        
        # Size based on stop loss
        if stop_loss_pct > 0:
            size_usd = risk_amount / stop_loss_pct
        else:
            size_usd = total_equity * 0.1  # Default 10% of equity
        
        # Apply leverage
        size_usd *= leverage
        
        # Limit by max position size
        max_size = self.config.trading.max_position_size_usd
        size_usd = min(size_usd, max_size)
        
        # Limit by available slots
        remaining_slots = max(0, max_positions - current_positions)
        if remaining_slots == 0:
            return 0.0
        
        # Divide remaining capacity
        max_total = self.config.trading.max_total_exposure_usd
        per_slot_max = max_total / max_positions
        size_usd = min(size_usd, per_slot_max)
        
        # Minimum viable size
        if size_usd < 50:
            return 0.0
        
        return round(size_usd, 2)
