"""
HYDRA Trading Engine

Main orchestrator integrating all 5 layers:
- Layer 1: Market Intelligence (data)
- Layer 2: Statistical Reality (tradability)
- Layer 3: Alpha Generation (signals)
- Layer 4: Risk Brain (sizing, leverage, stops)
- Layer 5: Execution (orders)

Handles:
- Entry decision flow
- Position management loop
- Exit execution
- Thesis tracking
- Kill switches

Per HYDRA_SPEC_TRADING.md
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum
from loguru import logger

from hydra.core.config import HydraConfig
from hydra.core.types import (
    MarketState, Signal, Position, Side, Regime,
    Order, OrderType, PositionAction, ThesisStatus
)
from hydra.layers.layer1_market_intel import MarketIntelligenceLayer, PERMITTED_PAIRS
from hydra.layers.layer2_statistical import (
    StatisticalRealityEngine, StatisticalResult, TradabilityStatus
)
from hydra.layers.layer3_alpha.signals import BehavioralSignalGenerator
from hydra.layers.layer4_risk import RiskCapitalBrain, RiskDecision, PortfolioState


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_POSITIONS = 5
MAX_POSITION_SIZE_USD = 10_000
MAX_TOTAL_EXPOSURE_USD = 50_000
MIN_POSITION_SIZE_USD = 100

MIN_SIGNAL_CONFIDENCE = 0.6
MIN_ML_SCORE = 0.6

DECISION_INTERVAL_SECONDS = 30
ORDER_TIMEOUT_SECONDS = 120

# Holding times by signal source (hours)
BASE_HOLDING_TIMES = {
    "FUNDING_SQUEEZE": 4,
    "LIQUIDATION_REVERSAL": 2,
    "OI_DIVERGENCE": 6,
    "CROWDING_FADE": 8,
    "FUNDING_CARRY": 24,
}


# =============================================================================
# THESIS HEALTH TRACKING
# =============================================================================

@dataclass
class PositionThesis:
    """Tracks the original thesis for a position."""
    signal_source: str
    thesis: str
    entry_time: datetime
    entry_price: float
    entry_funding: float
    entry_oi: float
    max_holding_hours: float
    stop_loss: float
    take_profit: float
    
    # Tracking
    health: float = 1.0
    status: ThesisStatus = ThesisStatus.INTACT
    partial_exits: int = 0


def evaluate_thesis_health(
    position: Position,
    thesis: PositionThesis,
    market_state: MarketState,
    stat_result: StatisticalResult,
) -> tuple[float, ThesisStatus]:
    """
    Evaluate if the original trade thesis is still valid.
    
    Returns: (health 0-1, status)
    """
    health = 1.0
    
    source = thesis.signal_source
    
    if "FUNDING_SQUEEZE" in source:
        funding = market_state.funding_rate.rate if market_state.funding_rate else 0
        
        if position.side == Side.SHORT:
            # We shorted because longs were paying
            if funding < 0:  # Funding flipped negative
                return 0.0, ThesisStatus.BROKEN
            elif funding < 0.0003:  # Funding normalized
                health = 0.5
        
        elif position.side == Side.LONG:
            # We longed because shorts were paying
            if funding > 0:  # Funding flipped positive
                return 0.0, ThesisStatus.BROKEN
            elif funding > -0.0003:
                health = 0.5
    
    elif "LIQUIDATION_REVERSAL" in source:
        pnl_pct = position.unrealized_pnl_pct
        if pnl_pct > 0.015:  # Got expected move
            health = 0.4  # Take profit zone
        elif pnl_pct < -0.01:  # Going against us
            health = 0.3
    
    elif "CROWDING_FADE" in source:
        # Would need L/S ratio data
        pass
    
    elif "OI_DIVERGENCE" in source:
        # Check if divergence still present
        oi_delta = market_state.open_interest.delta_pct if market_state.open_interest else 0
        if abs(oi_delta) < 0.01:  # OI normalized
            health = 0.6
    
    elif "FUNDING_CARRY" in source:
        # Check if still ranging with good funding
        if stat_result.regime != Regime.RANGING:
            health = 0.4
        if stat_result.volatility_regime in ["high", "extreme"]:
            health = 0.3
    
    # Time decay
    hours_held = (datetime.now(timezone.utc) - thesis.entry_time).total_seconds() / 3600
    time_ratio = hours_held / thesis.max_holding_hours
    if time_ratio > 0.8:
        health *= 0.7
    
    # Determine status
    if health <= 0.2:
        status = ThesisStatus.BROKEN
    elif health <= 0.5:
        status = ThesisStatus.WEAKENING
    else:
        status = ThesisStatus.INTACT
    
    return max(0.0, min(1.0, health)), status


def calculate_max_holding_time(
    signal_source: str,
    stat_result: StatisticalResult,
) -> float:
    """Calculate max holding time based on signal type and conditions."""
    base_time = BASE_HOLDING_TIMES.get(signal_source, 4)
    
    if stat_result.regime == Regime.HIGH_VOLATILITY:
        base_time *= 0.5
    elif stat_result.regime in [Regime.TRENDING_UP, Regime.TRENDING_DOWN]:
        base_time *= 1.5
    
    return base_time


def calculate_stop_loss(
    signal: Signal,
    price: float,
    stat_result: StatisticalResult,
) -> float:
    """Calculate stop-loss price."""
    eae = signal.expected_adverse_excursion
    
    # Adjust for volatility
    if stat_result.volatility_regime == "high":
        eae *= 1.3
    elif stat_result.volatility_regime == "extreme":
        eae *= 1.5
    
    # Bounds
    eae = max(0.01, min(eae, 0.10))
    
    if signal.side == Side.LONG:
        return price * (1 - eae)
    else:
        return price * (1 + eae)


def calculate_take_profit(
    signal: Signal,
    price: float,
    stop_loss: float,
) -> float:
    """Calculate take-profit price with 2:1 R:R."""
    risk = abs(price - stop_loss)
    reward = risk * 2
    
    if signal.side == Side.LONG:
        return price + reward
    else:
        return price - reward


# =============================================================================
# ENTRY CHECKER
# =============================================================================

def should_enter(
    symbol: str,
    signal: Signal,
    stat_result: StatisticalResult,
    risk_decision: RiskDecision,
    positions: Dict[str, Position],
    kill_switch_active: bool,
) -> tuple[bool, str]:
    """
    Check if a new position should be opened.
    
    Returns: (should_enter, reason if not)
    """
    # 1. No existing position
    if symbol in positions and positions[symbol].is_open:
        return False, "Already have position"
    
    # 2. Layer 2 allows entry
    if stat_result.trading_decision != TradabilityStatus.ALLOW:
        return False, f"Layer 2: {stat_result.trading_decision.value}"
    
    # 3. Signal exists with sufficient confidence
    if signal is None:
        return False, "No signal"
    if signal.confidence < MIN_SIGNAL_CONFIDENCE:
        return False, f"Low confidence: {signal.confidence:.2f}"
    
    # 4. ML score (if available)
    ml_score = signal.metadata.get("ml_score", 1.0)
    if ml_score < MIN_ML_SCORE:
        return False, f"ML score too low: {ml_score:.2f}"
    
    # 5. Risk brain approved
    if not risk_decision.approved:
        return False, f"Risk rejected: {risk_decision.veto_reason}"
    if risk_decision.veto:
        return False, f"Risk veto: {risk_decision.veto_reason}"
    
    # 6. Size is meaningful
    if risk_decision.recommended_size_usd < MIN_POSITION_SIZE_USD:
        return False, f"Size too small: ${risk_decision.recommended_size_usd:.0f}"
    
    # 7. Portfolio limits
    num_positions = sum(1 for p in positions.values() if p.is_open)
    if num_positions >= MAX_POSITIONS:
        return False, f"Max positions reached: {num_positions}"
    
    total_exposure = sum(abs(p.size_usd) for p in positions.values() if p.is_open)
    if total_exposure >= MAX_TOTAL_EXPOSURE_USD:
        return False, f"Max exposure reached: ${total_exposure:.0f}"
    
    # 8. No kill switch
    if kill_switch_active:
        return False, "Kill switch active"
    
    return True, "All checks passed"


# =============================================================================
# POSITION MANAGER
# =============================================================================

@dataclass
class PositionManagementResult:
    """Result of position management check."""
    action: PositionAction
    reason: str
    exit_fraction: float = 1.0
    urgency: str = "normal"  # normal, high, emergency


def manage_position(
    position: Position,
    thesis: PositionThesis,
    market_state: MarketState,
    stat_result: StatisticalResult,
    kill_switch_active: bool,
) -> PositionManagementResult:
    """
    Evaluate a position and determine action.
    
    Priority order per HYDRA_SPEC_TRADING.md:
    1. Kill switch → immediate market exit
    2. Layer 2 BLOCK → immediate market exit
    3. Liquidation distance < 5% → immediate exit
    4. Thesis broken → limit exit
    5. Stop-loss hit → stop triggers
    6. Take-profit hit → limit exit
    7. Max holding time → limit exit
    8. Thesis weakening → partial exit
    9. Regime against position → partial exit
    10. High funding burden → limit exit
    """
    now = datetime.now(timezone.utc)
    
    # === EMERGENCY EXITS ===
    
    # 1. Kill switch
    if kill_switch_active:
        return PositionManagementResult(
            action=PositionAction.FORCE_EXIT,
            reason="Kill switch active",
            exit_fraction=1.0,
            urgency="emergency"
        )
    
    # 2. Layer 2 BLOCK
    if stat_result.trading_decision == TradabilityStatus.BLOCK:
        return PositionManagementResult(
            action=PositionAction.FORCE_EXIT,
            reason=f"Layer 2 BLOCK: {stat_result.decision_reasons}",
            exit_fraction=1.0,
            urgency="emergency"
        )
    
    # 3. Liquidation risk
    if position.distance_to_liquidation < 0.05:
        return PositionManagementResult(
            action=PositionAction.FORCE_EXIT,
            reason=f"Liquidation risk: {position.distance_to_liquidation:.1%}",
            exit_fraction=1.0,
            urgency="emergency"
        )
    
    # === THESIS CHECK ===
    
    health, status = evaluate_thesis_health(position, thesis, market_state, stat_result)
    thesis.health = health
    thesis.status = status
    
    # 4. Thesis broken
    if health < 0.2:
        return PositionManagementResult(
            action=PositionAction.EXIT,
            reason="Thesis broken",
            exit_fraction=1.0,
            urgency="high"
        )
    
    # === PRICE LEVELS ===
    
    current_price = market_state.price
    
    # 5. Stop-loss (handled by exchange, but check here too)
    if position.side == Side.LONG and current_price <= thesis.stop_loss:
        return PositionManagementResult(
            action=PositionAction.EXIT,
            reason=f"Stop-loss hit: ${current_price:.2f} <= ${thesis.stop_loss:.2f}",
            exit_fraction=1.0,
            urgency="high"
        )
    if position.side == Side.SHORT and current_price >= thesis.stop_loss:
        return PositionManagementResult(
            action=PositionAction.EXIT,
            reason=f"Stop-loss hit: ${current_price:.2f} >= ${thesis.stop_loss:.2f}",
            exit_fraction=1.0,
            urgency="high"
        )
    
    # 6. Take-profit
    if position.side == Side.LONG and current_price >= thesis.take_profit:
        return PositionManagementResult(
            action=PositionAction.EXIT,
            reason=f"Take-profit hit: ${current_price:.2f}",
            exit_fraction=1.0,
            urgency="normal"
        )
    if position.side == Side.SHORT and current_price <= thesis.take_profit:
        return PositionManagementResult(
            action=PositionAction.EXIT,
            reason=f"Take-profit hit: ${current_price:.2f}",
            exit_fraction=1.0,
            urgency="normal"
        )
    
    # === TIME CHECK ===
    
    # 7. Max holding time
    hours_held = (now - thesis.entry_time).total_seconds() / 3600
    if hours_held > thesis.max_holding_hours:
        return PositionManagementResult(
            action=PositionAction.EXIT,
            reason=f"Max holding time: {hours_held:.1f}h > {thesis.max_holding_hours:.1f}h",
            exit_fraction=1.0,
            urgency="normal"
        )
    
    # === PARTIAL EXITS ===
    
    # 8. Thesis weakening (only one partial exit allowed)
    if health < 0.5 and thesis.partial_exits == 0:
        thesis.partial_exits += 1
        return PositionManagementResult(
            action=PositionAction.EXIT,
            reason=f"Thesis weakening: health={health:.2f}",
            exit_fraction=0.5,
            urgency="normal"
        )
    
    # 9. Regime against position
    if position.side == Side.LONG and stat_result.regime == Regime.TRENDING_DOWN:
        if thesis.partial_exits == 0:
            thesis.partial_exits += 1
            return PositionManagementResult(
                action=PositionAction.EXIT,
                reason="Regime bearish",
                exit_fraction=0.5,
                urgency="normal"
            )
    
    if position.side == Side.SHORT and stat_result.regime == Regime.TRENDING_UP:
        if thesis.partial_exits == 0:
            thesis.partial_exits += 1
            return PositionManagementResult(
                action=PositionAction.EXIT,
                reason="Regime bullish",
                exit_fraction=0.5,
                urgency="normal"
            )
    
    # === FUNDING CHECK ===
    
    # 10. High funding burden
    if market_state.funding_rate:
        funding = market_state.funding_rate.rate
        
        if position.side == Side.LONG and funding > 0.001 and health < 0.7:
            return PositionManagementResult(
                action=PositionAction.EXIT,
                reason=f"High funding burden: {funding*100:.3f}%",
                exit_fraction=1.0,
                urgency="normal"
            )
        
        if position.side == Side.SHORT and funding < -0.001 and health < 0.7:
            return PositionManagementResult(
                action=PositionAction.EXIT,
                reason=f"High funding burden: {funding*100:.3f}%",
                exit_fraction=1.0,
                urgency="normal"
            )
    
    # === HOLD ===
    
    return PositionManagementResult(
        action=PositionAction.HOLD,
        reason=f"Holding: health={health:.2f}, pnl={position.unrealized_pnl_pct:.2%}",
        exit_fraction=0.0,
        urgency="normal"
    )


# =============================================================================
# TRADING ENGINE
# =============================================================================

class TradingEngine:
    """
    Main trading engine orchestrating all layers.
    
    Flow:
    1. Refresh market data (Layer 1)
    2. Run statistical analysis (Layer 2)
    3. Generate signals (Layer 3)
    4. Evaluate through risk (Layer 4)
    5. Execute trades (Layer 5)
    6. Manage positions
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        
        # Layers
        self.layer1: Optional[MarketIntelligence] = None
        self.layer2: Optional[StatisticalRealityEngine] = None
        self.layer3: Optional[BehavioralSignalGenerator] = None
        self.layer4: Optional[RiskCapitalBrain] = None
        
        # State
        self.positions: Dict[str, Position] = {}
        self.theses: Dict[str, PositionThesis] = {}
        self.kill_switch_active = False
        self.kill_switch_reason = ""
        self.kill_switch_time: Optional[datetime] = None
        
        # Tracking
        self._running = False
        self._last_decision_time: Dict[str, datetime] = {}
        
        logger.info("Trading Engine initialized")
    
    async def initialize(self):
        """Initialize all layers."""
        logger.info("Initializing Trading Engine layers...")
        
        # Layer 1: Market Intelligence
        self.layer1 = MarketIntelligenceLayer()
        await self.layer1.initialize()
        logger.info("Layer 1 (Market Intelligence) ready")
        
        # Layer 2: Statistical Reality
        self.layer2 = StatisticalRealityEngine(self.config, use_ml_regime=True)
        await self.layer2.setup()
        logger.info("Layer 2 (Statistical Reality) ready")
        
        # Layer 3: Alpha Generation
        self.layer3 = BehavioralSignalGenerator()
        logger.info("Layer 3 (Alpha Generation) ready")
        
        # Layer 4: Risk Brain
        self.layer4 = RiskCapitalBrain(self.config)
        await self.layer4.setup()
        logger.info("Layer 4 (Risk Brain) ready")
        
        logger.info("All layers initialized")
    
    async def shutdown(self):
        """Shutdown all layers."""
        logger.info("Shutting down Trading Engine...")
        
        if self.layer1:
            await self.layer1.close()
        
        self._running = False
        logger.info("Trading Engine shutdown complete")
    
    async def run_decision_cycle(self, symbol: str) -> Dict[str, Any]:
        """
        Run a complete decision cycle for a symbol.
        
        Returns dict with decision details.
        """
        result = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc),
            "action": None,
            "signal": None,
            "stat_result": None,
            "risk_decision": None,
            "position_action": None,
        }
        
        try:
            # === LAYER 1: Get Market Data ===
            await self.layer1.refresh_symbol(symbol)
            market_state = self.layer1.get_market_state(symbol)
            
            if market_state is None or market_state.price == 0:
                logger.warning(f"No market data for {symbol}")
                return result
            
            # === LAYER 2: Statistical Analysis ===
            stat_result = await self.layer2.analyze(market_state)
            result["stat_result"] = stat_result
            
            # Check for kill switch conditions
            self._check_kill_switch(market_state, stat_result)
            
            # === POSITION MANAGEMENT ===
            if symbol in self.positions and self.positions[symbol].is_open:
                pos_result = await self._manage_existing_position(
                    symbol, market_state, stat_result
                )
                result["position_action"] = pos_result
                return result
            
            # === NEW ENTRY DECISION ===
            
            # Skip if blocked or restricted
            if stat_result.trading_decision != TradabilityStatus.ALLOW:
                logger.debug(f"{symbol}: Trading {stat_result.trading_decision.value}")
                return result
            
            # === LAYER 3: Signal Generation ===
            signals = self.layer3.generate_signals(
                market_state=market_state,
                stat_result=stat_result,
                long_short_ratio=1.0,  # TODO: Get from data provider
            )
            
            if not signals:
                logger.debug(f"{symbol}: No signals generated")
                return result
            
            # Take best signal
            signal = signals[0]
            result["signal"] = signal
            
            # === LAYER 4: Risk Evaluation ===
            risk_decision = await self.layer4.evaluate(
                signal=signal,
                market_state=market_state,
                stat_result=stat_result,
                current_position=self.positions.get(symbol),
                all_positions=self.positions,
            )
            result["risk_decision"] = risk_decision
            
            # === ENTRY CHECK ===
            can_enter, reason = should_enter(
                symbol=symbol,
                signal=signal,
                stat_result=stat_result,
                risk_decision=risk_decision,
                positions=self.positions,
                kill_switch_active=self.kill_switch_active,
            )
            
            if can_enter:
                result["action"] = "ENTER"
                logger.info(
                    f"ENTRY SIGNAL: {symbol} {signal.side.value} "
                    f"conf={signal.confidence:.2f} size=${risk_decision.recommended_size_usd:.0f}"
                )
                
                # Create thesis for tracking
                thesis = PositionThesis(
                    signal_source=signal.source,
                    thesis=signal.metadata.get("thesis", ""),
                    entry_time=datetime.now(timezone.utc),
                    entry_price=market_state.price,
                    entry_funding=market_state.funding_rate.rate if market_state.funding_rate else 0,
                    entry_oi=market_state.open_interest.open_interest_usd if market_state.open_interest else 0,
                    max_holding_hours=risk_decision.max_holding_time_hours,
                    stop_loss=risk_decision.stop_loss_price,
                    take_profit=risk_decision.take_profit_price,
                )
                self.theses[symbol] = thesis
            else:
                logger.debug(f"{symbol}: Entry rejected - {reason}")
            
        except Exception as e:
            logger.exception(f"Error in decision cycle for {symbol}: {e}")
        
        return result
    
    async def _manage_existing_position(
        self,
        symbol: str,
        market_state: MarketState,
        stat_result: StatisticalResult,
    ) -> PositionManagementResult:
        """Manage an existing position."""
        position = self.positions[symbol]
        thesis = self.theses.get(symbol)
        
        if thesis is None:
            # No thesis tracked, exit position
            return PositionManagementResult(
                action=PositionAction.EXIT,
                reason="No thesis tracked",
                exit_fraction=1.0,
            )
        
        result = manage_position(
            position=position,
            thesis=thesis,
            market_state=market_state,
            stat_result=stat_result,
            kill_switch_active=self.kill_switch_active,
        )
        
        if result.action in [PositionAction.EXIT, PositionAction.FORCE_EXIT]:
            logger.info(
                f"EXIT SIGNAL: {symbol} {position.side.value} "
                f"fraction={result.exit_fraction:.0%} reason={result.reason}"
            )
        
        return result
    
    def _check_kill_switch(
        self,
        market_state: MarketState,
        stat_result: StatisticalResult,
    ):
        """Check and update kill switch status."""
        # Cooldown check
        if self.kill_switch_active and self.kill_switch_time:
            cooldown = (datetime.now(timezone.utc) - self.kill_switch_time).total_seconds()
            if cooldown > 3600:  # 1 hour cooldown
                self.kill_switch_active = False
                self.kill_switch_reason = ""
                logger.info("Kill switch cooldown expired, trading enabled")
        
        if self.kill_switch_active:
            return
        
        # Check conditions
        conditions = [
            (stat_result.cascade_probability > 0.7, 
             f"Cascade probability {stat_result.cascade_probability:.1%}"),
            (stat_result.regime_break_alert and stat_result.volatility_regime == "extreme",
             "Regime break with extreme volatility"),
            (market_state.funding_rate and abs(market_state.funding_rate.rate) > 0.005,
             f"Funding spike {abs(market_state.funding_rate.rate)*100:.2f}%"),
        ]
        
        for condition, reason in conditions:
            if condition:
                self.kill_switch_active = True
                self.kill_switch_reason = reason
                self.kill_switch_time = datetime.now(timezone.utc)
                logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
                break
    
    async def run_loop(self, interval_seconds: int = 30):
        """
        Main trading loop.
        
        Runs decision cycle for all permitted pairs.
        """
        logger.info(f"Starting trading loop (interval: {interval_seconds}s)")
        self._running = True
        
        while self._running:
            cycle_start = datetime.now(timezone.utc)
            
            for symbol in PERMITTED_PAIRS:
                if not self._running:
                    break
                
                try:
                    result = await self.run_decision_cycle(symbol)
                    
                    if result.get("action"):
                        logger.info(f"{symbol}: {result['action']}")
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
            
            # Wait for next interval
            elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            sleep_time = max(0, interval_seconds - elapsed)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        logger.info("Trading loop stopped")
    
    def stop(self):
        """Stop the trading loop."""
        self._running = False
