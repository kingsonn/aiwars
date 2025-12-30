"""HYDRA Core Engine - Orchestrates all layers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

from hydra.core.config import HydraConfig
from hydra.core.types import (
    MarketState,
    Position,
    Signal,
    AgentVote,
    ExecutionPlan,
    Side,
    Regime,
)


class HydraEngine:
    """
    HYDRA Core Engine
    
    Orchestrates the 5-layer trading system:
    1. Market Intelligence Layer
    2. Statistical Reality Layer
    3. Alpha & Behavior Modeling Layer
    4. Risk, Leverage & Capital Brain
    5. Decision & Execution Engine
    
    Each layer can veto the next.
    """
    
    def __init__(self, config: Optional[HydraConfig] = None):
        self.config = config or HydraConfig.load()
        self._running = False
        self._initialized = False
        
        # Layer instances (initialized in setup)
        self._market_intel = None
        self._statistical_engine = None
        self._alpha_engine = None
        self._risk_brain = None
        self._execution_engine = None
        
        # State
        self._market_states: dict[str, MarketState] = {}
        self._positions: dict[str, Position] = {}
        self._current_regime: dict[str, Regime] = {}
        
        # Kill switch state
        self._killed = False
        self._kill_reason = ""
        
        logger.info("HYDRA Engine initialized")
    
    # Public properties to access layers
    @property
    def layer1(self):
        """Layer 1: Market Intelligence."""
        return self._market_intel
    
    @property
    def layer2(self):
        """Layer 2: Statistical Reality Engine."""
        return self._statistical_engine
    
    @property
    def layer3(self):
        """Layer 3: Alpha & Behavior Modeling."""
        return self._alpha_engine
    
    @property
    def layer4(self):
        """Layer 4: Risk, Leverage & Capital Brain."""
        return self._risk_brain
    
    @property
    def layer5(self):
        """Layer 5: Decision & Execution Engine."""
        return self._execution_engine
    
    async def setup(self) -> None:
        """Initialize all layers and connections."""
        if self._initialized:
            return
        
        logger.info("Setting up HYDRA layers...")
        
        # Import here to avoid circular imports
        from hydra.layers.layer1_market_intel import MarketIntelligenceLayer
        from hydra.layers.layer2_statistical import StatisticalRealityEngine
        from hydra.layers.layer3_alpha import AlphaBehaviorEngine
        from hydra.layers.layer4_risk import RiskCapitalBrain
        from hydra.layers.layer5_execution import DecisionExecutionEngine
        
        # Initialize layers
        self._market_intel = MarketIntelligenceLayer(self.config)
        self._statistical_engine = StatisticalRealityEngine(self.config)
        self._alpha_engine = AlphaBehaviorEngine(self.config)
        self._risk_brain = RiskCapitalBrain(self.config)
        self._execution_engine = DecisionExecutionEngine(self.config)
        
        # Setup each layer
        await self._market_intel.setup()
        await self._statistical_engine.setup()
        await self._alpha_engine.setup()
        await self._risk_brain.setup()
        await self._execution_engine.setup()
        
        self._initialized = True
        logger.info("HYDRA layers initialized successfully")
    
    async def start(self) -> None:
        """Start the trading engine."""
        if not self._initialized:
            await self.setup()
        
        if self._running:
            logger.warning("HYDRA is already running")
            return
        
        self._running = True
        self._killed = False
        
        logger.info("Starting HYDRA trading engine...")
        
        # Start data feeds
        await self._market_intel.start_feeds()
        
        # Main trading loop
        await self._run_trading_loop()
    
    async def stop(self) -> None:
        """Stop the trading engine gracefully."""
        logger.info("Stopping HYDRA...")
        self._running = False
        
        if self._market_intel:
            await self._market_intel.stop_feeds()
        
        if self._execution_engine:
            await self._execution_engine.cancel_all_orders()
        
        logger.info("HYDRA stopped")
    
    async def _run_trading_loop(self) -> None:
        """Main trading loop."""
        interval = self.config.system.decision_interval_seconds
        
        while self._running:
            try:
                cycle_start = datetime.now(timezone.utc)
                
                # Check kill switch
                if self._killed:
                    logger.warning(f"HYDRA killed: {self._kill_reason}")
                    await self._flatten_all_positions()
                    await asyncio.sleep(interval)
                    continue
                
                # Run decision cycle for each symbol
                for symbol in self.config.trading.symbols:
                    await self._run_decision_cycle(symbol)
                
                # Sleep until next cycle
                elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
                sleep_time = max(0, interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.exception(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    async def _run_decision_cycle(self, symbol: str) -> None:
        """
        Run complete decision cycle for a symbol.
        
        Flow:
        1. Get market state (Layer 1)
        2. Compute statistical reality (Layer 2)
        3. Generate alpha signals (Layer 3)
        4. Risk check and sizing (Layer 4)
        5. Execute if approved (Layer 5)
        """
        try:
            # LAYER 1: Market Intelligence
            market_state = await self._market_intel.get_market_state(symbol)
            if market_state is None:
                logger.debug(f"No market state for {symbol}")
                return
            
            self._market_states[symbol] = market_state
            
            # LAYER 2: Statistical Reality
            stat_result = await self._statistical_engine.analyze(market_state)
            
            # Check for regime break or danger zone
            if stat_result.regime_break_alert:
                logger.warning(f"Regime break detected for {symbol}")
                self._current_regime[symbol] = Regime.UNKNOWN
                # Don't open new positions during regime breaks
                return
            
            self._current_regime[symbol] = stat_result.regime
            
            # LAYER 3: Alpha & Behavior Modeling
            signals = await self._alpha_engine.generate_signals(
                market_state=market_state,
                stat_result=stat_result,
            )
            
            if not signals:
                return
            
            # Get best signal
            best_signal = max(signals, key=lambda s: s.confidence)
            
            # Minimum confidence check
            if best_signal.confidence < self.config.risk.min_confidence_threshold:
                logger.debug(f"Signal confidence {best_signal.confidence:.2f} below threshold")
                return
            
            # LAYER 4: Risk, Leverage & Capital Brain
            current_position = self._positions.get(symbol)
            
            risk_decision = await self._risk_brain.evaluate(
                signal=best_signal,
                market_state=market_state,
                stat_result=stat_result,
                current_position=current_position,
                all_positions=self._positions,
            )
            
            # Check for veto
            if risk_decision.veto:
                logger.info(f"Risk brain vetoed trade: {risk_decision.veto_reason}")
                return
            
            # Check for kill switch trigger
            if risk_decision.trigger_kill_switch:
                await self._trigger_kill_switch(risk_decision.kill_reason)
                return
            
            # LAYER 5: Decision & Execution
            # Multi-agent voting
            votes = await self._execution_engine.collect_votes(
                signal=best_signal,
                market_state=market_state,
                stat_result=stat_result,
                risk_decision=risk_decision,
            )
            
            # Check if trade is approved
            approved, execution_plan = self._execution_engine.evaluate_votes(
                votes=votes,
                signal=best_signal,
                risk_decision=risk_decision,
            )
            
            if not approved:
                logger.debug("Trade not approved by voting agents")
                return
            
            # Execute trade
            if execution_plan:
                await self._execution_engine.execute(execution_plan)
                logger.info(f"Executed trade for {symbol}: {execution_plan.target_side.value}")
            
        except Exception as e:
            logger.exception(f"Error in decision cycle for {symbol}: {e}")
    
    async def _trigger_kill_switch(self, reason: str) -> None:
        """Trigger kill switch - flatten all positions."""
        logger.error(f"KILL SWITCH TRIGGERED: {reason}")
        self._killed = True
        self._kill_reason = reason
        await self._flatten_all_positions()
    
    async def _flatten_all_positions(self) -> None:
        """Flatten all open positions."""
        logger.warning("Flattening all positions...")
        for symbol, position in self._positions.items():
            if position.is_open:
                await self._execution_engine.close_position(symbol)
        logger.info("All positions flattened")
    
    def get_status(self) -> dict:
        """Get current engine status."""
        return {
            "running": self._running,
            "initialized": self._initialized,
            "killed": self._killed,
            "kill_reason": self._kill_reason,
            "symbols": self.config.trading.symbols,
            "positions": {s: p.side.value for s, p in self._positions.items() if p.is_open},
            "regimes": {s: r.name for s, r in self._current_regime.items()},
        }
    
    async def manual_kill(self, reason: str = "Manual kill") -> None:
        """Manually trigger kill switch."""
        await self._trigger_kill_switch(reason)
    
    async def resume(self) -> None:
        """Resume trading after kill switch."""
        if self._killed:
            logger.info("Resuming HYDRA trading...")
            self._killed = False
            self._kill_reason = ""
