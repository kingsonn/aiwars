"""
HYDRA Paper Trading Engine

Main engine for paper trading that:
- Connects to real market data feeds
- Runs the full HYDRA decision pipeline
- Executes trades on the paper portfolio
- Tracks performance in real-time

DUAL MODE OPERATION:
- Entry Mode: For pairs WITHOUT positions (can LONG, SHORT, or FLAT)
- Position Management Mode: For pairs WITH positions (can only HOLD or EXIT)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable
from loguru import logger

from hydra.core.config import HydraConfig, PERMITTED_PAIRS, PAIR_DISPLAY_NAMES
from hydra.core.types import (
    MarketState, Signal, Side, Position,
    PositionAction, ThesisStatus, PositionManagementDecision, EntryDecision,
)
from hydra.core.engine import HydraEngine
from hydra.core.position_manager import PositionManager, OperatingMode
from hydra.paper_trading.portfolio import Portfolio
from hydra.paper_trading.dashboard import TradingDashboard
from hydra.layers.layer2_statistical import TradingDecision


class PaperTradingEngine:
    """
    Paper trading engine that simulates live trading.
    
    Uses real market data but executes on a simulated portfolio.
    Tracks all trades, P&L, and performance metrics.
    """
    
    def __init__(
        self,
        config: HydraConfig,
        initial_balance: float = 1000.0,
        data_dir: str = "./data/paper_trading",
    ):
        self.config = config
        self.config.trading.trading_mode = "paper"
        
        # Core engine for market data and signals
        self.hydra = HydraEngine(config)
        
        # Position manager for dual-mode operation
        self.position_manager = PositionManager(config)
        
        # Paper portfolio
        self.portfolio = Portfolio(
            initial_balance=initial_balance,
            data_dir=data_dir,
        )
        
        # Dashboard
        self.dashboard = TradingDashboard(self.portfolio)
        
        # State
        self._running = False
        self._last_snapshot_time: Optional[datetime] = None
        self._snapshot_interval = timedelta(minutes=5)
        
        # Price cache
        self._current_prices: dict[str, float] = {}
        
        # Callbacks
        self._on_trade: Optional[Callable] = None
        self._on_signal: Optional[Callable] = None
        
        logger.info(f"Paper Trading Engine initialized with ${initial_balance:.2f}")
        logger.info(f"Max leverage: {config.risk.max_leverage}x")
    
    async def start(self) -> None:
        """Start the paper trading engine."""
        logger.info("Starting Paper Trading Engine...")
        
        # Try to load existing state
        self.portfolio.load()
        
        # Setup HYDRA engine
        await self.hydra.setup()
        
        self._running = True
        
        # Start main loop
        try:
            await self._main_loop()
        except asyncio.CancelledError:
            logger.info("Paper trading cancelled")
        except Exception as e:
            logger.error(f"Paper trading error: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the paper trading engine."""
        logger.info("Stopping Paper Trading Engine...")
        self._running = False
        
        # Save portfolio state
        self.portfolio.save()
        
        # Take final snapshot
        self.portfolio.take_snapshot()
        
        # Stop HYDRA
        await self.hydra.stop()
        
        # Print final summary
        self.dashboard.print_summary()
    
    async def _main_loop(self) -> None:
        """Main trading loop."""
        logger.info("Paper trading loop started")
        
        decision_interval = self.config.system.decision_interval_seconds
        
        while self._running:
            try:
                cycle_start = datetime.now(timezone.utc)
                
                # Process each symbol
                for symbol in self.config.trading.symbols:
                    await self._process_symbol(symbol)
                
                # Update prices and P&L
                self.portfolio.update_prices(self._current_prices)
                
                # Take periodic snapshots
                if self._should_take_snapshot():
                    self.portfolio.take_snapshot()
                    self._last_snapshot_time = datetime.now(timezone.utc)
                
                # Save state periodically
                if len(self.portfolio.snapshots) % 10 == 0:
                    self.portfolio.save()
                
                # Wait for next cycle
                elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
                sleep_time = max(0, decision_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_symbol(self, symbol: str) -> None:
        """
        Process a single symbol with DUAL MODE operation.
        
        Entry Mode (no position): Can LONG, SHORT, or FLAT
        Management Mode (has position): Can only HOLD or EXIT
        """
        try:
            # Get market state from Layer 1
            market_state = await self.hydra.layer1.get_market_state(symbol)
            if not market_state:
                return
            
            # Update price cache
            self._current_prices[symbol] = market_state.price
            
            # Get current position
            current_position = self._get_position_as_type(symbol)
            
            # Determine operating mode
            mode = self.position_manager.get_operating_mode(symbol, current_position)
            
            # Get statistical analysis from Layer 2
            stat_result = await self.hydra.layer2.analyze(market_state)
            
            # === LAYER 2 SAFETY CHECK ===
            # For positions: BLOCK = FORCE EXIT
            force_exit, force_reason = self.position_manager.check_layer2_safety(
                stat_result, current_position
            )
            
            if force_exit and current_position:
                logger.warning(f"🚨 FORCE EXIT {symbol}: {force_reason}")
                self.portfolio.close_position(symbol, market_state.price, "force_exit_l2_block")
                self.position_manager.clear_position_context(symbol)
                return
            
            # Route to appropriate mode handler
            if mode == OperatingMode.MANAGEMENT:
                await self._process_position_management(
                    symbol, market_state, stat_result, current_position
                )
            else:
                await self._process_entry_evaluation(
                    symbol, market_state, stat_result
                )
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    async def _process_position_management(
        self,
        symbol: str,
        market_state: MarketState,
        stat_result,
        position: Position,
    ) -> None:
        """
        POSITION MANAGEMENT MODE - Can only HOLD or EXIT
        
        Layer 2: Already checked (BLOCK = force exit)
        Layer 3: Check thesis validity
        Layer 4: Exit decision based on PnL, time, funding, etc.
        Layer 5: Execute exit with reduce-only orders
        """
        display_name = PAIR_DISPLAY_NAMES.get(symbol, symbol)
        
        # Get position context
        context = self.position_manager.get_position_context(symbol)
        
        # === LAYER 3: THESIS CHECK ===
        thesis_status = self.position_manager.check_thesis_validity(
            position, market_state, stat_result, context
        )
        
        # === LAYER 4: EXIT DECISION ===
        decision = self.position_manager.evaluate_exit_decision(
            position, market_state, stat_result, thesis_status, context
        )
        
        # Log status
        pnl_str = f"{decision.unrealized_pnl_pct:+.2%}"
        time_str = f"{decision.time_in_position_hours:.1f}h"
        logger.debug(
            f"[{display_name}] {position.side.value.upper()} | "
            f"P&L: {pnl_str} | Time: {time_str} | "
            f"Thesis: {thesis_status.value} | Action: {decision.action.value}"
        )
        
        # === EXECUTE DECISION ===
        if decision.action in [PositionAction.EXIT, PositionAction.FORCE_EXIT]:
            logger.info(
                f"📤 EXIT {display_name}: {decision.exit_reason} "
                f"(P&L: {pnl_str}, {time_str})"
            )
            
            # Execute exit
            self.portfolio.close_position(
                symbol, 
                market_state.price, 
                decision.exit_reason,
                partial_pct=decision.exit_pct
            )
            
            # Clear context if full exit
            if decision.exit_pct >= 0.99:
                self.position_manager.clear_position_context(symbol)
        
        # HOLD - update context tracking
        elif decision.action == PositionAction.HOLD:
            if context:
                # Update peak tracking
                if position.unrealized_pnl > context.peak_unrealized_pnl:
                    context.peak_unrealized_pnl = position.unrealized_pnl
                    context.peak_unrealized_pnl_pct = position.unrealized_pnl_pct
    
    async def _process_entry_evaluation(
        self,
        symbol: str,
        market_state: MarketState,
        stat_result,
    ) -> None:
        """
        ENTRY MODE - Can LONG, SHORT, or stay FLAT
        
        Layer 2: Gate check (BLOCK = no entry)
        Layer 3: Generate entry signals
        Layer 4: Size and risk check
        Layer 5: Multi-agent vote and execute
        """
        display_name = PAIR_DISPLAY_NAMES.get(symbol, symbol)
        
        # === LAYER 2: GATE CHECK ===
        if stat_result.trading_decision == TradingDecision.BLOCK:
            logger.debug(f"[{display_name}] L2 BLOCK - no entry allowed")
            return
        
        # Get news for LLM context
        recent_news = []
        if hasattr(self.hydra, 'layer1') and self.hydra.layer1:
            news_sentiment = self.hydra.layer1.get_news_sentiment(symbol)
            if news_sentiment and news_sentiment.breaking_news:
                recent_news = [n.title for n in news_sentiment.breaking_news[:5]]
        
        # === LAYER 3: ALPHA GENERATION ===
        signals = await self.hydra.layer3.generate_signals(
            market_state, stat_result, None, recent_news
        )
        
        if not signals:
            return
        
        # Get best signal
        best_signal = max(signals, key=lambda s: s.confidence)
        
        if self._on_signal:
            self._on_signal(symbol, best_signal)
        
        # Minimum confidence check
        if best_signal.confidence < self.config.risk.min_confidence_threshold:
            logger.debug(
                f"[{display_name}] Signal confidence {best_signal.confidence:.1%} "
                f"below threshold {self.config.risk.min_confidence_threshold:.1%}"
            )
            return
        
        # === CALCULATE LEVERAGE ===
        portfolio_dd = self.portfolio.current_drawdown
        leverage_decision = self.position_manager.calculate_leverage(
            best_signal.confidence,
            stat_result,
            market_state,
            portfolio_dd,
        )
        
        # Apply L2 restriction
        if stat_result.trading_decision == TradingDecision.RESTRICT:
            # Reduce leverage for restricted conditions
            leverage_decision.leverage = min(
                leverage_decision.leverage,
                self.config.risk.base_leverage
            )
        
        # === CALCULATE POSITION SIZE ===
        all_positions = self._get_all_positions_as_type()
        stop_loss_pct = best_signal.expected_adverse_excursion or 0.02
        
        size_usd = self.position_manager.calculate_position_size(
            total_equity=self.portfolio.total_equity,
            signal_confidence=best_signal.confidence,
            leverage=leverage_decision.leverage,
            stop_loss_pct=stop_loss_pct,
            max_positions=self.config.trading.max_positions,
            current_positions=len(all_positions),
        )
        
        if size_usd < 50:
            logger.debug(f"[{display_name}] Position size too small: ${size_usd:.2f}")
            return
        
        # Convert to Signal type for Layer 4/5
        signal = Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            side=best_signal.direction,
            confidence=best_signal.confidence,
            expected_return=0.02,
            expected_adverse_excursion=best_signal.expected_adverse_excursion,
            holding_period_minutes=int(best_signal.expected_holding_period_hours * 60),
            source="hydra_alpha",
            regime=stat_result.regime,
            metadata={'thesis': best_signal.thesis},
        )
        
        # === LAYER 4: RISK EVALUATION ===
        current_position = self._get_position_as_type(symbol)
        
        risk_decision = await self.hydra.layer4.evaluate(
            signal, market_state, stat_result, current_position, all_positions
        )
        
        # Check kill switch
        if risk_decision.trigger_kill_switch:
            logger.warning(f"🚨 KILL SWITCH: {risk_decision.kill_reason}")
            await self._close_all_positions(market_state.price, "kill_switch")
            return
        
        # Check if approved
        if not risk_decision.approved:
            logger.debug(f"[{display_name}] L4 rejected: {risk_decision.veto_reason}")
            return
        
        # === LAYER 5: MULTI-AGENT VOTING ===
        votes = await self.hydra.layer5.collect_votes(
            signal, market_state, stat_result, risk_decision
        )
        
        approved, execution_plan = self.hydra.layer5.evaluate_votes(
            votes, signal, risk_decision
        )
        
        if not approved:
            logger.debug(f"[{display_name}] L5 voting rejected trade")
            return
        
        # === EXECUTE ENTRY ===
        logger.info(
            f"📥 ENTRY {display_name}: {signal.side.value.upper()} "
            f"${size_usd:.0f} @ {leverage_decision.leverage:.1f}x | "
            f"Confidence: {signal.confidence:.1%}"
        )
        
        await self._execute_trade(
            symbol=symbol,
            side=signal.side,
            size_usd=size_usd,
            price=market_state.price,
            leverage=leverage_decision.leverage,
            signal=signal,
        )
    
    async def _execute_trade(
        self,
        symbol: str,
        side: Side,
        size_usd: float,
        price: float,
        leverage: float,
        signal: Signal,
    ) -> None:
        """Execute a trade on the paper portfolio and store context."""
        # Check if we need to close existing opposite position
        existing = self.portfolio.positions.get(symbol.lower())
        if existing and existing.is_open and existing.side != side:
            self.portfolio.close_position(symbol, price, "flip")
            self.position_manager.clear_position_context(symbol)
        
        # Open new position
        trade = self.portfolio.open_position(
            symbol=symbol,
            side=side,
            size_usd=size_usd,
            price=price,
            leverage=leverage,
            signal_confidence=signal.confidence,
            signal_source=signal.source,
        )
        
        if trade:
            # Store position context for later thesis checking
            position = self._get_position_as_type(symbol)
            if position:
                # Get market context at entry
                market_state = await self.hydra.layer1.get_market_state(symbol)
                funding_rate = 0.0
                if market_state and market_state.funding_rate:
                    funding_rate = market_state.funding_rate.rate
                
                thesis = signal.metadata.get('thesis', '') if signal.metadata else ''
                
                self.position_manager.store_position_context(
                    symbol=symbol,
                    position=position,
                    thesis=thesis,
                    signal_source=signal.source,
                    confidence=signal.confidence,
                    funding_rate=funding_rate,
                )
            
            if self._on_trade:
                self._on_trade(trade)
    
    async def _reduce_position(self, symbol: str, price: float, reduce_to_pct: float) -> None:
        """Reduce an existing position."""
        close_pct = 1.0 - reduce_to_pct
        if close_pct > 0:
            self.portfolio.close_position(symbol, price, "risk_reduction", partial_pct=close_pct)
    
    async def _close_all_positions(self, price: float, reason: str) -> None:
        """Close all open positions (kill switch)."""
        for symbol, position in list(self.portfolio.positions.items()):
            if position.is_open:
                current_price = self._current_prices.get(symbol, price)
                self.portfolio.close_position(symbol, current_price, reason)
    
    def _get_position_as_type(self, symbol: str) -> Optional[Position]:
        """Convert portfolio position to core Position type."""
        pos = self.portfolio.positions.get(symbol.lower())
        if not pos or not pos.is_open:
            return None
        
        return Position(
            symbol=pos.symbol,
            side=pos.side,
            size=pos.size,
            size_usd=pos.size_usd,
            entry_price=pos.avg_entry_price,
            current_price=pos.current_price,
            leverage=pos.leverage,
            unrealized_pnl=pos.unrealized_pnl,
            unrealized_pnl_pct=pos.unrealized_pnl_pct,
            liquidation_price=pos.liquidation_price,
            margin_used=pos.margin_used,
            entry_time=pos.entry_time,
            funding_paid=pos.total_funding_paid,
        )
    
    def _get_all_positions_as_type(self) -> dict[str, Position]:
        """Get all positions as core Position types."""
        positions = {}
        for symbol in self.portfolio.positions:
            pos = self._get_position_as_type(symbol)
            if pos:
                positions[symbol] = pos
        return positions
    
    def _should_take_snapshot(self) -> bool:
        """Check if we should take a portfolio snapshot."""
        if self._last_snapshot_time is None:
            return True
        return datetime.now(timezone.utc) - self._last_snapshot_time > self._snapshot_interval
    
    # === Manual Trading Methods ===
    
    def manual_open(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        price: float,
        leverage: float = 3.0,
    ) -> bool:
        """
        Manually open a position (for testing/override).
        
        Args:
            symbol: Trading pair (e.g., "cmt_btcusdt")
            side: "long" or "short"
            size_usd: Position size in USD
            price: Entry price
            leverage: Leverage to use
            
        Returns:
            True if successful
        """
        side_enum = Side.LONG if side.lower() == "long" else Side.SHORT
        trade = self.portfolio.open_position(
            symbol=symbol,
            side=side_enum,
            size_usd=size_usd,
            price=price,
            leverage=leverage,
            signal_source="manual",
        )
        return trade is not None
    
    def manual_close(self, symbol: str, price: float, partial_pct: float = 1.0) -> bool:
        """
        Manually close a position.
        
        Args:
            symbol: Trading pair
            price: Exit price
            partial_pct: Percentage to close (1.0 = full)
            
        Returns:
            True if successful
        """
        trade = self.portfolio.close_position(symbol, price, "manual", partial_pct)
        return trade is not None
    
    def update_price(self, symbol: str, price: float) -> None:
        """Manually update a price (for testing)."""
        self._current_prices[symbol.lower()] = price
        self.portfolio.update_prices(self._current_prices)
    
    def show_dashboard(self) -> None:
        """Display the trading dashboard."""
        self.dashboard.print_dashboard()
    
    def show_positions(self) -> None:
        """Display open positions."""
        self.dashboard.print_positions()
    
    def show_trades(self, limit: int = 20) -> None:
        """Display recent trades."""
        self.dashboard.print_trades(limit=limit)
    
    def show_summary(self) -> None:
        """Display portfolio summary."""
        self.dashboard.print_summary()
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self.portfolio.total_equity
    
    def get_pnl(self) -> tuple[float, float]:
        """Get total P&L (absolute, percentage)."""
        pnl = self.portfolio.total_equity - self.portfolio.initial_balance
        pnl_pct = pnl / self.portfolio.initial_balance
        return pnl, pnl_pct
