"""
Layer 5: Decision & Execution Engine

Components:
1. Multi-Agent Vote System
   - Strategist (LLM) - narrative & leverage logic
   - Quant (Deep/RL) - statistical edge
   - Risk Manager - survival check
   - Executor - liquidity & slippage

2. Execution Engine (Non-HFT)
   - Post-only limits
   - Reduce-only exits
   - TWAP scaling
   - Slippage-aware sizing

Trade executes ONLY if all agents approve.
Execution is PATIENT, not fast.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from enum import Enum
import uuid
from loguru import logger

from hydra.core.config import HydraConfig, PAIR_DISPLAY_NAMES
from hydra.core.types import (
    MarketState,
    Signal,
    Position,
    Order,
    Trade,
    Side,
    OrderType,
    AgentVote,
    ExecutionPlan,
)
from hydra.layers.layer2_statistical import StatisticalResult
from hydra.layers.layer4_risk import RiskDecision


def _to_exchange_symbol(internal: str) -> str:
    """Convert internal symbol (cmt_btcusdt) to exchange format (BTC/USDT:USDT)."""
    display = PAIR_DISPLAY_NAMES.get(internal, internal)
    if "/" not in display:
        return internal
    base = display.split("/")[0]
    return f"{base}/USDT:USDT"


class VoteResult(Enum):
    """Result of multi-agent voting."""
    APPROVED = "approved"
    REJECTED = "rejected"
    VETOED = "vetoed"


@dataclass
class VotingResult:
    """Aggregated voting result."""
    result: VoteResult
    votes: list[AgentVote]
    approval_count: int
    veto_count: int
    consensus_direction: Side
    consensus_confidence: float
    rejection_reasons: list[str] = field(default_factory=list)


class StrategistAgent:
    """
    LLM-based strategist agent.
    Focuses on narrative and leverage logic.
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
    
    async def vote(
        self,
        signal: Signal,
        market_state: MarketState,
        stat_result: StatisticalResult,
    ) -> AgentVote:
        """Cast vote based on strategic analysis."""
        # Use signal metadata for LLM analysis
        metadata = signal.metadata or {}
        
        # Check for risk flags
        risk_flags = metadata.get('risk_flags', [])
        
        # Veto conditions
        veto = False
        veto_reason = ""
        
        if "CASCADE_RISK" in risk_flags and signal.confidence < 0.8:
            veto = True
            veto_reason = "Cascade risk present with insufficient confidence"
        
        if "REGIME_BREAK" in risk_flags:
            veto = True
            veto_reason = "Regime break - waiting for clarity"
        
        # Confidence adjustment
        confidence = signal.confidence
        
        # Boost for thesis quality
        thesis = metadata.get('thesis', '')
        if 'squeeze' in thesis.lower():
            confidence *= 1.1
        if 'trap' in thesis.lower():
            confidence *= 1.1
        
        confidence = min(1.0, confidence)
        
        return AgentVote(
            agent_name="Strategist",
            action=signal.side if not veto else Side.FLAT,
            confidence=confidence,
            reasoning=thesis or "Signal-based direction",
            veto=veto,
            veto_reason=veto_reason,
            metadata={'risk_flags': risk_flags},
        )


class QuantAgent:
    """
    Quantitative agent based on statistical/ML models.
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
    
    async def vote(
        self,
        signal: Signal,
        market_state: MarketState,
        stat_result: StatisticalResult,
    ) -> AgentVote:
        """Cast vote based on quantitative analysis."""
        veto = False
        veto_reason = ""
        
        # Veto on extreme volatility
        if stat_result.volatility_regime == "extreme":
            veto = True
            veto_reason = "Extreme volatility - models unreliable"
        
        # Veto on abnormal moves
        if stat_result.is_abnormal and abs(stat_result.abnormal_move_score) > 4:
            veto = True
            veto_reason = f"Abnormal move (z={stat_result.abnormal_move_score:.1f})"
        
        # Check regime alignment
        regime_aligned = True
        if signal.side == Side.LONG and stat_result.regime == Regime.TRENDING_DOWN:
            regime_aligned = False
        elif signal.side == Side.SHORT and stat_result.regime == Regime.TRENDING_UP:
            regime_aligned = False
        
        # Confidence based on signal and regime
        confidence = signal.confidence
        if not regime_aligned:
            confidence *= 0.5
        
        # Statistical edge check
        if stat_result.regime_confidence < 0.4:
            confidence *= 0.7
        
        return AgentVote(
            agent_name="Quant",
            action=signal.side if not veto else Side.FLAT,
            confidence=confidence,
            reasoning=f"Regime: {stat_result.regime.name}, Vol: {stat_result.volatility_regime}",
            veto=veto,
            veto_reason=veto_reason,
            metadata={
                'regime': stat_result.regime.name,
                'vol_zscore': stat_result.volatility_zscore,
            },
        )


class RiskManagerAgent:
    """
    Risk management agent - survival focused.
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
    
    async def vote(
        self,
        signal: Signal,
        risk_decision: RiskDecision,
    ) -> AgentVote:
        """Cast vote based on risk assessment."""
        # Risk manager respects the risk brain decision
        if risk_decision.veto:
            return AgentVote(
                agent_name="RiskManager",
                action=Side.FLAT,
                confidence=1.0,
                reasoning=risk_decision.veto_reason,
                veto=True,
                veto_reason=risk_decision.veto_reason,
            )
        
        # Approve with risk-adjusted confidence
        confidence = 1.0 - risk_decision.risk_score
        
        # Additional checks
        veto = False
        veto_reason = ""
        
        if risk_decision.risk_score > 0.8:
            veto = True
            veto_reason = f"Risk score {risk_decision.risk_score:.1%} too high"
        
        return AgentVote(
            agent_name="RiskManager",
            action=signal.side if not veto else Side.FLAT,
            confidence=confidence,
            reasoning=f"Risk score: {risk_decision.risk_score:.1%}, Size: ${risk_decision.recommended_size_usd:.0f}",
            veto=veto,
            veto_reason=veto_reason,
            metadata={
                'risk_score': risk_decision.risk_score,
                'recommended_size': risk_decision.recommended_size_usd,
            },
        )


class ExecutorAgent:
    """
    Execution agent - liquidity and slippage focused.
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
    
    async def vote(
        self,
        signal: Signal,
        market_state: MarketState,
        size_usd: float,
    ) -> AgentVote:
        """Cast vote based on execution feasibility."""
        veto = False
        veto_reason = ""
        
        # Check order book depth
        if market_state.order_book:
            ob = market_state.order_book
            
            # Check spread
            if ob.spread > 0.003:  # 30 bps
                veto = True
                veto_reason = f"Spread too wide: {ob.spread*10000:.0f} bps"
            
            # Check depth
            if signal.side == Side.LONG:
                depth = sum(q * p for p, q in ob.asks[:10])
            else:
                depth = sum(q * p for p, q in ob.bids[:10])
            
            # Size should be < 10% of visible depth
            if size_usd > depth * 0.1:
                veto = True
                veto_reason = f"Size ${size_usd:.0f} exceeds 10% of depth ${depth:.0f}"
        
        # Check for liquidity
        if market_state.volume_24h < size_usd * 100:
            veto = True
            veto_reason = "Insufficient 24h volume for size"
        
        confidence = 0.8  # Base confidence
        
        if market_state.order_book:
            # Favorable imbalance
            imb = market_state.order_book.imbalance
            if signal.side == Side.LONG and imb > 0:
                confidence += 0.1
            elif signal.side == Side.SHORT and imb < 0:
                confidence += 0.1
        
        return AgentVote(
            agent_name="Executor",
            action=signal.side if not veto else Side.FLAT,
            confidence=min(1.0, confidence),
            reasoning=f"Spread: {market_state.order_book.spread*10000:.0f}bps" if market_state.order_book else "No orderbook",
            veto=veto,
            veto_reason=veto_reason,
            metadata={
                'spread_bps': market_state.order_book.spread * 10000 if market_state.order_book else 0,
            },
        )


# Need to import Regime for QuantAgent
from hydra.core.types import Regime


class TWAPExecutor:
    """
    Time-Weighted Average Price executor.
    Slices orders over time to minimize impact.
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        self._active_executions: dict[str, ExecutionPlan] = {}
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        exchange: any,
    ) -> list[Trade]:
        """Execute a TWAP plan."""
        trades = []
        execution_id = str(uuid.uuid4())[:8]
        
        self._active_executions[execution_id] = plan
        
        try:
            slice_size = plan.target_size / plan.num_slices
            
            for i in range(plan.num_slices):
                if execution_id not in self._active_executions:
                    logger.info(f"Execution {execution_id} cancelled")
                    break
                
                # Place order
                order = await self._place_slice(
                    exchange,
                    plan.symbol,
                    plan.target_side,
                    slice_size,
                    plan.price_limit,
                )
                
                if order:
                    trade = Trade(
                        id=str(uuid.uuid4()),
                        symbol=plan.symbol,
                        side=plan.target_side,
                        quantity=order.filled_quantity,
                        price=order.average_fill_price,
                        fee=0,  # Would get from exchange
                        fee_currency="USDT",
                        timestamp=datetime.now(timezone.utc),
                        order_id=order.id,
                    )
                    trades.append(trade)
                
                if i < plan.num_slices - 1:
                    await asyncio.sleep(plan.slice_interval_seconds)
            
        finally:
            self._active_executions.pop(execution_id, None)
        
        return trades
    
    async def _place_slice(
        self,
        exchange,
        symbol: str,
        side: Side,
        size: float,
        price_limit: Optional[float],
    ) -> Optional[Order]:
        """Place a single slice order."""
        try:
            order_side = "buy" if side == Side.LONG else "sell"
            
            if self.config.trading.use_post_only:
                # Get best price
                ticker = await exchange.fetch_ticker(symbol)
                if side == Side.LONG:
                    price = ticker['bid']
                else:
                    price = ticker['ask']
                
                # Place limit order
                result = await exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=order_side,
                    amount=size,
                    price=price,
                    params={'postOnly': True},
                )
            else:
                # Market order
                result = await exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=order_side,
                    amount=size,
                )
            
            return Order(
                id=result['id'],
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT if self.config.trading.use_post_only else OrderType.MARKET,
                quantity=size,
                price=result.get('price'),
                status=result['status'],
                filled_quantity=result.get('filled', 0),
                average_fill_price=result.get('average', 0),
                created_at=datetime.now(timezone.utc),
            )
            
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return None
    
    def cancel_execution(self, execution_id: str) -> None:
        """Cancel an ongoing execution."""
        self._active_executions.pop(execution_id, None)


class DecisionExecutionEngine:
    """
    Layer 5: Decision & Execution Engine
    
    Orchestrates:
    1. Multi-agent voting
    2. Trade approval
    3. Order execution (TWAP)
    
    Trade executes ONLY if all agents approve.
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        
        # Voting agents
        self.strategist = StrategistAgent(config)
        self.quant = QuantAgent(config)
        self.risk_manager = RiskManagerAgent(config)
        self.executor_agent = ExecutorAgent(config)
        
        # Execution
        self.twap = TWAPExecutor(config)
        
        # Exchange connection
        self._exchange = None
        
        # Order tracking
        self._pending_orders: dict[str, Order] = {}
        self._open_orders: dict[str, Order] = {}
        
        logger.info("Decision Execution Engine initialized")
    
    async def setup(self) -> None:
        """Initialize execution engine."""
        import ccxt.async_support as ccxt
        
        exchange_id = self.config.exchange.primary_exchange
        
        if exchange_id == "binance":
            self._exchange = ccxt.binanceusdm({
                'apiKey': self.config.exchange.binance_api_key,
                'secret': self.config.exchange.binance_api_secret,
                'sandbox': self.config.exchange.binance_testnet,
                'options': {'defaultType': 'future'},
            })
        elif exchange_id == "bybit":
            self._exchange = ccxt.bybit({
                'apiKey': self.config.exchange.bybit_api_key,
                'secret': self.config.exchange.bybit_api_secret,
                'sandbox': self.config.exchange.bybit_testnet,
                'options': {'defaultType': 'linear'},
            })
        
        if self._exchange:
            await self._exchange.load_markets()
    
    async def collect_votes(
        self,
        signal: Signal,
        market_state: MarketState,
        stat_result: StatisticalResult,
        risk_decision: RiskDecision,
    ) -> list[AgentVote]:
        """Collect votes from all agents."""
        votes = await asyncio.gather(
            self.strategist.vote(signal, market_state, stat_result),
            self.quant.vote(signal, market_state, stat_result),
            self.risk_manager.vote(signal, risk_decision),
            self.executor_agent.vote(signal, market_state, risk_decision.recommended_size_usd),
        )
        
        return list(votes)
    
    def evaluate_votes(
        self,
        votes: list[AgentVote],
        signal: Signal,
        risk_decision: RiskDecision,
    ) -> tuple[bool, Optional[ExecutionPlan]]:
        """
        Evaluate votes and create execution plan if approved.
        
        Trade executes ONLY if:
        - No vetoes
        - At least min_agents_agreement agree
        """
        # Count vetoes and approvals
        veto_count = sum(1 for v in votes if v.veto)
        approval_count = sum(1 for v in votes if v.action == signal.side and not v.veto)
        
        # Log votes
        for vote in votes:
            status = "VETO" if vote.veto else vote.action.value.upper()
            logger.debug(f"Vote: {vote.agent_name} -> {status} ({vote.confidence:.1%})")
        
        # Check veto
        if veto_count > 0:
            veto_reasons = [v.veto_reason for v in votes if v.veto]
            logger.info(f"Trade vetoed: {', '.join(veto_reasons)}")
            return False, None
        
        # Check minimum agreement
        if approval_count < self.config.risk.min_agents_agreement:
            logger.info(f"Insufficient agreement: {approval_count}/{self.config.risk.min_agents_agreement}")
            return False, None
        
        # Create execution plan
        plan = ExecutionPlan(
            symbol=signal.symbol,
            target_side=signal.side,
            target_size=risk_decision.recommended_size_usd / market_state.price,
            target_size_usd=risk_decision.recommended_size_usd,
            execution_style="twap",
            num_slices=max(1, int(risk_decision.recommended_size_usd / 1000)),  # 1 slice per $1000
            slice_interval_seconds=30,
            max_slippage_bps=self.config.trading.max_slippage_bps,
            stop_loss=risk_decision.stop_loss_price,
            take_profit=risk_decision.take_profit_price,
        )
        
        return True, plan
    
    async def execute(self, plan: ExecutionPlan) -> list[Trade]:
        """Execute a trading plan."""
        logger.info(
            f"Executing: {plan.target_side.value} {plan.target_size:.4f} {plan.symbol} "
            f"(${plan.target_size_usd:.0f}) via {plan.execution_style}"
        )
        
        if self.config.trading.trading_mode == "paper":
            # Paper trading - simulate execution
            trade = Trade(
                id=str(uuid.uuid4()),
                symbol=plan.symbol,
                side=plan.target_side,
                quantity=plan.target_size,
                price=plan.price_limit or 0,
                fee=plan.target_size_usd * 0.0004,  # 4 bps
                fee_currency="USDT",
                timestamp=datetime.now(timezone.utc),
            )
            logger.info(f"Paper trade executed: {trade.id}")
            return [trade]
        
        elif self.config.trading.trading_mode == "live":
            if not self._exchange:
                logger.error("Exchange not connected")
                return []
            
            trades = await self.twap.execute_plan(plan, self._exchange)
            return trades
        
        return []
    
    async def close_position(self, symbol: str) -> Optional[Trade]:
        """Close an existing position."""
        if not self._exchange:
            return None
        
        try:
            # Get current position
            positions = await self._exchange.fetch_positions([symbol])
            
            for pos in positions:
                if pos['symbol'] == symbol and float(pos['contracts']) != 0:
                    side = "sell" if float(pos['contracts']) > 0 else "buy"
                    amount = abs(float(pos['contracts']))
                    
                    result = await self._exchange.create_order(
                        symbol=symbol,
                        type='market',
                        side=side,
                        amount=amount,
                        params={'reduceOnly': True},
                    )
                    
                    return Trade(
                        id=result['id'],
                        symbol=symbol,
                        side=Side.SHORT if side == "sell" else Side.LONG,
                        quantity=amount,
                        price=result.get('average', 0),
                        fee=0,
                        fee_currency="USDT",
                        timestamp=datetime.now(timezone.utc),
                        order_id=result['id'],
                    )
                    
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
        
        return None
    
    async def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        if not self._exchange:
            return
        
        try:
            for symbol in self.config.trading.symbols:
                exchange_symbol = _to_exchange_symbol(symbol)
                await self._exchange.cancel_all_orders(exchange_symbol)
            logger.info("All orders cancelled")
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
    
    async def get_positions(self) -> dict[str, Position]:
        """Get all current positions."""
        positions = {}
        
        if not self._exchange:
            return positions
        
        try:
            raw_positions = await self._exchange.fetch_positions()
            
            for pos in raw_positions:
                if float(pos.get('contracts', 0)) != 0:
                    symbol = pos['symbol']
                    contracts = float(pos['contracts'])
                    entry_price = float(pos.get('entryPrice', 0))
                    mark_price = float(pos.get('markPrice', 0))
                    
                    side = Side.LONG if contracts > 0 else Side.SHORT
                    size = abs(contracts)
                    
                    unrealized_pnl = float(pos.get('unrealizedPnl', 0))
                    
                    positions[symbol] = Position(
                        symbol=symbol,
                        side=side,
                        size=size,
                        size_usd=size * mark_price,
                        entry_price=entry_price,
                        current_price=mark_price,
                        leverage=float(pos.get('leverage', 1)),
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl / (size * entry_price) if entry_price > 0 else 0,
                        liquidation_price=float(pos.get('liquidationPrice', 0)),
                        margin_used=float(pos.get('initialMargin', 0)),
                    )
                    
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
        
        return positions
