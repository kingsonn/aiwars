"""
Layer 5: Order Execution Engine

Handles actual order placement on exchange:
- Entry execution (limit post-only)
- Exit execution (reduce-only)
- Stop-loss/take-profit placement
- Order monitoring and fill handling

Per HYDRA_SPEC_LAYERS.md and HYDRA_SPEC_TRADING.md

This is PATIENT execution, not HFT.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from enum import Enum
import os
from loguru import logger

try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("ccxt not available - execution will be simulated")

from hydra.core.types import Signal, Position, Side, Order, OrderType
from hydra.layers.layer4_risk import RiskDecision


# =============================================================================
# CONSTANTS
# =============================================================================

ORDER_TIMEOUT_SECONDS = 120
ENTRY_FILL_TIMEOUT = 120  # 2 minutes for entry fills
EXIT_FILL_TIMEOUT = 60    # 1 minute for exit fills

# Symbol mapping
SYMBOL_MAP = {
    "cmt_btcusdt": "BTC/USDT:USDT",
    "cmt_ethusdt": "ETH/USDT:USDT",
    "cmt_solusdt": "SOL/USDT:USDT",
    "cmt_bnbusdt": "BNB/USDT:USDT",
    "cmt_adausdt": "ADA/USDT:USDT",
    "cmt_xrpusdt": "XRP/USDT:USDT",
    "cmt_ltcusdt": "LTC/USDT:USDT",
    "cmt_dogeusdt": "DOGE/USDT:USDT",
}


# =============================================================================
# EXECUTION RESULT
# =============================================================================

class ExecutionStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of an order execution."""
    status: ExecutionStatus
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    filled_qty: float = 0.0
    filled_price: float = 0.0
    filled_usd: float = 0.0
    fees: float = 0.0
    message: str = ""
    raw_order: Dict = field(default_factory=dict)


# =============================================================================
# EXCHANGE CLIENT
# =============================================================================

class BinanceFuturesExecutor:
    """
    Binance Futures execution client.
    
    Uses ccxt for order management.
    Supports paper trading mode.
    """
    
    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.exchange = None
        self._initialized = False
        
        # Paper trading state
        self._paper_positions: Dict[str, Dict] = {}
        self._paper_orders: Dict[str, Dict] = {}
        self._paper_order_id = 0
    
    async def initialize(self):
        """Initialize exchange connection."""
        if not CCXT_AVAILABLE:
            logger.warning("ccxt not available, using paper trading only")
            self.paper_trading = True
            self._initialized = True
            return
        
        if self.paper_trading:
            logger.info("Executor initialized in PAPER TRADING mode")
            self._initialized = True
            return
        
        # Live trading setup
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        
        if not api_key or not api_secret:
            logger.warning("No API keys found, falling back to paper trading")
            self.paper_trading = True
            self._initialized = True
            return
        
        try:
            self.exchange = ccxt.binanceusdm({
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",
                }
            })
            
            # Test connection
            await self.exchange.load_markets()
            logger.info("Executor initialized for LIVE trading on Binance Futures")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            self.paper_trading = True
            self._initialized = True
    
    async def close(self):
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()
    
    def _to_exchange_symbol(self, internal_symbol: str) -> str:
        """Convert internal symbol to exchange format."""
        return SYMBOL_MAP.get(internal_symbol.lower(), internal_symbol)
    
    async def get_orderbook_top(self, symbol: str) -> tuple[float, float]:
        """Get best bid and ask prices."""
        exchange_symbol = self._to_exchange_symbol(symbol)
        
        if self.paper_trading or not self.exchange:
            # Return simulated prices
            return 0.0, 0.0
        
        try:
            ob = await self.exchange.fetch_order_book(exchange_symbol, limit=5)
            best_bid = ob["bids"][0][0] if ob["bids"] else 0
            best_ask = ob["asks"][0][0] if ob["asks"] else 0
            return best_bid, best_ask
        except Exception as e:
            logger.error(f"Failed to get orderbook: {e}")
            return 0.0, 0.0
    
    async def execute_entry(
        self,
        signal: Signal,
        risk_decision: RiskDecision,
        current_price: float,
    ) -> ExecutionResult:
        """
        Execute an entry order.
        
        Uses limit post-only to be maker.
        """
        symbol = signal.symbol
        exchange_symbol = self._to_exchange_symbol(symbol)
        
        # Calculate entry price and quantity
        if signal.side == Side.LONG:
            entry_price = current_price * 0.9999  # Slightly below for bid
            order_side = "buy"
        else:
            entry_price = current_price * 1.0001  # Slightly above for ask
            order_side = "sell"
        
        quantity = risk_decision.recommended_size_usd / entry_price
        
        logger.info(
            f"Executing ENTRY: {exchange_symbol} {order_side.upper()} "
            f"qty={quantity:.6f} @ ${entry_price:.2f} "
            f"(size: ${risk_decision.recommended_size_usd:.0f})"
        )
        
        if self.paper_trading:
            return await self._paper_entry(
                symbol, signal.side, quantity, entry_price, risk_decision
            )
        
        try:
            # Place limit post-only order
            order = await self.exchange.create_order(
                symbol=exchange_symbol,
                type="limit",
                side=order_side,
                amount=quantity,
                price=entry_price,
                params={"postOnly": True}
            )
            
            # Wait for fill
            filled = await self._wait_for_fill(order["id"], exchange_symbol, ENTRY_FILL_TIMEOUT)
            
            if not filled:
                # Cancel unfilled order
                await self.exchange.cancel_order(order["id"], exchange_symbol)
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    order_id=order["id"],
                    symbol=symbol,
                    side=order_side,
                    message="Entry order timed out - not chasing"
                )
            
            # Set stop-loss
            await self._set_stop_loss(
                exchange_symbol, signal.side, quantity, risk_decision.stop_loss_price
            )
            
            # Set take-profit
            await self._set_take_profit(
                exchange_symbol, signal.side, quantity, risk_decision.take_profit_price
            )
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                order_id=order["id"],
                symbol=symbol,
                side=order_side,
                filled_qty=quantity,
                filled_price=entry_price,
                filled_usd=risk_decision.recommended_size_usd,
                raw_order=order
            )
            
        except Exception as e:
            logger.error(f"Entry execution failed: {e}")
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                symbol=symbol,
                side=order_side,
                message=str(e)
            )
    
    async def execute_exit(
        self,
        position: Position,
        reason: str,
        fraction: float = 1.0,
        urgent: bool = False,
    ) -> ExecutionResult:
        """
        Execute an exit order.
        
        Uses limit reduce-only, falls back to market if urgent or timeout.
        """
        symbol = position.symbol
        exchange_symbol = self._to_exchange_symbol(symbol)
        
        exit_size = position.size * fraction
        
        if position.side == Side.LONG:
            order_side = "sell"
            exit_price = position.current_price * 0.9999
        else:
            order_side = "buy"
            exit_price = position.current_price * 1.0001
        
        logger.info(
            f"Executing EXIT: {exchange_symbol} {order_side.upper()} "
            f"qty={exit_size:.6f} @ ${exit_price:.2f} "
            f"(reason: {reason}, fraction: {fraction:.0%})"
        )
        
        if self.paper_trading:
            return await self._paper_exit(symbol, position.side, exit_size, exit_price)
        
        try:
            if urgent:
                # Immediate market exit
                order = await self.exchange.create_order(
                    symbol=exchange_symbol,
                    type="market",
                    side=order_side,
                    amount=exit_size,
                    params={"reduceOnly": True}
                )
            else:
                # Try limit first
                order = await self.exchange.create_order(
                    symbol=exchange_symbol,
                    type="limit",
                    side=order_side,
                    amount=exit_size,
                    price=exit_price,
                    params={"reduceOnly": True, "postOnly": True}
                )
                
                # Wait for fill
                filled = await self._wait_for_fill(order["id"], exchange_symbol, EXIT_FILL_TIMEOUT)
                
                if not filled:
                    # Cancel and use market
                    await self.exchange.cancel_order(order["id"], exchange_symbol)
                    order = await self.exchange.create_order(
                        symbol=exchange_symbol,
                        type="market",
                        side=order_side,
                        amount=exit_size,
                        params={"reduceOnly": True}
                    )
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                order_id=order["id"],
                symbol=symbol,
                side=order_side,
                filled_qty=exit_size,
                filled_price=exit_price,
                filled_usd=exit_size * exit_price,
                raw_order=order
            )
            
        except Exception as e:
            logger.error(f"Exit execution failed: {e}")
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                symbol=symbol,
                side=order_side,
                message=str(e)
            )
    
    async def _wait_for_fill(
        self,
        order_id: str,
        symbol: str,
        timeout: int
    ) -> bool:
        """Wait for order to fill."""
        start = datetime.now(timezone.utc)
        
        while (datetime.now(timezone.utc) - start).total_seconds() < timeout:
            try:
                order = await self.exchange.fetch_order(order_id, symbol)
                
                if order["status"] == "closed":
                    return True
                if order["status"] == "canceled":
                    return False
                
            except Exception as e:
                logger.warning(f"Error checking order status: {e}")
            
            await asyncio.sleep(2)
        
        return False
    
    async def _set_stop_loss(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        price: float
    ):
        """Set stop-loss order."""
        try:
            stop_side = "sell" if side == Side.LONG else "buy"
            
            await self.exchange.create_order(
                symbol=symbol,
                type="stop_market",
                side=stop_side,
                amount=quantity,
                params={
                    "stopPrice": price,
                    "reduceOnly": True
                }
            )
            logger.info(f"Stop-loss set: {symbol} @ ${price:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to set stop-loss: {e}")
    
    async def _set_take_profit(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        price: float
    ):
        """Set take-profit order."""
        try:
            tp_side = "sell" if side == Side.LONG else "buy"
            
            await self.exchange.create_order(
                symbol=symbol,
                type="take_profit_market",
                side=tp_side,
                amount=quantity,
                params={
                    "stopPrice": price,
                    "reduceOnly": True
                }
            )
            logger.info(f"Take-profit set: {symbol} @ ${price:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to set take-profit: {e}")
    
    async def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a symbol."""
        if self.paper_trading:
            self._paper_orders = {
                k: v for k, v in self._paper_orders.items()
                if v.get("symbol") != symbol
            }
            return
        
        try:
            exchange_symbol = self._to_exchange_symbol(symbol)
            await self.exchange.cancel_all_orders(exchange_symbol)
            logger.info(f"Cancelled all orders for {symbol}")
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
    
    async def flatten_all(self, reason: str):
        """Emergency flatten all positions."""
        logger.critical(f"FLATTENING ALL POSITIONS: {reason}")
        
        if self.paper_trading:
            for symbol, pos in list(self._paper_positions.items()):
                await self._paper_exit(symbol, pos["side"], pos["size"], pos["price"])
            self._paper_positions.clear()
            return
        
        try:
            positions = await self.exchange.fetch_positions()
            
            for pos in positions:
                if float(pos["contracts"]) > 0:
                    side = "sell" if pos["side"] == "long" else "buy"
                    await self.exchange.create_order(
                        symbol=pos["symbol"],
                        type="market",
                        side=side,
                        amount=float(pos["contracts"]),
                        params={"reduceOnly": True}
                    )
            
            logger.info("All positions flattened")
            
        except Exception as e:
            logger.error(f"Failed to flatten positions: {e}")
    
    # =========================================================================
    # PAPER TRADING
    # =========================================================================
    
    async def _paper_entry(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        price: float,
        risk_decision: RiskDecision,
    ) -> ExecutionResult:
        """Simulate entry order."""
        self._paper_order_id += 1
        order_id = f"paper_{self._paper_order_id}"
        
        # Simulate fill
        self._paper_positions[symbol] = {
            "side": side,
            "size": quantity,
            "price": price,
            "entry_time": datetime.now(timezone.utc),
            "stop_loss": risk_decision.stop_loss_price,
            "take_profit": risk_decision.take_profit_price,
        }
        
        logger.info(f"[PAPER] Entry filled: {symbol} {side.value} {quantity:.6f} @ ${price:.2f}")
        
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            order_id=order_id,
            symbol=symbol,
            side="buy" if side == Side.LONG else "sell",
            filled_qty=quantity,
            filled_price=price,
            filled_usd=quantity * price,
        )
    
    async def _paper_exit(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        price: float,
    ) -> ExecutionResult:
        """Simulate exit order."""
        self._paper_order_id += 1
        order_id = f"paper_{self._paper_order_id}"
        
        # Calculate PnL
        if symbol in self._paper_positions:
            entry_price = self._paper_positions[symbol]["price"]
            if side == Side.LONG:
                pnl = (price - entry_price) * quantity
            else:
                pnl = (entry_price - price) * quantity
            
            del self._paper_positions[symbol]
            logger.info(f"[PAPER] Exit filled: {symbol} {quantity:.6f} @ ${price:.2f} (PnL: ${pnl:.2f})")
        
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            order_id=order_id,
            symbol=symbol,
            side="sell" if side == Side.LONG else "buy",
            filled_qty=quantity,
            filled_price=price,
            filled_usd=quantity * price,
        )
    
    def get_paper_positions(self) -> Dict[str, Dict]:
        """Get current paper trading positions."""
        return self._paper_positions.copy()
