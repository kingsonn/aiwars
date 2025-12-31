"""
HYDRA Paper Trading Portfolio Manager

Tracks:
- All positions with entry prices
- Trade history
- P&L (realized and unrealized)
- Portfolio value over time
- Performance metrics
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from loguru import logger

from hydra.core.types import Side, Position, Trade, Order
from hydra.core.config import PERMITTED_PAIRS, PAIR_DISPLAY_NAMES


@dataclass
class TradeRecord:
    """Record of a single trade execution."""
    id: str
    timestamp: datetime
    symbol: str
    side: Side
    action: str  # "open", "close", "add", "reduce"
    quantity: float
    price: float
    fee: float
    fee_pct: float
    
    # Position context
    position_size_before: float
    position_size_after: float
    
    # P&L (for closes)
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    
    # Funding
    funding_paid: float = 0.0
    
    # Metadata
    signal_confidence: float = 0.0
    signal_source: str = ""
    notes: str = ""


@dataclass
class PositionRecord:
    """Active position with full history."""
    symbol: str
    side: Side
    
    # Size
    size: float  # in base currency
    size_usd: float
    
    # Entry
    entry_price: float
    entry_time: datetime
    avg_entry_price: float  # weighted average if scaled in
    
    # Current
    current_price: float = 0.0
    mark_price: float = 0.0
    
    # Leverage
    leverage: float = 1.0
    margin_used: float = 0.0
    liquidation_price: float = 0.0
    
    # P&L
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    # Funding
    total_funding_paid: float = 0.0
    
    # Trade history for this position
    trade_ids: list[str] = field(default_factory=list)
    
    @property
    def display_name(self) -> str:
        return PAIR_DISPLAY_NAMES.get(self.symbol, self.symbol)
    
    @property
    def is_open(self) -> bool:
        return self.size > 0
    
    def update_pnl(self, current_price: float) -> None:
        """Update unrealized P&L based on current price."""
        self.current_price = current_price
        if self.side == Side.LONG:
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.size
            self.unrealized_pnl_pct = (current_price - self.avg_entry_price) / self.avg_entry_price
        elif self.side == Side.SHORT:
            self.unrealized_pnl = (self.avg_entry_price - current_price) * self.size
            self.unrealized_pnl_pct = (self.avg_entry_price - current_price) / self.avg_entry_price


@dataclass
class PortfolioSnapshot:
    """Point-in-time snapshot of portfolio state."""
    timestamp: datetime
    
    # Value
    total_equity: float
    available_balance: float
    used_margin: float
    
    # P&L
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_funding_paid: float
    
    # Positions
    num_positions: int
    positions: dict[str, dict]  # symbol -> position summary
    
    # Performance
    pnl_pct: float  # Total P&L as % of initial
    drawdown: float  # Current drawdown from peak
    
    # Exposure
    gross_exposure: float
    net_exposure: float
    gross_leverage: float


class Portfolio:
    """
    Paper trading portfolio manager.
    
    Tracks all positions, trades, and portfolio value with full history.
    Persists state to disk for recovery.
    """
    
    def __init__(
        self,
        initial_balance: float = 1000.0,
        data_dir: str = "./data/paper_trading",
    ):
        self.initial_balance = initial_balance
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Current state
        self.balance = initial_balance  # Available USDC
        self.positions: dict[str, PositionRecord] = {}  # symbol -> position
        self.trades: list[TradeRecord] = []
        self.snapshots: list[PortfolioSnapshot] = []
        
        # Tracking
        self.peak_equity = initial_balance
        self.total_realized_pnl = 0.0
        self.total_funding_paid = 0.0
        self.trade_counter = 0
        
        # Fee structure (Binance futures taker)
        self.taker_fee_pct = 0.0004  # 0.04%
        self.maker_fee_pct = 0.0002  # 0.02%
        
        logger.info(f"Portfolio initialized with ${initial_balance:.2f} USDC")
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Sum of all unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.positions.values() if p.is_open)
    
    @property
    def total_equity(self) -> float:
        """Total portfolio value including unrealized P&L."""
        return self.balance + self.total_unrealized_pnl + sum(
            p.margin_used for p in self.positions.values() if p.is_open
        )
    
    @property
    def used_margin(self) -> float:
        """Total margin used by open positions."""
        return sum(p.margin_used for p in self.positions.values() if p.is_open)
    
    @property
    def available_balance(self) -> float:
        """Balance available for new positions."""
        return self.balance
    
    @property
    def can_trade(self) -> bool:
        """Check if we have enough balance to open new positions."""
        min_trade_size = 50  # Minimum $50 to open a position
        return self.available_balance >= min_trade_size
    
    @property
    def available_margin_pct(self) -> float:
        """Percentage of equity available as margin."""
        equity = self.total_equity
        if equity <= 0:
            return 0.0
        return self.available_balance / equity
    
    @property
    def gross_exposure(self) -> float:
        """Total absolute exposure."""
        return sum(p.size_usd for p in self.positions.values() if p.is_open)
    
    @property
    def net_exposure(self) -> float:
        """Net directional exposure."""
        net = 0.0
        for p in self.positions.values():
            if p.is_open:
                if p.side == Side.LONG:
                    net += p.size_usd
                else:
                    net -= p.size_usd
        return net
    
    @property
    def gross_leverage(self) -> float:
        """Gross leverage ratio."""
        equity = self.total_equity
        if equity <= 0:
            return 0.0
        return self.gross_exposure / equity
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        equity = self.total_equity
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - equity) / self.peak_equity
    
    def can_open_position(self, size_usd: float, leverage: float = 1.0) -> tuple[bool, str, float]:
        """
        Check if we can open a position of given size.
        
        Returns:
            (can_open, reason, max_size_possible)
        """
        margin_required = size_usd / leverage
        
        if margin_required <= 0:
            return False, "Invalid size", 0.0
        
        if margin_required > self.available_balance:
            max_size = self.available_balance * leverage * 0.95  # 5% buffer
            return False, f"Insufficient margin. Need ${margin_required:.2f}, have ${self.available_balance:.2f}", max_size
        
        # Check exposure limits
        max_exposure = self.total_equity * 5  # Max 5x gross leverage
        if self.gross_exposure + size_usd > max_exposure:
            remaining = max_exposure - self.gross_exposure
            return False, f"Would exceed max exposure. Remaining: ${remaining:.2f}", max(0, remaining)
        
        return True, "OK", size_usd
    
    def get_position_action(
        self,
        symbol: str,
        new_side: Side,
        signal_confidence: float,
    ) -> tuple[str, Optional[str]]:
        """
        Determine what action to take for a symbol given a new signal.
        
        Returns:
            (action, reason)
            action: "open" | "add" | "hold" | "close" | "flip" | "skip"
        """
        symbol = symbol.lower()
        existing = self.positions.get(symbol)
        
        if not existing or not existing.is_open:
            # No position - can open new
            if not self.can_trade:
                return "skip", "Insufficient balance for new position"
            return "open", None
        
        # Have existing position
        if existing.side == new_side:
            # Same direction - consider adding
            if signal_confidence > 0.7 and self.can_trade:
                return "add", "High confidence signal, adding to position"
            return "hold", "Already positioned in this direction"
        else:
            # Opposite direction
            if signal_confidence > 0.6:
                return "flip", "Signal suggests reversing position"
            return "hold", "Keeping existing position (low confidence reversal)"
    
    def open_position(
        self,
        symbol: str,
        side: Side,
        size_usd: float,
        price: float,
        leverage: float = 1.0,
        signal_confidence: float = 0.0,
        signal_source: str = "",
    ) -> Optional[TradeRecord]:
        """
        Open a new position or add to existing.
        
        Args:
            symbol: Trading pair (must be in PERMITTED_PAIRS)
            side: LONG or SHORT
            size_usd: Position size in USD
            price: Entry price
            leverage: Leverage to use
            signal_confidence: Confidence from signal
            signal_source: Source of the signal
            
        Returns:
            TradeRecord if successful, None if failed
        """
        symbol = symbol.lower()
        
        # Validate pair
        if symbol not in PERMITTED_PAIRS:
            logger.error(f"Invalid pair: {symbol}. Must be one of {PERMITTED_PAIRS}")
            return None
        
        # Calculate margin required
        margin_required = size_usd / leverage
        
        # Check if we can open this position
        can_open, reason, max_size = self.can_open_position(size_usd, leverage)
        if not can_open:
            # Try with reduced size if possible
            if max_size >= 50:  # Minimum viable size
                logger.info(f"Reducing position size from ${size_usd:.2f} to ${max_size:.2f}")
                size_usd = max_size
                margin_required = size_usd / leverage
            else:
                logger.warning(f"Cannot open position: {reason}")
                return None
        
        # Calculate fee
        fee = size_usd * self.taker_fee_pct
        
        # Calculate size in base currency
        size = size_usd / price
        
        # Generate trade ID
        self.trade_counter += 1
        trade_id = f"T{self.trade_counter:06d}"
        
        now = datetime.now(timezone.utc)
        
        # Check if adding to existing position
        existing = self.positions.get(symbol)
        position_size_before = existing.size if existing and existing.is_open else 0.0
        
        if existing and existing.is_open:
            if existing.side != side:
                # Opposite direction - close first
                logger.info(f"Closing existing {existing.side.value} position before opening {side.value}")
                self.close_position(symbol, price, "flip")
                existing = None
                position_size_before = 0.0
        
        if existing and existing.is_open:
            # Add to existing position
            action = "add"
            total_cost = existing.avg_entry_price * existing.size + price * size
            total_size = existing.size + size
            new_avg_price = total_cost / total_size
            
            existing.size = total_size
            existing.size_usd = total_size * price
            existing.avg_entry_price = new_avg_price
            existing.margin_used += margin_required
            existing.trade_ids.append(trade_id)
            
            position = existing
        else:
            # New position
            action = "open"
            
            # Calculate liquidation price (simplified)
            if side == Side.LONG:
                liq_price = price * (1 - 1/leverage + 0.005)  # 0.5% buffer
            else:
                liq_price = price * (1 + 1/leverage - 0.005)
            
            position = PositionRecord(
                symbol=symbol,
                side=side,
                size=size,
                size_usd=size_usd,
                entry_price=price,
                entry_time=now,
                avg_entry_price=price,
                current_price=price,
                leverage=leverage,
                margin_used=margin_required,
                liquidation_price=liq_price,
                trade_ids=[trade_id],
            )
            self.positions[symbol] = position
        
        # Deduct margin and fee from balance
        self.balance -= margin_required + fee
        
        # Create trade record
        trade = TradeRecord(
            id=trade_id,
            timestamp=now,
            symbol=symbol,
            side=side,
            action=action,
            quantity=size,
            price=price,
            fee=fee,
            fee_pct=self.taker_fee_pct,
            position_size_before=position_size_before,
            position_size_after=position.size,
            signal_confidence=signal_confidence,
            signal_source=signal_source,
        )
        self.trades.append(trade)
        
        logger.info(
            f"[{trade_id}] {action.upper()} {side.value} {position.display_name}: "
            f"{size:.6f} @ ${price:,.2f} (${size_usd:,.2f}, {leverage}x)"
        )
        
        return trade
    
    def close_position(
        self,
        symbol: str,
        price: float,
        reason: str = "signal",
        partial_pct: float = 1.0,
    ) -> Optional[TradeRecord]:
        """
        Close a position (fully or partially).
        
        Args:
            symbol: Trading pair
            price: Exit price
            reason: Reason for closing
            partial_pct: Percentage to close (1.0 = full)
            
        Returns:
            TradeRecord if successful
        """
        symbol = symbol.lower()
        position = self.positions.get(symbol)
        
        if not position or not position.is_open:
            logger.warning(f"No open position for {symbol}")
            return None
        
        # Calculate close size
        close_size = position.size * partial_pct
        close_size_usd = close_size * price
        
        # Calculate P&L
        if position.side == Side.LONG:
            pnl = (price - position.avg_entry_price) * close_size
        else:
            pnl = (position.avg_entry_price - price) * close_size
        
        pnl_pct = pnl / (position.avg_entry_price * close_size)
        
        # Calculate fee
        fee = close_size_usd * self.taker_fee_pct
        
        # Net P&L after fee
        net_pnl = pnl - fee - position.total_funding_paid * partial_pct
        
        # Generate trade ID
        self.trade_counter += 1
        trade_id = f"T{self.trade_counter:06d}"
        
        now = datetime.now(timezone.utc)
        position_size_before = position.size
        
        # Update position
        margin_released = position.margin_used * partial_pct
        
        if partial_pct >= 0.999:  # Full close
            action = "close"
            position.size = 0
            position.size_usd = 0
            position.margin_used = 0
        else:
            action = "reduce"
            position.size -= close_size
            position.size_usd = position.size * price
            position.margin_used -= margin_released
        
        position.trade_ids.append(trade_id)
        
        # Update balance
        self.balance += margin_released + net_pnl
        self.total_realized_pnl += net_pnl
        
        # Update peak equity
        equity = self.total_equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Create trade record
        trade = TradeRecord(
            id=trade_id,
            timestamp=now,
            symbol=symbol,
            side=position.side,
            action=action,
            quantity=close_size,
            price=price,
            fee=fee,
            fee_pct=self.taker_fee_pct,
            position_size_before=position_size_before,
            position_size_after=position.size,
            realized_pnl=net_pnl,
            realized_pnl_pct=pnl_pct,
            funding_paid=position.total_funding_paid * partial_pct,
            notes=reason,
        )
        self.trades.append(trade)
        
        logger.info(
            f"[{trade_id}] {action.upper()} {position.side.value} {position.display_name}: "
            f"{close_size:.6f} @ ${price:,.2f} | P&L: ${net_pnl:+,.2f} ({pnl_pct:+.2%})"
        )
        
        return trade
    
    def update_prices(self, prices: dict[str, float]) -> None:
        """
        Update current prices and recalculate P&L.
        
        Args:
            prices: Dict of symbol -> current price
        """
        for symbol, price in prices.items():
            symbol = symbol.lower()
            position = self.positions.get(symbol)
            if position and position.is_open:
                position.update_pnl(price)
        
        # Update peak equity
        equity = self.total_equity
        if equity > self.peak_equity:
            self.peak_equity = equity
    
    def apply_funding(self, symbol: str, funding_rate: float) -> None:
        """
        Apply funding payment to a position.
        
        Args:
            symbol: Trading pair
            funding_rate: Funding rate (positive = longs pay shorts)
        """
        symbol = symbol.lower()
        position = self.positions.get(symbol)
        
        if not position or not position.is_open:
            return
        
        # Calculate funding payment
        funding = position.size_usd * abs(funding_rate)
        
        # Determine who pays
        if (funding_rate > 0 and position.side == Side.LONG) or \
           (funding_rate < 0 and position.side == Side.SHORT):
            # We pay
            self.balance -= funding
            position.total_funding_paid += funding
            self.total_funding_paid += funding
            logger.debug(f"Paid ${funding:.4f} funding on {symbol}")
        else:
            # We receive
            self.balance += funding
            position.total_funding_paid -= funding
            self.total_funding_paid -= funding
            logger.debug(f"Received ${funding:.4f} funding on {symbol}")
    
    def take_snapshot(self) -> PortfolioSnapshot:
        """Take a snapshot of current portfolio state."""
        now = datetime.now(timezone.utc)
        
        positions_summary = {}
        for symbol, pos in self.positions.items():
            if pos.is_open:
                positions_summary[symbol] = {
                    "side": pos.side.value,
                    "size": pos.size,
                    "size_usd": pos.size_usd,
                    "entry_price": pos.avg_entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                    "leverage": pos.leverage,
                }
        
        snapshot = PortfolioSnapshot(
            timestamp=now,
            total_equity=self.total_equity,
            available_balance=self.available_balance,
            used_margin=self.used_margin,
            total_unrealized_pnl=self.total_unrealized_pnl,
            total_realized_pnl=self.total_realized_pnl,
            total_funding_paid=self.total_funding_paid,
            num_positions=len([p for p in self.positions.values() if p.is_open]),
            positions=positions_summary,
            pnl_pct=(self.total_equity - self.initial_balance) / self.initial_balance,
            drawdown=self.current_drawdown,
            gross_exposure=self.gross_exposure,
            net_exposure=self.net_exposure,
            gross_leverage=self.gross_leverage,
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_trade_history(self, limit: int = 50) -> list[dict]:
        """Get recent trade history as dicts."""
        trades = self.trades[-limit:]
        return [
            {
                "id": t.id,
                "time": t.timestamp.isoformat(),
                "symbol": PAIR_DISPLAY_NAMES.get(t.symbol, t.symbol),
                "side": t.side.value,
                "action": t.action,
                "qty": t.quantity,
                "price": t.price,
                "pnl": t.realized_pnl,
                "pnl_pct": t.realized_pnl_pct,
            }
            for t in reversed(trades)
        ]
    
    def get_open_positions(self) -> list[dict]:
        """Get all open positions as dicts."""
        positions = []
        for symbol, pos in self.positions.items():
            if pos.is_open:
                positions.append({
                    "symbol": pos.display_name,
                    "side": pos.side.value,
                    "size": pos.size,
                    "size_usd": pos.size_usd,
                    "entry_price": pos.avg_entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                    "leverage": pos.leverage,
                    "liquidation_price": pos.liquidation_price,
                    "funding_paid": pos.total_funding_paid,
                    "entry_time": pos.entry_time.isoformat(),
                })
        return positions
    
    def get_summary(self) -> dict:
        """Get portfolio summary."""
        winning_trades = [t for t in self.trades if t.realized_pnl > 0]
        losing_trades = [t for t in self.trades if t.realized_pnl < 0]
        
        return {
            "initial_balance": self.initial_balance,
            "total_equity": self.total_equity,
            "available_balance": self.available_balance,
            "used_margin": self.used_margin,
            "total_pnl": self.total_equity - self.initial_balance,
            "total_pnl_pct": (self.total_equity - self.initial_balance) / self.initial_balance,
            "realized_pnl": self.total_realized_pnl,
            "unrealized_pnl": self.total_unrealized_pnl,
            "funding_paid": self.total_funding_paid,
            "peak_equity": self.peak_equity,
            "current_drawdown": self.current_drawdown,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "gross_leverage": self.gross_leverage,
            "num_positions": len([p for p in self.positions.values() if p.is_open]),
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.trades) if self.trades else 0,
        }
    
    def save(self, filename: str = "portfolio_state.json") -> None:
        """Save portfolio state to disk."""
        filepath = self.data_dir / filename
        
        state = {
            "initial_balance": self.initial_balance,
            "balance": self.balance,
            "peak_equity": self.peak_equity,
            "total_realized_pnl": self.total_realized_pnl,
            "total_funding_paid": self.total_funding_paid,
            "trade_counter": self.trade_counter,
            "positions": {
                symbol: {
                    "symbol": pos.symbol,
                    "side": pos.side.value,
                    "size": pos.size,
                    "size_usd": pos.size_usd,
                    "entry_price": pos.entry_price,
                    "entry_time": pos.entry_time.isoformat(),
                    "avg_entry_price": pos.avg_entry_price,
                    "current_price": pos.current_price,
                    "leverage": pos.leverage,
                    "margin_used": pos.margin_used,
                    "liquidation_price": pos.liquidation_price,
                    "total_funding_paid": pos.total_funding_paid,
                    "trade_ids": pos.trade_ids,
                }
                for symbol, pos in self.positions.items()
            },
            "trades": [
                {
                    "id": t.id,
                    "timestamp": t.timestamp.isoformat(),
                    "symbol": t.symbol,
                    "side": t.side.value,
                    "action": t.action,
                    "quantity": t.quantity,
                    "price": t.price,
                    "fee": t.fee,
                    "realized_pnl": t.realized_pnl,
                    "realized_pnl_pct": t.realized_pnl_pct,
                }
                for t in self.trades
            ],
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Portfolio saved to {filepath}")
    
    def load(self, filename: str = "portfolio_state.json") -> bool:
        """Load portfolio state from disk."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"No saved state found at {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.initial_balance = state["initial_balance"]
            self.balance = state["balance"]
            self.peak_equity = state["peak_equity"]
            self.total_realized_pnl = state["total_realized_pnl"]
            self.total_funding_paid = state["total_funding_paid"]
            self.trade_counter = state["trade_counter"]
            
            # Restore positions
            self.positions = {}
            for symbol, pos_data in state.get("positions", {}).items():
                self.positions[symbol] = PositionRecord(
                    symbol=pos_data["symbol"],
                    side=Side(pos_data["side"]),
                    size=pos_data["size"],
                    size_usd=pos_data["size_usd"],
                    entry_price=pos_data["entry_price"],
                    entry_time=datetime.fromisoformat(pos_data["entry_time"]),
                    avg_entry_price=pos_data["avg_entry_price"],
                    current_price=pos_data["current_price"],
                    leverage=pos_data["leverage"],
                    margin_used=pos_data["margin_used"],
                    liquidation_price=pos_data["liquidation_price"],
                    total_funding_paid=pos_data["total_funding_paid"],
                    trade_ids=pos_data["trade_ids"],
                )
            
            # Restore trades
            self.trades = []
            for t_data in state.get("trades", []):
                self.trades.append(TradeRecord(
                    id=t_data["id"],
                    timestamp=datetime.fromisoformat(t_data["timestamp"]),
                    symbol=t_data["symbol"],
                    side=Side(t_data["side"]),
                    action=t_data["action"],
                    quantity=t_data["quantity"],
                    price=t_data["price"],
                    fee=t_data["fee"],
                    fee_pct=self.taker_fee_pct,
                    position_size_before=0,
                    position_size_after=0,
                    realized_pnl=t_data["realized_pnl"],
                    realized_pnl_pct=t_data["realized_pnl_pct"],
                ))
            
            logger.info(f"Portfolio loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load portfolio: {e}")
            return False
