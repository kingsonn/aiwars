"""Core type definitions for HYDRA trading system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional
import numpy as np


class Side(Enum):
    """Trade side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class PositionAction(Enum):
    """Actions for position management."""
    # For pairs WITHOUT position (Entry Mode)
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    STAY_FLAT = "stay_flat"
    
    # For pairs WITH position (Position Management Mode)
    HOLD = "hold"
    EXIT = "exit"
    FORCE_EXIT = "force_exit"  # Emergency exit (Layer 2 BLOCK)


class ThesisStatus(Enum):
    """Status of the original trade thesis."""
    INTACT = "intact"        # Original reason still valid
    WEAKENING = "weakening"  # Thesis deteriorating
    BROKEN = "broken"        # Thesis no longer valid - should exit


class OrderType(Enum):
    """Order types for execution."""
    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"
    REDUCE_ONLY = "reduce_only"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class Regime(Enum):
    """Market regime classification."""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING = auto()
    HIGH_VOLATILITY = auto()
    CASCADE_RISK = auto()
    SQUEEZE_LONG = auto()
    SQUEEZE_SHORT = auto()
    UNKNOWN = auto()


class TrapDirection(Enum):
    """Which side is trapped."""
    LONG_TRAP = "long_trap"
    SHORT_TRAP = "short_trap"
    NONE = "none"


@dataclass
class OHLCV:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    timeframe: str = "5m"

    def to_array(self) -> np.ndarray:
        return np.array([self.open, self.high, self.low, self.close, self.volume])


@dataclass
class FundingRate:
    """Funding rate data for perpetual futures."""
    timestamp: datetime
    symbol: str
    rate: float
    predicted_rate: Optional[float] = None
    next_funding_time: Optional[datetime] = None

    @property
    def annualized(self) -> float:
        """Annualized funding rate (assuming 8h periods)."""
        return self.rate * 3 * 365 * 100


@dataclass
class OpenInterest:
    """Open interest data."""
    timestamp: datetime
    symbol: str
    open_interest: float
    open_interest_usd: float
    delta: float = 0.0
    delta_pct: float = 0.0


@dataclass
class Liquidation:
    """Liquidation event data."""
    timestamp: datetime
    symbol: str
    side: Side
    quantity: float
    price: float
    usd_value: float


@dataclass
class OrderBookSnapshot:
    """Order book snapshot."""
    timestamp: datetime
    symbol: str
    bids: list[tuple[float, float]]  # (price, quantity)
    asks: list[tuple[float, float]]
    spread: float = 0.0
    imbalance: float = 0.0  # -1 to 1, positive = more bids

    def __post_init__(self):
        if self.bids and self.asks:
            best_bid = self.bids[0][0]
            best_ask = self.asks[0][0]
            self.spread = (best_ask - best_bid) / best_bid
            
            total_bids = sum(q for _, q in self.bids[:10])
            total_asks = sum(q for _, q in self.asks[:10])
            total = total_bids + total_asks
            if total > 0:
                self.imbalance = (total_bids - total_asks) / total


@dataclass
class MarketState:
    """Complete market state at a point in time."""
    timestamp: datetime
    symbol: str
    
    # Price data
    price: float
    ohlcv: dict[str, list[OHLCV]] = field(default_factory=dict)  # timeframe -> candles
    
    # Futures-specific
    funding_rate: Optional[FundingRate] = None
    open_interest: Optional[OpenInterest] = None
    recent_liquidations: list[Liquidation] = field(default_factory=list)
    order_book: Optional[OrderBookSnapshot] = None
    
    # Derived metrics
    volatility: float = 0.0
    volume_24h: float = 0.0
    price_change_24h: float = 0.0
    
    # Basis/Index
    index_price: float = 0.0
    mark_price: float = 0.0
    basis: float = 0.0  # (futures - spot) / spot


@dataclass
class Signal:
    """Trading signal from alpha models."""
    timestamp: datetime
    symbol: str
    side: Side
    confidence: float  # 0 to 1
    expected_return: float
    expected_adverse_excursion: float
    holding_period_minutes: int
    
    # Source info
    source: str = ""
    regime: Regime = Regime.UNKNOWN
    
    # Additional context
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class Position:
    """Current position state."""
    symbol: str
    side: Side
    size: float  # in contracts/coins
    size_usd: float
    entry_price: float
    current_price: float
    leverage: float
    
    # PnL
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    
    # Risk metrics
    liquidation_price: float = 0.0
    margin_used: float = 0.0
    
    # Timing
    entry_time: Optional[datetime] = None
    
    # Funding
    funding_paid: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.side != Side.FLAT and self.size > 0

    @property
    def distance_to_liquidation(self) -> float:
        """Distance to liquidation as percentage."""
        if self.liquidation_price <= 0:
            return float('inf')
        if self.side == Side.LONG:
            return (self.current_price - self.liquidation_price) / self.current_price
        else:
            return (self.liquidation_price - self.current_price) / self.current_price


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    
    # Status
    status: str = "pending"
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    
    # Timing
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Risk
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Execution
    reduce_only: bool = False
    post_only: bool = False
    
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """Executed trade record."""
    id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    fee: float
    fee_currency: str
    timestamp: datetime
    
    order_id: str = ""
    pnl: float = 0.0
    
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    timestamp: datetime
    
    # Exposure
    total_exposure_usd: float
    net_exposure_usd: float
    gross_leverage: float
    net_leverage: float
    
    # Risk measures
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    cvar_95: float  # Conditional VaR 95%
    
    # Drawdown
    current_drawdown: float
    max_drawdown: float
    
    # Correlation
    correlation_to_btc: float = 0.0
    
    # Funding
    projected_funding_8h: float = 0.0


@dataclass
class AgentVote:
    """Vote from a decision agent."""
    agent_name: str
    action: Side  # LONG, SHORT, or FLAT
    confidence: float
    reasoning: str
    veto: bool = False
    veto_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Execution plan for a trade."""
    symbol: str
    target_side: Side
    target_size: float
    target_size_usd: float
    
    # Execution params
    execution_style: str = "twap"  # twap, iceberg, limit
    num_slices: int = 1
    slice_interval_seconds: int = 30
    max_slippage_bps: float = 10.0
    
    # Limits
    price_limit: Optional[float] = None
    time_limit_seconds: int = 300
    
    # Risk
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Trading performance metrics."""
    period_start: datetime
    period_end: datetime
    
    # Returns
    total_pnl: float
    total_pnl_pct: float
    
    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown
    max_drawdown: float
    max_drawdown_duration_hours: float
    
    # Funding
    total_funding_paid: float
    total_funding_received: float
    net_funding: float
    
    # Per-trade
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    profit_factor: float
    
    # Exposure
    avg_leverage: float
    max_leverage: float


@dataclass
class PositionManagementDecision:
    """Decision for managing an existing position."""
    action: PositionAction
    thesis_status: ThesisStatus
    
    # Exit details (if action is EXIT or FORCE_EXIT)
    exit_reason: str = ""
    exit_pct: float = 1.0  # 1.0 = full exit, 0.5 = partial
    
    # Hold details (if action is HOLD)
    adjust_stop_loss: Optional[float] = None
    adjust_take_profit: Optional[float] = None
    
    # Metrics
    unrealized_pnl_pct: float = 0.0
    time_in_position_hours: float = 0.0
    funding_paid: float = 0.0
    
    # Confidence in decision
    confidence: float = 0.0
    reasoning: str = ""


@dataclass  
class EntryDecision:
    """Decision for entering a new position."""
    action: PositionAction
    side: Side
    
    # Sizing
    size_usd: float = 0.0
    leverage: float = 1.0
    
    # Risk levels
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    max_holding_hours: float = 24.0
    
    # Confidence
    confidence: float = 0.0
    thesis: str = ""
    
    # Veto info
    vetoed: bool = False
    veto_reason: str = ""
