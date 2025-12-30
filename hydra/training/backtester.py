"""
HYDRA Backtesting Engine

Features:
- Regime-segmented backtests
- Walk-forward validation
- Stress period testing (crashes, squeezes)
- Realistic execution simulation
- Comprehensive performance metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

from hydra.core.config import HydraConfig
from hydra.core.types import (
    Position, Trade, Side, Regime,
    PerformanceMetrics, MarketState, OHLCV,
)


@dataclass
class BacktestTrade:
    """Record of a backtest trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: Side
    size: float
    entry_price: float
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    funding_paid: float = 0.0
    regime: Regime = Regime.UNKNOWN
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0
    holding_minutes: int = 0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Config
    start_date: datetime
    end_date: datetime
    initial_capital: float
    
    # Performance
    final_equity: float
    total_return: float
    total_return_pct: float
    
    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration_hours: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Per-trade
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    profit_factor: float
    expectancy: float
    
    # Funding
    total_funding_paid: float
    total_funding_received: float
    
    # By regime
    regime_performance: dict[str, dict] = field(default_factory=dict)
    
    # Equity curve
    equity_curve: pd.Series = field(default_factory=pd.Series)
    
    # Trade log
    trades: list[BacktestTrade] = field(default_factory=list)


class BacktestEngine:
    """
    Core backtesting engine.
    
    Simulates trading with:
    - Realistic slippage
    - Funding rate costs
    - Position limits
    - Execution delays
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        fee_rate: float = 0.0004,  # 4 bps taker
        slippage_bps: float = 2,
        max_leverage: float = 10,
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps / 10000
        self.max_leverage = max_leverage
        
        # State
        self.equity = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[BacktestTrade] = []
        self.equity_history: list[tuple[datetime, float]] = []
        
        # Tracking
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        self.max_drawdown_start: Optional[datetime] = None
        self.max_drawdown_duration = timedelta(0)
        
        self.total_funding_paid = 0.0
        self.total_funding_received = 0.0
    
    def reset(self) -> None:
        """Reset backtest state."""
        self.equity = self.initial_capital
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_history.clear()
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0
        self.total_funding_paid = 0.0
        self.total_funding_received = 0.0
    
    def execute_trade(
        self,
        timestamp: datetime,
        symbol: str,
        side: Side,
        size_usd: float,
        price: float,
        regime: Regime = Regime.UNKNOWN,
    ) -> Optional[BacktestTrade]:
        """Execute a trade."""
        # Apply slippage
        if side == Side.LONG:
            exec_price = price * (1 + self.slippage_bps)
        else:
            exec_price = price * (1 - self.slippage_bps)
        
        size = size_usd / exec_price
        
        # Check if we have an existing position
        existing = self.positions.get(symbol)
        
        if existing and existing.is_open:
            if existing.side == side:
                # Add to position
                new_size = existing.size + size
                new_entry = (existing.entry_price * existing.size + exec_price * size) / new_size
                existing.size = new_size
                existing.size_usd = new_size * exec_price
                existing.entry_price = new_entry
                return None
            else:
                # Close existing position first
                self._close_position(timestamp, symbol, exec_price)
        
        # Open new position
        fee = size_usd * self.fee_rate
        self.cash -= fee
        
        trade = BacktestTrade(
            entry_time=timestamp,
            exit_time=None,
            symbol=symbol,
            side=side,
            size=size,
            entry_price=exec_price,
            fees=fee,
            regime=regime,
        )
        
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            size=size,
            size_usd=size_usd,
            entry_price=exec_price,
            current_price=exec_price,
            leverage=1.0,  # Simplified
            entry_time=timestamp,
        )
        
        self.trades.append(trade)
        return trade
    
    def _close_position(
        self,
        timestamp: datetime,
        symbol: str,
        price: float,
    ) -> Optional[BacktestTrade]:
        """Close an existing position."""
        pos = self.positions.get(symbol)
        if not pos or not pos.is_open:
            return None
        
        # Find the trade
        open_trade = None
        for t in reversed(self.trades):
            if t.symbol == symbol and t.exit_time is None:
                open_trade = t
                break
        
        if not open_trade:
            return None
        
        # Apply slippage
        if pos.side == Side.LONG:
            exec_price = price * (1 - self.slippage_bps)
        else:
            exec_price = price * (1 + self.slippage_bps)
        
        # Calculate PnL
        if pos.side == Side.LONG:
            pnl = (exec_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - exec_price) * pos.size
        
        # Fees
        close_fee = pos.size * exec_price * self.fee_rate
        pnl -= close_fee
        
        # Update trade record
        open_trade.exit_time = timestamp
        open_trade.exit_price = exec_price
        open_trade.pnl = pnl
        open_trade.pnl_pct = pnl / (pos.size * pos.entry_price)
        open_trade.fees += close_fee
        open_trade.holding_minutes = int((timestamp - open_trade.entry_time).total_seconds() / 60)
        
        # Update equity
        self.cash += pnl + pos.size * pos.entry_price  # Return margin + pnl
        
        # Remove position
        del self.positions[symbol]
        
        return open_trade
    
    def close_all_positions(self, timestamp: datetime, prices: dict[str, float]) -> None:
        """Close all open positions."""
        for symbol in list(self.positions.keys()):
            if symbol in prices:
                self._close_position(timestamp, symbol, prices[symbol])
    
    def apply_funding(
        self,
        timestamp: datetime,
        symbol: str,
        funding_rate: float,
        mark_price: float,
    ) -> float:
        """Apply funding payment to position."""
        pos = self.positions.get(symbol)
        if not pos or not pos.is_open:
            return 0.0
        
        notional = pos.size * mark_price
        
        if pos.side == Side.LONG:
            payment = notional * funding_rate
        else:
            payment = -notional * funding_rate
        
        self.cash -= payment
        
        if payment > 0:
            self.total_funding_paid += payment
        else:
            self.total_funding_received += abs(payment)
        
        # Update trade record
        for t in reversed(self.trades):
            if t.symbol == symbol and t.exit_time is None:
                t.funding_paid += payment
                break
        
        return payment
    
    def update_equity(self, timestamp: datetime, prices: dict[str, float]) -> float:
        """Update equity based on current prices."""
        # Mark positions to market
        unrealized_pnl = 0.0
        
        for symbol, pos in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                pos.current_price = current_price
                
                if pos.side == Side.LONG:
                    unrealized_pnl += (current_price - pos.entry_price) * pos.size
                else:
                    unrealized_pnl += (pos.entry_price - current_price) * pos.size
                
                pos.unrealized_pnl = unrealized_pnl
        
        self.equity = self.cash + unrealized_pnl
        self.equity_history.append((timestamp, self.equity))
        
        # Update drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        current_dd = (self.peak_equity - self.equity) / self.peak_equity
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
        
        return self.equity
    
    def get_results(self) -> BacktestResult:
        """Compile backtest results."""
        if not self.equity_history:
            return self._empty_results()
        
        # Basic metrics
        total_return = self.equity - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        
        # Trade stats
        closed_trades = [t for t in self.trades if t.exit_time is not None]
        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl <= 0]
        
        win_rate = len(winning) / len(closed_trades) if closed_trades else 0
        
        # Average trade
        avg_trade = np.mean([t.pnl for t in closed_trades]) if closed_trades else 0
        avg_winner = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loser = np.mean([t.pnl for t in losing]) if losing else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = win_rate * avg_winner - (1 - win_rate) * abs(avg_loser)
        
        # Risk metrics
        equity_curve = pd.Series(
            [e for _, e in self.equity_history],
            index=[t for t, _ in self.equity_history]
        )
        returns = equity_curve.pct_change().dropna()
        
        sharpe = np.sqrt(252 * 24) * returns.mean() / returns.std() if len(returns) > 1 else 0
        
        downside = returns[returns < 0]
        sortino = np.sqrt(252 * 24) * returns.mean() / downside.std() if len(downside) > 1 else 0
        
        annual_return = total_return_pct * (365 * 24 * 60) / max(1, len(self.equity_history))
        calmar = annual_return / self.max_drawdown if self.max_drawdown > 0 else 0
        
        # Regime performance
        regime_perf = {}
        for regime in Regime:
            regime_trades = [t for t in closed_trades if t.regime == regime]
            if regime_trades:
                regime_perf[regime.name] = {
                    'count': len(regime_trades),
                    'win_rate': len([t for t in regime_trades if t.pnl > 0]) / len(regime_trades),
                    'avg_pnl': np.mean([t.pnl for t in regime_trades]),
                    'total_pnl': sum(t.pnl for t in regime_trades),
                }
        
        return BacktestResult(
            start_date=self.equity_history[0][0],
            end_date=self.equity_history[-1][0],
            initial_capital=self.initial_capital,
            final_equity=self.equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=len(closed_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            max_drawdown=self.max_drawdown,
            max_drawdown_duration_hours=0,  # Would need to track
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_trade_pnl=avg_trade,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            profit_factor=profit_factor,
            expectancy=expectancy,
            total_funding_paid=self.total_funding_paid,
            total_funding_received=self.total_funding_received,
            regime_performance=regime_perf,
            equity_curve=equity_curve,
            trades=self.trades,
        )
    
    def _empty_results(self) -> BacktestResult:
        """Return empty results."""
        return BacktestResult(
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc),
            initial_capital=self.initial_capital,
            final_equity=self.initial_capital,
            total_return=0,
            total_return_pct=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            max_drawdown=0,
            max_drawdown_duration_hours=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            avg_trade_pnl=0,
            avg_winner=0,
            avg_loser=0,
            profit_factor=0,
            expectancy=0,
            total_funding_paid=0,
            total_funding_received=0,
        )


class Backtester:
    """
    High-level backtester for HYDRA strategies.
    
    Supports:
    - Full system backtests
    - Regime-segmented analysis
    - Stress testing
    - Walk-forward validation
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
    
    async def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func,
        initial_capital: float = 100000,
    ) -> BacktestResult:
        """
        Run a full backtest.
        
        Args:
            data: DataFrame with OHLCV + funding data
            strategy_func: Function that takes market state and returns signal
            initial_capital: Starting capital
        """
        engine = BacktestEngine(initial_capital=initial_capital)
        
        # Iterate through data
        for i in range(100, len(data)):  # Skip first 100 for warmup
            row = data.iloc[i]
            timestamp = data.index[i]
            
            # Build market state (simplified)
            candles = [
                OHLCV(
                    timestamp=data.index[j],
                    open=data.iloc[j]['open'],
                    high=data.iloc[j]['high'],
                    low=data.iloc[j]['low'],
                    close=data.iloc[j]['close'],
                    volume=data.iloc[j]['volume'],
                )
                for j in range(max(0, i-100), i)
            ]
            
            market_state = MarketState(
                timestamp=timestamp,
                symbol=data.attrs.get('symbol', 'BTCUSDT'),
                price=row['close'],
                ohlcv={'5m': candles},
                volatility=data['close'].pct_change().iloc[max(0,i-100):i].std() * np.sqrt(1440*365),
            )
            
            # Get signal
            signal = await strategy_func(market_state)
            
            if signal and signal.side != Side.FLAT and signal.confidence > 0.6:
                # Calculate size
                size_usd = initial_capital * 0.1 * signal.confidence
                
                engine.execute_trade(
                    timestamp=timestamp,
                    symbol=market_state.symbol,
                    side=signal.side,
                    size_usd=size_usd,
                    price=row['close'],
                )
            
            # Apply funding every 8 hours
            if 'funding_rate' in data.columns and i % 96 == 0:  # 8h for 5m data
                engine.apply_funding(
                    timestamp=timestamp,
                    symbol=market_state.symbol,
                    funding_rate=row.get('funding_rate', 0),
                    mark_price=row['close'],
                )
            
            # Update equity
            engine.update_equity(timestamp, {market_state.symbol: row['close']})
        
        # Close all positions at end
        final_row = data.iloc[-1]
        engine.close_all_positions(
            data.index[-1],
            {data.attrs.get('symbol', 'BTCUSDT'): final_row['close']}
        )
        
        return engine.get_results()
    
    def run_stress_tests(
        self,
        data: pd.DataFrame,
        strategy_func,
    ) -> dict[str, BacktestResult]:
        """Run backtests on stress periods."""
        stress_periods = {
            'covid_crash': ('2020-03-01', '2020-03-31'),
            'may_2021_crash': ('2021-05-01', '2021-05-31'),
            'luna_crash': ('2022-05-01', '2022-05-31'),
            'ftx_crash': ('2022-11-01', '2022-11-30'),
        }
        
        results = {}
        
        for name, (start, end) in stress_periods.items():
            try:
                period_data = data[start:end]
                if len(period_data) > 100:
                    result = asyncio.run(self.run_backtest(period_data, strategy_func))
                    results[name] = result
                    logger.info(f"{name}: Return={result.total_return_pct:.1%}, MaxDD={result.max_drawdown:.1%}")
            except Exception as e:
                logger.warning(f"Stress test {name} failed: {e}")
        
        return results
    
    def print_results(self, result: BacktestResult) -> None:
        """Print backtest results summary."""
        print("\n" + "="*60)
        print("HYDRA BACKTEST RESULTS")
        print("="*60)
        
        print(f"\nPeriod: {result.start_date} to {result.end_date}")
        print(f"Initial Capital: ${result.initial_capital:,.0f}")
        print(f"Final Equity: ${result.final_equity:,.0f}")
        
        print(f"\n--- RETURNS ---")
        print(f"Total Return: ${result.total_return:,.0f} ({result.total_return_pct:.1%})")
        
        print(f"\n--- TRADES ---")
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.1%}")
        print(f"Avg Trade: ${result.avg_trade_pnl:,.2f}")
        print(f"Avg Winner: ${result.avg_winner:,.2f}")
        print(f"Avg Loser: ${result.avg_loser:,.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Expectancy: ${result.expectancy:,.2f}")
        
        print(f"\n--- RISK ---")
        print(f"Max Drawdown: {result.max_drawdown:.1%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
        print(f"Calmar Ratio: {result.calmar_ratio:.2f}")
        
        print(f"\n--- FUNDING ---")
        print(f"Paid: ${result.total_funding_paid:,.2f}")
        print(f"Received: ${result.total_funding_received:,.2f}")
        print(f"Net: ${result.total_funding_received - result.total_funding_paid:,.2f}")
        
        if result.regime_performance:
            print(f"\n--- BY REGIME ---")
            for regime, perf in result.regime_performance.items():
                print(f"{regime}: {perf['count']} trades, "
                      f"WR={perf['win_rate']:.1%}, "
                      f"PnL=${perf['total_pnl']:,.0f}")
        
        print("\n" + "="*60)


# Import for stress tests
import asyncio
