"""
Layer 4: Risk, Leverage & Capital Brain

This layer is NON-NEGOTIABLE.

Components:
- Leverage Governance (dynamic caps, funding-aware sizing)
- Capital Throttling (risk budget, volatility scaling)
- Kill Switches (immediate flattening triggers)

HYDRA prefers not trading over dying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
import numpy as np
from loguru import logger

from hydra.core.config import HydraConfig, CORRELATION_GROUPS, PERMITTED_PAIRS
from hydra.core.types import (
    MarketState, 
    Position, 
    Signal, 
    Side, 
    Regime,
    RiskMetrics,
)
from hydra.layers.layer2_statistical import StatisticalResult


# Pre-computed correlation matrix for the 8 permitted pairs
# Based on historical BTC correlation coefficients (approximate)
PAIR_CORRELATIONS: dict[str, dict[str, float]] = {
    "cmt_btcusdt": {"cmt_btcusdt": 1.0, "cmt_ethusdt": 0.85, "cmt_solusdt": 0.75, "cmt_dogeusdt": 0.65, "cmt_xrpusdt": 0.70, "cmt_adausdt": 0.72, "cmt_bnbusdt": 0.78, "cmt_ltcusdt": 0.88},
    "cmt_ethusdt": {"cmt_btcusdt": 0.85, "cmt_ethusdt": 1.0, "cmt_solusdt": 0.80, "cmt_dogeusdt": 0.60, "cmt_xrpusdt": 0.68, "cmt_adausdt": 0.82, "cmt_bnbusdt": 0.75, "cmt_ltcusdt": 0.80},
    "cmt_solusdt": {"cmt_btcusdt": 0.75, "cmt_ethusdt": 0.80, "cmt_solusdt": 1.0, "cmt_dogeusdt": 0.55, "cmt_xrpusdt": 0.65, "cmt_adausdt": 0.70, "cmt_bnbusdt": 0.68, "cmt_ltcusdt": 0.72},
    "cmt_dogeusdt": {"cmt_btcusdt": 0.65, "cmt_ethusdt": 0.60, "cmt_solusdt": 0.55, "cmt_dogeusdt": 1.0, "cmt_xrpusdt": 0.58, "cmt_adausdt": 0.52, "cmt_bnbusdt": 0.55, "cmt_ltcusdt": 0.60},
    "cmt_xrpusdt": {"cmt_btcusdt": 0.70, "cmt_ethusdt": 0.68, "cmt_solusdt": 0.65, "cmt_dogeusdt": 0.58, "cmt_xrpusdt": 1.0, "cmt_adausdt": 0.72, "cmt_bnbusdt": 0.65, "cmt_ltcusdt": 0.68},
    "cmt_adausdt": {"cmt_btcusdt": 0.72, "cmt_ethusdt": 0.82, "cmt_solusdt": 0.70, "cmt_dogeusdt": 0.52, "cmt_xrpusdt": 0.72, "cmt_adausdt": 1.0, "cmt_bnbusdt": 0.68, "cmt_ltcusdt": 0.70},
    "cmt_bnbusdt": {"cmt_btcusdt": 0.78, "cmt_ethusdt": 0.75, "cmt_solusdt": 0.68, "cmt_dogeusdt": 0.55, "cmt_xrpusdt": 0.65, "cmt_adausdt": 0.68, "cmt_bnbusdt": 1.0, "cmt_ltcusdt": 0.75},
    "cmt_ltcusdt": {"cmt_btcusdt": 0.88, "cmt_ethusdt": 0.80, "cmt_solusdt": 0.72, "cmt_dogeusdt": 0.60, "cmt_xrpusdt": 0.68, "cmt_adausdt": 0.70, "cmt_bnbusdt": 0.75, "cmt_ltcusdt": 1.0},
}

# Volatility multipliers for each pair (relative to BTC)
PAIR_VOLATILITY_MULTIPLIER: dict[str, float] = {
    "cmt_btcusdt": 1.0,    # Base
    "cmt_ethusdt": 1.15,   # 15% more volatile than BTC
    "cmt_solusdt": 1.50,   # High volatility
    "cmt_dogeusdt": 2.00,  # Very high volatility (meme)
    "cmt_xrpusdt": 1.25,   # Moderate-high
    "cmt_adausdt": 1.30,   # Moderate-high
    "cmt_bnbusdt": 1.10,   # Similar to BTC
    "cmt_ltcusdt": 1.05,   # Very similar to BTC
}

# Max position size multiplier per pair (based on liquidity)
PAIR_SIZE_MULTIPLIER: dict[str, float] = {
    "cmt_btcusdt": 1.0,    # Full size allowed
    "cmt_ethusdt": 1.0,    # Full size allowed
    "cmt_solusdt": 0.8,    # Slightly reduced
    "cmt_dogeusdt": 0.5,   # Half size (meme volatility)
    "cmt_xrpusdt": 0.7,    # Reduced
    "cmt_adausdt": 0.6,    # Reduced
    "cmt_bnbusdt": 0.8,    # Slightly reduced
    "cmt_ltcusdt": 0.7,    # Reduced
}


@dataclass
class RiskDecision:
    """Output from Risk Brain."""
    # Approval (required fields first)
    approved: bool
    veto: bool
    
    # Position sizing (required)
    recommended_size_usd: float
    recommended_leverage: float
    max_position_pct: float  # % of max allowed
    
    # Risk limits (required)
    stop_loss_price: float
    take_profit_price: float
    max_holding_time_hours: float
    
    # Optional fields with defaults
    veto_reason: str = ""
    trigger_kill_switch: bool = False
    kill_reason: str = ""
    reduce_existing: bool = False
    reduce_to_pct: float = 1.0
    risk_score: float = 0.0  # 0-1, higher = more risky
    kelly_fraction: float = 0.0
    expected_var_95: float = 0.0


@dataclass
class PortfolioState:
    """Current portfolio state."""
    total_equity: float
    available_margin: float
    used_margin: float
    
    # Exposure
    total_exposure_usd: float
    net_exposure_usd: float
    gross_leverage: float
    
    # PnL
    unrealized_pnl: float
    realized_pnl_today: float
    
    # Drawdown
    peak_equity: float
    current_drawdown: float
    max_drawdown_today: float
    
    # Positions
    num_positions: int
    position_correlation: float  # Avg correlation between positions


class LeverageGovernor:
    """
    Dynamic leverage management.
    
    Adjusts leverage based on:
    - Volatility regime
    - Funding rate burden
    - Distance to liquidation
    - Portfolio correlation
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        self.base_leverage = config.risk.base_leverage
        self.max_leverage = config.risk.max_leverage
    
    def calculate_leverage(
        self,
        signal: Signal,
        market_state: MarketState,
        stat_result: StatisticalResult,
        portfolio: PortfolioState,
    ) -> float:
        """Calculate appropriate leverage for a trade."""
        leverage = self.base_leverage
        
        # Volatility adjustment
        if stat_result.volatility_regime == "extreme":
            leverage *= 0.3
        elif stat_result.volatility_regime == "high":
            leverage *= 0.5
        elif stat_result.volatility_regime == "low":
            leverage *= 1.2
        
        # Volatility z-score penalty
        if stat_result.volatility_zscore > 1:
            decay = self.config.risk.leverage_decay_per_sigma
            leverage *= (1 - decay * (stat_result.volatility_zscore - 1))
        
        # Funding rate adjustment
        if market_state.funding_rate:
            rate = abs(market_state.funding_rate.rate)
            if rate > self.config.risk.max_funding_rate_to_enter:
                leverage *= 0.5
            elif rate > 0.0005:
                leverage *= 0.7
        
        # Regime adjustment
        if stat_result.regime == Regime.CASCADE_RISK:
            leverage *= 0.3
        elif stat_result.regime in [Regime.SQUEEZE_LONG, Regime.SQUEEZE_SHORT]:
            leverage *= 0.6
        
        # Confidence scaling
        leverage *= signal.confidence
        
        # Portfolio correlation penalty
        if portfolio.position_correlation > 0.7:
            leverage *= 0.6
        
        # Existing exposure penalty
        if portfolio.gross_leverage > 2:
            remaining_capacity = max(0, self.max_leverage - portfolio.gross_leverage)
            leverage = min(leverage, remaining_capacity)
        
        # Drawdown penalty
        if portfolio.current_drawdown > 0.05:
            dd_penalty = 1 - (portfolio.current_drawdown / 0.15)
            leverage *= max(0.2, dd_penalty)
        
        # Apply limits
        leverage = max(1.0, min(leverage, self.max_leverage))
        
        return leverage


class PositionSizer:
    """
    Position sizing based on Kelly criterion and risk limits.
    Optimized for the 8 permitted pairs with pair-specific adjustments.
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        self.kelly_fraction = config.risk.kelly_fraction
        self.risk_per_trade = config.risk.risk_per_trade_pct / 100
    
    def calculate_size(
        self,
        signal: Signal,
        leverage: float,
        portfolio: PortfolioState,
        expected_adverse_excursion: float,
        existing_positions: Optional[dict[str, Position]] = None,
    ) -> tuple[float, float]:
        """
        Calculate position size with pair-specific and correlation adjustments.
        
        Returns:
            (size_usd, kelly_fraction_used)
        """
        symbol = signal.symbol.lower()
        equity = portfolio.total_equity
        
        # Kelly sizing
        if signal.expected_return > 0 and expected_adverse_excursion > 0:
            win_prob = signal.confidence
            win_loss_ratio = signal.expected_return / expected_adverse_excursion
            kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
            kelly = max(0, kelly)
        else:
            kelly = 0.1
        
        # Apply fraction (quarter Kelly typical)
        kelly_size = equity * kelly * self.kelly_fraction
        
        # Risk-based sizing
        if expected_adverse_excursion > 0:
            risk_size = (equity * self.risk_per_trade) / expected_adverse_excursion
        else:
            risk_size = equity * 0.1
        
        # Take minimum of approaches
        size = min(kelly_size, risk_size)
        
        # Apply leverage
        size *= leverage
        
        # === PAIR-SPECIFIC ADJUSTMENTS ===
        
        # 1. Volatility adjustment - reduce size for more volatile pairs
        vol_mult = PAIR_VOLATILITY_MULTIPLIER.get(symbol, 1.0)
        size /= vol_mult  # Higher vol = smaller size
        
        # 2. Liquidity/size multiplier
        size_mult = PAIR_SIZE_MULTIPLIER.get(symbol, 1.0)
        size *= size_mult
        
        # 3. Correlation-based reduction if holding correlated positions
        if existing_positions:
            corr_penalty = self._calculate_correlation_penalty(symbol, existing_positions)
            size *= (1 - corr_penalty)
        
        # Apply limits
        size = min(size, self.config.trading.max_position_size_usd * size_mult)
        size = min(size, portfolio.available_margin * leverage * 0.9)
        
        # Max positions limit
        if portfolio.num_positions >= self.config.trading.max_positions:
            size = 0
        
        # Total exposure limit
        remaining_exposure = self.config.trading.max_total_exposure_usd - portfolio.total_exposure_usd
        size = min(size, max(0, remaining_exposure))
        
        return size, kelly
    
    def _calculate_correlation_penalty(
        self, 
        new_symbol: str, 
        existing_positions: dict[str, Position]
    ) -> float:
        """
        Calculate size reduction based on correlation with existing positions.
        Returns penalty between 0 (no reduction) and 0.7 (max 70% reduction).
        """
        if not existing_positions:
            return 0.0
        
        new_symbol = new_symbol.lower()
        total_corr_exposure = 0.0
        total_exposure = 0.0
        
        for sym, pos in existing_positions.items():
            if not pos.is_open:
                continue
            sym = sym.lower()
            if sym in PAIR_CORRELATIONS and new_symbol in PAIR_CORRELATIONS.get(sym, {}):
                corr = PAIR_CORRELATIONS[sym].get(new_symbol, 0.5)
                # Weight by position size
                exposure = abs(pos.size_usd)
                total_corr_exposure += corr * exposure
                total_exposure += exposure
        
        if total_exposure == 0:
            return 0.0
        
        # Average correlation weighted by exposure
        avg_corr = total_corr_exposure / total_exposure
        
        # High correlation (>0.8) = up to 70% size reduction
        # Moderate correlation (0.5-0.8) = 20-50% reduction
        # Low correlation (<0.5) = minimal reduction
        if avg_corr > 0.8:
            return 0.7
        elif avg_corr > 0.6:
            return 0.4
        elif avg_corr > 0.4:
            return 0.2
        return 0.0


class KillSwitchManager:
    """
    Kill switch management - immediate risk circuit breakers.
    
    Triggers immediate flattening if:
    - Funding spikes beyond modeled bounds
    - Liquidation velocity explodes
    - Model disagreement exceeds threshold
    - Statistical reality layer flags regime break
    - Drawdown limits exceeded
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        
        # State tracking
        self._hourly_pnl: list[tuple[datetime, float]] = []
        self._last_check: Optional[datetime] = None
        self._kill_triggered = False
        self._kill_time: Optional[datetime] = None
    
    def check(
        self,
        portfolio: PortfolioState,
        market_state: MarketState,
        stat_result: StatisticalResult,
        model_disagreement: float = 0.0,
    ) -> tuple[bool, str]:
        """
        Check all kill switch conditions.
        
        Returns:
            (should_kill, reason)
        """
        now = datetime.now(timezone.utc)
        
        # Cooldown after kill
        if self._kill_triggered and self._kill_time:
            if (now - self._kill_time).total_seconds() < 3600:  # 1 hour cooldown
                return False, ""
        
        # Check conditions
        checks = [
            self._check_drawdown(portfolio),
            self._check_funding_spike(market_state),
            self._check_cascade_risk(stat_result),
            self._check_regime_break(stat_result),
            self._check_model_disagreement(model_disagreement),
            self._check_hourly_loss(portfolio, now),
        ]
        
        for should_kill, reason in checks:
            if should_kill:
                self._kill_triggered = True
                self._kill_time = now
                logger.critical(f"KILL SWITCH: {reason}")
                return True, reason
        
        return False, ""
    
    def _check_drawdown(self, portfolio: PortfolioState) -> tuple[bool, str]:
        """Check daily drawdown limit."""
        if portfolio.current_drawdown > self.config.risk.max_daily_drawdown_pct / 100:
            return True, f"Daily drawdown {portfolio.current_drawdown:.1%} exceeds {self.config.risk.max_daily_drawdown_pct}%"
        return False, ""
    
    def _check_funding_spike(self, market_state: MarketState) -> tuple[bool, str]:
        """Check for abnormal funding rates."""
        if market_state.funding_rate:
            rate = abs(market_state.funding_rate.rate)
            if rate > 0.005:  # 0.5% per 8h = extreme
                return True, f"Funding rate spike: {rate*100:.3f}%"
        return False, ""
    
    def _check_cascade_risk(self, stat_result: StatisticalResult) -> tuple[bool, str]:
        """Check for cascade/liquidation cascade."""
        if stat_result.cascade_probability > 0.7:
            return True, f"Cascade probability {stat_result.cascade_probability:.1%}"
        if stat_result.liquidation_velocity > 10:  # Normalized threshold
            return True, f"Liquidation velocity spike: {stat_result.liquidation_velocity:.1f}"
        return False, ""
    
    def _check_regime_break(self, stat_result: StatisticalResult) -> tuple[bool, str]:
        """Check for regime break with high uncertainty."""
        if stat_result.regime_break_alert and stat_result.volatility_regime == "extreme":
            return True, "Regime break with extreme volatility"
        return False, ""
    
    def _check_model_disagreement(self, disagreement: float) -> tuple[bool, str]:
        """Check model disagreement threshold."""
        if disagreement > self.config.risk.max_model_disagreement:
            return True, f"Model disagreement {disagreement:.1%} exceeds threshold"
        return False, ""
    
    def _check_hourly_loss(
        self, portfolio: PortfolioState, now: datetime
    ) -> tuple[bool, str]:
        """Check hourly loss limit."""
        # Track PnL
        self._hourly_pnl.append((now, portfolio.unrealized_pnl + portfolio.realized_pnl_today))
        
        # Clean old entries
        cutoff = now - timedelta(hours=1)
        self._hourly_pnl = [(t, p) for t, p in self._hourly_pnl if t > cutoff]
        
        if len(self._hourly_pnl) >= 2:
            oldest_pnl = self._hourly_pnl[0][1]
            current_pnl = self._hourly_pnl[-1][1]
            hourly_change = current_pnl - oldest_pnl
            
            hourly_loss_pct = -hourly_change / portfolio.total_equity if hourly_change < 0 else 0
            
            if hourly_loss_pct > self.config.risk.max_hourly_drawdown_pct / 100:
                return True, f"Hourly loss {hourly_loss_pct:.1%} exceeds limit"
        
        return False, ""
    
    def reset(self) -> None:
        """Reset kill switch state."""
        self._kill_triggered = False
        self._kill_time = None
        logger.info("Kill switch reset")


class RiskCapitalBrain:
    """
    Layer 4: Risk, Leverage & Capital Brain
    
    This layer is the final gatekeeper before execution.
    It can veto any trade and trigger kill switches.
    
    Core principle: HYDRA prefers not trading over dying.
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        
        self.leverage_governor = LeverageGovernor(config)
        self.position_sizer = PositionSizer(config)
        self.kill_switch = KillSwitchManager(config)
        
        # Portfolio tracking
        self._portfolio: Optional[PortfolioState] = None
        self._peak_equity: float = 0.0
        
        logger.info("Risk Capital Brain initialized")
    
    async def setup(self) -> None:
        """Initialize risk brain."""
        pass
    
    async def evaluate(
        self,
        signal: Signal,
        market_state: MarketState,
        stat_result: StatisticalResult,
        current_position: Optional[Position],
        all_positions: dict[str, Position],
    ) -> RiskDecision:
        """
        Evaluate a trading signal through risk lens.
        
        This is the final approval gate before execution.
        """
        # Build portfolio state
        portfolio = self._build_portfolio_state(all_positions)
        self._portfolio = portfolio
        
        # Check kill switches first
        kill, kill_reason = self.kill_switch.check(
            portfolio, market_state, stat_result
        )
        
        if kill:
            return RiskDecision(
                approved=False,
                veto=True,
                veto_reason=kill_reason,
                recommended_size_usd=0,
                recommended_leverage=0,
                max_position_pct=0,
                stop_loss_price=0,
                take_profit_price=0,
                max_holding_time_hours=0,
                trigger_kill_switch=True,
                kill_reason=kill_reason,
            )
        
        # Check if we should reduce existing position first
        if current_position and current_position.is_open:
            should_reduce, reduce_reason, reduce_to = self._check_position_health(
                current_position, market_state, stat_result
            )
            
            if should_reduce:
                return RiskDecision(
                    approved=False,
                    veto=True,
                    veto_reason=reduce_reason,
                    recommended_size_usd=0,
                    recommended_leverage=0,
                    max_position_pct=0,
                    stop_loss_price=0,
                    take_profit_price=0,
                    max_holding_time_hours=0,
                    reduce_existing=True,
                    reduce_to_pct=reduce_to,
                )
        
        # Calculate appropriate leverage
        leverage = self.leverage_governor.calculate_leverage(
            signal, market_state, stat_result, portfolio
        )
        
        # Calculate position size
        size_usd, kelly = self.position_sizer.calculate_size(
            signal, leverage, portfolio, signal.expected_adverse_excursion
        )
        
        # Veto if size too small
        if size_usd < 100:  # Minimum viable size
            return RiskDecision(
                approved=False,
                veto=True,
                veto_reason="Position size too small after risk adjustment",
                recommended_size_usd=0,
                recommended_leverage=0,
                max_position_pct=0,
                stop_loss_price=0,
                take_profit_price=0,
                max_holding_time_hours=0,
            )
        
        # Veto if funding rate unfavorable
        if market_state.funding_rate:
            rate = market_state.funding_rate.rate
            if signal.side == Side.LONG and rate > self.config.risk.max_funding_rate_to_enter:
                return RiskDecision(
                    approved=False,
                    veto=True,
                    veto_reason=f"Funding rate {rate*100:.3f}% unfavorable for long",
                    recommended_size_usd=0,
                    recommended_leverage=0,
                    max_position_pct=0,
                    stop_loss_price=0,
                    take_profit_price=0,
                    max_holding_time_hours=0,
                )
            elif signal.side == Side.SHORT and rate < -self.config.risk.max_funding_rate_to_enter:
                return RiskDecision(
                    approved=False,
                    veto=True,
                    veto_reason=f"Funding rate {rate*100:.3f}% unfavorable for short",
                    recommended_size_usd=0,
                    recommended_leverage=0,
                    max_position_pct=0,
                    stop_loss_price=0,
                    take_profit_price=0,
                    max_holding_time_hours=0,
                )
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_exit_levels(
            signal, market_state, stat_result
        )
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            signal, market_state, stat_result, portfolio, leverage
        )
        
        # Calculate VaR
        var_95 = self._calculate_var(size_usd, stat_result.realized_volatility)
        
        # Holding time based on regime
        if stat_result.regime in [Regime.TRENDING_UP, Regime.TRENDING_DOWN]:
            max_holding = 8.0
        elif stat_result.regime == Regime.HIGH_VOLATILITY:
            max_holding = 1.0
        else:
            max_holding = 4.0
        
        return RiskDecision(
            approved=True,
            veto=False,
            recommended_size_usd=size_usd,
            recommended_leverage=leverage,
            max_position_pct=size_usd / self.config.trading.max_position_size_usd,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            max_holding_time_hours=max_holding,
            risk_score=risk_score,
            kelly_fraction=kelly,
            expected_var_95=var_95,
        )
    
    def _build_portfolio_state(self, positions: dict[str, Position]) -> PortfolioState:
        """Build current portfolio state from positions."""
        # This would normally come from exchange
        total_equity = 100000  # Placeholder
        
        total_exposure = sum(p.size_usd for p in positions.values() if p.is_open)
        net_exposure = sum(
            p.size_usd if p.side == Side.LONG else -p.size_usd
            for p in positions.values() if p.is_open
        )
        
        unrealized = sum(p.unrealized_pnl for p in positions.values() if p.is_open)
        
        # Track peak equity
        current_equity = total_equity + unrealized
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        
        drawdown = (self._peak_equity - current_equity) / self._peak_equity if self._peak_equity > 0 else 0
        
        return PortfolioState(
            total_equity=total_equity,
            available_margin=total_equity - sum(p.margin_used for p in positions.values() if p.is_open),
            used_margin=sum(p.margin_used for p in positions.values() if p.is_open),
            total_exposure_usd=total_exposure,
            net_exposure_usd=net_exposure,
            gross_leverage=total_exposure / total_equity if total_equity > 0 else 0,
            unrealized_pnl=unrealized,
            realized_pnl_today=0,  # Would track separately
            peak_equity=self._peak_equity,
            current_drawdown=drawdown,
            max_drawdown_today=drawdown,  # Would track separately
            num_positions=len([p for p in positions.values() if p.is_open]),
            position_correlation=0.5,  # Would calculate from returns
        )
    
    def _check_position_health(
        self,
        position: Position,
        market_state: MarketState,
        stat_result: StatisticalResult,
    ) -> tuple[bool, str, float]:
        """Check if existing position needs to be reduced."""
        # Liquidation too close
        if position.distance_to_liquidation < self.config.risk.min_liquidation_distance_pct / 100:
            return True, "Liquidation distance too close", 0.5
        
        # Regime change against position
        if position.side == Side.LONG and stat_result.regime == Regime.TRENDING_DOWN:
            return True, "Regime turned bearish", 0.5
        elif position.side == Side.SHORT and stat_result.regime == Regime.TRENDING_UP:
            return True, "Regime turned bullish", 0.5
        
        # Large unrealized loss
        if position.unrealized_pnl_pct < -0.1:  # -10%
            return True, "Large unrealized loss", 0.5
        
        # Funding burden too high
        if market_state.funding_rate:
            rate = market_state.funding_rate.rate
            if position.side == Side.LONG and rate > 0.001:
                return True, "High funding burden for long", 0.7
            elif position.side == Side.SHORT and rate < -0.001:
                return True, "High funding burden for short", 0.7
        
        return False, "", 1.0
    
    def _calculate_exit_levels(
        self,
        signal: Signal,
        market_state: MarketState,
        stat_result: StatisticalResult,
    ) -> tuple[float, float]:
        """Calculate stop loss and take profit prices."""
        price = market_state.price
        
        # Base on expected adverse excursion
        eae = signal.expected_adverse_excursion
        if eae <= 0:
            eae = stat_result.realized_volatility * 2  # 2 sigma
        
        # Adjust for volatility
        if stat_result.volatility_regime == "high":
            eae *= 1.3
        elif stat_result.volatility_regime == "extreme":
            eae *= 1.5
        
        # Cap at reasonable levels
        eae = min(eae, 0.1)  # Max 10% stop
        eae = max(eae, 0.01)  # Min 1% stop
        
        # Calculate levels
        if signal.side == Side.LONG:
            stop_loss = price * (1 - eae)
            take_profit = price * (1 + eae * 2)  # 2:1 R:R minimum
        else:
            stop_loss = price * (1 + eae)
            take_profit = price * (1 - eae * 2)
        
        return stop_loss, take_profit
    
    def _calculate_risk_score(
        self,
        signal: Signal,
        market_state: MarketState,
        stat_result: StatisticalResult,
        portfolio: PortfolioState,
        leverage: float,
    ) -> float:
        """Calculate overall risk score (0-1)."""
        score = 0.0
        
        # Volatility contribution
        if stat_result.volatility_regime == "extreme":
            score += 0.3
        elif stat_result.volatility_regime == "high":
            score += 0.2
        
        # Leverage contribution
        score += (leverage / self.config.risk.max_leverage) * 0.2
        
        # Drawdown contribution
        score += portfolio.current_drawdown * 2
        
        # Cascade risk
        score += stat_result.cascade_probability * 0.2
        
        # Funding stress
        if market_state.funding_rate:
            score += abs(market_state.funding_rate.rate) * 100
        
        return min(1.0, score)
    
    def _calculate_var(self, size_usd: float, volatility: float) -> float:
        """Calculate 95% VaR."""
        # Assuming normal distribution (conservative)
        z_95 = 1.645
        daily_vol = volatility / np.sqrt(365)
        var_95 = size_usd * daily_vol * z_95
        return var_95
    
    def get_risk_metrics(self) -> Optional[RiskMetrics]:
        """Get current risk metrics."""
        if not self._portfolio:
            return None
        
        p = self._portfolio
        
        return RiskMetrics(
            timestamp=datetime.now(timezone.utc),
            total_exposure_usd=p.total_exposure_usd,
            net_exposure_usd=p.net_exposure_usd,
            gross_leverage=p.gross_leverage,
            net_leverage=abs(p.net_exposure_usd) / p.total_equity if p.total_equity > 0 else 0,
            var_95=0,  # Would calculate properly
            var_99=0,
            cvar_95=0,
            current_drawdown=p.current_drawdown,
            max_drawdown=p.max_drawdown_today,
        )
