"""
Market Simulator for HYDRA Training

Multi-agent self-play simulation for RL training.
Includes adversarial traders and synthetic cascades.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple
import numpy as np
from enum import IntEnum

from hydra.core.types import Side, Regime


class SimAction(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3


@dataclass
class SimState:
    """Simulation state for RL environment."""
    price: float
    position: float  # Positive = long, negative = short
    cash: float
    pnl: float
    funding_rate: float
    volatility: float
    momentum: float
    regime: int
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.price / 50000,  # Normalize
            self.position / 10,
            self.cash / 100000,
            self.pnl / 1000,
            self.funding_rate * 10000,
            self.volatility * 10,
            self.momentum * 10,
            self.regime / 5,
        ], dtype=np.float32)


class MarketSimulator:
    """
    Simulated market environment for RL training.
    
    Features:
    - Realistic price dynamics (GBM + jumps)
    - Funding rate simulation
    - Liquidation mechanics
    - Adversarial agents
    """
    
    def __init__(
        self,
        initial_price: float = 50000,
        initial_cash: float = 100000,
        volatility: float = 0.02,
        jump_probability: float = 0.01,
        funding_mean: float = 0.0001,
    ):
        self.initial_price = initial_price
        self.initial_cash = initial_cash
        self.base_volatility = volatility
        self.jump_probability = jump_probability
        self.funding_mean = funding_mean
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.price = self.initial_price
        self.cash = self.initial_cash
        self.position = 0.0
        self.entry_price = 0.0
        self.pnl = 0.0
        self.step_count = 0
        
        self.funding_rate = np.random.normal(self.funding_mean, 0.0002)
        self.volatility = self.base_volatility
        self.momentum = 0.0
        self.regime = 2  # Ranging
        
        self.price_history = [self.price]
        
        return self._get_state().to_array()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take a step in the environment."""
        self.step_count += 1
        
        # Execute action
        reward = self._execute_action(action)
        
        # Update market
        self._update_market()
        
        # Apply funding (every 480 steps = 8 hours at 1 min)
        if self.step_count % 480 == 0:
            reward += self._apply_funding()
        
        # Check termination
        done = self._check_done()
        
        # Update PnL
        if self.position != 0:
            if self.position > 0:
                self.pnl = (self.price - self.entry_price) * self.position
            else:
                self.pnl = (self.entry_price - self.price) * abs(self.position)
        
        state = self._get_state()
        
        info = {
            'price': self.price,
            'position': self.position,
            'pnl': self.pnl,
            'equity': self.cash + self.pnl,
        }
        
        return state.to_array(), reward, done, info
    
    def _execute_action(self, action: int) -> float:
        """Execute trading action."""
        reward = 0.0
        trade_size = 0.1  # 10% of capital per trade
        
        if action == SimAction.BUY:
            if self.position <= 0:
                # Close short if exists
                if self.position < 0:
                    reward += self._close_position()
                # Open long
                size = (self.cash * trade_size) / self.price
                self.position = size
                self.entry_price = self.price
                self.cash -= size * self.price * 0.0004  # Fee
                
        elif action == SimAction.SELL:
            if self.position >= 0:
                # Close long if exists
                if self.position > 0:
                    reward += self._close_position()
                # Open short
                size = (self.cash * trade_size) / self.price
                self.position = -size
                self.entry_price = self.price
                self.cash -= size * self.price * 0.0004  # Fee
                
        elif action == SimAction.CLOSE:
            if self.position != 0:
                reward += self._close_position()
        
        return reward
    
    def _close_position(self) -> float:
        """Close current position and return PnL."""
        if self.position == 0:
            return 0.0
        
        if self.position > 0:
            pnl = (self.price - self.entry_price) * self.position
        else:
            pnl = (self.entry_price - self.price) * abs(self.position)
        
        # Fee
        pnl -= abs(self.position) * self.price * 0.0004
        
        self.cash += pnl
        self.position = 0.0
        self.entry_price = 0.0
        self.pnl = 0.0
        
        return pnl / 1000  # Normalized reward
    
    def _update_market(self) -> None:
        """Update market state."""
        # GBM with jumps
        dt = 1 / 1440  # 1 minute
        
        # Regime-dependent dynamics
        if self.regime == 0:  # Trending up
            drift = 0.0005
        elif self.regime == 1:  # Trending down
            drift = -0.0005
        else:
            drift = 0.0
        
        # Random walk
        dW = np.random.normal(0, 1)
        return_pct = drift + self.volatility * np.sqrt(dt) * dW
        
        # Jump component
        if np.random.random() < self.jump_probability:
            jump = np.random.normal(0, 0.03)  # 3% jump
            return_pct += jump
            self.regime = 3 if abs(jump) > 0.02 else self.regime
        
        self.price *= (1 + return_pct)
        self.price_history.append(self.price)
        
        # Update derived state
        if len(self.price_history) > 20:
            prices = np.array(self.price_history[-20:])
            returns = np.diff(prices) / prices[:-1]  # 19 diffs / 19 prices
            self.volatility = np.std(returns) * np.sqrt(1440)
            self.momentum = np.mean(returns) * 100
        
        # Update funding
        self.funding_rate += np.random.normal(0, 0.00005)
        self.funding_rate = np.clip(self.funding_rate, -0.003, 0.003)
        
        # Regime transitions
        if np.random.random() < 0.01:
            self.regime = np.random.randint(0, 5)
    
    def _apply_funding(self) -> float:
        """Apply funding payment."""
        if self.position == 0:
            return 0.0
        
        notional = abs(self.position) * self.price
        
        if self.position > 0:
            payment = notional * self.funding_rate
        else:
            payment = -notional * self.funding_rate
        
        self.cash -= payment
        
        return -payment / 100  # Reward component
    
    def _check_done(self) -> bool:
        """Check if episode should end."""
        equity = self.cash + self.pnl
        
        # Liquidation
        if equity < self.initial_cash * 0.5:
            return True
        
        # Max steps
        if self.step_count >= 10000:
            return True
        
        return False
    
    def _get_state(self) -> SimState:
        """Get current state."""
        return SimState(
            price=self.price,
            position=self.position,
            cash=self.cash,
            pnl=self.pnl,
            funding_rate=self.funding_rate,
            volatility=self.volatility,
            momentum=self.momentum,
            regime=self.regime,
        )


class AdversarialTrader:
    """Adversarial trader for multi-agent simulation."""
    
    def __init__(self, style: str = "momentum"):
        self.style = style
        self.position = 0.0
    
    def act(self, state: SimState) -> int:
        """Decide action based on style."""
        if self.style == "momentum":
            if state.momentum > 0.5 and self.position <= 0:
                return SimAction.BUY
            elif state.momentum < -0.5 and self.position >= 0:
                return SimAction.SELL
                
        elif self.style == "mean_reversion":
            if state.momentum > 1.0 and self.position >= 0:
                return SimAction.SELL
            elif state.momentum < -1.0 and self.position <= 0:
                return SimAction.BUY
                
        elif self.style == "funding_farmer":
            if state.funding_rate > 0.0005 and self.position >= 0:
                return SimAction.SELL  # Collect positive funding
            elif state.funding_rate < -0.0005 and self.position <= 0:
                return SimAction.BUY
        
        return SimAction.HOLD
