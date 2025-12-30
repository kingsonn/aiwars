"""
Reinforcement Learning Agent (Execution-Centric)

RL does NOT predict price.

It learns:
- Entry timing
- Scaling rules
- Exit timing
- De-risking before funding flips
- When to stay flat

Reward function includes:
- Net PnL
- Funding paid/received
- Liquidation proximity penalty
- Drawdown penalty
- Overtrading penalty

Meta-RL adapts policies per regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from loguru import logger

from hydra.core.config import HydraConfig
from hydra.core.types import MarketState, Position, Side, Regime
from hydra.layers.layer2_statistical import StatisticalResult


class Action(IntEnum):
    """Discrete actions for the RL agent."""
    HOLD = 0  # Do nothing
    ENTER_LONG = 1  # Open/add long
    ENTER_SHORT = 2  # Open/add short
    EXIT_PARTIAL = 3  # Reduce position by 50%
    EXIT_FULL = 4  # Close entire position
    FLIP_LONG = 5  # Close and go long
    FLIP_SHORT = 6  # Close and go short


@dataclass
class RLState:
    """State representation for RL agent."""
    # Market features (normalized)
    price_momentum: float
    volatility: float
    funding_rate: float
    oi_delta: float
    orderbook_imbalance: float
    
    # Position features
    position_side: int  # -1, 0, 1
    position_size_pct: float  # % of max
    unrealized_pnl_pct: float
    time_in_position: float  # hours
    distance_to_liquidation: float
    
    # Risk features
    drawdown: float
    trades_last_hour: int
    funding_paid_session: float
    
    # Regime
    regime_code: int
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor."""
        return torch.FloatTensor([
            self.price_momentum,
            self.volatility,
            self.funding_rate,
            self.oi_delta,
            self.orderbook_imbalance,
            self.position_side,
            self.position_size_pct,
            self.unrealized_pnl_pct,
            self.time_in_position,
            self.distance_to_liquidation,
            self.drawdown,
            self.trades_last_hour / 10,
            self.funding_paid_session,
            self.regime_code / 8,
        ])


@dataclass
class RLDecision:
    """Output from RL agent."""
    action: Action
    action_probability: float
    value_estimate: float
    
    # Sizing
    size_multiplier: float  # 0.25 to 1.0
    
    # Risk overrides
    should_reduce_risk: bool = False
    reduce_reason: str = ""
    
    # Timing
    should_wait: bool = False
    wait_reason: str = ""


class PolicyNetwork(nn.Module):
    """Actor-Critic network for execution policy."""
    
    def __init__(
        self,
        state_dim: int = 14,  # Matches RLState.to_tensor() output (14 features)
        hidden_dim: int = 128,
        num_actions: int = 7,
    ):
        super().__init__()
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Size predictor (continuous 0-1)
        self.sizer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            action_logits: (batch, num_actions)
            value: (batch, 1)
            size: (batch, 1)
        """
        shared = self.shared(state)
        action_logits = self.actor(shared)
        value = self.critic(shared)
        size = self.sizer(shared)
        return action_logits, value, size
    
    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[int, float, float, float]:
        """
        Get action from policy.
        
        Returns:
            action, probability, value, size_multiplier
        """
        with torch.no_grad():
            logits, value, size = self.forward(state.unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                dist = Categorical(probs)
                action = dist.sample().item()
            
            prob = probs[0, action].item()
            
            # Size multiplier: 0.25 to 1.0
            size_mult = 0.25 + 0.75 * size[0, 0].item()
            
            return action, prob, value[0, 0].item(), size_mult


class RewardCalculator:
    """Calculate RL reward with multiple components."""
    
    def __init__(self, config: HydraConfig):
        self.config = config
        
        # Reward weights
        self.pnl_weight = 1.0
        self.funding_weight = 0.3
        self.liquidation_penalty_weight = 2.0
        self.drawdown_penalty_weight = 1.5
        self.overtrading_penalty_weight = 0.5
    
    def calculate(
        self,
        pnl_change: float,
        funding_paid: float,
        liquidation_distance: float,
        drawdown: float,
        trade_count_hour: int,
        position_size_pct: float,
    ) -> Tuple[float, dict]:
        """
        Calculate reward.
        
        Returns:
            total_reward, reward_breakdown
        """
        components = {}
        
        # PnL reward (scaled)
        pnl_reward = np.tanh(pnl_change * 10) * self.pnl_weight
        components['pnl'] = pnl_reward
        
        # Funding (positive if received, negative if paid)
        funding_reward = -funding_paid * self.funding_weight * 100
        components['funding'] = funding_reward
        
        # Liquidation proximity penalty
        if liquidation_distance < 0.2:  # Within 20%
            liq_penalty = -((0.2 - liquidation_distance) / 0.2) ** 2 * self.liquidation_penalty_weight
        else:
            liq_penalty = 0
        components['liquidation'] = liq_penalty
        
        # Drawdown penalty
        if drawdown > 0.05:  # Beyond 5%
            dd_penalty = -((drawdown - 0.05) / 0.1) ** 2 * self.drawdown_penalty_weight
        else:
            dd_penalty = 0
        components['drawdown'] = dd_penalty
        
        # Overtrading penalty
        if trade_count_hour > 5:
            overtrade_penalty = -(trade_count_hour - 5) * 0.1 * self.overtrading_penalty_weight
        else:
            overtrade_penalty = 0
        components['overtrading'] = overtrade_penalty
        
        # Flat position reward (small positive for not taking bad trades)
        if position_size_pct < 0.1:
            flat_bonus = 0.01  # Small reward for staying flat when uncertain
        else:
            flat_bonus = 0
        components['flat_bonus'] = flat_bonus
        
        total = sum(components.values())
        
        return total, components


class ExecutionRLAgent:
    """
    Reinforcement Learning Agent for execution decisions.
    
    This agent decides:
    - WHEN to enter (timing)
    - HOW MUCH to enter (sizing)
    - WHEN to exit (timing)
    - HOW to scale (partial vs full)
    
    It does NOT decide direction - that comes from alpha signals.
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        
        self.policy = PolicyNetwork()
        self.reward_calc = RewardCalculator(config)
        
        # Regime-specific policies (Meta-RL)
        self._regime_policies: dict[Regime, PolicyNetwork] = {}
        
        # Experience buffer for learning
        self._experience: list[tuple] = []
        self._max_experience = 10000
        
        # Trading state
        self._trades_this_hour: int = 0
        self._last_trade_time: Optional[datetime] = None
        self._session_funding_paid: float = 0.0
        self._peak_equity: float = 0.0
        self._current_drawdown: float = 0.0
        
        logger.info("Execution RL Agent initialized")
    
    async def setup(self) -> None:
        """Initialize agent."""
        # Initialize regime-specific policies
        for regime in Regime:
            self._regime_policies[regime] = PolicyNetwork()
        
        # Load pre-trained weights if available
        # self._load_weights()
    
    def decide(
        self,
        market_state: MarketState,
        stat_result: StatisticalResult,
        current_position: Optional[Position],
        proposed_direction: Side,
        proposed_confidence: float,
    ) -> RLDecision:
        """
        Make execution decision.
        
        Args:
            market_state: Current market data
            stat_result: Statistical analysis
            current_position: Current position if any
            proposed_direction: Direction from alpha models
            proposed_confidence: Confidence from alpha models
            
        Returns:
            RLDecision with action, sizing, and risk overrides
        """
        # Build state
        state = self._build_state(market_state, stat_result, current_position)
        state_tensor = state.to_tensor()
        
        # Select policy based on regime
        policy = self._regime_policies.get(stat_result.regime, self.policy)
        
        # Get action
        action_idx, prob, value, size_mult = policy.get_action(
            state_tensor, deterministic=False
        )
        action = Action(action_idx)
        
        # Apply safety overrides
        should_reduce, reduce_reason = self._check_risk_overrides(
            state, current_position, stat_result
        )
        
        should_wait, wait_reason = self._check_timing(
            market_state, stat_result, proposed_confidence
        )
        
        # Override action if needed
        if should_reduce and current_position and current_position.is_open:
            if state.distance_to_liquidation < 0.15:
                action = Action.EXIT_FULL
            else:
                action = Action.EXIT_PARTIAL
        
        if should_wait and action in [Action.ENTER_LONG, Action.ENTER_SHORT, 
                                      Action.FLIP_LONG, Action.FLIP_SHORT]:
            action = Action.HOLD
        
        # Validate action against proposed direction
        action = self._validate_action_direction(action, proposed_direction, current_position)
        
        return RLDecision(
            action=action,
            action_probability=prob,
            value_estimate=value,
            size_multiplier=size_mult,
            should_reduce_risk=should_reduce,
            reduce_reason=reduce_reason,
            should_wait=should_wait,
            wait_reason=wait_reason,
        )
    
    def _build_state(
        self,
        market_state: MarketState,
        stat_result: StatisticalResult,
        position: Optional[Position],
    ) -> RLState:
        """Build state representation."""
        # Market features
        price_momentum = market_state.price_change_24h * 10  # Scaled
        volatility = stat_result.realized_volatility
        
        funding_rate = 0.0
        if market_state.funding_rate:
            funding_rate = market_state.funding_rate.rate * 10000
        
        oi_delta = 0.0
        if market_state.open_interest:
            oi_delta = market_state.open_interest.delta_pct / 10
        
        ob_imbalance = 0.0
        if market_state.order_book:
            ob_imbalance = market_state.order_book.imbalance
        
        # Position features
        if position and position.is_open:
            position_side = 1 if position.side == Side.LONG else -1
            position_size_pct = position.size_usd / self.config.trading.max_position_size_usd
            unrealized_pnl_pct = position.unrealized_pnl_pct
            time_in_pos = 0.0
            if position.entry_time:
                time_in_pos = (datetime.now(timezone.utc) - position.entry_time).total_seconds() / 3600
            distance_to_liq = position.distance_to_liquidation
        else:
            position_side = 0
            position_size_pct = 0.0
            unrealized_pnl_pct = 0.0
            time_in_pos = 0.0
            distance_to_liq = 1.0
        
        # Regime encoding
        regime_map = {r: i for i, r in enumerate(Regime)}
        regime_code = regime_map.get(stat_result.regime, 0)
        
        return RLState(
            price_momentum=price_momentum,
            volatility=volatility,
            funding_rate=funding_rate,
            oi_delta=oi_delta,
            orderbook_imbalance=ob_imbalance,
            position_side=position_side,
            position_size_pct=position_size_pct,
            unrealized_pnl_pct=unrealized_pnl_pct,
            time_in_position=time_in_pos,
            distance_to_liquidation=distance_to_liq,
            drawdown=self._current_drawdown,
            trades_last_hour=self._trades_this_hour,
            funding_paid_session=self._session_funding_paid,
            regime_code=regime_code,
        )
    
    def _check_risk_overrides(
        self,
        state: RLState,
        position: Optional[Position],
        stat_result: StatisticalResult,
    ) -> Tuple[bool, str]:
        """Check if risk override is needed."""
        if not position or not position.is_open:
            return False, ""
        
        # Liquidation too close
        if state.distance_to_liquidation < 0.15:
            return True, f"Liquidation distance {state.distance_to_liquidation:.1%} < 15%"
        
        # Drawdown too high
        if self._current_drawdown > self.config.risk.max_daily_drawdown_pct / 100:
            return True, f"Drawdown {self._current_drawdown:.1%} exceeds limit"
        
        # Cascade risk
        if stat_result.cascade_probability > 0.5:
            return True, f"Cascade probability {stat_result.cascade_probability:.1%}"
        
        # Regime break
        if stat_result.regime_break_alert:
            return True, "Regime break detected"
        
        return False, ""
    
    def _check_timing(
        self,
        market_state: MarketState,
        stat_result: StatisticalResult,
        confidence: float,
    ) -> Tuple[bool, str]:
        """Check if we should wait before acting."""
        # Low confidence
        if confidence < self.config.risk.min_confidence_threshold:
            return True, f"Confidence {confidence:.2f} below threshold"
        
        # High volatility regime
        if stat_result.volatility_regime == "extreme":
            return True, "Extreme volatility - waiting"
        
        # Abnormal move in progress
        if stat_result.is_abnormal:
            return True, f"Abnormal move detected (z={stat_result.abnormal_move_score:.1f})"
        
        # Too many trades
        if self._trades_this_hour >= 10:
            return True, "Trade frequency limit reached"
        
        return False, ""
    
    def _validate_action_direction(
        self,
        action: Action,
        proposed_direction: Side,
        position: Optional[Position],
    ) -> Action:
        """Ensure action aligns with proposed direction."""
        # If alpha says long, don't go short (and vice versa)
        if proposed_direction == Side.LONG:
            if action == Action.ENTER_SHORT:
                return Action.HOLD
            if action == Action.FLIP_SHORT:
                return Action.EXIT_FULL
        elif proposed_direction == Side.SHORT:
            if action == Action.ENTER_LONG:
                return Action.HOLD
            if action == Action.FLIP_LONG:
                return Action.EXIT_FULL
        elif proposed_direction == Side.FLAT:
            # If alpha says flat, only allow exits
            if action in [Action.ENTER_LONG, Action.ENTER_SHORT,
                         Action.FLIP_LONG, Action.FLIP_SHORT]:
                if position and position.is_open:
                    return Action.EXIT_FULL
                return Action.HOLD
        
        return action
    
    def record_trade(self) -> None:
        """Record that a trade was made."""
        now = datetime.now(timezone.utc)
        
        # Reset hourly counter
        if self._last_trade_time:
            if (now - self._last_trade_time).total_seconds() > 3600:
                self._trades_this_hour = 0
        
        self._trades_this_hour += 1
        self._last_trade_time = now
    
    def update_equity(self, current_equity: float) -> None:
        """Update equity tracking for drawdown."""
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        
        if self._peak_equity > 0:
            self._current_drawdown = (self._peak_equity - current_equity) / self._peak_equity
    
    def record_funding(self, amount: float) -> None:
        """Record funding payment."""
        self._session_funding_paid += amount
    
    def store_experience(
        self,
        state: RLState,
        action: Action,
        reward: float,
        next_state: RLState,
        done: bool,
    ) -> None:
        """Store experience for learning."""
        self._experience.append((
            state.to_tensor(),
            action.value,
            reward,
            next_state.to_tensor(),
            done,
        ))
        
        if len(self._experience) > self._max_experience:
            self._experience = self._experience[-self._max_experience:]
    
    def save(self, path: str) -> None:
        """Save agent weights."""
        state = {
            'policy': self.policy.state_dict(),
            'regime_policies': {
                str(r): p.state_dict() 
                for r, p in self._regime_policies.items()
            },
        }
        torch.save(state, path)
        logger.info(f"RL Agent saved to {path}")
    
    def load(self, path: str) -> None:
        """Load agent weights."""
        state = torch.load(path)
        self.policy.load_state_dict(state['policy'])
        
        for regime_str, weights in state.get('regime_policies', {}).items():
            regime = Regime[regime_str.split('.')[-1]]
            if regime in self._regime_policies:
                self._regime_policies[regime].load_state_dict(weights)
        
        logger.info(f"RL Agent loaded from {path}")
