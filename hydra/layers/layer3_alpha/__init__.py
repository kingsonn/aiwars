"""
Layer 3: Alpha & Behavior Modeling

This is where HYDRA earns money.

Components:
- Behavioral Signal Generators (FUNDING_SQUEEZE, LIQUIDATION_REVERSAL, etc.)
- Deep Futures Models (Transformer, volatility predictor, squeeze classifier)
- LLM Market Structure Agent (thinks like a derivatives desk)
- Opponent & Crowd Modeling (behavioral clustering, inverse RL)
- Reinforcement Learning (execution-centric)
"""

from hydra.layers.layer3_alpha.engine import AlphaBehaviorEngine
from hydra.layers.layer3_alpha.signals import (
    BehavioralSignalGenerator,
    funding_squeeze,
    liquidation_reversal,
    oi_divergence,
    crowding_fade,
    funding_carry,
)
from hydra.layers.layer3_alpha.transformer_model import FuturesTransformer
from hydra.layers.layer3_alpha.llm_agent import LLMMarketStructureAgent
from hydra.layers.layer3_alpha.opponent_model import OpponentCrowdModel
from hydra.layers.layer3_alpha.rl_agent import ExecutionRLAgent

__all__ = [
    "AlphaBehaviorEngine",
    "BehavioralSignalGenerator",
    "funding_squeeze",
    "liquidation_reversal",
    "oi_divergence",
    "crowding_fade",
    "funding_carry",
    "FuturesTransformer",
    "LLMMarketStructureAgent",
    "OpponentCrowdModel",
    "ExecutionRLAgent",
]
