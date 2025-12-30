"""
HYDRA Training & Backtesting Module

Components:
- Data preparation and feature engineering
- Transformer model training
- RL agent training with self-play
- Backtesting engine with regime segmentation
- Walk-forward validation
"""

from hydra.training.data_pipeline import DataPipeline
from hydra.training.trainer import HydraTrainer
from hydra.training.backtester import Backtester
from hydra.training.simulator import MarketSimulator

__all__ = [
    "DataPipeline",
    "HydraTrainer", 
    "Backtester",
    "MarketSimulator",
]
