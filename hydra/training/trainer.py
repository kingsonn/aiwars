"""
HYDRA Model Trainer

Handles:
- Transformer model training
- RL agent training with PPO
- Multi-agent self-play
- Regime-specific model adaptation
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from hydra.core.config import HydraConfig, PERMITTED_PAIRS
from hydra.layers.layer3_alpha.transformer_model import FuturesTransformer, SYMBOL_TO_IDX
from hydra.layers.layer3_alpha.rl_agent import PolicyNetwork, Action
from hydra.training.data_pipeline import TrainingExample


@dataclass
class TrainingMetrics:
    """Training metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    direction_accuracy: float
    regime_accuracy: float
    learning_rate: float


class TransformerDataset(Dataset):
    """PyTorch dataset for transformer training with pair awareness."""
    
    def __init__(self, examples: list[TrainingExample]):
        self.examples = examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        
        # Get symbol index for pair-aware training
        symbol = ex.symbol.lower() if ex.symbol else 'cmt_btcusdt'
        symbol_idx = SYMBOL_TO_IDX.get(symbol, 0)
        
        return {
            'price_features': torch.FloatTensor(ex.price_features),
            'funding_features': torch.FloatTensor(ex.funding_features),
            'oi_features': torch.FloatTensor(ex.oi_features),
            'orderbook_features': torch.FloatTensor(ex.orderbook_features),
            'liq_features': torch.FloatTensor(ex.liq_features),
            'vol_features': torch.FloatTensor(ex.vol_features),
            'symbol_idx': torch.LongTensor([symbol_idx]),
            'direction_label': torch.LongTensor([ex.direction_label]),
            'adverse_excursion': torch.FloatTensor([ex.adverse_excursion]),
            'favorable_excursion': torch.FloatTensor([ex.favorable_excursion]),
            'regime_label': torch.LongTensor([ex.regime_label]),
            'volatility_1h': torch.FloatTensor([ex.volatility_1h]),
            'profit_potential': torch.FloatTensor([ex.profit_potential]),
            'raw_return': torch.FloatTensor([ex.raw_return]),
        }


class TransformerTrainer:
    """Trainer for FuturesTransformer model."""
    
    def __init__(
        self,
        config: HydraConfig,
        model: FuturesTransformer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.model.rl_learning_rate,
            weight_decay=0.01,
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Loss functions with asymmetric weighting
        # Penalize wrong direction predictions more heavily (they lose money)
        # Class weights: Long=1.0, Short=1.0, Flat=0.5 (less penalty for being cautious)
        self.direction_loss = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.0, 0.5]).to(device)
        )
        self.regression_loss = nn.SmoothL1Loss()
        self.regime_loss = nn.CrossEntropyLoss()
        
        # Profit-aware loss: penalize predictions that would lose money
        self.profit_loss = nn.SmoothL1Loss()
        
        # Best model tracking
        self._best_val_loss = float('inf')
        self._patience_counter = 0
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Transpose for transformer (seq_len, batch, features)
            price = batch['price_features'].transpose(0, 1)
            funding = batch['funding_features'].transpose(0, 1)
            oi = batch['oi_features'].transpose(0, 1)
            orderbook = batch['orderbook_features'].transpose(0, 1)
            liq = batch['liq_features'].transpose(0, 1)
            vol = batch['vol_features'].transpose(0, 1)
            symbol_idx = batch['symbol_idx'].squeeze()  # (batch,)
            
            # Forward pass with pair-aware symbol embedding
            outputs = self.model(price, funding, oi, orderbook, liq, vol, symbol_idx=symbol_idx)
            
            # Compute losses
            loss_dir = self.direction_loss(
                outputs['direction_logits'],
                batch['direction_label'].squeeze()
            )
            
            loss_ae = self.regression_loss(
                outputs['adverse_excursion'],
                batch['adverse_excursion'].squeeze()
            )
            
            loss_fe = self.regression_loss(
                outputs['favorable_excursion'],
                batch['favorable_excursion'].squeeze()
            )
            
            loss_regime = self.regime_loss(
                outputs['regime_logits'],
                batch['regime_label'].squeeze()
            )
            
            loss_vol = self.regression_loss(
                outputs['predicted_vol'],
                batch['volatility_1h'].squeeze()
            )
            
            # Asymmetric loss: penalize wrong predictions that would lose money
            # Get predicted direction and actual direction
            pred_dir = outputs['direction_logits'].argmax(dim=-1)
            actual_dir = batch['direction_label'].squeeze()
            raw_return = batch['raw_return'].squeeze()
            
            # Wrong direction penalty: if we predicted Long but it went Short (or vice versa)
            # This is the most costly mistake in trading
            wrong_direction_mask = (
                ((pred_dir == 0) & (actual_dir == 1)) |  # Predicted Long, was Short
                ((pred_dir == 1) & (actual_dir == 0))    # Predicted Short, was Long
            ).float()
            
            # Scale loss by how wrong we were (larger moves = bigger penalty)
            wrong_direction_penalty = (wrong_direction_mask * torch.abs(raw_return) * 10.0).mean()
            
            # Combined loss with profit awareness
            loss = (
                loss_dir * 1.0 +
                loss_ae * 0.5 +
                loss_fe * 0.5 +
                loss_regime * 0.3 +
                loss_vol * 0.2 +
                wrong_direction_penalty * 2.0  # Heavy penalty for wrong direction
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> tuple[float, float, float]:
        """Validate model. Returns (loss, direction_acc, regime_acc)."""
        self.model.eval()
        total_loss = 0.0
        correct_dir = 0
        correct_regime = 0
        total = 0
        
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            price = batch['price_features'].transpose(0, 1)
            funding = batch['funding_features'].transpose(0, 1)
            oi = batch['oi_features'].transpose(0, 1)
            orderbook = batch['orderbook_features'].transpose(0, 1)
            liq = batch['liq_features'].transpose(0, 1)
            vol = batch['vol_features'].transpose(0, 1)
            symbol_idx = batch['symbol_idx'].squeeze()
            
            outputs = self.model(price, funding, oi, orderbook, liq, vol, symbol_idx=symbol_idx)
            
            # Direction accuracy
            pred_dir = outputs['direction_logits'].argmax(dim=-1)
            correct_dir += (pred_dir == batch['direction_label'].squeeze()).sum().item()
            
            # Regime accuracy
            pred_regime = outputs['regime_logits'].argmax(dim=-1)
            correct_regime += (pred_regime == batch['regime_label'].squeeze()).sum().item()
            
            total += len(batch['direction_label'])
            
            # Loss
            loss = self.direction_loss(
                outputs['direction_logits'],
                batch['direction_label'].squeeze()
            )
            total_loss += loss.item()
        
        return (
            total_loss / len(dataloader),
            correct_dir / total,
            correct_regime / total,
        )
    
    def fit(
        self,
        train_examples: list[TrainingExample],
        val_examples: list[TrainingExample],
        epochs: int = None,
        batch_size: int = None,
    ) -> list[TrainingMetrics]:
        """Full training loop."""
        epochs = epochs or self.config.model.epochs
        batch_size = batch_size or self.config.model.batch_size
        
        train_loader = DataLoader(
            TransformerDataset(train_examples),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        val_loader = DataLoader(
            TransformerDataset(val_examples),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        metrics_history = []
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, dir_acc, regime_acc = self.validate(val_loader)
            
            self.scheduler.step()
            
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                direction_accuracy=dir_acc,
                regime_accuracy=regime_acc,
                learning_rate=self.optimizer.param_groups[0]['lr'],
            )
            metrics_history.append(metrics)
            
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, dir_acc={dir_acc:.2%}, "
                f"regime_acc={regime_acc:.2%}"
            )
            
            # Early stopping
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                # Save best model
                self.save_checkpoint("best_transformer.pt")
                logger.info(f"Saved best model (val_loss={val_loss:.4f})")
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.config.model.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Always save final model
        self.save_checkpoint("final_transformer.pt")
        logger.info("Saved final transformer model")
        
        return metrics_history
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = Path(self.config.model.models_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self._best_val_loss,
        }, path)
        
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = Path(self.config.model.models_dir) / filename
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self._best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from {path}")


class PPOTrainer:
    """PPO trainer for RL execution agent."""
    
    def __init__(
        self,
        config: HydraConfig,
        policy: PolicyNetwork,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.config = config
        self.policy = policy.to(device)
        self.device = device
        
        self.optimizer = optim.Adam(
            policy.parameters(),
            lr=config.model.rl_learning_rate,
        )
        
        self.gamma = config.model.rl_gamma
        self.gae_lambda = config.model.rl_gae_lambda
        self.clip_range = config.model.rl_clip_range
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Store a transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, last_value: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages."""
        advantages = np.zeros(len(self.rewards))
        returns = np.zeros(len(self.rewards))
        
        last_gae = 0
        last_return = last_value
        
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                next_value = 0
                last_gae = 0
            else:
                next_value = self.values[t + 1] if t + 1 < len(self.values) else last_value
            
            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae
            
            last_return = self.rewards[t] + self.gamma * last_return * (1 - self.dones[t])
            returns[t] = last_return
        
        return advantages, returns
    
    def update(self, last_value: float, epochs: int = 4, batch_size: int = 64) -> dict:
        """PPO update."""
        advantages, returns = self.compute_gae(last_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(epochs):
            indices = np.random.permutation(len(self.states))
            
            for start in range(0, len(self.states), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                # Get batch
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]
                
                # Forward pass
                logits, values, _ = self.policy(batch_states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * (batch_returns - values.squeeze()).pow(2).mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
        n_updates = epochs * (len(indices) // batch_size + 1)
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }


class HydraTrainer:
    """
    Master trainer for HYDRA system.
    
    Coordinates training of all components:
    - Transformer models
    - RL agents
    - Regime-specific adaptation
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"HydraTrainer initialized on {self.device}")
    
    def train_transformer(
        self,
        train_examples: list[TrainingExample],
        val_examples: list[TrainingExample],
    ) -> FuturesTransformer:
        """Train the futures transformer model."""
        model = FuturesTransformer(
            d_model=self.config.model.transformer_hidden_size,
            nhead=self.config.model.transformer_num_heads,
            num_layers=self.config.model.transformer_num_layers,
        )
        
        trainer = TransformerTrainer(self.config, model, self.device)
        metrics = trainer.fit(train_examples, val_examples)
        
        # Load best model
        trainer.load_checkpoint("best_transformer.pt")
        
        return model
    
    def train_rl_agent(
        self,
        env,  # Trading environment
        total_timesteps: int = 100000,
    ) -> PolicyNetwork:
        """Train the RL execution agent."""
        policy = PolicyNetwork()
        trainer = PPOTrainer(self.config, policy, self.device)
        
        state = env.reset()
        episode_reward = 0
        episode_count = 0
        
        for step in range(total_timesteps):
            # Get action
            state_t = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                logits, value, _ = policy(state_t.unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Step environment
            next_state, reward, done, info = env.step(action.item())
            
            # Store transition
            trainer.store_transition(
                state, action.item(), reward,
                value.item(), log_prob.item(), done
            )
            
            episode_reward += reward
            state = next_state
            
            if done:
                state = env.reset()
                episode_count += 1
                logger.debug(f"Episode {episode_count}: reward={episode_reward:.2f}")
                episode_reward = 0
            
            # Update every 2048 steps
            if (step + 1) % 2048 == 0:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).to(self.device)
                    _, last_value, _ = policy(state_t.unsqueeze(0))
                
                metrics = trainer.update(last_value.item())
                logger.info(
                    f"Step {step}: policy_loss={metrics['policy_loss']:.4f}, "
                    f"value_loss={metrics['value_loss']:.4f}"
                )
        
        return policy
