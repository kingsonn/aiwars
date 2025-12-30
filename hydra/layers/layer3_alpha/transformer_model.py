"""
Deep Futures Transformer Model

Trained on: Price + Funding + OI + Liquidations
Outputs:
- Directional bias (probabilistic)
- Expected adverse excursion
- Squeeze asymmetry score
- Regime classification
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from hydra.core.types import MarketState, Regime, Side
from hydra.core.config import PERMITTED_PAIRS

# Symbol to index mapping for embeddings
SYMBOL_TO_IDX: dict[str, int] = {symbol: idx for idx, symbol in enumerate(PERMITTED_PAIRS)}
NUM_SYMBOLS = len(PERMITTED_PAIRS)


@dataclass
class TransformerOutput:
    """Output from the Futures Transformer model."""
    # Directional
    long_probability: float
    short_probability: float
    flat_probability: float
    
    # Risk
    expected_adverse_excursion: float  # Max expected loss
    expected_favorable_excursion: float  # Max expected gain
    
    # Squeeze
    squeeze_probability: float
    squeeze_direction: Side  # Which side gets squeezed
    squeeze_asymmetry: float  # How imbalanced the squeeze potential is
    
    # Regime
    predicted_regime: Regime
    regime_confidence: float
    
    # Volatility
    predicted_volatility_1h: float
    volatility_shock_probability: float
    
    @property
    def directional_bias(self) -> Side:
        if self.long_probability > max(self.short_probability, self.flat_probability):
            return Side.LONG
        elif self.short_probability > max(self.long_probability, self.flat_probability):
            return Side.SHORT
        return Side.FLAT
    
    @property
    def confidence(self) -> float:
        return max(self.long_probability, self.short_probability, self.flat_probability)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class FuturesFeatureEncoder(nn.Module):
    """Encode raw futures data into embeddings."""
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        
        # Price features: OHLCV normalized
        self.price_encoder = nn.Sequential(
            nn.Linear(5, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
        )
        
        # Funding features
        self.funding_encoder = nn.Sequential(
            nn.Linear(3, d_model // 8),  # rate, predicted, hours_to_next
            nn.LayerNorm(d_model // 8),
            nn.GELU(),
        )
        
        # OI features
        self.oi_encoder = nn.Sequential(
            nn.Linear(3, d_model // 8),  # oi_normalized, delta, delta_pct
            nn.LayerNorm(d_model // 8),
            nn.GELU(),
        )
        
        # Orderbook features
        self.orderbook_encoder = nn.Sequential(
            nn.Linear(4, d_model // 8),  # imbalance, spread, bid_depth, ask_depth
            nn.LayerNorm(d_model // 8),
            nn.GELU(),
        )
        
        # Liquidation features
        self.liq_encoder = nn.Sequential(
            nn.Linear(4, d_model // 8),  # long_liq_vol, short_liq_vol, liq_ratio, liq_velocity
            nn.LayerNorm(d_model // 8),
            nn.GELU(),
        )
        
        # Volatility features
        self.vol_encoder = nn.Sequential(
            nn.Linear(3, d_model // 8),  # realized_vol, vol_zscore, vol_of_vol
            nn.LayerNorm(d_model // 8),
            nn.GELU(),
        )
        
        # Combine all features
        combined_dim = d_model // 4 + d_model // 8 * 5
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
    
    def forward(
        self,
        price_features: torch.Tensor,
        funding_features: torch.Tensor,
        oi_features: torch.Tensor,
        orderbook_features: torch.Tensor,
        liq_features: torch.Tensor,
        vol_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode all features into unified embedding.
        All inputs: (seq_len, batch, feature_dim)
        Output: (seq_len, batch, d_model)
        """
        price_emb = self.price_encoder(price_features)
        funding_emb = self.funding_encoder(funding_features)
        oi_emb = self.oi_encoder(oi_features)
        ob_emb = self.orderbook_encoder(orderbook_features)
        liq_emb = self.liq_encoder(liq_features)
        vol_emb = self.vol_encoder(vol_features)
        
        combined = torch.cat([
            price_emb, funding_emb, oi_emb, ob_emb, liq_emb, vol_emb
        ], dim=-1)
        
        return self.combiner(combined)


class FuturesTransformer(nn.Module):
    """
    Transformer model for perpetual futures prediction.
    
    Architecture:
    - Symbol embedding (pair-aware)
    - Multi-source feature encoding
    - Transformer encoder with self-attention (shared across pairs)
    - Multiple prediction heads with pair-specific adjustments
    
    This hybrid approach:
    - Learns general market dynamics from all pairs (shared encoder)
    - Learns pair-specific behavior via symbol embeddings
    - Can generalize while still specializing
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        num_symbols: int = NUM_SYMBOLS,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_symbols = num_symbols
        
        # Symbol embedding - learns pair-specific characteristics
        # Each pair gets a learned embedding that modifies predictions
        self.symbol_embedding = nn.Embedding(num_symbols, d_model)
        
        # Symbol-specific scaling factors for volatility adjustment
        # Initialized to 1.0, learns pair-specific vol scaling
        self.symbol_vol_scale = nn.Embedding(num_symbols, 1)
        nn.init.ones_(self.symbol_vol_scale.weight)
        
        # Feature encoder
        self.feature_encoder = FuturesFeatureEncoder(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Symbol context projection (combines symbol embedding with features)
        self.symbol_proj = nn.Linear(d_model * 2, d_model)
        
        # Transformer encoder (shared across all pairs)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads
        
        # Direction prediction (long/short/flat)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),
        )
        
        # Excursion prediction (adverse, favorable)
        self.excursion_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
            nn.Softplus(),  # Always positive
        )
        
        # Squeeze prediction (probability, direction, asymmetry)
        self.squeeze_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 4),  # prob, long_squeeze, short_squeeze, asymmetry
        )
        
        # Regime prediction (7 regimes from Regime enum)
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 8),
        )
        
        # Volatility prediction
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),  # vol_1h, shock_prob
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        price_features: torch.Tensor,
        funding_features: torch.Tensor,
        oi_features: torch.Tensor,
        orderbook_features: torch.Tensor,
        liq_features: torch.Tensor,
        vol_features: torch.Tensor,
        symbol_idx: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with pair-aware predictions.
        
        Args:
            All feature tensors: (seq_len, batch, feature_dim)
            symbol_idx: (batch,) tensor of symbol indices for pair-specific behavior
            src_mask: Optional attention mask
            
        Returns:
            Dictionary of predictions
        """
        seq_len = price_features.size(0)
        batch_size = price_features.size(1)
        device = price_features.device
        
        # Default to BTC if no symbol provided (backward compatible)
        if symbol_idx is None:
            symbol_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Get symbol embedding (batch, d_model)
        symbol_emb = self.symbol_embedding(symbol_idx)
        
        # Encode features
        x = self.feature_encoder(
            price_features, funding_features, oi_features,
            orderbook_features, liq_features, vol_features
        )
        
        # Inject symbol context into each timestep
        # Expand symbol embedding to (seq_len, batch, d_model)
        symbol_emb_expanded = symbol_emb.unsqueeze(0).expand(seq_len, -1, -1)
        
        # Combine features with symbol context
        x = self.symbol_proj(torch.cat([x, symbol_emb_expanded], dim=-1))
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding (shared across all pairs)
        x = self.transformer(x, src_mask)
        
        # Use last token for prediction
        last_hidden = x[-1]  # (batch, d_model)
        
        # Add symbol context to final prediction
        last_hidden = last_hidden + symbol_emb * 0.1  # Residual symbol influence
        
        # Generate predictions
        direction_logits = self.direction_head(last_hidden)
        excursions = self.excursion_head(last_hidden)
        squeeze_out = self.squeeze_head(last_hidden)
        regime_logits = self.regime_head(last_hidden)
        vol_out = self.volatility_head(last_hidden)
        
        # Apply pair-specific volatility scaling
        vol_scale = self.symbol_vol_scale(symbol_idx).squeeze(-1)  # (batch,)
        
        return {
            'direction_logits': direction_logits,
            'direction_probs': F.softmax(direction_logits, dim=-1),
            'adverse_excursion': excursions[:, 0] * vol_scale,
            'favorable_excursion': excursions[:, 1] * vol_scale,
            'squeeze_prob': torch.sigmoid(squeeze_out[:, 0]),
            'long_squeeze_prob': torch.sigmoid(squeeze_out[:, 1]),
            'short_squeeze_prob': torch.sigmoid(squeeze_out[:, 2]),
            'squeeze_asymmetry': torch.tanh(squeeze_out[:, 3]),
            'regime_logits': regime_logits,
            'regime_probs': F.softmax(regime_logits, dim=-1),
            'predicted_vol': F.softplus(vol_out[:, 0]) * vol_scale,
            'vol_shock_prob': torch.sigmoid(vol_out[:, 1]),
            'symbol_idx': symbol_idx,
        }
    
    def predict(self, market_state: MarketState, device: str = 'cpu') -> TransformerOutput:
        """
        Generate prediction from market state.
        """
        self.eval()
        
        with torch.no_grad():
            # Prepare features
            features = self._prepare_features(market_state)
            
            # Move to device
            features = {k: v.to(device) for k, v in features.items()}
            
            # Forward pass
            outputs = self.forward(**features)
            
            # Parse outputs
            dir_probs = outputs['direction_probs'][0].cpu().numpy()
            
            squeeze_asymmetry = outputs['squeeze_asymmetry'][0].item()
            if squeeze_asymmetry > 0:
                squeeze_direction = Side.SHORT  # Shorts will get squeezed
            elif squeeze_asymmetry < 0:
                squeeze_direction = Side.LONG
            else:
                squeeze_direction = Side.FLAT
            
            # Map regime
            regime_idx = outputs['regime_probs'][0].argmax().item()
            regime_map = {
                0: Regime.TRENDING_UP,
                1: Regime.TRENDING_DOWN,
                2: Regime.RANGING,
                3: Regime.HIGH_VOLATILITY,
                4: Regime.CASCADE_RISK,
                5: Regime.SQUEEZE_LONG,
                6: Regime.SQUEEZE_SHORT,
                7: Regime.UNKNOWN,
            }
            
            return TransformerOutput(
                long_probability=float(dir_probs[0]),
                short_probability=float(dir_probs[1]),
                flat_probability=float(dir_probs[2]),
                expected_adverse_excursion=float(outputs['adverse_excursion'][0]),
                expected_favorable_excursion=float(outputs['favorable_excursion'][0]),
                squeeze_probability=float(outputs['squeeze_prob'][0]),
                squeeze_direction=squeeze_direction,
                squeeze_asymmetry=abs(squeeze_asymmetry),
                predicted_regime=regime_map.get(regime_idx, Regime.UNKNOWN),
                regime_confidence=float(outputs['regime_probs'][0].max()),
                predicted_volatility_1h=float(outputs['predicted_vol'][0]),
                volatility_shock_probability=float(outputs['vol_shock_prob'][0]),
            )
    
    def _prepare_features(self, market_state: MarketState) -> dict[str, torch.Tensor]:
        """Prepare feature tensors from market state."""
        candles = market_state.ohlcv.get("5m", [])[-self.max_seq_len:]
        seq_len = len(candles)
        
        if seq_len == 0:
            seq_len = 1
            candles = [type('OHLCV', (), {
                'open': market_state.price,
                'high': market_state.price,
                'low': market_state.price,
                'close': market_state.price,
                'volume': 0,
            })()]
        
        # Price features: normalize by first close
        base_price = candles[0].close if candles[0].close > 0 else 1
        price_features = np.array([
            [c.open/base_price, c.high/base_price, c.low/base_price, 
             c.close/base_price, np.log1p(c.volume)]
            for c in candles
        ])
        
        # Funding features (broadcast to sequence)
        if market_state.funding_rate:
            fr = market_state.funding_rate
            funding_feat = np.array([fr.rate * 10000, (fr.predicted_rate or fr.rate) * 10000, 8.0])
        else:
            funding_feat = np.zeros(3)
        funding_features = np.tile(funding_feat, (seq_len, 1))
        
        # OI features (with None safety)
        if market_state.open_interest:
            oi = market_state.open_interest
            oi_feat = np.array([
                (oi.open_interest_usd or 0) / 1e9,  # Normalize to billions
                (oi.delta or 0) / 1e6,
                (oi.delta_pct or 0) / 10,
            ])
        else:
            oi_feat = np.zeros(3)
        oi_features = np.tile(oi_feat, (seq_len, 1))
        
        # Orderbook features
        if market_state.order_book:
            ob = market_state.order_book
            bid_depth = sum(q for _, q in ob.bids[:5])
            ask_depth = sum(q for _, q in ob.asks[:5])
            ob_feat = np.array([ob.imbalance, ob.spread * 10000, bid_depth, ask_depth])
        else:
            ob_feat = np.zeros(4)
        orderbook_features = np.tile(ob_feat, (seq_len, 1))
        
        # Liquidation features
        liqs = market_state.recent_liquidations
        if liqs:
            long_liq = sum(l.usd_value for l in liqs if l.side == Side.LONG)
            short_liq = sum(l.usd_value for l in liqs if l.side == Side.SHORT)
            total = long_liq + short_liq
            liq_feat = np.array([
                long_liq / 1e6, short_liq / 1e6,
                (long_liq - short_liq) / max(total, 1),
                len(liqs) / 100,
            ])
        else:
            liq_feat = np.zeros(4)
        liq_features = np.tile(liq_feat, (seq_len, 1))
        
        # Volatility features
        vol_feat = np.array([
            market_state.volatility,
            0.0,  # Would be vol zscore
            0.0,  # Vol of vol
        ])
        vol_features = np.tile(vol_feat, (seq_len, 1))
        
        # Get symbol index for pair-aware prediction
        symbol = market_state.symbol.lower()
        symbol_idx = SYMBOL_TO_IDX.get(symbol, 0)  # Default to BTC if unknown
        
        # Convert to tensors (seq_len, batch=1, features)
        return {
            'price_features': torch.FloatTensor(price_features).unsqueeze(1),
            'funding_features': torch.FloatTensor(funding_features).unsqueeze(1),
            'oi_features': torch.FloatTensor(oi_features).unsqueeze(1),
            'orderbook_features': torch.FloatTensor(orderbook_features).unsqueeze(1),
            'liq_features': torch.FloatTensor(liq_features).unsqueeze(1),
            'vol_features': torch.FloatTensor(vol_features).unsqueeze(1),
            'symbol_idx': torch.LongTensor([symbol_idx]),
        }
    
    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str, device: str = 'cpu') -> None:
        """Load model weights."""
        self.load_state_dict(torch.load(path, map_location=device))
        logger.info(f"Model loaded from {path}")
