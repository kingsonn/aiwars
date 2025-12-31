# HYDRA — ML Model Specifications
## Models, Features, Training Process, and LLM Integration

**Version:** 2.0  
**Last Updated:** December 31, 2024  

---

# MODELS OVERVIEW

HYDRA uses two AI components for signal filtering:

| Component | Purpose | Type | Input | Output |
|-----------|---------|------|-------|--------|
| **ML Signal Scorer** | Score signal profitability | XGBoost Classifier | 49 features | P(profitable) |
| **LLM News Analyst** | News-based trade gating | Claude LLM | News + pairs | Action per pair |

Additional models (optional/future):

| Model | Purpose | Type | Status |
|-------|---------|------|--------|
| Regime Classifier | Detect market regime | Multi-class | Rule-based currently |
| Direction Predictor | Price direction | Transformer | Placeholder |
| Exit Predictor | Optimal exit timing | Regression | Not implemented |

---

# ML SIGNAL SCORER

## Purpose
Score how likely a generated behavioral signal will be profitable after transaction costs.

## Architecture
- **Type**: XGBoost Gradient Boosted Trees
- **Why**: Fast inference, handles tabular data well, interpretable feature importance
- **Model file**: `models/signal_scorer.pkl`
- **Threshold**: 0.45 (signals below this are rejected)

## The 49 Input Features

```python
SIGNAL_SCORER_FEATURES = [
    # === SIGNAL FEATURES (9) ===
    "signal_direction",                    # 1=LONG, -1=SHORT
    "signal_confidence",                   # From behavioral generator (0-1)
    "signal_source_funding_squeeze",       # One-hot encoded
    "signal_source_liq_reversal",          # One-hot encoded
    "signal_source_oi_divergence",         # One-hot encoded
    "signal_source_crowding_fade",         # One-hot encoded
    "signal_source_funding_carry",         # One-hot encoded
    "expected_return",                     # Expected profit %
    "expected_adverse_excursion",          # Expected max loss %
    
    # === PRICE FEATURES (10) ===
    "return_1m",                           # 1-minute return
    "return_5m",                           # 5-minute return
    "return_15m",                          # 15-minute return
    "return_1h",                           # 1-hour return
    "volatility_5m",                       # 5-minute rolling volatility
    "volatility_1h",                       # 1-hour rolling volatility
    "price_vs_sma_20",                     # Price relative to 20-period SMA
    "price_vs_sma_50",                     # Price relative to 50-period SMA
    "rsi_14",                              # 14-period RSI
    "atr_14",                              # 14-period ATR (normalized)
    
    # === FUNDING FEATURES (4) ===
    "funding_rate",                        # Current funding rate
    "funding_rate_zscore",                 # Z-score vs historical
    "funding_annualized",                  # Annualized funding cost
    "funding_momentum",                    # Change over recent periods
    
    # === OI FEATURES (3) ===
    "oi_delta_pct",                        # OI change percentage
    "oi_delta_zscore",                     # Z-score of OI change
    "oi_price_divergence",                 # OI vs price direction mismatch
    
    # === LIQUIDATION FEATURES (3) ===
    "liq_imbalance",                       # (long_liq - short_liq) / total
    "liq_velocity",                        # Recent liquidations / OI
    "liq_zscore",                          # Z-score of liquidation velocity
    
    # === ORDER BOOK FEATURES (4) ===
    "ob_imbalance",                        # Bid vs ask volume imbalance
    "spread_bps",                          # Spread in basis points
    "bid_depth_usd",                       # Bid side depth in USD
    "ask_depth_usd",                       # Ask side depth in USD
    
    # === POSITIONING FEATURES (2) ===
    "long_short_ratio",                    # Long/short account ratio
    "taker_buy_sell_ratio",                # Taker buy/sell volume ratio
    
    # === REGIME FEATURES (9) ===
    "regime_trending_up",                  # One-hot: trending up
    "regime_trending_down",                # One-hot: trending down
    "regime_ranging",                      # One-hot: ranging
    "regime_high_vol",                     # One-hot: high volatility
    "regime_cascade",                      # One-hot: cascade risk
    "regime_squeeze_long",                 # One-hot: long squeeze
    "regime_squeeze_short",                # One-hot: short squeeze
    "volatility_regime",                   # 0=low, 1=normal, 2=high
    "cascade_probability",                 # From Layer 2
    
    # === TIME FEATURES (5) ===
    "hour_of_day_sin",                     # Cyclical hour encoding (sin)
    "hour_of_day_cos",                     # Cyclical hour encoding (cos)
    "day_of_week_sin",                     # Cyclical day encoding (sin)
    "day_of_week_cos",                     # Cyclical day encoding (cos)
    "minutes_to_funding",                  # Minutes until next funding
]
```

## Output
```python
# Binary classification
output = model.predict_proba(features)
# Returns: [P(unprofitable), P(profitable)]
# Use: score = output[1]  # P(profitable)
```

## Label Definition
```python
def create_label(signal, future_prices, holding_period_minutes):
    """
    Label = 1 if trade would have been profitable
    """
    entry_price = future_prices[0]
    exit_price = future_prices[holding_period_minutes]
    
    if signal.side == Side.LONG:
        return_pct = (exit_price - entry_price) / entry_price
    else:
        return_pct = (entry_price - exit_price) / entry_price
    
    # Account for fees (~0.1% round trip)
    net_return = return_pct - 0.001
    
    return 1 if net_return > 0 else 0
```

## Training Data Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Samples per pair | 10,000 | 50,000+ |
| Time span | 3 months | 1+ year |
| Include regimes | All | All |
| Class balance | ~40-60% | Natural |

## Training Process

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit

def train_signal_scorer(X, y, categorical_features):
    # Time-series cross-validation (no data leakage)
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        early_stopping_rounds=50,
        cat_features=categorical_features,
    )
    
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train,
                  eval_set=(X_val, y_val),
                  verbose=100)
        
        score = model.score(X_val, y_val)
        scores.append(score)
    
    print(f"CV Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    # Final model on all data
    model.fit(X, y)
    return model
```

---

# MODEL 2: REGIME CLASSIFIER

## Purpose
Classify current market regime for strategy selection.

## Architecture
- **Type**: Random Forest or XGBoost multi-class
- **Classes**: 7 regimes

```python
class Regime(Enum):
    TRENDING_UP = 0
    TRENDING_DOWN = 1
    RANGING = 2
    HIGH_VOLATILITY = 3
    CASCADE_RISK = 4
    SQUEEZE_LONG = 5
    SQUEEZE_SHORT = 6
```

## Input Features

```python
REGIME_FEATURES = [
    # === TREND FEATURES ===
    "return_1h", "return_4h", "return_24h",
    "sma_20_slope", "sma_50_slope",
    "price_vs_sma_20", "price_vs_sma_50",
    "adx_14",                    # Trend strength
    
    # === VOLATILITY FEATURES ===
    "volatility_5m", "volatility_1h", "volatility_24h",
    "volatility_zscore",
    "atr_14", "atr_zscore",
    "bollinger_width",
    
    # === FUNDING/POSITIONING ===
    "funding_rate", "funding_zscore",
    "oi_delta_pct", "oi_zscore",
    "long_short_ratio",
    
    # === LIQUIDATION ===
    "liq_velocity",
    "liq_imbalance",
    "cascade_probability",       # From statistical model
    
    # === VOLUME ===
    "volume_zscore",
    "cvd_momentum",              # CVD change
]
```

## Label Definition

```python
def label_regime(window_data):
    """
    Label based on what happened in the window.
    """
    returns = window_data["returns"]
    volatility = window_data["volatility"]
    funding = window_data["funding_rate"]
    oi_delta = window_data["oi_delta"]
    liq_velocity = window_data["liq_velocity"]
    
    # Cascade risk
    if liq_velocity > 0.05:
        return Regime.CASCADE_RISK
    
    # Squeeze detection
    if funding > 0.001 and oi_delta > 0.02 and returns.mean() < 0:
        return Regime.SQUEEZE_SHORT
    if funding < -0.001 and oi_delta > 0.02 and returns.mean() > 0:
        return Regime.SQUEEZE_LONG
    
    # High volatility
    if volatility > volatility_95th_percentile:
        return Regime.HIGH_VOLATILITY
    
    # Trending
    trend = returns.sum()
    if trend > 0.02:
        return Regime.TRENDING_UP
    if trend < -0.02:
        return Regime.TRENDING_DOWN
    
    return Regime.RANGING
```

## Training

```python
from sklearn.ensemble import RandomForestClassifier

def train_regime_classifier(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
    )
    
    # Time-series split
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv)
    
    print(f"CV Accuracy: {np.mean(scores):.3f}")
    
    model.fit(X, y)
    return model
```

---

# MODEL 3: DIRECTION PREDICTOR (TRANSFORMER)

## Purpose
Predict short-term price direction from sequence data.

## Architecture
- **Type**: Transformer encoder
- **Input**: Sequence of candles + features
- **Output**: P(price up in next N minutes)

```python
import torch
import torch.nn as nn

class DirectionTransformer(nn.Module):
    def __init__(
        self,
        input_dim=15,        # Features per timestep
        d_model=64,
        nhead=4,
        num_layers=3,
        seq_len=100,
        dropout=0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x[:, -1, :]  # Last timestep
        return self.classifier(x)
```

## Input Features (per timestep)

```python
SEQUENCE_FEATURES = [
    # OHLCV normalized
    "open_norm", "high_norm", "low_norm", "close_norm", "volume_norm",
    
    # Returns
    "return",
    
    # Volatility
    "volatility",
    
    # Funding
    "funding_rate",
    
    # OI
    "oi_delta_pct",
    
    # Order book
    "ob_imbalance",
    
    # Liquidations
    "liq_imbalance",
    
    # Technical
    "rsi_norm",
    "macd_norm",
    "bb_position",  # Position within Bollinger Bands
]
```

## Label Definition

```python
def create_direction_label(prices, horizon_minutes=30):
    """
    Label = 1 if price higher in horizon_minutes
    """
    current_price = prices[0]
    future_price = prices[horizon_minutes]
    return 1 if future_price > current_price else 0
```

## Training

```python
def train_transformer(train_loader, val_loader, epochs=100):
    model = DirectionTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = model(X).squeeze()
            loss = criterion(pred, y.float())
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                pred = (model(X).squeeze() > 0.5).long()
                correct += (pred == y).sum().item()
                total += len(y)
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
        
        print(f"Epoch {epoch}: Val Acc = {val_acc:.3f}")
    
    return model
```

---

# MODEL 4: EXIT PREDICTOR

## Purpose
Predict optimal exit timing for open positions.

## Architecture
- **Type**: Gradient Boosted Trees
- **Output**: P(should exit now)

## Input Features

```python
EXIT_PREDICTOR_FEATURES = [
    # === POSITION FEATURES ===
    "position_side",             # 1=LONG, -1=SHORT
    "unrealized_pnl_pct",
    "time_in_position_hours",
    "distance_to_stop_loss",
    "distance_to_take_profit",
    "distance_to_liquidation",
    "funding_paid_pct",
    
    # === ORIGINAL THESIS ===
    "thesis_health",             # From thesis tracker
    "original_signal_source",    # One-hot
    "original_confidence",
    
    # === CURRENT MARKET ===
    "current_funding_rate",
    "funding_vs_entry",          # How funding changed
    "current_oi_delta",
    "oi_vs_entry",
    "current_regime",            # One-hot
    "regime_changed",            # 1 if different from entry
    
    # === VOLATILITY ===
    "current_volatility",
    "volatility_vs_entry",
    "volatility_regime",
    
    # === MOMENTUM ===
    "return_since_entry",
    "return_last_1h",
    "return_last_15m",
    
    # === RISK ===
    "cascade_probability",
    "liq_velocity",
]
```

## Label Definition

```python
def create_exit_label(position, future_data, lookahead_minutes=60):
    """
    Label = 1 if exiting NOW is better than holding
    """
    current_pnl = position.unrealized_pnl_pct
    
    # Simulate holding for lookahead period
    future_prices = future_data["prices"][:lookahead_minutes]
    
    if position.side == Side.LONG:
        future_pnl = (future_prices[-1] - position.entry_price) / position.entry_price
        max_drawdown = min(future_prices) / position.entry_price - 1
    else:
        future_pnl = (position.entry_price - future_prices[-1]) / position.entry_price
        max_drawdown = 1 - max(future_prices) / position.entry_price
    
    # Exit now is better if:
    # 1. Future PnL is worse than current
    # 2. OR max drawdown in future is severe
    should_exit = (future_pnl < current_pnl - 0.005) or (max_drawdown < -0.03)
    
    return 1 if should_exit else 0
```

---

# TRAINING DATA PIPELINE

## Data Collection

```python
async def collect_training_data(symbols, start_date, end_date):
    """
    Collect historical data for training.
    """
    all_data = {}
    
    for symbol in symbols:
        data = {
            "ohlcv_1m": await fetch_historical_ohlcv(symbol, "1m", start_date, end_date),
            "ohlcv_5m": await fetch_historical_ohlcv(symbol, "5m", start_date, end_date),
            "funding": await fetch_historical_funding(symbol, start_date, end_date),
            "oi": await fetch_historical_oi(symbol, start_date, end_date),
            "liquidations": await fetch_historical_liquidations(symbol, start_date, end_date),
        }
        all_data[symbol] = data
    
    return all_data
```

## Feature Engineering

```python
def engineer_features(raw_data):
    """
    Create all features from raw data.
    """
    df = pd.DataFrame()
    
    # Price features
    df["return_1m"] = raw_data["close"].pct_change(1)
    df["return_5m"] = raw_data["close"].pct_change(5)
    df["return_15m"] = raw_data["close"].pct_change(15)
    df["return_1h"] = raw_data["close"].pct_change(60)
    
    # Volatility
    df["volatility_5m"] = df["return_1m"].rolling(5).std() * np.sqrt(365*24*60)
    df["volatility_1h"] = df["return_1m"].rolling(60).std() * np.sqrt(365*24*60)
    
    # Technical indicators
    df["sma_20"] = raw_data["close"].rolling(20).mean()
    df["sma_50"] = raw_data["close"].rolling(50).mean()
    df["price_vs_sma_20"] = raw_data["close"] / df["sma_20"] - 1
    df["price_vs_sma_50"] = raw_data["close"] / df["sma_50"] - 1
    
    # RSI
    delta = raw_data["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / loss))
    
    # ATR
    high_low = raw_data["high"] - raw_data["low"]
    high_close = abs(raw_data["high"] - raw_data["close"].shift())
    low_close = abs(raw_data["low"] - raw_data["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    
    # Funding features
    df["funding_rate"] = raw_data["funding_rate"]
    df["funding_zscore"] = (df["funding_rate"] - df["funding_rate"].rolling(100).mean()) / df["funding_rate"].rolling(100).std()
    
    # OI features
    df["oi_delta_pct"] = raw_data["oi"].pct_change()
    df["oi_zscore"] = (df["oi_delta_pct"] - df["oi_delta_pct"].rolling(100).mean()) / df["oi_delta_pct"].rolling(100).std()
    
    # Time features
    df["hour_sin"] = np.sin(2 * np.pi * raw_data.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * raw_data.index.hour / 24)
    
    return df.dropna()
```

## Dataset Creation

```python
def create_signal_scorer_dataset(features_df, signals_df, lookahead=30):
    """
    Create labeled dataset for signal scorer.
    """
    X = []
    y = []
    
    for idx, signal in signals_df.iterrows():
        # Get features at signal time
        signal_features = features_df.loc[signal.timestamp]
        
        # Get label (was it profitable?)
        future_idx = signal.timestamp + timedelta(minutes=lookahead)
        if future_idx not in features_df.index:
            continue
        
        entry_price = features_df.loc[signal.timestamp, "close"]
        exit_price = features_df.loc[future_idx, "close"]
        
        if signal.side == "LONG":
            profitable = exit_price > entry_price * 1.001  # Account for fees
        else:
            profitable = exit_price < entry_price * 0.999
        
        X.append(signal_features)
        y.append(1 if profitable else 0)
    
    return pd.DataFrame(X), pd.Series(y)
```

---

# TRAINING SCHEDULE

## Initial Training

```
1. Collect 1 year of historical data (all 8 pairs)
2. Engineer features
3. Generate synthetic signals using strategies
4. Create labeled datasets
5. Train all models
6. Validate on held-out period (last 2 months)
7. Deploy if metrics acceptable
```

## Retraining Schedule

| Model | Frequency | Trigger |
|-------|-----------|---------|
| Signal Scorer | Weekly | Or if accuracy drops 5% |
| Regime Classifier | Weekly | Or on regime distribution shift |
| Direction Predictor | Monthly | Or on significant underperformance |
| Exit Predictor | Weekly | Or if exit timing degrades |

## Model Monitoring

```python
def monitor_model_performance(model, recent_predictions, recent_outcomes):
    """
    Track model performance and alert if degraded.
    """
    accuracy = (recent_predictions == recent_outcomes).mean()
    baseline_accuracy = 0.65  # Expected
    
    if accuracy < baseline_accuracy - 0.05:
        alert("Model accuracy degraded", accuracy)
        trigger_retraining()
    
    # Track calibration
    for bucket in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = (recent_predictions >= bucket) & (recent_predictions < bucket + 0.1)
        if mask.sum() > 10:
            actual_rate = recent_outcomes[mask].mean()
            expected_rate = bucket + 0.05
            if abs(actual_rate - expected_rate) > 0.1:
                alert(f"Calibration drift at {bucket}", actual_rate)
```

---

# MINIMUM VIABLE MODELS

For initial deployment without ML, use rule-based alternatives:

## Signal Scorer (Rule-Based)

```python
def rule_based_signal_score(signal, market_state, stat_result):
    score = signal.confidence
    
    # Boost for favorable conditions
    if stat_result.volatility_regime == "normal":
        score *= 1.1
    if abs(market_state.funding_rate.rate) > 0.0005:
        score *= 1.1  # Strong funding = stronger signal
    
    # Penalize for unfavorable
    if stat_result.volatility_regime == "high":
        score *= 0.8
    if stat_result.cascade_probability > 0.3:
        score *= 0.7
    
    return min(1.0, score)
```

## Regime Classifier (Rule-Based)

```python
def rule_based_regime(market_state, stat_result):
    if stat_result.cascade_probability > 0.5:
        return Regime.CASCADE_RISK
    
    if stat_result.volatility_regime == "extreme":
        return Regime.HIGH_VOLATILITY
    
    funding = market_state.funding_rate.rate
    oi_delta = market_state.open_interest.delta_pct
    
    if funding > 0.001 and oi_delta > 0.02:
        return Regime.SQUEEZE_SHORT
    if funding < -0.001 and oi_delta > 0.02:
        return Regime.SQUEEZE_LONG
    
    # Simple trend detection
    sma_20 = calculate_sma(market_state.ohlcv["5m"], 20)
    sma_50 = calculate_sma(market_state.ohlcv["5m"], 50)
    
    if sma_20 > sma_50 * 1.02:
        return Regime.TRENDING_UP
    if sma_20 < sma_50 * 0.98:
        return Regime.TRENDING_DOWN
    
    return Regime.RANGING
```

---

# LLM NEWS ANALYST

## Purpose
Provide news-based context for trading decisions by analyzing crypto news every 30 minutes.

## Architecture
- **LLM**: Claude 3.5 Sonnet (Anthropic)
- **News Source**: CryptoCompare API
- **Scan Interval**: 30 minutes
- **Output**: Per-pair action recommendations

## How It Works

```
Every 30 minutes:
│
├── Fetch News
│   ├── CryptoCompare news API
│   ├── Filter last 2 hours
│   └── Extract headlines + summaries
│
├── Build Prompt
│   ├── Include news context
│   ├── List all 8 trading pairs
│   └── Request JSON output
│
├── Call Claude
│   ├── Rate-limited (max 10 calls/hour)
│   └── Parse JSON response
│
└── Cache Results
    ├── Store per-pair analysis
    └── Used by Layer 3 for gating
```

## Per-Pair Analysis Structure

```python
@dataclass
class PairAnalysis:
    symbol: str           # e.g., "BTCUSDT"
    action: str           # "bullish" | "bearish" | "hold" | "exit"
    confidence: str       # "high" | "medium" | "low"
    reason: str           # Brief explanation
    timestamp: datetime   # When analysis was generated
```

## Action Definitions

| Action | Meaning | Trade Impact |
|--------|---------|--------------|
| `bullish` | Positive news/sentiment | Allow LONG, block SHORT |
| `bearish` | Negative news/sentiment | Allow SHORT, block LONG |
| `hold` | Unclear or mixed signals | Block all new entries |
| `exit` | Significant negative news | Block entries, recommend exit |

## Trade Gating Logic

```python
def should_trade(symbol: str, direction: str) -> tuple[bool, str]:
    """Check if LLM supports this trade direction."""
    analysis = get_cached_analysis(symbol)
    
    if not analysis:
        return True, "No LLM analysis, allowing trade"
    
    # Always block if LLM says exit
    if analysis.action == "exit":
        return False, f"LLM exit: {analysis.reason}"
    
    # Block conflicting directions
    if analysis.action == "bearish" and direction == "long":
        return False, f"LLM bearish, blocking long"
    
    if analysis.action == "bullish" and direction == "short":
        return False, f"LLM bullish, blocking short"
    
    # Block if hold
    if analysis.action == "hold":
        return False, f"LLM hold: {analysis.reason}"
    
    return True, f"LLM {analysis.action}"
```

## Example LLM Response

```json
[
  {
    "symbol": "BTCUSDT",
    "action": "bullish",
    "confidence": "medium",
    "reason": "Institutional inflows continue, ETF demand strong"
  },
  {
    "symbol": "ETHUSDT", 
    "action": "hold",
    "confidence": "low",
    "reason": "Mixed signals on L2 adoption metrics"
  },
  {
    "symbol": "SOLUSDT",
    "action": "bearish",
    "confidence": "high",
    "reason": "Network congestion issues reported"
  }
]
```

---

# SUMMARY

## Current Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| ML Signal Scorer | ✅ Active | `models/signal_scorer.pkl` |
| LLM News Analyst | ✅ Active | `hydra/layers/llm_analyst.py` |
| Regime Classifier | ⚙️ Rule-based | `hydra/layers/layer2_statistical.py` |
| Direction Predictor | 📝 Placeholder | `hydra/layers/layer3_alpha/transformer_model.py` |
| Exit Predictor | ❌ Not implemented | - |

## Training Commands

```bash
# Train Signal Scorer
python scripts/train_signal_scorer.py

# Output:
# - models/signal_scorer.pkl (trained model)
# - Training metrics and feature importance
```

## Configuration

```env
# ML Thresholds
ML_SCORE_THRESHOLD=0.45
MIN_SIGNAL_CONFIDENCE=0.50

# LLM Settings
ANTHROPIC_API_KEY=your_key
LLM_MODEL=claude-3-5-sonnet-20241022
LLM_SCAN_INTERVAL=30  # minutes

# News API
CRYPTOCOMPARE_API_KEY=your_key
```

## Model Performance Expectations

| Metric | Target | Notes |
|--------|--------|-------|
| ML Accuracy | > 55% | Better than random |
| ML Precision | > 60% | Minimize false positives |
| ML Recall | > 50% | Catch profitable signals |
| LLM Agreement | > 70% | Align with market moves |
