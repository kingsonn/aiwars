# HYDRA Profitability Audit - Honest Assessment

## Question: Is this the best we can do?

**Short answer: No.** The current implementation has significant gaps that would likely result in **losses** if deployed, even in paper trading.

---

## Critical Issues Found

### 1. ❌ Model-to-Execution Disconnect

**Problem:** The transformer model is trained on labels that don't match how we trade.

```
Training Labels:
- Direction: 0.5% threshold over 1 hour (12 candles)
- This means: "Will price move >0.5% in 1 hour?"

Execution Reality:
- Trades have fees (~0.04% taker × 2 = 0.08%)
- Slippage (~0.05-0.2%)
- Funding payments (up to ±0.1% every 8h)
- Stop losses get triggered by noise

Net cost per trade: ~0.15-0.30%
Required move to break even: ~0.3%
```

**Impact:** Model says "go long" for a +0.5% predicted move, but after costs we make +0.2%. Over many trades, this barely breaks even.

### 2. ❌ Class Imbalance in Labels

**Problem:** With 0.5% threshold:
- ~20-30% of candles are "Long" signals
- ~20-30% are "Short" signals  
- ~40-60% are "Flat"

The model learns to predict "Flat" too often because it's the safe bet. A 92% "accuracy" might mean it's just predicting Flat 60% of the time correctly.

### 3. ❌ No Transaction Cost Awareness

**Problem:** The model doesn't know about trading costs.

```python
# Current loss function (trainer.py line 151-159):
loss = (
    loss_dir * 1.0 +      # Direction classification
    loss_ae * 0.5 +       # Adverse excursion
    loss_fe * 0.5 +       # Favorable excursion
    loss_regime * 0.3 +   # Regime
    loss_vol * 0.2        # Volatility
)
```

**Missing:** Cost-adjusted return prediction, risk-adjusted profit optimization.

### 4. ❌ Leaky Labels (Look-Ahead Bias)

**Problem:** The labels use future data that wouldn't be available at trade time.

```python
# data_pipeline.py line 145-147:
for i in range(len(df) - horizon):
    future_returns[i] = (close[i + horizon] - close[i]) / close[i]
```

This is fine for training, but the model might learn patterns that only exist because we know the future. Real trading doesn't have this.

### 5. ❌ Symbol Embedding Not Leveraged Properly

**Current:** Symbol embedding just modifies volatility scaling.

**Should be:** Different pairs have different:
- Optimal holding periods (BTC: hours, DOGE: minutes)
- Momentum characteristics
- Correlation lead/lag relationships
- Funding rate patterns

### 6. ❌ No Cross-Pair Signals

**Problem:** BTC often leads alts by 5-30 minutes. Current model processes each pair independently.

**Should have:** Attention mechanism that sees all pairs simultaneously:
```
If BTC just moved +2%, SOL/DOGE will likely follow
If BTC is flat but DOGE spikes, it's likely to revert
```

### 7. ❌ Orderbook/Liquidation Features Are Zeros

```python
# data_pipeline.py line 321-323:
orderbook_feat = np.zeros((len(df), 4))
liq_feat = np.zeros((len(df), 4))
```

These are placeholder zeros. The model is learning nothing from orderbook imbalance or liquidation cascades - which are the most predictive features in perps.

### 8. ❌ Funding Rate Not In Training Data

```python
# data_pipeline.py line 86-89:
def compute_funding_features(self, df: pd.DataFrame) -> np.ndarray:
    if 'funding_rate' not in df.columns:
        return np.zeros((len(df), 3))  # Returns zeros!
```

We fetch OHLCV but don't fetch/merge funding rates into training data.

---

## What Would Actually Make Money

### Fix 1: Return-Based Labels (Not Direction)

Instead of classifying direction, predict **risk-adjusted return**:

```python
# Better label: Expected profit after costs
expected_profit = (future_return - transaction_cost) / realized_volatility
```

### Fix 2: Asymmetric Loss Function

Penalize false positives more than false negatives:
- Wrong direction + entry = lose money
- No entry + missed opportunity = just neutral

```python
# Asymmetric loss
loss = torch.where(
    (pred_long & actual_short) | (pred_short & actual_long),
    base_loss * 3.0,  # 3x penalty for wrong direction
    base_loss
)
```

### Fix 3: Fetch Real Funding Rates

```python
# Merge funding into training data
funding_df = await pipeline.fetch_funding_history(symbol)
df = df.merge(funding_df, left_index=True, right_index=True, how='left')
df['funding_rate'].fillna(method='ffill', inplace=True)
```

### Fix 4: Cross-Pair Attention

Add a "market context" encoder that processes BTC state before predicting alts:

```python
class CrossPairAttention(nn.Module):
    def __init__(self, d_model):
        self.btc_context = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, 8)
```

### Fix 5: Time-Weighted Sampling

Recent data should matter more:

```python
# Weight recent samples higher
weights = np.exp(-np.arange(len(examples))[::-1] / (len(examples) / 3))
sampler = WeightedRandomSampler(weights, len(examples))
```

### Fix 6: Proper Backtesting Before Paper Trading

Current system has no proper walk-forward backtest with realistic fills.

---

## Recommended Priority Fixes

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| **P0** | Fetch real funding rates | 2 hours | High |
| **P0** | Fix return-based labels | 4 hours | Critical |
| **P1** | Add BTC lead signal for alts | 4 hours | High |
| **P1** | Asymmetric loss function | 2 hours | Medium |
| **P2** | Cross-pair attention | 8 hours | High |
| **P2** | Proper backtest validation | 8 hours | Critical |
| **P3** | Real orderbook data | Days | High |

---

## Honest Probability of Profitability

| Scenario | Current System | After Fixes |
|----------|----------------|-------------|
| Profitable after 30 days | 15% | 45% |
| Break-even after 30 days | 30% | 35% |
| Loses money after 30 days | 55% | 20% |

**Why so pessimistic?**

Even hedge funds with 100x our resources struggle in crypto. The market is:
- Highly efficient for simple patterns
- Dominated by HFT firms with microsecond latency
- Subject to random news/regulation events
- Manipulated by whales

**What we CAN exploit:**
- Retail leverage crowding (funding rates)
- Cascade/squeeze events
- Alt correlation lag behind BTC
- Regime-based position sizing

---

## Conclusion

The current architecture is **reasonable but incomplete**. To have a real chance at profitability:

1. Fix the training labels to be cost-aware
2. Add funding rate data
3. Implement BTC→alt lead signal
4. Do proper walk-forward backtesting

Without these fixes, you're essentially gambling with extra steps.
