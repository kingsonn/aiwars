# HYDRA — Layer Specifications
## Detailed Technical Specification for Each Layer

**Version:** 2.0  
**Last Updated:** December 31, 2024  

---

# LAYER 1: MARKET INTELLIGENCE

## Purpose
Collect ALL market data needed for decisions. No opinions — just facts.

**Key responsibilities:**
- Fetch real-time price, funding, OI, liquidation data
- Maintain multi-timeframe candle history
- Provide clean, normalized data to downstream layers

## Data Sources

### Primary Data (Exchange API)

| Data | Update Freq | Fields |
|------|-------------|--------|
| OHLCV | Every candle | open, high, low, close, volume |
| Funding Rate | 8 hours | rate, predicted_rate, next_time |
| Open Interest | 15 sec | oi_contracts, oi_usd, delta, delta_pct |
| Order Book | 5 sec | top 20 bids/asks, spread, imbalance |
| Trades | Continuous | last 100 trades for CVD |
| Liquidations | Continuous | side, size, price, usd_value |

### External Data (Optional)

| Source | Data | Use |
|--------|------|-----|
| Binance API | Long/Short Ratio | Crowd positioning |
| Binance API | Taker Buy/Sell | Aggressor flow |
| CoinGlass | Aggregated liquidations | Market stress |
| Alternative.me | Fear & Greed | Sentiment |

## Output: MarketState

```python
@dataclass
class MarketState:
    timestamp: datetime
    symbol: str
    
    # Price
    price: float
    ohlcv: dict[str, list[OHLCV]]  # timeframe → candles
    
    # Futures-specific
    funding_rate: FundingRate
    open_interest: OpenInterest
    recent_liquidations: list[Liquidation]
    order_book: OrderBookSnapshot
    
    # Derived
    volatility: float
    volume_24h: float
    price_change_24h: float
```

## Data Requirements

**Per symbol, keep:**
- 500 candles × 5 timeframes (1m, 5m, 15m, 1h, 4h)
- Last 100 liquidations
- Last 100 trades
- Current funding, OI, order book

**Update cycle:** 30 seconds

---

# LAYER 2: STATISTICAL REALITY

## Purpose
Answer: **"Is it safe to trade right now?"**

## Output

```python
class TradabilityStatus(Enum):
    ALLOW = "allow"       # Safe to trade
    RESTRICT = "restrict" # Exits only, no entries
    BLOCK = "block"       # Close everything
```

## Checks Performed

### 1. Volatility Regime
```python
def check_volatility(returns: np.array) -> str:
    vol = np.std(returns) * np.sqrt(365 * 24 * 12)  # Annualized
    percentile = rank_vs_history(vol)
    
    if percentile > 95: return "extreme"    # → BLOCK
    if percentile > 80: return "high"       # → RESTRICT
    if percentile > 50: return "elevated"   # → Caution
    return "normal"                         # → ALLOW
```

### 2. Cascade Risk
```python
def check_cascade(liquidations: list, oi: float) -> float:
    recent_liq = sum(l.usd_value for l in liquidations[-100:])
    velocity = recent_liq / oi
    
    if velocity > 0.05: return 0.9   # → BLOCK
    if velocity > 0.02: return 0.6   # → RESTRICT
    return velocity * 10              # Linear
```

### 3. ML-Based Regime Detection

Layer 2 uses **Model 2: Regime Classifier** (XGBoost) to detect market regimes:

```python
class Regime(Enum):
    TRENDING_UP = 0       # Clear upward trend
    TRENDING_DOWN = 1     # Clear downward trend
    RANGING = 2           # Sideways consolidation
    HIGH_VOLATILITY = 3   # Elevated volatility
    CASCADE_RISK = 4      # Liquidation cascade danger
    SQUEEZE_LONG = 5      # Shorts getting squeezed
    SQUEEZE_SHORT = 6     # Longs getting squeezed
```

**Model Details:**
- **Type**: XGBoost Multi-class Classifier
- **Classes**: 7 regimes
- **File**: `models/regime_classifier.pkl`
- **Features**: Trend indicators, volatility metrics, funding/positioning, liquidation data, volume

**Usage in Layer 2:**
```python
from hydra.training.models import load_regime_classifier

regime_classifier = load_regime_classifier()
features = extract_regime_features(market_state)
regime_proba = regime_classifier.predict_proba(features)
regime = regime_classifier.predict(features)[0]
regime_confidence = regime_proba.max()
```

### 4. Data Health
```python
def check_data_health(state: MarketState) -> bool:
    checks = [
        (now - state.timestamp).seconds < 60,  # Fresh
        state.funding_rate is not None,
        state.open_interest is not None,
        len(state.ohlcv.get("5m", [])) >= 20,
        state.order_book.spread < 0.01,  # <1% spread
    ]
    return all(checks)
```

## Output Structure

```python
@dataclass
class StatisticalResult:
    tradability: TradabilityStatus
    tradability_reason: str
    
    regime: Regime
    regime_confidence: float
    
    volatility_regime: str  # low/normal/high/extreme
    volatility_zscore: float
    realized_volatility: float
    
    cascade_probability: float
    liquidation_velocity: float
    
    expected_range_1h: tuple[float, float]
    regime_break_alert: bool
```

## Decision Matrix

| Condition | Result | Action |
|-----------|--------|--------|
| Data health fail | BLOCK | Stop all |
| Vol extreme | BLOCK | Close positions |
| Cascade > 70% | BLOCK | Close positions |
| Vol high | RESTRICT | Exits only |
| Funding > 0.1% | RESTRICT | Reduce leverage |
| Normal | ALLOW | Trade |

---

# LAYER 3: ALPHA GENERATION

## Purpose
Generate signals: LONG, SHORT, or FLAT with confidence score.

**Key responsibilities:**
- Generate behavioral signals from market data
- Score signals with ML model (49 features → P(profitable))
- Gate signals with LLM news analysis
- Return best signal for trading decision

## Architecture

```
Layer 3 Flow:
│
├── BEHAVIORAL SIGNAL GENERATORS
│   ├── FUNDING_SQUEEZE
│   ├── LIQUIDATION_REVERSAL
│   ├── OI_DIVERGENCE
│   ├── CROWDING_FADE
│   └── FUNDING_CARRY
│   
├── ML SIGNAL SCORER
│   ├── Extract 49 features
│   ├── XGBoost model: predict P(profitable)
│   ├── Add score to signal metadata
│   └── Mark as approved if score >= 0.45
│   
└── LLM NEWS ANALYST (parallel, every 30 min)
    ├── Fetch crypto news
    ├── Per-pair analysis with Claude
    ├── Cache recommendations
    └── Gate trades based on sentiment
```

## Signal Structure

```python
@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    side: Side              # LONG / SHORT / FLAT
    confidence: float       # 0.0 to 1.0
    expected_return: float  # Expected profit %
    expected_adverse_excursion: float  # Expected max loss %
    holding_period_minutes: int
    source: str             # Which strategy
    regime: Regime          # Market regime at signal time
    metadata: dict          # Contains ml_score, ml_approved, thesis
```

## Signal Sources (Behavioral Primitives)

### 1. FUNDING_SQUEEZE
**Logic:** Extreme funding bleeds one side → they capitulate

```python
def funding_squeeze(state: MarketState) -> Signal | None:
    funding = state.funding_rate.rate
    oi_delta = state.open_interest.delta_pct
    
    # Longs paying → go SHORT
    if funding > 0.0005 and oi_delta > 0:
        return Signal(
            side=Side.SHORT,
            confidence=min(0.8, funding * 500),
            expected_return=0.02,
            expected_adverse_excursion=0.01,
            source="FUNDING_SQUEEZE",
            thesis=f"Longs paying {funding*100:.3f}%"
        )
    
    # Shorts paying → go LONG
    if funding < -0.0005 and oi_delta > 0:
        return Signal(
            side=Side.LONG,
            confidence=min(0.8, abs(funding) * 500),
            ...
        )
```

### 2. LIQUIDATION_REVERSAL
**Logic:** After cascade, forced sellers exhausted → reversal

```python
def liquidation_reversal(state: MarketState, stat: StatisticalResult) -> Signal | None:
    long_liq = sum(l.usd for l in state.liquidations if l.side == Side.LONG)
    short_liq = sum(l.usd for l in state.liquidations if l.side == Side.SHORT)
    imbalance = (long_liq - short_liq) / (long_liq + short_liq)
    
    # Heavy long liquidations → longs exhausted → LONG for bounce
    if imbalance > 0.7 and stat.cascade_probability < 0.3:
        return Signal(side=Side.LONG, confidence=0.65, ...)
    
    # Heavy short liquidations → SHORT for pullback
    if imbalance < -0.7 and stat.cascade_probability < 0.3:
        return Signal(side=Side.SHORT, confidence=0.65, ...)
```

### 3. OI_DIVERGENCE
**Logic:** Price vs OI divergence = weak move

```python
def oi_divergence(state: MarketState) -> Signal | None:
    price_change = state.price_change_24h
    oi_change = state.open_interest.delta_pct
    
    # Price up but OI down = weak rally → SHORT
    if price_change > 0.02 and oi_change < -0.02:
        return Signal(side=Side.SHORT, confidence=0.55, ...)
    
    # Price down but OI down = weak selloff → LONG
    if price_change < -0.02 and oi_change < -0.02:
        return Signal(side=Side.LONG, confidence=0.55, ...)
```

### 4. CROWDING_FADE
**Logic:** Everyone on same side → fade them

```python
def crowding_fade(state: MarketState, ls_ratio: float) -> Signal | None:
    funding = state.funding_rate.rate
    
    # Extreme long crowding → SHORT
    if ls_ratio > 2.0 and funding > 0.0003:
        return Signal(side=Side.SHORT, confidence=0.6, ...)
    
    # Extreme short crowding → LONG
    if ls_ratio < 0.5 and funding < -0.0003:
        return Signal(side=Side.LONG, confidence=0.6, ...)
```

### 5. FUNDING_CARRY
**Logic:** In ranges, collect funding

```python
def funding_carry(state: MarketState, stat: StatisticalResult) -> Signal | None:
    if stat.regime != Regime.RANGING:
        return None
    if stat.volatility_regime in ["high", "extreme"]:
        return None
    
    funding = state.funding_rate.rate
    
    # Positive funding → SHORT to receive
    if funding > 0.0005:
        return Signal(side=Side.SHORT, confidence=0.5, ...)
    
    # Negative funding → LONG to receive
    if funding < -0.0005:
        return Signal(side=Side.LONG, confidence=0.5, ...)
```

## ML Signal Scoring

The ML Signal Scorer evaluates every generated signal using 49 features:

### Feature Categories

| Category | Count | Features |
|----------|-------|----------|
| Signal | 9 | direction, confidence, 5 source one-hots, expected_return, expected_adverse_excursion |
| Price | 10 | return_1m/5m/15m/1h, volatility_5m/1h, price_vs_sma_20/50, rsi_14, atr_14 |
| Funding | 4 | rate, zscore, annualized, momentum |
| OI | 3 | delta_pct, delta_zscore, price_divergence |
| Liquidation | 3 | imbalance, velocity, zscore |
| Order Book | 4 | imbalance, spread_bps, bid_depth_usd, ask_depth_usd |
| Positioning | 2 | long_short_ratio, taker_buy_sell_ratio |
| Regime | 9 | 7 regime one-hots, volatility_regime, cascade_probability |
| Time | 5 | hour_sin/cos, day_sin/cos, minutes_to_funding |

### Scoring Flow

```python
def score_signals_with_ml(signals: list[Signal], market_state, stat_result) -> list[Signal]:
    """Score ALL signals with ML - doesn't filter, just adds scores."""
    for signal in signals:
        # Extract 49 features
        features = extract_ml_features(signal, market_state, stat_result)
        
        # Get probability from XGBoost model
        score = ml_model.predict_proba(features)[0, 1]  # P(profitable)
        
        # Add to signal metadata
        signal.metadata["ml_score"] = score
        signal.metadata["ml_approved"] = score >= ML_SCORE_THRESHOLD
    
    # Sort by approval status, then confidence
    signals.sort(key=lambda s: (s.metadata["ml_approved"], s.confidence), reverse=True)
    return signals

ML_SCORE_THRESHOLD = 0.45
```

### Training

The model is trained using `scripts/train_signal_scorer.py`:
- Historical data: OHLCV, funding, OI, liquidations
- Signal generation: Same behavioral generators used in live
- Labels: 1 if signal would have been profitable after fees, 0 otherwise
- Model: XGBoost with time-series cross-validation

## LLM News Analyst

The LLM News Analyst runs independently every 30 minutes, analyzing news for each trading pair:

### Architecture

```python
class LLMNewsAnalyst:
    """Independent LLM-powered news analysis for trading pairs."""
    
    scan_interval: int = 30  # minutes
    _pair_analysis: dict[str, PairAnalysis]  # Cached per-pair results
    
    async def scan_all_pairs(self, force: bool = False):
        """Scan news and analyze all pairs every 30 minutes."""
        if not force and not self._should_scan():
            return
        
        # Fetch recent crypto news
        news = await self._fetch_news()
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(news)
        
        # Get LLM response (Claude)
        response = await self._call_llm(prompt)
        
        # Parse and cache per-pair analysis
        self._parse_and_cache(response)
    
    def should_trade(self, symbol: str, direction: str) -> tuple[bool, str]:
        """Check if LLM analysis supports this trade."""
        analysis = self._pair_analysis.get(symbol)
        if not analysis:
            return True, "No LLM analysis available"
        
        # Block if LLM recommends exit
        if analysis.action == "exit":
            return False, f"LLM recommends exit: {analysis.reason}"
        
        # Block if direction conflicts
        if analysis.action == "bearish" and direction == "long":
            return False, f"LLM bearish, blocking long: {analysis.reason}"
        if analysis.action == "bullish" and direction == "short":
            return False, f"LLM bullish, blocking short: {analysis.reason}"
        
        return True, f"LLM {analysis.action}: {analysis.reason}"
```

### Per-Pair Analysis Output

```python
@dataclass
class PairAnalysis:
    symbol: str
    action: str      # "bullish" | "bearish" | "hold" | "exit"
    confidence: str  # "high" | "medium" | "low"
    reason: str      # Brief explanation
    timestamp: datetime
```

### LLM Prompt Structure

```
Analyze these crypto pairs based on recent news:
[News headlines and summaries]

For each pair (BTC, ETH, SOL, BNB, ADA, XRP, LTC, DOGE), provide:
- action: bullish/bearish/hold/exit
- confidence: high/medium/low  
- reason: 1-2 sentence explanation

Return as JSON array.
```

---

# LAYER 4: RISK BRAIN

## Purpose
Final gatekeeper. Decides:
1. Approve or veto trade
2. Position size
3. Leverage
4. Stop-loss / take-profit

**Principle: HYDRA prefers not trading over dying.**

## Position Sizing

### Kelly Criterion
```python
def kelly_size(win_prob: float, win_loss_ratio: float) -> float:
    """Optimal bet fraction."""
    p, q, b = win_prob, 1 - win_prob, win_loss_ratio
    return max(0, (p * b - q) / b)

# We use quarter Kelly
KELLY_FRACTION = 0.25
```

### Full Calculation
```python
def calculate_size(signal, equity, leverage, vol_mult, size_mult, corr_penalty):
    # Kelly sizing
    kelly = kelly_size(signal.confidence, 
                       signal.expected_return / signal.expected_adverse_excursion)
    kelly_size_usd = equity * kelly * KELLY_FRACTION
    
    # Risk-based sizing (1% risk per trade)
    risk_size_usd = (equity * 0.01) / signal.expected_adverse_excursion
    
    # Take minimum
    base = min(kelly_size_usd, risk_size_usd)
    
    # Apply adjustments
    size = base * leverage
    size /= vol_mult        # Higher vol = smaller
    size *= size_mult       # Liquidity adjustment
    size *= corr_penalty    # Correlation reduction
    
    # Apply limits
    size = min(size, MAX_POSITION_SIZE)
    size = min(size, remaining_exposure_capacity)
    
    return size
```

### Pair Adjustments
```python
VOLATILITY_MULT = {
    "cmt_btcusdt": 1.0, "cmt_ethusdt": 1.15, "cmt_solusdt": 1.50,
    "cmt_bnbusdt": 1.10, "cmt_adausdt": 1.30, "cmt_xrpusdt": 1.25,
    "cmt_ltcusdt": 1.05, "cmt_dogeusdt": 2.00
}

SIZE_MULT = {
    "cmt_btcusdt": 1.0, "cmt_ethusdt": 1.0, "cmt_solusdt": 0.8,
    "cmt_bnbusdt": 0.8, "cmt_adausdt": 0.6, "cmt_xrpusdt": 0.7,
    "cmt_ltcusdt": 0.7, "cmt_dogeusdt": 0.5
}
```

### Correlation Penalty
```python
def correlation_penalty(new_symbol, existing_positions) -> float:
    """Reduce size if correlated positions exist."""
    avg_corr = calculate_avg_correlation(new_symbol, existing_positions)
    
    if avg_corr > 0.8: return 0.3  # 70% reduction
    if avg_corr > 0.6: return 0.6  # 40% reduction
    if avg_corr > 0.4: return 0.8  # 20% reduction
    return 1.0
```

## Leverage Calculation

```python
def calculate_leverage(signal, stat_result, portfolio) -> float:
    leverage = BASE_LEVERAGE  # 5x
    
    # Volatility
    if stat_result.volatility_regime == "extreme": leverage *= 0.3
    elif stat_result.volatility_regime == "high": leverage *= 0.5
    
    # Funding
    if abs(funding) > 0.001: leverage *= 0.5
    elif abs(funding) > 0.0005: leverage *= 0.7
    
    # Regime
    if stat_result.regime == Regime.CASCADE_RISK: leverage *= 0.3
    
    # Confidence
    leverage *= signal.confidence
    
    # Drawdown penalty
    if portfolio.drawdown > 0.05:
        leverage *= max(0.2, 1 - portfolio.drawdown / 0.15)
    
    return max(1.0, min(leverage, MAX_LEVERAGE))
```

## Stop-Loss / Take-Profit

```python
def calculate_exits(signal, state, stat_result):
    price = state.price
    eae = signal.expected_adverse_excursion
    
    # Adjust for volatility
    if stat_result.volatility_regime == "high": eae *= 1.3
    if stat_result.volatility_regime == "extreme": eae *= 1.5
    
    # Cap at reasonable levels
    eae = max(0.01, min(eae, 0.10))  # 1% to 10%
    
    if signal.side == Side.LONG:
        stop_loss = price * (1 - eae)
        take_profit = price * (1 + eae * 2)  # 2:1 R:R
    else:
        stop_loss = price * (1 + eae)
        take_profit = price * (1 - eae * 2)
    
    return stop_loss, take_profit
```

## Kill Switches

```python
KILL_CONDITIONS = [
    ("Daily drawdown > 5%", lambda p: p.drawdown > 0.05),
    ("Funding spike > 0.5%", lambda s: abs(s.funding_rate.rate) > 0.005),
    ("Cascade probability > 70%", lambda r: r.cascade_probability > 0.7),
    ("Regime break + extreme vol", lambda r: r.regime_break_alert and r.volatility_regime == "extreme"),
]

def check_kill_switch(portfolio, state, stat_result) -> tuple[bool, str]:
    for reason, check in KILL_CONDITIONS:
        if check(portfolio) or check(state) or check(stat_result):
            return True, reason
    return False, ""
```

## Output

```python
@dataclass
class RiskDecision:
    approved: bool
    veto: bool
    veto_reason: str
    
    recommended_size_usd: float
    recommended_leverage: float
    
    stop_loss_price: float
    take_profit_price: float
    max_holding_time_hours: float
    
    trigger_kill_switch: bool
    kill_reason: str
```

---

# LAYER 5: EXECUTION ENGINE

## Purpose
Place orders on exchange. Handle fills, failures, partial fills.

## Order Types

| Type | Use | Description |
|------|-----|-------------|
| LIMIT | Entries | Post at best bid/ask |
| POST_ONLY | Entries | Maker only |
| REDUCE_ONLY | Exits | Can only close |
| STOP_LOSS | Risk | Trigger on price |
| TAKE_PROFIT | Exits | Trigger on target |

## Entry Execution

```python
async def execute_entry(signal, risk_decision):
    # Get prices
    best_bid, best_ask = await get_orderbook_top(signal.symbol)
    
    # Entry price
    entry_price = best_bid if signal.side == Side.LONG else best_ask
    quantity = risk_decision.recommended_size_usd / entry_price
    
    # Place limit order
    order = await exchange.create_order(
        symbol=signal.symbol,
        type="limit",
        side="buy" if signal.side == Side.LONG else "sell",
        amount=quantity,
        price=entry_price,
        params={"postOnly": True}
    )
    
    # Wait for fill (timeout 2 min)
    filled = await wait_for_fill(order, timeout=120)
    
    if not filled:
        await exchange.cancel_order(order.id)
        return None
    
    # Set stops
    await set_stop_loss(signal.symbol, signal.side, risk_decision.stop_loss_price)
    await set_take_profit(signal.symbol, signal.side, risk_decision.take_profit_price)
    
    return order
```

## Exit Execution

```python
async def execute_exit(position, reason):
    best_bid, best_ask = await get_orderbook_top(position.symbol)
    
    exit_price = best_bid if position.side == Side.LONG else best_ask
    order_side = "sell" if position.side == Side.LONG else "buy"
    
    # Reduce-only order
    order = await exchange.create_order(
        symbol=position.symbol,
        type="limit",
        side=order_side,
        amount=position.size,
        price=exit_price,
        params={"reduceOnly": True, "postOnly": True}
    )
    
    filled = await wait_for_fill(order, timeout=60)
    
    # If not filled, use market
    if not filled:
        await exchange.cancel_order(order.id)
        order = await exchange.create_order(
            type="market",
            params={"reduceOnly": True}
        )
    
    return order
```

---

**Continue to HYDRA_SPEC_TRADING.md for position management and exit logic →**
