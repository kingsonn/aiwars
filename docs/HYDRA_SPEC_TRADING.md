# HYDRA — Trading Logic Specification
## Position Sizing, Leverage, Entry/Exit Rules

**Version:** 2.0  
**Last Updated:** December 31, 2024  

---

# TRADE DECISION FLOW

Before any position sizing or entry logic, trades must pass multiple gates:

```
Signal Generated
      │
      ▼
┌─────────────────┐
│ ML Score Check  │ ── Score < 0.45 ──→ ML REJECT
│ (49 features)   │
└────────┬────────┘
         │ Score ≥ 0.45
         ▼
┌─────────────────┐
│ LLM Gate Check  │ ── Action=exit/hold ──→ LLM VETO
│ (per-pair cache)│ ── Direction conflict ──→ LLM VETO
└────────┬────────┘
         │ LLM approves
         ▼
┌─────────────────┐
│ Portfolio Check │ ── No margin ──→ NO MARGIN
│                 │ ── Already in position ──→ HOLDING
└────────┬────────┘
         │ Can trade
         ▼
┌─────────────────┐
│ Risk Brain (L4) │ ── Kill switch ──→ L4 VETO
│ Size + Leverage │ ── Limits exceeded ──→ L4 VETO
└────────┬────────┘
         │ Approved
         ▼
┌─────────────────┐
│ Execution (L5)  │ ── Order fails ──→ ERROR
│ Place order     │
└────────┬────────┘
         │ Filled
         ▼
    TRADE EXECUTED
```

---

# POSITION SIZING

## The Complete Formula

```
Position Size (USD) = min(
    Kelly_Size × 0.25 × Leverage,
    Risk_Size × Leverage,
    MAX_POSITION_SIZE × Pair_Size_Mult,
    Remaining_Exposure
) × Correlation_Penalty ÷ Volatility_Mult
```

## Step-by-Step Example

**Scenario:**
- Portfolio: $10,000
- Signal: LONG SOL, 70% confidence
- Expected return: 2%, Expected adverse excursion: 1%
- Existing position: LONG BTC $3,000

| Step | Calculation | Result |
|------|-------------|--------|
| 1. Kelly fraction | (0.7 × 2 - 0.3) / 2 | 0.55 |
| 2. Kelly size | $10,000 × 0.55 × 0.25 | $1,375 |
| 3. Risk size | ($10,000 × 0.01) / 0.01 | $10,000 |
| 4. Take minimum | min($1,375, $10,000) | $1,375 |
| 5. Apply leverage (5x) | $1,375 × 5 | $6,875 |
| 6. Volatility adj (SOL=1.5x) | $6,875 / 1.5 | $4,583 |
| 7. Size mult (SOL=0.8) | $4,583 × 0.8 | $3,667 |
| 8. Correlation (SOL-BTC=0.75) | $3,667 × 0.6 | $2,200 |
| 9. Check limits | min($2,200, $10,000) | **$2,200** |

**Result: Open $2,200 LONG SOL at 5x leverage**

## Hard Limits

```python
MAX_POSITION_SIZE_USD = 10_000    # Per position
MAX_TOTAL_EXPOSURE_USD = 50_000   # All positions
MAX_POSITIONS = 5                  # Max concurrent
MIN_POSITION_SIZE_USD = 100        # Don't bother below this
RISK_PER_TRADE_PCT = 1.0          # 1% of equity at risk
KELLY_FRACTION = 0.25             # Quarter Kelly
```

## Pair-Specific Multipliers

```python
# Volatility multiplier (divide size by this)
VOLATILITY_MULT = {
    "cmt_btcusdt": 1.00,   # Base
    "cmt_ethusdt": 1.15,   # 15% more volatile
    "cmt_solusdt": 1.50,   # 50% more volatile
    "cmt_bnbusdt": 1.10,
    "cmt_adausdt": 1.30,
    "cmt_xrpusdt": 1.25,
    "cmt_ltcusdt": 1.05,
    "cmt_dogeusdt": 2.00,  # Very volatile
}

# Size multiplier (multiply size by this)
SIZE_MULT = {
    "cmt_btcusdt": 1.0,    # Full size
    "cmt_ethusdt": 1.0,    # Full size
    "cmt_solusdt": 0.8,    # 80% max
    "cmt_bnbusdt": 0.8,
    "cmt_adausdt": 0.6,    # 60% max
    "cmt_xrpusdt": 0.7,
    "cmt_ltcusdt": 0.7,
    "cmt_dogeusdt": 0.5,   # Half size only
}
```

## Correlation Matrix

```python
PAIR_CORRELATIONS = {
    "cmt_btcusdt": {"cmt_ethusdt": 0.85, "cmt_solusdt": 0.75, "cmt_ltcusdt": 0.88, ...},
    "cmt_ethusdt": {"cmt_btcusdt": 0.85, "cmt_solusdt": 0.80, "cmt_adausdt": 0.82, ...},
    # ... full matrix in code
}

# Penalty based on average correlation with existing positions
# avg_corr > 0.8 → 70% size reduction (multiply by 0.3)
# avg_corr > 0.6 → 40% size reduction (multiply by 0.6)
# avg_corr > 0.4 → 20% size reduction (multiply by 0.8)
# avg_corr ≤ 0.4 → no reduction (multiply by 1.0)
```

---

# LEVERAGE RULES

## Base Calculation

```python
def calculate_leverage(signal, stat_result, portfolio):
    leverage = 5.0  # Base leverage
    
    # === VOLATILITY ADJUSTMENTS ===
    vol_multipliers = {
        "extreme": 0.3,   # 5x → 1.5x
        "high": 0.5,      # 5x → 2.5x
        "elevated": 0.8,  # 5x → 4x
        "normal": 1.0,    # 5x → 5x
        "low": 1.2,       # 5x → 6x
    }
    leverage *= vol_multipliers[stat_result.volatility_regime]
    
    # === FUNDING ADJUSTMENTS ===
    funding = abs(market_state.funding_rate.rate)
    if funding > 0.001:      # > 0.1%
        leverage *= 0.5
    elif funding > 0.0005:   # > 0.05%
        leverage *= 0.7
    
    # === REGIME ADJUSTMENTS ===
    regime_multipliers = {
        Regime.CASCADE_RISK: 0.3,
        Regime.SQUEEZE_LONG: 0.6,
        Regime.SQUEEZE_SHORT: 0.6,
        Regime.HIGH_VOLATILITY: 0.4,
        Regime.TRENDING_UP: 1.0,
        Regime.TRENDING_DOWN: 1.0,
        Regime.RANGING: 0.9,
    }
    leverage *= regime_multipliers.get(stat_result.regime, 1.0)
    
    # === CONFIDENCE SCALING ===
    leverage *= signal.confidence  # 70% conf → 70% leverage
    
    # === PORTFOLIO ADJUSTMENTS ===
    if portfolio.position_correlation > 0.7:
        leverage *= 0.6
    
    if portfolio.gross_leverage > 2:
        remaining = max(0, 20 - portfolio.gross_leverage)
        leverage = min(leverage, remaining)
    
    # === DRAWDOWN PENALTY ===
    if portfolio.current_drawdown > 0.05:
        penalty = 1 - (portfolio.current_drawdown / 0.15)
        leverage *= max(0.2, penalty)
    
    # === APPLY LIMITS ===
    return max(1.0, min(leverage, 20.0))
```

## Leverage by Scenario

| Scenario | Base | After Adjustments |
|----------|------|-------------------|
| Normal market, 70% conf | 5x | 3.5x |
| Low vol, 80% conf | 5x | 4.8x |
| High vol, 60% conf | 5x | 1.5x |
| Extreme vol | 5x | 1x or no trade |
| Cascade risk | 5x | 1.5x |
| Already 5x leveraged | 5x | ≤5x (remaining) |
| 8% drawdown | 5x | ~2x |

## Per-Pair Leverage Caps

```python
PAIR_MAX_LEVERAGE = {
    "cmt_btcusdt": 20,
    "cmt_ethusdt": 20,
    "cmt_solusdt": 15,
    "cmt_bnbusdt": 15,
    "cmt_adausdt": 10,
    "cmt_xrpusdt": 10,
    "cmt_ltcusdt": 15,
    "cmt_dogeusdt": 10,  # Dangerous
}
```

---

# ENTRY LOGIC

## When to Enter

A new position is opened when ALL of these gates pass:

```python
def should_enter(symbol, signal, stat_result, risk_decision, portfolio, llm_analyst):
    checks = [
        # 1. No existing position
        not has_position(symbol),
        
        # 2. Layer 2 allows entry
        stat_result.tradability == TradabilityStatus.ALLOW,
        
        # 3. Signal exists with sufficient confidence
        signal is not None,
        signal.confidence >= 0.50,  # MIN_SIGNAL_CONFIDENCE
        
        # 4. ML score above threshold
        signal.metadata.get("ml_approved", False),  # score >= 0.45
        
        # 5. LLM analysis supports direction
        llm_analyst.should_trade(symbol, signal.side.value)[0],
        
        # 6. Risk brain approved
        risk_decision.approved,
        not risk_decision.veto,
        
        # 7. Size is meaningful
        risk_decision.recommended_size_usd >= 100,
        
        # 8. Portfolio limits not exceeded
        portfolio.num_positions < MAX_POSITIONS,
        portfolio.total_exposure < MAX_TOTAL_EXPOSURE,
        
        # 9. No kill switch active
        not kill_switch_active,
    ]
    return all(checks)
```

## Gate Failure Codes

| Check Failed | Result Code | Action |
|--------------|-------------|--------|
| Existing position | HOLDING | Skip, manage existing |
| Layer 2 BLOCK | BLOCKED | Skip pair entirely |
| Low confidence | NO SIGNAL | Skip, wait for better |
| ML score low | ML REJECT | Signal shown but blocked |
| LLM conflict | LLM VETO | Signal shown but blocked |
| Risk veto | L4 VETO | Size/leverage issues |
| No margin | NO MARGIN | Skip, insufficient funds |
| Order fails | ERROR | Log and continue |

## Entry Execution Steps

```
1. Get current order book
2. Calculate entry price (best bid for LONG, best ask for SHORT)
3. Calculate quantity: size_usd / entry_price
4. Place LIMIT POST_ONLY order
5. Wait for fill (max 2 minutes)
6. If filled:
   - Record entry in portfolio
   - Place stop-loss order
   - Place take-profit order
   - Log entry with thesis
7. If not filled:
   - Cancel order
   - Skip (don't chase)
```

## Entry Price Logic

```python
def get_entry_price(side, orderbook):
    if side == Side.LONG:
        # Buy at best bid (maker, receive fee rebate)
        return orderbook.bids[0][0]
    else:
        # Sell at best ask (maker)
        return orderbook.asks[0][0]
```

---

# EXIT LOGIC

## Exit Triggers (Priority Order)

| Priority | Trigger | Action |
|----------|---------|--------|
| 1 | Kill switch activated | Immediate market exit |
| 2 | Layer 2 BLOCK | Immediate market exit |
| 3 | Liquidation distance < 5% | Immediate market exit |
| 4 | Thesis broken (health < 0.2) | Limit exit |
| 5 | Stop-loss hit | Stop order triggers |
| 6 | Take-profit hit | Limit exit |
| 7 | Max holding time exceeded | Limit exit |
| 8 | Thesis weakening (health < 0.5) | Partial exit (50%) |
| 9 | Regime change against position | Partial exit (50%) |
| 10 | High funding burden + weak thesis | Limit exit |

## Thesis Health Tracking

```python
def evaluate_thesis_health(position, market_state) -> float:
    """
    Returns 0.0 (thesis dead) to 1.0 (thesis intact)
    """
    thesis = position.metadata["thesis"]
    health = 1.0
    
    if "FUNDING_SQUEEZE" in thesis:
        funding = market_state.funding_rate.rate
        
        if position.side == Side.SHORT:
            # We shorted because longs were paying
            if funding < 0:  # Funding flipped
                health = 0.0  # Thesis broken
            elif funding < 0.0003:  # Funding normalized
                health = 0.5
        
        elif position.side == Side.LONG:
            # We longed because shorts were paying
            if funding > 0:  # Funding flipped
                health = 0.0
            elif funding > -0.0003:
                health = 0.5
    
    if "LIQUIDATION_REVERSAL" in thesis:
        pnl_pct = position.unrealized_pnl_pct
        if pnl_pct > 0.015:  # Got expected move
            health = 0.4  # Take profit zone
        elif pnl_pct < -0.01:  # Going against us
            health = 0.3
    
    if "CROWDING_FADE" in thesis:
        ls_ratio = get_long_short_ratio(position.symbol)
        if position.side == Side.SHORT and ls_ratio < 1.5:
            health = 0.5  # Crowd dispersed
        elif position.side == Side.LONG and ls_ratio > 0.7:
            health = 0.5
    
    return max(0.0, min(1.0, health))
```

## Stop-Loss Calculation

```python
def calculate_stop_loss(signal, price, stat_result):
    # Base: expected adverse excursion
    eae = signal.expected_adverse_excursion
    
    # Adjust for volatility
    if stat_result.volatility_regime == "high":
        eae *= 1.3
    elif stat_result.volatility_regime == "extreme":
        eae *= 1.5
    
    # Bounds: 1% minimum, 10% maximum
    eae = max(0.01, min(eae, 0.10))
    
    if signal.side == Side.LONG:
        return price * (1 - eae)
    else:
        return price * (1 + eae)
```

## Take-Profit Calculation

```python
def calculate_take_profit(signal, price, stop_loss):
    # Minimum 2:1 reward-to-risk ratio
    risk = abs(price - stop_loss)
    reward = risk * 2
    
    if signal.side == Side.LONG:
        return price + reward
    else:
        return price - reward
```

## Max Holding Time

```python
def calculate_max_holding_time(stat_result, signal_source):
    # Base times by signal type
    base_times = {
        "FUNDING_SQUEEZE": 4,     # Hours
        "LIQUIDATION_REVERSAL": 2,
        "OI_DIVERGENCE": 6,
        "CROWDING_FADE": 8,
        "FUNDING_CARRY": 24,
    }
    
    time = base_times.get(signal_source, 4)
    
    # Reduce in volatile regimes
    if stat_result.regime == Regime.HIGH_VOLATILITY:
        time *= 0.5
    elif stat_result.regime in [Regime.TRENDING_UP, Regime.TRENDING_DOWN]:
        time *= 1.5
    
    return time
```

## Exit Execution

```python
async def execute_exit(position, reason, fraction=1.0):
    """
    Exit position (full or partial).
    fraction: 1.0 = full exit, 0.5 = half exit
    """
    exit_size = position.size * fraction
    
    # Get exit price
    if position.side == Side.LONG:
        exit_price = orderbook.bids[0][0]  # Sell at bid
        order_side = "sell"
    else:
        exit_price = orderbook.asks[0][0]  # Buy at ask
        order_side = "buy"
    
    # Place reduce-only limit order
    order = await exchange.create_order(
        symbol=position.symbol,
        type="limit",
        side=order_side,
        amount=exit_size,
        price=exit_price,
        params={"reduceOnly": True}
    )
    
    # Wait for fill
    filled = await wait_for_fill(order, timeout=60)
    
    # If not filled, use market order
    if not filled:
        await exchange.cancel_order(order.id)
        order = await exchange.create_order(
            type="market",
            amount=exit_size,
            params={"reduceOnly": True}
        )
    
    # Log exit
    log_exit(position, reason, order)
    return order
```

---

# POSITION MANAGEMENT LOOP

## Every 30 Seconds Per Position

```python
async def manage_position(position, market_state, stat_result):
    # === EMERGENCY EXITS ===
    
    # Kill switch?
    if kill_switch_active:
        await execute_exit(position, "Kill switch", fraction=1.0)
        return
    
    # Layer 2 BLOCK?
    if stat_result.tradability == TradabilityStatus.BLOCK:
        await execute_exit(position, "Layer 2 BLOCK", fraction=1.0)
        return
    
    # Liquidation too close?
    if position.distance_to_liquidation < 0.05:
        await execute_exit(position, "Liquidation risk", fraction=1.0)
        return
    
    # === THESIS CHECK ===
    
    thesis_health = evaluate_thesis_health(position, market_state)
    
    if thesis_health < 0.2:
        await execute_exit(position, "Thesis broken", fraction=1.0)
        return
    
    if thesis_health < 0.5:
        await execute_exit(position, "Thesis weakening", fraction=0.5)
        # Continue holding remaining 50%
    
    # === TIME CHECK ===
    
    hours_held = (now - position.entry_time).total_seconds() / 3600
    if hours_held > position.max_holding_hours:
        await execute_exit(position, "Max time exceeded", fraction=1.0)
        return
    
    # === REGIME CHECK ===
    
    if position.side == Side.LONG and stat_result.regime == Regime.TRENDING_DOWN:
        await execute_exit(position, "Regime bearish", fraction=0.5)
    
    if position.side == Side.SHORT and stat_result.regime == Regime.TRENDING_UP:
        await execute_exit(position, "Regime bullish", fraction=0.5)
    
    # === FUNDING CHECK ===
    
    funding = market_state.funding_rate.rate
    
    if position.side == Side.LONG and funding > 0.001 and thesis_health < 0.7:
        await execute_exit(position, "High funding burden", fraction=1.0)
        return
    
    if position.side == Side.SHORT and funding < -0.001 and thesis_health < 0.7:
        await execute_exit(position, "High funding burden", fraction=1.0)
        return
    
    # === DEFAULT: HOLD ===
    log(f"Holding {position.symbol}: health={thesis_health:.2f}, pnl={position.unrealized_pnl_pct:.2%}")
```

---

# KILL SWITCHES

## Conditions That Trigger Immediate Flatten

```python
KILL_CONDITIONS = {
    "daily_drawdown": {
        "check": lambda p: p.current_drawdown > 0.05,
        "message": "Daily drawdown exceeded 5%"
    },
    "funding_spike": {
        "check": lambda s: abs(s.funding_rate.rate) > 0.005,
        "message": "Funding rate spike > 0.5%"
    },
    "cascade_risk": {
        "check": lambda r: r.cascade_probability > 0.7,
        "message": "Cascade probability > 70%"
    },
    "extreme_vol_regime_break": {
        "check": lambda r: r.regime_break_alert and r.volatility_regime == "extreme",
        "message": "Regime break with extreme volatility"
    },
    "liquidation_velocity": {
        "check": lambda r: r.liquidation_velocity > 10,
        "message": "Liquidation velocity explosion"
    },
}
```

## Kill Switch Behavior

1. **Trigger**: Any condition becomes true
2. **Action**: Close ALL positions immediately (market orders)
3. **Cooldown**: 1 hour before new trades allowed
4. **Reset**: Manual or automatic after cooldown

---

# CONFIGURATION REFERENCE

```python
# === RISK LIMITS ===
MAX_DAILY_DRAWDOWN_PCT = 5.0
MAX_HOURLY_DRAWDOWN_PCT = 3.0
MAX_POSITION_LOSS_PCT = 3.0
MIN_LIQUIDATION_DISTANCE_PCT = 5.0

# === POSITION LIMITS ===
MAX_POSITION_SIZE_USD = 10_000
MAX_TOTAL_EXPOSURE_USD = 50_000
MAX_POSITIONS = 5
MIN_POSITION_SIZE_USD = 100

# === LEVERAGE ===
BASE_LEVERAGE = 5.0
MAX_LEVERAGE = 20.0
LEVERAGE_DECAY_PER_SIGMA = 0.2

# === ENTRY THRESHOLDS ===
MIN_SIGNAL_CONFIDENCE = 0.50     # Minimum confidence from behavioral generator
ML_SCORE_THRESHOLD = 0.45        # Minimum ML score to approve signal
MAX_FUNDING_TO_ENTER = 0.0005    # 0.05%

# === SIZING ===
RISK_PER_TRADE_PCT = 1.0
KELLY_FRACTION = 0.25

# === TIMING ===
DECISION_INTERVAL_SECONDS = 60   # Main loop cycle
LLM_SCAN_INTERVAL_MINUTES = 30   # LLM news analysis
ORDER_TIMEOUT_SECONDS = 120
```

---

# DASHBOARD METRICS

The Streamlit dashboard displays:

## Top Row (4 metrics)
| Metric | Description |
|--------|-------------|
| Total Equity | Current portfolio value |
| Available Balance | Free margin for new trades |
| P&L | Profit/Loss since start |
| Trades | Number of executed trades |

## Pipeline Table (per pair)
| Column | Description |
|--------|-------------|
| Symbol | Trading pair |
| Price | Current price |
| L2 Regime | Detected market regime |
| L2 Trade | Tradability status |
| Cascade % | Cascade probability |
| L3 Signals | Number of signals generated |
| Best Signal | Top signal source + direction |
| Conf | Signal confidence |
| ML Score | ML model score (0-1) |
| L4 Size | Recommended position size |
| L4 Lev | Recommended leverage |
| Final | Final action code |

## Final Action Codes
| Code | Color | Meaning |
|------|-------|---------|
| LONG | Green | Long trade executed |
| SHORT | Green | Short trade executed |
| NO SIGNAL | Gray | No behavioral signal |
| ML REJECT | Red | ML score too low |
| LLM VETO | Red | LLM blocked trade |
| L4 VETO | Red | Risk brain blocked |
| BLOCKED | Red | Layer 2 blocked |
| NO MARGIN | Gray | Insufficient margin |
| HOLDING | Gray | Managing existing position |
| ERROR | Red | System error |

---

**Continue to HYDRA_SPEC_ML.md for ML model specifications →**
