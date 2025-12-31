# HYDRA — Complete Algorithm Specification
## AI-Native Perpetual Futures Trading System

**Version:** 2.0  
**Last Updated:** December 31, 2024  
**Target Audience:** Developers and traders seeking to understand HYDRA's architecture  

---

# TABLE OF CONTENTS

This specification is split into parts due to length:

1. **HYDRA_FINAL_SPEC.md** (this file) - Overview, Concepts, Architecture
2. **HYDRA_SPEC_LAYERS.md** - Detailed Layer specifications (1-5)
3. **HYDRA_SPEC_TRADING.md** - Position sizing, leverage, entry/exit logic
4. **HYDRA_SPEC_ML.md** - ML models, training data, training process

---

# 1. EXECUTIVE SUMMARY

## What is HYDRA?

HYDRA is an automated trading system for **perpetual futures** on 8 cryptocurrency pairs with up to 20x leverage. It combines:

- **5-Layer Pipeline**: Data → Statistics → Alpha → Risk → Execution
- **ML Model 1 - Signal Scorer**: CatBoost with 49 features predicting P(profitable)
- **ML Model 2 - Regime Classifier**: XGBoost detecting 7 market regimes
- **LLM News Analyst**: Claude-powered per-pair news analysis every 30 minutes
- **Behavioral Signals**: 5 signal types based on market participant behavior
- **Multi-Gate Risk**: Every trade must pass ML, LLM, and risk gates

## Core Philosophy

> **HYDRA does not predict price. HYDRA predicts WHO IS FORCED TO ACT.**

In leveraged markets, traders get **liquidated** when price moves against them. HYDRA's edge:
1. Detect when one side (longs or shorts) is trapped via funding + OI analysis
2. Score signals with ML to filter low-probability setups
3. Gate trades with LLM news analysis for market context
4. Position to profit when trapped traders are forced to exit
5. Never be the trapped one through strict risk management

---

# 2. CORE CONCEPTS FOR NON-TRADERS

## 2.1 Perpetual Futures (Perps)

A contract to bet on price going up (LONG) or down (SHORT) with borrowed money.

**Key features:**
- You don't own the cryptocurrency
- Use leverage (borrowed money)
- Pay/receive **funding fees** every 8 hours
- Can get **liquidated** if price moves too much against you

## 2.2 Leverage Explained

| Your Money | Leverage | Position Size | 5% Price Move |
|------------|----------|---------------|---------------|
| $1,000 | 1x | $1,000 | ±$50 (±5%) |
| $1,000 | 5x | $5,000 | ±$250 (±25%) |
| $1,000 | 10x | $10,000 | ±$500 (±50%) |
| $1,000 | 20x | $20,000 | ±$1,000 (±100%) |

**Rule: At 20x leverage, a 5% move against you = 100% loss = LIQUIDATED**

## 2.3 Funding Rate

Fee paid between traders every 8 hours:
- **Positive funding** → Longs pay shorts
- **Negative funding** → Shorts pay longs

**Example:** 0.1% funding, $10,000 position = $10 every 8h = $30/day = $210/week

**Why it matters:** Extreme funding = one side is bleeding money = unsustainable = reversal likely

## 2.4 Open Interest (OI)

Total value of all open positions.

| OI Change | Price Change | Meaning |
|-----------|--------------|---------|
| ↑ Rising | ↑ Rising | New longs entering (bullish) |
| ↑ Rising | ↓ Falling | New shorts entering (bearish) |
| ↓ Falling | ↑ Rising | Shorts closing (squeeze) |
| ↓ Falling | ↓ Falling | Longs closing (capitulation) |

## 2.5 Liquidation

Exchange force-closes your position when you can't cover losses.

**Why it matters:** Liquidations are forced selling/buying → violent price moves → if we know where they cluster, we can profit.

---

# 3. FIXED TRADING UNIVERSE

## 3.1 The 8 Pairs (NEVER CHANGE)

```python
PERMITTED_PAIRS = [
    "cmt_btcusdt",   # Bitcoin
    "cmt_ethusdt",   # Ethereum  
    "cmt_solusdt",   # Solana
    "cmt_bnbusdt",   # Binance Coin
    "cmt_adausdt",   # Cardano
    "cmt_xrpusdt",   # Ripple
    "cmt_ltcusdt",   # Litecoin
    "cmt_dogeusdt",  # Dogecoin
]
```

## 3.2 Pair Characteristics

| Pair | Volatility | BTC Correlation | Max Size Mult | Notes |
|------|------------|-----------------|---------------|-------|
| BTC | 1.0x (base) | 1.00 | 100% | Most stable |
| ETH | 1.15x | 0.85 | 100% | Second most stable |
| SOL | 1.50x | 0.75 | 80% | High volatility |
| BNB | 1.10x | 0.78 | 80% | Exchange token |
| ADA | 1.30x | 0.72 | 60% | Alt-L1 |
| XRP | 1.25x | 0.70 | 70% | Legacy coin |
| LTC | 1.05x | 0.88 | 70% | BTC proxy |
| DOGE | 2.00x | 0.65 | 50% | Meme, dangerous |

---

# 4. SYSTEM ARCHITECTURE

## 4.1 The Complete System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYDRA TRADING SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐            │
│  │   LLM NEWS      │   │   ML SIGNAL     │   │   5-LAYER       │            │
│  │   ANALYST       │   │   SCORER        │   │   PIPELINE      │            │
│  │                 │   │                 │   │                 │            │
│  │ • Every 30 min  │   │ • 49 features   │   │ L1: Market Data │            │
│  │ • Per-pair scan │   │ • XGBoost model │   │ L2: Statistics  │            │
│  │ • News fetch    │   │ • P(profitable) │   │ L3: Alpha Gen   │            │
│  │ • Action: bull/ │   │ • Threshold:    │   │ L4: Risk Brain  │            │
│  │   bear/hold/exit│   │   0.45          │   │ L5: Execution   │            │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘            │
│           │                     │                     │                      │
│           └─────────────────────┼─────────────────────┘                      │
│                                 │                                            │
│                        ┌────────▼────────┐                                   │
│                        │  TRADE DECISION │                                   │
│                        │  All gates must │                                   │
│                        │  approve        │                                   │
│                        └─────────────────┘                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 4.2 The 5 Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 5: EXECUTION ENGINE                 │
│         Place orders → Monitor fills → Log trades            │
├─────────────────────────────────────────────────────────────┤
│                    LAYER 4: RISK BRAIN                       │
│   Kelly sizing → Leverage calc → Kill switches → APPROVE/VETO│
├─────────────────────────────────────────────────────────────┤
│                    LAYER 3: ALPHA GENERATION                 │
│   Behavioral signals → ML score → LLM gate → Best signal     │
├─────────────────────────────────────────────────────────────┤
│                 LAYER 2: STATISTICAL REALITY                 │
│   Regime detection → Volatility → Cascade risk → ALLOW/BLOCK │
├─────────────────────────────────────────────────────────────┤
│                  LAYER 1: MARKET INTELLIGENCE                │
│   Price → Funding → OI → Liquidations → Order book → News    │
└─────────────────────────────────────────────────────────────┘
```

## 4.3 Decision Flow (Every 60 Seconds)

```
MAIN LOOP (every 60 seconds):
│
├── For each of 8 pairs:
│   │
│   ├── LAYER 1: Fetch market data
│   │   └── Price, funding, OI, liquidations, order book
│   │
│   ├── LAYER 2: Analyze safety
│   │   ├── Regime: TRENDING_UP/DOWN, RANGING, HIGH_VOL, CASCADE_RISK
│   │   ├── Volatility regime: normal/high/extreme
│   │   ├── Cascade probability
│   │   └── Decision: ALLOW / RESTRICT / BLOCK
│   │       └── If BLOCK → Skip pair, exit if holding
│   │
│   ├── LAYER 3: Generate signals
│   │   ├── Run 5 behavioral generators:
│   │   │   • FUNDING_SQUEEZE
│   │   │   • LIQUIDATION_REVERSAL
│   │   │   • OI_DIVERGENCE
│   │   │   • CROWDING_FADE
│   │   │   • FUNDING_CARRY
│   │   │
│   │   ├── ML SCORING (for each signal):
│   │   │   ├── Extract 49 features
│   │   │   ├── Predict P(profitable)
│   │   │   └── If score < 0.45 → ML REJECT
│   │   │
│   │   └── LLM CHECK:
│   │       ├── Get cached per-pair analysis
│   │       └── If action = exit/hold → LLM VETO
│   │
│   ├── LAYER 4: Risk evaluation
│   │   ├── Calculate position size (Kelly + risk-based)
│   │   ├── Calculate leverage (volatility-adjusted)
│   │   ├── Check kill switches
│   │   └── Decision: APPROVE / VETO
│   │
│   └── LAYER 5: Execute (if all approved)
│       ├── Place limit order
│       ├── Set stop-loss
│       ├── Set take-profit
│       └── Log trade
│
└── LLM SCAN (every 30 minutes):
    ├── Fetch news for all crypto topics
    ├── Analyze each pair with LLM
    └── Cache per-pair recommendations
```

## 4.4 Decision Gates

Every trade must pass **6 gates**:

| Gate | Component | Blocks If |
|------|-----------|-----------|
| **Data Health** | L1 | Missing or stale data |
| **Statistical** | L2 | BLOCK status (extreme vol, cascade) |
| **ML Score** | L3 | Score < 0.45 |
| **LLM Analysis** | L3 | LLM recommends exit or hold |
| **Risk Approval** | L4 | Limits exceeded, kill switch active |
| **Execution** | L5 | Order fails, timeout |

## 4.5 Veto Power

Any component can stop a trade:
- **Layer 2**: BLOCK (dangerous conditions)
- **ML Scorer**: REJECT (low probability signal)
- **LLM Analyst**: VETO (news-based concern)
- **Layer 4**: VETO (risk limits exceeded)
- **Layer 5**: ABORT (execution issues)

---

# 5. QUICK REFERENCE

## 5.1 Key Thresholds

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Signal confidence threshold | 0.50 | Minimum confidence to consider signal |
| ML score threshold | 0.45 | ML model must score above this |
| Max funding to enter | 0.05% | Don't enter if funding > 0.05% |
| Max daily drawdown | 5% | Stop trading if down 5% today |
| Max position size | $10,000 | Per position |
| Max total exposure | $50,000 | All positions combined |
| Max positions | 5 | Out of 8 pairs |
| Base leverage | 5x | Starting point |
| Max leverage | 20x | Hard cap |
| LLM scan interval | 30 min | News analysis frequency |

## 5.2 Signal Types

| Signal | Logic | Direction |
|--------|-------|-----------|
| FUNDING_SQUEEZE | Extreme funding, one side bleeding | Against payers |
| LIQUIDATION_REVERSAL | After cascade, forced sellers exhausted | Counter to cascade |
| OI_DIVERGENCE | Price vs OI moving opposite | Against weak move |
| CROWDING_FADE | Everyone on same side | Against crowd |
| FUNDING_CARRY | Range market, collect funding | Receive funding |

## 5.3 ML Model 1: Signal Scorer Features (49 total)

| Category | Count | Features |
|----------|-------|----------|
| Signal | 9 | direction, confidence, 5 source one-hots, expected return/risk |
| Price | 10 | returns (1m/5m/15m/1h), volatility, SMA ratios, RSI, ATR |
| Funding | 4 | rate, z-score, annualized, momentum |
| OI | 3 | delta_pct, z-score, price_divergence |
| Liquidation | 3 | imbalance, velocity, z-score |
| Order Book | 4 | imbalance, spread, bid/ask depth |
| Positioning | 2 | long_short_ratio, taker_buy_sell_ratio |
| Regime | 9 | 7 regime one-hots, volatility_regime, cascade_probability |
| Time | 5 | hour/day cyclical encoding, minutes_to_funding |

**Model:** CatBoost Classifier  
**Threshold:** 0.45  
**File:** `models/signal_scorer.pkl`

## 5.4 ML Model 2: Regime Classifier (7 classes)

| Regime | Description | Strategy Impact |
|--------|-------------|----------------|
| TRENDING_UP | Clear upward trend | Favor longs, reduce shorts |
| TRENDING_DOWN | Clear downward trend | Favor shorts, reduce longs |
| RANGING | Sideways consolidation | Mean reversion strategies |
| HIGH_VOLATILITY | Elevated volatility | Reduce leverage, wider stops |
| CASCADE_RISK | Liquidation cascade danger | Reduce exposure, exit risky positions |
| SQUEEZE_LONG | Longs getting squeezed | Favor shorts, avoid longs |
| SQUEEZE_SHORT | Shorts getting squeezed | Favor longs, avoid shorts |

**Model:** XGBoost Multi-class Classifier  
**Used by:** Layer 2 for regime detection  
**File:** `models/regime_classifier.pkl`

## 5.5 LLM News Analyst Actions

| Action | Meaning | Trade Impact |
|--------|---------|--------------|
| `bullish` | Positive sentiment for pair | Allow LONG trades |
| `bearish` | Negative sentiment for pair | Allow SHORT trades |
| `hold` | Unclear, wait for clarity | Block new entries |
| `exit` | Negative news, close positions | Block entries, signal exit |

## 5.6 Exit Triggers

1. **ML Reject** - Signal score below threshold
2. **LLM Veto** - News analysis recommends exit/hold
3. **Thesis broken** - Original reason no longer valid
4. **Stop-loss hit** - Price hit stop level
5. **Take-profit hit** - Target reached
6. **Max time exceeded** - Held too long
7. **Liquidation close** - Within 5% of liquidation
8. **Regime change** - Market changed against position
9. **Layer 2 BLOCK** - Emergency exit
10. **Kill switch** - Portfolio-level risk trigger

## 5.7 Final Action Codes

| Code | Meaning |
|------|---------|
| LONG / SHORT | Trade executed |
| NO SIGNAL | No behavioral signal generated |
| ML REJECT | Signal rejected by ML scorer |
| LLM VETO | Signal blocked by LLM analysis |
| L4 VETO | Risk brain rejected trade |
| BLOCKED | Layer 2 blocked trading |
| NO MARGIN | Insufficient margin available |
| HOLDING | Already in position, holding |
| ERROR | System error occurred |

---

**Continue to HYDRA_SPEC_LAYERS.md for detailed layer specifications →**
