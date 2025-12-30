# HYDRA — Complete Algorithm Specification
## AI-Native Perpetual Futures Trading System

**Version:** 1.0 Final  
**Last Updated:** December 31, 2024  
**Target Audience:** Developers with no trading experience  

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
- Real-time market data analysis
- Statistical models to detect danger
- Machine learning for signal generation
- LLM for market structure analysis
- Strict risk management

## Core Philosophy

> **HYDRA does not predict price. HYDRA predicts WHO IS FORCED TO ACT.**

In leveraged markets, traders get **liquidated** when price moves against them. HYDRA's edge:
1. Detect when one side (longs or shorts) is trapped
2. Position to profit when they're forced to exit
3. Never be the trapped one

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

## 4.1 The 5 Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1: MARKET INTELLIGENCE              │
│         Raw data (price, funding, OI, liquidations)          │
│                         ↓                                    │
├─────────────────────────────────────────────────────────────┤
│                    LAYER 2: STATISTICAL REALITY              │
│         Safety gate: ALLOW / RESTRICT / BLOCK                │
│                         ↓                                    │
├─────────────────────────────────────────────────────────────┤
│                    LAYER 3: ALPHA GENERATION                 │
│         Signals: LONG / SHORT / FLAT + confidence            │
│                         ↓                                    │
├─────────────────────────────────────────────────────────────┤
│                    LAYER 4: RISK BRAIN                       │
│         Size, leverage, stops: APPROVE / VETO                │
│                         ↓                                    │
├─────────────────────────────────────────────────────────────┤
│                    LAYER 5: EXECUTION ENGINE                 │
│         Place orders on exchange                             │
└─────────────────────────────────────────────────────────────┘
```

## 4.2 Decision Flow (Every 30 Seconds)

```
1. Layer 1: Fetch data for all 8 pairs
2. For each pair:
   a. Layer 2: Safe to trade?
      - BLOCK → Skip entirely (close positions if any)
      - RESTRICT → Exits only
      - ALLOW → Continue
   
   b. If position exists → Manage it (hold/exit)
   
   c. If no position + ALLOW:
      - Layer 3: Generate signal
      - Layer 4: Size + approve/veto
      - Layer 5: Execute if approved
```

## 4.3 Veto Power

Any layer can stop a trade:
- Layer 2: BLOCK (dangerous conditions)
- Layer 4: VETO (risk limits exceeded)
- Layer 5: ABORT (execution issues)

---

# 5. QUICK REFERENCE

## 5.1 Key Thresholds

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Signal confidence threshold | 0.6 | Don't trade below 60% confidence |
| Max funding to enter | 0.05% | Don't enter if funding > 0.05% |
| Max daily drawdown | 5% | Stop trading if down 5% today |
| Max position size | $10,000 | Per position |
| Max total exposure | $50,000 | All positions combined |
| Max positions | 5 | Out of 8 pairs |
| Base leverage | 5x | Starting point |
| Max leverage | 20x | Hard cap |

## 5.2 Signal Types

| Signal | Logic | Direction |
|--------|-------|-----------|
| FUNDING_SQUEEZE | Extreme funding, one side bleeding | Against payers |
| LIQUIDATION_REVERSAL | After cascade, forced sellers exhausted | Counter to cascade |
| OI_DIVERGENCE | Price vs OI moving opposite | Against weak move |
| CROWDING_FADE | Everyone on same side | Against crowd |
| FUNDING_CARRY | Range market, collect funding | Receive funding |

## 5.3 Exit Triggers

1. **Thesis broken** - Original reason no longer valid
2. **Stop-loss hit** - Price hit stop level
3. **Take-profit hit** - Target reached
4. **Max time exceeded** - Held too long
5. **Liquidation close** - Within 5% of liquidation
6. **Regime change** - Market changed against position
7. **Layer 2 BLOCK** - Emergency exit

---

**Continue to HYDRA_SPEC_LAYERS.md for detailed layer specifications →**
