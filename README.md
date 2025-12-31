# HYDRA

## AI-Native Short-Term Crypto Perpetual Futures Trading System

HYDRA is not a price predictor. HYDRA is a **market-participant behavior engine**.

Designed to:
- 🎯 Exploit leverage imbalances and funding rate dynamics
- 💥 Anticipate forced liquidations before they cascade
- 📊 Monetize volatility, crowding, and market inefficiencies
- 🛡️ Survive regime shifts with multi-layer risk management
- 🤖 Combine ML signal scoring with LLM market analysis
- 🧠 Adapt through continuous learning and model retraining

> In perpetual futures, **who is forced to act** matters more than where price "should" go.

---

## System Overview

HYDRA combines **5 specialized layers**, **2 ML models**, and **LLM news analysis** into a cohesive trading system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYDRA TRADING SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  ML MODEL 1  │  │  ML MODEL 2  │  │  LLM NEWS    │  │   5-LAYER    │    │
│  │  SIGNAL      │  │  REGIME      │  │  ANALYST     │  │   PIPELINE   │    │
│  │  SCORER      │  │  CLASSIFIER  │  │              │  │              │    │
│  │              │  │              │  │ • 30min scan │  │ L1: Intel    │    │
│  │ • 49 feat    │  │ • 7 regimes  │  │ • Per-pair   │  │ L2: Stats+ML │    │
│  │ • CatBoost   │  │ • XGBoost    │  │ • News fetch │  │ L3: Alpha+ML │    │
│  │ • P(profit)  │  │ • Layer 2    │  │ • Trade gate │  │ L4: Risk     │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │ L5: Execute  │    │
│         │                 │                 │          └──────┬───────┘    │
│         └─────────────────┴─────────────────┴─────────────────┘            │
│                                   │                                         │
│                          ┌────────▼────────┐                                │
│                          │  TRADE DECISION │                                │
│                          │  All gates pass │                                │
│                          └─────────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### The 5-Layer Pipeline

Each layer can **veto** the next. A trade only executes if ALL layers approve:

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 5: EXECUTION                        │
│          Order Placement → Fill Management → Logging         │
├─────────────────────────────────────────────────────────────┤
│                    LAYER 4: RISK BRAIN                       │
│   Kelly Sizing → Leverage Calc → Kill Switches → Approve    │
├─────────────────────────────────────────────────────────────┤
│                    LAYER 3: ALPHA ENGINE                     │
│   Behavioral Signals → ML Scoring → LLM Gate → Best Signal  │
├─────────────────────────────────────────────────────────────┤
│                 LAYER 2: STATISTICAL REALITY                 │
│      Regime Detection → Volatility → Cascade Risk → Gate    │
├─────────────────────────────────────────────────────────────┤
│                  LAYER 1: MARKET INTEL                       │
│   Price → Funding → OI → Liquidations → Orderbook → News    │
└─────────────────────────────────────────────────────────────┘
```

### Decision Gates

| Gate | Layer | Blocks Trade If |
|------|-------|-----------------|
| **Data Health** | L1 | Missing or stale data |
| **Statistical** | L2 | BLOCK status (extreme vol, cascade) |
| **ML Score** | L3 | Score < 0.45 threshold |
| **LLM Analysis** | L3 | LLM recommends exit/hold |
| **Risk Approval** | L4 | Veto (limits exceeded, kill switch) |
| **Execution** | L5 | Order fails, timeout |

---

## Quick Start

### 1. Installation

```bash
# Clone and enter directory
cd aiwars

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install HYDRA
pip install -e .
```

### 2. Configuration

```bash
# Copy example config
copy .env.example .env

# Edit with your API keys
notepad .env
```

**Required API keys:**

| Service | Purpose | Required |
|---------|---------|----------|
| **Binance/Bybit** | Exchange trading | ✅ Yes |
| **Anthropic/OpenAI** | LLM news analysis | ⚡ Recommended |
| **CryptoCompare** | News data | ⚡ Recommended |

### 3. Run Dashboard

```bash
# Start the Streamlit dashboard (recommended)
streamlit run hydra/dashboard/app_v2.py --server.port 8502

# Or use CLI
hydra run --mode paper
```

### 4. Train ML Models

```bash
# Train both models (signal scorer + regime classifier)
python scripts/train_ml_models.py --days 90

# Train only signal scorer
python scripts/train_signal_scorer.py

# Train only regime classifier
python scripts/train_ml_models.py --regime-only

# Models saved to:
# - models/signal_scorer.pkl
# - models/regime_classifier.pkl
```

---

## Key Components

### 1. Behavioral Signal Generators

HYDRA generates signals from **market participant behavior**, not price prediction:

| Signal | Logic | Direction |
|--------|-------|-----------|
| **FUNDING_SQUEEZE** | Extreme funding bleeds one side → capitulation | Against payers |
| **LIQUIDATION_REVERSAL** | After cascade, forced sellers exhausted | Counter to cascade |
| **OI_DIVERGENCE** | Price vs OI moving opposite = weak move | Against weak move |
| **CROWDING_FADE** | Everyone on same side → fade them | Against crowd |
| **FUNDING_CARRY** | Range market, collect funding fees | Receive funding |

### 2. ML Signal Scorer (Model 1)

CatBoost model trained on historical signals to predict P(profitable):

**49 Features:**
- Signal features (9): direction, confidence, source encoding, expected return/risk
- Price features (10): returns, volatility, SMA ratios, RSI, ATR
- Funding features (4): rate, z-score, annualized, momentum
- OI features (3): delta, z-score, price divergence
- Liquidation features (3): imbalance, velocity, z-score
- Order book features (4): imbalance, spread, depth
- Positioning features (2): long/short ratio, taker ratio
- Regime features (9): regime encoding, volatility regime, cascade probability
- Time features (5): hour/day cyclical encoding, minutes to funding

**Training:**
```bash
python scripts/train_signal_scorer.py
# Uses historical data to generate signals and label profitability
# Cross-validated with time-series split
```

### 3. ML Regime Classifier (Model 2)

XGBoost multi-class classifier for market regime detection:

**7 Regime Classes:**
- TRENDING_UP - Clear upward trend
- TRENDING_DOWN - Clear downward trend
- RANGING - Sideways consolidation
- HIGH_VOLATILITY - Elevated volatility regime
- CASCADE_RISK - Liquidation cascade danger
- SQUEEZE_LONG - Longs getting squeezed
- SQUEEZE_SHORT - Shorts getting squeezed

**Features:**
- Trend indicators (ADX, SMA slopes, price vs SMAs)
- Volatility metrics (realized vol, ATR, Bollinger width)
- Funding & positioning (funding rate, OI delta, long/short ratio)
- Liquidation metrics (velocity, imbalance, cascade probability)
- Volume indicators (volume z-score, CVD momentum)

**Training:**
```bash
python scripts/train_ml_models.py --regime-only
# Trains regime classifier on historical market data
# Saved to models/regime_classifier.pkl
```

### 3. LLM News Analyst

Fetches crypto news and analyzes each trading pair every 30 minutes:

**Features:**
- Independent news scanning on 30-minute intervals
- Per-pair analysis with action recommendations
- Trade gating based on LLM sentiment
- Rate limiting to prevent API abuse

**Actions per pair:**
- `bullish` - Favor long trades
- `bearish` - Favor short trades  
- `hold` - Wait for clarity
- `exit` - Close existing positions

### 5. Risk Management

**Position Sizing:**
- Kelly criterion (quarter-Kelly for safety)
- Risk-based sizing (1% equity at risk per trade)
- Correlation penalties for similar positions
- Volatility-adjusted sizing per pair

**Kill Switches (immediate flatten):**
- Daily drawdown > 5%
- Funding spike > 0.5%
- Cascade probability > 70%
- Regime break + extreme volatility

---

## Asset Universe

HYDRA trades **8 perpetual futures contracts** on Binance:

| Pair | Volatility | Size Mult | Max Leverage | Notes |
|------|------------|-----------|--------------|-------|
| BTC/USDT | 1.0x (base) | 100% | 20x | Most stable, highest liquidity |
| ETH/USDT | 1.15x | 100% | 20x | Second most liquid |
| SOL/USDT | 1.50x | 80% | 15x | High volatility alt |
| BNB/USDT | 1.10x | 80% | 15x | Exchange token |
| ADA/USDT | 1.30x | 60% | 10x | Alt-L1, moderate risk |
| XRP/USDT | 1.25x | 70% | 10x | Legacy coin |
| LTC/USDT | 1.05x | 70% | 15x | BTC proxy |
| DOGE/USDT | 2.00x | 50% | 10x | High risk, meme |

---

## Dashboard

The Streamlit dashboard provides real-time monitoring:

**Metrics:**
- Total Equity, Available Balance, P&L, Trades

**Pipeline Table:**
| Symbol | Price | L2 Regime | Best Signal | Conf | ML Score | L4 Size | Final |
|--------|-------|-----------|-------------|------|----------|---------|-------|

**Tabs:**
- 📊 Dashboard - Overview and pipeline results
- 📝 Verbose Logs - Detailed per-cycle logging
- 💹 Trades - Trade history
- 🔬 Layer Details - Layer-by-layer breakdown

---

## Project Structure

```
aiwars/
├── hydra/
│   ├── __init__.py
│   ├── cli.py                    # Command line interface
│   ├── __main__.py               # Entry point
│   ├── core/
│   │   ├── config.py             # Configuration management
│   │   ├── engine.py             # Main orchestrator
│   │   └── types.py              # Type definitions (Signal, MarketState, etc.)
│   ├── layers/
│   │   ├── layer1_market_intel.py  # Data fetching
│   │   ├── layer2_statistical.py   # Regime detection, volatility
│   │   ├── layer3_alpha/
│   │   │   ├── engine.py           # Alpha orchestrator
│   │   │   ├── signals.py          # Behavioral signal generators
│   │   │   ├── transformer_model.py
│   │   │   ├── llm_agent.py
│   │   │   ├── opponent_model.py
│   │   │   └── rl_agent.py
│   │   ├── layer4_risk.py          # Risk brain, position sizing
│   │   ├── layer5_execution.py     # Order execution
│   │   ├── layer5_executor.py      # Executor implementation
│   │   ├── llm_analyst.py          # LLM news analyst
│   │   └── data_providers.py       # External data APIs
│   ├── dashboard/
│   │   └── app_v2.py               # Streamlit dashboard
│   ├── paper_trading/
│   │   ├── engine.py               # Paper trading engine
│   │   └── portfolio.py            # Portfolio management
│   └── training/
│       ├── signal_scorer_data.py   # Feature engineering for ML
│       ├── historical_data.py      # Historical data fetching
│       ├── data_pipeline.py        # Data processing
│       ├── trainer.py              # Model training
│       └── backtester.py           # Backtesting engine
├── scripts/
│   ├── train_signal_scorer.py      # Train ML model
│   ├── test_layer1.py              # Layer 1 tests
│   └── test_layer2.py              # Layer 2 tests
├── models/
│   └── signal_scorer.pkl           # Trained ML model
├── requirements.txt
├── pyproject.toml
├── .env.example
├── README.md
├── HYDRA_FINAL_SPEC.md             # Full specification
├── HYDRA_SPEC_LAYERS.md            # Layer details
├── HYDRA_SPEC_ML.md                # ML model specs
└── HYDRA_SPEC_TRADING.md           # Trading logic specs
```

---

## Configuration

Key settings in `.env`:

```env
# === EXCHANGE ===
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BINANCE_TESTNET=true

# === LLM (for news analysis) ===
ANTHROPIC_API_KEY=your_key
LLM_MODEL=claude-3-5-sonnet-20241022

# === NEWS DATA ===
CRYPTOCOMPARE_API_KEY=your_key

# === TRADING ===
TRADING_MODE=paper
INITIAL_BALANCE=10000

# === RISK LIMITS ===
MAX_LEVERAGE=10
MAX_POSITION_SIZE_USD=10000
MAX_TOTAL_EXPOSURE_USD=50000
MAX_POSITIONS=5
RISK_PER_TRADE_PCT=1.0

# === ML THRESHOLDS ===
ML_SCORE_THRESHOLD=0.45
MIN_SIGNAL_CONFIDENCE=0.50
```

---

## Trading Flow

```
Every 60 seconds:
├── For each of 8 pairs:
│   ├── L1: Fetch market data (price, funding, OI, liquidations)
│   ├── L2: Analyze regime, volatility, cascade risk
│   │   └── If BLOCK → Skip pair
│   ├── L3: Generate behavioral signals
│   │   ├── Score with ML model (49 features → P(profitable))
│   │   │   └── If ML score < 0.45 → Reject signal
│   │   └── Check LLM analysis
│   │       └── If LLM says exit/hold → Block trade
│   ├── L4: Calculate position size, leverage, stops
│   │   └── If veto (limits, kill switch) → Block trade
│   └── L5: Execute order if all gates pass
│
├── Every 30 minutes:
│   └── LLM: Scan news for all pairs, update analysis cache
│
└── Continuous:
    ├── Monitor open positions
    ├── Check thesis health
    └── Execute exits when triggered
```

---

## Documentation

| Document | Contents |
|----------|----------|
| **HYDRA_FINAL_SPEC.md** | Complete system overview, concepts, architecture |
| **HYDRA_SPEC_LAYERS.md** | Detailed Layer 1-5 specifications |
| **HYDRA_SPEC_ML.md** | ML models, features, training process |
| **HYDRA_SPEC_TRADING.md** | Position sizing, leverage, entry/exit logic |

---

## Performance Metrics

HYDRA tracks:
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns
- **ML Accuracy**: Signal scorer prediction accuracy

---

## Disclaimer

⚠️ **WARNING**: Trading perpetual futures carries extreme risk. You can lose more than your initial investment.

- This software is for educational and research purposes
- Past performance does not guarantee future results
- Never trade with money you cannot afford to lose
- Always start with paper trading
- The ML models require historical data and proper training

---

## License

Proprietary. All rights reserved.

---

**HYDRA is not a strategy. HYDRA is an adaptive trading organism that combines machine learning, market microstructure analysis, and risk management into a cohesive system.**
