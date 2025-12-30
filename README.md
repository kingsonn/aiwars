# HYDRA

## AI-Native Short-Term Crypto Perpetual Futures Trading System

HYDRA is not a price predictor. HYDRA is a **market-participant behavior engine**.

Designed to:
- 🎯 Exploit leverage imbalances
- 💥 Anticipate forced liquidations
- 📊 Monetize volatility and crowding
- 🛡️ Survive regime shifts
- 🤖 Out-adapt other AI systems

> In perpetual futures, **who is forced to act** matters more than where price "should" go.

---

## Architecture

HYDRA is a 5-layer system where **each layer can veto the next**:

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 5: EXECUTION                        │
│     Multi-Agent Vote → TWAP Execution → Order Management    │
├─────────────────────────────────────────────────────────────┤
│                    LAYER 4: RISK BRAIN                       │
│   Leverage Governance → Position Sizing → Kill Switches     │
├─────────────────────────────────────────────────────────────┤
│                    LAYER 3: ALPHA ENGINE                     │
│   Transformer → LLM Agent → Crowd Model → RL Execution      │
├─────────────────────────────────────────────────────────────┤
│                 LAYER 2: STATISTICAL REALITY                 │
│      GBM → Jump-Diffusion → Hawkes Process → Regime         │
├─────────────────────────────────────────────────────────────┤
│                  LAYER 1: MARKET INTEL                       │
│   Price → Funding → OI → Liquidations → Orderbook → Chain   │
└─────────────────────────────────────────────────────────────┘
```

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

Required API keys:
- **Exchange**: Binance or Bybit API keys
- **LLM**: OpenAI or Anthropic API key (for market structure agent)

### 3. Run HYDRA

```bash
# Paper trading (recommended first)
hydra run --mode paper

# Check system status
hydra status

# Run backtest
hydra backtest --symbol "BTC/USDT:USDT" --start 2024-01-01 --end 2024-06-01

# Train models
hydra train --component transformer --epochs 100
```

---

## Trading Philosophy

### What HYDRA Does

1. **Models Participants, Not Prices**
   - Identifies crowded trades via funding + OI
   - Detects trapped traders before liquidation cascades
   - Fades retail leverage while respecting smart money

2. **Exploits Leverage Dynamics**
   - Funding rate arbitrage awareness
   - Squeeze probability estimation
   - Liquidation cluster mapping

3. **Survives First, Profits Second**
   - Dynamic leverage based on volatility regime
   - Kill switches for tail events
   - Funding-aware position sizing

### What HYDRA Does NOT Do

- ❌ High-frequency trading
- ❌ Latency arbitrage
- ❌ Wash trading
- ❌ Spot market trading
- ❌ Predict exact prices

---

## Asset Universe

HYDRA trades only **top-liquidity perpetual futures**:

| Contract | Criteria |
|----------|----------|
| BTC-PERP | ✅ Top OI, deep books |
| ETH-PERP | ✅ High institutional participation |
| SOL-PERP | ✅ Stable funding history |
| BNB-PERP | ✅ Low manipulation risk |

**Excluded:**
- 🚫 Illiquid perps
- 🚫 New contracts (< 30 days)
- 🚫 Meme leverage traps

---

## Layer Details

### Layer 1: Market Intelligence

Data sources:
- **Price**: Multi-timeframe OHLCV (1m, 5m, 15m, 1h, 4h)
- **Funding**: Current rate, predicted rate, historical
- **Open Interest**: Absolute, delta, velocity
- **Liquidations**: Long vs short, volume, clustering
- **Order Book**: Imbalance, depth, spread
- **On-Chain**: Exchange flows, whale behavior (contextual)

### Layer 2: Statistical Reality

This layer defines **what is random vs meaningful**:

| Model | Purpose |
|-------|---------|
| Geometric Brownian Motion | Volatility envelope |
| Jump-Diffusion | Liquidation events |
| Hawkes Process | Cascade detection |

**Outputs**: Expected range, abnormal move score, jump probability, regime alerts

⚠️ This layer **never predicts direction** - it defines danger zones.

### Layer 3: Alpha & Behavior

This is where HYDRA earns money:

1. **Futures Transformer**
   - Trained on price + funding + OI + liquidations
   - Outputs: directional bias, adverse excursion, squeeze probability

2. **LLM Market Structure Agent**
   - Thinks like a derivatives desk
   - Answers: Who is trapped? Where are forced exits?

3. **Opponent Model**
   - Clusters trader archetypes (CTAs, funding farmers, retail)
   - Enables fade-the-crowd and pre-squeeze positioning

4. **Execution RL Agent**
   - Learns entry/exit timing
   - Optimizes for funding + slippage + drawdown

### Layer 4: Risk Brain

**Non-negotiable.** Features:

- Dynamic leverage caps (volatility-aware)
- Kelly-based position sizing
- Liquidation distance buffers
- Correlation exposure limits

**Kill Switches** (immediate flattening):
- Funding spikes beyond bounds
- Liquidation velocity explosion
- Model disagreement threshold
- Regime break detection

> HYDRA prefers **not trading** over dying.

### Layer 5: Decision & Execution

**Multi-Agent Voting:**
| Agent | Focus |
|-------|-------|
| Strategist (LLM) | Narrative & leverage logic |
| Quant (Deep/RL) | Statistical edge |
| Risk Manager | Survival check |
| Executor | Liquidity & slippage |

Trade executes **only if all approve**.

**Execution** (Non-HFT):
- Post-only limits
- Reduce-only exits
- TWAP scaling
- Slippage-aware sizing

---

## Benchmarks

HYDRA must outperform:
- Buy & hold futures
- Momentum perps
- Mean-reversion bots
- Static ML traders
- Single-LLM traders

Metrics:
- Absolute PnL
- Max drawdown
- CVaR (tail risk)
- Stability across regimes

---

## Project Structure

```
aiwars/
├── hydra/
│   ├── __init__.py
│   ├── cli.py                 # Command line interface
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   ├── engine.py          # Main orchestrator
│   │   └── types.py           # Type definitions
│   ├── layers/
│   │   ├── layer1_market_intel.py
│   │   ├── layer2_statistical.py
│   │   ├── layer3_alpha/
│   │   │   ├── engine.py
│   │   │   ├── transformer_model.py
│   │   │   ├── llm_agent.py
│   │   │   ├── opponent_model.py
│   │   │   └── rl_agent.py
│   │   ├── layer4_risk.py
│   │   └── layer5_execution.py
│   └── training/
│       ├── data_pipeline.py
│       ├── trainer.py
│       ├── backtester.py
│       └── simulator.py
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Commands

```bash
# Start trading
hydra run --mode [live|paper|backtest]

# Run backtest
hydra backtest --symbol BTC/USDT:USDT --start 2024-01-01 --end 2024-06-01

# Train models
hydra train --component [transformer|rl|all]

# Check status
hydra status

# Emergency kill
hydra kill

# Version
hydra version
```

---

## Configuration

Key settings in `.env`:

```env
# Exchange
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BINANCE_TESTNET=true

# LLM (for market structure agent)
ANTHROPIC_API_KEY=your_key
LLM_MODEL=claude-3-5-sonnet-20241022

# Trading
TRADING_MODE=paper
MAX_LEVERAGE=10
MAX_POSITION_SIZE_USD=10000

# Risk
RISK_PER_TRADE_PCT=1.0
```

---

## Disclaimer

⚠️ **WARNING**: Trading perpetual futures carries extreme risk. You can lose more than your initial investment.

- This software is for educational purposes
- Past performance does not guarantee future results
- Never trade with money you cannot afford to lose
- Always start with paper trading

---

## License

Proprietary. All rights reserved.

---

**HYDRA is not a strategy. HYDRA is an adaptive trading organism.**
