"""
HYDRA Trading Dashboard V2 - New Layers Integration

Features:
- Run/Stop controls with balance input
- VERBOSE logging from all 5 layers
- Fresh data fetch each cycle (60 sec interval)
- Pipeline tables with detailed layer results
- Active positions tracking
- ML Signal Scorer integration
"""

import streamlit as st
import asyncio
import threading
import queue
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Page config
st.set_page_config(
    page_title="HYDRA Trading Dashboard V2",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = []
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'positions' not in st.session_state:
    st.session_state.positions = {}
if 'cycle_count' not in st.session_state:
    st.session_state.cycle_count = 0
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = None
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = queue.Queue()
if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue()
if 'layer_status' not in st.session_state:
    st.session_state.layer_status = {}
if 'equity' not in st.session_state:
    st.session_state.equity = 0
if 'pnl' not in st.session_state:
    st.session_state.pnl = 0


def log(msg: str, log_queue: queue.Queue, level: str = "INFO"):
    """Helper to log with timestamp."""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    log_queue.put({
        'time': timestamp,
        'level': level,
        'message': msg
    })


def run_hydra_layers(balance: float, stop_event: threading.Event, result_queue: queue.Queue, log_queue: queue.Queue):
    """Run HYDRA layers in a separate thread with asyncio - VERBOSE LOGGING."""
    
    async def run_layers():
        from hydra.core.config import HydraConfig, PERMITTED_PAIRS
        from hydra.core.types import Side, Regime
        from hydra.layers.layer1_market_intel import MarketIntelligenceLayer
        from hydra.layers.layer2_statistical import StatisticalRealityEngine, TradabilityStatus
        from hydra.layers.layer3_alpha.signals import BehavioralSignalGenerator
        from hydra.layers.layer4_risk import RiskCapitalBrain
        from hydra.layers.llm_analyst import LLMNewsAnalyst, get_llm_analyst
        from hydra.paper_trading.portfolio import Portfolio
        
        PAIR_DISPLAY = {
            "cmt_btcusdt": "BTC/USDT",
            "cmt_ethusdt": "ETH/USDT", 
            "cmt_solusdt": "SOL/USDT",
            "cmt_bnbusdt": "BNB/USDT",
            "cmt_adausdt": "ADA/USDT",
            "cmt_xrpusdt": "XRP/USDT",
            "cmt_ltcusdt": "LTC/USDT",
            "cmt_dogeusdt": "DOGE/USDT",
        }
        
        log("=" * 70, log_queue)
        log("🐍 HYDRA V2 INITIALIZATION - NEW LAYERS ARCHITECTURE", log_queue)
        log("=" * 70, log_queue)
        
        config = HydraConfig()
        
        # Initialize paper trading portfolio
        portfolio = Portfolio(initial_balance=balance, data_dir="./data/dashboard_v2")
        log(f"💰 Portfolio initialized: ${balance:.2f} USDT", log_queue)
        
        # Position theses for tracking
        theses = {}
        
        # Initialize Layer 1
        log("", log_queue)
        log("📡 [LAYER 1] Market Intelligence - Initializing...", log_queue)
        layer1 = MarketIntelligenceLayer()
        await layer1.initialize()
        log("  ✓ Layer 1 ready: OHLCV, Funding, OI, Order Book, Liquidations", log_queue)
        
        # Initialize Layer 2
        log("", log_queue)
        log("📊 [LAYER 2] Statistical Reality - Initializing...", log_queue)
        layer2 = StatisticalRealityEngine(config, use_ml_regime=True)
        await layer2.setup()
        ml_regime_status = "ML Regime Detector loaded" if layer2.ml_regime_detector else "Rule-based regime"
        log(f"  ✓ Layer 2 ready: {ml_regime_status}", log_queue)
        
        # Initialize Layer 3
        log("", log_queue)
        log("🎯 [LAYER 3] Alpha Generation - Initializing...", log_queue)
        layer3 = BehavioralSignalGenerator()
        ml_scorer_status = "ML Signal Scorer loaded" if layer3.ml_scorer else "No ML scorer (signals unfiltered)"
        log(f"  ✓ Layer 3 ready: {ml_scorer_status}", log_queue)
        log("  → Signal types: FUNDING_SQUEEZE, LIQUIDATION_REVERSAL, OI_DIVERGENCE, CROWDING_FADE, FUNDING_CARRY", log_queue)
        
        # Initialize Layer 4
        log("", log_queue)
        log("⚖️ [LAYER 4] Risk Brain - Initializing...", log_queue)
        layer4 = RiskCapitalBrain(config)
        await layer4.setup()
        log("  ✓ Layer 4 ready: Leverage, Position Sizing, Kill Switches", log_queue)
        
        # Initialize LLM News Analyst
        log("", log_queue)
        log("🧠 [LLM] News Analyst - Initializing...", log_queue)
        llm_analyst = get_llm_analyst()
        if llm_analyst.api_key:
            log("  ✓ LLM ready: Groq Qwen3-32B for news analysis", log_queue)
            log(f"  → Scan interval: {llm_analyst.scan_interval} minutes", log_queue)
            log("  → Running initial scan for all pairs...", log_queue)
            # Initial LLM scan on startup
            await llm_analyst.initialize()
            for sym, analysis in llm_analyst.get_all_analyses().items():
                log(f"    [{llm_analyst.PAIR_NAMES.get(sym, sym)[:3]}] {analysis.get_log_string()}", log_queue)
        else:
            log("  ⚠ LLM disabled: GROQ_API_KEY not set", log_queue, "WARN")
        
        log("", log_queue)
        log("=" * 70, log_queue)
        log("✅ ALL LAYERS INITIALIZED - Starting trading loop (60s interval)", log_queue)
        log("=" * 70, log_queue)
        
        cycle_count = 0
        
        while not stop_event.is_set():
            cycle_count += 1
            cycle_start = datetime.now(timezone.utc)
            
            log("", log_queue)
            log("═" * 70, log_queue)
            log(f"🔄 CYCLE {cycle_count} STARTED @ {cycle_start.strftime('%H:%M:%S')} UTC", log_queue)
            log("═" * 70, log_queue)
            
            cycle_results = []
            layer_details = []
            
            # LLM News Scan (runs independently every 30 minutes)
            if llm_analyst.api_key and llm_analyst.should_scan():
                log("", log_queue)
                log("🧠 [LLM] Running scheduled news scan...", log_queue)
                await llm_analyst.scan_all_pairs()
                for sym, analysis in llm_analyst.get_all_analyses().items():
                    log(f"  [{llm_analyst.PAIR_NAMES.get(sym, sym)[:3]}] {analysis.get_log_string()}", log_queue)
            
            # Log portfolio status
            log("", log_queue)
            log(f"💰 Portfolio: ${portfolio.total_equity:,.2f} | Available: ${portfolio.available_balance:,.2f} | Positions: {len([p for p in portfolio.positions.values() if p.is_open])}", log_queue)
            if not portfolio.can_trade:
                log("  ⚠ Insufficient balance for new trades - managing existing positions only", log_queue, "WARN")
            
            # Build positions dict for Layer 4 from portfolio
            active_positions: dict = {}
            for sym, pos_rec in portfolio.positions.items():
                if pos_rec.is_open:
                    from hydra.core.types import Position
                    active_positions[sym] = Position(
                        symbol=sym,
                        side=pos_rec.side,
                        size=pos_rec.size,
                        size_usd=pos_rec.size_usd,
                        entry_price=pos_rec.entry_price,
                        current_price=pos_rec.current_price,
                        leverage=pos_rec.leverage,
                        unrealized_pnl=pos_rec.unrealized_pnl,
                        unrealized_pnl_pct=pos_rec.unrealized_pnl_pct,
                        margin_used=pos_rec.margin_used,
                        entry_time=pos_rec.entry_time,
                    )
            
            for symbol in PERMITTED_PAIRS:  # Process all 8 permitted pairs
                display_name = PAIR_DISPLAY.get(symbol, symbol)
                short_name = display_name.split("/")[0]
                
                log("", log_queue)
                log(f"─── Processing {display_name} ───", log_queue)
                
                result = {
                    'symbol': symbol,
                    'display': short_name,
                    'price': 0.0,
                    'l1_status': '—',
                    'l2_regime': '—',
                    'l2_tradability': '—',
                    'l2_cascade_prob': 0.0,
                    'l3_signals': 0,
                    'l3_best_signal': '—',
                    'l3_confidence': 0.0,
                    'l3_ml_score': 0.0,
                    'llm_action': '—',
                    'llm_sentiment': '—',
                    'l4_approved': False,
                    'l4_size': 0.0,
                    'l4_leverage': 0.0,
                    'l4_stop': 0.0,
                    'l4_tp': 0.0,
                    'l5_status': '—',
                    'final_action': 'FLAT',
                    'error': None,
                }
                
                try:
                    # ═══════════════════════════════════════════════════════════════
                    # LAYER 1: MARKET INTELLIGENCE
                    # ═══════════════════════════════════════════════════════════════
                    log(f"  📡 [L1] Fetching market data...", log_queue)
                    
                    await layer1.refresh_symbol(symbol)
                    market_state = layer1.get_market_state(symbol)
                    
                    if market_state is None or market_state.price == 0:
                        log(f"  ❌ [L1] No market data available", log_queue, "ERROR")
                        result['l1_status'] = 'NO DATA'
                        result['final_action'] = 'NO DATA'
                        cycle_results.append(result)
                        continue
                    
                    result['price'] = market_state.price
                    result['l1_status'] = 'OK'
                    
                    # Log Layer 1 details
                    candles_5m = len(market_state.ohlcv.get("5m", []))
                    log(f"  [L1] Price: ${market_state.price:,.2f}", log_queue)
                    log(f"  [L1] OHLCV: {candles_5m} candles (5m)", log_queue)
                    
                    if market_state.funding_rate:
                        fr = market_state.funding_rate
                        log(f"  [L1] Funding Rate: {fr.rate*100:.4f}% ({fr.rate * 3 * 365 * 100:.1f}% ann)", log_queue)
                    
                    if market_state.open_interest:
                        oi = market_state.open_interest
                        log(f"  [L1] Open Interest: ${oi.open_interest_usd/1e9:.2f}B | Δ: {oi.delta_pct*100:+.2f}%", log_queue)
                    
                    if market_state.order_book:
                        ob = market_state.order_book
                        log(f"  [L1] Order Book: Imbalance {ob.imbalance:+.2f} | Spread {ob.spread*10000:.1f}bps", log_queue)
                    
                    if market_state.recent_liquidations:
                        total_liq = sum(l.usd_value for l in market_state.recent_liquidations)
                        log(f"  [L1] Recent Liquidations: ${total_liq/1e6:.2f}M", log_queue)
                    
                    # ═══════════════════════════════════════════════════════════════
                    # LAYER 2: STATISTICAL REALITY
                    # ═══════════════════════════════════════════════════════════════
                    log(f"  📊 [L2] Running statistical analysis...", log_queue)
                    
                    stat_result = await layer2.analyze(market_state)
                    
                    result['l2_regime'] = stat_result.regime.name
                    result['l2_tradability'] = stat_result.trading_decision.value
                    result['l2_cascade_prob'] = stat_result.cascade_probability
                    
                    # Log Layer 2 details
                    log(f"  [L2] Regime: {stat_result.regime.name} (confidence: {stat_result.regime_confidence:.1%})", log_queue)
                    log(f"  [L2] Volatility Regime: {stat_result.volatility_regime} (z-score: {stat_result.volatility_zscore:.2f})", log_queue)
                    log(f"  [L2] Cascade Probability: {stat_result.cascade_probability:.1%}", log_queue)
                    log(f"  [L2] Regime Break Alert: {stat_result.regime_break_alert}", log_queue)
                    log(f"  [L2] Trading Decision: {stat_result.trading_decision.value.upper()}", log_queue)
                    
                    # Check tradability gate
                    if stat_result.trading_decision == TradabilityStatus.BLOCK:
                        log(f"  ⛔ [L2] BLOCKED - Too dangerous to trade", log_queue, "WARN")
                        result['final_action'] = 'BLOCKED'
                        cycle_results.append(result)
                        continue
                    
                    # ═══════════════════════════════════════════════════════════════
                    # LAYER 3: ALPHA GENERATION
                    # ═══════════════════════════════════════════════════════════════
                    log(f"  🎯 [L3] Generating behavioral signals...", log_queue)
                    
                    signals = layer3.generate_signals(
                        market_state=market_state,
                        stat_result=stat_result,
                        long_short_ratio=1.0,  # Default L/S ratio
                    )
                    
                    result['l3_signals'] = len(signals)
                    
                    # Log Layer 3 details
                    log(f"  [L3] Signals generated: {len(signals)}", log_queue)
                    
                    if not signals:
                        log(f"  [L3] No signals - market conditions don't match patterns", log_queue)
                        result['final_action'] = 'NO SIGNAL'
                        cycle_results.append(result)
                        continue
                    
                    # Show each signal with ML score
                    for i, sig in enumerate(signals[:3]):
                        ml_score = sig.metadata.get('ml_score', 0)
                        ml_approved = sig.metadata.get('ml_approved', True)
                        ml_status = "✓" if ml_approved else "✗"
                        log(f"  [L3] Signal {i+1}: {sig.source} {sig.side.value} @ {sig.confidence:.1%} | ML: {ml_score:.2f} {ml_status}", log_queue)
                        if sig.metadata.get('thesis'):
                            log(f"       Thesis: {sig.metadata['thesis'][:60]}...", log_queue)
                    
                    # Get best signal (first one, sorted by ML approval then confidence)
                    best_signal = signals[0]
                    result['l3_best_signal'] = f"{best_signal.source} {best_signal.side.value}"
                    result['l3_confidence'] = best_signal.confidence
                    result['l3_ml_score'] = best_signal.metadata.get('ml_score', 0)
                    result['l3_ml_approved'] = best_signal.metadata.get('ml_approved', True)
                    
                    # Check if ML rejected the signal
                    if not best_signal.metadata.get('ml_approved', True):
                        ml_score = best_signal.metadata.get('ml_score', 0)
                        log(f"  ❌ [ML] Signal rejected: score {ml_score:.2f} < 0.45 threshold", log_queue)
                        result['final_action'] = 'ML REJECT'
                        cycle_results.append(result)
                        continue
                    
                    # Check position action (can we trade or just hold)
                    position_action, action_reason = portfolio.get_position_action(
                        symbol, best_signal.side, best_signal.confidence
                    )
                    
                    if position_action == "skip":
                        log(f"  ⚠ [PORTFOLIO] {action_reason}", log_queue, "WARN")
                        result['final_action'] = 'NO MARGIN'
                        cycle_results.append(result)
                        continue
                    elif position_action == "hold":
                        log(f"  ℹ [PORTFOLIO] {action_reason}", log_queue)
                        result['final_action'] = 'HOLDING'
                        cycle_results.append(result)
                        continue
                    
                    # LLM Check - Does LLM agree with this trade direction?
                    if llm_analyst.api_key:
                        llm_ok, llm_reason = llm_analyst.should_trade(symbol, best_signal.side.value)
                        pair_analysis = llm_analyst.get_pair_analysis(symbol)
                        
                        if pair_analysis:
                            result['llm_action'] = pair_analysis.action
                            result['llm_sentiment'] = pair_analysis.sentiment.value.replace('_', ' ').title()
                            log(f"  🧠 [LLM] {pair_analysis.get_log_string()}", log_queue)
                        else:
                            result['llm_action'] = 'N/A'
                            result['llm_sentiment'] = 'N/A'
                        
                        if not llm_ok:
                            log(f"  ❌ [LLM] Trade blocked: {llm_reason}", log_queue)
                            result['final_action'] = 'LLM VETO'
                            cycle_results.append(result)
                            continue
                        else:
                            log(f"  ✓ [LLM] {llm_reason}", log_queue)
                    else:
                        result['llm_action'] = 'Disabled'
                        result['llm_sentiment'] = 'Disabled'
                    
                    # ═══════════════════════════════════════════════════════════════
                    # LAYER 4: RISK BRAIN
                    # ═══════════════════════════════════════════════════════════════
                    log(f"  ⚖️ [L4] Evaluating risk...", log_queue)
                    
                    risk_decision = await layer4.evaluate(
                        signal=best_signal,
                        market_state=market_state,
                        stat_result=stat_result,
                        current_position=None,
                        all_positions=active_positions,
                        portfolio_equity=portfolio.total_equity,  # Pass actual equity
                    )
                    
                    result['l4_approved'] = risk_decision.approved
                    result['l4_size'] = risk_decision.recommended_size_usd
                    result['l4_leverage'] = risk_decision.recommended_leverage
                    result['l4_stop'] = risk_decision.stop_loss_price
                    result['l4_tp'] = risk_decision.take_profit_price
                    
                    # Log Layer 4 details
                    log(f"  [L4] Approved: {'✓' if risk_decision.approved else '✗'} | Veto: {risk_decision.veto}", log_queue)
                    if risk_decision.veto_reason:
                        log(f"  [L4] Veto Reason: {risk_decision.veto_reason}", log_queue)
                    log(f"  [L4] Position Size: ${risk_decision.recommended_size_usd:,.0f}", log_queue)
                    log(f"  [L4] Leverage: {risk_decision.recommended_leverage:.1f}x", log_queue)
                    log(f"  [L4] Stop Loss: ${risk_decision.stop_loss_price:,.2f}", log_queue)
                    log(f"  [L4] Take Profit: ${risk_decision.take_profit_price:,.2f}", log_queue)
                    log(f"  [L4] Risk Score: {risk_decision.risk_score:.2f}", log_queue)
                    log(f"  [L4] Max Hold: {risk_decision.max_holding_time_hours:.1f}h", log_queue)
                    
                    if risk_decision.trigger_kill_switch:
                        log(f"  🚨 [L4] KILL SWITCH TRIGGERED: {risk_decision.kill_reason}", log_queue, "CRITICAL")
                        result['final_action'] = 'KILL SWITCH'
                        cycle_results.append(result)
                        continue
                    
                    if not risk_decision.approved or risk_decision.veto:
                        log(f"  ❌ [L4] Trade rejected by Risk Brain", log_queue)
                        result['final_action'] = 'L4 VETO'
                        cycle_results.append(result)
                        continue
                    
                    # ═══════════════════════════════════════════════════════════════
                    # LAYER 5: EXECUTION (Paper Trading)
                    # ═══════════════════════════════════════════════════════════════
                    log(f"  🚀 [L5] Executing paper trade...", log_queue)
                    result['l5_status'] = 'PAPER'
                    
                    trade_record = portfolio.open_position(
                        symbol=symbol,
                        side=best_signal.side,
                        size_usd=risk_decision.recommended_size_usd,
                        price=market_state.price,
                        leverage=risk_decision.recommended_leverage,
                        signal_confidence=best_signal.confidence,
                        signal_source=best_signal.source,
                    )
                    
                    if trade_record:
                        log(f"  [L5] ✅ PAPER TRADE EXECUTED", log_queue)
                        log(f"  [L5] Size: ${risk_decision.recommended_size_usd:,.2f} @ {risk_decision.recommended_leverage:.1f}x", log_queue)
                        log(f"  [L5] Entry: ${market_state.price:,.2f}", log_queue)
                        log(f"  [L5] Stop: ${risk_decision.stop_loss_price:,.2f} | TP: ${risk_decision.take_profit_price:,.2f}", log_queue)
                        result['final_action'] = best_signal.side.value.upper()
                        
                        # Store thesis for position management
                        theses[symbol] = {
                            'source': best_signal.source,
                            'entry_time': datetime.now(timezone.utc),
                            'entry_price': market_state.price,
                            'stop_loss': risk_decision.stop_loss_price,
                            'take_profit': risk_decision.take_profit_price,
                            'max_hold_hours': risk_decision.max_holding_time_hours,
                        }
                        
                        # Record trade
                        result_queue.put({
                            'type': 'trade',
                            'data': {
                                'time': datetime.now().strftime('%H:%M:%S'),
                                'symbol': display_name,
                                'action': best_signal.side.value.upper(),
                                'price': market_state.price,
                                'size': risk_decision.recommended_size_usd,
                                'leverage': risk_decision.recommended_leverage,
                                'source': best_signal.source,
                                'ml_score': result['l3_ml_score'],
                                'stop': risk_decision.stop_loss_price,
                                'tp': risk_decision.take_profit_price,
                            }
                        })
                    else:
                        log(f"  [L5] ❌ Trade failed (insufficient balance or limits)", log_queue)
                        result['final_action'] = 'L5 FAILED'
                    
                except Exception as e:
                    log(f"  ❌ ERROR: {str(e)}", log_queue, "ERROR")
                    result['error'] = str(e)
                    result['final_action'] = 'ERROR'
                
                cycle_results.append(result)
            
            # Update portfolio prices and get positions
            positions_data = []
            for sym, pos in portfolio.positions.items():
                if pos.is_open:
                    # Get current price from last market state
                    positions_data.append({
                        'symbol': PAIR_DISPLAY.get(sym, sym),
                        'side': pos.side.value.upper(),
                        'size_usd': pos.size_usd,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'leverage': pos.leverage,
                        'pnl': pos.unrealized_pnl,
                        'pnl_pct': pos.unrealized_pnl_pct,
                        'margin': pos.margin_used,
                    })
            
            # Get LLM stats
            llm_stats = llm_analyst.get_stats() if llm_analyst.api_key else {}
            
            # Calculate average ML score for this cycle
            ml_scores = [r.get('l3_ml_score', 0) for r in cycle_results if r.get('l3_ml_score', 0) > 0]
            avg_ml_score = sum(ml_scores) / len(ml_scores) if ml_scores else 0
            
            # Send cycle results
            result_queue.put({
                'type': 'cycle',
                'data': {
                    'cycle': cycle_count,
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'results': cycle_results,
                    'equity': portfolio.total_equity,
                    'balance': portfolio.balance,
                    'available_balance': portfolio.available_balance,
                    'used_margin': portfolio.used_margin,
                    'pnl': portfolio.total_equity - balance,
                    'pnl_pct': (portfolio.total_equity - balance) / balance if balance > 0 else 0,
                    'positions': positions_data,
                    'num_positions': len(positions_data),
                    'drawdown': portfolio.current_drawdown,
                    'llm_stats': llm_stats,
                    'avg_ml_score': avg_ml_score,
                }
            })
            
            log("", log_queue)
            log(f"═══ CYCLE {cycle_count} COMPLETE ═══", log_queue)
            log(f"Processed {len(cycle_results)} symbols", log_queue)
            log(f"💰 Equity: ${portfolio.total_equity:,.2f} | Available: ${portfolio.available_balance:,.2f} | Used Margin: ${portfolio.used_margin:,.2f}", log_queue)
            if llm_analyst.api_key:
                mins_to_next = max(0, llm_analyst.scan_interval - int((datetime.now(timezone.utc) - llm_analyst._last_scan_time).total_seconds() / 60)) if llm_analyst._last_scan_time else 0
                log(f"🧠 LLM: {llm_analyst.total_calls} calls | {len(llm_analyst._pair_analysis)} pairs | Next scan: {mins_to_next} min", log_queue)
            log(f"Next cycle in 60 seconds...", log_queue)
            
            # Wait 60 seconds
            for i in range(60):
                if stop_event.is_set():
                    break
                await asyncio.sleep(1)
                if i % 15 == 14:
                    log(f"  ... {60 - i - 1}s until next cycle", log_queue)
        
        # Cleanup
        log("", log_queue)
        log("🛑 STOPPING HYDRA...", log_queue)
        await layer1.close()
        log("✓ HYDRA stopped", log_queue)
    
    asyncio.run(run_layers())


def main():
    # Sidebar
    with st.sidebar:
        st.title("🐍 HYDRA V2")
        st.caption("New Layers Architecture")
        st.markdown("---")
        
        balance = st.number_input(
            "Starting Balance (USDT)",
            min_value=100.0,
            max_value=1000000.0,
            value=1000.0,
            step=100.0,
            disabled=st.session_state.running
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Run", disabled=st.session_state.running, use_container_width=True):
                st.session_state.pipeline_results = []
                st.session_state.trades = []
                st.session_state.logs = []
                st.session_state.cycle_count = 0
                st.session_state.equity = balance
                st.session_state.pnl = 0
                
                st.session_state.stop_event = threading.Event()
                st.session_state.log_queue = queue.Queue()
                st.session_state.result_queue = queue.Queue()
                
                thread = threading.Thread(
                    target=run_hydra_layers,
                    args=(balance, st.session_state.stop_event, st.session_state.result_queue, st.session_state.log_queue),
                    daemon=True
                )
                thread.start()
                st.session_state.running = True
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop", disabled=not st.session_state.running, use_container_width=True):
                if st.session_state.stop_event:
                    st.session_state.stop_event.set()
                st.session_state.running = False
                st.rerun()
        
        if st.session_state.running:
            st.success("🟢 Running")
            st.metric("Cycles", st.session_state.cycle_count)
        else:
            st.info("🔴 Stopped")
        
        st.markdown("---")
        
        # Layer status
        st.subheader("Layer Status")
        layer_info = """
        - **L1**: Market Intel
        - **L2**: Statistical Reality  
        - **L3**: Alpha (ML Scorer)
        - **L4**: Risk Brain
        - **L5**: Execution
        """
        st.markdown(layer_info)
        
        st.markdown("---")
        st.caption("60s interval • Verbose logging")
    
    # Process queues
    if st.session_state.running:
        while not st.session_state.log_queue.empty():
            try:
                log_entry = st.session_state.log_queue.get_nowait()
                st.session_state.logs.append(log_entry)
                if len(st.session_state.logs) > 1000:
                    st.session_state.logs = st.session_state.logs[-1000:]
            except queue.Empty:
                break
        
        while not st.session_state.result_queue.empty():
            try:
                result = st.session_state.result_queue.get_nowait()
                if result['type'] == 'cycle':
                    st.session_state.pipeline_results.insert(0, result['data'])
                    st.session_state.cycle_count = result['data']['cycle']
                    st.session_state.equity = result['data']['equity']
                    st.session_state.available_balance = result['data'].get('available_balance', 0)
                    st.session_state.used_margin = result['data'].get('used_margin', 0)
                    st.session_state.pnl = result['data']['pnl']
                    st.session_state.llm_stats = result['data'].get('llm_stats', {})
                    st.session_state.avg_ml_score = result['data'].get('avg_ml_score', 0)
                elif result['type'] == 'trade':
                    st.session_state.trades.insert(0, result['data'])
            except queue.Empty:
                break
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📝 Verbose Logs", "💹 Trades", "🔬 Layer Details"])
    
    # Tab 1: Dashboard
    with tab1:
        st.header("HYDRA V2 Dashboard")
        
        # Portfolio summary - Keep only 4 metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Equity", f"${st.session_state.equity:,.2f}")
        with col2:
            avail = st.session_state.get('available_balance', st.session_state.equity)
            st.metric("Available Balance", f"${avail:,.2f}")
        with col3:
            pnl = st.session_state.pnl
            st.metric("P&L", f"${pnl:+,.2f}")
        with col4:
            st.metric("Trades", len(st.session_state.trades))
        
        st.markdown("---")
        
        # Pipeline Summary Table
        st.subheader("🔄 Pipeline Results")
        
        if st.session_state.pipeline_results:
            latest = st.session_state.pipeline_results[0]
            
            # Create summary table
            rows = []
            for r in latest['results']:
                rows.append({
                    'Symbol': r['display'],
                    'Price': f"${r['price']:,.2f}" if r['price'] > 0 else "—",
                    'L2 Regime': r['l2_regime'],
                    'L2 Trade': r['l2_tradability'],
                    'Best Signal': r['l3_best_signal'],
                    'Conf': f"{r['l3_confidence']:.0%}" if r['l3_confidence'] > 0 else "—",
                    'ML Score': f"{r['l3_ml_score']:.2f}" if r['l3_ml_score'] > 0 else "—",
                    'LLM': r.get('llm_action', '—'),
                    'Sentiment': r.get('llm_sentiment', '—'),
                    'L4 Size': f"${r['l4_size']:,.0f}" if r['l4_size'] > 0 else "—",
                    'L4 Lev': f"{r['l4_leverage']:.1f}x" if r['l4_leverage'] > 0 else "—",
                    'Final': r['final_action'],
                })
            
            df = pd.DataFrame(rows)
            
            # Style the dataframe
            def highlight_action(val):
                if val in ['LONG', 'SHORT']:
                    return 'background-color: #1a472a; color: white'
                elif val in ['BLOCKED', 'L4 VETO', 'ERROR', 'ML REJECT', 'LLM VETO']:
                    return 'background-color: #472a1a; color: white'
                return ''
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=250,
            )
            
            st.caption(f"Cycle {latest['cycle']} @ {latest['time']}")
        else:
            st.info("Click 'Run' to start HYDRA")
        
        st.markdown("---")
        
        # Layer Results Breakdown
        st.subheader("📈 Layer Flow Summary")
        
        if st.session_state.pipeline_results:
            latest = st.session_state.pipeline_results[0]
            results = latest['results']
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                l1_ok = sum(1 for r in results if r['l1_status'] == 'OK')
                st.metric("L1 Data OK", f"{l1_ok}/{len(results)}")
            
            with col2:
                l2_allow = sum(1 for r in results if r['l2_tradability'] == 'allow')
                st.metric("L2 Tradable", f"{l2_allow}/{len(results)}")
            
            with col3:
                l3_signals = sum(1 for r in results if r['l3_signals'] > 0)
                st.metric("L3 Signals", f"{l3_signals}/{len(results)}")
            
            with col4:
                l4_approved = sum(1 for r in results if r['l4_approved'])
                st.metric("L4 Approved", f"{l4_approved}/{len(results)}")
            
            with col5:
                executed = sum(1 for r in results if r['final_action'] in ['LONG', 'SHORT'])
                st.metric("Executed", f"{executed}/{len(results)}")
    
    # Tab 2: Verbose Logs
    with tab2:
        st.header("📝 Verbose System Logs")
        st.caption("Detailed logs from all 5 layers")
        
        logs = st.session_state.logs
        if logs:
            col1, col2 = st.columns([1, 4])
            with col1:
                show_all = st.checkbox("Show all levels", value=True)
            
            if show_all:
                display_logs = logs
            else:
                display_logs = [l for l in logs if l.get('level', 'INFO') in ['INFO', 'WARN', 'ERROR', 'CRITICAL']]
            
            # Format logs
            log_lines = []
            for log_entry in display_logs[-300:]:
                level = log_entry.get('level', 'INFO')
                level_colors = {
                    'INFO': '',
                    'WARN': '⚠️ ',
                    'ERROR': '❌ ',
                    'CRITICAL': '🚨 ',
                }
                prefix = level_colors.get(level, '')
                log_lines.append(f"[{log_entry['time']}] {prefix}{log_entry['message']}")
            
            log_text = "\n".join(log_lines)
            st.code(log_text, language="")
        else:
            st.info("No logs yet - click 'Run' to start")
    
    # Tab 3: Trades
    with tab3:
        st.header("💹 Trade History")
        
        trades = st.session_state.trades
        if trades:
            trades_df = pd.DataFrame(trades)
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades executed yet")
    
    # Tab 4: Layer Details
    with tab4:
        st.header("🔬 Layer Architecture Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Layer 1: Market Intelligence")
            st.markdown("""
            **Data Sources:**
            - OHLCV candles (5m, 15m, 1h)
            - Funding rates (8h)
            - Open Interest (contracts + USD)
            - Order book (bids/asks depth)
            - Recent liquidations
            
            **Output:** `MarketState` object
            """)
            
            st.subheader("Layer 2: Statistical Reality")
            st.markdown("""
            **Analysis:**
            - Regime classification (ML or rule-based)
            - Volatility regime (normal/high/extreme)
            - Cascade probability
            - Regime break detection
            
            **Output:** `StatisticalResult` with trading decision
            """)
        
        with col2:
            st.subheader("Layer 3: Alpha Generation")
            st.markdown("""
            **Signal Types:**
            - FUNDING_SQUEEZE
            - LIQUIDATION_REVERSAL
            - OI_DIVERGENCE
            - CROWDING_FADE
            - FUNDING_CARRY
            
            **ML Filtering:** 49-feature Signal Scorer model
            """)
            
            st.subheader("Layer 4: Risk Brain")
            st.markdown("""
            **Decisions:**
            - Position sizing (Kelly)
            - Leverage (1-5x)
            - Stop-loss / Take-profit
            - Kill switch triggers
            
            **Output:** `RiskDecision` (approve/veto)
            """)
        
        st.subheader("Layer 5: Execution Engine")
        st.markdown("""
        **Execution Strategy:**
        - Entry: Limit post-only orders
        - Exit: Reduce-only with market fallback
        - Paper trading simulation mode
        
        **Output:** `ExecutionResult` with fill details
        """)
    
    # Auto-refresh when running
    if st.session_state.running:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
