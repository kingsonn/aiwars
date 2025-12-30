"""
HYDRA Trading Dashboard - Verbose Logging Edition

Features:
- Run/Stop controls with balance input
- VERBOSE logging showing every step, calculation, LLM prompts/responses
- Fresh data fetch each cycle (60 sec interval)
- Pipeline tables with pagination (latest 10)
- Active positions with real-time value calculation
- Trades tab
"""

import streamlit as st
import asyncio
import threading
import queue
import time
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
from loguru import logger
import sys

# Page config
st.set_page_config(
    page_title="HYDRA Trading Dashboard",
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
    st.session_state.positions = []
if 'cycle_count' not in st.session_state:
    st.session_state.cycle_count = 0
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = None
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = queue.Queue()
if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue()


def log(msg: str, log_queue: queue.Queue, level: str = "INFO"):
    """Helper to log with timestamp."""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    log_queue.put({
        'time': timestamp,
        'level': level,
        'message': msg
    })


def run_hydra_async(balance: float, stop_event: threading.Event, result_queue: queue.Queue, log_queue: queue.Queue):
    """Run HYDRA in a separate thread with asyncio - VERBOSE LOGGING."""
    
    async def run_hydra():
        from hydra.core.config import HydraConfig, PAIR_DISPLAY_NAMES
        from hydra.core.types import Side, Signal, PositionAction
        from hydra.paper_trading.engine import PaperTradingEngine
        from hydra.core.position_manager import OperatingMode
        from hydra.layers.layer2_statistical import TradingDecision
        from hydra.layers.layer3_alpha.llm_agent import get_last_llm_interaction
        
        log("=" * 60, log_queue)
        log("HYDRA INITIALIZATION STARTING", log_queue)
        log("=" * 60, log_queue)
        
        config = HydraConfig.load()
        log(f"Config loaded: {len(config.trading.symbols)} symbols", log_queue)
        log(f"Symbols: {config.trading.symbols}", log_queue)
        log(f"LLM Provider: {config.llm.llm_provider} | Model: {config.llm.llm_model}", log_queue)
        
        engine = PaperTradingEngine(
            config=config,
            initial_balance=balance,
            data_dir="./data/dashboard_trading",
        )
        log(f"Paper trading engine created with ${balance:.2f} balance", log_queue)
        
        # Initial setup - this loads historical data
        log("Setting up HYDRA layers...", log_queue)
        log("  → Layer 1: Market Intelligence (loading OHLCV, funding, OI, orderbook)...", log_queue)
        await engine.hydra.setup()
        log("  → Layer 2: Statistical Reality (ready)", log_queue)
        log("  → Layer 3: Alpha & Behavior (LLM + Transformer ready)", log_queue)
        log("  → Layer 4: Risk & Capital (ready)", log_queue)
        log("  → Layer 5: Decision & Execution (ready)", log_queue)
        log("✓ All layers initialized", log_queue)
        
        cycle_count = 0
        
        while not stop_event.is_set():
            cycle_count += 1
            cycle_start = datetime.now(timezone.utc)
            
            log("", log_queue)
            log("=" * 60, log_queue)
            log(f"CYCLE {cycle_count} STARTED @ {cycle_start.strftime('%H:%M:%S')}", log_queue)
            log("=" * 60, log_queue)
            
            # REFRESH DATA EACH CYCLE - Critical for real-time trading
            log("", log_queue)
            log("📡 REFRESHING MARKET DATA...", log_queue)
            log("  → Fetching fresh OHLCV, funding, OI, orderbook, sentiment...", log_queue)
            await engine.hydra.layer1.refresh_all_data()
            log("✓ All market data refreshed", log_queue)
            
            cycle_results = []
            
            for symbol in config.trading.symbols:
                display_name = PAIR_DISPLAY_NAMES.get(symbol, symbol)
                short_name = display_name.split("/")[0]
                
                log("", log_queue)
                log(f"─── Processing {short_name} ───", log_queue)
                
                result = {
                    'symbol': symbol,
                    'display': short_name,
                    'mode': 'ENTRY',
                    'l2_decision': None,
                    'l3_signal': None,
                    'l3_confidence': 0.0,
                    'l4_approved': False,
                    'l4_size': 0.0,
                    'l4_leverage': 0.0,
                    'l5_approved': False,
                    'final_action': 'FLAT',
                    'pnl_pct': 0.0,
                    'current_price': 0.0,
                }
                
                try:
                    # Get current position
                    position = engine._get_position_as_type(symbol)
                    mode = engine.position_manager.get_operating_mode(symbol, position)
                    result['mode'] = mode.value.upper()
                    log(f"  Mode: {result['mode']} {'(has position)' if position else '(no position)'}", log_queue)
                    
                    if position:
                        result['pnl_pct'] = position.unrealized_pnl_pct
                        log(f"  Position: {position.side.value.upper()} ${position.size_usd:.0f} @ {position.leverage:.1f}x, P&L: {result['pnl_pct']:+.2%}", log_queue)
                    
                    # Get market state
                    log(f"  [L1] Getting market state...", log_queue)
                    market_state = await engine.hydra.layer1.get_market_state(symbol)
                    if not market_state:
                        log(f"  ❌ No market data available", log_queue, "ERROR")
                        result['final_action'] = 'NO DATA'
                        cycle_results.append(result)
                        continue
                    
                    result['current_price'] = market_state.price
                    engine._current_prices[symbol] = market_state.price
                    
                    # Log market state details
                    log(f"  [L1] Price: ${market_state.price:,.2f} | 24h: {market_state.price_change_24h*100:+.2f}%", log_queue)
                    if market_state.funding_rate:
                        fr = market_state.funding_rate
                        log(f"  [L1] Funding: {fr.rate*100:.4f}% ({fr.annualized:.1f}% ann)", log_queue)
                    if market_state.open_interest:
                        oi = market_state.open_interest
                        log(f"  [L1] OI: {oi.open_interest:,.0f} contracts = ${(oi.open_interest_usd or 0)/1e9:.2f}B", log_queue)
                    if market_state.order_book:
                        ob = market_state.order_book
                        log(f"  [L1] Orderbook: Imbalance {ob.imbalance:+.2f} | Spread {ob.spread*10000:.1f}bps", log_queue)
                    
                    # Log L/S ratio
                    ls_ratio = engine.hydra.layer1.get_long_short_ratio(symbol)
                    if ls_ratio:
                        log(f"  [L1] L/S Ratio: {ls_ratio['ratio']:.2f} (L:{ls_ratio['longAccount']:.1%} S:{ls_ratio['shortAccount']:.1%})", log_queue)
                    
                    # Layer 2: Statistical Analysis
                    log(f"  [L2] Running statistical analysis...", log_queue)
                    stat_result = await engine.hydra.layer2.analyze(market_state)
                    result['l2_decision'] = stat_result.trading_decision.value.upper()
                    
                    log(f"  [L2] Regime: {stat_result.regime.name} (conf: {stat_result.regime_confidence:.2f})", log_queue)
                    log(f"  [L2] Volatility: {stat_result.realized_volatility*100:.1f}% ({stat_result.volatility_regime})", log_queue)
                    log(f"  [L2] Danger Score: {stat_result.danger_score}/100", log_queue)
                    log(f"  [L2] Jump Prob: {stat_result.jump_probability*100:.1f}% | Cascade: {stat_result.cascade_probability*100:.1f}%", log_queue)
                    log(f"  [L2] Decision: {result['l2_decision']}", log_queue)
                    
                    # L2 Safety check for existing positions
                    force_exit, force_reason = engine.position_manager.check_layer2_safety(
                        stat_result, position
                    )
                    
                    if force_exit and position:
                        log(f"  ⚠️ L2 SAFETY TRIGGER: {force_reason}", log_queue, "WARN")
                        result['final_action'] = 'FORCE EXIT'
                        engine.portfolio.close_position(symbol, market_state.price, "force_exit_l2_block")
                        engine.position_manager.clear_position_context(symbol)
                        
                        result_queue.put({
                            'type': 'trade',
                            'data': {
                                'time': datetime.now().strftime('%H:%M:%S'),
                                'symbol': display_name,
                                'action': 'FORCE EXIT',
                                'side': position.side.value.upper(),
                                'price': market_state.price,
                                'reason': force_reason,
                            }
                        })
                        cycle_results.append(result)
                        continue
                    
                    # Position Management Mode
                    if mode == OperatingMode.MANAGEMENT and position:
                        log(f"  [MGT] Evaluating position management...", log_queue)
                        context = engine.position_manager.get_position_context(symbol)
                        thesis_status = engine.position_manager.check_thesis_validity(
                            position, market_state, stat_result, context
                        )
                        log(f"  [MGT] Thesis status: {thesis_status.value}", log_queue)
                        
                        decision = engine.position_manager.evaluate_exit_decision(
                            position, market_state, stat_result, thesis_status, context
                        )
                        log(f"  [MGT] Exit decision: {decision.action.value} (reason: {decision.exit_reason})", log_queue)
                        
                        result['l3_signal'] = position.side.value.upper()
                        
                        if decision.action in [PositionAction.EXIT, PositionAction.FORCE_EXIT]:
                            result['final_action'] = 'EXIT'
                            engine.portfolio.close_position(
                                symbol, market_state.price, decision.exit_reason,
                                partial_pct=decision.exit_pct
                            )
                            if decision.exit_pct >= 0.99:
                                engine.position_manager.clear_position_context(symbol)
                            
                            result_queue.put({
                                'type': 'trade',
                                'data': {
                                    'time': datetime.now().strftime('%H:%M:%S'),
                                    'symbol': display_name,
                                    'action': 'EXIT',
                                    'side': position.side.value.upper(),
                                    'price': market_state.price,
                                    'reason': decision.exit_reason,
                                }
                            })
                            log(f"  → EXITED position", log_queue)
                        else:
                            result['final_action'] = 'HOLD'
                            log(f"  → HOLDING position", log_queue)
                        
                        cycle_results.append(result)
                        continue
                    
                    # Entry Mode - check L2 gate
                    if stat_result.trading_decision == TradingDecision.BLOCK:
                        log(f"  ❌ BLOCKED by L2 (danger too high)", log_queue)
                        result['final_action'] = 'BLOCKED'
                        cycle_results.append(result)
                        continue
                    
                    # Layer 3: Alpha Generation
                    log(f"  [L3] Generating alpha signals (LLM + Transformer)...", log_queue)
                    
                    recent_news = []
                    news_sentiment = engine.hydra.layer1.get_news_sentiment(symbol)
                    if news_sentiment and news_sentiment.breaking_news:
                        recent_news = [n.title for n in news_sentiment.breaking_news[:5]]
                        log(f"  [L3] News: {len(recent_news)} headlines", log_queue)
                    
                    signals = await engine.hydra.layer3.generate_signals(
                        market_state, stat_result, None, recent_news
                    )
                    
                    # Log LLM interaction
                    try:
                        llm_prompt, llm_response = get_last_llm_interaction()
                        if llm_prompt:
                            log(f"  [L3-LLM] Prompt ({len(llm_prompt)} chars):", log_queue)
                            for line in llm_prompt.split('\n')[:5]:
                                log(f"       {line}", log_queue)
                        if llm_response:
                            log(f"  [L3-LLM] Response ({len(llm_response)} chars):", log_queue)
                            log(f"       {llm_response[:200]}...", log_queue)
                    except:
                        pass
                    
                    if not signals:
                        log(f"  [L3] No signals generated", log_queue)
                        result['final_action'] = 'NO SIGNAL'
                        cycle_results.append(result)
                        continue
                    
                    best_signal = max(signals, key=lambda s: s.confidence)
                    result['l3_signal'] = best_signal.side.value.upper()
                    result['l3_confidence'] = best_signal.confidence
                    
                    log(f"  [L3] Signal: {result['l3_signal']} @ {result['l3_confidence']:.1%} confidence", log_queue)
                    if best_signal.metadata:
                        thesis = best_signal.metadata.get('thesis', '')[:80]
                        if thesis:
                            log(f"  [L3] Thesis: {thesis}...", log_queue)
                    
                    if best_signal.confidence < config.risk.min_confidence_threshold:
                        log(f"  [L3] ❌ Confidence {best_signal.confidence:.1%} < threshold {config.risk.min_confidence_threshold:.1%}", log_queue)
                        result['final_action'] = 'LOW CONF'
                        cycle_results.append(result)
                        continue
                    
                    # Calculate leverage
                    log(f"  [L4] Calculating position parameters...", log_queue)
                    leverage_decision = engine.position_manager.calculate_leverage(
                        best_signal.confidence, stat_result, market_state,
                        engine.portfolio.current_drawdown
                    )
                    result['l4_leverage'] = leverage_decision.leverage
                    log(f"  [L4] Leverage: {leverage_decision.leverage:.1f}x (reason: {leverage_decision.reasoning})", log_queue)
                    
                    # Calculate size
                    all_positions = engine._get_all_positions_as_type()
                    stop_loss_pct = best_signal.expected_adverse_excursion or 0.02
                    size_usd = engine.position_manager.calculate_position_size(
                        total_equity=engine.portfolio.total_equity,
                        signal_confidence=best_signal.confidence,
                        leverage=leverage_decision.leverage,
                        stop_loss_pct=stop_loss_pct,
                        max_positions=config.trading.max_positions,
                        current_positions=len(all_positions),
                    )
                    result['l4_size'] = size_usd
                    log(f"  [L4] Size: ${size_usd:.0f} (equity: ${engine.portfolio.total_equity:.0f}, stop: {stop_loss_pct:.1%})", log_queue)
                    
                    if size_usd < 50:
                        log(f"  [L4] ❌ Size ${size_usd:.0f} too small (min $50)", log_queue)
                        result['final_action'] = 'SIZE TOO SMALL'
                        cycle_results.append(result)
                        continue
                    
                    # Layer 4: Risk Evaluation
                    log(f"  [L4] Running risk evaluation...", log_queue)
                    signal = Signal(
                        timestamp=datetime.now(timezone.utc),
                        symbol=symbol,
                        side=best_signal.side,
                        confidence=best_signal.confidence,
                        expected_return=0.02,
                        expected_adverse_excursion=best_signal.expected_adverse_excursion,
                        holding_period_minutes=int(best_signal.expected_holding_period_hours * 60),
                        source="hydra_alpha",
                        regime=stat_result.regime,
                        metadata={'thesis': best_signal.metadata.get('thesis', '') if best_signal.metadata else ''},
                    )
                    
                    risk_decision = await engine.hydra.layer4.evaluate(
                        signal, market_state, stat_result, position, all_positions
                    )
                    
                    log(f"  [L4] Approved: {risk_decision.approved} | Veto: {risk_decision.veto}", log_queue)
                    if risk_decision.veto:
                        log(f"  [L4] Veto reason: {risk_decision.veto_reason}", log_queue)
                    
                    if risk_decision.trigger_kill_switch:
                        log(f"  ⚠️ KILL SWITCH TRIGGERED", log_queue, "CRITICAL")
                        result['final_action'] = 'KILL SWITCH'
                        cycle_results.append(result)
                        continue
                    
                    result['l4_approved'] = risk_decision.approved
                    if not risk_decision.approved:
                        log(f"  [L4] ❌ Risk rejected: {risk_decision.veto_reason}", log_queue)
                        result['final_action'] = 'L4 VETO'
                        cycle_results.append(result)
                        continue
                    
                    # Layer 5: Multi-Agent Voting
                    log(f"  [L5] Collecting agent votes...", log_queue)
                    votes = await engine.hydra.layer5.collect_votes(
                        signal, market_state, stat_result, risk_decision
                    )
                    
                    for vote in votes:
                        log(f"  [L5] {vote.agent}: {'✓' if vote.approved else '✗'} (conf: {vote.confidence:.1%}) - {vote.reasoning[:50]}", log_queue)
                    
                    approved, plan = engine.hydra.layer5.evaluate_votes(votes, signal, risk_decision)
                    result['l5_approved'] = approved
                    
                    log(f"  [L5] Final vote: {'APPROVED' if approved else 'REJECTED'}", log_queue)
                    
                    if not approved:
                        log(f"  [L5] ❌ Trade not approved by agents", log_queue)
                        result['final_action'] = 'L5 VETO'
                        cycle_results.append(result)
                        continue
                    
                    # Execute trade
                    log(f"  ✅ EXECUTING TRADE: {signal.side.value.upper()} ${size_usd:.0f} @ {leverage_decision.leverage:.1f}x", log_queue)
                    result['final_action'] = signal.side.value.upper()
                    await engine._execute_trade(
                        symbol=symbol,
                        side=signal.side,
                        size_usd=size_usd,
                        price=market_state.price,
                        leverage=leverage_decision.leverage,
                        signal=signal,
                    )
                    
                    thesis = signal.metadata.get('thesis', '') if signal.metadata else ''
                    result_queue.put({
                        'type': 'trade',
                        'data': {
                            'time': datetime.now().strftime('%H:%M:%S'),
                            'symbol': display_name,
                            'action': signal.side.value.upper(),
                            'side': signal.side.value.upper(),
                            'price': market_state.price,
                            'size': size_usd,
                            'leverage': leverage_decision.leverage,
                            'reason': thesis[:50] if thesis else '',
                        }
                    })
                    
                except Exception as e:
                    result['final_action'] = 'ERROR'
                    log(f"  ❌ ERROR: {e}", log_queue, "ERROR")
                    import traceback
                    log(f"  {traceback.format_exc()[:200]}", log_queue, "ERROR")
                
                cycle_results.append(result)
            
            # Update portfolio prices
            engine.portfolio.update_prices(engine._current_prices)
            
            # Get positions with current values
            positions_data = []
            for sym, pos in engine.portfolio.positions.items():
                if pos.is_open:
                    current_price = engine._current_prices.get(sym, pos.current_price)
                    entry_value = pos.size_usd
                    margin_used = entry_value / pos.leverage
                    
                    if pos.side == Side.LONG:
                        price_change_pct = (current_price - pos.entry_price) / pos.entry_price
                    else:
                        price_change_pct = (pos.entry_price - current_price) / pos.entry_price
                    
                    pnl = entry_value * price_change_pct
                    pnl_pct = price_change_pct * pos.leverage
                    current_value = margin_used + pnl
                    
                    positions_data.append({
                        'symbol': PAIR_DISPLAY_NAMES.get(sym, sym),
                        'side': pos.side.value.upper(),
                        'size_usd': entry_value,
                        'margin': margin_used,
                        'leverage': pos.leverage,
                        'entry_price': pos.entry_price,
                        'current_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'current_value': current_value,
                        'entry_time': pos.entry_time.strftime('%H:%M:%S') if pos.entry_time else '',
                    })
            
            # Send cycle results
            result_queue.put({
                'type': 'cycle',
                'data': {
                    'cycle': cycle_count,
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'results': cycle_results,
                    'positions': positions_data,
                    'equity': engine.portfolio.total_equity,
                    'balance': engine.portfolio.balance,
                    'pnl': engine.portfolio.total_equity - balance,
                    'pnl_pct': (engine.portfolio.total_equity - balance) / balance,
                }
            })
            
            log("", log_queue)
            log(f"═══ CYCLE {cycle_count} COMPLETE ═══", log_queue)
            log(f"Equity: ${engine.portfolio.total_equity:.2f} | P&L: ${engine.portfolio.total_equity - balance:+.2f}", log_queue)
            log(f"Next cycle in 60 seconds...", log_queue)
            
            # Wait 60 seconds or until stop
            for i in range(60):
                if stop_event.is_set():
                    break
                await asyncio.sleep(1)
                if i % 15 == 14:  # Log every 15 seconds
                    log(f"  ... {60 - i - 1}s until next cycle", log_queue)
        
        # Cleanup
        log("", log_queue)
        log("STOPPING HYDRA...", log_queue)
        await engine.hydra.stop()
        engine.portfolio.save()
        log("✓ HYDRA stopped and portfolio saved", log_queue)
    
    # Run the async function
    asyncio.run(run_hydra())


def main():
    # Sidebar
    with st.sidebar:
        st.title("🐍 HYDRA")
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
                st.session_state.positions = []
                st.session_state.cycle_count = 0
                
                st.session_state.stop_event = threading.Event()
                st.session_state.log_queue = queue.Queue()
                st.session_state.result_queue = queue.Queue()
                
                thread = threading.Thread(
                    target=run_hydra_async,
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
                st.session_state.pipeline_results = []
                st.session_state.trades = []
                st.session_state.logs = []
                st.session_state.positions = []
                st.session_state.cycle_count = 0
                st.rerun()
        
        if st.session_state.running:
            st.success("🟢 Running")
            st.metric("Cycles", st.session_state.cycle_count)
        else:
            st.info("🔴 Stopped")
        
        st.markdown("---")
        st.caption("60s interval • Verbose logging")
    
    # Process queues
    if st.session_state.running:
        while not st.session_state.log_queue.empty():
            try:
                log_entry = st.session_state.log_queue.get_nowait()
                st.session_state.logs.append(log_entry)
                if len(st.session_state.logs) > 500:
                    st.session_state.logs = st.session_state.logs[-500:]
            except queue.Empty:
                break
        
        while not st.session_state.result_queue.empty():
            try:
                result = st.session_state.result_queue.get_nowait()
                if result['type'] == 'cycle':
                    st.session_state.pipeline_results.insert(0, result['data'])
                    st.session_state.cycle_count = result['data']['cycle']
                    st.session_state.positions = result['data']['positions']
                elif result['type'] == 'trade':
                    st.session_state.trades.insert(0, result['data'])
            except queue.Empty:
                break
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📝 Logs (Verbose)", "💹 Trades"])
    
    with tab1:
        st.header("HYDRA Dashboard")
        
        # Portfolio summary
        if st.session_state.pipeline_results:
            latest = st.session_state.pipeline_results[0]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Equity", f"${latest['equity']:,.2f}")
            with col2:
                pnl = latest['pnl']
                st.metric("P&L", f"${pnl:+,.2f}", f"{latest['pnl_pct']:+.2%}")
            with col3:
                st.metric("Balance", f"${latest['balance']:,.2f}")
            with col4:
                st.metric("Cycle", latest['cycle'])
        
        st.markdown("---")
        
        # Active Positions
        st.subheader("📍 Active Positions (Real-time Value)")
        positions = st.session_state.positions
        if positions:
            pos_df = pd.DataFrame(positions)
            
            display_df = pos_df.copy()
            display_df['size_usd'] = display_df['size_usd'].apply(lambda x: f"${x:,.0f}")
            display_df['margin'] = display_df['margin'].apply(lambda x: f"${x:,.2f}")
            display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:,.2f}")
            display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
            display_df['current_value'] = display_df['current_value'].apply(lambda x: f"${x:,.2f}")
            display_df['pnl_display'] = pos_df.apply(lambda r: f"${r['pnl']:+,.2f} ({r['pnl_pct']:+.1%})", axis=1)
            display_df['leverage'] = display_df['leverage'].apply(lambda x: f"{x:.1f}x")
            
            st.dataframe(
                display_df[['symbol', 'side', 'size_usd', 'margin', 'leverage', 'entry_price', 'current_price', 'current_value', 'pnl_display', 'entry_time']],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No active positions")
        
        st.markdown("---")
        
        # Pipeline Summary
        st.subheader("🔄 Pipeline Summary (Latest 10)")
        
        pipeline_results = st.session_state.pipeline_results
        if pipeline_results:
            page_size = 10
            total_pages = max(1, (len(pipeline_results) + page_size - 1) // page_size)
            
            if total_pages > 1:
                page = st.selectbox("Page", range(1, total_pages + 1), key="pipeline_page")
            else:
                page = 1
            
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, len(pipeline_results))
            
            for i, cycle_data in enumerate(pipeline_results[start_idx:end_idx]):
                with st.expander(f"Cycle {cycle_data['cycle']} - {cycle_data['time']}", expanded=(i == 0)):
                    results = cycle_data['results']
                    
                    rows = []
                    for r in results:
                        rows.append({
                            'Symbol': r['display'],
                            'Mode': r['mode'][:4],
                            'L2 Gate': r['l2_decision'] or 'N/A',
                            'L3 Signal': f"{r['l3_signal']} {r['l3_confidence']:.0%}" if r['l3_signal'] and r['l3_confidence'] > 0 else (r['l3_signal'] or 'N/A'),
                            'L4 Risk': f"${r['l4_size']:.0f} @ {r['l4_leverage']:.1f}x" if r['l4_size'] > 0 else ('OK' if r['l4_approved'] else 'N/A'),
                            'L5 Vote': '✓' if r['l5_approved'] else ('✗' if r['final_action'] == 'L5 VETO' else '—'),
                            'Final': r['final_action'],
                        })
                    
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Click 'Run' to start HYDRA")
    
    with tab2:
        st.header("📝 Verbose System Logs")
        st.caption("Shows every detail: LLM prompts, responses, calculations, layer outputs")
        
        logs = st.session_state.logs
        if logs:
            # Filter options
            col1, col2 = st.columns([1, 4])
            with col1:
                show_all = st.checkbox("Show all", value=True)
            
            if show_all:
                display_logs = logs
            else:
                display_logs = [l for l in logs if l.get('level', 'INFO') != 'DEBUG']
            
            # Display logs in a scrollable text area
            log_text = "\n".join([
                f"[{log['time']}] {log.get('level', 'INFO'):8} | {log['message']}" 
                for log in display_logs[-200:]  # Last 200 logs
            ])
            st.code(log_text, language="")
        else:
            st.info("No logs yet")
    
    with tab3:
        st.header("💹 Trades & Actions")
        
        trades = st.session_state.trades
        if trades:
            trades_df = pd.DataFrame(trades)
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades yet")
    
    # Auto-refresh
    if st.session_state.running:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
