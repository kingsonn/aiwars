"""
HYDRA Trading Dashboard

Streamlit-based dashboard for HYDRA paper trading with:
- Run/Stop controls with balance input
- Real-time pipeline summary tables (latest 10, pagination)
- Active positions with real-time value calculation
- Logs tab
- Trades/Actions tab
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
import io

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
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = []  # List of cycle results
if 'trades' not in st.session_state:
    st.session_state.trades = []  # List of trades/actions
if 'logs' not in st.session_state:
    st.session_state.logs = []  # List of log messages
if 'positions' not in st.session_state:
    st.session_state.positions = {}  # Current positions
if 'current_prices' not in st.session_state:
    st.session_state.current_prices = {}
if 'cycle_count' not in st.session_state:
    st.session_state.cycle_count = 0
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = None
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = queue.Queue()
if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue()


class StreamlitLogHandler:
    """Custom log handler that sends logs to a queue."""
    def __init__(self, log_queue):
        self.log_queue = log_queue
    
    def write(self, message):
        if message.strip():
            self.log_queue.put({
                'time': datetime.now().strftime('%H:%M:%S'),
                'message': message.strip()
            })
    
    def flush(self):
        pass


def run_hydra_async(balance: float, stop_event: threading.Event, result_queue: queue.Queue, log_queue: queue.Queue):
    """Run HYDRA in a separate thread with asyncio."""
    
    # Setup logging to queue
    log_handler = StreamlitLogHandler(log_queue)
    logger.remove()
    logger.add(log_handler.write, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
    
    async def run_hydra():
        from hydra.core.config import HydraConfig, PAIR_DISPLAY_NAMES
        from hydra.core.types import Side, Signal, PositionAction
        from hydra.paper_trading.engine import PaperTradingEngine
        from hydra.core.position_manager import OperatingMode
        from hydra.layers.layer2_statistical import TradingDecision
        
        config = HydraConfig.load()
        engine = PaperTradingEngine(
            config=config,
            initial_balance=balance,
            data_dir="./data/dashboard_trading",
        )
        
        await engine.hydra.setup()
        log_queue.put({'time': datetime.now().strftime('%H:%M:%S'), 'message': '✓ HYDRA initialized'})
        
        cycle_count = 0
        
        while not stop_event.is_set():
            cycle_count += 1
            cycle_start = datetime.now(timezone.utc)
            log_queue.put({'time': datetime.now().strftime('%H:%M:%S'), 'message': f'━━━ Cycle {cycle_count} Started ━━━'})
            
            cycle_results = []
            
            for symbol in config.trading.symbols:
                display_name = PAIR_DISPLAY_NAMES.get(symbol, symbol)
                result = {
                    'symbol': symbol,
                    'display': display_name.split("/")[0],
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
                    
                    if position:
                        result['pnl_pct'] = position.unrealized_pnl_pct
                    
                    # Get market state
                    market_state = await engine.hydra.layer1.get_market_state(symbol)
                    if not market_state:
                        result['final_action'] = 'NO DATA'
                        cycle_results.append(result)
                        continue
                    
                    result['current_price'] = market_state.price
                    engine._current_prices[symbol] = market_state.price
                    
                    # Layer 2
                    stat_result = await engine.hydra.layer2.analyze(market_state)
                    result['l2_decision'] = stat_result.trading_decision.value.upper()
                    
                    # L2 Safety check
                    force_exit, force_reason = engine.position_manager.check_layer2_safety(
                        stat_result, position
                    )
                    
                    if force_exit and position:
                        result['final_action'] = 'FORCE EXIT'
                        engine.portfolio.close_position(symbol, market_state.price, "force_exit_l2_block")
                        engine.position_manager.clear_position_context(symbol)
                        
                        # Record trade
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
                        context = engine.position_manager.get_position_context(symbol)
                        thesis_status = engine.position_manager.check_thesis_validity(
                            position, market_state, stat_result, context
                        )
                        decision = engine.position_manager.evaluate_exit_decision(
                            position, market_state, stat_result, thesis_status, context
                        )
                        
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
                        else:
                            result['final_action'] = 'HOLD'
                        
                        cycle_results.append(result)
                        continue
                    
                    # Entry Mode
                    if stat_result.trading_decision == TradingDecision.BLOCK:
                        result['final_action'] = 'BLOCKED'
                        cycle_results.append(result)
                        continue
                    
                    # Layer 3
                    recent_news = []
                    news_sentiment = engine.hydra.layer1.get_news_sentiment(symbol)
                    if news_sentiment and news_sentiment.breaking_news:
                        recent_news = [n.title for n in news_sentiment.breaking_news[:5]]
                    
                    signals = await engine.hydra.layer3.generate_signals(
                        market_state, stat_result, None, recent_news
                    )
                    
                    if not signals:
                        result['final_action'] = 'NO SIGNAL'
                        cycle_results.append(result)
                        continue
                    
                    best_signal = max(signals, key=lambda s: s.confidence)
                    result['l3_signal'] = best_signal.side.value.upper()
                    result['l3_confidence'] = best_signal.confidence
                    
                    if best_signal.confidence < config.risk.min_confidence_threshold:
                        result['final_action'] = 'LOW CONF'
                        cycle_results.append(result)
                        continue
                    
                    # Calculate leverage
                    leverage_decision = engine.position_manager.calculate_leverage(
                        best_signal.confidence, stat_result, market_state,
                        engine.portfolio.current_drawdown
                    )
                    result['l4_leverage'] = leverage_decision.leverage
                    
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
                    
                    if size_usd < 50:
                        result['final_action'] = 'SIZE TOO SMALL'
                        cycle_results.append(result)
                        continue
                    
                    # Layer 4
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
                    
                    if risk_decision.trigger_kill_switch:
                        result['final_action'] = 'KILL SWITCH'
                        cycle_results.append(result)
                        continue
                    
                    result['l4_approved'] = risk_decision.approved
                    if not risk_decision.approved:
                        result['final_action'] = 'L4 VETO'
                        cycle_results.append(result)
                        continue
                    
                    # Layer 5
                    votes = await engine.hydra.layer5.collect_votes(
                        signal, market_state, stat_result, risk_decision
                    )
                    approved, plan = engine.hydra.layer5.evaluate_votes(votes, signal, risk_decision)
                    result['l5_approved'] = approved
                    
                    if not approved:
                        result['final_action'] = 'L5 VETO'
                        cycle_results.append(result)
                        continue
                    
                    # Execute trade
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
                    log_queue.put({'time': datetime.now().strftime('%H:%M:%S'), 'message': f'Error {symbol}: {e}'})
                
                cycle_results.append(result)
            
            # Update portfolio prices
            engine.portfolio.update_prices(engine._current_prices)
            
            # Get positions with current values
            positions_data = []
            for sym, pos in engine.portfolio.positions.items():
                if pos.is_open:
                    current_price = engine._current_prices.get(sym, pos.current_price)
                    # Calculate current value with leverage
                    entry_value = pos.size_usd
                    margin_used = entry_value / pos.leverage
                    
                    # Calculate P&L
                    if pos.side == Side.LONG:
                        price_change_pct = (current_price - pos.entry_price) / pos.entry_price
                    else:
                        price_change_pct = (pos.entry_price - current_price) / pos.entry_price
                    
                    pnl = entry_value * price_change_pct
                    pnl_pct = price_change_pct * pos.leverage  # Leveraged P&L %
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
            
            log_queue.put({'time': datetime.now().strftime('%H:%M:%S'), 'message': f'━━━ Cycle {cycle_count} Complete ━━━'})
            
            # Wait 60 seconds or until stop
            for _ in range(60):
                if stop_event.is_set():
                    break
                await asyncio.sleep(1)
        
        # Cleanup
        await engine.hydra.stop()
        engine.portfolio.save()
        log_queue.put({'time': datetime.now().strftime('%H:%M:%S'), 'message': '✓ HYDRA stopped and state saved'})
    
    # Run the async function
    asyncio.run(run_hydra())


def main():
    # Sidebar
    with st.sidebar:
        st.title("🐍 HYDRA")
        st.markdown("---")
        
        # Balance input
        balance = st.number_input(
            "Starting Balance (USDT)",
            min_value=100.0,
            max_value=1000000.0,
            value=1000.0,
            step=100.0,
            disabled=st.session_state.running
        )
        
        st.markdown("---")
        
        # Run/Stop buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Run HYDRA", disabled=st.session_state.running, use_container_width=True):
                # Clear previous state
                st.session_state.pipeline_results = []
                st.session_state.trades = []
                st.session_state.logs = []
                st.session_state.positions = {}
                st.session_state.cycle_count = 0
                
                # Create stop event and queues
                st.session_state.stop_event = threading.Event()
                st.session_state.log_queue = queue.Queue()
                st.session_state.result_queue = queue.Queue()
                
                # Start HYDRA thread
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
                # Clear everything
                st.session_state.pipeline_results = []
                st.session_state.trades = []
                st.session_state.logs = []
                st.session_state.positions = {}
                st.session_state.cycle_count = 0
                st.rerun()
        
        # Status
        if st.session_state.running:
            st.success("🟢 HYDRA Running")
            st.metric("Cycles", st.session_state.cycle_count)
        else:
            st.info("🔴 HYDRA Stopped")
        
        st.markdown("---")
        st.caption("60 second interval between cycles")
    
    # Process queues
    if st.session_state.running:
        # Process log queue
        while not st.session_state.log_queue.empty():
            try:
                log = st.session_state.log_queue.get_nowait()
                st.session_state.logs.append(log)
                # Keep only last 100 logs
                if len(st.session_state.logs) > 100:
                    st.session_state.logs = st.session_state.logs[-100:]
            except queue.Empty:
                break
        
        # Process result queue
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
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📝 Logs", "💹 Trades"])
    
    with tab1:
        # Dashboard tab
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
        st.subheader("📍 Active Positions")
        positions = st.session_state.positions
        if positions:
            pos_df = pd.DataFrame(positions)
            
            # Style the dataframe
            def color_pnl(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val >= 0 else 'red'
                    return f'color: {color}'
                return ''
            
            # Format columns
            pos_df['size_usd'] = pos_df['size_usd'].apply(lambda x: f"${x:,.0f}")
            pos_df['margin'] = pos_df['margin'].apply(lambda x: f"${x:,.2f}")
            pos_df['entry_price'] = pos_df['entry_price'].apply(lambda x: f"${x:,.2f}")
            pos_df['current_price'] = pos_df['current_price'].apply(lambda x: f"${x:,.2f}")
            pos_df['current_value'] = pos_df['current_value'].apply(lambda x: f"${x:,.2f}")
            pos_df['pnl_display'] = pos_df.apply(lambda r: f"${r['pnl']:+,.2f} ({r['pnl_pct']:+.1%})", axis=1)
            pos_df['leverage'] = pos_df['leverage'].apply(lambda x: f"{x:.1f}x")
            
            st.dataframe(
                pos_df[['symbol', 'side', 'size_usd', 'margin', 'leverage', 'entry_price', 'current_price', 'current_value', 'pnl_display', 'entry_time']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'symbol': 'Symbol',
                    'side': 'Side',
                    'size_usd': 'Position Size',
                    'margin': 'Margin Used',
                    'leverage': 'Leverage',
                    'entry_price': 'Entry Price',
                    'current_price': 'Current Price',
                    'current_value': 'Current Value',
                    'pnl_display': 'P&L',
                    'entry_time': 'Entry Time',
                }
            )
        else:
            st.info("No active positions")
        
        st.markdown("---")
        
        # Pipeline Summary Tables (latest 10 with pagination)
        st.subheader("🔄 Pipeline Summary")
        
        pipeline_results = st.session_state.pipeline_results
        if pipeline_results:
            # Pagination
            page_size = 10
            total_pages = max(1, (len(pipeline_results) + page_size - 1) // page_size)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page = st.selectbox("Page", range(1, total_pages + 1), key="pipeline_page")
            
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, len(pipeline_results))
            
            for i, cycle_data in enumerate(pipeline_results[start_idx:end_idx]):
                with st.expander(f"Cycle {cycle_data['cycle']} - {cycle_data['time']}", expanded=(i == 0)):
                    results = cycle_data['results']
                    
                    # Create dataframe
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
                    
                    # Color coding for final action
                    def style_final(val):
                        if val in ['LONG']:
                            return 'background-color: #1a472a; color: white'
                        elif val in ['SHORT']:
                            return 'background-color: #4a1a1a; color: white'
                        elif val == 'HOLD':
                            return 'background-color: #4a4a1a; color: white'
                        elif val in ['EXIT', 'FORCE EXIT']:
                            return 'background-color: #4a1a4a; color: white'
                        elif val == 'BLOCKED':
                            return 'background-color: #4a1a1a; color: white'
                        return ''
                    
                    st.dataframe(
                        df.style.applymap(style_final, subset=['Final']),
                        use_container_width=True,
                        hide_index=True
                    )
        else:
            st.info("No pipeline data yet. Click 'Run HYDRA' to start.")
    
    with tab2:
        # Logs tab
        st.header("📝 System Logs")
        
        logs = st.session_state.logs
        if logs:
            # Display logs in reverse order (latest first)
            log_text = "\n".join([f"[{log['time']}] {log['message']}" for log in reversed(logs[-50:])])
            st.code(log_text, language="")
        else:
            st.info("No logs yet")
    
    with tab3:
        # Trades tab
        st.header("💹 Trades & Actions")
        
        trades = st.session_state.trades
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Style based on action
            def style_action(val):
                if val == 'LONG':
                    return 'background-color: #1a472a; color: white'
                elif val == 'SHORT':
                    return 'background-color: #4a1a1a; color: white'
                elif val == 'EXIT':
                    return 'background-color: #4a1a4a; color: white'
                elif val == 'FORCE EXIT':
                    return 'background-color: #4a0a0a; color: white'
                return ''
            
            st.dataframe(
                trades_df.style.applymap(style_action, subset=['action']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No trades yet")
    
    # Auto-refresh when running
    if st.session_state.running:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
