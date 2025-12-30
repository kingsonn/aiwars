"""
HYDRA Paper Trading Dashboard

Real-time display of:
- Portfolio value and P&L
- Open positions with entry prices
- Trade history
- Performance metrics
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

from hydra.paper_trading.portfolio import Portfolio, PortfolioSnapshot
from hydra.core.config import PAIR_DISPLAY_NAMES


class TradingDashboard:
    """
    Rich-based trading dashboard for paper trading.
    
    Displays:
    - Portfolio summary
    - Open positions
    - Recent trades
    - Performance metrics
    """
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.console = Console()
    
    def _create_header(self) -> Panel:
        """Create header panel."""
        now = datetime.now(timezone.utc)
        summary = self.portfolio.get_summary()
        
        pnl = summary["total_pnl"]
        pnl_pct = summary["total_pnl_pct"]
        pnl_color = "green" if pnl >= 0 else "red"
        
        header_text = Text()
        header_text.append("HYDRA ", style="bold cyan")
        header_text.append("Paper Trading Dashboard\n", style="bold white")
        header_text.append(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC\n", style="dim")
        header_text.append(f"\nEquity: ", style="white")
        header_text.append(f"${summary['total_equity']:,.2f}", style="bold white")
        header_text.append(f"  |  P&L: ", style="white")
        header_text.append(f"${pnl:+,.2f} ({pnl_pct:+.2%})", style=f"bold {pnl_color}")
        
        return Panel(header_text, box=box.DOUBLE)
    
    def _create_portfolio_table(self) -> Table:
        """Create portfolio summary table."""
        summary = self.portfolio.get_summary()
        
        table = Table(title="Portfolio Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Initial Balance", f"${summary['initial_balance']:,.2f}")
        table.add_row("Total Equity", f"${summary['total_equity']:,.2f}")
        table.add_row("Available Balance", f"${summary['available_balance']:,.2f}")
        table.add_row("Used Margin", f"${summary['used_margin']:,.2f}")
        table.add_row("─" * 20, "─" * 15)
        
        pnl_color = "green" if summary['realized_pnl'] >= 0 else "red"
        table.add_row("Realized P&L", f"[{pnl_color}]${summary['realized_pnl']:+,.2f}[/]")
        
        upnl_color = "green" if summary['unrealized_pnl'] >= 0 else "red"
        table.add_row("Unrealized P&L", f"[{upnl_color}]${summary['unrealized_pnl']:+,.2f}[/]")
        
        table.add_row("Funding Paid", f"${summary['funding_paid']:,.2f}")
        table.add_row("─" * 20, "─" * 15)
        table.add_row("Peak Equity", f"${summary['peak_equity']:,.2f}")
        table.add_row("Current Drawdown", f"[yellow]{summary['current_drawdown']:.2%}[/]")
        table.add_row("Gross Leverage", f"{summary['gross_leverage']:.2f}x")
        table.add_row("─" * 20, "─" * 15)
        table.add_row("Total Trades", str(summary['total_trades']))
        table.add_row("Win Rate", f"{summary['win_rate']:.1%}")
        
        return table
    
    def _create_positions_table(self) -> Table:
        """Create open positions table."""
        positions = self.portfolio.get_open_positions()
        
        table = Table(title=f"Open Positions ({len(positions)})", box=box.ROUNDED)
        table.add_column("Symbol", style="cyan")
        table.add_column("Side", justify="center")
        table.add_column("Size", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")
        table.add_column("Lev", justify="center")
        table.add_column("Liq Price", justify="right")
        
        if not positions:
            table.add_row("No open positions", "", "", "", "", "", "", "", "")
        else:
            for pos in positions:
                side_color = "green" if pos["side"] == "long" else "red"
                pnl_color = "green" if pos["unrealized_pnl"] >= 0 else "red"
                
                table.add_row(
                    pos["symbol"],
                    f"[{side_color}]{pos['side'].upper()}[/]",
                    f"${pos['size_usd']:,.2f}",
                    f"${pos['entry_price']:,.2f}",
                    f"${pos['current_price']:,.2f}",
                    f"[{pnl_color}]${pos['unrealized_pnl']:+,.2f}[/]",
                    f"[{pnl_color}]{pos['unrealized_pnl_pct']:+.2%}[/]",
                    f"{pos['leverage']:.1f}x",
                    f"${pos['liquidation_price']:,.2f}",
                )
        
        return table
    
    def _create_trades_table(self, limit: int = 10) -> Table:
        """Create recent trades table."""
        trades = self.portfolio.get_trade_history(limit=limit)
        
        table = Table(title=f"Recent Trades (Last {limit})", box=box.ROUNDED)
        table.add_column("ID", style="dim")
        table.add_column("Time", style="dim")
        table.add_column("Symbol", style="cyan")
        table.add_column("Action", justify="center")
        table.add_column("Side", justify="center")
        table.add_column("Price", justify="right")
        table.add_column("P&L", justify="right")
        
        if not trades:
            table.add_row("No trades yet", "", "", "", "", "", "")
        else:
            for trade in trades:
                side_color = "green" if trade["side"] == "long" else "red"
                
                action_colors = {
                    "open": "green",
                    "close": "red",
                    "add": "cyan",
                    "reduce": "yellow",
                }
                action_color = action_colors.get(trade["action"], "white")
                
                pnl_str = ""
                if trade["pnl"] != 0:
                    pnl_color = "green" if trade["pnl"] >= 0 else "red"
                    pnl_str = f"[{pnl_color}]${trade['pnl']:+,.2f}[/]"
                
                # Parse time
                time_str = trade["time"].split("T")[1][:8] if "T" in trade["time"] else trade["time"]
                
                table.add_row(
                    trade["id"],
                    time_str,
                    trade["symbol"],
                    f"[{action_color}]{trade['action'].upper()}[/]",
                    f"[{side_color}]{trade['side'].upper()}[/]",
                    f"${trade['price']:,.2f}",
                    pnl_str,
                )
        
        return table
    
    def print_dashboard(self) -> None:
        """Print the full dashboard to console."""
        self.console.clear()
        self.console.print(self._create_header())
        self.console.print()
        
        # Create two-column layout
        self.console.print(self._create_portfolio_table())
        self.console.print()
        self.console.print(self._create_positions_table())
        self.console.print()
        self.console.print(self._create_trades_table())
    
    def print_positions(self) -> None:
        """Print only positions table."""
        self.console.print(self._create_positions_table())
    
    def print_trades(self, limit: int = 20) -> None:
        """Print only trades table."""
        self.console.print(self._create_trades_table(limit=limit))
    
    def print_summary(self) -> None:
        """Print portfolio summary."""
        summary = self.portfolio.get_summary()
        
        self.console.print("\n[bold cyan]═══ HYDRA Portfolio Summary ═══[/bold cyan]\n")
        
        pnl = summary["total_pnl"]
        pnl_pct = summary["total_pnl_pct"]
        pnl_color = "green" if pnl >= 0 else "red"
        
        self.console.print(f"  Initial Balance:  [white]${summary['initial_balance']:,.2f}[/white]")
        self.console.print(f"  Current Equity:   [bold white]${summary['total_equity']:,.2f}[/bold white]")
        self.console.print(f"  Total P&L:        [{pnl_color}]${pnl:+,.2f} ({pnl_pct:+.2%})[/{pnl_color}]")
        self.console.print()
        self.console.print(f"  Open Positions:   {summary['num_positions']}")
        self.console.print(f"  Total Trades:     {summary['total_trades']}")
        self.console.print(f"  Win Rate:         {summary['win_rate']:.1%}")
        self.console.print(f"  Max Drawdown:     [yellow]{summary['current_drawdown']:.2%}[/yellow]")
        self.console.print()
    
    def print_position_detail(self, symbol: str) -> None:
        """Print detailed info for a specific position."""
        symbol = symbol.lower()
        position = self.portfolio.positions.get(symbol)
        
        if not position or not position.is_open:
            self.console.print(f"[yellow]No open position for {symbol}[/yellow]")
            return
        
        pnl_color = "green" if position.unrealized_pnl >= 0 else "red"
        side_color = "green" if position.side.value == "long" else "red"
        
        self.console.print(f"\n[bold cyan]═══ Position: {position.display_name} ═══[/bold cyan]\n")
        self.console.print(f"  Side:             [{side_color}]{position.side.value.upper()}[/{side_color}]")
        self.console.print(f"  Size:             {position.size:.6f} ({position.size_usd:,.2f} USD)")
        self.console.print(f"  Leverage:         {position.leverage:.1f}x")
        self.console.print()
        self.console.print(f"  Entry Price:      ${position.avg_entry_price:,.2f}")
        self.console.print(f"  Current Price:    ${position.current_price:,.2f}")
        self.console.print(f"  Liquidation:      ${position.liquidation_price:,.2f}")
        self.console.print()
        self.console.print(f"  Unrealized P&L:   [{pnl_color}]${position.unrealized_pnl:+,.2f} ({position.unrealized_pnl_pct:+.2%})[/{pnl_color}]")
        self.console.print(f"  Funding Paid:     ${position.total_funding_paid:,.2f}")
        self.console.print(f"  Margin Used:      ${position.margin_used:,.2f}")
        self.console.print(f"  Entry Time:       {position.entry_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        self.console.print()
