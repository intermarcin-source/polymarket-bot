import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

console = Console()
DATA_DIR = Path(__file__).parent.parent / "data"


def show_dashboard():
    """Display a rich terminal dashboard of bot performance."""

    console.clear()
    console.print(Panel(
        "[bold cyan]POLYMARKET TRADING BOT[/bold cyan] - [yellow]SIMULATION MODE[/yellow]",
        style="bold white",
    ))

    # Portfolio
    portfolio_file = DATA_DIR / "sim_portfolio.json"
    if portfolio_file.exists():
        with open(portfolio_file) as f:
            portfolio = json.load(f)

        table = Table(title="Portfolio", show_header=False, border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Balance", f"${portfolio.get('balance', 0):,.2f}")
        table.add_row("Starting Balance", f"${portfolio.get('starting_balance', 0):,.2f}")
        table.add_row("Total P&L", f"${portfolio.get('total_pnl', 0):+,.2f}")
        table.add_row("Last Updated", portfolio.get("last_updated", "N/A")[:19])
        console.print(table)
    else:
        console.print("[dim]No portfolio data yet. Run the bot first.[/dim]")

    # Open Positions
    if portfolio_file.exists():
        positions = portfolio.get("positions", [])
        open_pos = [p for p in positions if p.get("status") == "open"]

        if open_pos:
            pos_table = Table(title=f"Open Positions ({len(open_pos)})", border_style="green")
            pos_table.add_column("Market", max_width=45)
            pos_table.add_column("Outcome", style="cyan")
            pos_table.add_column("Source", style="yellow")
            pos_table.add_column("Entry", justify="right")
            pos_table.add_column("Current", justify="right")
            pos_table.add_column("Size", justify="right")
            pos_table.add_column("P&L", justify="right")

            for pos in open_pos:
                pnl = pos.get("pnl", 0)
                pnl_style = "green" if pnl >= 0 else "red"
                pos_table.add_row(
                    pos.get("market_question", "?")[:45],
                    pos.get("outcome", "?"),
                    pos.get("source", "?"),
                    f"${pos.get('price', 0):.4f}",
                    f"${pos.get('current_price', pos.get('price', 0)):.4f}",
                    f"${pos.get('size_usdc', 0):.2f}",
                    f"[{pnl_style}]${pnl:+.2f}[/{pnl_style}]",
                )
            console.print(pos_table)
        else:
            console.print("[dim]No open positions.[/dim]")

    # Trade History
    trades_file = DATA_DIR / "sim_trades.json"
    if trades_file.exists():
        with open(trades_file) as f:
            trades = json.load(f)

        if trades:
            recent = trades[-10:]  # last 10 trades
            trade_table = Table(title=f"Recent Trades (last 10 of {len(trades)})", border_style="blue")
            trade_table.add_column("Time", max_width=16)
            trade_table.add_column("Market", max_width=40)
            trade_table.add_column("Outcome")
            trade_table.add_column("Source")
            trade_table.add_column("Price", justify="right")
            trade_table.add_column("Size", justify="right")
            trade_table.add_column("Status")

            for t in reversed(recent):
                status = t.get("status", "open")
                status_style = {"open": "yellow", "won": "bold green", "lost": "bold red"}.get(status, "white")
                trade_table.add_row(
                    t.get("timestamp", "")[:16],
                    t.get("market_question", "?")[:40],
                    t.get("outcome", "?"),
                    t.get("source", "?"),
                    f"${t.get('price', 0):.4f}",
                    f"${t.get('size_usdc', 0):.2f}",
                    f"[{status_style}]{status.upper()}[/{status_style}]",
                )
            console.print(trade_table)

    # Whale Wallets
    whale_file = DATA_DIR / "whale_wallets.json"
    if whale_file.exists():
        with open(whale_file) as f:
            whales = json.load(f)
        if whales:
            whale_table = Table(title=f"Tracked Whales ({len(whales)})", border_style="magenta")
            whale_table.add_column("Label")
            whale_table.add_column("Address", max_width=15)
            whale_table.add_column("Win Rate", justify="right")
            whale_table.add_column("P&L", justify="right")

            for addr, meta in list(whales.items())[:10]:
                whale_table.add_row(
                    meta.get("label", "?"),
                    f"{addr[:6]}...{addr[-4:]}",
                    f"{meta.get('win_rate', 0):.0%}",
                    f"${meta.get('total_pnl', 0):,.0f}",
                )
            console.print(whale_table)

    # Tracked Bots
    bot_file = DATA_DIR / "tracked_bots.json"
    if bot_file.exists():
        with open(bot_file) as f:
            bots = json.load(f)
        if bots:
            bot_table = Table(title=f"Tracked Bots ({len(bots)})", border_style="yellow")
            bot_table.add_column("Label")
            bot_table.add_column("Address", max_width=15)
            bot_table.add_column("Bot Score", justify="right")
            bot_table.add_column("Win Rate", justify="right")

            for addr, meta in list(bots.items())[:10]:
                bot_table.add_row(
                    meta.get("label", "?"),
                    f"{addr[:6]}...{addr[-4:]}",
                    f"{meta.get('bot_score', 0):.2f}",
                    f"{meta.get('win_rate', 0):.0%}",
                )
            console.print(bot_table)

    console.print("\n[dim]Press Ctrl+C to exit[/dim]")


if __name__ == "__main__":
    show_dashboard()
