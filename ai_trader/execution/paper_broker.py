"""PaperBroker — simulated execution with realistic position sizing.

Tracks balance, positions, and P&L.  Uses config-driven commission and
stamp tax rates rather than hardcoded ±$100 adjustments.
"""

from typing import Dict, Optional

from ai_trader.config import config


class PaperBroker:
    """Simulated broker with config-driven fees and position tracking.

    Set journal=True to persist trades to paper_trades SQLite table.
    """

    def __init__(self, initial_balance: Optional[float] = None, journal: bool = False):
        self.balance = initial_balance or config.trading.initial_balance
        self.position: Optional[Dict] = None  # {symbol, quantity, avg_price}
        self.trade_log: list = []
        self._journal = journal

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, signal: Dict) -> Dict:
        """
        Execute a trade signal against the paper account.

        Returns a trade confirmation dict.
        """
        action = signal["action"]
        price = signal["price"]
        confidence = signal["confidence"]

        if action == "BUY":
            return self._open_long(price, signal)
        elif action == "SELL":
            return self._close_long(price, signal)
        else:
            return {"status": "SKIPPED", "reason": "HOLD signal"}

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _open_long(self, price: float, signal: Dict) -> Dict:
        if self.position is not None:
            return {"status": "SKIPPED", "reason": "Already holding a position"}

        qty = self._calc_position_size(price)
        if qty <= 0:
            return {"status": "SKIPPED", "reason": "Insufficient balance"}

        cost = qty * price
        commission = cost * config.trading.commission_rate
        total_cost = cost + commission

        if total_cost > self.balance:
            return {"status": "SKIPPED", "reason": "Insufficient balance after fees"}

        self.balance -= total_cost
        self.position = {
            "symbol": signal.get("symbol", "XAUUSD"),
            "quantity": qty,
            "avg_price": price,
            "cost_basis": total_cost,  # includes commission
        }
        confirm = {
            "status": "FILLED",
            "action": "BUY",
            "quantity": qty,
            "price": price,
            "commission": round(commission, 2),
            "cost": round(total_cost, 2),
            "balance": round(self.balance, 2),
            "confidence": signal["confidence"],
            "reason": signal.get("reason", ""),
        }
        self.trade_log.append(confirm)
        self._print_trade(confirm)
        self._maybe_journal(confirm)
        return confirm

    def _close_long(self, price: float, signal: Dict) -> Dict:
        if self.position is None:
            return {"status": "SKIPPED", "reason": "No position to close"}

        pos = self.position
        qty = pos["quantity"]
        revenue = qty * price
        commission = revenue * config.trading.commission_rate
        stamp_tax = revenue * config.trading.stamp_tax_rate
        net_revenue = revenue - commission - stamp_tax

        pnl = net_revenue - pos["cost_basis"]
        self.balance += net_revenue

        confirm = {
            "status": "FILLED",
            "action": "SELL",
            "symbol": pos["symbol"],
            "quantity": qty,
            "price": price,
            "entry_price": pos["avg_price"],
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / pos["cost_basis"] * 100, 2),
            "commission": round(commission, 2),
            "stamp_tax": round(stamp_tax, 2),
            "balance": round(self.balance, 2),
            "confidence": signal["confidence"],
            "reason": signal.get("reason", ""),
        }
        self.position = None
        self.trade_log.append(confirm)
        self._print_trade(confirm)
        self._maybe_journal(confirm)
        return confirm

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_journal(self, confirm: Dict):
        """Persist to paper_trades if journal mode is enabled."""
        if not self._journal:
            return
        try:
            from ai_trader.trade_journal import record_trade
            side = "buy" if confirm["action"] == "BUY" else "sell"
            symbol = confirm.get("symbol", "")
            qty = confirm["quantity"]
            price = confirm["price"]
            record_trade(
                code=symbol,
                name=symbol,
                side=side,
                price=price,
                quantity=qty,
                amount=confirm.get("cost", qty * price) if side == "buy" else qty * price,
                notes=confirm.get("reason", ""),
                table="paper",
            )
        except Exception:
            pass  # journal failure must not break execution

    def _calc_position_size(self, price: float) -> int:
        """Calculate position size respecting max_position_ratio."""
        max_capital = self.balance * config.trading.max_position_ratio
        if max_capital < config.trading.min_trade_amount:
            return 0
        qty = int(max_capital / price)
        return max(qty, 0)

    def _print_trade(self, confirm: Dict):
        status = confirm["status"]
        action = confirm["action"]
        price = confirm["price"]
        bal = confirm["balance"]
        if confirm.get("pnl") is not None:
            print(
                f"[PAPER] {status} {action} @ {price} | "
                f"PnL: {confirm['pnl']:+} ({confirm['pnl_pct']}%) | "
                f"Balance: {bal}"
            )
        else:
            qty = confirm["quantity"]
            print(
                f"[PAPER] {status} {action} {qty} @ {price} | "
                f"Balance: {bal}"
            )

    def get_summary(self) -> Dict:
        """Return current account summary."""
        pos_val = 0.0
        if self.position:
            pos_val = self.position["quantity"] * self.position["avg_price"]
        return {
            "balance": round(self.balance, 2),
            "position": self.position,
            "position_value": round(pos_val, 2),
            "equity": round(self.balance + pos_val, 2),
            "total_trades": len(self.trade_log),
        }
