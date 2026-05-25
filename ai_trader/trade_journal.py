"""交易日志 — MySQL 持久化交割单。

两表：
  - paper_trades：AI 模拟/回测产生的交易
  - real_trades：用户真实买卖记录
"""

from datetime import datetime
from typing import Literal

from ai_trader.database import query, fetchone, execute


def record_trade(code: str, name: str, side: str, price: float,
                 quantity: int, amount: float, notes: str = "",
                 table: Literal["paper", "real"] = "real",
                 user_id: int = 0) -> dict:
    """记录一笔交易到指定表。"""
    if table not in ("paper", "real"):
        raise ValueError(f"table must be 'paper' or 'real', got {table}")
    tbl = f"{table}_trades"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tid = execute(
        f"INSERT INTO {tbl} (user_id, created_at, code, name, side, price, quantity, amount, notes) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (user_id, now, str(code), str(name), side, float(price), int(quantity), float(amount), str(notes)),
    )
    return {"id": tid, "created_at": now, "source": table, "status": "recorded"}


def list_trades(limit: int = 50, table: str = "all", user_id: int = 0) -> list[dict]:
    """查询最近 N 条交易记录。

    table: 'paper' | 'real' | 'all'
    """
    if table == "paper":
        rows = query(
            "SELECT *, 'paper' AS source FROM paper_trades "
            "WHERE user_id=%s ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        )
    elif table == "real":
        rows = query(
            "SELECT *, 'real' AS source FROM real_trades "
            "WHERE user_id=%s ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        )
    else:
        rows = query(
            "SELECT *, 'paper' AS source FROM paper_trades WHERE user_id=%s "
            "UNION ALL "
            "SELECT *, 'real' AS source FROM real_trades WHERE user_id=%s "
            "ORDER BY created_at DESC LIMIT %s",
            (user_id, user_id, limit),
        )
    return [dict(r) for r in rows]


def get_summary(table: str = "all", user_id: int = 0) -> dict:
    """汇总统计。"""
    def _sum(tbl):
        buy = fetchone(
            f"SELECT COALESCE(SUM(amount), 0) AS s FROM {tbl} WHERE user_id=%s AND side='buy'",
            (user_id,),
        )["s"]
        sell = fetchone(
            f"SELECT COALESCE(SUM(amount), 0) AS s FROM {tbl} WHERE user_id=%s AND side='sell'",
            (user_id,),
        )["s"]
        count = fetchone(
            f"SELECT COUNT(*) AS c FROM {tbl} WHERE user_id=%s", (user_id,),
        )["c"]
        return float(buy), float(sell), int(count)

    if table == "paper":
        total_buy, total_sell, count = _sum("paper_trades")
    elif table == "real":
        total_buy, total_sell, count = _sum("real_trades")
    else:
        p_buy, p_sell, p_count = _sum("paper_trades")
        r_buy, r_sell, r_count = _sum("real_trades")
        total_buy = p_buy + r_buy
        total_sell = p_sell + r_sell
        count = p_count + r_count

    return {
        "total_buy_amount": round(total_buy, 2),
        "total_sell_amount": round(total_sell, 2),
        "net": round(total_sell - total_buy, 2),
        "trade_count": count,
    }
