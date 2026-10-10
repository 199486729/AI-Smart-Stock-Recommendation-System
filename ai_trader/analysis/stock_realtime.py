"""个股实时数据 — 为 AI 聊天提供 MCP 工具调用。

使用 AkShare 新浪全市场快照，并通过服务器共享缓存控制请求频率。
"""

import logging
import numpy as np
import pandas as pd

from .data_fetcher import fetch_stock_history, fetch_realtime_quotes

logger = logging.getLogger("stock_realtime")


def get_stock_realtime(code: str) -> dict:
    """获取某只股票的实时快照 + 近期技术面摘要。

    Args:
        code: 股票代码，如 sh600519 或 600519

    Returns:
        dict with price, change_pct, volume, turnover, ma, trend, etc.
    """
    # 标准化代码
    code = str(code).strip()
    if "." in code:
        parts = code.split(".")
        code = parts[0] if parts[0].isdigit() else parts[-1]
    code = code.lower().removeprefix("sh").removeprefix("sz").removeprefix("bj")

    quote = fetch_realtime_quotes([code]).get(code)
    if not quote:
        return {"error": "新浪实时行情不可用，已停止使用旧价格分析"}
    if not quote.get("is_fresh"):
        return {
            "error": "实时行情已过期，已停止使用旧价格分析",
            "quote_time": quote.get("quote_time"),
            "market_status": quote.get("market_status"),
        }
    name = quote.get("name")
    price = quote["price"]
    pre_close = quote.get("pre_close")
    change_amt = quote.get("change")
    change_pct = quote.get("change_pct")
    volume = quote.get("volume")
    amount = quote.get("amount")
    high = quote.get("high")
    low = quote.get("low")
    open_ = quote.get("open")

    # K线技术面摘要
    tech = {}
    try:
        hist = fetch_stock_history(code, days=60)
        if hist is not None and not hist.empty:
            close_col = "收盘" if "收盘" in hist.columns else ("close" if "close" in hist.columns else None)
            if close_col:
                close = pd.to_numeric(hist[close_col], errors="coerce").dropna().values
                if len(close) >= 20:
                    tech["ma5"] = round(float(pd.Series(close).rolling(5).mean().iloc[-1]), 2)
                    tech["ma20"] = round(float(pd.Series(close).rolling(20).mean().iloc[-1]), 2)
                    tech["ma60"] = round(float(pd.Series(close).rolling(min(60, len(close))).mean().iloc[-1]), 2) if len(close) >= 60 else tech["ma20"]
                    tech["high_20d"] = round(float(np.max(close[-20:])), 2)
                    tech["low_20d"] = round(float(np.min(close[-20:])), 2)

                    if price and price > 0 and tech["high_20d"] > 0:
                        tech["dist_to_high_pct"] = round((price - tech["high_20d"]) / tech["high_20d"] * 100, 2)
                    if price and tech.get("ma20"):
                        tech["dist_to_ma20_pct"] = round((price - tech["ma20"]) / tech["ma20"] * 100, 2)

                    # 趋势判断
                    if price and tech.get("ma5") and tech.get("ma20"):
                        if price > tech["ma5"] > tech["ma20"]:
                            tech["trend"] = "多头排列，短期强势"
                        elif price > tech["ma20"]:
                            tech["trend"] = "站在MA20上方，中期偏多"
                        elif price < tech["ma20"]:
                            tech["trend"] = "在MA20下方运行，偏弱"
                        else:
                            tech["trend"] = "均线交织，震荡格局"

                    # 连涨连跌
                    consec = 0
                    for i in range(len(close) - 1, 0, -1):
                        if close[i] > close[i - 1]:
                            consec += 1
                        else:
                            break
                    tech["consecutive_up"] = consec

                    consec_d = 0
                    for i in range(len(close) - 1, 0, -1):
                        if close[i] < close[i - 1]:
                            consec_d += 1
                        else:
                            break
                    tech["consecutive_down"] = consec_d
    except Exception:
        pass

    result = {
        "code": code,
        "name": str(name) if name else "",
        "price": round(price, 2) if price else None,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "change_amt": round(change_amt, 2) if change_amt is not None else None,
        "open": round(open_, 2) if open_ else None,
        "high": round(high, 2) if high else None,
        "low": round(low, 2) if low else None,
        "pre_close": round(pre_close, 2) if pre_close else None,
        "volume_hands": int(volume / 100) if volume else None,
        "amount_yuan": round(amount, 2) if amount else None,
        "turnover_rate": None,
        "pe": None,
        "total_mv": None,
        "quote_time": quote["quote_time"],
        "quote_age_seconds": quote["age_seconds"],
        "quote_source": quote["source"],
        "market_status": quote["market_status"],
        "session_quote": {
            "open": round(quote["open"], 2),
            "high": round(quote["high"], 2),
            "low": round(quote["low"], 2),
            "close": round(quote["price"], 2),
            "volume": quote["volume"],
            "amount": quote["amount"],
        },
    }
    if tech:
        result["technical"] = tech

    return result


def _safe_float(row, candidates: list[str]) -> float | None:
    for c in candidates:
        v = row.get(c)
        if v is not None and str(v) not in ("", "nan", "-"):
            try:
                f = float(v)
                if not np.isnan(f) and not np.isinf(f):
                    return f
            except (ValueError, TypeError):
                pass
    return None


def _safe_str(row, candidates: list[str]) -> str | None:
    for c in candidates:
        v = row.get(c)
        if v is not None and str(v) not in ("", "nan"):
            return str(v)
    return None
