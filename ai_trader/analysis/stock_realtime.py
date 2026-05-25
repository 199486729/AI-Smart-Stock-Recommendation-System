"""个股实时数据 — 为 AI 聊天提供 MCP 工具调用。

使用 akshare stock_zh_a_spot_em 获取全市场实时快照，
支持个股/ETF 查询。数据延迟约 3-5 分钟。
"""

import logging
import numpy as np
import pandas as pd

from .data_fetcher import fetch_stock_history

logger = logging.getLogger("stock_realtime")


def get_stock_realtime(code: str) -> dict:
    """获取某只股票的实时快照 + 近期技术面摘要。

    Args:
        code: 股票代码，如 sh600519 或 600519

    Returns:
        dict with price, change_pct, volume, turnover, ma, trend, etc.
    """
    try:
        import akshare as ak
    except ImportError:
        return {"error": "akshare 未安装，无法获取实时行情"}

    # 标准化代码
    code = str(code).strip()
    if "." in code:
        code = code.split(".")[-1]

    try:
        df = ak.stock_zh_a_spot_em()
    except Exception as e:
        return {"error": f"获取全市场行情失败: {e}"}

    code_col = None
    for col in ["代码", "code"]:
        if col in df.columns:
            code_col = col
            break
    if code_col is None:
        return {"error": "无法解析行情数据格式"}

    row = df[df[code_col] == code]
    if row.empty:
        return {"error": f"未找到股票 {code}，请检查代码是否正确（如 600519）"}

    r = row.iloc[0]

    name = _safe_str(r, ["名称", "name"])
    price = _safe_float(r, ["最新价", "price", "c"])
    change_pct = _safe_float(r, ["涨跌幅", "change_pct", "pct_chg"])
    change_amt = _safe_float(r, ["涨跌额", "change", "change_amount"])
    volume = _safe_float(r, ["成交量", "volume", "vol"])
    amount = _safe_float(r, ["成交额", "amount", "turnover"])
    turnover = _safe_float(r, ["换手率", "turnover_rate", "turn"])
    high = _safe_float(r, ["最高", "high", "h"])
    low = _safe_float(r, ["最低", "low", "l"])
    open_ = _safe_float(r, ["今开", "open", "o"])
    pre_close = _safe_float(r, ["昨收", "pre_close", "preclose"])
    pe = _safe_float(r, ["市盈率-动态", "pe", "peTTM"])
    total_mv = _safe_float(r, ["总市值", "total_mv", "market_cap"])

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
        "volume_hands": int(volume) if volume else None,
        "amount_yuan": round(amount, 2) if amount else None,
        "turnover_rate": round(turnover, 2) if turnover is not None else None,
        "pe": round(pe, 2) if pe is not None else None,
        "total_mv": round(total_mv / 1e8, 2) if total_mv and total_mv > 0 else None,  # 亿
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
