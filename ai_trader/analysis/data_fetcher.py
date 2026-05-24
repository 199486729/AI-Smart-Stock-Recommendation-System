"""统一数据拉取层 — 所有 akshare API 调用封装，独立容错。

每个函数独立 try/except，单个 API 失败不影响整体分析。
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("data_fetcher")

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _safe_call(name, fn, *args, **kwargs):
    """安全调用 akshare API，含重试和退避。失败返回 None。"""
    import time as _time
    last_exc = None
    for attempt in range(2):
        try:
            result = fn(*args, **kwargs)
            if result is None:
                logger.warning("%s 返回 None", name)
                return None
            if hasattr(result, "empty") and result.empty:
                logger.warning("%s 返回空数据", name)
                return None
            return result
        except Exception as e:
            last_exc = e
            if attempt < 1:
                wait = 1.5
                logger.info("%s 失败，%.1fs后重试...", name, wait)
                _time.sleep(wait)
    logger.warning("%s 失败: %s", name, last_exc)
    return None


def _safe_call_no_retry(name, fn, *args, **kwargs):
    """安全调用 akshare API，不重试（用于K线等容易被限流的接口）。"""
    try:
        result = fn(*args, **kwargs)
        if result is None:
            return None
        if hasattr(result, "empty") and result.empty:
            return None
        return result
    except Exception:
        return None


def _compute_mas(prices: np.ndarray):
    """计算 5/10/20/60 日均线及相对 MA20/MA60 的位置。"""
    n = len(prices)
    mas = {}
    for w in [5, 10, 20, 60]:
        if n >= w:
            mas[f"ma{w}"] = float(pd.Series(prices).rolling(w).mean().iloc[-1])
        else:
            mas[f"ma{w}"] = None
    current = float(prices[-1])
    mas["price"] = current
    mas["above_ma20"] = (current > mas["ma20"]) if mas.get("ma20") else None
    mas["above_ma60"] = (current > mas["ma60"]) if mas.get("ma60") else None
    mas["ma20_direction"] = "up" if (n >= 2 and mas.get("ma20") and float(pd.Series(prices).rolling(20).mean().iloc[-1]) > float(pd.Series(prices).rolling(20).mean().iloc[-2])) else "down"
    return mas


# ---------------------------------------------------------------------------
# 1. 大盘指数
# ---------------------------------------------------------------------------

KEY_INDICES = {
    "上证综指": "sh000001",
    "深证成指": "sz399001",
    "创业板指": "sz399006",
    "科创50":   "sh000688",
    "沪深300":  "sh000300",
    "中证500":  "sh000905",
}


def fetch_indices() -> dict | None:
    """拉取关键指数日线数据，计算 MA 和趋势。"""
    import akshare as ak
    result = {}
    errors = []
    for name, symbol in KEY_INDICES.items():
        df = _safe_call(f"指数-{name}", ak.stock_zh_index_daily, symbol)
        if df is None:
            errors.append(name)
            continue
        close = df["close"].values.astype(float)
        vol = df["volume"].values.astype(float)
        mas = _compute_mas(close)

        # 涨跌幅
        if len(close) >= 2:
            change_pct = round((close[-1] - close[-2]) / close[-2] * 100, 2)
        else:
            change_pct = 0.0

        # 5日涨跌幅
        if len(close) >= 6:
            change_5d = round((close[-1] - close[-6]) / close[-6] * 100, 2)
        else:
            change_5d = None

        # 20日高点距离
        if len(close) >= 20:
            high_20d = float(np.max(close[-20:]))
            dist_to_high_20d = round((mas["price"] - high_20d) / high_20d * 100, 2)
        else:
            dist_to_high_20d = None

        result[name] = {
            "symbol": symbol,
            "price": round(mas["price"], 2),
            "change_pct": change_pct,
            "change_5d": change_5d,
            "ma20": round(mas["ma20"], 2) if mas["ma20"] else None,
            "ma60": round(mas["ma60"], 2) if mas["ma60"] else None,
            "above_ma20": mas["above_ma20"],
            "above_ma60": mas["above_ma60"],
            "dist_to_high_20d": dist_to_high_20d,
            "volume": int(vol[-1]) if len(vol) > 0 else 0,
        }

    if errors:
        logger.warning("指数数据缺失: %s", errors)
    return result if result else None


# ---------------------------------------------------------------------------
# 2. 市场广度
# ---------------------------------------------------------------------------

def fetch_market_breadth() -> dict | None:
    """涨跌家数、涨停跌停、成交额、总市值。"""
    import akshare as ak
    items = _safe_call("市场广度", ak.stock_market_activity_legu)
    if items is None:
        return None
    try:
        d = {}
        for row in items.itertuples() if hasattr(items, "itertuples") else items.iterrows():
            key = str(row[0]) if hasattr(row, "_fields") else str(row[1].iloc[0] if hasattr(row[1], "iloc") else row[1])
            val = row[1] if hasattr(row, "_fields") and len(row) > 1 else (row[1].iloc[1] if hasattr(row[1], "iloc") and len(row[1]) > 1 else None)
            d[key] = val
        # 直接用序号访问，因为返回格式是特殊的
        if hasattr(items, "iloc"):
            return _parse_breadth_df(items)
        # fallback: 尝试转成 dict
        raw = {}
        for i in range(min(len(items), 20)):
            try:
                k = str(items.iloc[i, 0]) if hasattr(items, "iloc") else ""
                v = items.iloc[i, 1] if hasattr(items, "iloc") else ""
                raw[k] = v
            except Exception:
                pass
        return raw if raw else None
    except Exception as e:
        logger.warning("市场广度解析失败: %s", e)
        return None


def _parse_breadth_df(df) -> dict:
    """解析 stock_market_activity_legu 返回的特殊格式。"""
    try:
        # 这个 API 返回的是类似 Series/DataFrame 的结构，每行是 item: value
        result = {}
        for _, row in df.iterrows():
            key = str(row.iloc[0]) if hasattr(row, "iloc") else str(row[0])
            val = row.iloc[1] if hasattr(row, "iloc") else row[1]
            result[key] = val
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3. 北向资金
# ---------------------------------------------------------------------------

def fetch_north_flow() -> dict | None:
    """北向资金历史流向，提取今日/5日/20日净买额。"""
    import akshare as ak
    df = _safe_call("北向资金", ak.stock_hsgt_hist_em, "北向资金")
    if df is None:
        return None
    try:
        net_col = None
        for c in ["当日成交净买额", "净买额", "当日资金流入"]:
            if c in df.columns:
                net_col = c
                break
        if net_col is None:
            return None

        vals = pd.to_numeric(df[net_col], errors="coerce").dropna()
        if vals.empty:
            return None

        today_net = float(vals.iloc[-1]) if len(vals) > 0 else 0
        net_5d = float(vals.iloc[-5:].sum()) if len(vals) >= 5 else float(vals.sum())
        net_20d = float(vals.iloc[-20:].sum()) if len(vals) >= 20 else float(vals.sum())

        direction = "大幅流入" if net_5d > 100 else ("小幅流入" if net_5d > 0 else ("小幅流出" if net_5d > -100 else "大幅流出"))

        return {
            "net_buy_today": round(today_net, 2),
            "net_buy_5d": round(net_5d, 2),
            "net_buy_20d": round(net_20d, 2),
            "direction": direction,
        }
    except Exception as e:
        logger.warning("北向资金解析失败: %s", e)
        return None


# ---------------------------------------------------------------------------
# 4. 板块数据 (1007 个板块)
# ---------------------------------------------------------------------------

def fetch_all_boards() -> pd.DataFrame | None:
    """拉取全市场板块异动数据。"""
    import akshare as ak
    return _safe_call("板块异动", ak.stock_board_change_em)


def fetch_industry_fund_flow() -> pd.DataFrame | None:
    """拉取行业资金流向。"""
    import akshare as ak
    return _safe_call("行业资金流", ak.stock_fund_flow_industry, "即时")


def fetch_sector_spot() -> pd.DataFrame | None:
    """拉取新浪行业板块行情。"""
    import akshare as ak
    return _safe_call("新浪行业", ak.stock_sector_spot, "新浪行业")


# ---------------------------------------------------------------------------
# 5. ETF 数据
# ---------------------------------------------------------------------------

def fetch_etf_spot() -> pd.DataFrame | None:
    """拉取全市场 ETF 实时行情（1475只，37列）。"""
    import akshare as ak
    return _safe_call("ETF行情", ak.fund_etf_spot_em)


def fetch_etf_category() -> pd.DataFrame | None:
    """拉取 ETF 分类数据。"""
    import akshare as ak
    return _safe_call("ETF分类", ak.fund_etf_category_ths, "全部")


# ---------------------------------------------------------------------------
# 6. 涨停池
# ---------------------------------------------------------------------------

def fetch_limit_up_pool() -> pd.DataFrame | None:
    """拉取当日涨停股票池。"""
    import akshare as ak
    today = datetime.now().strftime("%Y%m%d")
    df = _safe_call("涨停池", ak.stock_zt_pool_em, today)
    # 如果当天没数据（非交易日），往前推
    if df is None:
        for back in range(1, 6):
            d = (datetime.now() - timedelta(days=back)).strftime("%Y%m%d")
            df = _safe_call(f"涨停池-{d}", ak.stock_zt_pool_em, d)
            if df is not None:
                break
    return df


# ---------------------------------------------------------------------------
# 7. 个股融资融券
# ---------------------------------------------------------------------------

def fetch_margin_detail() -> pd.DataFrame | None:
    """拉取融资融券明细。"""
    import akshare as ak
    today = datetime.now().strftime("%Y%m%d")
    df = _safe_call("融资融券", ak.stock_margin_detail_sse, today)
    if df is None:
        # 往前找最近交易日
        for back in range(1, 10):
            d = (datetime.now() - timedelta(days=back)).strftime("%Y%m%d")
            df = _safe_call(f"融资融券-{d}", ak.stock_margin_detail_sse, d)
            if df is not None:
                break
    return df


# ---------------------------------------------------------------------------
# 8. 个股/ETF 历史 K 线
# ---------------------------------------------------------------------------

# ---- baostock session (module-level, lazy init) ----
import threading as _threading

_BS_READY = False
_BS_LOCK = _threading.Lock()

def _ensure_bs_login():
    global _BS_READY
    if not _BS_READY:
        import baostock as bs
        bs.login()
        _BS_READY = True


def _bs_fetch_history(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """用 baostock 拉取个股日线（前复权），线程安全。"""
    try:
        import baostock as bs
        with _BS_LOCK:
            _ensure_bs_login()

            code = str(symbol).zfill(6)
            if code.startswith(("6", "9")):
                bs_symbol = f"sh.{code}"
            else:
                bs_symbol = f"sz.{code}"

            bs_start = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
            bs_end = f"{end[:4]}-{end[4:6]}-{end[6:8]}"

            rs_query = bs.query_history_k_data_plus(
                bs_symbol,
                "date,code,open,high,low,close,volume,amount,turn",
                start_date=bs_start, end_date=bs_end,
                frequency="d", adjustflag="2",
            )
            rows = []
            while rs_query.next():
                rows.append(rs_query.get_row_data())

        if not rows:
            return None

        df_bs = pd.DataFrame(rows, columns=["date", "code", "open", "high", "low", "close", "volume", "amount", "turn"])
        for c in ["open", "high", "low", "close", "volume", "amount", "turn"]:
            df_bs[c] = pd.to_numeric(df_bs[c], errors="coerce")
        df_bs["date"] = pd.to_datetime(df_bs["date"])
        return df_bs
    except Exception:
        return None


def fetch_stock_history(symbol: str, days: int = 120) -> pd.DataFrame | None:
    """拉取个股日线历史（前复权）。akshare 优先，被限流时降级到 baostock。"""
    import akshare as ak
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")

    # 先尝试 akshare
    df = _safe_call_no_retry(f"个股K线-{symbol}", ak.stock_zh_a_hist, symbol, "daily", start, end, "qfq")
    if df is not None and not df.empty:
        return df

    # 降级到 baostock（持久会话，无 login/logout 开销）
    return _bs_fetch_history(symbol, start, end)


def fetch_etf_history(symbol: str, days: int = 120) -> pd.DataFrame | None:
    """拉取 ETF 日线历史。不重试，K线限流时立即降级。"""
    import akshare as ak
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")
    return _safe_call_no_retry(f"ETF K线-{symbol}", ak.fund_etf_hist_em, symbol, "daily", start, end, "")


# ---------------------------------------------------------------------------
# 9. 个股实时行情
# ---------------------------------------------------------------------------

def fetch_stock_spot() -> pd.DataFrame | None:
    """拉取全市场个股实时行情（分沪市+深市）。"""
    import akshare as ak
    sh = _safe_call("沪市个股", ak.stock_sh_a_spot_em)
    sz = _safe_call("深市个股", ak.stock_sz_a_spot_em)

    if sh is not None and sz is not None:
        return pd.concat([sh, sz], ignore_index=True)
    elif sh is not None:
        return sh
    elif sz is not None:
        return sz
    # fallback to combined API
    return _safe_call("全市场个股", ak.stock_zh_a_spot_em)


# ---------------------------------------------------------------------------
# 10. 北向资金个股持仓
# ---------------------------------------------------------------------------

def fetch_north_holdings() -> pd.DataFrame | None:
    """拉取北向资金个股持仓。"""
    import akshare as ak
    return _safe_call("北向个股持仓", ak.stock_hsgt_individual_em, "沪股通")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def fetch_all_data() -> dict:
    """一键拉取全部数据。并行调用所有 akshare API，单个失败不阻塞全局。"""
    t0 = time.time()
    result = {
        "indices": None,
        "breadth": None,
        "north_flow": None,
        "boards": None,
        "industry_flow": None,
        "sectors": None,
        "etfs": None,
        "limit_up": None,
        "stock_spot": None,
        "errors": [],
        "fetch_time": datetime.now().isoformat(),
        "elapsed": 0,
    }

    # 并行拉取所有独立 API（IO 密集型，ThreadPoolExecutor 足够）
    tasks = {
        "indices": fetch_indices,
        "breadth": fetch_market_breadth,
        "north_flow": fetch_north_flow,
        "boards": fetch_all_boards,
        "industry_flow": fetch_industry_fund_flow,
        "sectors": fetch_sector_spot,
        "etfs": fetch_etf_spot,
        "limit_up": fetch_limit_up_pool,
    }

    max_workers = min(4, len(tasks))  # 东方财富反爬，并发太高更容易限流
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(future_map, timeout=45):
            key = future_map[future]
            try:
                result[key] = future.result(timeout=30)
            except Exception:
                result[key] = None
                result["errors"].append(f"{key}拉取超时或异常")

    # 质量检查
    checks = {
        "指数数据": result["indices"],
        "市场广度": result["breadth"],
        "北向资金": result["north_flow"],
        "板块数据": result["boards"],
        "行业资金流": result["industry_flow"],
        "ETF行情": result["etfs"],
        "涨停池": result["limit_up"],
    }
    for name, data in checks.items():
        if data is None:
            result["errors"].append(f"{name}拉取失败")
        elif hasattr(data, "empty") and data.empty:
            result["errors"].append(f"{name}返回空")

    result["elapsed"] = round(time.time() - t0, 2)
    logger.info("数据拉取完成，耗时 %.1fs，%d 个错误", result["elapsed"], len(result["errors"]))
    return result
