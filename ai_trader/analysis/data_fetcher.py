"""统一数据拉取层 — 所有 akshare API 调用封装，独立容错。

每个函数独立 try/except，单个 API 失败不影响整体分析。
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("data_fetcher")
_REALTIME_SOURCE = "新浪实时行情 via AkShare"
_SINA_SPOT_CACHE_SECONDS = 60
_last_realtime_status = {"ok": False, "source": _REALTIME_SOURCE, "quote_count": 0}
_sina_spot_cache: dict[str, dict] = {}
_sina_spot_cache_at = 0.0
_sina_spot_lock = threading.Lock()
_sina_etf_cache: dict[str, dict] = {}
_sina_etf_cache_at = 0.0
_sina_etf_lock = threading.Lock()


def get_realtime_status() -> dict:
    return dict(_last_realtime_status)


def _to_ts_code(symbol: str) -> str:
    code = str(symbol).strip().upper()
    if "." in code:
        left, right = code.split(".", 1)
        if left.isdigit():
            return f"{left}.{right}"
    code = code.removeprefix("SH").removeprefix("SZ").removeprefix("BJ")
    if code.startswith(("4", "8")):
        suffix = "BJ"
    elif code.startswith(("5", "6", "9")):
        suffix = "SH"
    else:
        suffix = "SZ"
    return f"{code}.{suffix}"


def _market_status(now: datetime) -> str:
    if now.weekday() >= 5:
        return "closed"
    hm = now.hour * 60 + now.minute
    if 570 <= hm <= 690 or 780 <= hm <= 900:
        return "trading"
    if 690 < hm < 780:
        return "lunch_break"
    return "closed"


def _normalize_stock_code(symbol: str) -> str:
    code = str(symbol).strip().upper()
    if "." in code:
        parts = code.split(".")
        code = parts[0] if parts[0].isdigit() else parts[-1]
    return code.removeprefix("SH").removeprefix("SZ").removeprefix("BJ")


def _load_sina_spot_cache() -> tuple[dict[str, dict], float]:
    """全站最多每60秒请求一次新浪全市场快照。"""
    global _sina_spot_cache, _sina_spot_cache_at
    cache_age = time.monotonic() - _sina_spot_cache_at
    if _sina_spot_cache and cache_age < _SINA_SPOT_CACHE_SECONDS:
        return _sina_spot_cache, cache_age

    acquired = _sina_spot_lock.acquire(blocking=False)
    if not acquired:
        if _sina_spot_cache:
            return _sina_spot_cache, cache_age
        with _sina_spot_lock:
            return _sina_spot_cache, time.monotonic() - _sina_spot_cache_at

    try:
        cache_age = time.monotonic() - _sina_spot_cache_at
        if _sina_spot_cache and cache_age < _SINA_SPOT_CACHE_SECONDS:
            return _sina_spot_cache, cache_age

        import akshare as ak
        df = ak.stock_zh_a_spot()
        if df is None or df.empty:
            raise RuntimeError("新浪全市场行情返回空数据")

        refreshed = {}
        for _, row in df.iterrows():
            code = _normalize_stock_code(row.get("代码", ""))
            if code:
                refreshed[code] = row.to_dict()
        _sina_spot_cache = refreshed
        _sina_spot_cache_at = time.monotonic()
        return _sina_spot_cache, 0.0
    except Exception as exc:
        logger.warning("新浪实时行情失败: %s", exc)
        return _sina_spot_cache, time.monotonic() - _sina_spot_cache_at
    finally:
        _sina_spot_lock.release()


def _load_sina_etf_cache() -> tuple[dict[str, dict], float]:
    """全站最多每60秒请求一次新浪 ETF 快照。"""
    global _sina_etf_cache, _sina_etf_cache_at
    cache_age = time.monotonic() - _sina_etf_cache_at
    if _sina_etf_cache and cache_age < _SINA_SPOT_CACHE_SECONDS:
        return _sina_etf_cache, cache_age
    with _sina_etf_lock:
        cache_age = time.monotonic() - _sina_etf_cache_at
        if _sina_etf_cache and cache_age < _SINA_SPOT_CACHE_SECONDS:
            return _sina_etf_cache, cache_age
        try:
            import akshare as ak
            df = ak.fund_etf_category_sina(symbol="ETF基金")
            if df is None or df.empty:
                raise RuntimeError("新浪 ETF 行情返回空数据")
            fetched_at = datetime.now()
            refreshed = {}
            for _, source_row in df.iterrows():
                row = source_row.to_dict()
                code = _normalize_stock_code(row.get("代码", ""))
                if code:
                    row["时间戳"] = fetched_at.strftime("%H:%M:%S")
                    refreshed[code] = row
            _sina_etf_cache = refreshed
            _sina_etf_cache_at = time.monotonic()
            return _sina_etf_cache, 0.0
        except Exception as exc:
            logger.warning("新浪 ETF 实时行情失败: %s", exc)
            return _sina_etf_cache, time.monotonic() - _sina_etf_cache_at


def _sina_quote_datetime(value, now: datetime) -> datetime | None:
    try:
        parsed = datetime.strptime(str(value), "%H:%M:%S").time()
        quote_time = datetime.combine(now.date(), parsed)
        if quote_time > now + timedelta(minutes=1):
            quote_time -= timedelta(days=1)
        return quote_time
    except Exception:
        return None


def fetch_realtime_quotes(symbols: list[str], max_age_seconds: int = 120) -> dict[str, dict]:
    """从60秒共享的新浪全市场快照中提取指定股票/ETF。"""
    global _last_realtime_status
    if not symbols:
        return {}

    data, stock_cache_age = _load_sina_spot_cache()
    requested_codes = [_normalize_stock_code(symbol) for symbol in symbols]
    missing_codes = [code for code in requested_codes if code not in data]
    etf_data, etf_cache_age = ({}, 0.0)
    if missing_codes:
        etf_data, etf_cache_age = _load_sina_etf_cache()
    now = datetime.now()
    status = _market_status(now)
    result = {}
    for symbol in symbols:
        code = _normalize_stock_code(symbol)
        row = data.get(code) or etf_data.get(code)
        if not row:
            continue
        cache_age = etf_cache_age if code in etf_data and code not in data else stock_cache_age
        quote_time = _sina_quote_datetime(row.get("时间戳"), now)
        if quote_time is None:
            continue
        age_seconds = max(0, int((now - quote_time).total_seconds()))

        if status == "trading":
            is_fresh = age_seconds <= max_age_seconds
        elif status == "lunch_break":
            is_fresh = quote_time.date() == now.date() and quote_time.hour == 11 and quote_time.minute >= 29
        else:
            after_close = now.weekday() < 5 and (now.hour * 60 + now.minute) > 900
            is_fresh = after_close and quote_time.date() == now.date() and quote_time.hour >= 15

        def number(key, default=0.0):
            value = pd.to_numeric(row.get(key), errors="coerce")
            return default if pd.isna(value) else float(value)

        result[code] = {
            "code": code,
            "name": str(row.get("名称", "")),
            "price": number("最新价"),
            "pre_close": number("昨收"),
            "change": number("涨跌额"),
            "change_pct": number("涨跌幅"),
            "open": number("今开"),
            "high": number("最高"),
            "low": number("最低"),
            "volume": number("成交量"),
            "amount": number("成交额"),
            "quote_time": quote_time.strftime("%Y-%m-%d %H:%M:%S"),
            "age_seconds": age_seconds,
            "cache_age_seconds": round(cache_age, 1),
            "market_status": status,
            "is_fresh": is_fresh,
            "source": _REALTIME_SOURCE,
        }

    fresh_count = sum(1 for q in result.values() if q["is_fresh"])
    _last_realtime_status = {
        "ok": fresh_count > 0,
        "source": _REALTIME_SOURCE,
        "quote_count": len(result),
        "fresh_count": fresh_count,
        "cache_age_seconds": round(cache_age, 1),
        "checked_at": now.isoformat(timespec="seconds"),
    }
    return result


def fetch_stock_reference(symbol: str) -> dict:
    """用 Tushare 补充股票名称及当前交易日的昨收基准。"""
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        return {}
    try:
        import tushare as ts
        pro = ts.pro_api(token)
        ts_code = _to_ts_code(symbol)
        result = {}

        basic = pro.stock_basic(ts_code=ts_code, fields="ts_code,name")
        if basic is not None and not basic.empty:
            result["name"] = str(basic.iloc[0]["name"])

        now = datetime.now()
        daily = pro.daily(
            ts_code=ts_code,
            start_date=(now - timedelta(days=15)).strftime("%Y%m%d"),
            end_date=now.strftime("%Y%m%d"),
        )
        if daily is not None and not daily.empty:
            row = daily.sort_values("trade_date", ascending=False).iloc[0]
            if str(row["trade_date"]) == now.strftime("%Y%m%d"):
                result["pre_close"] = float(row["pre_close"])
            else:
                result["pre_close"] = float(row["close"])
            result["reference_trade_date"] = str(row["trade_date"])
        return result
    except Exception as exc:
        logger.warning("Tushare股票基础数据失败: %s", exc)
        return {}

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
    """新浪实时指数作为当前点位，日线仅用于 MA 与趋势计算。"""
    import akshare as ak
    result = {}
    errors = []
    spot = _safe_call("新浪实时指数", ak.stock_zh_index_spot_sina)
    spot_by_code = {}
    if spot is not None and not spot.empty and "代码" in spot.columns:
        spot_by_code = {str(row["代码"]): row for _, row in spot.iterrows()}

    for name, symbol in KEY_INDICES.items():
        df = _safe_call(f"指数-{name}", ak.stock_zh_index_daily, symbol)
        if df is None:
            errors.append(name)
            continue
        close = df["close"].values.astype(float)
        vol = df["volume"].values.astype(float)
        mas = _compute_mas(close)

        realtime = spot_by_code.get(symbol)
        realtime_price = float(realtime["最新价"]) if realtime is not None else None
        price = realtime_price if realtime_price and realtime_price > 0 else float(close[-1])
        if realtime is not None:
            change_pct = round(float(realtime["涨跌幅"]), 2)
        elif len(close) >= 2:
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
            dist_to_high_20d = round((price - high_20d) / high_20d * 100, 2)
        else:
            dist_to_high_20d = None

        result[name] = {
            "symbol": symbol,
            "price": round(price, 2),
            "change_pct": change_pct,
            "change_5d": change_5d,
            "ma20": round(mas["ma20"], 2) if mas["ma20"] else None,
            "ma60": round(mas["ma60"], 2) if mas["ma60"] else None,
            "above_ma20": (price > mas["ma20"]) if mas.get("ma20") else None,
            "above_ma60": (price > mas["ma60"]) if mas.get("ma60") else None,
            "dist_to_high_20d": dist_to_high_20d,
            "volume": int(realtime["成交量"]) if realtime is not None else (int(vol[-1]) if len(vol) > 0 else 0),
            "quote_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if realtime is not None else None,
            "quote_source": "新浪实时指数" if realtime is not None else "指数日线降级",
            "is_realtime": realtime is not None,
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
    """拉取个股日线历史（前复权）。Tushare 优先，失败时降级。"""
    import akshare as ak
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")

    token = os.environ.get("TUSHARE_TOKEN", "")
    if token:
        try:
            import tushare as ts
            pro = ts.pro_api(token)
            df = ts.pro_bar(
                ts_code=_to_ts_code(symbol),
                api=pro,
                adj="qfq",
                start_date=start,
                end_date=end,
            )
            if df is not None and not df.empty:
                return df.sort_values("trade_date").reset_index(drop=True)
        except Exception as exc:
            logger.warning("Tushare个股日线失败 %s: %s", symbol, exc)

    # Tushare 暂时不可用时再尝试 AkShare。
    df = _safe_call_no_retry(f"个股K线-{symbol}", ak.stock_zh_a_hist, symbol, "daily", start, end, "qfq")
    if df is not None and not df.empty:
        return df

    # 降级到 baostock（持久会话，无 login/logout 开销）
    return _bs_fetch_history(symbol, start, end)


def fetch_etf_history(symbol: str, days: int = 120) -> pd.DataFrame | None:
    """拉取 ETF 日线历史。Tushare 优先，失败时降级到 AkShare。"""
    import akshare as ak
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")
    token = os.environ.get("TUSHARE_TOKEN", "")
    if token:
        try:
            import tushare as ts
            pro = ts.pro_api(token)
            df = pro.fund_daily(
                ts_code=_to_ts_code(symbol),
                start_date=start,
                end_date=end,
            )
            if df is not None and not df.empty:
                return df.sort_values("trade_date").reset_index(drop=True)
        except Exception as exc:
            logger.warning("Tushare ETF日线失败 %s: %s", symbol, exc)
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
