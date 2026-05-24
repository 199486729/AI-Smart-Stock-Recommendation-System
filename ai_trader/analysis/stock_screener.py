"""个股筛选 — 在强势板块中精选龙头个股。

筛选链：
1. 取 Top N 板块的成分股
2. 用实时行情快速初筛（涨跌幅、换手率、成交额）
3. 对候选股拉取 K 线做深度技术分析
4. 按趋势质量 + 回踩位置 + 量能评分排序
5. 输出个股推荐（含入场、止损、仓位）
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .data_fetcher import fetch_stock_history

logger = logging.getLogger("stock_screener")

# 需要过滤的个股类型
_EXCLUDE_STOCK_TYPES = ["ST", "*ST", "退市", "N "]

# 技术面参数
_TREND_LOOKBACK = 60
_PULLBACK_MIN = -8.0            # 距20日高点回撤不低于 -8%
_PULLBACK_MAX = -0.5            # 距20日高点回撤不超过 -0.5%
_MAX_DIST_TO_MA20 = 6.0         # 距 MA20 不超过 6%
_STOP_LOSS_PCT = 0.065          # 固定 6.5% 止损


def screen_stocks(
    top_sectors: list[dict],
    leading_stocks: list[dict] | None = None,
    limit_up_pool: pd.DataFrame | None = None,
    stock_spot: pd.DataFrame | None = None,
    max_deep_checks: int = 12,
    max_final: int = 10,
) -> list[dict]:
    """主入口：从强势板块中筛选龙头个股。

    策略：不依赖被限流的成分股 API，而是从三个来源聚合候选股：
    1. 板块异动最频繁个股（来自 stock_board_change_em）
    2. 当日涨停池中属于强势板块行业的个股（来自 stock_zt_pool_em）
    3. 实时行情（spot）中按板块关键词匹配（降级用）

    Args:
        top_sectors: sector_ranker 输出的板块排名
        leading_stocks: 板块异动最频繁个股列表
        limit_up_pool: 涨停池 DataFrame
        stock_spot: 全市场个股实时行情（可选降级）
        max_deep_checks: 最多做深度分析的候选数
        max_final: 最终输出最多几只
    """
    candidates_seeds: list[dict] = []
    # 全板块信息（用于查找 chase_level）
    sector_info = {s["name"]: s for s in top_sectors}
    # 强势板块名集合（用于优先排序，不做硬过滤）
    strong_sectors = {s["name"] for s in top_sectors[:30] if s.get("chase_level") != "danger"}

    # —— 来源 1：全市场板块异动最频繁个股（452只，不限于头部板块）——
    if leading_stocks:
        for ls in leading_stocks:
            sec_info = sector_info.get(ls["sector"], {})
            candidates_seeds.append({
                "code": ls["code"], "name": ls["name"],
                "sector": ls["sector"],
                "sector_rank": ls.get("sector_rank", 99),
                "sector_score": ls.get("sector_score", 0),
                "sector_chase_level": sec_info.get("chase_level", "normal"),
                "is_strong_sector": ls["sector"] in strong_sectors,
                "source": "板块异动龙头",
            })

    # —— 来源 2：涨停池个股（按行业关键词匹配强势板块）——
    if limit_up_pool is not None and not limit_up_pool.empty:
        zt_code_col = _find_col(limit_up_pool, ["代码", "code"])
        zt_name_col = _find_col(limit_up_pool, ["名称", "name"])
        zt_industry_col = _find_col(limit_up_pool, ["所属行业", "行业", "industry"])
        zt_pct_col = _find_col(limit_up_pool, ["涨跌幅", "change_pct"])
        zt_price_col = _find_col(limit_up_pool, ["最新价", "price"])
        zt_turnover_col = _find_col(limit_up_pool, ["换手率", "turnover_rate"])

        if zt_code_col and zt_name_col:
            for _, row in limit_up_pool.iterrows():
                industry = str(row.get(zt_industry_col, "")) if zt_industry_col else ""
                code = str(row.get(zt_code_col, ""))
                name = str(row.get(zt_name_col, ""))
                matched_sector = _match_industry_to_sector(industry, strong_sectors)
                if matched_sector:
                    pct = _safe_float(row.get(zt_pct_col)) if zt_pct_col else None
                    price = _safe_float(row.get(zt_price_col)) if zt_price_col else None
                    turnover = _safe_float(row.get(zt_turnover_col)) if zt_turnover_col else None
                    candidates_seeds.append({
                        "code": code, "name": name,
                        "sector": matched_sector,
                        "sector_rank": sector_info.get(matched_sector, {}).get("rank", 99),
                        "sector_score": sector_info.get(matched_sector, {}).get("score", 0),
                        "sector_chase_level": sector_info.get(matched_sector, {}).get("chase_level", "normal"),
                        "current_price": round(price, 2) if price else 0,
                        "change_pct": round(pct, 2) if pct else None,
                        "turnover_rate": round(turnover, 2) if turnover else None,
                        "amount": 0,
                        "is_strong_sector": True,
                        "source": "涨停池",
                    })

    # 去重
    seen_codes = set()
    unique_seeds = []
    for s in candidates_seeds:
        if s["code"] not in seen_codes and not _is_excluded(s.get("name", "")):
            seen_codes.add(s["code"])
            unique_seeds.append(s)

    if not unique_seeds:
        logger.warning("无候选个股（龙头股和涨停池均无匹配）")
        return []

    logger.info("聚合候选股 %d 只，开始深度K线分析", len(unique_seeds))

    # 排序：强势板块优先，然后按板块排名
    unique_seeds.sort(key=lambda x: (
        not x.get("is_strong_sector", False),  # 强势板块优先
        x.get("sector_rank", 999),
    ))

    # —— 并行 K 线深度分析 ——
    deep_checks = unique_seeds[:max_deep_checks]
    candidates = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {
            executor.submit(_deep_analyze, s["code"], s.get("current_price", 0)): s
            for s in deep_checks
        }
        for future in as_completed(future_map, timeout=90):
            stock = future_map[future]
            try:
                analysis = future.result(timeout=25)
                if analysis.get("pass"):
                    candidates.append({**stock, **analysis})
            except Exception:
                pass

    # 按质量分排名
    candidates.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
    return candidates[:max_final]


def _is_excluded(name: str) -> bool:
    # N开头 = 新股首日，无K线历史，排除
    if name.startswith("N"):
        return True
    for kw in _EXCLUDE_STOCK_TYPES:
        if kw in name:
            return True
    return False


def _match_industry_to_sector(industry: str, sector_names: set[str]) -> str | None:
    """将涨停池个股的所属行业匹配到我们的强势板块名。

    匹配策略：
    1. 行业名直接包含板块名（或反之）
    2. 关键词映射表
    """
    if not industry:
        return None

    # 直接匹配：行业名和板块名互相包含
    for sname in sector_names:
        if sname in industry or industry in sname:
            return sname
        # 拆词匹配（2字片段）
        for i in range(len(sname) - 1):
            chunk = sname[i:i+2]
            if len(chunk) >= 2 and chunk in industry:
                return sname

    # 关键词映射
    _INDUSTRY_TO_SECTOR = {
        "半导体": ["半导体", "芯片", "集成电路"],
        "软件": ["人工智能", "AI", "计算机", "软件"],
        "IT": ["人工智能", "AI", "计算机"],
        "通信": ["CPO", "光通信", "光模块", "5G"],
        "医疗": ["医药", "创新药", "生物医药", "医疗"],
        "医药": ["医药", "创新药", "生物医药"],
        "新能源": ["新能源", "光伏", "电池", "锂电"],
        "汽车": ["汽车", "智能驾驶", "新能源车"],
        "电子": ["消费电子", "电子", "半导体", "芯片"],
        "机械": ["机器人", "智能制造", "机械"],
        "电力": ["电力", "公用事业"],
        "化工": ["化工", "新材料"],
        "金融": ["证券", "银行", "保险"],
        "消费": ["消费", "白酒", "食品"],
    }

    for ind_key, sector_keywords in _INDUSTRY_TO_SECTOR.items():
        if ind_key in industry:
            for skw in sector_keywords:
                if skw in sector_names:
                    return skw

    return None


# ===========================================================================
# 深度技术分析
# ===========================================================================

def _deep_analyze(code: str, current_price: float) -> dict:
    """拉取 K 线，做完整技术面筛选。"""
    df = fetch_stock_history(code, days=_TREND_LOOKBACK + 20)
    if df is None or df.empty:
        return {"pass": False}

    close_col = _find_col(df, ["收盘", "close"])
    if close_col is None:
        return {"pass": False}

    close = pd.to_numeric(df[close_col], errors="coerce").dropna().values
    if len(close) < 20:
        return {"pass": False}

    price = current_price if current_price > 0 else float(close[-1])

    # ——— 均线 ———
    ma20 = float(pd.Series(close).rolling(20).mean().iloc[-1])
    ma60 = float(pd.Series(close).rolling(min(60, len(close))).mean().iloc[-1]) if len(close) >= 60 else ma20
    ma5 = float(pd.Series(close).rolling(5).mean().iloc[-1]) if len(close) >= 5 else ma20

    # ——— 趋势条件 ———
    # 严格：价格 > MA5 > MA20
    uptrend = price > ma5 > ma20
    structural_ok = price > ma20 and ma20 > ma60

    if not uptrend and not structural_ok:
        return {"pass": False, "reason": "趋势不满足"}

    # ——— 回踩幅度 ———
    high20 = float(np.max(close[-20:]))
    dist_to_high = (price - high20) / high20 * 100

    # 如果涨超20日高点（创新高），也不算回踩，是突破
    if dist_to_high > 2:
        return {"pass": False, "reason": "已创新高，不是回踩买点"}

    # 距高点太近（没回踩）或太远（破位）
    if dist_to_high > _PULLBACK_MAX:
        return {"pass": False, "reason": f"距高点仅{abs(dist_to_high):.1f}%，尚未回踩"}

    # ——— 距均线 ———
    dist_to_ma20 = (price - ma20) / ma20 * 100
    if dist_to_ma20 > _MAX_DIST_TO_MA20:
        return {"pass": False, "reason": f"远离MA20 {dist_to_ma20:.1f}%"}
    if dist_to_ma20 < -3:
        return {"pass": False, "reason": "跌破MA20"}

    # ——— 量能 ———
    vol_ok = True
    vol_col = _find_col(df, ["成交量", "volume"])
    if vol_col:
        vol = pd.to_numeric(df[vol_col], errors="coerce").dropna().values
        if len(vol) >= 20:
            vol_5d = float(np.mean(vol[-5:]))
            vol_20d = float(np.mean(vol[-20:]))
            if vol_20d > 0 and vol_5d / vol_20d < 0.3:
                vol_ok = False

    # ——— 连涨/连跌 ———
    consecutive_up = 0
    for i in range(len(close) - 1, 0, -1):
        if close[i] > close[i - 1]:
            consecutive_up += 1
        else:
            break
    consecutive_down = 0
    for i in range(len(close) - 1, 0, -1):
        if close[i] < close[i - 1]:
            consecutive_down += 1
        else:
            break

    # ——— 综合质量分 ———
    quality = _calc_quality(
        dist_to_high=dist_to_high,
        dist_to_ma20=dist_to_ma20,
        vol_ok=vol_ok,
        consecutive_up=consecutive_up,
        consecutive_down=consecutive_down,
        uptrend=uptrend,
        structural_ok=structural_ok,
    )

    return {
        "pass": True,
        "price": round(price, 2),
        "ma5": round(ma5, 2),
        "ma20": round(ma20, 2),
        "ma60": round(ma60, 2),
        "high_20d": round(high20, 2),
        "dist_to_high_pct": round(dist_to_high, 2),
        "dist_to_ma20_pct": round(dist_to_ma20, 2),
        "vol_ok": vol_ok,
        "consecutive_up": consecutive_up,
        "consecutive_down": consecutive_down,
        "uptrend": uptrend,
        "structural_ok": structural_ok,
        "quality_score": round(quality, 2),
        "trend_label": "均线多头" if uptrend else ("多头排列" if structural_ok else "震荡偏强"),
    }


def _calc_quality(
    dist_to_high: float,
    dist_to_ma20: float,
    vol_ok: bool,
    consecutive_up: int,
    consecutive_down: int,
    uptrend: bool,
    structural_ok: bool,
) -> float:
    """质量分：越高越好。"""
    score = 0.0

    # 回踩深度得分（最优 -2% ~ -6%）
    if -6 <= dist_to_high <= -2:
        score += 8.0
    elif -8 <= dist_to_high < -6 or -2 < dist_to_high <= -0.5:
        score += 4.0

    # 距均线得分（最优 0~3%）
    if 0 <= dist_to_ma20 <= 3:
        score += 6.0
    elif 3 < dist_to_ma20 <= 6:
        score += 2.0

    # 趋势得分
    if uptrend:
        score += 8.0
    elif structural_ok:
        score += 4.0

    # 量能
    if vol_ok:
        score += 3.0

    # 连涨天数（1-3天最好，说明刚开始涨；连涨太多=追高）
    if 1 <= consecutive_up <= 3:
        score += 4.0
    elif 4 <= consecutive_up <= 5:
        score += 1.0
    elif consecutive_up >= 6:
        score -= 2.0

    # 连跌后转涨（最佳：跌1-2天后今天涨）
    if consecutive_down in (1, 2) and consecutive_up >= 1:
        score += 4.0

    return score


# ===========================================================================
# 工具
# ===========================================================================

def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _safe_float(val) -> float | None:
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (ValueError, TypeError):
        return None
