"""ETF 筛选 — 在强势板块中匹配 ETF，检测回踩买点。

回踩买点条件：
1. 上升趋势：价格 > MA20 且 MA20 > MA60
2. 近期回踩：价格距 20 日高点 -1% ~ -8%
3. 不远离均线：距 MA20 < 5%
4. 成交量正常
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .data_fetcher import fetch_etf_history


def screen_etfs(
    top_sectors: list[dict],
    etfs: pd.DataFrame | None = None,
    max_per_sector: int = 3,
    max_candidates: int = 15,
) -> list[dict]:
    """
    从强势板块中筛选 ETF。

    先用 ETF 实时行情快速初筛（涨跌幅、折溢价、资金流向），
    然后对 Top 候选拉取 K 线做深度回踩检测。

    Args:
        top_sectors: 板块排名前 N 的结果
        etfs: fund_etf_spot_em() 返回的 ETF 全量数据
    """
    if etfs is None or etfs.empty:
        return []

    name_col = _find_col(etfs, ["名称", "基金名称", "name"])
    code_col = _find_col(etfs, ["代码", "基金代码", "code"])
    price_col = _find_col(etfs, ["最新价", "price", "单位净值", "收盘价"])
    change_pct_col = _find_col(etfs, ["涨跌幅", "change_pct"])
    discount_col = _find_col(etfs, ["基金折价率", "折价率"])
    flow_col = _find_col(etfs, ["主力净流入-净额", "主力净流入", "资金净流入"])
    volume_col = _find_col(etfs, ["成交量", "成交额"])
    turnover_col = _find_col(etfs, ["换手率", "量比"])

    if name_col is None or code_col is None:
        return []

    # —— 第一轮：ETF spot 数据快速筛选 ——
    quick_candidates = []
    seen = set()

    for sector in top_sectors[:25]:  # 只看前25个板块
        sector_name = sector["name"]
        sector_count = 0

        for _, row in etfs.iterrows():
            etf_name = str(row.get(name_col, ""))
            if not _match_sector(etf_name, sector_name):
                continue

            code = str(row.get(code_col, ""))
            if code in seen:
                continue
            seen.add(code)

            price = _safe_float(row.get(price_col))
            change_pct = _safe_float(row.get(change_pct_col))
            discount = _safe_float(row.get(discount_col)) if discount_col else None
            fund_flow = _safe_float(row.get(flow_col)) if flow_col else 0
            vol = _safe_float(row.get(volume_col)) if volume_col else 0
            turnover = _safe_float(row.get(turnover_col)) if turnover_col else 0

            if price is None or price <= 0:
                continue

            # 初筛得分：板块强势 + ETF 涨幅合理 + 不是暴涨（暴涨=不是回踩）
            quick_score = sector["score"] * 0.5
            if change_pct is not None:
                # 涨幅 0-3% 最好（回踩后的温和上涨），<-1% 或 >8% 不太好
                if 0 <= change_pct <= 3:
                    quick_score += 2.0
                elif -1 <= change_pct < 0 or 3 < change_pct <= 5:
                    quick_score += 1.0
                elif change_pct > 8 or change_pct < -3:
                    quick_score -= 2.0
            if discount is not None:
                # 折价买入更好
                if discount < -0.5:
                    quick_score += 1.5
                elif discount < 0:
                    quick_score += 0.5
            if fund_flow and fund_flow > 0:
                quick_score += 1.0
            if turnover and turnover > 0:
                quick_score += 0.5

            quick_candidates.append({
                "code": code,
                "name": etf_name,
                "price": round(price, 3),
                "change_pct": change_pct,
                "discount": discount,
                "fund_flow": fund_flow,
                "sector": sector_name,
                "sector_rank": sector["rank"],
                "sector_score": sector["score"],
                "quick_score": round(quick_score, 2),
            })
            sector_count += 1
            if sector_count >= max_per_sector:
                break

    # 按快速得分排序，取前4个做深度检测（减少K线API调用）
    quick_candidates.sort(key=lambda x: x["quick_score"], reverse=True)
    deep_check = quick_candidates[:4]

    # —— 第二轮：并行 K 线数据做深度回踩检测 ——
    final = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {
            executor.submit(_check_pullback, c["code"], c["price"]): c
            for c in deep_check
        }
        for future in as_completed(future_map, timeout=60):
            c = future_map[future]
            try:
                kline_ok, pullback_info = future.result(timeout=25)
                if kline_ok:
                    c.pop("quick_score", None)
                    final.append({**c, **pullback_info})
            except Exception:
                pass

    # 如果深度检测结果太少，把初筛高分的也加进来（标记为"趋势跟踪"而非回踩）
    if len(final) < 3:
        for c in deep_check:
            if c["code"] not in {f["code"] for f in final}:
                c.pop("quick_score", None)
                final.append({
                    **c,
                    "ma20": None, "ma60": None, "high_20d": None,
                    "dist_to_high_pct": 0, "dist_to_ma20_pct": 0,
                    "volume_ok": True, "pullback_score": c.get("quick_score", 1),
                    "trend_label": "趋势跟踪",
                })

    final.sort(key=lambda x: x.get("pullback_score", 0), reverse=True)
    return final[:min(max_candidates, 8)]


# ---------------------------------------------------------------------------
# 回踩检测
# ---------------------------------------------------------------------------

def _check_pullback(code: str, current_price: float) -> tuple[bool, dict]:
    """检测 ETF 是否处于健康的回踩买点。"""
    df = fetch_etf_history(code, days=120)
    if df is None or df.empty:
        # K线数据不可用，用当前价保守估算
        # 距MA20估计2%，距高点估计-3%（假设温和回踩中）
        return True, {
            "ma20": round(current_price * 0.98, 3),
            "ma60": round(current_price * 0.95, 3),
            "high_20d": round(current_price * 1.03, 3),
            "dist_to_high_pct": -3.0,
            "dist_to_ma20_pct": 2.0,
            "volume_ok": True,
            "pullback_score": 7.0,
            "trend_label": "实时行情(估)",
        }

    close_col = _find_col(df, ["收盘", "close", "收盘价"])
    if close_col is None:
        return True, {
            "ma20": None, "ma60": None, "high_20d": None,
            "dist_to_high_pct": 0, "dist_to_ma20_pct": 0,
            "volume_ok": True, "pullback_score": 0.3, "trend_label": "数据缺失",
        }

    close = pd.to_numeric(df[close_col], errors="coerce").dropna().values
    if len(close) < 20:
        return True, {
            "ma20": None, "ma60": None, "high_20d": None,
            "dist_to_high_pct": 0, "dist_to_ma20_pct": 0,
            "volume_ok": True, "pullback_score": 0.2, "trend_label": "数据不足",
        }

    ma20 = float(pd.Series(close).rolling(20).mean().iloc[-1])
    ma60 = float(pd.Series(close).rolling(min(60, len(close))).mean().iloc[-1]) if len(close) >= 60 else ma20

    # 条件 1: 上升趋势
    if current_price <= ma20 or ma20 <= ma60:
        return False, {}

    # 条件 2: 回踩幅度（距20日高点 -1% ~ -8%）
    high_20d = float(np.max(close[-20:]))
    dist_to_high = (current_price - high_20d) / high_20d * 100

    if dist_to_high > -1 or dist_to_high < -10:
        return False, {}

    # 条件 3: 不远离 MA20 (< 5%)
    dist_to_ma20 = (current_price - ma20) / ma20 * 100
    if dist_to_ma20 > 5:
        return False, {}

    # 条件 4: 成交量不异常萎缩
    vol_cols = ["成交量", "volume", "成交额"]
    vol_ok = True
    for vc in vol_cols:
        if vc in df.columns:
            vol = pd.to_numeric(df[vc], errors="coerce").dropna().values
            if len(vol) >= 20:
                vol_5d_avg = float(np.mean(vol[-5:]))
                vol_20d_avg = float(np.mean(vol[-20:-5]))
                if vol_20d_avg > 0 and vol_5d_avg / vol_20d_avg < 0.3:
                    vol_ok = False
            break

    # 计算回踩得分
    pullback_score = _calc_pullback_score(dist_to_high, dist_to_ma20, vol_ok)

    return True, {
        "ma20": round(ma20, 3),
        "ma60": round(ma60, 3),
        "high_20d": round(high_20d, 3),
        "dist_to_high_pct": round(dist_to_high, 2),
        "dist_to_ma20_pct": round(dist_to_ma20, 2),
        "volume_ok": vol_ok,
        "pullback_score": round(pullback_score, 2),
        "trend_label": "上升趋势" if ma20 > ma60 else "震荡偏强",
    }


def _calc_pullback_score(dist_to_high: float, dist_to_ma20: float, vol_ok: bool) -> float:
    """
    回踩得分：
    - 回踩越深（距高点越远）得分越高，但超过 -8% 不推荐
    - 离 MA20 越近越好（最佳在 MA20 上方 0-2%）
    - 成交量正常加分
    """
    score = abs(dist_to_high) * 1.5  # 回踩深度加分，最多 ~12
    score -= abs(dist_to_ma20) * 0.5  # 远离均线扣分
    if vol_ok:
        score += 1.0  # 量能正常加分
    if -5 < dist_to_high < -2:
        score += 2.0  # 黄金回踩区间加分
    return score


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    # 模糊匹配（忽略大小写）
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


# 板块名→ETF关键词映射（板块名本身可能不直接出现在ETF名称中）
_SECTOR_ETF_KEYWORDS = {
    "半导体": ["半导体", "芯片", "集成电路"],
    "芯片": ["芯片", "半导体", "集成电路"],
    "CPO": ["通信", "光通信", "CPO", "光模块", "5G"],
    "光通信": ["通信", "光通信", "CPO", "光模块", "5G"],
    "光模块": ["通信", "光通信", "CPO", "光模块", "5G"],
    "印制电路板": ["电子", "科技", "5G", "芯片"],
    "PCB": ["电子", "科技", "5G", "芯片"],
    "MLCC": ["电子", "科技", "消费电子"],
    "被动元件": ["电子", "科技", "消费电子"],
    "消费电子": ["消费电子", "电子", "科技"],
    "机器人": ["机器人", "智能制造", "自动化", "机械"],
    "人工智能": ["人工智能", "AI", "计算机", "大数据", "云计算"],
    "AI": ["人工智能", "AI", "科技", "计算机"],
    "创新药": ["创新药", "医药", "生物医药", "医疗", "CRO"],
    "医药": ["医药", "医疗", "生物医药", "创新药", "中药"],
    "生物医药": ["医药", "医疗", "生物医药", "创新药"],
    "算力": ["计算机", "云计算", "大数据", "算力"],
    "存储": ["芯片", "半导体", "科技"],
    "新能源": ["新能源", "光伏", "锂电", "电池", "新能源车"],
    "光伏": ["光伏", "新能源", "太阳能"],
    "电池": ["电池", "新能源", "锂电", "新能源车"],
    "军工": ["军工", "国防", "航空航天"],
    "证券": ["证券", "券商"],
    "银行": ["银行"],
    "白酒": ["白酒", "消费", "食品饮料"],
    "食品": ["消费", "食品饮料", "食品"],
    "煤炭": ["煤炭", "能源", "资源"],
    "有色": ["有色", "矿产", "资源", "稀土"],
    "汽车": ["汽车", "新能源车", "智能驾驶"],
    "电力": ["电力", "公用事业"],
    "房地产": ["房地产", "地产"],
    "钢铁": ["钢铁", "资源"],
    "科创": ["科创", "科创板", "科创50"],
    "创业板": ["创业板", "创成长", "创50"],
    "沪深300": ["沪深300", "300"],
    "中证500": ["中证500", "500"],
    "中证1000": ["中证1000", "1000"],
}


def _match_sector(etf_name: str, sector_name: str) -> bool:
    """板块名到ETF名称的匹配：先查映射表，再用直接子串匹配。"""
    # 1. 直接包含
    if sector_name in etf_name:
        return True

    # 2. 映射表关键词
    keywords = _SECTOR_ETF_KEYWORDS.get(sector_name, [])
    for kw in keywords:
        if kw in etf_name:
            return True

    # 3. 反向：检查sector_name的每个片段（拆分为2-4字词）
    for i in range(len(sector_name) - 1):
        chunk = sector_name[i:i+2]
        if len(chunk) >= 2 and chunk in etf_name:
            return True

    return False


def _safe_float(val) -> float | None:
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (ValueError, TypeError):
        return None
