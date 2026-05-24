"""板块排名 — 机构级选股：趋势质量 > 单日涨幅。

核心理念：
1. 连涨3天 > 单日暴拉（持续性 > 爆发力）
2. 温和放量 > 缩量急拉
3. 资金持续流入 > 脉冲式流入
4. 单日暴拉 + 前期连跌 = 追涨陷阱（标记为高风险）
"""

import numpy as np
import pandas as pd

from .data_fetcher import fetch_all_boards, fetch_industry_fund_flow, fetch_sector_spot

_EXCLUDE_KEYWORDS = [
    "融资融券", "昨日涨停", "昨日跌停", "昨日连板", "昨日",
    "转债", "百元股", "基金重仓", "券商重仓",
    "机构重仓", "信托重仓", "保险重仓", "社保持仓",
    "QFII", "MSCI", "罗素", "行业龙头", "大盘", "小盘",
]

FOCUS_KEYWORDS = [
    "人工智能", "AI", "ChatGPT", "AIGC", "大模型", "智能",
    "半导体", "芯片", "集成电路", "封测", "光刻",
    "创新药", "生物医药", "CRO", "CXO", "医药",
    "算力", "服务器", "光模块", "CPO", "液冷",
    "存储", "HBM", "闪存", "内存",
    "消费电子", "机器人", "自动驾驶", "低空经济",
    "科创", "创业板", "科创板",
]


def rank_sectors(
    boards: pd.DataFrame | None = None,
    industry_flow: pd.DataFrame | None = None,
    sectors: pd.DataFrame | None = None,
    top_n: int = 30,
) -> dict:
    """板块排名入口，三层降级。"""
    if boards is not None and not boards.empty:
        result = _rank_from_boards(boards, top_n)
    elif industry_flow is not None and not industry_flow.empty:
        result = _rank_from_industry_flow(industry_flow, top_n)
    elif sectors is not None and not sectors.empty:
        result = _rank_from_sectors(sectors, top_n)
    else:
        return {"ranked_boards": [], "top_sectors": [], "focus_in_top": [],
                "leading_stocks": [],
                "source": "无数据", "board_count": 0}

    # 标注追涨风险板块
    _flag_spike_risk(result["top_sectors"])

    return result


# ===========================================================================
# 主排名 — stock_board_change_em
# ===========================================================================

def _rank_from_boards(df: pd.DataFrame, top_n: int) -> dict:
    clean = _filter_boards(df)
    cols = list(clean.columns)

    name_col = _find_col(clean, ["板块名称", "名称"])
    pct_col = _find_col(clean, ["涨跌幅", "板块涨跌幅", "change_pct", "rate"])
    flow_col = _find_col(clean, ["主力净流入", "主力资金净流入", "资金净流入", "主力净流入-净额"])
    anomaly_col = _find_col(clean, ["板块异动总次数", "异动次数", "异动总次数"])
    volume_col = _find_col(clean, ["成交额", "总成交额", "成交金额"])
    lead_code_col = _find_col(clean, ["板块异动最频繁个股及所属类型-股票代码"])
    lead_name_col = _find_col(clean, ["板块异动最频繁个股及所属类型-股票名称"])

    if name_col is None or pct_col is None:
        return {"ranked_boards": [], "top_sectors": [], "focus_in_top": [],
                "source": "板块数据列名不匹配", "board_count": 0}

    names = clean[name_col].astype(str).tolist()
    pcts = pd.to_numeric(clean[pct_col], errors="coerce").fillna(0).values
    flows = pd.to_numeric(clean[flow_col], errors="coerce").fillna(0).values if flow_col else np.zeros(len(clean))
    anomalies = pd.to_numeric(clean[anomaly_col], errors="coerce").fillna(0).values if anomaly_col else np.zeros(len(clean))
    volumes = pd.to_numeric(clean[volume_col], errors="coerce").fillna(0).values if volume_col else np.zeros(len(clean))

    # 提取板块龙头股（异动最频繁个股）
    lead_codes = clean[lead_code_col].astype(str).tolist() if lead_code_col else [""] * len(clean)
    lead_names = clean[lead_name_col].astype(str).tolist() if lead_name_col else [""] * len(clean)

    z_pct = _robust_zscore(pcts)
    z_flow = _robust_zscore(flows)
    z_anomaly = _robust_zscore(anomalies)
    z_volume = _robust_zscore(volumes)

    # 趋势质量分：异动次数多且涨幅温和 → 持续性行情而非脉冲
    spike_ratio = np.zeros_like(pcts)
    for i in range(len(pcts)):
        if anomalies[i] >= 3 and abs(pcts[i]) > 0:
            spike_ratio[i] = abs(anomalies[i] / max(abs(pcts[i]), 0.01))
    z_continuity = _robust_zscore(spike_ratio)

    scores = (0.20 * z_pct +
              0.35 * z_flow +
              0.15 * z_anomaly +
              0.10 * z_volume +
              0.20 * z_continuity)

    order = np.argsort(scores)[::-1]
    ranked = []
    for rank, idx in enumerate(order, 1):
        name = names[idx]
        spike_flag = _detect_spike(pcts[idx], anomalies[idx])
        ranked.append({
            "rank": rank,
            "name": name,
            "change_pct": round(float(pcts[idx]), 2),
            "fund_flow": round(float(flows[idx]), 2) if flow_col else 0,
            "anomaly_count": int(anomalies[idx]) if anomaly_col else 0,
            "volume": round(float(volumes[idx]), 2) if volume_col else 0,
            "score": round(float(scores[idx]), 2),
            "is_focus": _is_focus(name),
            "spike_warning": spike_flag,
            "lead_code": lead_codes[idx] if lead_codes[idx] and lead_codes[idx] != "nan" else "",
            "lead_name": lead_names[idx] if lead_names[idx] and lead_names[idx] != "nan" else "",
        })

    top = ranked[:top_n]
    # 收集所有板块的龙头股（用于个股筛选种子）
    leading_stocks = []
    seen_codes = set()
    for r in ranked:
        code = r.get("lead_code", "")
        name = r.get("lead_name", "")
        if code and code not in seen_codes:
            seen_codes.add(code)
            leading_stocks.append({
                "code": code, "name": name,
                "sector": r["name"], "sector_rank": r["rank"],
                "sector_score": r["score"],
            })

    return {
        "ranked_boards": ranked,
        "top_sectors": top,
        "focus_in_top": [b for b in top if b["is_focus"]],
        "leading_stocks": leading_stocks,
        "source": "stock_board_change_em (全板块)",
        "board_count": len(ranked),
    }


# ===========================================================================
# 降级方案
# ===========================================================================

def _rank_from_industry_flow(df: pd.DataFrame, top_n: int) -> dict:
    name_col = _find_col(df, ["行业", "名称", "板块"])
    pct_col = _find_col(df, ["行业-涨跌幅", "涨跌幅", "change_pct"])
    flow_col = _find_col(df, ["净额", "主力净流入", "净流入"])

    if name_col is None:
        return {"ranked_boards": [], "top_sectors": [], "focus_in_top": [],
                "source": "行业数据无名称列", "board_count": 0}

    names = df[name_col].astype(str).tolist()
    pcts = pd.to_numeric(df[pct_col], errors="coerce").fillna(0).values if pct_col else np.zeros(len(df))
    flows = pd.to_numeric(df[flow_col], errors="coerce").fillna(0).values if flow_col else np.zeros(len(df))

    z_pct = _robust_zscore(pcts)
    z_flow = _robust_zscore(flows)
    scores = 0.35 * z_pct + 0.65 * z_flow

    order = np.argsort(scores)[::-1]
    ranked = []
    for rank, idx in enumerate(order, 1):
        name = names[idx]
        ranked.append({
            "rank": rank, "name": name,
            "change_pct": round(float(pcts[idx]), 2),
            "fund_flow": round(float(flows[idx]), 2),
            "anomaly_count": 0, "volume": 0,
            "score": round(float(scores[idx]), 2),
            "is_focus": _is_focus(name),
            "spike_warning": False,
        })

    top = ranked[:top_n]
    return {
        "ranked_boards": ranked, "top_sectors": top,
        "focus_in_top": [b for b in top if b["is_focus"]],
        "leading_stocks": [],
        "source": "stock_fund_flow_industry (行业)", "board_count": len(ranked),
    }


def _rank_from_sectors(df: pd.DataFrame, top_n: int) -> dict:
    name_col = _find_col(df, ["板块", "名称", "板块名称"])
    pct_col = _find_col(df, ["涨跌幅", "平均涨跌幅"])

    if name_col is None:
        return {"ranked_boards": [], "top_sectors": [], "focus_in_top": [],
                "source": "行业数据无名称列", "board_count": 0}

    names = df[name_col].astype(str).tolist()
    pcts = pd.to_numeric(df[pct_col], errors="coerce").fillna(0).values if pct_col else np.zeros(len(df))
    scores = _robust_zscore(pcts)

    order = np.argsort(scores)[::-1]
    ranked = []
    for rank, idx in enumerate(order, 1):
        name = names[idx]
        ranked.append({
            "rank": rank, "name": name,
            "change_pct": round(float(pcts[idx]), 2),
            "fund_flow": 0, "anomaly_count": 0, "volume": 0,
            "score": round(float(scores[idx]), 2),
            "is_focus": _is_focus(name),
            "spike_warning": False,
        })
    top = ranked[:top_n]
    return {
        "ranked_boards": ranked, "top_sectors": top,
        "focus_in_top": [b for b in top if b["is_focus"]],
        "leading_stocks": [],
        "source": "stock_sector_spot (新浪行业)", "board_count": len(ranked),
    }


# ===========================================================================
# 追涨检测
# ===========================================================================

def _detect_spike(pct: float, anomaly_count: int) -> bool:
    """单日涨幅 > 4% 且异动次数很少 → 脉冲式暴拉，追涨风险高。"""
    if pct > 4.0 and anomaly_count <= 1:
        return True
    if pct > 6.0:
        return True
    return False


def _flag_spike_risk(sectors: list):
    """给板块标注追涨风险等级。"""
    for s in sectors:
        pct = s.get("change_pct", 0) or 0
        anomaly = s.get("anomaly_count", 0) or 0

        if pct > 6 and anomaly <= 1:
            s["chase_level"] = "danger"
            s["chase_label"] = "单日暴拉无持续异动，追高风险极大"
        elif pct > 4 and anomaly <= 1:
            s["chase_level"] = "warning"
            s["chase_label"] = "单日涨幅较大，建议等回踩"
        elif pct > 3 and anomaly >= 3:
            s["chase_level"] = "safe"
            s["chase_label"] = "涨幅温和+多次异动，趋势健康"
        elif anomaly >= 5:
            s["chase_level"] = "safe"
            s["chase_label"] = "持续异动，板块热度高"
        else:
            s["chase_level"] = "normal"
            s["chase_label"] = ""


# ===========================================================================
# 工具
# ===========================================================================

def _filter_boards(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for kw in _EXCLUDE_KEYWORDS:
        for col in df.columns[:3]:
            mask &= ~df[col].astype(str).str.contains(kw, na=False)
    return df[mask].copy()


def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    arr = values.astype(float)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad < 1e-12:
        return np.zeros_like(arr)
    z = (arr - med) / (mad * 1.4826)
    return np.clip(z, -5, 5)


def _is_focus(name: str) -> bool:
    for kw in FOCUS_KEYWORDS:
        if kw in name:
            return True
    return False
