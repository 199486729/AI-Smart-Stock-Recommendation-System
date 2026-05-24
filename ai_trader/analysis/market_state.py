"""市场状态分类 — 机构级多空信号分析 + 方向预测 + 追涨风险评估。

核心理念：不只看当前状态，更要判断接下来大概率怎么走。
用多空信号计数法，信号越一致，置信度越高。
"""

from typing import Optional

import numpy as np


def compute_market_state(
    indices: Optional[dict] = None,
    breadth: Optional[dict] = None,
    north_flow: Optional[dict] = None,
) -> dict:
    """综合多维度数据，输出方向判断、置信度、追涨风险。"""
    bull = []
    bear = []

    # ——— 1. 指数趋势信号 ———
    trend_result = _analyze_trend(indices, bull, bear)

    # ——— 2. 量价关系信号 ———
    volume_result = _analyze_volume_price(indices, bull, bear)

    # ——— 3. 市场广度信号 ———
    breadth_result = _analyze_breadth(breadth, bull, bear)

    # ——— 4. 北向资金信号 ———
    north_result = _analyze_north(north_flow, bull, bear)

    # ——— 5. 修复/反转检测 ———
    recovery = _detect_recovery(indices, breadth)

    # ——— 6. 追涨风险评估 ———
    chase_risk = _assess_chase_risk(recovery, trend_result, breadth_result)

    # ——— 7. 综合方向判断 ———
    direction, conviction, direction_score = _compute_direction(bull, bear)

    # ——— 8. 市场状态分类 ———
    composite, risk, suitable = _classify(direction, conviction, recovery, chase_risk)

    # ——— 9. 关键位 ———
    key_levels = _find_key_levels(indices)

    # ——— 10. 中文总结 ———
    summary = _build_summary(
        composite, direction, conviction, bull, bear,
        recovery, chase_risk, volume_result, north_result, indices
    )

    return {
        "composite": composite,
        "direction": direction,
        "direction_score": direction_score,
        "conviction": conviction,
        "risk": risk,
        "suitable": suitable,
        "chase_risk": chase_risk,
        "bull_signals": bull,
        "bear_signals": bear,
        "trend": trend_result["label"],
        "trend_detail": trend_result["detail"],
        "breadth": breadth_result["label"],
        "breadth_raw": breadth_result["raw"],
        "north_direction": north_result.get("direction", "未知"),
        "north_5d": north_result.get("net_5d", 0),
        "north_20d": north_result.get("net_20d", 0),
        "recovery": recovery,
        "volume_analysis": volume_result,
        "key_levels": key_levels,
        "summary": summary,
        "indices_snapshot": _indices_snapshot(indices) if indices else {},
    }


# ===========================================================================
# 信号分析
# ===========================================================================

def _analyze_trend(indices, bull, bear) -> dict:
    """指数趋势：MA位置 + MA方向 + 近期变化率。"""
    if not indices:
        bull.append("⚠ 指数数据缺失")
        return {"label": "未知", "detail": {}}

    above_ma20 = 0
    above_ma60 = 0
    total = len(indices)
    detail = {}

    for name, d in indices.items():
        a20 = d.get("above_ma20")
        a60 = d.get("above_ma60")
        chg = d.get("change_pct", 0) or 0
        chg5 = d.get("change_5d")

        detail[name] = {
            "price": d["price"],
            "change_pct": chg,
            "change_5d": chg5,
            "above_ma20": a20,
            "above_ma60": a60,
        }
        if a20 is True:
            above_ma20 += 1
        if a60 is True:
            above_ma60 += 1

    ratio_ma20 = above_ma20 / total
    ratio_ma60 = above_ma60 / total

    if ratio_ma20 >= 0.67:
        label = "上升趋势"
        bull.append(f"✓ {above_ma20}/{total} 指数站上MA20（均线多头排列）")
    elif ratio_ma20 <= 0.33:
        label = "下降趋势"
        bear.append(f"✗ 仅{above_ma20}/{total} 指数站上MA20（均线空头排列）")
    else:
        label = "震荡整理"
        if ratio_ma20 >= 0.5:
            bull.append(f"△ {above_ma20}/{total} 指数站上MA20（偏多震荡）")
        else:
            bear.append(f"△ 仅{above_ma20}/{total} 指数站上MA20（偏空震荡）")

    return {"label": label, "ratio_ma20": round(ratio_ma20, 2),
            "ratio_ma60": round(ratio_ma60, 2), "detail": detail}


def _analyze_volume_price(indices, bull, bear) -> dict:
    """量价关系：涨放量=健康，涨缩量=虚，跌放量=恐慌。"""
    if not indices:
        return {"status": "unknown", "label": "无数据"}

    vol_healthy = 0
    vol_unhealthy = 0
    for name, d in indices.items():
        chg = d.get("change_pct", 0) or 0
        chg5 = d.get("change_5d")
        # 用价格方向和成交量的关系粗略判断
        # 真实的量价分析需要日线明细，这里用5日涨跌+量能趋势近似
        if chg5 is not None:
            # 5日涨 + 今日也在涨 = 持续上行
            if chg5 > 0 and chg > 0:
                vol_healthy += 1
            elif chg5 < -3 and chg > 2:
                # 5日大跌但今日暴拉 = 修复反弹，量价背离风险
                vol_unhealthy += 1

    if vol_unhealthy >= 2:
        bear.append("✗ 疑似修复性反弹：5日跌幅大+今日暴拉，量价可能背离")
        return {"status": "divergence", "label": "量价背离：缩量反弹风险"}
    else:
        total = len(indices)
        if vol_healthy >= total * 0.67:
            bull.append("✓ 多数指数量价配合正常")
            return {"status": "healthy", "label": "量价配合正常"}
    return {"status": "normal", "label": "量价正常"}


def _analyze_breadth(breadth, bull, bear) -> dict:
    """市场广度：涨跌比、涨停跌停数。"""
    row = breadth or {}

    def _num(key):
        v = row.get(key, 1)
        try:
            return float(v)
        except (ValueError, TypeError):
            return 1

    up = max(_num("上涨"), 1)
    down = max(_num("下跌"), 1)
    limit_up = _num("涨停")
    limit_down = _num("跌停")
    ratio = up / down

    if ratio > 2.0:
        label = "普涨"
        if ratio > 4:
            bull.append(f"✓ 市场广度极强：涨跌比 {ratio:.1f}（全面普涨）")
            return {"label": label, "ratio": round(ratio, 3),
                    "raw": _make_breadth_raw(up, down, limit_up, limit_down)}
        bull.append(f"✓ 市场广度健康：涨跌比 {ratio:.1f}（普涨格局）")
    elif ratio > 1.2:
        label = "结构性上涨"
        bull.append(f"△ 涨跌比 {ratio:.1f}（结构性行情）")
    elif ratio > 0.8:
        label = "结构性下跌"
        bear.append(f"△ 涨跌比 {ratio:.1f}（偏弱格局）")
    else:
        label = "普跌"
        bear.append(f"✗ 涨跌比 {ratio:.1f}（普跌）")

    # 涨跌停信号
    if limit_up >= 80 and limit_down <= 10:
        bull.append(f"✓ 涨停{int(limit_up)}家 vs 跌停{int(limit_down)}家，短线情绪亢奋")
    elif limit_down >= 30:
        bear.append(f"✗ 跌停{int(limit_down)}家，恐慌情绪")

    return {"label": label, "ratio": round(ratio, 3),
            "raw": _make_breadth_raw(up, down, limit_up, limit_down)}


def _make_breadth_raw(up, down, limit_up, limit_down) -> dict:
    return {"上涨": int(up), "下跌": int(down),
            "涨停": int(limit_up), "跌停": int(limit_down),
            "涨跌比": round(up / down, 2)}


def _analyze_north(north, bull, bear) -> dict:
    """北向资金信号。"""
    n = north or {}
    net5 = n.get("net_buy_5d", 0)
    net20 = n.get("net_buy_20d", 0)

    if net5 > 80:
        bull.append(f"✓ 北向5日净流入 {net5:.0f}亿（大幅流入）")
    elif net5 > 30:
        bull.append(f"△ 北向5日净流入 {net5:.0f}亿（小幅流入）")
    elif net5 < -80:
        bear.append(f"✗ 北向5日净流出 {abs(net5):.0f}亿（大幅流出）")
    elif net5 < -30:
        bear.append(f"△ 北向5日净流出 {abs(net5):.0f}亿（小幅流出）")

    direction = "大幅流入" if net5 > 80 else ("小幅流入" if net5 > 0 else
                 ("大幅流出" if net5 < -80 else ("小幅流出" if net5 < 0 else "持平")))

    return {"direction": direction, "net_5d": round(net5, 1), "net_20d": round(net20, 1)}


# ===========================================================================
# 修复/反转检测
# ===========================================================================

def _detect_recovery(indices, breadth) -> dict:
    """检测是否处于暴跌后的修复反弹。"""
    if not indices:
        return {"detected": False, "label": ""}

    for name in ["上证综指", "创业板指", "深证成指"]:
        if name not in indices:
            continue
        d = indices[name]
        chg = d.get("change_pct", 0) or 0
        chg5 = d.get("change_5d")

        # 今日大涨 +2% 以上，但 5 日还是跌的
        if chg > 1.5 and chg5 is not None and chg5 < -1:
            recovery_pct = abs(chg)
            prior_drop = abs(chg5 - chg)  # 前4天大概跌了多少
            recovery_ratio = recovery_pct / max(prior_drop, 0.01)

            # 反弹只收复了不到50%的跌幅
            if recovery_ratio < 0.5:
                return {
                    "detected": True,
                    "type": "弱修复",
                    "index": name,
                    "today_gain": round(chg, 2),
                    "drop_5d": round(chg5, 2),
                    "recovery_ratio": round(recovery_ratio * 100, 1),
                    "label": f"{name}今日+{chg:.1f}%，但5日仍跌{chg5:.1f}%，仅修复{recovery_ratio*100:.0f}%，确认弱修复",
                }
            elif recovery_ratio < 0.8:
                return {
                    "detected": True,
                    "type": "半修复",
                    "index": name,
                    "today_gain": round(chg, 2),
                    "drop_5d": round(chg5, 2),
                    "recovery_ratio": round(recovery_ratio * 100, 1),
                    "label": f"{name}今日+{chg:.1f}%，收复{recovery_ratio*100:.0f}%跌幅，修复中但未确认反转",
                }
            else:
                return {
                    "detected": True,
                    "type": "强修复",
                    "index": name,
                    "today_gain": round(chg, 2),
                    "drop_5d": round(chg5, 2),
                    "recovery_ratio": round(recovery_ratio * 100, 1),
                    "label": f"{name}今日+{chg:.1f}%，收复{recovery_ratio*100:.0f}%跌幅，修复力度强",
                }

        # 今日大涨 +2%，5日也涨了 → 不是修复，是真的趋势
        if chg > 1.5 and chg5 is not None and chg5 > 0:
            return {"detected": False, "type": "趋势延续", "label": ""}

    return {"detected": False, "type": "正常", "label": ""}


# ===========================================================================
# 追涨风险评估
# ===========================================================================

def _assess_chase_risk(recovery, trend_result, breadth_result) -> dict:
    """评估当前追涨买入的风险等级。"""
    risk_score = 0
    reasons = []

    # 修复反弹中的追涨 = 高风险
    if recovery.get("detected"):
        if recovery.get("type") == "弱修复":
            risk_score += 40
            reasons.append("弱修复反弹追涨风险极高")
        elif recovery.get("type") == "半修复":
            risk_score += 25
            reasons.append("半修复状态，追涨性价比低")
        elif recovery.get("type") == "强修复":
            risk_score += 10
            reasons.append("强修复但需确认反转")

    # 趋势不明确时追涨 = 风险
    if trend_result["label"] == "下降趋势":
        risk_score += 30
        reasons.append("下降趋势中的反弹不值得追")
    elif trend_result["label"] == "震荡整理":
        risk_score += 15
        reasons.append("震荡市追涨空间有限")

    # 广度极端 = 可能是高潮
    if breadth_result.get("ratio", 1) > 4:
        risk_score += 15
        reasons.append("全面普涨可能是短期高潮")

    if risk_score >= 40:
        level = "高"
        label = "不建议追涨，等待回踩确认"
    elif risk_score >= 20:
        level = "中"
        label = "可轻仓试探，不宜重仓追入"
    else:
        level = "低"
        label = "趋势确立，追涨风险可控"

    return {"level": level, "score": risk_score, "label": label, "reasons": reasons}


# ===========================================================================
# 方向判断
# ===========================================================================

def _compute_direction(bull, bear) -> tuple:
    """多空信号 → 方向 + 置信度。"""
    b = len([s for s in bull if not s.startswith("⚠") and not s.startswith("△")])
    be = len(bear)

    # 加权：✓=2, △=1, ✗=2
    bull_weight = sum(2 if s.startswith("✓") else 1 for s in bull)
    bear_weight = sum(2 if s.startswith("✗") else 1 for s in bear)

    net = bull_weight - bear_weight
    score = max(-100, min(100, net * 15))

    if net >= 4:
        direction = "看多"
    elif net >= 1:
        direction = "偏多"
    elif net >= -1:
        direction = "中性"
    elif net >= -4:
        direction = "偏空"
    else:
        direction = "看空"

    total_signals = b + be
    agree_ratio = abs(bull_weight - bear_weight) / max(bull_weight + bear_weight, 1)
    if total_signals >= 5 and agree_ratio > 0.6:
        conviction = "高"
    elif total_signals >= 3 and agree_ratio > 0.3:
        conviction = "中"
    else:
        conviction = "低"

    return direction, conviction, score


# ===========================================================================
# 市场状态分类
# ===========================================================================

def _classify(direction, conviction, recovery, chase_risk) -> tuple:
    recovery_type = recovery.get("type", "")

    if recovery_type == "弱修复":
        return "修复反弹（弱）", "高", False
    if recovery_type == "半修复":
        return "修复反弹（待确认）", "中", False

    if direction in ("看多", "偏多") and conviction in ("高", "中"):
        if chase_risk["level"] == "低":
            return "进攻市", "低", True
        else:
            return "震荡偏多", "中", True

    if direction in ("看空", "偏空"):
        return "防守市", "高", False

    return "震荡市", "中", True


# ===========================================================================
# 关键位
# ===========================================================================

def _find_key_levels(indices) -> dict:
    """找最近支撑和阻力（简化版：用MA20/MA60做参考）。"""
    if not indices:
        return {}
    name = "上证综指"
    d = indices.get(name, list(indices.values())[0])
    price = d["price"]
    ma20 = d.get("ma20", price)
    ma60 = d.get("ma60", price)
    high_20d_raw = d.get("dist_to_high_20d")

    support = round(min(ma20, ma60), 2) if ma20 and ma60 else round(price * 0.97, 2)
    if high_20d_raw is not None:
        resistance = round(price * (1 + abs(high_20d_raw) / 100), 2)
    else:
        resistance = round(max(ma20, ma60, price * 1.02), 2)

    return {
        "index": name,
        "price": price,
        "support": support,
        "resistance": resistance,
        "upside_pct": round((resistance - price) / price * 100, 2),
        "downside_pct": round((price - support) / price * 100, 2),
    }


# ===========================================================================
# 中文总结
# ===========================================================================

def _build_summary(composite, direction, conviction, bull, bear,
                   recovery, chase_risk, volume, north, indices) -> str:
    parts = [f"【{composite}】，方向判断：{direction}（置信度{conviction}）"]

    # 多空信号摘要
    active_bull = [s for s in bull if not s.startswith("⚠")]
    active_bear = [s for s in bear if not s.startswith("⚠")]
    parts.append(f"；多头信号{len(active_bull)}个，空头信号{len(active_bear)}个")

    # 修复检测
    if recovery.get("detected"):
        parts.append(f"；⚠ {recovery['label']}")

    # 追涨风险
    parts.append(f"；追涨风险：{chase_risk['level']}")

    # 指数快照
    if indices:
        idx_strs = []
        for name in ["上证综指", "创业板指"]:
            if name in indices:
                d = indices[name]
                sign = "+" if (d.get("change_pct") or 0) >= 0 else ""
                idx_strs.append(f"{name} {d['price']}({sign}{d.get('change_pct', 0):.2f}%)")
        parts.append("；" + " ".join(idx_strs))

    # 操作建议
    if chase_risk["level"] == "高":
        parts.append("。不建议追高，等待回踩确认后再入场。")
    elif composite == "进攻市":
        parts.append("。趋势确立，可积极参与强势板块龙头。")
    elif composite == "震荡偏多":
        parts.append("。适度参与，优选回调到位的标的，控制仓位。")
    elif composite == "防守市":
        parts.append("。建议观望或极轻仓，等待趋势明朗。")
    else:
        parts.append("。控制仓位，精选标的，严格止损。")

    return "".join(parts)


# ===========================================================================
# 工具
# ===========================================================================

def _indices_snapshot(indices: dict) -> list:
    snap = []
    for name, d in indices.items():
        snap.append({
            "name": name,
            "price": d["price"],
            "change_pct": d.get("change_pct"),
            "trend": "↑" if d.get("above_ma20") else ("↓" if d.get("above_ma20") is False else "—"),
        })
    return snap
