"""买入推荐生成器 — 星级制精选推荐。

星级制度：
- 满星 5★：极高胜率，BUY_NOW + 高质量 + 市场顺风 → 15%仓位
- 4★：高胜率，BUY_NOW + 良好质量 → 12%仓位
- 3★：标准买入，BUY_NOW 或优质回调 → 10%仓位（一成仓）
- 2★：可以买但需谨慎，等待回调 → 7%仓位
- 1★：勉强可入，等更好点位 → 5%仓位
- 0★ / 不推：追涨风险高、质量差、市场不适合

只推 2★ 及以上，最多推 3 只。
"""

import math

_ALLOC_PER_STAR = {5: 0.15, 4: 0.12, 3: 0.10, 2: 0.07, 1: 0.05}
_TRANCHE_RATIOS = [0.50, 0.30, 0.20]
_TRANCHE_OFFSETS = [0.0, -0.02, -0.05]
_STOP_LOSS_PCT = 0.065
_DEFAULT_CAPITAL = 100000
_MIN_STARS_TO_RECOMMEND = 2
_MAX_RECS = 3


def generate_recommendations(
    market_state: dict,
    top_sectors: list[dict],
    candidates: list[dict] | None = None,
    stocks: list[dict] | None = None,
    available_capital: float = _DEFAULT_CAPITAL,
    max_recs: int = _MAX_RECS,
) -> list[dict]:
    """合并个股和 ETF 候选，按星级排名输出精选推荐。

    只推 >= _MIN_STARS_TO_RECOMMEND 星的标的，最多 max_recs 只。
    """
    composite = market_state.get("composite", "震荡市")
    chase_level = market_state.get("chase_risk", {}).get("level", "中")

    all_candidates = []

    # 个股优先（过滤非主板）
    if stocks:
        for s in stocks:
            code = str(s.get("code", ""))
            if code.startswith("300") or code.startswith("301") or code.startswith("688"):
                continue
            s["_type"] = "stock"
            s["_sort_score"] = s.get("quality_score", 0)
            all_candidates.append(s)

    # ETF 作为补充
    if candidates:
        for c in candidates:
            c["_type"] = "etf"
            c["_sort_score"] = c.get("pullback_score", 0)
            all_candidates.append(c)

    # 去重 + 按质量分排
    seen_names = set()
    unique = []
    for c in sorted(all_candidates, key=lambda x: x.get("_sort_score", 0), reverse=True):
        name = c.get("name", "")
        if name not in seen_names:
            seen_names.add(name)
            unique.append(c)

    # 构建推荐 + 算星 + 过滤
    recs = []
    sector_count = {}
    for c in unique:
        if len(recs) >= max_recs:
            break
        sec = c.get("sector", "其他")
        if sector_count.get(sec, 0) >= 2:
            continue

        rec = _build_recommendation(c, market_state, available_capital, chase_level)
        stars = rec.get("stars", 0)
        if stars < _MIN_STARS_TO_RECOMMEND:
            continue

        recs.append(rec)
        sector_count[sec] = sector_count.get(sec, 0) + 1

    for i, r in enumerate(recs):
        r["recommendation_id"] = i + 1

    return recs


# ===========================================================================
# 单条推荐
# ===========================================================================

def _build_recommendation(
    c: dict,
    market_state: dict,
    capital: float,
    chase_level: str,
) -> dict:
    ctype = c.get("_type", "etf")
    price = c.get("current_price") or c.get("price", 0)
    code = c.get("code", "")
    name = c.get("name", "")
    sector = c.get("sector", "")
    sector_rank = c.get("sector_rank", 0)

    # ——— 决定进场等级 ———
    entry_grade, entry_reason = _determine_entry_grade(c, market_state, chase_level)

    # ——— 计算星级 ———
    stars = _calculate_stars(c, entry_grade, market_state, chase_level)

    # ——— 仓位（基于星级） ———
    alloc_pct = _ALLOC_PER_STAR.get(stars, 0.05)
    total_amount = capital * alloc_pct

    # ——— 入场计划 ———
    ma20 = c.get("ma20") or price * 0.98
    stop_loss = round(price * (1 - _STOP_LOSS_PCT), 2)
    max_loss = round(total_amount * _STOP_LOSS_PCT, 2)

    entry_plan = {
        "entry_grade": entry_grade,
        "entry_reason": entry_reason,
        "current_price": round(price, 2),
        "ma20": round(ma20, 2) if ma20 else None,
        "stop_loss": stop_loss,
        "stop_loss_pct": round(_STOP_LOSS_PCT * 100, 1),
        "max_loss_amount": max_loss,
    }

    # ——— 分批计划 ———
    if entry_grade == "BUY_NOW" and stars >= 2:
        entry_plan["tranches"] = _build_tranches(price, total_amount)
    elif entry_grade == "WAIT_PULLBACK":
        target = round(ma20 * 1.01, 2) if ma20 else round(price * 0.97, 2)
        entry_plan["target_price"] = target
        entry_plan["tranches"] = _build_tranches(target, total_amount)
    else:
        entry_plan["tranches"] = []

    # ——— 仓位计划 ———
    position_plan = {
        "total_allocation_pct": round(alloc_pct * 100, 1),
        "total_amount": round(total_amount, 2),
    }

    # ——— 技术面 ———
    technical = {
        "ma5": c.get("ma5"),
        "ma20": c.get("ma20"),
        "ma60": c.get("ma60"),
        "high_20d": c.get("high_20d"),
        "dist_to_high_pct": c.get("dist_to_high_pct"),
        "dist_to_ma20_pct": c.get("dist_to_ma20_pct"),
        "vol_ok": c.get("vol_ok"),
        "trend": c.get("trend_label", ""),
    }

    # ——— 理由 ———
    reasoning = _build_reasoning(c, market_state, entry_grade, entry_reason, sector, sector_rank, ctype, stars)

    # ——— 置信度 ———
    if stars >= 5:
        confidence = "高"
    elif stars >= 3:
        confidence = "中"
    else:
        confidence = "低"

    return {
        "type": ctype.upper(),
        "code": code,
        "name": name,
        "sector": sector,
        "sector_rank": sector_rank,
        "current_price": round(price, 2),
        "change_pct": c.get("change_pct"),
        "turnover_rate": c.get("turnover_rate"),
        "stars": stars,
        "stars_label": "★" * stars + "☆" * (5 - stars),
        "alloc_pct": round(alloc_pct * 100, 1),
        "entry_plan": entry_plan,
        "position_plan": position_plan,
        "technical": technical,
        "reasoning": reasoning,
        "confidence": confidence,
        "quote_time": c.get("quote_time"),
        "quote_age_seconds": c.get("quote_age_seconds"),
        "quote_source": c.get("quote_source"),
        "market_status": c.get("market_status"),
    }


# ===========================================================================
# 星级计算
# ===========================================================================

def _calculate_stars(c: dict, entry_grade: str, market_state: dict, chase_level: str) -> int:
    """综合评估，输出 0-5 星。

    考量维度：
    1. 技术质量分 (quality_score / pullback_score)
    2. 进场等级 (BUY_NOW / WAIT_PULLBACK / AVOID_CHASE)
    3. 市场状态 (进攻/震荡/防守)
    4. 追涨风险级别
    5. 回踩深度、均线位置
    """
    if entry_grade == "AVOID_CHASE":
        return 0

    quality = c.get("quality_score") or c.get("pullback_score", 0)
    dist_high = c.get("dist_to_high_pct", 0) or 0
    dist_ma = c.get("dist_to_ma20_pct", 0) or 0
    vol_ok = c.get("vol_ok", True)
    consec_up = c.get("consecutive_up", 0)
    sector_chase = c.get("sector_chase_level", "normal")

    stars = 0.0

    # ---- 技术质量基础分 ----
    if quality >= 16:
        stars += 2.5
    elif quality >= 12:
        stars += 2.0
    elif quality >= 8:
        stars += 1.5
    elif quality >= 5:
        stars += 1.0
    else:
        stars += 0.0

    # ---- 进场等级 ----
    if entry_grade == "BUY_NOW":
        stars += 1.5
    elif entry_grade == "WAIT_PULLBACK":
        stars += 0.5

    # ---- 回踩质量 ----
    if -8 <= dist_high <= -2:
        stars += 1.0
    elif -2 < dist_high <= -0.5:
        stars += 0.5
    elif dist_high > 0:
        stars -= 0.5  # 高于前高，追涨风险

    # ---- 均线位置 ----
    dist_ma_abs = abs(dist_ma)
    if dist_ma_abs <= 2:
        stars += 0.5
    elif dist_ma_abs <= 5:
        stars += 0.2

    # ---- 量能 ----
    if vol_ok:
        stars += 0.3

    # ---- 连涨惩罚 ----
    if consec_up >= 5:
        stars -= 1.0
    elif consec_up >= 3:
        stars -= 0.3

    # ---- 市场顺风/逆风 ----
    composite = market_state.get("composite", "")
    if composite in ("进攻市",):
        stars += 0.5
    elif composite in ("震荡偏多",):
        stars += 0.3
    elif composite in ("防守市", "修复反弹（弱）"):
        stars -= 0.5

    # ---- 板块追涨风险 ----
    if sector_chase == "danger":
        stars -= 1.5

    # ---- 市场追涨风险 ----
    if chase_level == "高":
        stars -= 1.5
    elif chase_level == "中":
        stars -= 0.5

    return max(0, min(5, round(stars)))


# ===========================================================================
# 进场等级
# ===========================================================================

def _determine_entry_grade(c: dict, market_state: dict, chase_level: str) -> tuple:
    """判断是否可以现在买，还是要等回调。"""
    quality = c.get("quality_score") or c.get("pullback_score", 0)
    dist_high = c.get("dist_to_high_pct", 0) or 0
    dist_ma = c.get("dist_to_ma20_pct", 0) or 0
    pct = c.get("change_pct", 0) or 0
    consec_up = c.get("consecutive_up", 0)
    sector_chase = c.get("sector_chase_level", "normal")

    if sector_chase == "danger" or chase_level == "高":
        return "AVOID_CHASE", "板块或市场追涨风险高，不建议此时入场"

    if consec_up >= 5:
        return "WAIT_PULLBACK", f"已连涨{consec_up}天，短期获利盘压力大，等回踩"

    if pct > 9 and dist_high > -2:
        return "AVOID_CHASE", f"今日涨停+已接近20日高点，追涨风险极大"

    if pct > 9 and dist_high <= -2:
        note = f"今日涨{pct:.1f}%但距20日高点仍差{abs(dist_high):.1f}%，超跌反弹早期" if dist_high <= -5 else f"今日强劲反弹，距高点{abs(dist_high):.1f}%，仍有空间"
        return "BUY_NOW", note

    if pct > 5 and dist_high <= -2 and dist_ma <= 5:
        return "BUY_NOW", f"回踩{abs(dist_high):.1f}%，今日温和放量上涨，可入场"

    if pct > 7:
        if dist_high > -3:
            return "WAIT_PULLBACK", f"今日涨幅{pct:.1f}%偏大，回踩空间不够，等回调"

    if quality >= 10 and dist_high <= -0.5 and dist_ma <= 4:
        return "BUY_NOW", f"回踩{abs(dist_high):.1f}%到位，距MA20仅{dist_ma:.1f}%，入场位置好"

    if quality >= 7:
        return "BUY_NOW", "技术面良好，可分批建仓"

    return "WAIT_PULLBACK", "条件尚不完全满足，建议等更好买点"


# ===========================================================================
# 分批计划
# ===========================================================================

def _build_tranches(base_price: float, total: float) -> list[dict]:
    tranches = []
    for i, (ratio, offset) in enumerate(zip(_TRANCHE_RATIOS, _TRANCHE_OFFSETS)):
        bp = round(base_price * (1 + offset), 3)
        amt = total * ratio
        qty = max(100, int(amt / bp) // 100 * 100) if bp > 0 else 0
        if i == 0:
            note, ptype = "当前价位买入", "市价"
        elif i == 1:
            note, ptype = f"回落至{bp}加仓", "限价"
        else:
            note, ptype = f"深度回落至{bp}加仓", "限价"
        tranches.append({
            "batch": i + 1, "ratio": int(ratio * 100),
            "amount": round(amt, 2), "price": bp,
            "quantity": qty, "price_type": ptype, "note": note,
        })
    return tranches


# ===========================================================================
# 理由生成
# ===========================================================================

def _build_reasoning(
    c: dict, market_state: dict, grade: str, grade_reason: str,
    sector: str, sector_rank: int, ctype: str, stars: int,
) -> str:
    parts = []
    pct = c.get("change_pct", 0) or 0
    dist_high = c.get("dist_to_high_pct", 0) or 0

    label = "个股" if ctype == "stock" else "ETF"
    parts.append(f"{label} {c['name']}，所属{sector}板块排名第{sector_rank}。")

    if ctype == "stock" and c.get("consecutive_down", 0) > 0:
        parts.append(f"此前连跌{c['consecutive_down']}天后今日转涨，是健康的回调结束信号。")
    elif ctype == "stock" and c.get("consecutive_up", 0):
        parts.append(f"已连涨{c['consecutive_up']}天，趋势确立。")

    if dist_high <= -2:
        parts.append(f"从20日高点回调{abs(dist_high):.1f}%，回踩幅度健康。")
    elif dist_high < 0:
        parts.append(f"从20日高点小幅回踩{abs(dist_high):.1f}%。")

    if c.get("dist_to_ma20_pct", 0) is not None:
        dma = c["dist_to_ma20_pct"]
        if 0 <= dma <= 2:
            parts.append("紧贴MA20，是理想的低吸位置。")

    if c.get("vol_ok") is True:
        parts.append("成交量正常，无缩量风险。")

    # 星级说明
    star_note = {5: "★★★★★ 满星推荐，各项条件均优，建议重点配置",
                 4: "★★★★ 高胜率，技术面+市场环境配合良好",
                 3: "★★★ 标准买点，可分配一成仓位",
                 2: "★★ 条件可接受，建议小仓位或等更好确认",
                 1: "★ 勉强符合，安全边际较薄"}.get(stars, "")

    if grade == "BUY_NOW":
        parts.append(f"结论：{star_note}。{grade_reason}")
    elif grade == "WAIT_PULLBACK":
        parts.append(f"结论：{star_note}。{grade_reason}")
    else:
        parts.append(f"结论：不建议追高。{grade_reason}")

    sl = round(c.get("current_price", c.get("price", 0)) * (1 - _STOP_LOSS_PCT), 2)
    parts.append(f"止损{sl}（-{round(_STOP_LOSS_PCT*100,1)}%），触及无条件离场。")

    return "".join(parts)
