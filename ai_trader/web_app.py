"""AlphaMind AI Trader V2 — Web Dashboard（含认证 + 持久化）。"""

import json
import os
import threading
import time
from datetime import datetime, timedelta

from anthropic import Anthropic
from flask import Flask, jsonify, render_template, request, Response, session, redirect, url_for, g

from ai_trader.agent.market_agent import MarketAgent
from ai_trader.agent.signal_agent import SignalAgent
from ai_trader.execution.paper_broker import PaperBroker
from ai_trader.analysis.data_fetcher import fetch_all_data
from ai_trader.analysis.market_state import compute_market_state
from ai_trader.analysis.sector_ranker import rank_sectors
from ai_trader.analysis.etf_screener import screen_etfs
from ai_trader.analysis.stock_screener import screen_stocks
from ai_trader.analysis.recommender import generate_recommendations
from ai_trader.analysis.stock_realtime import get_stock_realtime
from ai_trader.trade_journal import record_trade, list_trades, get_summary as trade_summary
from ai_trader.auth import auth_bp, login_required, current_user
from ai_trader.persistence import (
    create_chat_session, get_chat_sessions, update_session_title,
    save_message, get_messages,
    save_recommendation_snapshot, get_recommendation_snapshots, get_snapshot_detail,
    save_fact, get_facts,
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "alphamind-dev-key-change-in-production")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

# 注册认证蓝图
app.register_blueprint(auth_bp)

# Anthropic-compatible AI client
_ai_client = Anthropic(
    base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
    api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN", ""),
)
_ai_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

market = MarketAgent(cache_ttl=30)
signal = SignalAgent()
broker = PaperBroker()

# ---------------------------------------------------------------------------
# XAUUSD trading state (unchanged)
# ---------------------------------------------------------------------------
_state = {
    "market_data": None, "trade_signal": None, "last_result": None,
    "last_updated": None, "error": None, "auto_trade": False,
}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 全局 before_request：未登录重定向
# ---------------------------------------------------------------------------

@app.before_request
def _check_auth():
    """非 API 页面需要登录，API 路由由 @login_required 保护。"""
    path = request.path
    if path.startswith("/api/auth/"):
        return
    if path.startswith("/login"):
        return
    if path.startswith("/api/"):
        return  # API 由各路由的 @login_required 控制
    # 静态资源不加限制
    if path.startswith("/static/"):
        return
    # 页面路由：未登录重定向到登录页
    if not session.get("user_id"):
        return redirect(url_for("login_page"))


# ---------------------------------------------------------------------------
# 页面路由
# ---------------------------------------------------------------------------

@app.route("/login")
def login_page():
    if session.get("user_id"):
        return redirect(url_for("ashare"))
    return render_template("login.html")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ashare")
def ashare():
    return render_template("ashare_dashboard.html")


# ---------------------------------------------------------------------------
# AI 聊天工具定义
# ---------------------------------------------------------------------------

_CHAT_TOOLS = [
    {
        "name": "get_stock_realtime",
        "description": (
            "获取某只A股或ETF的实时行情和技术面数据。"
            "返回当前价格、涨跌幅、换手率、成交额、市盈率，以及MA5/MA20/MA60均线、"
            "距20日高点距离、连涨连跌天数等。用于判断个股当前是否处于买点或卖点。"
            "行情延迟约3-5分钟。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "6位股票代码，如 600519、000001、159915"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "record_trade",
        "description": (
            "将用户的**真实买卖**记录写入交割单数据库（real_trades 表）。"
            "当用户说'我买了XX'、'我卖出了XX'、'帮我记一笔'时调用。"
            "务必在回复中告知用户已记录到真实交割单，并显示摘要。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "6位股票代码"},
                "name": {"type": "string", "description": "股票名称"},
                "side": {"type": "string", "enum": ["buy", "sell"], "description": "买入(buy)或卖出(sell)"},
                "price": {"type": "number", "description": "成交价格"},
                "quantity": {"type": "integer", "description": "成交股数（A股一手=100股）"},
                "amount": {"type": "number", "description": "成交金额（元），必填"},
                "notes": {"type": "string", "description": "备注，如买入理由"},
            },
            "required": ["code", "name", "side", "price", "amount"],
        },
    },
]


# ---------------------------------------------------------------------------
# 工具执行
# ---------------------------------------------------------------------------

def _execute_tool(tool_name: str, tool_input: dict) -> str:
    user_id = session.get("user_id", 0)
    if tool_name == "get_stock_realtime":
        code = str(tool_input.get("code", "")).strip()
        if not code:
            return json.dumps({"error": "请提供一个有效的股票代码"}, ensure_ascii=False)
        result = get_stock_realtime(code)
        return json.dumps(result, ensure_ascii=False, default=str)

    elif tool_name == "record_trade":
        code = str(tool_input.get("code", "")).strip()
        name = str(tool_input.get("name", "")).strip()
        side = str(tool_input.get("side", "buy")).strip()
        price = float(tool_input.get("price", 0))
        quantity = int(tool_input.get("quantity", 0))
        amount = float(tool_input.get("amount", 0))
        notes = str(tool_input.get("notes", ""))

        if quantity == 0 and price > 0:
            quantity = max(100, int(amount / price) // 100 * 100)

        result = record_trade(
            code=code, name=name, side=side,
            price=price, quantity=quantity, amount=amount, notes=notes,
            table="real", user_id=user_id,
        )
        summary = trade_summary(table="real", user_id=user_id)
        return json.dumps({
            "recorded": result,
            "summary": summary,
            "message": f"已记录{side} {name}({code}) @ {price} 金额{amount}元",
        }, ensure_ascii=False, default=str)

    return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)


def _build_system_prompt() -> str:
    """构建包含用户记忆、市场上下文的 system prompt。"""
    user_id = session.get("user_id", 0)
    parts = [
        "你是一个A股交易助手，可以调用工具获取实时个股数据、记录用户交易。",
        "基于实时数据和市场状态为用户提供分析。",
    ]

    # ── 用户关键事实（长期记忆）──
    if user_id:
        facts = get_facts(user_id)
        if facts:
            parts.append("\n== 用户交易偏好（长期记忆）==")
            for f in facts:
                parts.append(f"- {f['fact']}")

    # ── 近期市场数据 ──
    ms = _latest_market_state or {}
    sectors = _latest_sectors or []
    recs = _latest_recs or []

    parts.append("\n== 当前市场状态 ==")
    if ms:
        parts.append(f"市场状态：{ms.get('composite', '?')}，方向：{ms.get('direction', '?')}，置信度：{ms.get('conviction', '?')}")
        parts.append(f"追涨风险：{ms.get('chase_risk', {}).get('level', '?')}")
        parts.append(f"总结：{ms.get('summary', '无')}")
    else:
        parts.append("（尚未获取市场数据，请提示用户先点击「一键获取市场分析」）")

    if sectors:
        parts.append("\n前5强势板块：")
        for s in sectors[:5]:
            parts.append(f"  #{s['rank']} {s['name']} 涨跌{s.get('change_pct',0):+.2f}% 评分{s.get('score',0):.2f}")

    if recs:
        parts.append("\n当前推荐：")
        for r in recs:
            g = r.get('entry_plan', {}).get('entry_grade', '')
            gl = {"BUY_NOW": "可入", "WAIT_PULLBACK": "等回调", "AVOID_CHASE": "不追"}.get(g, g)
            parts.append(f"  {'★'*r.get('stars', 0)}{'☆'*(5-r.get('stars', 0))} {r['name']}({r['code']}) {r.get('type','')} {gl} {r.get('alloc_pct',0)}%仓位")

    # ── 近期会话摘要（中期记忆）──
    if user_id:
        recent_sessions = get_chat_sessions(user_id, limit=3)
        summaries = [s for s in recent_sessions if s.get("summary")]
        if summaries:
            parts.append("\n== 近期对话摘要 ==")
            for s in summaries:
                parts.append(f"- {s['updated_at']}: {s['summary']}")

    parts.append("\n用中文回答，简洁专业。当用户询问具体个股时，要主动调用 get_stock_realtime 获取实时数据再作答，不要编造。当用户说买了/卖了股票时，要主动调用 record_trade 记录。")
    return "\n".join(parts)


# ── 最新分析结果缓存（用于 system prompt 注入）──
_latest_market_state = None
_latest_sectors = None
_latest_recs = None
_ms_lock = threading.Lock()


# ---------------------------------------------------------------------------
# A-Share API（需登录）
# ---------------------------------------------------------------------------

@app.route("/api/ashare/state")
@login_required
def api_ashare_state():
    user_id = session["user_id"]
    return jsonify({
        "market_state": _latest_market_state,
        "ranked_sectors": _latest_sectors,
        "recommendations": _latest_recs,
        "user": current_user(),
    })


@app.route("/api/ashare/analyze", methods=["POST"])
@login_required
def api_ashare_analyze():
    global _latest_market_state, _latest_sectors, _latest_recs
    user_id = session["user_id"]

    try:
        all_data = fetch_all_data()
        errors = all_data.get("errors", [])

        data_quality = {
            "indices_ok": all_data.get("indices") is not None,
            "breadth_ok": all_data.get("breadth") is not None,
            "north_ok": all_data.get("north_flow") is not None,
            "boards_ok": all_data.get("boards") is not None,
            "industry_flow_ok": all_data.get("industry_flow") is not None,
            "sectors_ok": all_data.get("sectors") is not None,
            "etfs_ok": all_data.get("etfs") is not None,
            "limit_up_ok": all_data.get("limit_up") is not None,
            "stock_spot_ok": all_data.get("stock_spot") is not None,
            "errors": errors,
        }

        market_state = compute_market_state(
            indices=all_data.get("indices"),
            breadth=all_data.get("breadth"),
            north_flow=all_data.get("north_flow"),
        )

        sector_result = rank_sectors(
            boards=all_data.get("boards"),
            industry_flow=all_data.get("industry_flow"),
            sectors=all_data.get("sectors"),
        )

        screened_stocks = screen_stocks(
            top_sectors=sector_result.get("top_sectors", []),
            leading_stocks=sector_result.get("leading_stocks", []),
            limit_up_pool=all_data.get("limit_up"),
            stock_spot=all_data.get("stock_spot"),
            max_deep_checks=80,
            max_final=15,
        )

        etf_candidates = screen_etfs(
            top_sectors=sector_result.get("top_sectors", []),
            etfs=all_data.get("etfs"),
        )

        recs = generate_recommendations(
            market_state=market_state,
            top_sectors=sector_result.get("top_sectors", []),
            candidates=etf_candidates,
            stocks=screened_stocks,
        )

        update_time = datetime.now().isoformat()
        top_sectors = sector_result.get("top_sectors", [])

        # ── 持久化推荐快照 ──
        try:
            save_recommendation_snapshot(user_id, market_state, top_sectors, recs)
        except Exception:
            pass

        # ── 更新全局缓存（用于 AI system prompt）──
        with _ms_lock:
            _latest_market_state = market_state
            _latest_sectors = top_sectors
            _latest_recs = recs

        return jsonify({
            "ok": True,
            "market_state": market_state,
            "ranked_sectors": top_sectors,
            "focus_sectors": sector_result.get("focus_in_top", []),
            "recommendations": recs,
            "screened_stocks": screened_stocks,
            "data_quality": data_quality,
            "last_updated": update_time,
            "elapsed": all_data.get("elapsed", 0),
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# 分析历史
# ---------------------------------------------------------------------------

@app.route("/api/ashare/history")
@login_required
def api_ashare_history():
    user_id = session["user_id"]
    limit = request.args.get("limit", 10, type=int)
    snapshots = get_recommendation_snapshots(user_id, limit)
    return jsonify({"ok": True, "snapshots": snapshots})


@app.route("/api/ashare/history/<int:snap_id>")
@login_required
def api_ashare_history_detail(snap_id):
    user_id = session["user_id"]
    snap = get_snapshot_detail(snap_id, user_id)
    if not snap:
        return jsonify({"ok": False, "error": "快照不存在"}), 404
    return jsonify({"ok": True, "snapshot": {
        "id": snap["id"],
        "created_at": str(snap["created_at"]),
        "market_state": json.loads(snap["market_state_json"]) if isinstance(snap["market_state_json"], str) else snap["market_state_json"],
        "sectors": json.loads(snap["sectors_json"]) if isinstance(snap["sectors_json"], str) else snap["sectors_json"],
        "recommendations": json.loads(snap["recommendations_json"]) if isinstance(snap["recommendations_json"], str) else snap["recommendations_json"],
    }})


# ---------------------------------------------------------------------------
# AI 聊天 API
# ---------------------------------------------------------------------------

@app.route("/api/ashare/chat/sessions")
@login_required
def api_chat_sessions():
    user_id = session["user_id"]
    sessions = get_chat_sessions(user_id)
    return jsonify({"ok": True, "sessions": sessions})


@app.route("/api/ashare/chat/sessions", methods=["POST"])
@login_required
def api_create_chat_session():
    user_id = session["user_id"]
    body = request.get_json(force=True) or {}
    title = body.get("title", "新对话")
    sid = create_chat_session(user_id, title)
    return jsonify({"ok": True, "session_id": sid})


@app.route("/api/ashare/chat/sessions/<int:sid>/messages")
@login_required
def api_chat_messages(sid):
    messages = get_messages(sid)
    return jsonify({"ok": True, "messages": [
        {"role": m["role"], "content": m["content"], "created_at": str(m["created_at"])}
        for m in messages
    ]})


@app.route("/api/ashare/chat", methods=["POST"])
@login_required
def api_ashare_chat():
    """AI 对话 + MCP 工具调用（非流式，用于快速回复）。"""
    user_id = session["user_id"]
    try:
        body = request.get_json(force=True) or {}
        user_msg = (body.get("message") or "").strip()
        session_id = body.get("session_id", 0)
        if not user_msg:
            return jsonify({"ok": False, "error": "消息不能为空"}), 400

        # 新建会话
        if not session_id:
            title = user_msg[:30] + ("..." if len(user_msg) > 30 else "")
            session_id = create_chat_session(user_id, title)

        # 保存用户消息
        save_message(session_id, "user", user_msg)

        system_prompt = _build_system_prompt()

        # 加载历史消息
        db_messages = get_messages(session_id, limit=40)
        api_messages = []
        for m in db_messages:
            api_messages.append({"role": m["role"], "content": m["content"]})

        # MCP 循环
        for _ in range(4):
            resp = _ai_client.messages.create(
                model=_ai_model,
                max_tokens=2000,
                system=system_prompt,
                tools=_CHAT_TOOLS,
                messages=api_messages,
            )

            tool_uses = []
            text_parts = []
            for block in resp.content:
                if block.type == "tool_use":
                    tool_uses.append(block)
                elif block.type == "text" and hasattr(block, "text"):
                    text_parts.append(block.text)

            if not tool_uses:
                reply = "".join(text_parts)
                # 保存 AI 回复
                save_message(session_id, "assistant", reply)
                return jsonify({"ok": True, "reply": reply, "session_id": session_id})

            # 执行工具
            api_messages.append({
                "role": "assistant",
                "content": [b.model_dump() if hasattr(b, "model_dump") else dict(b) for b in resp.content],
            })

            tool_results = []
            for tu in tool_uses:
                result = _execute_tool(tu.name, tu.input if hasattr(tu, "input") else {})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result,
                })

            api_messages.append({"role": "user", "content": tool_results})

        # 超轮次，强制总结
        api_messages.append({"role": "user", "content": "请基于以上工具返回的数据，用中文给出简洁的交易分析和操作建议。"})
        resp = _ai_client.messages.create(
            model=_ai_model, max_tokens=2000, system=system_prompt, messages=api_messages,
        )
        reply = "".join([b.text for b in resp.content if b.type == "text" and hasattr(b, "text")]) or "抱歉，AI 未返回有效回复。"
        save_message(session_id, "assistant", reply)
        return jsonify({"ok": True, "reply": reply, "session_id": session_id})

    except Exception as e:
        return jsonify({"ok": False, "error": f"AI对话失败: {str(e)}"}), 500


@app.route("/api/ashare/chat/stream", methods=["POST"])
@login_required
def api_ashare_chat_stream():
    """AI 对话 SSE 流式输出 + MCP 工具调用 + 持久化。"""
    user_id = session["user_id"]

    body = request.get_json(force=True) or {}
    user_msg = (body.get("message") or "").strip()
    history = body.get("history", []) or []
    session_id = body.get("session_id", 0)
    system_prompt = _build_system_prompt()

    def generate():
        nonlocal session_id
        try:
            if not user_msg:
                yield _sse({"type": "error", "text": "消息不能为空"})
                return

            # 新建会话
            if not session_id:
                title = user_msg[:30] + ("..." if len(user_msg) > 30 else "")
                session_id = create_chat_session(user_id, title)

            # 保存用户消息
            save_message(session_id, "user", user_msg)

            # 构建消息列表
            messages = []
            for h in history[-30:]:
                role = h.get("role", "")
                content = h.get("content", "")
                if role in ("user", "assistant") and isinstance(content, str) and content:
                    messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": user_msg})

            # 第一轮：流式输出 + 检测 tool_use
            tool_uses = []
            with _ai_client.messages.stream(
                model=_ai_model, max_tokens=2000, system=system_prompt,
                tools=_CHAT_TOOLS, messages=messages,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        d = event.delta
                        if hasattr(d, "type") and d.type == "text_delta" and hasattr(d, "text"):
                            yield _sse({"type": "delta", "text": d.text})
                final_msg = stream.get_final_message()

            for block in final_msg.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_uses.append(block)

            if not tool_uses:
                # 提取完整回复并保存
                full_reply = "".join([
                    b.text for b in final_msg.content
                    if hasattr(b, "type") and b.type == "text" and hasattr(b, "text")
                ])
                if full_reply:
                    save_message(session_id, "assistant", full_reply)
                yield _sse({"type": "done", "session_id": session_id})
                return

            # 工具调用
            tool_names = [tu.name for tu in tool_uses]
            yield _sse({"type": "status", "text": f"\n\n> 调用工具: {', '.join(tool_names)}...\n\n"})

            messages.append({
                "role": "assistant",
                "content": [b.model_dump() if hasattr(b, "model_dump") else dict(b) for b in final_msg.content],
            })

            tool_results = []
            for tu in tool_uses:
                result = _execute_tool(tu.name, tu.input if hasattr(tu, "input") else {})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

            # 第二轮（不给工具，避免递归）
            full_reply2 = ""
            with _ai_client.messages.stream(
                model=_ai_model, max_tokens=2000, system=system_prompt, messages=messages,
            ) as stream2:
                for event in stream2:
                    if event.type == "content_block_delta":
                        d = event.delta
                        if hasattr(d, "type") and d.type == "text_delta" and hasattr(d, "text"):
                            full_reply2 += d.text
                            yield _sse({"type": "delta", "text": d.text})

            if full_reply2:
                save_message(session_id, "assistant", full_reply2)
            yield _sse({"type": "done", "session_id": session_id})

        except Exception as e:
            yield _sse({"type": "error", "text": f"AI对话失败: {str(e)}"})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# 交易日志 API
# ---------------------------------------------------------------------------

@app.route("/api/ashare/trades")
@login_required
def api_trades():
    user_id = session["user_id"]
    limit = request.args.get("limit", 50, type=int)
    table = request.args.get("type", "all")
    return jsonify({
        "trades": list_trades(limit, table=table, user_id=user_id),
        "summary": trade_summary(table=table, user_id=user_id),
    })


@app.route("/api/ashare/trades/paper")
@login_required
def api_trades_paper():
    user_id = session["user_id"]
    limit = request.args.get("limit", 50, type=int)
    return jsonify({
        "trades": list_trades(limit, table="paper", user_id=user_id),
        "summary": trade_summary(table="paper", user_id=user_id),
    })


@app.route("/api/ashare/trades/real")
@login_required
def api_trades_real():
    user_id = session["user_id"]
    limit = request.args.get("limit", 50, type=int)
    return jsonify({
        "trades": list_trades(limit, table="real", user_id=user_id),
        "summary": trade_summary(table="real", user_id=user_id),
    })


@app.route("/api/ashare/trades/record", methods=["POST"])
@login_required
def api_trades_record():
    user_id = session["user_id"]
    try:
        body = request.get_json(force=True) or {}
        code = str(body.get("code", "")).strip()
        name = str(body.get("name", "")).strip()
        side = str(body.get("side", "buy")).strip()
        price = float(body.get("price", 0))
        quantity = int(body.get("quantity", 0))
        amount = float(body.get("amount", 0))
        notes = str(body.get("notes", ""))
        table = str(body.get("table", "real")).strip()

        if table not in ("paper", "real"):
            return jsonify({"ok": False, "error": "table 必须为 paper 或 real"}), 400
        if not code or price <= 0 or amount <= 0:
            return jsonify({"ok": False, "error": "code/price/amount 为必填且需>0"}), 400

        result = record_trade(code, name, side, price, quantity, amount, notes,
                              table=table, user_id=user_id)
        return jsonify({"ok": True, "trade": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ---------------------------------------------------------------------------
# 原有 XAUUSD 路由（保持不变）
# ---------------------------------------------------------------------------

def _run_cycle():
    data = market.fetch_market_data("XAUUSD")
    sig = signal.generate(data)
    result = None
    if _state.get("auto_trade") and signal.should_act(sig):
        result = broker.execute(sig)
    elif not _state.get("auto_trade"):
        result = {"status": "MANUAL", "reason": "Auto-trade disabled"}
    with _lock:
        _state["market_data"] = data
        _state["trade_signal"] = sig
        _state["last_result"] = result
        _state["last_updated"] = datetime.now().isoformat()
        _state["error"] = None


@app.route("/api/state")
def api_state():
    with _lock:
        return jsonify({
            "market_data": _state["market_data"],
            "trade_signal": _state["trade_signal"],
            "last_result": _state["last_result"],
            "last_updated": _state["last_updated"],
            "error": _state["error"],
            "auto_trade": _state["auto_trade"],
            "summary": broker.get_summary(),
            "trade_log": broker.trade_log[-20:],
        })


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    try:
        _run_cycle()
        with _lock:
            return jsonify({"ok": True, "last_updated": _state["last_updated"]})
    except Exception as e:
        with _lock:
            _state["error"] = str(e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/execute", methods=["POST"])
def api_execute():
    with _lock:
        sig = _state.get("trade_signal")
    if sig is None:
        return jsonify({"ok": False, "error": "No signal yet"}), 400
    if not signal.should_act(sig):
        return jsonify({"ok": False, "error": "Signal not actionable"}), 400
    result = broker.execute(sig)
    with _lock:
        _state["last_result"] = result
    return jsonify({"ok": True, "result": result})


@app.route("/api/auto", methods=["POST"])
def api_auto():
    enabled = request.json.get("enabled", False)
    with _lock:
        _state["auto_trade"] = enabled
    return jsonify({"ok": True, "auto_trade": enabled})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global broker
    broker = PaperBroker()
    with _lock:
        _state["last_result"] = None
        _state["trade_log"] = []
    return jsonify({"ok": True, "summary": broker.get_summary()})


def _auto_loop():
    while True:
        time.sleep(30)
        try:
            with _lock:
                auto = _state.get("auto_trade", False)
            if auto:
                _run_cycle()
        except Exception:
            pass


def main():
    threading.Thread(target=_auto_loop, daemon=True).start()
    print("=" * 56)
    print("  AlphaMind AI Trader V2 — Web Dashboard")
    print("  http://127.0.0.1:5000")
    print("=" * 56)
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
