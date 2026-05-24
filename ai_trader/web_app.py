"""AlphaMind AI Trader V2 — Web Dashboard."""

import json
import threading
import time
from datetime import datetime

from flask import Flask, jsonify, render_template, request

from ai_trader.agent.market_agent import MarketAgent
from ai_trader.agent.signal_agent import SignalAgent
from ai_trader.execution.paper_broker import PaperBroker
from ai_trader.analysis.data_fetcher import fetch_all_data
from ai_trader.analysis.market_state import compute_market_state
from ai_trader.analysis.sector_ranker import rank_sectors
from ai_trader.analysis.etf_screener import screen_etfs
from ai_trader.analysis.stock_screener import screen_stocks
from ai_trader.analysis.recommender import generate_recommendations

app = Flask(__name__)

market = MarketAgent(cache_ttl=30)
signal = SignalAgent()
broker = PaperBroker()

# In-memory state for the dashboard
_state = {
    "market_data": None,
    "trade_signal": None,
    "last_result": None,
    "last_updated": None,
    "error": None,
    "auto_trade": False,
}

_ashare_state = {
    "market_state": None,
    "ranked_sectors": None,
    "recommendations": None,
    "stocks": None,
    "data_quality": None,
    "last_updated": None,
    "error": None,
}


_lock = threading.Lock()


def _run_cycle():
    """Fetch data, generate signal, optionally execute."""
    data = market.fetch_market_data("XAUUSD")
    sig = signal.generate(data)

    result = None
    if _state.get("auto_trade") and signal.should_act(sig):
        result = broker.execute(sig)
    elif not _state.get("auto_trade"):
        result = {"status": "MANUAL", "reason": "Auto-trade disabled; use manual execute"}

    with _lock:
        _state["market_data"] = data
        _state["trade_signal"] = sig
        _state["last_result"] = result
        _state["last_updated"] = datetime.now().isoformat()
        _state["error"] = None


@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------------------------------------------------
# A-Share 智能推荐系统路由
# -----------------------------------------------------------------------

@app.route("/ashare")
def ashare():
    return render_template("ashare_dashboard.html")


@app.route("/api/ashare/state")
def api_ashare_state():
    with _lock:
        return jsonify({
            "market_state": _ashare_state["market_state"],
            "ranked_sectors": _ashare_state["ranked_sectors"],
            "recommendations": _ashare_state["recommendations"],
            "stocks": _ashare_state["stocks"],
            "data_quality": _ashare_state["data_quality"],
            "last_updated": _ashare_state["last_updated"],
            "error": _ashare_state["error"],
        })


@app.route("/api/ashare/analyze", methods=["POST"])
def api_ashare_analyze():
    try:
        # 1. 拉取全部数据
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

        # 2. 市场状态分类
        market_state = compute_market_state(
            indices=all_data.get("indices"),
            breadth=all_data.get("breadth"),
            north_flow=all_data.get("north_flow"),
        )

        # 3. 板块排名
        sector_result = rank_sectors(
            boards=all_data.get("boards"),
            industry_flow=all_data.get("industry_flow"),
            sectors=all_data.get("sectors"),
        )

        # 4. 个股筛选（优先）
        screened_stocks = screen_stocks(
            top_sectors=sector_result.get("top_sectors", []),
            leading_stocks=sector_result.get("leading_stocks", []),
            limit_up_pool=all_data.get("limit_up"),
            stock_spot=all_data.get("stock_spot"),
            max_deep_checks=80,
            max_final=15,
        )

        # 5. ETF 筛选（作为补充）
        etf_candidates = screen_etfs(
            top_sectors=sector_result.get("top_sectors", []),
            etfs=all_data.get("etfs"),
        )

        # 6. 生成推荐（个股优先）
        recs = generate_recommendations(
            market_state=market_state,
            top_sectors=sector_result.get("top_sectors", []),
            candidates=etf_candidates,
            stocks=screened_stocks,
            max_recs=10,
        )

        update_time = datetime.now().isoformat()

        with _lock:
            _ashare_state["market_state"] = market_state
            _ashare_state["ranked_sectors"] = sector_result.get("top_sectors", [])
            _ashare_state["recommendations"] = recs
            _ashare_state["stocks"] = screened_stocks
            _ashare_state["data_quality"] = data_quality
            _ashare_state["last_updated"] = update_time
            _ashare_state["error"] = None if not errors else "; ".join(errors)

        return jsonify({
            "ok": True,
            "market_state": market_state,
            "ranked_sectors": sector_result.get("top_sectors", []),
            "focus_sectors": sector_result.get("focus_in_top", []),
            "recommendations": recs,
            "screened_stocks": screened_stocks,
            "data_quality": data_quality,
            "last_updated": update_time,
            "elapsed": all_data.get("elapsed", 0),
        })

    except Exception as e:
        with _lock:
            _ashare_state["error"] = str(e)
        return jsonify({"ok": False, "error": str(e)}), 500


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
        return jsonify({"ok": False, "error": "No signal yet; refresh first"}), 400
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
    """Background thread: auto-refresh when auto_trade is on."""
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
