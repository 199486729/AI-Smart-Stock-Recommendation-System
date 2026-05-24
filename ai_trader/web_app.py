"""AlphaMind AI Trader V2 — Web Dashboard."""

import json
import threading
import time
from datetime import datetime

from flask import Flask, jsonify, render_template, request

from ai_trader.agent.market_agent import MarketAgent
from ai_trader.agent.signal_agent import SignalAgent
from ai_trader.execution.paper_broker import PaperBroker

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
