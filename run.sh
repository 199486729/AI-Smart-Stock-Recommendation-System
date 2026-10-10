#!/bin/bash
# AlphaMind A-Share Trader — Dual-instance launcher
# Usage: ./run.sh {start|stop|restart|rolling|status}
#
# start   — launch both instances (5001, 5002)
# stop    — stop both
# restart — rolling restart (zero-downtime): start new on 5002, wait, stop 5001, restart 5001
# status  — health check both

set -euo pipefail
cd "$(dirname "$0")"

PORT1=5001
PORT2=5002
LOG1=/tmp/flask-${PORT1}.log
LOG2=/tmp/flask-${PORT2}.log
PYTHON=./venv/bin/python
APP="from ai_trader.web_app import app; import logging; log = logging.getLogger('werkzeug'); log.setLevel(logging.ERROR); app.run(host='0.0.0.0', port="

start_one() {
    local port=$1 log=$2
    if ss -ltn "sport = :$port" 2>/dev/null | grep -q ":$port "; then
        echo "[WARN] Port $port already in use, skipping"
        return 1
    fi
    echo "[START] Flask on :$port → log $log"
    nohup "$PYTHON" -c "${APP}${port}, debug=False, threaded=True)" > "$log" 2>&1 &
    local pid=$!
    echo "        PID=$pid"
}

stop_one() {
    local port=$1
    local pids
    pids=$(lsof -t -i :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "[STOP] Killing :$port → PID $pids"
        kill $pids 2>/dev/null || true
        sleep 1
        kill -9 $pids 2>/dev/null || true
    else
        echo "[STOP] :$port — no process"
    fi
}

health() {
    local port=$1
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://127.0.0.1:${port}/ashare" 2>/dev/null || echo "000")
    echo "$code"
}

wait_ready() {
    local port=$1 max=${2:-30}
    echo -n "[WAIT] :$port"
    for _ in $(seq 1 $max); do
        local code
        code=$(health "$port")
        if [ "$code" = "200" ] || [ "$code" = "302" ]; then
            echo " OK (HTTP $code)"
            return 0
        fi
        sleep 1
        echo -n "."
    done
    echo " TIMEOUT"
    return 1
}

# ─── Commands ──────────────────────────────────────────────────

start() {
    start_one $PORT1 "$LOG1"
    start_one $PORT2 "$LOG2"
    sleep 2
    wait_ready $PORT1
    wait_ready $PORT2
    echo "[OK] Both instances running — http://127.0.0.1:${PORT1}/ashare  http://127.0.0.1:${PORT2}/ashare"
}

stop() {
    stop_one $PORT1
    stop_one $PORT2
}

restart() {
    # Rolling (zero-downtime) restart
    echo "=== Rolling restart ==="
    echo "[STEP 1/4] Starting new instance on :$PORT2"
    stop_one $PORT2  # ensure clean
    start_one $PORT2 "$LOG2"
    wait_ready $PORT2 || { echo "[FATAL] New instance failed to start, aborting"; exit 1; }

    echo "[STEP 2/4] Stopping old instance on :$PORT1"
    stop_one $PORT1
    sleep 1

    echo "[STEP 3/4] Restarting :$PORT1"
    start_one $PORT1 "$LOG1"
    wait_ready $PORT1 || echo "[WARN] :$PORT1 did not come up — :$PORT2 still serving"

    echo "[STEP 4/4] Done. Both instances live."
    status
}

status() {
    echo "=== Status ==="
    for port in $PORT1 $PORT2; do
        local code
        code=$(health "$port" || echo "000")
        local label="DOWN"
        [ "$code" = "200" ] || [ "$code" = "302" ] && label="UP"
        echo "  :$port → HTTP $code [$label]"
    done
}

# ─── Dispatch ──────────────────────────────────────────────────

case "${1:-}" in
    start)    start ;;
    stop)     stop ;;
    restart)  restart ;;
    rolling)  restart ;;  # alias
    status)   status ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "  start    Launch Flask on :$PORT1 + :$PORT2"
        echo "  stop     Kill both"
        echo "  restart  Zero-downtime rolling restart"
        echo "  status   Health check"
        exit 1
        ;;
esac
