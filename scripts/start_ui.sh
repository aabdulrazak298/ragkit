#!/usr/bin/env bash
# Start the rag-kit Gradio UI locally (Linux/WSL).
#
#   ./scripts/start_ui.sh                 # foreground, http://127.0.0.1:7860
#   ./scripts/start_ui.sh -d              # background daemon (log: ~/.rag-kit/ui.log)
#   ./scripts/start_ui.sh --stop          # stop the daemon
#   ./scripts/start_ui.sh -p 8000 --share # forward any extra flag to `rag-kit ui`
#
# Auto-detects the repo venv and adds torch's bundled nvidia CUDA libs to
# LD_LIBRARY_PATH so docling's GPU OCR works without manual exports.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PIDFILE="$HOME/.rag-kit/ragkit-ui.pid"
LOG="$HOME/.rag-kit/ui.log"

PORT=7860
HOST=127.0.0.1
DAEMON=0
STOP=0
NO_BROWSER=0
DB=""
EXTRA_ARGS=()

# ── venv discovery ─────────────────────────────────────────────────────
PY=""
for cand in "$REPO_DIR/.venv-test/bin/python" "$REPO_DIR/.venv/bin/python"; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then PY="$(command -v python3 || echo python3)"; fi

# ── CUDA libs for docling OCR (torch's bundled nvidia pip packages) ────
if SITE_PACKAGES="$("$PY" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)" \
    && [ -n "$SITE_PACKAGES" ] && [ -d "$SITE_PACKAGES/nvidia" ]; then
    NVIDIA_LIBS="$(ls -d "$SITE_PACKAGES"/nvidia/*/lib 2>/dev/null | paste -sd: -)"
    if [ -n "$NVIDIA_LIBS" ]; then
        export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "[ui] CUDA libs from torch's nvidia packages added to LD_LIBRARY_PATH (GPU OCR)"
    fi
fi

# ── arg parsing ────────────────────────────────────────────────────────
while [ "$#" -gt 0 ]; do
    case "$1" in
        -p|--port) PORT="$2"; shift 2 ;;
        -H|--host) HOST="$2"; shift 2 ;;
        -d|--daemon) DAEMON=1; shift ;;
        --stop) STOP=1; shift ;;
        --no-browser) NO_BROWSER=1; shift ;;
        --db) DB="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

stop_daemon() {
    if [ -f "$PIDFILE" ]; then
        kill "$(cat "$PIDFILE")" 2>/dev/null && echo "[ui] stopped (pid $(cat "$PIDFILE"))"
        rm -f "$PIDFILE"
    else
        echo "[ui] no daemon running (no $PIDFILE)"
    fi
    exit 0
}
[ "$STOP" = 1 ] && stop_daemon

URL="http://$HOST:$PORT"

wait_for_server() {
    local i
    for i in $(seq 1 60); do
        if curl -sf -o /dev/null "$URL" 2>/dev/null; then return 0; fi
        sleep 1
    done
    echo "[ui] server did not respond at $URL after 60s — check the log: $LOG" >&2
    return 1
}

open_browser() {
    if [ "$NO_BROWSER" = 1 ]; then return; fi
    if command -v wslview >/dev/null 2>&1; then wslview "$URL" >/dev/null 2>&1 &
    elif command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$URL" >/dev/null 2>&1 &
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 &
    fi
}

CMD=("$PY" -m rag_kit)
[ -n "$DB" ] && CMD+=(--db "$DB")
CMD+=(ui --host "$HOST" --port "$PORT" "${EXTRA_ARGS[@]}")

if [ "$DAEMON" = 1 ]; then
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "[ui] already running (pid $(cat "$PIDFILE")) at $URL"
        exit 0
    fi
    mkdir -p "$HOME/.rag-kit"
    nohup "${CMD[@]}" >"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    echo "[ui] starting in background (pid $!): $URL"
    echo "[ui] log: $LOG   |   stop with: $0 --stop"
    wait_for_server && open_browser
else
    echo "[ui] starting: $URL   (Ctrl+C to stop)"
    "${CMD[@]}" &
    UI_PID=$!
    wait_for_server && open_browser
    wait "$UI_PID"
fi
