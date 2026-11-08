#!/usr/bin/env bash
# Userspace Ollama, without root or systemd.
#
# The release tarball is extracted into ~/.local and the server runs as an
# ordinary background process, configured through environment variables.
#
#   ./scripts/ollama.sh install   download and extract into ~/.local
#   ./scripts/ollama.sh start
#   ./scripts/ollama.sh stop
#   ./scripts/ollama.sh status
#   ./scripts/ollama.sh models    pull the models the pipeline needs
set -euo pipefail

PREFIX="${MYSCHOLAR_OLLAMA_PREFIX:-$HOME/.local}"
BIN="$PREFIX/bin/ollama"
PORT="${MYSCHOLAR_OLLAMA_PORT:-11434}"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/myscholar"
LOGFILE="$STATE/ollama.log"
PIDFILE="$STATE/ollama.pid"

# Answer synthesis.
GENERATOR="${MYSCHOLAR_GENERATOR_MODEL:-qwen3:14b}"
# Classification and checks, called several times per question, so it must be fast.
# It must be -instruct: plain qwen3:4b spends its whole token budget reasoning and
# short calls come back empty.
UTILITY="${MYSCHOLAR_UTILITY_MODEL:-qwen3:4b-instruct}"

info() { printf '\033[36m==>\033[0m %s\n' "$1"; }
ok()   { printf '\033[32mok\033[0m  %s\n' "$1"; }
die()  { printf '\033[31merror\033[0m %s\n' "$1" >&2; exit 1; }

serving() { curl -sf "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1; }

cmd_install() {
    if [[ -x "$BIN" ]]; then
        ok "already installed: $("$BIN" --version 2>&1 | grep -o '[0-9.]*' | head -1)"
        return
    fi
    info "downloading the latest linux-amd64 release"
    mkdir -p "$PREFIX" "$STATE"
    local tmp
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' RETURN
    # Newer releases ship .tar.zst, older ones .tgz.
    if curl -fsSL "https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst" \
            -o "$tmp/ollama.tar.zst" 2>/dev/null; then
        tar --zstd -xf "$tmp/ollama.tar.zst" -C "$PREFIX"
    elif curl -fsSL "https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tgz" \
            -o "$tmp/ollama.tgz" 2>/dev/null; then
        tar -xzf "$tmp/ollama.tgz" -C "$PREFIX"
    else
        die "could not download an ollama release asset"
    fi
    [[ -x "$BIN" ]] || die "extraction did not produce $BIN"
    ok "installed to $BIN"
}

cmd_start() {
    [[ -x "$BIN" ]] || die "ollama not installed; run: $0 install"
    if serving; then
        ok "already serving on port $PORT"
        return
    fi
    mkdir -p "$STATE"
    info "starting ollama on port $PORT"

    # KEEP_ALIVE=-1 keeps weights loaded; Ollama otherwise evicts after five idle
    # minutes. MAX_LOADED_MODELS=2 because each question alternates between the
    # generator and the utility model.
    #
    # NUM_PARALLEL=1: four slots allocate four KV caches per model (qwen3:14b at
    # 38GB, qwen3:4b-instruct at 22GB), so both could not stay resident and every
    # switch forced a reload. A stalled reload ran past the 90s reasoning deadline
    # and the question was refused with no error logged.
    #
    # CONTEXT_LENGTH: with one slot and no cap, qwen3:4b-instruct gets its full
    # 262144-token window (45GB) and spills to the CPU. The largest prompt is
    # about 6,000 tokens, so 16384 leaves room.
    OLLAMA_HOST="127.0.0.1:$PORT" \
    OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_MAX_LOADED_MODELS=2 \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_CONTEXT_LENGTH=16384 \
        nohup "$BIN" serve >"$LOGFILE" 2>&1 &
    echo $! >"$PIDFILE"

    for _ in $(seq 1 30); do
        serving && { ok "serving on port $PORT (log: $LOGFILE)"; return; }
        sleep 1
    done
    die "ollama did not come up; see $LOGFILE"
}

cmd_stop() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        kill "$(cat "$PIDFILE")"
        rm -f "$PIDFILE"
        ok "stopped"
    else
        pkill -f "$BIN serve" 2>/dev/null && ok "stopped" || ok "not running"
    fi
}

cmd_status() {
    if serving; then
        ok "serving on port $PORT"
        OLLAMA_HOST="127.0.0.1:$PORT" "$BIN" list
    else
        echo "not running"
        return 1
    fi
}

cmd_models() {
    serving || cmd_start
    for model in "$GENERATOR" "$UTILITY"; do
        if OLLAMA_HOST="127.0.0.1:$PORT" "$BIN" list | awk '{print $1}' | grep -qx "$model"; then
            ok "$model already present"
        else
            info "pulling $model"
            OLLAMA_HOST="127.0.0.1:$PORT" "$BIN" pull "$model"
        fi
    done
    echo
    ok "models ready"
    echo
    echo "for .env:"
    echo "  MYSCHOLAR_LLM__PROVIDER=ollama"
    echo "  MYSCHOLAR_LLM__BASE_URL=http://127.0.0.1:$PORT"
    echo "  MYSCHOLAR_LLM__GENERATOR_MODEL=$GENERATOR"
    echo "  MYSCHOLAR_LLM__UTILITY_MODEL=$UTILITY"
}

case "${1:-}" in
    install) cmd_install ;;
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    models)  cmd_models ;;
    *) die "usage: $0 {install|start|stop|status|models}" ;;
esac
