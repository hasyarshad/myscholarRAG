#!/usr/bin/env bash
# Userspace Ollama — no root, no systemd.
#
# The release tarball is extracted into ~/.local and the server runs as an
# ordinary background process. Everything systemd would have configured is set
# here as environment variables instead.
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

# Answer synthesis: larger, called once or twice per question.
GENERATOR="${MYSCHOLAR_GENERATOR_MODEL:-qwen3:14b}"
# Classification and grading: called six to twelve times per question, so it
# has to be fast. A 14b here would dominate end-to-end latency.
#
# The -instruct suffix is load-bearing. Plain qwen3:4b is reasoning-only and
# spends its entire token budget thinking before emitting anything, so short
# grading calls come back empty.
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
    # Assets moved from .tgz to .tar.zst; keep both paths so an older or newer
    # release layout still installs.
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

    # KEEP_ALIVE=-1 pins weights in VRAM. Ollama otherwise evicts a model after
    # five idle minutes, and the next question pays a multi-second reload that
    # is easy to misread as a slow pipeline.
    #
    # MAX_LOADED_MODELS=2 for the generator and the utility model. Both must be
    # resident at once, because one question alternates between them several
    # times — route and grade on the small one, generation on the large one.
    #
    # NUM_PARALLEL=1, and this is the setting that matters. It was 4, which
    # allocates four KV-cache slots per model at full context and inflated both
    # models about fourfold: `ollama ps` reported qwen3:14b at 38GB and
    # qwen3:4b-instruct at 22GB. That is 60GB of the 64GB across two cards, so
    # the two could not be resident together and every generate-then-grade
    # transition forced a 38GB eviction and a 22GB load. Measured with curl
    # alone, no application code involved: the 14b request returned in 9.2s and
    # an identical 4b request sent immediately afterwards had still not returned
    # after 200 seconds. The same request took 1.8s when the model was already
    # resident.
    #
    # In the pipeline this surfaced as an intermittent wrong answer rather than
    # as a slow one: the stalled call blew the 90s reasoning deadline, the graph
    # took its budget_exhausted edge, and the question came back "I don't have
    # anything on that in these papers" — for a question the corpus answers, on
    # every second attempt. Nothing in the retrieval or generation code was
    # involved and no error was logged.
    #
    # Four parallel slots buy nothing here: this is a single-user development
    # box serving one question at a time. Concurrency belongs in production,
    # which runs against a hosted model and never loads one locally at all.
    # CONTEXT_LENGTH caps the window both models are loaded with. Without it,
    # dropping NUM_PARALLEL to 1 had the opposite effect to the one intended:
    # Ollama stopped rationing context between slots and gave qwen3:4b-instruct
    # its full 262144-token window, which weighs 45GB and no longer fitted, so
    # `ollama ps` showed it running 5%/95% CPU/GPU. Trading a stall for a
    # partial CPU offload is not a fix.
    #
    # 16384 is set from the measured prompts rather than guessed: the largest
    # request the pipeline sends is about 23,000 characters, roughly 6,000
    # tokens, and generation adds a few hundred more. That leaves both models
    # wholly on the GPU with room over.
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
