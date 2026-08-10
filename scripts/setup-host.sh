#!/usr/bin/env bash
# Host setup requiring root. Run once:
#
#   ./scripts/setup-host.sh
#
# Everything else in this project runs unprivileged.
set -euo pipefail

info() { printf '\033[36m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[33m!!\033[0m  %s\n' "$1"; }
ok()   { printf '\033[32mok\033[0m  %s\n' "$1"; }

# --- Docker group ----------------------------------------------------------
# Without this, every docker command needs sudo, which makes the Makefile
# targets unusable.
if id -nG "$USER" | tr ' ' '\n' | grep -qx docker; then
    ok "$USER is already in the docker group"
else
    info "adding $USER to the docker group"
    sudo usermod -aG docker "$USER"
    warn "log out and back in (or run: newgrp docker) for this to take effect"
fi

# --- Ollama ----------------------------------------------------------------
if command -v ollama >/dev/null 2>&1; then
    ok "ollama already installed ($(ollama --version 2>&1 | head -1))"
else
    info "installing ollama"
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Keep weights resident. Ollama unloads a model after five idle minutes by
# default, so the next question pays a multi-second reload that looks like a
# slow pipeline rather than a cold model.
info "configuring ollama to keep models loaded"
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_MAX_LOADED_MODELS=3"
Environment="OLLAMA_NUM_PARALLEL=4"
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now ollama
sudo systemctl restart ollama

info "waiting for ollama to accept connections"
for _ in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then break; fi
    sleep 1
done

# --- Models ----------------------------------------------------------------
# qwen3:14b  — answer synthesis. Called once or twice per question.
# qwen3:4b   — classification and grading. Called many times per question, so
#              it needs to be fast; a 14b here would dominate total latency.
for model in qwen3:14b qwen3:4b; do
    if ollama list 2>/dev/null | grep -q "^${model%%:*}.*${model##*:}"; then
        ok "$model already pulled"
    else
        info "pulling $model"
        ollama pull "$model"
    fi
done

echo
ok "host setup complete"
echo
echo "next:"
echo "  newgrp docker      # or log out and back in"
echo "  make db-up         # start Postgres with pgvector"
echo "  make check         # lint, typecheck, test"
