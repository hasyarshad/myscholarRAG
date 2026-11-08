#!/usr/bin/env bash
# Userspace PostgreSQL with pgvector, for machines without root or Docker.
#
# conda-forge ships postgresql and pgvector, so the database runs as an ordinary
# user process with its data directory in $HOME. Same engine as Supabase.
#
#   ./scripts/db.sh setup    create the cluster and database (once)
#   ./scripts/db.sh start
#   ./scripts/db.sh stop
#   ./scripts/db.sh status
#   ./scripts/db.sh psql
#   ./scripts/db.sh destroy  delete the cluster entirely
set -euo pipefail

CONDA_ENV="${MYSCHOLAR_DB_ENV:-myscholar-db}"
PGDATA="${MYSCHOLAR_PGDATA:-$HOME/.local/share/myscholar/pgdata}"
PGPORT="${MYSCHOLAR_PGPORT:-5433}"
PGDATABASE="${MYSCHOLAR_PGDATABASE:-myscholar}"
LOGFILE="$PGDATA/server.log"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
BIN="$CONDA_ROOT/envs/$CONDA_ENV/bin"

info() { printf '\033[36m==>\033[0m %s\n' "$1"; }
ok()   { printf '\033[32mok\033[0m  %s\n' "$1"; }
die()  { printf '\033[31merror\033[0m %s\n' "$1" >&2; exit 1; }

[[ -x "$BIN/pg_ctl" ]] || die "postgres not found in conda env '$CONDA_ENV'.
  Create it with:
    conda create -y -n $CONDA_ENV -c conda-forge postgresql=17 pgvector"

export PATH="$BIN:$PATH"

cmd_setup() {
    if [[ -f "$PGDATA/PG_VERSION" ]]; then
        ok "cluster already initialised at $PGDATA"
    else
        info "initialising cluster at $PGDATA"
        mkdir -p "$PGDATA"
        # Trust auth is safe here: the cluster listens on 127.0.0.1 only and is
        # owned by this user.
        initdb -D "$PGDATA" -U "$USER" --auth-local=trust --auth-host=trust -E UTF8 >/dev/null
    fi

    cmd_start

    if psql -h 127.0.0.1 -p "$PGPORT" -U "$USER" -lqt postgres | cut -d'|' -f1 | grep -qw "$PGDATABASE"; then
        ok "database '$PGDATABASE' already exists"
    else
        info "creating database '$PGDATABASE'"
        createdb -h 127.0.0.1 -p "$PGPORT" -U "$USER" "$PGDATABASE"
    fi

    info "enabling extensions"
    psql -h 127.0.0.1 -p "$PGPORT" -U "$USER" -d "$PGDATABASE" -v ON_ERROR_STOP=1 -q <<'SQL'
-- In `extensions`, not `public`: see PostgresStore._EXTENSIONS.
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;
SQL

    local version
    version=$(psql -h 127.0.0.1 -p "$PGPORT" -U "$USER" -d "$PGDATABASE" -tAc \
        "SELECT extversion FROM pg_extension WHERE extname='vector'")
    ok "pgvector $version ready on port $PGPORT"
    echo
    echo "connection string for .env:"
    echo "  MYSCHOLAR_DATABASE__URL=postgresql+asyncpg://$USER@127.0.0.1:$PGPORT/$PGDATABASE"
}

cmd_start() {
    if cmd_status >/dev/null 2>&1; then
        ok "already running on port $PGPORT"
        return
    fi
    info "starting postgres on port $PGPORT"
    pg_ctl -D "$PGDATA" -l "$LOGFILE" \
        -o "-p $PGPORT -k $PGDATA -c listen_addresses=127.0.0.1" -w start >/dev/null
    ok "started (log: $LOGFILE)"
}

cmd_stop() {
    if ! cmd_status >/dev/null 2>&1; then
        ok "not running"
        return
    fi
    info "stopping postgres"
    pg_ctl -D "$PGDATA" -m fast -w stop >/dev/null
    ok "stopped"
}

cmd_status() {
    pg_ctl -D "$PGDATA" status
}

cmd_psql() {
    exec psql -h 127.0.0.1 -p "$PGPORT" -U "$USER" -d "$PGDATABASE" "$@"
}

cmd_destroy() {
    read -rp "delete the cluster at $PGDATA? [y/N] " reply
    [[ "$reply" == "y" ]] || { echo "cancelled"; exit 0; }
    cmd_stop || true
    rm -rf "$PGDATA"
    ok "removed $PGDATA"
}

case "${1:-}" in
    setup)   cmd_setup ;;
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    psql)    shift; cmd_psql "$@" ;;
    destroy) cmd_destroy ;;
    *) die "usage: $0 {setup|start|stop|status|psql|destroy}" ;;
esac
