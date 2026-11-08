#!/usr/bin/env bash
# Copy the locally built index into a hosted Postgres (Supabase).
#
# Ingestion runs locally; production only reads its result. This moves data
# only: the schema comes from `myscholar db migrate`, because the role that
# serves traffic must not be able to change the schema it reads.
#
#   ./scripts/push_index.sh              # dump locally, then load into $SUPABASE_DB_URL
#   ./scripts/push_index.sh --dump-only  # just write the dump, inspect it first
#
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-myscholar-db}"
BIN="$CONDA_ROOT/envs/$CONDA_ENV/bin"
PGPORT="${MYSCHOLAR_PGPORT:-5433}"
PGDATABASE="${MYSCHOLAR_PGDATABASE:-myscholar}"
DUMP="${MYSCHOLAR_DUMP:-artifacts/index.sql}"

[[ -x "$BIN/pg_dump" ]] || { echo "pg_dump not found in $BIN. Run: make db-setup"; exit 1; }
export PATH="$BIN:$PATH"

# Tables in dependency order. conversation_turns is left out: visitor chat
# history must never be copied between environments.
TABLES=(documents sections chunks chunk_embeddings document_summaries)

mkdir -p "$(dirname "$DUMP")"
echo "dumping ${#TABLES[@]} tables from localhost:$PGPORT/$PGDATABASE"

# No --disable-triggers: it needs superuser, which Supabase does not grant
# ("RI_ConstraintTrigger_... is a system trigger"). It is also unneeded, since
# pg_dump emits these tables parent-first and the foreign keys hold during the load.
pg_dump \
    --host=127.0.0.1 --port="$PGPORT" --username="$USER" --dbname="$PGDATABASE" \
    --data-only --no-owner --no-privileges \
    "${TABLES[@]/#/--table=}" \
    > "$DUMP"

rows=$(grep -c '^COPY' "$DUMP" || true)
echo "wrote $DUMP ($(du -h "$DUMP" | cut -f1), $rows COPY blocks)"

# Check the parent-first order: otherwise a foreign-key violation would stop the
# load halfway through, after the production tables were truncated.
order=$(grep -oE '^COPY public\.[a-z_]+' "$DUMP" | sed 's/COPY public\.//' | tr '\n' ' ')
expected="documents sections chunks chunk_embeddings document_summaries "
if [[ "$order" != "$expected" ]]; then
    echo
    echo "Unexpected table order in the dump:"
    echo "  got:      $order"
    echo "  expected: $expected"
    echo "Loading in this order would violate foreign keys. Not continuing."
    exit 1
fi

if [[ "${1:-}" == "--dump-only" ]]; then
    exit 0
fi

# Read from .env without sourcing it: the secrets may contain shell metacharacters.
TARGET="${SUPABASE_DB_URL:-$(grep -E '^SUPABASE_DB_URL=' .env 2>/dev/null | cut -d= -f2- | tr -d '"'"'"'' || true)}"
if [[ -z "$TARGET" ]]; then
    echo
    echo "SUPABASE_DB_URL is not set in .env (see .env.example)."
    echo "Dump is ready; load it yourself with:"
    echo "  psql \"\$SUPABASE_DB_URL\" -v ON_ERROR_STOP=1 -f $DUMP"
    exit 1
fi

# psql rejects SQLAlchemy's +asyncpg scheme.
TARGET="${TARGET/postgresql+asyncpg:/postgresql:}"

echo
echo "target: ${TARGET##*@}"
echo "This replaces the index rows on that database. Ctrl-C now if that is wrong."
read -r -p "type 'push' to continue: " reply
[[ "$reply" == "push" ]] || { echo "aborted"; exit 1; }

# Truncate first so a re-push replaces rows instead of failing on duplicate keys.
# CASCADE follows the foreign keys down from documents.
psql "$TARGET" -v ON_ERROR_STOP=1 -q -c "TRUNCATE documents CASCADE;"
psql "$TARGET" -v ON_ERROR_STOP=1 -q -f "$DUMP"

# Build the HNSW index over the loaded rows. A graph built over the finished
# data agrees with an exact scan; one built row by row during the load did not.
echo
echo "building the vector index over the loaded data"
psql "$TARGET" -v ON_ERROR_STOP=1 -q -c "DROP INDEX IF EXISTS ix_chunk_embeddings_hnsw;"
psql "$TARGET" -v ON_ERROR_STOP=1 -q -c "
    CREATE INDEX ix_chunk_embeddings_hnsw
    ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);"

echo
psql "$TARGET" -tAc "
    SELECT 'documents: ' || count(*) FROM documents
    UNION ALL SELECT 'chunks: ' || count(*) FROM chunks
    UNION ALL SELECT 'embeddings: ' || count(*) FROM chunk_embeddings
    UNION ALL SELECT 'summaries: ' || count(*) FROM document_summaries
    UNION ALL SELECT 'model: ' || string_agg(DISTINCT model_id, ', ') FROM chunk_embeddings;
"
echo "done"
