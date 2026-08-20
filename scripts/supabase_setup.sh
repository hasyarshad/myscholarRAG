#!/usr/bin/env bash
# Prepare a hosted Postgres to receive the index: extension, schema, indexes.
#
# Reads SUPABASE_DB_URL from .env, because bash does not. That is the trap this
# script exists to remove: `.env` is read by *pydantic*, inside the Python
# process, so `$SUPABASE_DB_URL` in a shell is empty unless it was exported.
# Passing it anyway sets the variable to an empty string, which then overrides
# the working local default and produces a confusing connection error.
#
#   ./scripts/supabase_setup.sh --check    # report only, change nothing
#   ./scripts/supabase_setup.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

# Always succeeds. Under `set -e` a grep that matches nothing is a failed
# command, and inside an assignment that aborts the script with no message.
value() { grep -E "^$1=" .env 2>/dev/null | cut -d= -f2- | tr -d '"'"'"'' || true; }

URL="${SUPABASE_DB_URL:-$(value SUPABASE_DB_URL)}"
if [[ -z "$URL" ]]; then
    echo "SUPABASE_DB_URL is not set in .env — see .env.example."
    echo "Paste the *session pooler* URI from:"
    echo "  Supabase -> Project Settings -> Database -> Connection string -> URI"
    echo "and change the scheme to postgresql+asyncpg://"
    exit 1
fi

if [[ "$URL" != postgresql+asyncpg://* ]]; then
    echo "SUPABASE_DB_URL must start with postgresql+asyncpg://"
    echo "  (SQLAlchemy picks its driver from the scheme; plain postgresql://"
    echo "   loads the synchronous one, which the async engine cannot use)"
    exit 1
fi

# Host and port only — never the credentials.
echo "target: $(printf '%s' "$URL" | sed -E 's#.*@##; s#/.*##')"

if [[ "${1:-}" == "--check" ]]; then
    MYSCHOLAR_DATABASE__URL="$URL" uv run python - <<'PY'
import asyncio, os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

async def main() -> None:
    engine = create_async_engine(os.environ["MYSCHOLAR_DATABASE__URL"])
    async with engine.connect() as conn:
        ext = (await conn.execute(text(
            "select extname from pg_extension where extname = 'vector'"))).scalar()
        tables = [r[0] for r in await conn.execute(text(
            "select table_name from information_schema.tables "
            "where table_schema = 'public' order by table_name"))]
        counts = {}
        for name in ("documents", "chunks", "chunk_embeddings", "document_summaries"):
            if name in tables:
                counts[name] = (await conn.execute(text(f"select count(*) from {name}"))).scalar()
        indexes = [r[0] for r in await conn.execute(text(
            "select indexname from pg_indexes where schemaname = 'public' "
            "and (indexname like '%hnsw%' or indexname like '%tsv%')"))]
    await engine.dispose()

    print(f"  pgvector : {ext or 'NOT INSTALLED'}")
    print(f"  tables   : {len(tables)} ({', '.join(tables) or 'none'})")
    for name, n in counts.items():
        print(f"  {name:<17}: {n}")
    print(f"  indexes  : {', '.join(indexes) or 'none — searches will scan every row'}")

asyncio.run(main())
PY
    exit 0
fi

echo "1/3  creating extension and schema"
MYSCHOLAR_DATABASE__URL="$URL" uv run myscholar db migrate

echo "2/3  building the text index"
# The full-text index only. The *vector* index is deliberately not built here.
#
# An HNSW graph built on an empty table and then populated by a bulk COPY is a
# far worse graph than one built over the finished data — every row is inserted
# into a structure that has no idea what the data looks like yet. Building it
# here cost real accuracy silently: the top hit for "how are weighting factors
# tuned" came back at 0.5567 and was the wrong paper, while an exact scan over
# the same rows returned the right one at 0.6260. Nothing errored.
#
# So push_index.sh builds it after the data lands. See create_vector_indexes().
MYSCHOLAR_DATABASE__URL="$URL" uv run python - <<'PY'
import asyncio, os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

STATEMENTS = ("CREATE INDEX IF NOT EXISTS ix_chunks_tsv ON chunks USING gin (tsv)",)

async def main() -> None:
    engine = create_async_engine(os.environ["MYSCHOLAR_DATABASE__URL"])
    async with engine.begin() as conn:
        for statement in STATEMENTS:
            await conn.execute(text(statement))
            print(f"     {statement.split()[5]}")
    await engine.dispose()

asyncio.run(main())
PY

echo "3/3  verifying"
exec "$0" --check
