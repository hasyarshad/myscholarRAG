-- Close the public API surface on a hosted Supabase database.
--
-- Supabase serves every table in the `public` schema over PostgREST, and grants
-- the `anon` and `authenticated` roles full DML on them. The anon key is public
-- by design — it ships inside client applications — so without row-level
-- security those grants mean anyone holding it can read the corpus and, because
-- the grants include DELETE and TRUNCATE, destroy it.
--
-- That matters here for two specific reasons: the chunks table contains the
-- text of publisher copyright papers, and conversation_turns contains what
-- visitors typed.
--
-- This project never uses the REST API. It connects directly as the table
-- owner, and **the owner bypasses row-level security**, so enabling RLS with no
-- policies costs the application nothing and leaves the REST surface returning
-- nothing.
--
--   psql "$SUPABASE_DB_URL" -f scripts/supabase_harden.sql
--
-- Idempotent: safe to run repeatedly.

-- 1. Take the write grants away. Defence in depth: even if RLS were disabled
--    later, or a policy were added carelessly, no anonymous write is possible.
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM anon, authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM anon, authenticated;

-- 2. Enable row-level security everywhere, with no policies. Default-deny: any
--    role that is not the owner sees nothing.
ALTER TABLE documents           ENABLE ROW LEVEL SECURITY;
ALTER TABLE sections            ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks              ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunk_embeddings    ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_summaries  ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_turns  ENABLE ROW LEVEL SECURITY;

-- 3. Drop the duplicate index. `created_at` carried both an explicit
--    Index(...) and index=True on the column, so two identical btree indexes
--    existed — paid for on every insert, useful only once.
DROP INDEX IF EXISTS ix_conversation_turns_created_at;

-- 4. Get the extensions out of `public`, which Supabase's linter warns about
--    under "Extension in Public". `public` is the schema PostgREST exposes and
--    every role can reach, so an extension's functions sitting there are
--    callable by roles that should not have them and shadowable by a
--    same-named function created later in the same schema.
--
--    pg_trgm is dropped rather than moved. Nothing has ever used it: lexical
--    search is tsvector/ts_rank_cd over a GIN index, there is no similarity()
--    call or gin_trgm_ops index anywhere in the schema, and pg_depend reports
--    zero dependent objects. Without CASCADE, deliberately — if that stops
--    being true this should fail loudly rather than drop an index nobody
--    remembered.
DROP EXTENSION IF EXISTS pg_trgm;

--    pgvector is moved, since the corpus is unusable without it. The type and
--    the `<=>` operator become schema-qualified, which is why the application
--    connects with `search_path = public,extensions` — see create_engine.
CREATE SCHEMA IF NOT EXISTS extensions;
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_extension e
          JOIN pg_namespace n ON n.oid = e.extnamespace
         WHERE e.extname = 'vector' AND n.nspname <> 'extensions'
    ) THEN
        ALTER EXTENSION vector SET SCHEMA extensions;
    END IF;
END $$;
