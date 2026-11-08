-- Close the public API surface on a hosted Supabase database.
--
-- Supabase serves every `public` table over PostgREST and grants `anon` and
-- `authenticated` full DML on them, and the anon key ships in client apps. Here
-- that would expose copyrighted paper text (chunks) and visitor messages
-- (conversation_turns). The app never uses the REST API and connects as the
-- table owner, which bypasses RLS, so RLS with no policies costs it nothing.
--
--   psql "$SUPABASE_DB_URL" -f scripts/supabase_harden.sql
--
-- Idempotent.

-- 1. Revoke the grants, so no anonymous write is possible even if RLS is later
--    disabled or a careless policy is added.
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM anon, authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM anon, authenticated;

-- 2. RLS with no policies: every role except the owner sees nothing.
ALTER TABLE documents           ENABLE ROW LEVEL SECURITY;
ALTER TABLE sections            ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks              ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunk_embeddings    ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_summaries  ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_turns  ENABLE ROW LEVEL SECURITY;

-- 3. Drop a duplicate btree index on created_at (the model declared it twice),
--    which cost every insert and added nothing.
DROP INDEX IF EXISTS ix_conversation_turns_created_at;

-- 4. Move extensions out of `public` (Supabase lint "Extension in Public"),
--    where every role can call their functions and a same-named function can
--    shadow them.
--
--    pg_trgm is unused (lexical search is tsvector over GIN), so it is dropped.
--    No CASCADE, so this fails instead of dropping a dependent object.
DROP EXTENSION IF EXISTS pg_trgm;

--    pgvector is moved; the application connects with
--    `search_path = public,extensions` (see create_engine) to resolve it.
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
