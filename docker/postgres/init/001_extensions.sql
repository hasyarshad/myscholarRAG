-- Extensions required by the retrieval layer.
--
-- vector   : embedding storage and approximate nearest-neighbour search
-- pg_trgm  : trigram similarity, used for fuzzy title matching when linking
--            CV publication entries to indexed papers
--
-- Runs once, on first initialisation of the data volume.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
