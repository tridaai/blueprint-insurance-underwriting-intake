-- 0001: extensions the application schema depends on.
-- pgvector backs the guideline embeddings table (0003). The compose
-- stack runs the pgvector/pgvector image, where the extension is
-- installed; on a managed Postgres it must be available to CREATE.
CREATE EXTENSION IF NOT EXISTS vector;
