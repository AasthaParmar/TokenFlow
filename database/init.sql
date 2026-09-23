CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cache_entries (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    embedding vector(3072) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Gemini embeddings are 3072-d; pgvector HNSW indexes cap at 2000-d.
-- Semantic search still works without an index at dev/eval scale.
