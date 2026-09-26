from contextlib import contextmanager

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import settings

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS repos (
    id           SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    url          TEXT NOT NULL UNIQUE,
    branch       TEXT,
    head_sha     TEXT,
    status       TEXT NOT NULL DEFAULT 'queued',   -- queued | indexing | ready | error
    progress     TEXT,
    error        TEXT,
    chunk_count  INT NOT NULL DEFAULT 0,
    commit_count INT NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    indexed_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    repo_id      INT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    path         TEXT NOT NULL,
    language     TEXT NOT NULL,
    symbol       TEXT,
    kind         TEXT NOT NULL,          -- function | class | method | block | window
    start_line   INT NOT NULL,
    end_line     INT NOT NULL,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    lexemes      TEXT NOT NULL,
    tsv          TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', lexemes)) STORED,
    embedding    VECTOR({settings.embed_dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks (repo_id);
CREATE INDEX IF NOT EXISTS chunks_hash_idx ON chunks (content_hash);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx  ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_emb_idx  ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS commits (
    id           BIGSERIAL PRIMARY KEY,
    repo_id      INT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    sha          TEXT NOT NULL,
    author       TEXT NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL,
    message      TEXT NOT NULL,
    files        TEXT[] NOT NULL DEFAULT '{{}}',
    content_hash TEXT NOT NULL,
    lexemes      TEXT NOT NULL,
    tsv          TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', lexemes)) STORED,
    embedding    VECTOR({settings.embed_dim}) NOT NULL,
    UNIQUE (repo_id, sha)
);
CREATE INDEX IF NOT EXISTS commits_repo_idx ON commits (repo_id);
CREATE INDEX IF NOT EXISTS commits_hash_idx ON commits (content_hash);
CREATE INDEX IF NOT EXISTS commits_tsv_idx  ON commits USING gin (tsv);
CREATE INDEX IF NOT EXISTS commits_emb_idx  ON commits USING hnsw (embedding vector_cosine_ops);
"""


def _configure(conn):
    register_vector(conn)
    # Keep filtered ANN queries (WHERE repo_id = ...) from under-returning.
    conn.execute("SET hnsw.ef_search = 100")
    conn.execute("SET hnsw.iterative_scan = relaxed_order")
    conn.commit()


pool: ConnectionPool | None = None


def init_db() -> None:
    global pool
    # Create the extension before any connection tries to register the vector type.
    import psycopg

    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute(SCHEMA)
    pool = ConnectionPool(settings.database_url, min_size=2, max_size=10, configure=_configure, open=True)


@contextmanager
def connection():
    assert pool is not None, "init_db() not called"
    with pool.connection() as conn:
        yield conn
