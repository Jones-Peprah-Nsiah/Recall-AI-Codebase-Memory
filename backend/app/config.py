import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://recall:recall@localhost:5433/recall")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    repos_dir: str = os.getenv("REPOS_DIR", "/data/repos")

    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    embed_dim: int = int(os.getenv("EMBED_DIM", "384"))
    embed_batch: int = int(os.getenv("EMBED_BATCH", "64"))

    # Chunking
    max_chunk_lines: int = 80
    min_chunk_lines: int = 4
    window_overlap: int = 10
    max_file_bytes: int = 400_000
    max_commits: int = 2000

    # Search
    candidate_pool: int = 50     # per retriever, before fusion
    rrf_k: int = 60
    result_ttl: int = 600        # seconds
    embedding_ttl: int = 7 * 24 * 3600


settings = Settings()
