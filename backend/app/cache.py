import hashlib
import json

import numpy as np
import redis

from .config import settings

r = redis.Redis.from_url(settings.redis_url)

INDEX_VERSION_KEY = "recall:index_version"


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def get_query_embedding(query: str) -> np.ndarray | None:
    raw = r.get(f"emb:{settings.embed_model}:{_h(query)}")
    return np.frombuffer(raw, dtype=np.float32) if raw else None


def set_query_embedding(query: str, vec: np.ndarray) -> None:
    r.set(f"emb:{settings.embed_model}:{_h(query)}", vec.astype(np.float32).tobytes(), ex=settings.embedding_ttl)


def index_version() -> int:
    return int(r.get(INDEX_VERSION_KEY) or 0)


def bump_index_version() -> None:
    """Invalidate every cached result set; called whenever an index changes."""
    r.incr(INDEX_VERSION_KEY)


def _result_key(params: dict) -> str:
    return f"search:v{index_version()}:{_h(json.dumps(params, sort_keys=True))}"


def get_results(params: dict) -> list | None:
    raw = r.get(_result_key(params))
    return json.loads(raw) if raw else None


def set_results(params: dict, results: list) -> None:
    r.set(_result_key(params), json.dumps(results, default=str), ex=settings.result_ttl)
