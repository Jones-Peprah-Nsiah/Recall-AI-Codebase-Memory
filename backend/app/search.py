"""Hybrid retrieval: pgvector ANN + Postgres full-text, fused with Reciprocal Rank Fusion."""
import re
import time

from . import cache
from .chunker import CAMEL
from .config import settings
from .db import connection
from .embeddings import embed_query

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "by", "with", "how",
    "what", "where", "when", "which", "does", "do", "that", "this", "it", "from", "as", "at", "code",
    "function", "find", "show", "me", "i", "we", "get", "gets",
}

TABLES = {
    "code": {
        "table": "chunks",
        "cols": "t.id, t.path, t.language, t.symbol, t.kind, t.start_line, t.end_line, t.content",
    },
    "commits": {
        "table": "commits",
        "cols": "t.id, t.sha, t.author, t.committed_at, t.message, t.files",
    },
}


def to_tsquery(q: str) -> str:
    words = set()
    for w in re.findall(r"[A-Za-z0-9_]+", q):
        for part in [w, *CAMEL.findall(w)]:
            part = part.lower()
            if len(part) > 1 and part not in STOPWORDS:
                words.add(part)
    # OR the terms together (with prefix match) and let ts_rank_cd reward documents matching more of them.
    return " | ".join(f"{w}:*" if len(w) >= 4 else w for w in sorted(words))


def _search_table(conn, kind: str, qvec, tsq: str, repo_id: int | None, k: int, mode: str) -> list[dict]:
    spec = TABLES[kind]
    table, n = spec["table"], settings.candidate_pool
    where = "WHERE repo_id = %(repo)s" if repo_id else ""
    and_repo = "AND repo_id = %(repo)s" if repo_id else ""

    vec_cte = f"""
        SELECT id, row_number() OVER (ORDER BY dist) AS rank, 1 - dist AS similarity FROM (
            SELECT id, embedding <=> %(qvec)s AS dist FROM {table} {where}
            ORDER BY embedding <=> %(qvec)s LIMIT %(n)s
        ) v"""
    fts_cte = f"""
        SELECT id, row_number() OVER (ORDER BY r DESC) AS rank FROM (
            SELECT id, ts_rank_cd(tsv, query, 32) AS r
            FROM {table}, to_tsquery('simple', %(tsq)s) query
            WHERE tsv @@ query {and_repo}
            ORDER BY r DESC LIMIT %(n)s
        ) f"""
    empty = "SELECT NULL::bigint AS id, NULL::bigint AS rank, NULL::float AS similarity WHERE false"
    if mode == "keyword" or qvec is None:
        vec_cte = empty
    if mode == "vector" or not tsq:
        fts_cte = empty.replace(", NULL::float AS similarity", "")

    sql = f"""
        WITH vec AS ({vec_cte}), fts AS ({fts_cte})
        SELECT {spec['cols']}, r.name AS repo,
               COALESCE(1.0 / (%(rrf)s + vec.rank), 0) + COALESCE(1.0 / (%(rrf)s + fts.rank), 0) AS score,
               vec.rank AS vector_rank, fts.rank AS keyword_rank, vec.similarity
        FROM vec FULL OUTER JOIN fts USING (id)
        JOIN {table} t USING (id)
        JOIN repos r ON r.id = t.repo_id
        ORDER BY score DESC, vec.rank NULLS LAST
        LIMIT %(k)s"""
    params = {"qvec": qvec, "tsq": tsq, "repo": repo_id, "n": n, "k": k, "rrf": settings.rrf_k}
    cur = conn.execute(sql, params)
    names = [d.name for d in cur.description]
    rows = [dict(zip(names, row)) for row in cur.fetchall()]
    for row in rows:
        row["type"] = "code" if kind == "code" else "commit"
        row["score"] = float(row["score"])
        if row["similarity"] is not None:
            row["similarity"] = round(float(row["similarity"]), 4)
    return rows


def search(query: str, kind: str = "code", repo_id: int | None = None, k: int = 10,
           mode: str = "hybrid", use_cache: bool = True) -> dict:
    t0 = time.perf_counter()
    params = {"q": query, "kind": kind, "repo": repo_id, "k": k, "mode": mode}

    if use_cache and (hit := cache.get_results(params)) is not None:
        return {"results": hit, "timings": {"total_ms": _ms(t0)}, "cache": {"results": True, "embedding": True}}

    emb_cached, qvec, embed_ms = False, None, 0.0
    if mode != "keyword":
        t1 = time.perf_counter()
        qvec = cache.get_query_embedding(query) if use_cache else None
        emb_cached = qvec is not None
        if qvec is None:
            qvec = embed_query(query)
            cache.set_query_embedding(query, qvec)
        embed_ms = _ms(t1)

    t2 = time.perf_counter()
    tsq = to_tsquery(query)
    kinds = ["code", "commits"] if kind == "all" else [kind]
    with connection() as conn:
        results = [row for kd in kinds for row in _search_table(conn, kd, qvec, tsq, repo_id, k, mode)]
    results = sorted(results, key=lambda r: -r["score"])[:k]
    db_ms = _ms(t2)

    if use_cache:
        cache.set_results(params, results)
    return {
        "results": results,
        "timings": {"embed_ms": embed_ms, "db_ms": db_ms, "total_ms": _ms(t0)},
        "cache": {"results": False, "embedding": emb_cached},
    }


def _ms(t: float) -> float:
    return round((time.perf_counter() - t) * 1000, 2)
