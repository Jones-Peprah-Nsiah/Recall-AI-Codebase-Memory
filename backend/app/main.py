import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import cache
from .config import settings
from .db import connection, init_db
from .embeddings import model as load_model
from .ingest import index_repo, repo_name
from .search import search as run_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    load_model()  # warm the ONNX session so the first query isn't slow
    with connection() as conn:  # anything mid-index when we last stopped will never finish
        conn.execute("UPDATE repos SET status='error', error='interrupted by restart' WHERE status IN ('queued','indexing')")
    yield


app = FastAPI(title="Recall", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

REPO_COLS = "id, name, url, branch, head_sha, status, progress, error, chunk_count, commit_count, created_at, indexed_at"


class RepoIn(BaseModel):
    url: str
    ref: str | None = None  # branch, tag or sha; defaults to the remote HEAD


def _repo_rows(sql: str, params=()) -> list[dict]:
    with connection() as conn:
        cur = conn.execute(sql, params)
        names = [d.name for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]


@app.get("/api/health")
def health():
    with connection() as conn:
        conn.execute("SELECT 1")
    cache.r.ping()
    return {"ok": True}


@app.get("/api/repos")
def list_repos():
    return _repo_rows(f"SELECT {REPO_COLS} FROM repos ORDER BY id")


@app.post("/api/repos", status_code=202)
def add_repo(body: RepoIn, bg: BackgroundTasks):
    url = body.url.strip()
    with connection() as conn:
        row = conn.execute(
            """INSERT INTO repos (name, url, branch, status) VALUES (%s, %s, %s, 'queued')
               ON CONFLICT (url) DO UPDATE SET branch = EXCLUDED.branch
               WHERE repos.status NOT IN ('queued', 'indexing')
               RETURNING id""",
            (repo_name(url), url, body.ref),
        ).fetchone()
        if row is None:
            raise HTTPException(409, "repo is already being indexed")
        conn.execute("UPDATE repos SET status='queued', error=NULL WHERE id=%s", (row[0],))
    bg.add_task(index_repo, row[0], url, body.ref)
    return _repo_rows(f"SELECT {REPO_COLS} FROM repos WHERE id=%s", (row[0],))[0]


@app.post("/api/repos/{repo_id}/reindex", status_code=202)
def reindex(repo_id: int, bg: BackgroundTasks):
    rows = _repo_rows(f"SELECT {REPO_COLS} FROM repos WHERE id=%s", (repo_id,))
    if not rows:
        raise HTTPException(404)
    if rows[0]["status"] in ("queued", "indexing"):
        raise HTTPException(409, "repo is already being indexed")
    with connection() as conn:
        conn.execute("UPDATE repos SET status='queued', error=NULL WHERE id=%s", (repo_id,))
    bg.add_task(index_repo, repo_id, rows[0]["url"], rows[0]["branch"])
    return {"queued": True}


@app.delete("/api/repos/{repo_id}", status_code=204)
def delete_repo(repo_id: int):
    with connection() as conn:
        conn.execute("DELETE FROM repos WHERE id=%s", (repo_id,))
    cache.bump_index_version()


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, max_length=500),
    kind: Literal["code", "commits", "all"] = "code",
    repo_id: int | None = None,
    k: int = Query(10, ge=1, le=50),
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    cache_: bool = Query(True, alias="cache"),
):
    return run_search(q, kind=kind, repo_id=repo_id, k=k, mode=mode, use_cache=cache_)


@app.get("/api/chunks/{chunk_id}/context")
def chunk_context(chunk_id: int, around: int = Query(15, ge=0, le=200)):
    """Return the chunk plus surrounding lines from the checked-out file."""
    with connection() as conn:
        row = conn.execute(
            "SELECT c.path, c.start_line, c.end_line, r.id, r.name FROM chunks c JOIN repos r ON r.id=c.repo_id WHERE c.id=%s",
            (chunk_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404)
    path, start, end, rid, name = row
    file = Path(settings.repos_dir) / f"{rid}-{name}" / path
    lines = file.read_text(encoding="utf-8").splitlines()
    s, e = max(1, start - around), min(len(lines), end + around)
    return {"path": path, "start_line": s, "end_line": e, "highlight": [start, end], "content": "\n".join(lines[s - 1:e])}


@app.get("/api/stats")
def stats():
    with connection() as conn:
        chunks, commits, repos = conn.execute(
            "SELECT (SELECT count(*) FROM chunks), (SELECT count(*) FROM commits), (SELECT count(*) FROM repos)"
        ).fetchone()
    return {"chunks": chunks, "commits": commits, "repos": repos, "index_version": cache.index_version()}
