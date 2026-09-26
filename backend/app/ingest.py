"""Clone a git repo, chunk its files, embed, and store everything in pgvector."""
import hashlib
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from . import cache
from .chunker import SKIP_DIRS, Chunk, chunk_file, language_for
from .config import settings
from .db import connection
from .embeddings import embed_documents

log = logging.getLogger("recall.ingest")


def repo_name(url: str) -> str:
    return re.sub(r"\.git$", "", url.rstrip("/")).split("/")[-1].split(":")[-1]


def _git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def _set_status(repo_id: int, status: str, progress: str | None = None, error: str | None = None):
    with connection() as conn:
        conn.execute("UPDATE repos SET status=%s, progress=%s, error=%s WHERE id=%s", (status, progress, error, repo_id))


def checkout(url: str, ref: str | None, dest: Path) -> str:
    if (dest / ".git").exists():
        _git("fetch", "--tags", "--force", "origin", cwd=dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Treeless-blob clone: full history for `git log`, blobs fetched only for the checked-out tree.
        _git("clone", "--filter=blob:none", "--no-checkout", url, str(dest))
    target = ref or _git("symbolic-ref", "--short", "refs/remotes/origin/HEAD", cwd=dest).strip()
    _git("checkout", "--force", "--detach", target, cwd=dest)
    return _git("rev-parse", "HEAD", cwd=dest).strip()


def collect_chunks(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            full = Path(dirpath) / fn
            rel = full.relative_to(root).as_posix()
            if language_for(rel) is None or full.stat().st_size > settings.max_file_bytes:
                continue
            try:
                text = full.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            chunks.extend(chunk_file(rel, text))
    return chunks


def collect_commits(root: Path) -> list[dict]:
    out = _git("log", "--no-merges", f"-n{settings.max_commits}", "--name-only",
               "--pretty=format:%x1e%H%x1f%an%x1f%aI%x1f%B%x1f", cwd=root)
    commits = []
    for rec in out.split("\x1e")[1:]:
        sha, author, date, message, files = rec.split("\x1f")
        commits.append({
            "sha": sha,
            "author": author,
            "committed_at": datetime.fromisoformat(date),
            "message": message.strip(),
            "files": [f for f in files.strip().splitlines() if f],
        })
    return commits


def commit_text(c: dict) -> str:
    files = c["files"][:25]
    return c["message"] + ("\n\nFiles changed: " + ", ".join(files) if files else "")


def _hash(text: str) -> str:
    return hashlib.sha256(f"{settings.embed_model}\0{text}".encode()).hexdigest()


def embed_with_reuse(conn, table: str, texts: list[str], hashes: list[str], on_progress) -> list:
    """Embed texts, reusing vectors already stored for identical content (makes re-indexing cheap)."""
    rows = conn.execute(
        f"SELECT DISTINCT ON (content_hash) content_hash, embedding FROM {table} WHERE content_hash = ANY(%s)",
        (list(set(hashes)),),
    ).fetchall()
    known = {h: e for h, e in rows}
    todo = [(h, t) for h, t in dict(zip(hashes, texts)).items() if h not in known]
    batch = settings.embed_batch * 4
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        for (h, _), vec in zip(part, embed_documents([t for _, t in part])):
            known[h] = vec
        on_progress(min(i + batch, len(todo)), len(todo))
    return [known[h] for h in hashes]


def index_repo(repo_id: int, url: str, ref: str | None) -> None:
    t0 = time.perf_counter()
    try:
        _set_status(repo_id, "indexing", "cloning")
        dest = Path(settings.repos_dir) / f"{repo_id}-{repo_name(url)}"
        head = checkout(url, ref, dest)

        _set_status(repo_id, "indexing", "chunking files")
        chunks = collect_chunks(dest)
        commits = collect_commits(dest)

        with connection() as conn:
            c_texts = [c.embedding_text() for c in chunks]
            c_hashes = [_hash(t) for t in c_texts]
            c_vecs = embed_with_reuse(conn, "chunks", c_texts, c_hashes,
                                      lambda d, n: _set_status(repo_id, "indexing", f"embedding code {d}/{n}"))
            m_texts = [commit_text(c) for c in commits]
            m_hashes = [_hash(t) for t in m_texts]
            m_vecs = embed_with_reuse(conn, "commits", m_texts, m_hashes,
                                      lambda d, n: _set_status(repo_id, "indexing", f"embedding commits {d}/{n}"))

            # Swap the index atomically so searches never see a half-built repo.
            with conn.transaction():
                conn.execute("DELETE FROM chunks WHERE repo_id=%s", (repo_id,))
                conn.execute("DELETE FROM commits WHERE repo_id=%s", (repo_id,))
                with conn.cursor() as cur:
                    cur.executemany(
                        """INSERT INTO chunks (repo_id, path, language, symbol, kind, start_line, end_line,
                                               content, content_hash, lexemes, embedding)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        [(repo_id, c.path, c.language, c.symbol, c.kind, c.start_line, c.end_line,
                          c.content, h, c.lexemes(), v) for c, h, v in zip(chunks, c_hashes, c_vecs)],
                    )
                    cur.executemany(
                        """INSERT INTO commits (repo_id, sha, author, committed_at, message, files,
                                                content_hash, lexemes, embedding)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        [(repo_id, c["sha"], c["author"], c["committed_at"], c["message"], c["files"],
                          h, t, v) for c, t, h, v in zip(commits, m_texts, m_hashes, m_vecs)],
                    )
                conn.execute(
                    """UPDATE repos SET status='ready', progress=NULL, error=NULL, head_sha=%s,
                              chunk_count=%s, commit_count=%s, indexed_at=now() WHERE id=%s""",
                    (head, len(chunks), len(commits), repo_id),
                )
        cache.bump_index_version()
        log.info("indexed repo %s: %d chunks, %d commits in %.1fs",
                 url, len(chunks), len(commits), time.perf_counter() - t0)
    except subprocess.CalledProcessError as e:
        log.exception("git failed for %s", url)
        _set_status(repo_id, "error", error=(e.stderr or str(e)).strip()[-500:])
    except Exception as e:  # noqa: BLE001 — surface any failure on the repo row
        log.exception("indexing failed for %s", url)
        _set_status(repo_id, "error", error=str(e)[-500:])
