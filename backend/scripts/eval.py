"""Measure retrieval quality and latency against a labelled query set.

    docker compose exec api python scripts/eval.py eval/flask_queries.json

Quality: Recall@1 / Recall@5 / MRR@10 for each retrieval mode. A query counts as a hit
when a result's (path, symbol) matches any expected answer.
Latency: end-to-end HTTP time for every query, repeated, with the Redis cache off and on.
"""
import argparse
import json
import statistics
import time

import httpx

MODES = ["vector", "keyword", "hybrid"]


def first_hit_rank(results: list[dict], expect: set[tuple[str, str]]) -> int | None:
    for i, r in enumerate(results, start=1):
        if (r.get("path"), r.get("symbol")) in expect:
            return i
    return None


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--rounds", type=int, default=5, help="latency repetitions per query")
    ap.add_argument("--misses", action="store_true", help="print the hybrid misses")
    args = ap.parse_args()

    spec = json.load(open(args.queries))
    client = httpx.Client(base_url=args.api, timeout=30)
    repo = next(r for r in client.get("/api/repos").json() if r["name"] == spec["repo"])
    queries = spec["queries"]
    print(f"repo={repo['name']}@{repo['head_sha'][:8]}  chunks={repo['chunk_count']}  queries={len(queries)}\n")

    def search(q, mode, cache, k=10):
        t = time.perf_counter()
        resp = client.get("/api/search", params={"q": q, "repo_id": repo["id"], "k": k, "mode": mode,
                                                 "cache": str(cache).lower()})
        resp.raise_for_status()
        return resp.json(), (time.perf_counter() - t) * 1000

    # ---- quality
    print(f"{'mode':<9}{'R@1':>7}{'R@5':>7}{'MRR@10':>9}")
    misses = []
    for mode in MODES:
        ranks = []
        for item in queries:
            body, _ = search(item["q"], mode, cache=False)
            rank = first_hit_rank(body["results"], {tuple(e) for e in item["expect"]})
            ranks.append(rank)
            if mode == "hybrid" and (rank is None or rank > 5):
                misses.append((item, body["results"][:5], rank))
        n = len(ranks)
        r1 = sum(1 for r in ranks if r == 1) / n
        r5 = sum(1 for r in ranks if r and r <= 5) / n
        mrr = sum(1 / r for r in ranks if r) / n
        print(f"{mode:<9}{r1:>7.1%}{r5:>7.1%}{mrr:>9.3f}")

    if args.misses and misses:
        print("\nhybrid misses (not in top 5):")
        for item, top, rank in misses:
            print(f"  - {item['q']!r}  expected {[e[1] for e in item['expect']]}  (rank {rank})")
            for r in top:
                print(f"      {r['path']}  {r['symbol']}")

    # ---- latency (hybrid, end to end over HTTP)
    print(f"\nlatency, hybrid, {args.rounds} rounds x {len(queries)} queries (ms)")
    print(f"{'':<16}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}")
    for label, cache in [("no cache", False), ("redis cache", True)]:
        samples = []
        if cache:
            for item in queries:  # warm
                search(item["q"], "hybrid", cache=True)
        for _ in range(args.rounds):
            for item in queries:
                samples.append(search(item["q"], "hybrid", cache=cache)[1])
        print(f"{label:<16}{pct(samples, 50):>8.1f}{pct(samples, 95):>8.1f}{pct(samples, 99):>8.1f}{max(samples):>8.1f}")
    print(f"  mean no-cache server time: {statistics.mean(search(q['q'], 'hybrid', False)[0]['timings']['total_ms'] for q in queries):.1f} ms")


if __name__ == "__main__":
    main()
