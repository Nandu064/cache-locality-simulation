"""
Multi-cluster crossover experiment.

Streams Twitter twemcache traces directly from CMU FTP — no full download needed.
Runs all 4 cache configs on each cluster, computes P95 gap (LRU - Partitioned).
Saves crossover_results.json for dashboard rendering.

Usage:
    python3 run_crossover.py
"""

import sys
import os
import json
import struct
import subprocess
import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple

# ── Reuse latency/cache machinery from simulate.py ─────────────
sys.path.insert(0, os.path.dirname(__file__))
from simulate import LRUCache, backend_latency, cache_hit_latency, \
    partitioned_hit_latency, affinity_hit_latency, compute_mrc

# ── Cluster definitions ─────────────────────────────────────────
#
# alpha values from Yiyang's metadata:
# https://github.com/twitter/cache-trace/blob/master/stat/2020Mar.md
#
BASE_URL = "https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source"

CLUSTERS = [
    {
        "id":         "cluster10",
        "alpha":      0.092,
        "label":      "cluster10 (streaming, α≈0.09)",
        "local_path": "traces/cluster10.sort.sample10.zst",
        "url":        f"{BASE_URL}/cluster10.sort.zst",
    },
    {
        "id":         "cluster19",
        "alpha":      0.735,
        "label":      "cluster19 (moderate, α≈0.735)",
        "local_path": "traces/cluster19.sort.zst",
        "url":        f"{BASE_URL}/cluster19.sort.zst",
    },
    {
        "id":         "cluster7",
        "alpha":      1.067,
        "label":      "cluster7 (high-reuse, α≈1.067)",
        "local_path": "traces/cluster7.sort.zst",
        "url":        f"{BASE_URL}/cluster7.sort.zst",
    },
    {
        "id":         "cluster24",
        "alpha":      1.585,
        "label":      "cluster24 (high-reuse, α≈1.585)",
        "local_path": "traces/cluster24.sort.sample100.zst",
        "url":        f"{BASE_URL}/cluster24.sort.zst",
    },
]

MAX_REQUESTS = 500_000
CONCURRENCY  = 10_000
N_EDGE_NODES = 8
CACHE_RATIO  = 0.10   # 10% of unique keys


# ── Streaming trace loader ──────────────────────────────────────

def load_trace(source: str, max_requests: int = MAX_REQUESTS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load GET/GETS requests from a CSV twemcache trace.
    source: either a local file path (.zst) or a URL to stream via curl.
    Returns: (requests, client_ids) as int64 arrays.
    """
    import zstandard as zstd

    key_map    = {}
    client_map = {}
    requests   = []
    client_ids = []

    dctx = zstd.ZstdDecompressor()

    if source.startswith("http"):
        print(f"    Streaming from URL (will stop after {max_requests:,} GET ops)...")
        proc = subprocess.Popen(
            ["curl", "-s", "--max-time", "300", source],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        raw_stream = proc.stdout
    else:
        raw_stream = open(source, "rb")
        proc = None

    try:
        with dctx.stream_reader(raw_stream) as reader:
            buf = b""
            while len(requests) < max_requests:
                chunk = reader.read(512 * 1024)
                if not chunk:
                    break
                buf += chunk
                lines = buf.split(b"\n")
                buf = lines[-1]
                for line in lines[:-1]:
                    parts = line.split(b",")
                    if len(parts) < 6:
                        continue
                    op = parts[5].strip()
                    if op not in (b"get", b"gets"):
                        continue
                    key = parts[1]
                    cid = parts[4]
                    if key not in key_map:
                        key_map[key] = len(key_map)
                    if cid not in client_map:
                        client_map[cid] = len(client_map)
                    requests.append(key_map[key])
                    client_ids.append(client_map[cid])
                    if len(requests) >= max_requests:
                        break
    finally:
        if proc:
            proc.kill()
            proc.wait()
        elif hasattr(raw_stream, "close"):
            raw_stream.close()

    return (
        np.array(requests,   dtype=np.int64),
        np.array(client_ids, dtype=np.int64),
    )


# ── Experiment runner ───────────────────────────────────────────

def run_experiment(requests, client_ids, cache_size, seed=42) -> Dict:
    """Run all 4 configs on the same trace stream."""
    n_clients      = int(client_ids.max()) + 1 if len(client_ids) else 1
    queue_pressure = CONCURRENCY * 0.15
    results        = {}
    configs        = ["NO_CACHE", "LRU_CACHE", "PARTITIONED_CACHE", "CLIENT_AFFINITY"]

    for config in configs:
        rng      = np.random.default_rng(seed)
        latencies  = []
        errors     = 0
        backend_h  = 0

        if config == "NO_CACHE":
            for key in requests:
                lat = backend_latency(rng, queue_pressure)
                latencies.append(lat)
                if lat > 800: errors += 1
                backend_h += 1
            hit_rate = 0.0

        elif config == "LRU_CACHE":
            cache = LRUCache(capacity=cache_size)
            for key in requests:
                hit = cache.get(int(key))
                lat = cache_hit_latency(rng) if hit else backend_latency(rng, queue_pressure * 0.05)
                latencies.append(lat)
                if not hit: backend_h += 1
                if lat > 800: errors += 1
            hit_rate = cache.hit_rate

        elif config == "PARTITIONED_CACHE":
            node_cap = max(1, cache_size // N_EDGE_NODES)
            caches   = [LRUCache(capacity=node_cap) for _ in range(N_EDGE_NODES)]
            for key in requests:
                node = int(key) % N_EDGE_NODES
                hit  = caches[node].get(int(key))
                lat  = partitioned_hit_latency(rng) if hit else backend_latency(rng, queue_pressure * 0.03)
                latencies.append(lat)
                if not hit: backend_h += 1
                if lat > 800: errors += 1
            total_h  = sum(c.hits for c in caches)
            total_a  = sum(c.hits + c.misses for c in caches)
            hit_rate = total_h / total_a if total_a else 0.0

        elif config == "CLIENT_AFFINITY":
            node_cap = max(1, cache_size // max(1, n_clients))
            caches   = [LRUCache(capacity=node_cap) for _ in range(max(1, n_clients))]
            for key, cid in zip(requests, client_ids):
                node = int(cid) % max(1, n_clients)
                hit  = caches[node].get(int(key))
                lat  = affinity_hit_latency(rng) if hit else backend_latency(rng, queue_pressure * 0.025)
                latencies.append(lat)
                if not hit: backend_h += 1
                if lat > 800: errors += 1
            total_h  = sum(c.hits for c in caches)
            total_a  = sum(c.hits + c.misses for c in caches)
            hit_rate = total_h / total_a if total_a else 0.0

        arr = np.array(latencies)
        results[config] = {
            "hit_rate":   round(float(hit_rate) * 100, 2),
            "p50_ms":     round(float(np.percentile(arr, 50)), 1),
            "p95_ms":     round(float(np.percentile(arr, 95)), 1),
            "p99_ms":     round(float(np.percentile(arr, 99)), 1),
            "error_rate": round(errors / len(requests), 4),
        }

    # Crossover metric: positive = partitioned wins P95
    lru_p95  = results["LRU_CACHE"]["p95_ms"]
    part_p95 = results["PARTITIONED_CACHE"]["p95_ms"]
    results["_crossover_delta_ms"] = round(lru_p95 - part_p95, 1)
    results["_lru_wins"]           = lru_p95 < part_p95

    return results


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = []

    for cluster in CLUSTERS:
        cid   = cluster["id"]
        label = cluster["label"]
        alpha = cluster["alpha"]

        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")

        # Prefer local file; fall back to streaming URL
        local = cluster.get("local_path", "")
        if local and os.path.exists(local):
            print(f"  Loading from local file: {local}")
            source = local
        else:
            if not local or not os.path.exists(local):
                print(f"  Local file not found — streaming from URL")
            source = cluster["url"]

        print(f"  Reading up to {MAX_REQUESTS:,} GET/GETS ops...")
        requests, client_ids = load_trace(source, max_requests=MAX_REQUESTS)
        n_unique  = len(np.unique(requests))
        n_clients = int(client_ids.max()) + 1 if len(client_ids) else 1
        cache_size = max(1, int(n_unique * CACHE_RATIO))

        print(f"  Loaded {len(requests):,} requests | "
              f"{n_unique:,} unique keys | "
              f"{n_clients:,} clients | "
              f"cache_size={cache_size:,}")

        print(f"  Running 4 configs @ concurrency={CONCURRENCY:,}...")
        results = run_experiment(requests, client_ids, cache_size)

        # MRC
        print(f"  Computing MRC...")
        mrc_raw = compute_mrc(requests, max_size=n_unique, steps=30)
        mrc     = [[int(s), round(float(m), 4)] for s, m in mrc_raw]

        entry = {
            "id":         cid,
            "alpha":      alpha,
            "label":      label,
            "n_requests": len(requests),
            "n_unique":   n_unique,
            "n_clients":  n_clients,
            "cache_size": cache_size,
            "mrc":        mrc,
            "results":    results,
        }
        all_results.append(entry)

        delta = results["_crossover_delta_ms"]
        lru_p95  = results["LRU_CACHE"]["p95_ms"]
        part_p95 = results["PARTITIONED_CACHE"]["p95_ms"]
        winner   = "Partitioned" if delta > 0 else "LRU"
        print(f"\n  LRU P95={lru_p95}ms  |  Partitioned P95={part_p95}ms  "
              f"|  Δ={delta:+.1f}ms  →  {winner} wins")

    out_path = "crossover_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"  Results saved to {out_path}")
    print(f"{'='*60}")

    # Print crossover summary
    print("\n  CROSSOVER SUMMARY (LRU P95 − Partitioned P95):")
    print(f"  {'Cluster':<30} {'α':>6}  {'Δ P95 (ms)':>12}  {'Winner'}")
    print(f"  {'-'*60}")
    for r in sorted(all_results, key=lambda x: x["alpha"]):
        delta = r["results"]["_crossover_delta_ms"]
        winner = "Partitioned ✓" if delta > 0 else "LRU"
        print(f"  {r['label']:<30} {r['alpha']:>6.3f}  {delta:>+12.1f}  {winner}")
