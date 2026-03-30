"""
Cache Effectiveness and Tail Latency in High-Concurrency API Workloads
Simulation Engine v1.2 — Harish Yerraguntla

Changelog from v1.1:
  - Renamed EDGE_CACHE -> PARTITIONED_CACHE (honest: key-sharded, not geographic edge)
  - Added CLIENT_AFFINITY model: each client has a hot subset, requests merge into global
    stream — now concurrency genuinely affects locality structure
  - 5 seeds per condition, report mean + 95% CI for P95 and throughput
  - MRC computed per-client-stream to show how client count shifts reuse distance
  - Throughput labeled "effective throughput under latency model" (not wall-clock)
  - Backend-hit count tracked separately from cache-hit rate
  - Research question corrected to match what is actually modeled

Research question (v1.2):
  When do partitioned caches outperform shared LRU because of locality concentration,
  and when does global LRU win because of cross-client reuse?

Four configurations:
  1. NO_CACHE           -- every request hits backend
  2. LRU_CACHE          -- shared global LRU (simulates Redis single instance)
  3. PARTITIONED_CACHE  -- key-sharded per node (honest rename from EDGE_CACHE)
  4. CLIENT_AFFINITY    -- clients route to assigned nodes by session affinity;
                           each client has a private hot subset + shared long tail
"""

import numpy as np
import pandas as pd
import json
import math
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict
from scipy import stats as sp_stats


# ──────────────────────────────────────────────
# Workload generators
# ──────────────────────────────────────────────

def zipfian_probs(n_keys: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, n_keys + 1, dtype=np.float64)
    probs = 1.0 / ranks ** alpha
    return probs / probs.sum()


def single_stream(n_requests: int, n_keys: int, alpha: float, seed: int) -> np.ndarray:
    """Single client stream — baseline Zipfian."""
    rng = np.random.default_rng(seed)
    return rng.choice(n_keys, size=n_requests, p=zipfian_probs(n_keys, alpha))


def multi_client_stream(
    n_requests: int,
    n_keys: int,
    alpha: float,
    n_clients: int,
    hot_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate n_clients each with:
      - a private hot subset (hot_fraction of keys)
      - access to global tail with normal Zipfian weight

    Returns:
      requests  -- merged global key access sequence
      client_ids -- which client issued each request

    This makes concurrency genuinely change locality:
    more clients -> more hot subsets -> reuse distance increases
    for a global LRU, but stays low for a partitioned cache
    that routes each client to its own node.
    """
    rng = np.random.default_rng(seed)
    hot_size = max(1, int(n_keys * hot_fraction))
    client_hot = []
    for c in range(n_clients):
        offset = (c * hot_size) % n_keys
        hot_keys = np.arange(offset, offset + hot_size) % n_keys
        client_hot.append(hot_keys)

    all_requests = []
    all_clients = []
    per_client = n_requests // n_clients

    global_probs = zipfian_probs(n_keys, alpha)

    for c in range(n_clients):
        # 70% requests from private hot subset, 30% from global Zipfian
        n_hot = int(per_client * 0.70)
        n_global = per_client - n_hot
        hot_keys = client_hot[c]
        hot_probs = zipfian_probs(len(hot_keys), alpha)
        hot_req = rng.choice(hot_keys, size=n_hot, p=hot_probs)
        global_req = rng.choice(n_keys, size=n_global, p=global_probs)
        client_requests = np.concatenate([hot_req, global_req])
        rng.shuffle(client_requests)
        all_requests.append(client_requests)
        all_clients.extend([c] * len(client_requests))

    # Interleave all client streams (simulates concurrent arrival)
    combined = list(zip(np.concatenate(all_requests), all_clients))
    rng.shuffle(combined)
    requests_arr = np.array([r for r, _ in combined])
    clients_arr = np.array([c for _, c in combined])
    return requests_arr, clients_arr


# ──────────────────────────────────────────────
# LRU Cache
# ──────────────────────────────────────────────

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.backend_hits = 0

    def get(self, key: int) -> bool:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return True
        self.misses += 1
        self.backend_hits += 1
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = True
        return False

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ──────────────────────────────────────────────
# Latency model
# ──────────────────────────────────────────────

def backend_latency(rng, queue_pressure: float) -> float:
    base = float(rng.lognormal(mean=4.8, sigma=0.6))   # ~120ms mean
    wait = float(rng.exponential(queue_pressure))
    return base + wait

def cache_hit_latency(rng) -> float:
    return float(rng.lognormal(mean=0.7, sigma=0.3))   # ~2ms mean

def partitioned_hit_latency(rng) -> float:
    return float(rng.lognormal(mean=2.1, sigma=0.35))  # ~8ms mean

def affinity_hit_latency(rng) -> float:
    return float(rng.lognormal(mean=1.8, sigma=0.3))   # ~6ms mean


# ──────────────────────────────────────────────
# Single-run simulation
# ──────────────────────────────────────────────

@dataclass
class RunResult:
    config: str
    concurrency: int
    n_clients: int
    n_requests: int
    n_keys: int
    zipf_alpha: float
    cache_size: int
    hit_rate: float
    backend_hit_rate: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    effective_throughput_rps: float
    error_rate: float
    seed: int


def run_single(
    config: str,
    n_requests: int,
    n_keys: int,
    zipf_alpha: float,
    cache_size: int,
    concurrency: int,
    n_clients: int = 10,
    n_edge_nodes: int = 8,
    hot_fraction: float = 0.05,
    seed: int = 42,
) -> RunResult:

    rng = np.random.default_rng(seed)
    queue_pressure = concurrency * 0.15  # scales backend wait with load

    if config == "CLIENT_AFFINITY":
        requests, client_ids = multi_client_stream(
            n_requests, n_keys, zipf_alpha, n_clients, hot_fraction, seed
        )
    else:
        requests = single_stream(n_requests, n_keys, zipf_alpha, seed)
        client_ids = np.zeros(n_requests, dtype=int)

    latencies = []
    errors = 0
    total_backend_hits = 0

    if config == "NO_CACHE":
        for key in requests:
            lat = backend_latency(rng, queue_pressure)
            if lat > 800:
                errors += 1
            latencies.append(lat)
            total_backend_hits += 1
        hit_rate = 0.0

    elif config == "LRU_CACHE":
        cache = LRUCache(capacity=cache_size)
        for key in requests:
            hit = cache.get(key)
            if hit:
                lat = cache_hit_latency(rng)
            else:
                lat = backend_latency(rng, queue_pressure * 0.05)
                total_backend_hits += 1
            if lat > 800:
                errors += 1
            latencies.append(lat)
        hit_rate = cache.hit_rate

    elif config == "PARTITIONED_CACHE":
        node_capacity = max(1, cache_size // n_edge_nodes)
        caches = [LRUCache(capacity=node_capacity) for _ in range(n_edge_nodes)]
        for key in requests:
            node = int(key) % n_edge_nodes
            hit = caches[node].get(key)
            if hit:
                lat = partitioned_hit_latency(rng)
            else:
                lat = backend_latency(rng, queue_pressure * 0.03)
                total_backend_hits += 1
            if lat > 800:
                errors += 1
            latencies.append(lat)
        total_hits = sum(c.hits for c in caches)
        total_all = sum(c.hits + c.misses for c in caches)
        hit_rate = total_hits / total_all if total_all > 0 else 0.0

    elif config == "CLIENT_AFFINITY":
        # Each client routes to its assigned node
        # Each node has its own LRU — client's hot keys stay warm there
        node_capacity = max(1, cache_size // n_clients)
        caches = [LRUCache(capacity=node_capacity) for _ in range(n_clients)]
        for key, client_id in zip(requests, client_ids):
            node = int(client_id) % n_clients
            hit = caches[node].get(key)
            if hit:
                lat = affinity_hit_latency(rng)
            else:
                lat = backend_latency(rng, queue_pressure * 0.025)
                total_backend_hits += 1
            if lat > 800:
                errors += 1
            latencies.append(lat)
        total_hits = sum(c.hits for c in caches)
        total_all = sum(c.hits + c.misses for c in caches)
        hit_rate = total_hits / total_all if total_all > 0 else 0.0

    arr = np.array(latencies)
    # Effective throughput under latency model (not wall-clock)
    sim_duration_s = arr.sum() / (concurrency * 1000)
    throughput = n_requests / max(sim_duration_s, 0.001)
    backend_hit_rate = total_backend_hits / n_requests

    return RunResult(
        config=config,
        concurrency=concurrency,
        n_clients=n_clients,
        n_requests=n_requests,
        n_keys=n_keys,
        zipf_alpha=zipf_alpha,
        cache_size=cache_size,
        hit_rate=round(hit_rate, 4),
        backend_hit_rate=round(backend_hit_rate, 4),
        p50_ms=round(float(np.percentile(arr, 50)), 2),
        p95_ms=round(float(np.percentile(arr, 95)), 2),
        p99_ms=round(float(np.percentile(arr, 99)), 2),
        effective_throughput_rps=round(throughput, 1),
        error_rate=round(errors / n_requests, 4),
        seed=seed,
    )


# ──────────────────────────────────────────────
# Multi-trial runner with CI
# ──────────────────────────────────────────────

@dataclass
class AggResult:
    config: str
    concurrency: int
    n_clients: int
    zipf_alpha: float
    cache_size: int
    hit_rate_mean: float
    backend_hit_rate_mean: float
    p50_mean: float
    p50_ci95: float
    p95_mean: float
    p95_ci95: float
    p99_mean: float
    p99_ci95: float
    throughput_mean: float
    throughput_ci95: float
    error_rate_mean: float
    n_trials: int


def run_trials(
    config: str,
    n_requests: int,
    n_keys: int,
    zipf_alpha: float,
    cache_size: int,
    concurrency: int,
    n_clients: int = 10,
    n_trials: int = 5,
    base_seed: int = 100,
) -> AggResult:
    runs = [
        run_single(
            config=config,
            n_requests=n_requests,
            n_keys=n_keys,
            zipf_alpha=zipf_alpha,
            cache_size=cache_size,
            concurrency=concurrency,
            n_clients=n_clients,
            seed=base_seed + t * 37,   # deterministic but independent seeds
        )
        for t in range(n_trials)
    ]

    def ci95(vals):
        if len(vals) < 2:
            return 0.0
        return float(sp_stats.sem(vals) * sp_stats.t.ppf(0.975, len(vals) - 1))

    p50s = [r.p50_ms for r in runs]
    p95s = [r.p95_ms for r in runs]
    p99s = [r.p99_ms for r in runs]
    thru = [r.effective_throughput_rps for r in runs]

    return AggResult(
        config=config,
        concurrency=concurrency,
        n_clients=n_clients,
        zipf_alpha=zipf_alpha,
        cache_size=cache_size,
        hit_rate_mean=round(np.mean([r.hit_rate for r in runs]), 4),
        backend_hit_rate_mean=round(np.mean([r.backend_hit_rate for r in runs]), 4),
        p50_mean=round(float(np.mean(p50s)), 2),
        p50_ci95=round(ci95(p50s), 2),
        p95_mean=round(float(np.mean(p95s)), 2),
        p95_ci95=round(ci95(p95s), 2),
        p99_mean=round(float(np.mean(p99s)), 2),
        p99_ci95=round(ci95(p99s), 2),
        throughput_mean=round(float(np.mean(thru)), 1),
        throughput_ci95=round(ci95(thru), 1),
        error_rate_mean=round(float(np.mean([r.error_rate for r in runs])), 4),
        n_trials=n_trials,
    )


# ──────────────────────────────────────────────
# MRC with client-count variation
# Now concurrency (via n_clients) affects locality
# ──────────────────────────────────────────────

def compute_mrc(requests: np.ndarray, max_size: int, steps: int = 35) -> List[Tuple[int, float]]:
    sizes = np.unique(np.logspace(0, np.log10(max_size), steps).astype(int))
    mrc = []
    for size in sizes:
        cache = LRUCache(capacity=int(size))
        for key in requests:
            cache.get(key)
        mrc.append((int(size), round(1.0 - cache.hit_rate, 4)))
    return mrc


def compute_mrc_by_clients(
    n_requests: int,
    n_keys: int,
    alpha: float,
    client_counts: List[int],
    hot_fraction: float = 0.05,
    seed: int = 42,
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Show how increasing client count (= concurrency) shifts the MRC.
    More clients -> more distinct hot subsets -> global LRU sees higher reuse distance.
    """
    result = {}
    for nc in client_counts:
        requests, _ = multi_client_stream(n_requests, n_keys, alpha, nc, hot_fraction, seed)
        mrc = compute_mrc(requests, max_size=n_keys, steps=35)
        result[str(nc)] = mrc
    return result


# ──────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Cache Effectiveness and Tail Latency in High-Concurrency API Workloads")
    print("Simulation v1.2 — Harish Yerraguntla")
    print("=" * 65)

    N_REQ    = 10000    # per trial (5 trials -> 50K total per condition)
    N_KEYS   = 5000
    CACHE    = 500      # 10% of key space
    N_TRIALS = 5
    CONFIGS  = ["NO_CACHE", "LRU_CACHE", "PARTITIONED_CACHE", "CLIENT_AFFINITY"]
    CONCS    = [100, 500, 1000, 5000, 10000, 25000, 50000]
    ALPHAS   = [0.8, 1.0, 1.3, 1.5]

    # ── Experiment 1: concurrency sweep ──────────────────
    print("\n[1/4] Concurrency sweep (alpha=1.3, 5 trials each)...")
    conc_results = []
    for conc in CONCS:
        for cfg in CONFIGS:
            r = run_trials(cfg, N_REQ, N_KEYS, 1.3, CACHE, conc, n_trials=N_TRIALS)
            conc_results.append(asdict(r))
            print(f"  {cfg:20s} | conc={conc:6d} | "
                  f"P95={r.p95_mean:7.1f}±{r.p95_ci95:.1f}ms | "
                  f"hit={r.hit_rate_mean:.1%} | "
                  f"backend={r.backend_hit_rate_mean:.1%} | "
                  f"err={r.error_rate_mean:.2%}")

    # ── Experiment 2: alpha sweep ─────────────────────────
    print("\n[2/4] Alpha sweep (concurrency=10000, 5 trials each)...")
    alpha_results = []
    for alpha in ALPHAS:
        for cfg in CONFIGS:
            r = run_trials(cfg, N_REQ, N_KEYS, alpha, CACHE, 10000, n_trials=N_TRIALS)
            alpha_results.append(asdict(r))
            print(f"  {cfg:20s} | alpha={alpha} | "
                  f"P50={r.p50_mean:7.1f}ms | P95={r.p95_mean:7.1f}ms | "
                  f"hit={r.hit_rate_mean:.1%}")

    # ── Experiment 3: MRC by alpha (single stream) ────────
    print("\n[3/4] MRC by alpha (single stream)...")
    mrc_alpha = {}
    for alpha in ALPHAS:
        reqs = single_stream(20000, N_KEYS, alpha, seed=42)
        mrc_alpha[str(alpha)] = compute_mrc(reqs, max_size=N_KEYS)
        print(f"  alpha={alpha}: {len(mrc_alpha[str(alpha)])} points")

    # ── Experiment 4: MRC by client count ─────────────────
    # KEY ADDITION: this is where concurrency genuinely changes locality
    print("\n[4/4] MRC by client count (alpha=1.3) — concurrency shifts locality...")
    client_counts = [1, 2, 5, 10, 20, 50]
    mrc_clients = compute_mrc_by_clients(20000, N_KEYS, 1.3, client_counts)
    for nc, mrc in mrc_clients.items():
        # Find miss rate at 10% cache size
        pt = next((m for s, m in mrc if s >= CACHE * 0.9), mrc[-1][1])
        print(f"  {nc:2s} clients -> miss rate at 10% cache = {pt:.1%}")

    # ── Save ─────────────────────────────────────────────
    output = {
        "metadata": {
            "version": "1.2",
            "n_requests_per_trial": N_REQ,
            "n_trials": N_TRIALS,
            "n_keys": N_KEYS,
            "cache_size": CACHE,
            "cache_size_pct": CACHE / N_KEYS,
            "research_question": (
                "When do partitioned caches outperform shared LRU because of "
                "locality concentration, and when does global LRU win because "
                "of cross-client reuse?"
            ),
            "honest_note": (
                "Throughput is effective throughput under latency model, "
                "not wall-clock measurement. Latency model uses lognormal "
                "distributions calibrated to Redis/DB typical values."
            ),
        },
        "concurrency_sweep": conc_results,
        "alpha_sweep": alpha_results,
        "mrc_by_alpha": mrc_alpha,
        "mrc_by_clients": {k: v for k, v in mrc_clients.items()},
    }

    with open("results_v2.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nAll results saved to results_v2.json")
    print(f"Total conditions run: {len(conc_results) + len(alpha_results)}")
    print(f"Total trials: {(len(conc_results) + len(alpha_results)) * N_TRIALS}")
