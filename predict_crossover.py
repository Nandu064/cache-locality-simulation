"""
Crossover prediction model.

Question: Can a simple non-stochastic metric computed from a trace
predict which cache architecture wins on P95 — without running the
full experiment?

We test: avg_req_per_obj and one_hit_frac as predictors of the
P95 crossover between Partitioned and Client Affinity.

Outputs prediction_model.json for the dashboard.
"""

import json
import subprocess
import numpy as np
from collections import Counter
from scipy import stats as sp_stats
from typing import Tuple

BASE_URL = "https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source"

CLUSTERS_META = [
    {"id":"cluster10", "alpha":0.092,
     "local":"traces/cluster10.sort.sample10.zst", "url":f"{BASE_URL}/cluster10.sort.zst"},
    {"id":"cluster19", "alpha":0.735,
     "local":"traces/cluster19.sort.zst",          "url":f"{BASE_URL}/cluster19.sort.zst"},
    {"id":"cluster7",  "alpha":1.067,
     "local":"traces/cluster7.sort.zst",           "url":f"{BASE_URL}/cluster7.sort.zst"},
    {"id":"cluster24", "alpha":1.585,
     "local":"traces/cluster24.sort.sample100.zst","url":f"{BASE_URL}/cluster24.sort.zst"},
]

MAX_REQ = 500_000


# ── Trace loader (reuses streaming approach) ────────────────────

def load_requests(source: str, max_req: int = MAX_REQ) -> np.ndarray:
    import zstandard as zstd
    key_map = {}
    reqs = []
    dctx = zstd.ZstdDecompressor()

    if source.startswith("http"):
        proc = subprocess.Popen(
            ["curl", "-s", "--max-time", "300", source],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw = proc.stdout
    else:
        raw = open(source, "rb")
        proc = None

    try:
        with dctx.stream_reader(raw) as reader:
            buf = b""
            while len(reqs) < max_req:
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
                    if parts[5].strip() not in (b"get", b"gets"):
                        continue
                    k = parts[1]
                    if k not in key_map:
                        key_map[k] = len(key_map)
                    reqs.append(key_map[k])
                    if len(reqs) >= max_req:
                        break
    finally:
        if proc:
            proc.kill(); proc.wait()
        elif hasattr(raw, "close"):
            raw.close()

    return np.array(reqs, dtype=np.int64)


# ── Non-stochastic metrics from request sequence ────────────────

def compute_nonstochastic(reqs: np.ndarray) -> dict:
    """
    Compute non-stochastic metrics from a request sequence.
    These are approximations from a sample (not oracle vtime),
    but are directly measurable from any production trace.
    """
    n         = len(reqs)
    freq      = Counter(reqs.tolist())
    n_unique  = len(freq)
    freqs_arr = np.array(list(freq.values()))

    avg_req_per_obj = n / n_unique

    # One-hit fraction: objects seen exactly once in sample
    # (proxy for true one-hit wonders; overestimates slightly in finite sample)
    n_one_hit     = int((freqs_arr == 1).sum())
    one_hit_frac  = n_one_hit / n_unique

    # Reuse distance P50: median gap between consecutive accesses to same object
    # Sample up to 2000 objects with ≥2 accesses for speed
    from collections import defaultdict
    positions = defaultdict(list)
    for i, k in enumerate(reqs):
        positions[k].append(i)

    reuse_dists = []
    for k, pos in positions.items():
        if len(pos) >= 2:
            reuse_dists.extend(np.diff(pos).tolist())
        if len(reuse_dists) > 200_000:
            break

    rd_p50 = float(np.percentile(reuse_dists, 50)) if reuse_dists else -1.0
    rd_p95 = float(np.percentile(reuse_dists, 95)) if reuse_dists else -1.0

    return {
        "n_requests":       n,
        "n_unique":         n_unique,
        "avg_req_per_obj":  round(avg_req_per_obj, 3),
        "one_hit_frac":     round(one_hit_frac, 4),
        "one_hit_pct":      round(one_hit_frac * 100, 2),
        "reuse_dist_p50":   round(rd_p50, 1),
        "reuse_dist_p95":   round(rd_p95, 1),
    }


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # Load crossover outcomes
    with open("crossover_results.json") as f:
        crossover = {r["id"]: r for r in json.load(f)}

    data_points = []
    print(f"\n{'Computing non-stochastic metrics for all clusters':^60}")
    print("=" * 60)

    for meta in CLUSTERS_META:
        cid = meta["id"]
        src = meta["local"] if os.path.exists(meta["local"]) else meta["url"]
        print(f"\n[{cid}] Loading from {'local file' if not src.startswith('http') else 'URL stream'}...")

        reqs = load_requests(src)
        ns   = compute_nonstochastic(reqs)

        cr   = crossover[cid]["results"]
        lru_p95  = cr["LRU_CACHE"]["p95_ms"]
        part_p95 = cr["PARTITIONED_CACHE"]["p95_ms"]
        aff_p95  = cr["CLIENT_AFFINITY"]["p95_ms"]
        lru_hr   = cr["LRU_CACHE"]["hit_rate"]

        # δ > 0 means Partitioned wins over Affinity on P95
        delta_part_over_aff = round(aff_p95 - part_p95, 1)

        print(f"  avg_req/obj:  {ns['avg_req_per_obj']:.3f}")
        print(f"  one_hit_pct:  {ns['one_hit_pct']:.1f}%")
        print(f"  reuse_p50:    {ns['reuse_dist_p50']:.0f} positions")
        print(f"  LRU hit rate: {lru_hr:.1f}%")
        print(f"  Δ (Aff-Part): {delta_part_over_aff:+.1f}ms  "
              f"→  {'Partitioned wins' if delta_part_over_aff > 0 else 'Affinity wins'}")

        data_points.append({
            "id":                   cid,
            "alpha":                meta["alpha"],
            **ns,
            "lru_hit_rate":         lru_hr,
            "part_p95":             part_p95,
            "aff_p95":              aff_p95,
            "lru_p95":              lru_p95,
            "delta_part_over_aff":  delta_part_over_aff,
            "partitioned_wins":     delta_part_over_aff > 0,
        })

    # ── Model fitting ──────────────────────────────────────────
    print(f"\n{'MODEL FITTING':^60}")
    print("=" * 60)

    avgs    = np.array([d["avg_req_per_obj"] for d in data_points])
    ohf     = np.array([d["one_hit_frac"]     for d in data_points])
    rd_p50  = np.array([d["reuse_dist_p50"]   for d in data_points])
    lru_hr  = np.array([d["lru_hit_rate"]      for d in data_points])
    delta   = np.array([d["delta_part_over_aff"] for d in data_points])

    # Fit 1: log(avg_req_per_obj) → delta
    log_avgs = np.log(avgs)
    s1, i1, r1, _, _ = sp_stats.linregress(log_avgs, delta)
    r2_1 = r1 ** 2
    threshold_avg = float(np.exp(-i1 / s1))

    # Fit 2: one_hit_frac → delta
    s2, i2, r2, _, _ = sp_stats.linregress(ohf, delta)
    r2_2 = r2 ** 2
    threshold_ohf = float(-i2 / s2)

    # Fit 3: lru_hit_rate → delta
    s3, i3, r3, _, _ = sp_stats.linregress(lru_hr, delta)
    r2_3 = r3 ** 2
    threshold_lruhr = float(-i3 / s3)

    print(f"\n  Predictor 1: log(avg_req_per_obj)")
    print(f"    R² = {r2_1:.3f}  |  Crossover threshold: avg_req/obj ≈ {threshold_avg:.2f}")
    print(f"\n  Predictor 2: one_hit_frac")
    print(f"    R² = {r2_2:.3f}  |  Crossover threshold: one_hit_frac ≈ {threshold_ohf:.3f} "
          f"({threshold_ohf*100:.1f}%)")
    print(f"\n  Predictor 3: LRU hit rate")
    print(f"    R² = {r2_3:.3f}  |  Crossover threshold: hit_rate ≈ {threshold_lruhr:.1f}%")

    # Generate fitted curve for dashboard
    avg_range = np.linspace(0.8, 40, 200)
    fitted_curve = (s1 * np.log(avg_range) + i1).tolist()

    # ── Decision rule ──────────────────────────────────────────
    print(f"\n{'DECISION RULE':^60}")
    print("=" * 60)
    print(f"""
  IF avg_req_per_obj < {threshold_avg:.1f}  (one_hit_frac > {threshold_ohf*100:.0f}%)
      → Use PARTITIONED or CLIENT AFFINITY interchangeably
        (workload is in streaming regime; caching provides <30% hit rate;
         architecture choice affects queue routing only, not cache efficiency)

  IF avg_req_per_obj ≥ {threshold_avg:.1f}  (one_hit_frac ≤ {threshold_ohf*100:.0f}%)
      → Use PARTITIONED over CLIENT AFFINITY
        (workload has temporal locality; per-client slots too small;
         cross-client key sharing collapses affinity hit rate)

  Best single metric: log(avg_req/obj), R²={r2_1:.3f}
  Runner-up:          one_hit_frac,     R²={r2_2:.3f}
  Also useful:        LRU hit rate,     R²={r2_3:.3f}
""")

    # ── Save model output ──────────────────────────────────────
    model_out = {
        "data_points": data_points,
        "model": {
            "predictor":          "log(avg_req_per_obj)",
            "slope":              round(s1, 4),
            "intercept":          round(i1, 4),
            "r_squared":          round(r2_1, 4),
            "threshold_avg_req":  round(threshold_avg, 2),
            "threshold_one_hit_pct": round(threshold_ohf * 100, 1),
            "threshold_lru_hit_rate": round(threshold_lruhr, 1),
            "r2_one_hit_frac":    round(r2_2, 4),
            "r2_lru_hit_rate":    round(r2_3, 4),
        },
        "fitted_curve": {
            "avg_req": avg_range.tolist(),
            "delta":   fitted_curve,
        },
        "decision_rule": (
            f"Partitioned wins when avg_req_per_obj ≥ {threshold_avg:.1f} "
            f"(one_hit_frac ≤ {threshold_ohf*100:.0f}%)"
        ),
    }

    with open("prediction_model.json", "w") as f:
        json.dump(model_out, f, indent=2)
    print("  Saved to prediction_model.json")
