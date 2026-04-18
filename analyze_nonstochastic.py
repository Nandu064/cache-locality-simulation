"""
Non-stochastic behavior analysis of cache traces.
Uses the oracleGeneral binary format which includes next_access_vtime —
the virtual time until the next access to the same object (-1 = never).

This directly measures the behaviors Yiyang described:
  - One-hit wonders (next_access_vtime == -1)
  - Reuse distance distribution
  - Temporal locality structure
  - IRM deviation

Binary record format (24 bytes, packed):
  uint32_t  timestamp          (4 bytes)
  uint64_t  obj_id             (8 bytes)
  uint32_t  obj_size           (4 bytes)
  int64_t   next_access_vtime  (8 bytes)  <- -1 if no future access
"""

import struct
import numpy as np
import json
import zstandard as zstd
from collections import Counter

RECORD_FMT  = "<IQIq"   # little-endian: uint32, uint64, uint32, int64
RECORD_SIZE = struct.calcsize(RECORD_FMT)  # 24 bytes

def read_oracle_trace(path: str, max_records: int = 2_000_000):
    """
    Read binary oracle trace. Returns arrays of:
      obj_ids, sizes, reuse_vtimes (next_access_vtime)
    """
    dctx = zstd.ZstdDecompressor()
    obj_ids, sizes, reuse_vtimes = [], [], []

    with open(path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            buf = b""
            while len(obj_ids) < max_records:
                chunk = reader.read(256 * 1024)
                if not chunk:
                    break
                buf += chunk
                while len(buf) >= RECORD_SIZE and len(obj_ids) < max_records:
                    rec = buf[:RECORD_SIZE]
                    buf = buf[RECORD_SIZE:]
                    ts, oid, sz, nav = struct.unpack(RECORD_FMT, rec)
                    obj_ids.append(oid)
                    sizes.append(sz)
                    reuse_vtimes.append(nav)

    return (
        np.array(obj_ids,      dtype=np.uint64),
        np.array(sizes,        dtype=np.uint32),
        np.array(reuse_vtimes, dtype=np.int64),
    )


def analyze_nonstochastic(obj_ids, sizes, reuse_vtimes, label="trace"):
    """
    Compute non-stochastic behavior metrics:
      1. One-hit wonder fraction
      2. Reuse distance distribution
      3. Working set size over time
      4. Object frequency distribution (effective Zipfian alpha)
    """
    from scipy import stats as sp_stats

    n = len(obj_ids)
    n_unique = len(np.unique(obj_ids))

    # ── 1. One-hit wonders ───────────────────────────────────
    one_hit_mask = reuse_vtimes == -1
    n_one_hit = int(one_hit_mask.sum())
    one_hit_frac = n_one_hit / n

    # ── 2. Reuse distance distribution ──────────────────────
    reuse_positive = reuse_vtimes[reuse_vtimes > 0]
    if len(reuse_positive) > 0:
        rd_p50  = float(np.percentile(reuse_positive, 50))
        rd_p95  = float(np.percentile(reuse_positive, 95))
        rd_p99  = float(np.percentile(reuse_positive, 99))
        rd_mean = float(np.mean(reuse_positive))
    else:
        rd_p50 = rd_p95 = rd_p99 = rd_mean = -1.0

    # ── 3. Frequency distribution + Zipfian alpha ───────────
    freq = Counter(obj_ids.tolist())
    freqs_sorted = sorted(freq.values(), reverse=True)
    repeated = [f for f in freqs_sorted if f > 1]
    fitted_alpha = None
    r_squared    = None
    if len(repeated) > 20:
        ranks = np.arange(1, len(repeated) + 1)
        slope, intercept, r, p, se = sp_stats.linregress(
            np.log(ranks), np.log(repeated)
        )
        fitted_alpha = round(-slope, 4)
        r_squared    = round(r ** 2, 4)

    # ── 4. IRM deviation score ───────────────────────────────
    # Under IRM, access probability is stationary.
    # Proxy: coefficient of variation of inter-access intervals
    # High CoV = bursty/periodic = far from IRM
    irm_deviation = None
    freq_counts = np.array(list(Counter(obj_ids.tolist()).values()))
    n_objs_3plus = int((freq_counts >= 3).sum())
    frac_objs_3plus = round(n_objs_3plus / n_unique, 4) if n_unique > 0 else 0

    sample_ids = list(set(obj_ids.tolist()))[:200]  # sample 200 objects
    covs = []
    for oid in sample_ids:
        positions = np.where(obj_ids == oid)[0]
        if len(positions) > 2:
            intervals = np.diff(positions).astype(float)
            mean_iv = intervals.mean()
            if mean_iv > 0:
                covs.append(intervals.std() / mean_iv)
    if covs:
        irm_deviation = round(float(np.mean(covs)), 4)
        # CoV=1 is pure Poisson (closest to IRM), >1 is bursty, <1 is periodic

    results = {
        "label":               label,
        "n_requests":          n,
        "n_unique_objects":    n_unique,
        "avg_requests_per_obj": round(n / n_unique, 2) if n_unique > 0 else 0,
        "one_hit_wonder_frac": round(one_hit_frac, 4),
        "one_hit_wonder_pct":  round(one_hit_frac * 100, 2),
        "frac_objects_3plus":  frac_objs_3plus,
        "pct_objects_3plus":   round(frac_objs_3plus * 100, 2),
        "fitted_alpha":        fitted_alpha,
        "r_squared":           r_squared,
        "reuse_distance": {
            "p50":  round(rd_p50, 1),
            "p95":  round(rd_p95, 1),
            "p99":  round(rd_p99, 1),
            "mean": round(rd_mean, 1),
        },
        "irm_deviation_cov": irm_deviation,
        "irm_note": (
            "CoV of inter-access intervals. "
            "CoV≈1 = Poisson/IRM-like. CoV>1 = bursty. CoV<1 = periodic."
        ),
    }
    return results


def print_report(r):
    print(f"\n{'='*60}")
    print(f"  Non-stochastic Analysis: {r['label']}")
    print(f"{'='*60}")
    print(f"  Requests:          {r['n_requests']:>12,}")
    print(f"  Unique objects:    {r['n_unique_objects']:>12,}")
    print(f"  Avg req/object:    {r['avg_requests_per_obj']:>12.2f}")
    print(f"  One-hit wonders:   {r['one_hit_wonder_pct']:>11.1f}%")
    print(f"  Fitted Zipf alpha: {str(r['fitted_alpha']):>12}  (R²={r['r_squared']})")
    print(f"  Reuse distance P50:{r['reuse_distance']['p50']:>12.1f}")
    print(f"  Reuse distance P95:{r['reuse_distance']['p95']:>12.1f}")
    print(f"  IRM deviation CoV: {str(r['irm_deviation_cov']):>12}  (1.0 = IRM-like)")
    print()


if __name__ == "__main__":
    import os, sys

    traces = [
        ("traces/cluster10.oracleGeneral.sample10.zst", "cluster10 (streaming, α≈0.09)"),
    ]

    all_results = []
    for path, label in traces:
        if not os.path.exists(path):
            print(f"Skipping {label} — file not found: {path}")
            continue
        print(f"Reading {label}...")
        obj_ids, sizes, reuse_vtimes = read_oracle_trace(path, max_records=1_000_000)
        print(f"  Loaded {len(obj_ids):,} records")
        r = analyze_nonstochastic(obj_ids, sizes, reuse_vtimes, label=label)
        print_report(r)
        all_results.append(r)

    with open("results_nonstochastic.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved to results_nonstochastic.json")
