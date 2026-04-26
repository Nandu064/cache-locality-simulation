"""
Chi-square stationarity analysis — reproducing Leo's approach.

Algorithm (from Yiyang/Leo):
  1. One pass: count frequencies of all object IDs
  2. Select top-m% IDs as bins + one "OTHER" bin
  3. Divide trace into S equal segments
  4. Compute Pearson chi-square across all segments and bins
  5. Normalize: X²_norm = X² / ((S-1) * m_bins)

Reference values from Leo's CSV (m1s2):
  cluster10: norm≈2.14   (barely above stochastic baseline ~1.0)
  cluster24: norm≈8930   (highly non-stationary)
  cluster7:  norm≈2175   (moderately non-stationary)
  cluster19: norm≈24.8   (mildly non-stationary)

Outputs chi_square_results.json for dashboard integration.
"""

import subprocess
import json
import numpy as np
from collections import Counter
from scipy.stats import chi2 as chi2_dist

BASE_URL = "https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source"

CLUSTERS = [
    {"id": "cluster10", "alpha": 0.092,
     "local": "traces/cluster10.sort.sample10.zst",
     "url":   f"{BASE_URL}/cluster10.sort.zst",
     "leo_norm_m1s2": 2.14},
    {"id": "cluster19", "alpha": 0.735,
     "local": "traces/cluster19.sort.zst",
     "url":   f"{BASE_URL}/cluster19.sort.zst",
     "leo_norm_m1s2": 24.81},
    {"id": "cluster7",  "alpha": 1.067,
     "local": "traces/cluster7.sort.zst",
     "url":   f"{BASE_URL}/cluster7.sort.zst",
     "leo_norm_m1s2": 2174.80},
    {"id": "cluster24", "alpha": 1.585,
     "local": "traces/cluster24.sort.sample100.zst",
     "url":   f"{BASE_URL}/cluster24.sort.zst",
     "leo_norm_m1s2": 8930.65},
]

MAX_REQ = 800_000   # higher limit to capture more of the trace


# ── Trace loader (all ops, not just GET) ────────────────────────

def load_trace(source: str, max_req: int = MAX_REQ) -> np.ndarray:
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
                    if len(parts) < 2:
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


# ── Chi-square computation ───────────────────────────────────────

def compute_chi_square(reqs: np.ndarray, m_pct: float, s: int) -> dict:
    """
    m_pct : percentage of unique objects to use as explicit bins (e.g. 1.0 = 1%)
    s     : number of temporal segments
    """
    n        = len(reqs)
    freq     = Counter(reqs.tolist())
    n_unique = len(freq)

    # Top-m bins + OTHER
    m = max(1, int(n_unique * m_pct / 100.0))
    top_ids  = {k for k, _ in freq.most_common(m)}
    id_to_bin = {k: i for i, (k, _) in enumerate(freq.most_common(m))}

    B = m + 1  # m explicit bins + 1 OTHER

    # Overall probability per bin
    p = np.zeros(B)
    for k, c in freq.most_common(m):
        p[id_to_bin[k]] = c / n
    other_total = sum(c for k, c in freq.items() if k not in top_ids)
    p[m] = other_total / n

    # Segment trace and accumulate chi-square
    X2 = 0.0
    seg_len = n // s
    for seg_idx in range(s):
        start = seg_idx * seg_len
        end   = start + seg_len if seg_idx < s - 1 else n
        L     = end - start

        O = np.zeros(B)
        for k in reqs[start:end]:
            O[id_to_bin.get(k, m)] += 1

        E    = p * L
        mask = E > 0
        X2  += float(np.sum((O[mask] - E[mask]) ** 2 / E[mask]))

    dof          = (s - 1) * m          # Leo's normalization denominator
    normalized   = X2 / dof if dof > 0 else 0.0
    p_value      = float(chi2_dist.sf(X2, dof)) if dof > 0 else 1.0

    return {
        "m_pct":        m_pct,
        "s":            s,
        "m_bins":       m,
        "X2":           round(X2, 3),
        "normalized_X2": round(normalized, 4),
        "p_value":      round(p_value, 6),
        "n":            n,
        "n_unique":     n_unique,
    }


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # Load crossover results for avg_req/obj comparison
    with open("crossover_results.json") as f:
        crossover = {r["id"]: r for r in json.load(f)}

    M_PCTS = [1.0, 10.0]
    S_VALS = [2, 4, 8, 16]

    all_results = []

    print(f"\n{'CHI-SQUARE STATIONARITY ANALYSIS':^70}")
    print("=" * 70)

    for meta in CLUSTERS:
        cid = meta["id"]
        src = meta["local"] if os.path.exists(meta["local"]) else meta["url"]
        print(f"\n[{cid}]  α={meta['alpha']}  "
              f"source={'local' if not src.startswith('http') else 'stream'}")
        print("-" * 50)

        reqs = load_trace(src)
        n_unique = len(set(reqs.tolist()))
        avg_req_per_obj = round(len(reqs) / n_unique, 3)

        # Avg req/obj from crossover data
        cr_avg = crossover.get(cid, {})
        cr_n   = cr_avg.get("n_requests", len(reqs))
        cr_u   = cr_avg.get("n_unique",   n_unique)

        configs = []
        for m_pct in M_PCTS:
            for s in S_VALS:
                res = compute_chi_square(reqs, m_pct, s)
                configs.append(res)

                leo_ref = meta.get("leo_norm_m1s2") if (m_pct == 1.0 and s == 2) else None
                ref_str = f"  [Leo ref: {leo_ref}]" if leo_ref else ""
                print(f"  m={int(m_pct):2d}% s={s:2d} | "
                      f"norm_X2={res['normalized_X2']:10.3f} | "
                      f"p={res['p_value']:.4f}{ref_str}")

        all_results.append({
            "id":              cid,
            "alpha":           meta["alpha"],
            "avg_req_per_obj": avg_req_per_obj,
            "leo_norm_m1s2":   meta.get("leo_norm_m1s2"),
            "configs":         configs,
        })

    # ── Comparison: normalized X2 vs avg_req/obj ────────────────
    print(f"\n{'COMPARISON: norm_X2 (m1s2) vs avg_req/obj':^70}")
    print("=" * 70)
    print(f"  {'Cluster':12s} {'α':6s} {'avg_req/obj':>12s} {'norm_X2(m1s2)':>15s} {'Leo ref':>10s}")
    print("  " + "-" * 60)
    for r in all_results:
        our = next(c for c in r["configs"] if c["m_pct"]==1.0 and c["s"]==2)
        print(f"  {r['id']:12s} {r['alpha']:<6.3f} "
              f"{r['avg_req_per_obj']:>12.3f} "
              f"{our['normalized_X2']:>15.3f} "
              f"{r['leo_norm_m1s2']:>10.2f}")

    # Save
    with open("chi_square_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved → chi_square_results.json")
