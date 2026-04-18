# Cache Effectiveness and Tail Latency in High-Concurrency API Workloads

A simulation study examining when partitioned caches outperform shared LRU — validated against four Twitter production traces.

**Harish Yerraguntla** · [Portfolio](https://nandu064.github.io/portfolio/) · [GitHub](https://github.com/Nandu064)

---

## Research question

> When does partitioned caching outperform shared LRU because of locality concentration — and can a simple workload metric predict the crossover without running the experiment?

---

## Key findings

**1. Partitioned consistently beats LRU on P95 across all workload types.**

The advantage is backend queue pressure distributed across shards, not hit rate. Both LRU and Partitioned track near-identical hit rates across every cluster tested.

| Concurrency | NO_CACHE | LRU | Partitioned | Client Affinity |
|---|---|---|---|---|
| 1K | 638ms | 191 ± 5ms | 187 ± 4ms | 224 ± 6ms |
| 10K | 4617ms | 281 ± 7ms | 240 ± 4ms | 266 ± 6ms |
| 25K | 11328ms | 431 ± 12ms | 330 ± 3ms | 356 ± 7ms |
| 50K | 22517ms | 684 ± 23ms | 480 ± 7ms | 511 ± 13ms |

**2. The real crossover is Partitioned vs Client Affinity — not vs LRU.**

Tested across 4 real Twitter twemcache clusters spanning α = 0.09–1.59:

| Cluster | α | LRU hit rate | Part. P95 | Affinity P95 | Winner |
|---|---|---|---|---|---|
| cluster10 | 0.092 | 0.04% | 388ms | 375ms | Affinity (barely) |
| cluster19 | 0.735 | 9.0% | 379ms | 375ms | Affinity (barely) |
| cluster7  | 1.067 | 71.0% | 272ms | 375ms | **Partitioned +103ms** |
| cluster24 | 1.585 | 90.9% | 157ms | 374ms | **Partitioned +217ms** |

Client affinity wins at low α via node-count distribution (queue routing benefit). It collapses above α ≈ 0.8 once clients share popular keys and per-client cache slots become insufficient.

**3. A single metric predicts the crossover with R²=0.977.**

`log(avg_requests_per_unique_object)` linearly predicts the Partitioned vs Affinity P95 gap across all 4 clusters. No simulation needed — compute this from any access log.

**Decision rule:**
- `avg_req/obj < 1.1` → streaming regime, architecture choice doesn't affect cache efficiency
- `avg_req/obj ≥ 1.1` → reuse regime, use Partitioned over Client Affinity

**4. Non-stochastic oracle trace analysis confirms streaming workload diagnosis.**

Analysis of cluster10 using the `oracleGeneral` binary format (`next_access_vtime` field):

| Metric | Value | Interpretation |
|---|---|---|
| One-hit wonder fraction | 50% | Half of all requests go to objects never seen again |
| Median reuse distance | 500,009 | Spans the full trace window — no temporal locality |
| Objects with ≥3 accesses | 0.06% | Essentially no object is accessed frequently enough to warm a cache |
| Fitted Zipf α (oracle) | 0.002 (R²=0.034) | Flat distribution — no Zipfian structure |

This directly quantifies IRM assumption violation, extending the qualitative finding ("streaming workloads are uncacheable") to a precise measurement.

---

## Methodology

**Workload model.** All four configurations are evaluated against the same merged multi-client request stream. Each client has a private hot key subset (5% of keys, Zipfian) plus a shared global tail. Streams are interleaved to simulate concurrent arrival. 5 trials per condition, 95% CI reported. 220 total trials.

**Cache models.**

| Config | Model | Routing |
|---|---|---|
| `NO_CACHE` | None | — |
| `LRU_CACHE` | Single shared LRU (10% of key space) | Key → global cache |
| `PARTITIONED_CACHE` | 8 shards, each LRU (total = 10%) | Key mod 8 → shard |
| `CLIENT_AFFINITY` | Per-client LRU (total = 10%) | Client ID → node |

**Latency model.** Backend latency: lognormal (mean ≈ 120ms) + exponential queue wait scaling with concurrency. Cache hit: lognormal (mean ≈ 2ms LRU, ≈ 8ms partitioned/affinity). Error threshold: >800ms.

**Trace validation.** Traces streamed directly from [CMU FTP](https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source/) — no full download required. First 500K GET/GETS operations loaded per cluster.

---

## Repo structure

```
simulate.py              — experiment engine (workload gen, cache models, MRC, CI)
run_crossover.py         — multi-cluster crossover experiment (streams traces from URL)
predict_crossover.py     — prediction model: log(avg_req/obj) → P95 gap, R²=0.977
analyze_nonstochastic.py — oracle trace analysis (next_access_vtime, one-hit wonders)
index.html               — interactive dashboard (7 tabs including Crossover Analysis)
results.json             — synthetic experiment results (220 trials)
results_cluster24.json   — cluster24 trace experiment results
crossover_results.json   — 4-cluster crossover experiment results
prediction_model.json    — fitted model output and decision rule
results_nonstochastic.json — oracle trace non-stochastic metrics
requirements.txt
```

---

## Threats to validity

- **Latency model is synthetic.** Lognormal parameters calibrated to typical Redis/DB values, not measured. Results are directionally valid, not quantitative benchmarks.
- **Queue model is approximate.** Exponential wait rather than M/M/c. Underestimates tail latency under sustained overload.
- **IRM deviation is approximated.** CoV of inter-access intervals used as IRM deviation proxy; rigorous metrics (Leo Sciortino's toolkit, pending) would strengthen this.
- **Prediction model has 4 data points.** R²=0.977 is promising but needs validation on additional clusters to confirm generality.

---

## Next steps

- Validate prediction model on additional Twitter clusters across the α=0.1–2.0 range
- Incorporate Leo Sciortino's IRM deviation metrics to replace the CoV approximation
- Extend to write-heavy and TTL workloads

---

## Running it

```bash
pip install -r requirements.txt

# Synthetic experiments
python simulate.py

# Multi-cluster crossover (streams traces from CMU FTP — no download needed)
python run_crossover.py

# Prediction model
python predict_crossover.py

# Non-stochastic oracle analysis (requires oracleGeneral .zst trace)
python analyze_nonstochastic.py

# Dashboard
python3 -m http.server 8080
# visit http://localhost:8080
```

---

## Related work

- Ding et al., "Continuous-Time Modeling of Zipfian Workload Locality," TOMPECS 2025
- Hu et al., "LAMA: Optimized Locality-aware Memory Allocation for Key-value Cache," USENIX ATC 2015
- Yang et al., "A Large-scale Analysis of Hundreds of In-memory Key-value Cache Clusters at Twitter," OSDI 2020
