# Cache Effectiveness and Tail Latency in High-Concurrency API Workloads

A simulation study examining when partitioned caches outperform shared LRU — and when they don't.

**Harish Yerraguntla** · [Portfolio](https://nandu064.github.io/portfolio/) · [GitHub](https://github.com/Nandu064)

---

## Research question

> When does partitioned caching outperform shared LRU because of locality concentration, and when does global LRU win because of cross-client reuse?

I built this to answer a question I kept running into at work: our Redis cache would hold 89% hit rate under normal load, but P95 latency would still blow past SLA thresholds during traffic spikes. The P50 looked fine. The P95 told a different story.

That gap — between median and tail behavior under concurrent load — is what this simulation studies. It's not a Redis benchmark. It's a controlled experiment on how cache architecture affects latency distribution as concurrency scales.

---

## Key findings

**LRU wins on P50. Partitioned wins on P95 at scale.**

At 10K concurrency with alpha=1.3 (Twitter-typical Zipfian skew):

| Config | P95 (ms) | Hit rate | Error rate |
|---|---|---|---|
| No cache | 4617ms | — | 64.6% |
| LRU (shared) | 214 ± 10ms | 88.3% | — |
| Partitioned | 183 ± 7ms | 88.1% | — |
| Client affinity | 266 ± 6ms | 69.6% | — |

The partitioned advantage grows with concurrency. At 50K: LRU P95 = 482ms, partitioned = 350ms. The gap is backend queue pressure distributed across shards, not hit rate — both LRU and partitioned sit near 88% hits.

**Concurrency genuinely shifts locality structure.**

Each client has its own hot 5% key subset. As client count grows, the global working set expands. Miss rate at a fixed 10% cache size:

| Clients | Miss rate at 10% cache |
|---|---|
| 1 | 4.1% |
| 5 | 12.0% |
| 10 | 18.5% |
| 20 | 27.2% |

This is the mechanism Ding's partition approximation (TOMPECS'25) is built for — not just reduced contention, but genuine per-partition locality concentration.

**Alpha is the primary predictor of cache value.**

At alpha=0.8 (flat distribution), LRU hits 41.4% and P50 stays above 100ms. At alpha=1.5, it hits 94.6% with P50 = 2.1ms. The MRC shape changes completely. High-alpha workloads yield most of their locality from tiny caches.

---

## Methodology

**Workload model.** All four cache configurations are evaluated against the same merged multi-client request stream, ensuring a fair apples-to-apples comparison. Each of `n_clients` clients generates requests from its own private hot subset (5% of the key space, Zipfian-distributed) plus a shared global tail. Streams are interleaved to simulate concurrent arrival.

**Cache models.**

| Config | Cache model | Routing |
|---|---|---|
| `NO_CACHE` | None — every request hits backend | — |
| `LRU_CACHE` | Single shared LRU (10% of key space) | Key → global cache |
| `PARTITIONED_CACHE` | 8 shards, each LRU (total = 10% of key space) | Key mod 8 → shard |
| `CLIENT_AFFINITY` | Per-client LRU (total = 10% of key space) | Client ID → node |

**Latency model.** Backend latency is lognormal (mean ≈ 120ms) plus exponential queue wait that scales with concurrency. Cache hit latency is lognormal (mean ≈ 2ms for LRU, ≈ 8ms for partitioned/affinity due to routing overhead). Error threshold: requests over 800ms are counted as errors.

**Statistical rigor.** Each condition is run 5 times with independent seeds. Results report mean ± 95% confidence interval (t-distribution). Total: 220 trials across 44 conditions.

**Miss ratio curves (MRC).** Computed via LRU stack simulation across log-spaced cache sizes. MRC by client count shows how concurrency shifts the global reuse distance distribution.

---

## Repo structure

```
simulate.py     — experiment engine (workload gen, cache models, multi-trial CI runner, MRC)
index.html      — interactive dashboard (5 tabs, CI error bars, MRC by client count)
results.json    — raw output from 220 simulation trials
requirements.txt
```

---

## Threats to validity

- **Latency model is synthetic.** Lognormal parameters are calibrated to typical Redis/DB values but not derived from measured traces. Results are directionally valid; they should not be treated as quantitative benchmarks.
- **Queue model is approximate.** Backend wait is modeled as exponential rather than an M/M/c queue. This underestimates tail latency under sustained overload.
- **Throughput is not wall-clock.** "Effective throughput" divides total request count by cumulative simulated latency divided by concurrency. It measures work density under the latency model, not observed RPS.
- **Workload is synthetic.** Real cache traces (Twitter, Meta) have temporal locality patterns, key expiry, and write traffic not captured here. The natural extension is to replay public traces through this framework.

---

## Next steps

Replay Twitter's public cache trace through the same MRC framework to validate whether the client-count locality shift holds empirically. That's the bridge from this simulation to something submittable to MEMSYS or HotStorage.

---

## Running it

```bash
pip install -r requirements.txt
python simulate.py
# outputs results.json

# view the dashboard:
python3 -m http.server 8080
# visit http://localhost:8080
```

---

## Related work

- Ding et al., "Continuous-Time Modeling of Zipfian Workload Locality," TOMPECS 2025
- Hu et al., "LAMA: Optimized Locality-aware Memory Allocation for Key-value Cache," USENIX ATC 2015
- Yang et al., "A Large-scale Analysis of Hundreds of In-memory Key-value Cache Clusters at Twitter," OSDI 2020
