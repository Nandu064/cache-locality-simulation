# Cache Effectiveness and Tail Latency in High-Concurrency API Workloads

A simulation study examining when partitioned caches outperform shared LRU — and when they don't.

**Harish Yerraguntla** · [Portfolio](https://nandu064.github.io/portfolio/) · [GitHub](https://github.com/Nandu064)

---

## What this is

I built this to answer a question I kept running into at work: our Redis cache would hold 89% hit rate under normal load, but P95 latency would still blow past SLA thresholds during traffic spikes. The P50 looked fine. The P95 told a different story.

That gap — between median and tail behavior under concurrent load — is what this simulation studies. It's not a Redis benchmark. It's a controlled experiment on how cache architecture affects latency distribution as concurrency scales.

## Research question

> When does partitioned caching outperform shared LRU because of locality concentration, and when does global LRU win because of cross-client reuse?

## Four configurations

| Config | What it models |
|---|---|
| `NO_CACHE` | Every request hits the backend |
| `LRU_CACHE` | Single shared LRU — simulates one Redis instance |
| `PARTITIONED_CACHE` | Key-sharded across nodes — honest name for what's often called "edge caching" |
| `CLIENT_AFFINITY` | Each client routes to its own node; each client has a private hot key subset |

## What I found

**LRU wins on P50. Partitioned wins on P95 at scale.**

At 10K concurrency with alpha=1.3 (Twitter-typical Zipfian skew):
- No cache: P95 = 4617ms, error rate = 64.6%
- LRU: P95 = 214ms ± 10ms
- Partitioned: P95 = 183ms ± 7ms
- Client affinity: P95 = 266ms ± 6ms — worse hit rate (69.6% vs 88.3%) because per-node capacity is too small for the global tail

The partitioned advantage grows with concurrency. At 50K: LRU P95 = 482ms, partitioned = 350ms. The gap is backend queue spikes, not hit rate — both sit at ~88% hits, but partitioned distributes miss-induced pressure across shards.

**The finding I didn't expect: concurrency genuinely shifts locality structure.**

In v1.1 of this sim, concurrency only changed queue wait times — the request sequence was fixed regardless of load. That's not honest. In v1.2, each client has its own hot 5% key subset. As client count grows, the global working set expands. Miss rate at a fixed 10% cache size:

| Clients | Miss rate at 10% cache |
|---|---|
| 1 | 4.1% |
| 5 | 12.0% |
| 10 | 18.5% |
| 20 | 27.2% |

This is the mechanism Ding's partition approximation (TOMPECS'25) is built for — not just reduced contention, but genuine per-partition locality concentration.

**Alpha is the primary predictor of cache value.**

At alpha=0.8 (flat distribution), LRU hits 41.4% and P50 stays above 100ms. At alpha=1.5, it hits 94.6% with P50 = 2.1ms. The MRC shape changes completely. High-alpha workloads yield most of their locality from tiny caches — which is why a 10% cache captures 96% of requests under a single highly skewed client stream.

## What this doesn't claim

The throughput numbers are "effective throughput under the latency model" — not wall-clock measurements. Queue contention is an exponential approximation, not a proper M/M/c model. The latency distributions (lognormal, calibrated to ~2ms cache hit / ~120ms backend) are reasonable but not measured. These results are directionally valid. They shouldn't be published without replacing the latency model with real Redis traces.

## Next steps

The natural extension is to replay Twitter or Meta's public cache traces through the same framework and see whether the MRC shift prediction holds empirically. That's the bridge between this simulation and something submittable to MEMSYS or HotStorage.

## Running it

```bash
pip install numpy scipy pandas matplotlib
python simulate.py
# outputs results.json
# open index.html via a local server:
python3 -m http.server 8080
# then visit http://localhost:8080
```

## Structure

```
simulate.py     — experiment engine (workload gen, cache models, multi-trial runner)
index.html      — interactive dashboard (5 tabs, CI error bars, MRC by client count)
results.json    — raw output from 220 simulation trials
```

## Related work

This simulation was built in conversation with Prof. Chen Ding's work on locality theory and key-value cache modeling, particularly:
- Ding et al., "Continuous-Time Modeling of Zipfian Workload Locality," TOMPECS 2025
- Hu et al., "LAMA: Optimized Locality-aware Memory Allocation for Key-value Cache," USENIX ATC 2015
- Yang et al., "A Large-scale Analysis of Hundreds of In-memory Key-value Cache Clusters at Twitter," OSDI 2020
