# Reconstruction: all eight declared seeds

The reviewed reconstruction baseline remains unchanged. Two warmup and five measured training batches; four validation batches before and after, batch size two. No million-sentence campaign.

| Seed | HEAD before | Candidate before | HEAD after | Candidate after | HEAD process | Candidate process |
|---|---:|---:|---:|---:|---|---|
| 0 | 0.10109799 | 0.10989956 | 0.10851453 | 0.10920948 | complete | complete |
| 1 | 0.12225494 | 0.10667161 | 0.11884609 | 0.11788479 | complete | complete |
| 2 | 0.12085174 | 0.12410101 | 0.11443525 | unavailable | complete | memory |
| 3 | 0.09851541 | 0.09847647 | 0.10134130 | 0.11899159 | complete | complete |
| 4 | 0.09506519 | 0.09858430 | 0.10199065 | 0.10001503 | complete | complete |
| 5 | 0.11952911 | 0.11903835 | 0.12380376 | 0.14672459 | complete | complete |
| 6 | 0.11327300 | 0.11370735 | 0.10896933 | 0.11613330 | complete | complete |
| 7 | 0.08667692 | 0.09762206 | 0.11200417 | 0.10322671 | complete | complete |

## One unguarded diagnostic for each memory stop

These values are separate from the guarded results above. Each original memory stop remains red.

| Seed | Tree | Before | After | Diagnostic process | Peak GiB |
|---|---|---:|---:|---|---:|
| 2 | candidate | 0.12410101 | 0.12183350 | complete | 7.22 |

All process results, training means, timings, initial atom fingerprints, definition counts and context-read counts are in [the comparison](reconstruction-comparison-final.json). These are concurrent-run timings, not a controlled speed comparison.
