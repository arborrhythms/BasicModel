# Teacher reconstruction one-hour benchmark

The accepted Teacher comparison artifact is:

```text
output/runs/basicmodel_teacher_1h_b24_20260727_083632/
```

It uses the same B24 / W256 comparison geometry and 2,000-document
FineWeb-Edu slice as the pre-Teacher baseline. The run starts from a fresh
basin and uses MPS, Inductor `default`, and fullgraph compilation. One global
3,600-second training deadline spans 100 possible epochs, so a faster model
cannot terminate early after its first corpus traversal. Periodic checkpoints
were disabled because the accepted baseline completed only 71 bricks and
therefore never reached its 100-brick periodic-save interval; final autosave
remained enabled.

The Teacher's objective source address is exposed outside the compiled model
carrier. It neither reads nor modifies the model's subjective `.where` and
`.when`. This clean throughput gate does not yet let an objective-address
encoder affect the numerical path.

## Result

| Interval | Complete sentences | Training seconds | Optimizer bricks | Sentences/s | Reconstruction loss |
|---|---:|---:|---:|---:|---:|
| Epoch 1 | 74,559 | 1,850.368 | 246 | 40.294 | 2.4273 |
| Epoch 2 | 74,559 | 1,714.707 | 246 | 43.482 | 2.2894 |
| Epoch 3 partial | 2,272 | 40.347 | 6 | 56.311 | 2.3344 |
| **Training-window aggregate** | **151,390** | **3,605.422** | **498** | **41.989** | — |

The cap is checked between complete optimizer bricks, accounting for the
5.422-second overshoot. Total process time, including setup and the final
4.65 GB checkpoint write, was 3,649.94 seconds; throughput over that broader
interval was 41.477 sentences/s.

Peak resident memory reported by `/usr/bin/time -l` was:

```text
14,571,520,000 bytes = 14.572 GB = 13.571 GiB
```

The run included both real growth costs:

- aligned conceptual prefixes 32K -> 65K -> 131K -> 262K;
- PartSpace radix 32K -> 65K, including one 76.874-second graph
  invalidation/recompile brick.

After each transition, steady brick time returned to roughly 6.7-7.3 seconds.

## Accepted baseline comparison

The preserved pre-Teacher comparison reported 24,746 sentences in 3,637.495
training seconds, or 6.803 sentences/s.

| Metric | Pre-Teacher | Teacher | Change |
|---|---:|---:|---:|
| Sentences/s | 6.803 | 41.989 | **6.172x** |
| Throughput gain | — | — | **+517.2%** |
| Milliseconds/sentence | 146.993 | 23.815 | **-83.8%** |
| Checkpoint bytes | 4,911,831,883 | 4,645,553,753 | **-5.4%** |

The loss values are not step-matched: the baseline stopped after 71 bricks
and one third of its first pass, while the Teacher run completed 498 bricks
and two full passes. They demonstrate finite training but must not be used as
a controlled quality comparison.

Artifact identities:

```text
2db62794dc2b34468e7675448e3823b87e7a28f6f3f204e37793b17bcf903655  BasicModel.ckpt
8ba4dd20b30d08d4dd579fae379bc16e36f5ca4735349e4269b0c7e99d44cb6f  BasicModel.xml
```

During the first attempted multi-epoch endurance run, the faster model
reached a previously unexercised end-of-epoch path and exposed a bounded
prefetch-queue sentinel race. The producer now retries the terminal sentinel
until the consumer frees a slot, with a deterministic regression test that
keeps the queue full beyond the retired one-second timeout. The accepted run
above completed two epoch boundaries after that fix.
