# Reconstruction Fidelity + Config Matrix + Speed — Design (APPROVED)

> **STATUS: APPROVED 2026-07-03 (Alec, brainstorm session).** Successor scope
> to the two-phase plan's deferred "recon-fidelity design pass (where-band
> aliasing, loss magnitude)" and todo.md's Task 10. Execution plan:
> [2026-07-03-reconstruction-fidelity-execution.md](2026-07-03-reconstruction-fidelity-execution.md).

## Decisions (Alec, 2026-07-03)

1. **Success bar: EXACT ROUND-TRIP.** The decoded reconstruction reproduces
   the input presentation exactly (fixed seed, bounded epochs), pinned by
   tests — first on MM_20M_xor, then on MM_20M_grammar.
2. **Test breadth: canonical path + variants** (~8 named configurations),
   not a pairwise sweep.
3. **Speed goal: production training throughput** — epochs/hour on
   MM_20M_grammar, benchmarked on ArborStudio (M4, 36 GB;
   `arogers@ArborStudio.local`, key `~/.ssh/id_ed25519_arborstudio`,
   existing `ARBORSTUDIO_*` rsync targets in the parent `Makefile.local`,
   dest `~/WikiOracle/` — a native path, no iCloud, no spaces, so
   `MODEL_COMPILE=auto`/inductor is plausibly unblocked there; `~/github`
   available as an `ARBORSTUDIO_DEST` override).
4. **New analysis/synthesis defaults: meronomy / meronomy** — MM_20M_grammar
   moves off lexicon/byte onto the mereological pair MM_20M_xor already uses.
5. **Debug and speed run IN CONJUNCTION**: one shared harness records
   fidelity AND timing on every run, so reconstruction debugging
   accumulates the benchmark baselines as a side effect.

## Scope

**In:** subsymbolic reconstruction quality (the PS $\to$ `.what` / WS $\to$
`.where` story, Architecture sec B); MM_20M_xor exact round-trip;
MM_20M_grammar reshape to meronomy/meronomy + exact round-trip; the
canonical config matrix with tests; profile-driven speed work on the
predominant (grammar) path.

**Parked (explicitly out):** wave brightness / training dynamics
(MM_sparse_concept, sO $\ge 1$ PARALLEL — grammar is serial and XOR is
sO=0, so the conceptual wave never runs in this scope) and Task 11
nVectors wiring (same reason; queued behind this phase).

## The three known symptoms (from the deferred pass)

1. **Weak loss magnitude** — reconstruction loss contributes little
   (channel scales today: `reconstruction_scale` 0.5, what/where/when
   0.7/0.2/0.1); unknown whether the problem is scaling or a
   near-constant signal.
2. **Near-zero `.where` recovery** — the "where-band aliasing" item; WS
   reverse `.where` codes do not recover true spans.
3. **Decode granularity** — reconstructed content decodes at the wrong
   granularity (slab vs word vs byte).

Architecture sec B already sketches the `.where` direction: under a serial
tiling, placement is the RUNNING SUM of part sizes, so WS should supply
TYPE tiling (word/space/punct), not absolute coordinates.

## Design

### The shared harness

One driver (promoted from the `test/bench_throughput.py` pattern) that
builds any config, runs $N$ epochs, and emits a structured run record
(JSON row in `output/`): **fidelity** — exact-match rate (the bar),
per-channel what/where/when reconstruction losses, a `.where`-recovery
score — and **timing** — wall-clock per epoch, optional torch.profiler
top-$k$. Remote mode: rsync via the existing `ARBORSTUDIO_*` targets, ssh
the same harness, retrieve records. Every debug run is a benchmark sample.

### Root-cause on XOR, then fixes

MM_20M_xor is the isolation vehicle (already meronomy/meronomy,
`mereologyRaise` on, sO=0 parallel). Probe each symptom; fixes are
evidence-contingent but the anticipated territory is: per-channel loss
rebalancing (magnitude), type-tiling + running-sum `.where` (recovery),
word-granularity decode via PS codebook rows (granularity). Every fix
lands TDD against `test/test_reconstruction_roundtrip.py`.

### Grammar switch + canonical matrix

MM_20M_grammar reshapes for meronomy/meronomy (its lexicon/byte-era
dimensions — PS nVectors 8 / nDim 12 — get reworked toward the XOR shape
adapted for serial+grammar), keeping `butterfly=false` (the $\sim$1M-param
square fold) unless profiling argues otherwise. Matrix (~8): grammar-
meronomy (predominant), xor, legacy (bpe/byte back-compat), stack-on,
mereologyRaise-off, sO=3-parallel (smoke only — wave still dark),
readingAttention-on, two-pass learning. Fast build+epoch+recon smokes run
in `make test`; full round-trips behind `RUN_SLOW`.

### Speed pass (gated by fidelity)

On ArborStudio against accumulated baselines: profiler top-$k$ on grammar;
`MODEL_COMPILE=auto` on the native path; prefetch (`numWorkers>0`);
batch-size sweep; MPS-eager probe (MPS is compile-incompatible but may
beat CPU eager). HARD RULE: an optimization lands only if the round-trip
tests stay green.

## Gates (Alec commits at each)

- **Gate 1**: harness + remote mode + recorded baselines (xor, grammar
  pre-switch, legacy; local + ArborStudio).
- **Gate 2**: XOR exact round-trip green.
- **Gate 3**: grammar-meronomy exact round-trip + matrix green;
  `make test` green.
- **Gate 4**: speed report — before/after epochs/hour on MM_20M_grammar
  on ArborStudio; `make test` green.

## Success criteria

1. `test_reconstruction_roundtrip.py` pins exact round-trip for
   MM_20M_xor and MM_20M_grammar(meronomy) at fixed seed within a bounded
   epoch count.
2. The canonical matrix is tested: fast smokes in `make test`, full
   round-trips under `RUN_SLOW`.
3. A recorded before/after epochs-per-hour for MM_20M_grammar on
   ArborStudio, with the optimization log.
4. No fidelity regression at any speed step (round-trip suite green
   throughout).

## References

- doc/Architecture.md sec B (reconstruction mandate; type-tiling note)
- doc/plans/2026-07-02-two-phase-loops-sparse-relation.md (deferred list)
- doc/plans/2026-07-03-iterated-symbolic-loop-execution.md EXECUTION
  NOTES items 12-13 (wave-dark, nVectors — the parked findings)
- todo.md Task-10 entry; parent `Makefile.local` `ARBORSTUDIO_*`
