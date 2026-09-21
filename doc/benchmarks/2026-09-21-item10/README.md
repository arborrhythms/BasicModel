# Item 10: reconstruction measurements and seed-selection audit

Baseline includes local documentation commit `0c8bea8`, which is published with
this landing. The MentalModel/seed bullet was taken early from countdown item 2
at Alec's request. This record does not close any expectation learning gate.

## MentalModel overflow

The failing initialization is seed 3, `data/MentalModel.xml`, untrained CPU
fp32, input `hello world`. Recursive chart composition fed the old approximate
`compact_soft` slab back into the reducer. A reduction could reuse an operand
that an earlier tile had already consumed; padded slots also remained live in
later rounds. A four-leaf product made the error independent of initialization:
`2 * 3 * 5 * 7` returned **1260**, instead of **210**. The seed-3 trace grew to
about `2.01e32` before a candidate product overflowed.

The replacement dynamic program tracks source position and packed output
ordinal. It computes the tiling distribution's expected packed output and
matches hard compaction for every one-hot tiling. Hard recursive rounds carry
live lengths and stop at one remaining slot. Padded inputs are zeroed before
candidate evaluation; padding cannot be reduced or earn local structural
credit. Arithmetic operators and the non-finite guard
are unchanged; there is no clamp or replacement selected seed.

The initial regression selection has **6 failures / 15 cases**
([receipt](compaction-red-result.json.gz)). After the runtime fix, the only
remaining failure was an invalid old gradient assertion: summing additive
outputs is invariant under every legal tiling, so its routing gradient is
zero. That fixture now uses a squared objective that varies with the tiling;
it no longer depends on the compactor duplicating mass or floating-point
roundoff. The corrected affected selection passes **35/35**
([receipt](compaction-green-result.json.gz)). Final source additionally checks
the enumerated output and derivatives under fullgraph capture.

The ordinary MentalModel compatibility assertion now uses ambient RNG. A
separate regression keeps **failing seed 3** and requires a finite forward.
The declared measurement range **0–31 passes 32/32**, each in a fresh process
under 8 GiB / 120 seconds. All outcomes and the exact source snapshot are in
[the seed records](mentalmodel-records.tar.gz); [probe](mentalmodel_probe.py).
The largest observed reducer input over that range is `7.73e16`. This is a
bounded measurement of this input/configuration, not a guarantee that an
arbitrary arithmetic expression cannot overflow fp32. The guard remains live.
The sweep preceded the final padding-only local-loss and degenerate-length
refinements; the final full receipt covers the final runtime and both
MentalModel assertions.

## Reconstruction baseline

[probe.py](probe.py) extends the original item-1d probe. The native run keeps
`data/MM_ladder.xml`, seed **42**, CPU fp32, one Torch thread, batch **2**,
seven training batches (two warmup, five timed), and four validation batches
before and after. Only compilation and checkpoint writes are disabled, as in
item 1d. The dictionary still uses the unit-sphere rotation owner at rate .01
and is not an optimizer parameter.

| Native reconstruction loss | Item 1d rotation baseline | Current |
| --- | ---: | ---: |
| Before training | 0.10059325583279133 | 0.10059325583279133 |
| Five timed training batches | 0.09279306977987290 | 0.09279306977987290 |
| After training | 0.09232672862708569 | 0.09232672862708569 |

These are identical, not rounded matches. [Current record](final-source/baseline.json),
[process receipt](final-source/baseline-process.json),
[original rotation record](../2026-09-21-item1d/rotation.json).
This preserves the native configuration's existing reconstruction path; the
separate comparison below explicitly enables the current tied traversal.

The [parity XML](parity.xml) differs from `MM_ladder.xml` only by word capacity
and bucket **8 → 16**, plus `reconstructInLoop=true`. Its two rows contain
`["9 plus 1", "14 plus 1"]` and `["2 plus 1", "3 plus 1"]`, the first four
validation presentations. Both modes warm the same vocabulary in the same
order before measuring, have identical parameter and dictionary fingerprints,
and perform no optimizer step. The native lexer treats whitespace as units;
the joining space belongs to the first sentence in each packed row and is
included in the corresponding single-sentence presentation. Both modes use
`runBatch` and the outer loop's flush/reset/compaction boundaries.

| Tied byte cost | Packed | Single sentence |
| --- | ---: | ---: |
| 9 plus 1 (including its separator) | 0.6540764570 | 0.6137780547 |
| 14 plus 1 | 0.9466432929 | 0.7259087563 |
| 2 plus 1 (including its separator) | 0.7255163193 | 0.6725499630 |
| 3 plus 1 | 0.8205345273 | 0.7232422829 |
| Mean over four sentences | **0.7866926491** | **0.6838697642** |

**Parity is not demonstrated.** All four retained leaf slabs are identical;
the first sentence's sealed state in each row is also identical, yet its
reconstruction differs. Later sealed states differ too. This is not a tolerance
adjustment or a learned-utility result. The comparison records the gap as the
current baseline and leaves its resolution visible in the countdown before
expectation learning comparisons. Items 7 and 5 must preserve these measurements
unless an explicit reviewed correction replaces the baseline.
[Comparison and tolerances](final-source/parity-comparison.json),
[packed record](final-source/packed.json),
[single record](final-source/single.json), [measurement implementation](parity.py),
[comparison script](compare.py).

All three measurements were repeated after the binder correction. Their
630-file source maps match the final full receipt, and the
[measurement manifest](final-source/measurement-manifest.json) records the
probe/config hashes and confirms the source stayed unchanged. The native
means and complete packed/single measurement payloads are identical to the
earlier records retained in this directory. Each final process exited zero;
the largest peak was 1.62 GiB. A successful measurement process does not make
the reported parity null a pass.

To repeat all three fixed workloads from BasicModel, use the existing
virtualenv and a new output directory:

```sh
.venv/bin/python doc/benchmarks/2026-09-21-item10/run_measurements.py --out output/item10-reconstruction
```

The [driver](run_measurements.py) gives each workload a fresh CPU process,
an 8 GiB cap and a 600-second deadline; it records source/probe hashes and
every process result. It reports a parity null without retrying or changing
the seed. It refuses to overwrite an existing output directory.

Development attempts are retained: the first incorrectly used the 8-unit
config without tied reconstruction; the second staged packing before installing
the unit lexer; the next comparisons lacked matching separator units or common
vocabulary allocation; and one diagnostic read a transient end-state attribute
after output had cleared it. The final probe reads the owned answer program's
last sealed state. These setup attempts do not supply parity evidence.

## Seed audit and test dispositions

| Tests | Disposition |
| --- | --- |
| `test_hierarchical.py` MentalModel compatibility | Removed successful seed 0. Seed 3 now reproduces the fixed bug and must forward successfully; ordinary smoke test is unseeded. |
| `test_mm_xor.py` three learning gates | Removed selected seeds and best-of-retry loops. One initialization per gate, existing 600/900/200-epoch budgets and .26/.20/.20 thresholds. |
| `test_sigmapi.py` direct bounded pair | Removed the nine-seed search; the unpinned failure prompted Alec's nonlinearity question. Pi's tanh is cancelled by Sigma's entry atanh away from clipping. Replaced this incorrect XOR capability claim with a direct interior chart-composition identity check. The adjacent exponential-Pi/linear-Sigma XOR gate remains unseeded, with its unchanged 10-epoch LBFGS budget, MSE < 1e-4 and output tolerance .02. |
| `test_explicit_dimensions.py` XOR exact and grammar gates | Removed passing seed 4 from CLI tests, the unused grammar seed argument/constant, and instructions to pick another passing seed. Accuracy/MSE thresholds remain. Removed the two grammar expected-failure markers after the audit exposed a pre-training W=6 configuration error instead of their stated learning failure. |
| `test_reconstruction_roundtrip.py` | Removed selected initialization from model fixtures and harness calls; fresh seeds are recorded by `recon_bench(seed=None)`. Existing positive bars and 160-epoch budget remain. The old assertion that free derivation must score zero is replaced with the actual exact-recovery goal. Removed a stale exact-recovery expected-failure marker after an unseeded run passed. |
| `test_blind_decode.py` | Removed selected initialization from training fixtures and harness calls; positive recovery bars remain. |
| `test_output_path_supervised.py`, `test_generation_catalog.py` | Removed model seed 0 and sampling seed 11. The ordinary output probe records absent shared-map credit as absent: a walk may stop or select parameter-free operations. It still requires live conditioner training, finite gradients and zero conclusion credit; reached shared parameters must receive an optimizer step. The existing fixed-action shared-inverse test still requires nonzero shared credit. |

The ordinary output assertion initially failed without its selected sampling
seed ([receipt](unpinned-output-red-result.json.gz)); its invalid universal
branch-coverage assumption is now separated from the existing controlled
mechanism assertion. The corrected generation/prepared-answer tests pass.
This is not evidence that a free policy learned to use the shared maps.

The audit also found `test_mm_boolean.py` already unseeded, so it receives no
new seed or threshold adjustment. `test_reference_table.py`'s “seed-pinned”
comment describes the symbol-code construction itself, not a selected training
initialization. The [remaining RNG inventory](remaining-seed-inventory.json)
contains 306 explicit calls in 109 files, including fixed numerical inputs,
paired state comparisons and known-failure reproducers. Those calls are not
blanket-certified as seed-independent; the selections with an explicit
passing-basin rationale and the newly exposed output sampling dependence are
dispositioned above. [Inventory script](seed_inventory.py).

## Validation and limits

The unpinned bounded Pi/Sigma gate fails at **MSE .2560509443**, threshold
**.1** ([affected receipt](seed-affected-result.json.gz)). The subsequent source
inspection answers Alec's nonlinearity question: in the interior,
`Pi(x) = tanh(atanh(x) @ Wp + bp/2)` and, with the test's plain linear host,
`Sigma(z) = tanh(atanh(z) @ Ws)` (that host applies bias separately, and
Sigma's forward does not request it). Their composition has only a monotone
readout of an affine input chart, with no XOR decision boundary. The selected
seeds were not valid evidence of a hidden nonlinear feature layer. Whether a
historical passing seed exploited clipping/saturation is unmeasured.

The incorrect capability assertion is replaced with the chart identity
regression, and the layer docstrings now state the actual technique. This
changes no numerical operator. The actual exponential-feature XOR learning
assertion remains unchanged and unseeded; passing the chart identity does not
satisfy it. This disposition removes an invalid claim, not a failing example
from an otherwise valid XOR learning gate.

The explicit slow audit runs **14/14** cases with `RUN_SLOW=1`, CPU, one case
per worker, four workers and unchanged 8/28 GiB caps
([receipt](slow-seed-audit-result.json.gz)). Nine pass, three fail and two hit
old expected-failure markers. The grammar MM XOR gate measures **.2175724208**
after its 900-epoch budget against **.20**; free derivation recovers **0/4**
against exact recovery. One strict expected failure was an **unexpected pass**
of scaffold exact recovery; its marker is removed. Both grammar CLI expected
failures stop at the unsupported **W=6** peer-pipeline configuration before
learning, so their markers are removed too. These are distinct from null
learning measurements. The unwaived follow-up records both W=6 errors as
failures and scaffold exact recovery as a pass (3/3 completed,
[receipt](unwaived-result.json.gz)).

The final affected run `20260921-101810-915d13` passes **37/37** cases
([receipt](final-affected-result.json.gz),
[source manifest](final-affected-source-manifest.json)), including exact
compaction values/gradients, fullgraph capture, both MentalModel forwards,
the interior chart identity and the exponential-feature XOR learning gate.
The preceding algebra fixture mistakenly included a Sigma bias that the
plain linear host's forward does not apply; that fixture failure and its
source are preserved in [the development receipt](chart-fixture-red-result.json.gz).
The correction changes the expected formula, not the numerical operator.

The first full selection, `20260921-101843-869d6a`, exposed a contextual-bind
regression: the padding guard copied even an unpadded input, breaking the
contract that the binder retains the caller's live slab. The run was explicitly
interrupted before changing source, after **1,068/4,706** completed cases
(978 passed, 89 skipped, 1 failed; exit 130). Its
[partial receipt](interrupted-integration-result.json.gz) is not a full-suite
pass. Calls without explicit lengths now preserve the original tensor;
length-bearing recursive calls still mask padding. The existing binder
identity assertion is unchanged.

The corrected affected selection `20260921-102302-f01230` passes **48/48**
([receipt](bind-affected-result.json.gz),
[source manifest](bind-affected-source-manifest.json)).

## Full landing receipt

The default full selection `20260921-102325-6e2fd3` completes **4,706/4,706**
cases: **4,375 passed, 330 skipped, 1 existing expected failure**, exit **0**,
in **1,064.73 seconds**. There is **no waived failure and no cache retry**.
The run uses eight-case/one-file batches, ten workers, the unchanged
8 GiB per-worker / 28 GiB aggregate caps, and a **10.78 GiB** peak aggregate
footprint. The separately bounded measurement processes each remain below
1.62 GiB; even the sum of their largest peak and the full suite's largest
peak is below the 28 GiB cap.

The [full receipt](full-result.json.gz), [source manifest](full-source-manifest.json)
and [summary](full-summary.json) match all **630** final source files.
SHA-256 of the sorted compact validated-source map:
`0e47c3133f953bd5920e1a02f977509cefda0f6394ff3cf32b4bd02980a5e0d2`.
All three final reconstruction records have the same source map. Unique case
counts are reported separately from repeated phase reports. This default
receipt does not erase the explicit slow learning failures above.

Final documentation links pass **70/70**, `20260921-104236-e8452e`, exit 0,
with the same source map ([receipt](doc-links-result.json.gz)). The
[landing metadata](receipt-info.json) ties the implementation commit, full
receipt, documentation check and final measurement source maps together.
