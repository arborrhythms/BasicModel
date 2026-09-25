# Item 10 review resolution and native CLI XOR

**Superseded inverse.**

This is the historical receipt before Alec's final reconstruction correction.
The saved native perceptual views described below were rejected. The
[forward-artifact receipt](../2026-09-25-item10-forward/README.md) records their
deletion, activity-based reconstruction and the asserted mode exclusions.

Prepared September 25, 2026. **Uncommitted and unpushed; stopped for code review.**
BasicModel HEAD remains `606683a8e32aac66569669a5e607f64eeec3ae32`;
WikiOracle remains `8e7123a1c55ce10ce924d1f27b1e2037dc8890c7`.
Item 9 has not started.

## The three decisions

Refine-before-raise is unchanged. `mereologyRefinePatience=3` counts completed
optimizer updates without strict improvement in the worst local both residual.
Improvement resets patience; a pure or unknown read clears it. Inference
passes do not advance the clock. The prior [11c residue record](../2026-09-24-item10/README.md)
documents the support geometry and accepted learning policy.

**Item 10 is decided: dropped.** The normalization mixin, Sigma/Pi mode,
lift/lower arguments, XML defaults, schema entries, Params entries and mode
tests are deleted. The [evaluation scripts and measurements](../2026-09-24-item10/README.md)
remain. Their exact measured source is preserved in
[evaluation-source.tar.gz](../2026-09-24-item10/evaluation-source.tar.gz), and
[review-resolution.patch](review-resolution.patch) shows its change to this
review source, including the deletions that were never committed.

The reasons for dropping it are measured: composed grammar operands exceed
unit energy (maximum norm 1.775326), so normalized leaf codes do not establish
the L2 bound. Thirty alternating means keep .166289 of the spread. The
geometric floor has zero gradient below it and slope 353.553 immediately
above it in the two-input probe. Convex bias gives nonzero output on all-zero
input. Selective normalization worsens the short reconstruction comparison
from .0891618710 to .0914028883, and two timing pairs show no consistent
speedup. No rejected mode remains callable in the model.

**XOR_exact passes through conceptual space.** Both original CLI assertion
bodies and decorators are unchanged relative to HEAD; the
[AST comparison](unchanged-cli-assertions.json) records that fact. The output
bar remains MSE < .05; the existing reconstruction assertion remains at its
original threshold. Neither test selects an initialization.

## Failing probe and fix

The [initial unseeded probe](diagnostics/xor-before-slow-worker-000.log.gz)
fails the crisp-output assertion at MSE .066293785. The old config ran at
symbolic order zero and named unary tower folds deleted by 11c; those rules
were pass-throughs. No gate is retired or marked expected-failure.

[XOR_exact.xml](../../../data/XOR_exact.xml) now uses symbolic order 1,
conceptual pi, aligned native perception and a four-row provisional pool.
`00`, `01`, `10`, `11` each supply two primitive positions in one word extent.
The [primitive/name lesson](../../../data/xor_concept_lessons.json) teaches
the property “is a one,” independently of the XOR labels. The ordinary
target-free observer witnesses all four located pure cases. Their pi is
performed at order 0 in the field; no pi edge exists above order 0 or in
the symbolic loop.

The order-1 output concept starts with zero sigma weights to every witnessed
case. Ordinary supervised output learning selects the two XOR cases. The
new `OutputSpace.conceptIds` binding reads that concept's positive pole by
persistent identity. It does not read a pooled both corner or a distributed
projection. Native evidence already has range [0,1], so the numeric output
denormalizer is bypassed for these bindings. Grammar `lift` and `lower` own
their chart operators in conceptual space; no perceptual fold is restored.

The generic [lesson loader](../../../bin/ConceptLessons.py) is wired before
the normal training optimizer through `architecture.data.conceptLessons`.
It teaches primitive memberships and name evidence, then observes unlabelled
training inputs and creates unwritten output edges. XOR labels enter only
the ordinary supervised output objective. Use earns pool participation by
EWMA. The config uses 64 epochs at .03 without selecting a random seed.

The first corrected forward passed the crisp check but reconstructed 0/4
inputs. Native aligned binding already retained both perceptual views; the
inverse had still been taking the older dense route. The understanding now
owns the native views used by its inverse, and ordered PartSpace vectors
decode against canonical part spans. Words need not have an interned row.
Anagrams still share the max-composed distributed code; their ordered
parts distinguish their reconstructions.

A follow-up [failing ownership probe](diagnostics/owned-inverse-before-worker-000.log.gz)
caught an inverse reading the latest live field when its understanding had
no captured views. The fix requires the understanding's explicit views;
serial derivations and free inverses do not substitute a later perception.
The serial comparison returned to its preserved values after that correction.
The regression and three native output/inverse learning runs pass.

The new symbolic configuration also exposed one strict checkpoint migration
failure: a duplicate legacy VQ key lacked its own current entry, so the
carrier ownership rewrite missed it. Migration now resolves its owner W
through the existing ownership map before removing the duplicate. The strict
restore assertion is unchanged.

## Validation

| Selection | Completed | Passed | Existing skips | Existing expected failures |
|---|---:|---:|---:|---:|
| [Full suite](full-result.json.gz) | 4,818 | 4,497 | 320 | 1 |
| [Affected files](affected-final-result.json.gz) | 423 | 361 | 61 | 1 |
| [Unchanged XOR CLI gates](xor-final-result.json.gz) | 2 | 2 | 0 | 0 |
| [Explicit slow checks](explicit-final-result.json.gz) | 23 | 23 | 0 | 0 |

All four selections exit zero, with no failed or unexpectedly passing cases.
The full suite takes 1100.2 seconds, peaks at 13.74 GiB aggregate memory and
uses no compiler-cache retries. Its [summary](full-summary.json) and
[source manifest](full-source-manifest.json) retain the bounds and hashes.
The final documentation-link selection passes **83/83**, on the same source snapshot.

The six existing grounded XOR runs (pools 4 and 8, three each) and all three
new native CLI-curriculum runs learn `[0, 1, 1, 0]` from initial MSE .5.
Every declared result is in [native-xor.json](native-xor.json). The new runs
also reconstruct all four ordered inputs and assert exact-zero output for
unrelated content. A separate identity probe swaps field rows between batch
members and verifies selection by concept id, with no negative-pole gradient.
The restored `MM_sparse_concept` smoke asserts positive evidence and no
negative evidence for each present word. Serial reconstruction ownership,
word occurrence activations and the free-generation boundary are explicitly
checked with slow tests enabled.

The only expected failure is the pre-existing
`test_topk_recovered_words_overlap_input`. No new skip or expected-failure
marker was introduced. Earlier failed probes, the interrupted pre-correction
full run, and the slow selection's aggregate-memory stop remain diagnostics.
The memory-limited selection was rerun with one worker. Only the final
complete full run supplies the full count.

All final tests and serial measurements match **648 source files** exactly.
SHA-256 of the sorted source map:

`f3976bc4b6427491c59625d228128aece337212b47c02037b433eba458ec9baf`

The [source delta](review-source-delta.json) gives per-file hashes against the
archived evaluation. Runtime source was frozen before the final validations;
the documentation and receipt were then completed and link-checked.

## Serial reconstruction and priors

The [preserved measurement](serial/comparison.json) uses the same workload
and measurement seed 42 as the `d4dc385` baseline. It defines no tolerance
or pass/fail gate. Pi off, pi on and frozen primitive memberships agree:

| Phase | d4dc385 | Review source | Delta |
|---|---:|---:|---:|
| Before training | .1005932558 | .1006144527 | +.0000211969 |
| Five timed training updates | .0927930698 | .0879226878 | −.0048703820 |
| After seven total updates | .0923267286 | .0891618710 | −.0031648576 |

All eight 11a prior rows over 256 bytes, discarded-byte signatures and
160 constant-signature runs over 32 sentences are unchanged;
[priors.json](serial/priors.json) reports zero differences.

## Affected files and reproduction

Runtime changes are in `bin/Layers.py`, `bin/Models.py`, `bin/Spaces.py` and
the new `bin/ConceptLessons.py`. Configuration is in `data/XOR_exact.xml`,
`data/xor_concept_lessons.json`, `data/model.xml` and `data/model.xsd`.
`bin/Language.py`, `test/test_sigmapi.py` and the direct GrammarLayer inheritance
test return to their pre-evaluation forms. The experimental
`test/test_normalized_folds.py` and `test/fold_evaluation.py` are removed and
remain available only in the archived source.

The new contract tests are in `test/test_concept_output.py`. Existing embedding
ownership tests use `XOR_pos.xml`, which still owns an Embedding; native
`XOR_exact.xml` owns radix percepts. The explicit codebook opt-out test now
constructs its own opt-out, and the output-shape fixture resizes all peer
slots consistently. These fixture corrections retain their original
assertions. The training-update probe loads the migrated config's inline
inputs. The two CLI gate functions are unchanged.

Architecture's three-operations note, Params, Testing, the 11c plan, todo
and the historical evaluation record document the accepted decisions.
The existing 11c residue changes to support geometry, refinement history,
checkpoint release and their tests remain part of the working tree.

From `basicmodel/`, using unused output directories:

```sh
env -u RUN_SLOW BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager .venv/bin/python test/test_report.py --workers 10 --memory-gib 20 --batch-size 8 --max-files 1 --run-dir output/review-full
RUN_SLOW=1 BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager .venv/bin/python test/test_report.py test/test_explicit_dimensions.py::TestXorExactCliReconstruction --workers 2 --memory-gib 8 --batch-size 1 --max-files 1 --run-dir output/review-xor
.venv/bin/python doc/benchmarks/2026-09-24-item11c/run_measurements.py --out output/review-serial
```

Each receipt stores its exact selected node ids and resource bounds.
[package_receipt.py](package_receipt.py) verifies complete coverage, no red
outcomes and source equality before packaging. Publishing waits for review;
no commit, push or parent bump has been made.
