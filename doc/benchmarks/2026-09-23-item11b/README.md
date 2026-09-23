# Item 11b: membership read

This landing implements [Alec's brief](spec.md). Order-0 concepts use the
existing sparse pi scatter over their signed feature definitions. Native
PartSpace percept rows and WholeSpace property rows have independent
addresses. Both symbol channels use products within an occurrence; the
extent readout unions contained occurrences. Duplicate span brackets count
once. Empty definitions and missing observations supply neither pole.

The learned coordinate maps, projection read, admission floor, calibration
harness and projection residual are deleted. Historical calibration results
remain in the earlier receipts; their deleted scripts are linked to the
baseline commit. Distributed codes follow feature definitions at the
sentence boundary and continue to serve retrieval and tied reconstruction.
The feature weights use the existing definition-sparsity penalty.

Candidate growth now runs at the sentence boundary beside promotion.
`getParameters` is pure. Optimizer construction attaches the sparse leaves
before any boundary growth, preserving moments and frozen-row hooks.
Maximum normalization remains limited to assigned provisional disjunctive
rows, as accepted. The 11a normalization residue is closed.

## Learning protocol and exact controls

Each native model receives primitive byte observations. Two initially
unnamed property rows learn one and zero membership from the examples
`0` and `1`. The conceptual name first witnesses the positive one property;
both signs of the other feature are offered at zero. Thirty-two updates on
single-byte occurrences teach positive evidence for `1`, counterevidence for
`0`, and neither for an unrelated `A`. The feature definition learns the
negative zero membership. No two-position activation table is installed.

The two-position examples then produce the required corners directly:
`11` true-only, `00` false-only, and `01`/`10` both. Four provisional
conjunction hypotheses start with the positive symbol of this one property.
Their negative-symbol candidates start at zero. After 120 supervised XOR
updates, every hypothesis must have learned the conjunction of P⁺ and P⁻.
Participation starts at zero and follows EWMA use. This is supervised
primitive naming and concept learning, not an unsupervised discovery claim.

There are three unseeded runs at each pool size, 4 and 8. The XOR read is the
positive channel of the conjunction, P⁺P⁻. Above order 0, the existing dual
fold also retains counterevidence for mixed input; its pair is not replaced
by a classical complement. The earlier OR/AND formula remains a separate
composition check.

The learned definition must read letters, a different digit, and punctuation
as exact zero in both channels at position, extent and symbol scope. The
same definition algebra is checked at 1, 8, 256 and 1,024 positions. Missing
memberships remain zero under the union, so the old 256-position accumulation
ceiling is removed. An observed binary counterexample still supplies
negative evidence; an unrelated observation supplies neither required
product. No threshold makes that distinction.

## Failing probes

The [initial probe](failing-probe.log) fails on the absent membership API and
on parameter reads growing an opposite-pole edge. The
[ownership probe](failing-ownership.log) exposes orphaned Adam state when
candidate growth precedes optimizer attachment. The
[control probe](failing-controls.log) preserves a failed attempt to learn
the primitive name through saturated multi-position unions. Naming is now
taught at single primitive occurrences; the two-position truth table is
tested only after that learning, through the ordinary extent readout.

## Reconstruction and prior protocol

[The measurement driver](run_measurements.py) runs the preserved serial
`d4dc385` workload with conceptual pi off and on. Two warmup batches precede
five measured updates; seed 42 fixes the measurement, never an assertion.
The frozen-membership ablation repeats the 11a diagnosis. Each process has
a 4 GiB limit and a 600-second deadline, with CPU/eager execution and one
Torch thread. The prior comparison checks all 8 × 256 coefficients and the
same 32-sentence, 160-run corpus used in 11a.

[The reconstruction comparison](final-source/comparison.json) gives:

| Phase | d4dc385 | 11b, pi off and on | Memberships frozen |
| --- | ---: | ---: | ---: |
| Before training | .1005932558 | .1005932558 | .1005932558 |
| Five-update training mean | .0927930698 | .0949083805 | .0927930698 |
| After training | .0923267286 | .0924575571 | .0923267286 |

The native values match 11a exactly. Relative to d4dc385, the training mean
is 2.28% higher and the final reconstruction is 0.142% higher; freezing the
learned primitive memberships reproduces the reference exactly. This serial
control does not exercise the parallel membership cutover. Packed/single
parity remains the first landing of item 9.

[The prior comparison](final-source/priors.json) has zero coefficient
error, zero differing byte signatures, and zero changed sentences or runs.

## Validation

The affected runtime files are `Spaces.py`, `Layers.py`, `Models.py`,
`PerceptProperties.py` and `ConceptEvidence.py`; the model defaults and
schema delete the floor. The affected tests cover native membership reads,
learned XOR, sparse definitions, promotion, optimizer ownership, codebook
reads, dual towers, primitive properties and reconstruction. Architecture,
parameter documentation and the todo record describe the resulting contract.

The [final affected-file run](witness-fix-result.json.gz) completed all 251
selected cases: 233 passed and 18 skipped. Its
[source manifest](witness-fix-source-manifest.json) matches the 637-file
snapshot in the [measurement manifest](final-source/manifest.json).
[The six XOR records](xor.json) have MSE `2.842170943040401e-14` each;
every run also asserts the four extent corners and exact-zero unrelated
controls at all three scopes.

The supplementary [contract run](contracts-result.json.gz) passed 71 cases,
including the slow symbolic checks. It predates the final missing-witness
guard in `Models.py`. Earlier probes and their manifests are retained;
[source differences](validation-source-deltas.json) distinguish them from
the final source. The [first full attempt](interrupted-full-result.json.gz)
was interrupted after 396 cases when non-radix PartSpace models exposed an
absent native witness. Missing observations now supply neither pole through
the same membership path; the 251-case run checks that fix.

The full run uses CPU/eager execution, ten workers, a 20 GiB aggregate limit,
eight-case batches and at most one test file per worker:

```sh
BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager .venv/bin/python test/test_report.py \
  --workers 10 --memory-gib 20 --batch-size 8 --max-files 1 \
  --run-dir output/item11b-full-verified
```

The [full receipt](full-result.json.gz) completed all 4,775 unique selected
cases: **4,444 passed, 330 skipped, one existing expected failure**, exit 0.
It took 1,084.86 seconds and peaked at 17.03 GiB aggregate memory. The raw
worker records contain four additional passing reports for the same
`TestOrthogonalFlags::test_flags_match_expected` case; counts here use unique
node IDs. No worker failed or required a compile-cache retry.

The existing expected failure is
`test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input`:
reversal from the cleared STM cache still fails its recovered-word overlap
criterion. Its marker and implementation are unchanged. The six native XOR
gates all pass without an expected-failure waiver or seed selection.

The [full source manifest](full-source-manifest.json), final affected-file
manifest and measurement manifest have identical 637-file source maps.
Their SHA-256 over the sorted JSON source map is
`11407c2dfee90f1d84a7952dd2a9ef0327c03bb17e84e361451b9e3b75fd4e61`.
These maps cover runtime, tests, model data and root build/configuration
files. Mutable prose is recorded separately. After completing the receipt
and todo, all 76 documentation-link checks passed. The staged source and
prose passed `git diff --check`; raw logs and generated measurement XML
retain their captured whitespace and are excluded from that formatting check.
