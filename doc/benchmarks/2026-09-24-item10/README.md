# Item 11c residue and item 10 — evaluation record

Prepared September 24–25, 2026. **Uncommitted and unpushed; stopped for code
review before publication.** BasicModel baseline is `606683a8e32aac66569669a5e607f64eeec3ae32`
and WikiOracle remains at `8e7123a1c55ce10ce924d1f27b1e2037dc8890c7`.
Claude's pre-existing Language and todo edits are preserved in the resulting
documents. Item 9 has not started.

**Resolution (Alec, September 25): normalized means are dropped.** The mode,
its XML parameters and its tests have been deleted. Refine-before-raise is
accepted with patience three. The CLI XOR failure recorded in this evaluation
has been corrected through native conceptual evidence; see the [final review
resolution](../2026-09-25-item10-resolution/README.md) for current counts and
source hashes. The measurements below describe the earlier evaluated source,
preserved byte-for-byte in [evaluation-source.tar.gz](evaluation-source.tar.gz).
No rejected mode remains in the live model.

## 11c: support geometry and refinement before raising

A both reading descends through its symbol definition to the retained
order-0 evidence. Run geometry uses actual witnesses inside the subject:
PartSpace marks the matching canonical part-id tiles, including the
constituents of an unformed ordered group; WholeSpace marks the pervading
property occurrence. PartSpace's **containment read remains extent-wide**.
Its truth value cannot serve as a location mask: doing so filled unrelated
gaps between its witnesses and incorrectly made discontiguous support look
contiguous. The [failing probe](diagnostics/ps-support-before.txt) and
[passing correction selection](diagnostics/ps-support-after.txt) retain that
distinction.

Contiguous support stays in order-0 pi refinement. Discontiguous support
can assign a provisional sigma row above order 1 only after local refinement
stalls. Ordinary order-0 → order-1 symbolization remains available.
`ConceptualSpace.maybe_raise_order` is the gate on the live context-matched
promotion path. Count-based WS raising, `K_many`, and unused
`WholeSpace.passback_action` are deleted. No pi edge is introduced above
order 0 or in the symbolic loop; participation remains EWMA use.

The operational choice **accepted by Alec on September 25** is `mereologyRefinePatience=3`
completed optimizer updates without strict improvement in `min(c⁺, c⁻)`.
The worst subject reading for the definition is observed; any reduction
restarts patience, and a pure or unknown reading clears it. Repeated reads
at the same update do not advance the clock. Only scalar convergence
history persists with the definition's concept-id mapping in checkpoints.
The current field's brackets and permission to raise are released per turn;
recycled provisional definitions start with fresh history. The numerical policy was presented for review and is now accepted.

[Refinement tests](../../../test/test_refine_raise.py) cover contiguous and
scattered support, ordered groups, genuine update counts, improving/pure
readings, checkpoint recovery, batch-specific release, and the live context
observer. Three unseeded controlled learning runs supply primitive positive
memberships, learn the initially zero counterweight, retain a pure
order-1 particular, then assign an order-2 sigma over scattered particulars
after unsuccessful local refinement. “Felix” and “cat” name those fixture
roles; this is not autonomous word or object-kind discovery. Testimony-based
object kinds remain item 7.

The six [native XOR runs](xor-full.json), pools 4 and 8 with three fresh runs
each, still learn a row's positive pole `[0, 1, 1, 0]` from initial MSE .5 to
0. Located pi occurs at order 0; the learned sigma is at order 1. Exact-zero
unrelated controls, negative witnessing, nonzero pass-back focus, transient
field identity, grammar reference resolution, and getter purity remain in
the full receipt. The [MM_sparse_concept smoke](smoke.log) passes with positive
evidence for the present word and no negative evidence.

## Item 10: real fold evaluation

The evaluated implementation was in the existing `SigmaLayer` / `PiLayer` and their
grammar owners `LiftLayer` / `LowerLayer`. `normalizeSigma` and `normalizePi`
were XML evaluation parameters, both false by default; they are now removed. Perceptual fold layers are
not restored and the native min/max membership read and max concept pyramid
are unchanged. Dense factors with width-scaled initialization are evaluated;
normalized butterfly mode raises an explicit configuration error.

Signed Sigma uses the raw chart, L2-normalized columns, and convex bias
toward a learned signed constant; the candidate feeds today's Pi unchanged.
Monotone Sigma and Pi separately use nonnegative LDU factors, L1 columns,
arithmetic/geometric means and convex chart bias. The both-normalized signed
odds path is a comparison control. Reverse uses the same column scale and
tied inverse. No activation normalization, floor-gradient cap, or new
definition penalty is added.

### XOR and the domain restriction

All **64 declared unseeded trials** are in [folds.json](measurements/folds.json).
Each uses 2,500 Adam updates at .03 on four primitive input combinations,
real folds, and at least four hidden coordinates. No seed or run is selected.
Unit-energy inputs have norm .9; cube inputs have coordinates ±.9. MSE below
is on targets ±1, with a crisp bar of .05.

| Pair / input domain | Width | Crisp runs | Final MSE range |
|---|---:|---:|---:|
| Current same-chart pair, unit energy | 4 | 0/8 | ≈1.000000 |
| Current same-chart pair, unit energy | 8 | 0/8 | ≈1.000000 |
| Raw L2 Sigma → current Pi, unit energy | 4 | 8/8 | .00004679–.00044711 |
| Raw L2 Sigma → current Pi, unit energy | 8 | 8/8 | .00001187–.00004088 |
| Raw L2 Sigma → current Pi, cube | 4 | 0/8 | 1.000004–1.999997 |
| Raw L2 Sigma → current Pi, cube | 8 | 1/8 | .00025200–1.000104 |
| Both normalized, unit energy | 4 | 0/8 | .368780–.445901 |
| Both normalized, unit energy | 8 | 0/8 | .223094–.233190 |

The successful unit-energy selective runs never enter Pi's clamp; the cube
runs put 31.25–50% of hidden values there. L2 bounds a column on the unit
ball, not the coordinate cube. All six additional explicit slow assertions
pass: three width-4 runs in `test_sigmapi` and three width-8 runs in
`test_explicit_dimensions`. These are controlled layer-learning gates.

The inherited `TestXorExactCliReconstruction::test_output_mse_is_crisp`
fails unseeded at **.6183774975**; its reconstruction companion passes.
An untouched `606683a` archive also fails unseeded, with all four input
reconstructions mismatching in that separate run. A paired **measurement**
at seed 42 gives exactly the same four reported predictions on baseline,
current defaults and selective normalization: output MSE **.2763440875**
and one of four input reconstructions matched. See the [paired comparison](measurements/cli/comparison.json)
and [measurement source map](measurements/cli/manifest.json).

This fixture uses unary `P = sigma(P)`, `C = pi(C)`, `S = sigma(S)` rules
whose former tower hosts were deleted by 11c. The accepted behavior is
pass-through on an absent host, as [Language](../../Language.md) records.
The normalization parameters wire the grammar's actual lift/lower folds;
they cannot repair this fixture's missing computation. Its old butterfly
convergence explanation no longer describes the executed model. The test
and historical failure remain visible, with no skip, expected-failure marker,
seed selection or weaker target. Alec rejected leaving this gate red; the
[review resolution](../2026-09-25-item10-resolution/README.md) corrects its
configuration and readout through native conceptual space. This archived
failure does not count as a passing end-to-end XOR gate.

### Findings 4–9 in the real layers

| Question | Measurement |
|---|---|
| Width initialization | At width 1032, fixed raw off-diagonal −5 leaves own-input mean .048646, condition number 33.067. Width-scaled initialization gives .948200 and 1.05517; spread ratio .941805. |
| Log floor | At input 0, 1e−7 and 1e−6 the two-input geometric probe has output .001 and zero gradient. At 2e−6 its slope is 353.553; at 1e−4 it is 50. The singular behavior was moved, not removed. |
| Native operand domain | Of 68 operands in 34 actual binary windows, 25 have norm above 1. Maximum norm 1.775326, maximum absolute coordinate 1.069514. Leaf norm at most .761595 does not bound later compositions. |
| AM/GM on sampled operands | After mapping the sampled coordinates to `[0,1]`, the uniform AM–GM gap is .001257 on average and .003846 at maximum. This diagnostic is not a new live activation rescale. |
| Depth | Thirty alternating monotone means at width 264 keep .166289 of input spread; the input gradient norm ratio is .950647. Bounded values and nonzero gradients do not prevent contrast loss. |
| Signed column norms | At width 264 after 30 layers, L1 keeps 5.116e−9 of input norm with reverse gain 3.485e8; L2 keeps .81756 with reverse gain 2.07796. The remaining attenuation includes the convex bias gate. |
| Binary reverse | Parent round trips are within 1.8e−7 in float32. Balanced children still have MSE .0064–.0071 against distinct originals. Known-reference double-precision inverses pass separately. |

Native ranges come from owned `AnswerProgram` leaves and actions retained
from the last four validation batches after seven optimizer updates. They
are replayed through the real grammar **after timing**; maximum error from
the retained root is 1.20e−7 for current folds, 1.50e−8 for selective mode.
The [complete window sample](measurements/native-complete/current-ranges.json)
includes all committed binary windows, which are candidate fold inputs even
when another operator wins. The current sample selects Pi twice and Sigma
zero times; selective mode selects neither. Those counts limit conclusions
about trained use of the candidate itself. The [earlier selected-fold-only
sample](measurements/native-selected-only/current-ranges.json) and its
driver are also retained.

At width 64 on one-thread CPU, measured forward operator counts and times are:

| Layer | Current ATen calls / µs | Normalized ATen calls / µs |
|---|---:|---:|
| Sigma | 52 / 53.71 | 60 / 58.23 |
| Pi | 91 / 71.25 | 85 / 74.10 |

These are CPU ATen call counts, **not GPU kernel measurements**. Removing a
chart also adds normalization and bias work. The actual implementation does
not demonstrate a launch-count or elapsed-time gain for Sigma.

### Native cost and reconstruction

The preserved `MM_ladder` serial workload uses measurement seed 42, two
warm-up updates followed by five timed training updates, and four validation
batches before and after. This is a short measurement, not a convergence or
item-9 learning claim. Both declared timing pairs are preserved:

| Measurement | Current training sentences/s | Selective training sentences/s | Current final eval sentences/s | Selective final eval sentences/s |
|---|---:|---:|---:|---:|
| Selected-fold instrumentation | .51076 | .49881 | 7.08057 | 6.74816 |
| Complete-window instrumentation | .49271 | .53588 | 7.36313 | 7.52358 |

The probes run on the shared CPU while validation is active. The direction
of the timing difference changes between repeats; **no consistent speedup
is demonstrated**. Instrumentation is outside the timed inner loop.
Both pairs have identical reconstruction values:

| Phase | Current | Selective normalization |
|---|---:|---:|
| Before training | .1006144527 | .1006144527 |
| Five timed training updates | .0879226878 | .0920557365 |
| After seven total updates | .0891618710 | .0914028883 |

### Union and unknown evidence

At fan-in eight with one member fully active, five arithmetic-mean rungs
give `.125, .015625, .001953125, .000244141, .000030518`. Max and
probabilistic union keep `1` through all five. The old tanh-sum hop gives
`.76159, .64201, .56627, .51261, .47198`.

The historical [11a projection limit of 256 positions](../2026-09-23-item11a/README.md#measurements)
is retained for comparison. Current native membership evidence on unrelated
input is exactly zero; max readout stays zero at 1, 8, 256 and 4096 positions,
so there is no accumulation limit. Proposed normalized membership operators
give **.003346425** for Sigma and **.000001091799** for Pi on all-zero input,
because of convex bias and the log floor. They do not preserve unknown and
are unsuitable for adoption as membership operators in this definition.

**Recommendation for Alec: drop as a production replacement.** The controlled
chart mismatch does enable XOR on the promised unit-energy domain, but
composed native inputs leave that domain, final reconstruction is higher in
the short comparison, there is no consistent cost win, and the monotone
proposal conflicts with exact unknown evidence. Alec accepted the drop recommendation and the evaluation mode has been
deleted. The scripts require the archived evaluation source to replay.

## Preserved reconstruction and priors against d4dc385

The [serial comparison](measurements/serial/comparison.json) uses the same
preserved workload and measurement seed as the earlier receipts. Pi off,
pi on, and frozen primitive memberships agree in every phase:

| Phase | d4dc385 | Current | Delta |
|---|---:|---:|---:|
| Before training | .1005932558 | .1006144527 | +.0000211969 |
| Five timed training updates | .0927930698 | .0879226878 | −.0048703820 |
| After seven total updates | .0923267286 | .0891618710 | −.0031648576 |

No tolerance or pass/fail criterion is applied to reconstruction. All eight
11a prior rows over 256 bytes are identical, as are the discard signature
and all **160 constant-signature runs over 32 sentences**; the [prior
comparison](measurements/serial/priors.json) has zero differing signatures
and zero differing sentences.

## Validation and provenance

The final [full receipt](full-result.json.gz) completes **4,829 unique
cases: 4,502 passed, 326 existing skips and 1 existing
expected failure**, exit zero. The bounded run takes 1085.5 seconds,
with 13.98 GiB peak aggregate memory; 0 compiler-cache retries.
This is the one final full receipt. The interrupted runs are diagnostics,
not partial successes to combine into a full count.

The [affected selection](diagnostics/affected.txt) passes **140 cases**.
The final [fixture correction selection](fixture-corrections-result.json.gz)
passes **34/34**. The [explicit slow receipt](explicit-result.json.gz)
completes **28/28: 27 passed, one failed**, the inherited CLI gate discussed
above. That result includes the native smoke and all six new normalized
fold learning assertions. No new skips or expected failures are introduced.
The final documentation-link check passes **82/82**.

[Source manifest](full-source-manifest.json): **647 files**, SHA-256
of its sorted JSON source map:

`6f9f495ed94f89e6f55f293808d41bf02693180305252e05171a99befcf8ac2e`

The full and final fixture receipts match the archived evaluation source byte for byte.
All runtime, configuration and measured-gate source files also match the
measurements and explicit slow receipt. Only three test files changed after
measurement: `test_pi_sigma_inherit_grammarlayer.py` and
`test_relative_sentence_codebook_insertion.py`, and `test_structural_checkpoint.py`.
Their exact before/after
hashes are in [validation-source-delta.json](validation-source-delta.json).
The first now asserts GrammarLayer remains a **direct base**, allowing the
normalization mixin; anonymous rule identity and instance contracts remain.
The second supplies three distinct known codebook rows before testing their
nearest-row resolution and relation insertion. Its previous hand-made
queries could resolve to the same random row, so a two-child assertion had
an unstated fixture assumption. The corrected test also asserts exact
child identities. The third now asserts that deleted WS count-raise state
is neither saved nor restored, while live identity and sparse values still
round-trip. No learning capability assertion is weakened.

Earlier probes are retained under `diagnostics/`: missing normalization
mode, PS-support gap filling, a learning fixture that trained the wrong
read, missing documentation links, the two MRO assumptions and the
relation fixture. A native instrumentation attempt also failed because a
Python recorder was placed inside `torch.while_loop`; the working probe
records at the host boundary and replays retained programs after timing.
The three interrupted full attempts were stopped for the support fix,
missing receipt links, and test-fixture corrections respectively. A complete
diagnostic then covered all 4,829 cases and found only the obsolete
checkpoint-state assertion; that result is preserved separately from the
final passing receipt.
Earlier source revisions' fold measurements are also preserved under
`diagnostics/`; they are not added to or selected for the final 64-trial table.

### Affected files

- Runtime: [Spaces.py](../../../bin/Spaces.py), [Layers.py](../../../bin/Layers.py),
  [Models.py](../../../bin/Models.py), [Language.py](../../../bin/Language.py).
- Configuration: [model.xml](../../../data/model.xml), [model.xsd](../../../data/model.xsd).
- New probes: [test_refine_raise.py](../../../test/test_refine_raise.py),
  `test_normalized_folds.py` and `fold_evaluation.py` (removed; preserved in
  [the evaluated source](evaluation-source.tar.gz)).
- Existing affected tests: `test_cs_symbol_table`, `test_mereology_raise`,
  `test_mereology_word_binding`, `test_property_tiling`, `test_where_attention_handoff`,
  `test_sigmapi`, `test_explicit_dimensions`, `test_pi_sigma_inherit_grammarlayer`,
  `test_relative_sentence_codebook_insertion`, `test_structural_checkpoint`.
- Records: Architecture, Language, Mereology, Params, FutureWork, Testing,
  the 11c plan, todo and this measurement directory.

### Reproduction

For historical replay, extract `evaluation-source.tar.gz` into an isolated
copy of the BasicModel root, then run on CPU with `MODEL_COMPILE=eager`.
The rejected mode is absent from the current checkout.
`measure_folds.py` and the normalized XOR tests never set a seed. The native
and paired CLI measurements use seed 42 for comparison, never to make an
assertion pass. The saved JSON contains every declared learning run.

```sh
env -u RUN_SLOW BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager .venv/bin/python test/test_report.py --workers 10 --memory-gib 20 --batch-size 8 --max-files 1
BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager OMP_NUM_THREADS=1 .venv/bin/python doc/benchmarks/2026-09-24-item10/measure_folds.py --out output/new-fold-measurement.json
.venv/bin/python doc/benchmarks/2026-09-24-item10/run_native_measurements.py --out output/new-native-measurement
.venv/bin/python doc/benchmarks/2026-09-24-item10/run_native_measurements.py --selected-only --out output/new-selected-measurement
.venv/bin/python doc/benchmarks/2026-09-24-item11c/run_measurements.py --out output/new-serial-comparison
```

The [CLI driver](measure_cli.py) takes `--baseline-dir` containing a `git archive`
of `606683a` and an unused `--out` directory. The [package script](package_receipt.py)
checks exit statuses, complete case coverage, source maps, script hashes and
all six native XOR outputs before creating this review record.
