# 2026-06-09 build batch -- remaining work (granular)

Status: **active execution plan.** Landed this session: **#11** (regression
head) verified; **#13** (PS-codebook retire / SS-quantize default) APPLIED
but left **4 regressions** (the subagent was killed before it verified the
full suite). Suite is at **13 failed / 2131 passed** (9 deferred IS-shape +
4 from #13). Each task below returns the suite to the **9-failed** baseline
(regression-verified; the **D** semantic validation is deferred).

Design cross-refs: `2026-06-09-asymmetric-vq-symbolic-ss.md` ($\S$4 routing,
$\S$5 combiner, $\S$6 semantic; $\S$7 tasks 7--14). Decisions:
`2026-06-08-mm20m-architecture-backlog.md` $\S$1 (#11), $\S$2 (#13).

Verification (every task): sibling venv --
`BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager PYTHONPATH=bin "<WikiOracle/.venv
python>" -m pytest test/ -q` (the tests; `make test`'s report writer also
needs `PIL` in `basicmodel/.venv`). Each task: suite returns to **9 failed**
(no NEW failures), `test_basicmodel` green, XOR still solves + reconstructs.

## Landed
- **#11** -- config-honoured regression head + `<readout>` knob
  (`MM_20M` $\to$ `sigmoid`; default `identity` is byte-identical to the old
  bare-linear head). Verified; `test_basicmodel` 206.
- **#13** -- PS `<codebook>` removed (PS $=$ subsymbolic `none`); SS
  `quantize` default. **APPLIED, NOT clean** -- 4 regressions (Task 0).

## Task 0 -- stabilize #13 (the 4 regressions)
PS-codebook-removal side-effects. Resolve so PS $=$ `none` does not break the
paths that relied on a PS codebook surface:
- `test_basicmodel.py::TestReconstructionSymbols::test_forward_output_shape_unchanged`
- `test_explicit_dimensions.py::TestIdempotentCliRuns::test_cli_does_not_crash`
- `test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct`
- `test_ir_fullgraph_compile.py::test_idempotent_forward_compiles_fullgraph_eager`

Likely a config/reader fix (XOR_exact still needs a percept surface for
reconstruction; the forward output shape must be preserved). **User
hypothesis to test first:** some of these may be *resolved by Task 1 / Task 3*
(parallel quantize / the new combiner) rather than a standalone patch -- run
those and re-check before hand-fixing.
Acceptance: suite back to **9 failed**; `test_basicmodel` green.

## Task 1 -- C-8: parallel SS quantize
Make `Codebook.quantize()` genuinely fire in the parallel `_forward_per_stage`
path (today it is a no-op there). Decision (2026-06-09): parallel SS
quantizes -- the symbolic/quantized path is NOT serial-only.
Acceptance: the SS VQ is live in parallel; XOR still solves + reconstructs;
suite 9-failed.

## Task 2 -- C-9 + C-11/12: asymmetric STE routing, drop commitment + EMA
- Forward / output: STE, $z_q = z_e + \mathrm{sg}(e[\mathrm{idx}] - z_e)$
  (gradient $\to$ encoder; code detached).
- Reverse / recon: plain gather $z_q = e[\mathrm{idx}]$ (no STE; gradient
  $\to$ codebook).
- Remove the standard commitment loss and the EMA codebook update for the SS
  VQ. Keep the $\pm 1$ operating range (encoder near its code so STE stays
  honest).
Acceptance: invertibility + XOR regression; suite 9-failed.

## Task 3 -- C-10: 2-stream ILL combiner
Replace the 3-stream `combine(PS, SS, CS)` with $\mathrm{CS} =
\mathrm{ILL}([\mathrm{PS} \Vert \mathrm{SS}])$ -- the butterfly
`InvertibleLinearLayer`. The CS layer materialises PS $+$ SS into one
double-wide event (internal only), runs the ILL, stores a **single** CS
subspace, and returns the PS / SS `.active` views (same `.what` Basis);
`OutputSpace` accepts both. No new parameters.
Acceptance: round-trip invertibility + XOR regression; suite 9-failed.

## Task 4 -- BPE/MPHF: shared byte codebook
Share the byte / `PerceptStore` codebook across chunking front ends so a
`bpe`/`mphf` chunk decodes through the SAME byte codebook on the reverse path
that `radix` uses (segmentation becomes a strategy over one shared
embedding/codebook, not a private table). See
`2026-06-08-mm20m-architecture-backlog.md` $\S$4.
Acceptance: `bpe` and `mphf` reconstruct the smoke prompts byte-perfect under
`lexer=byte`, equivalent to `radix`/`lexicon`; suite 9-failed.

## Task 5 -- C-13: semantic arrangement (mechanism only)
Wire the post-sentence SS-heat pode/antipode mechanism (semantic-centroid
attraction $+$ antipode repulsion over SS heat across conceptual orders).
Mechanism only -- the semantic payoff is validated under **D (deferred)**.

## Deferred
- **D** -- corpus validation on the serial / grammar path. XOR cannot
  validate the symbolic side (`asymmetric-vq` $\S$8: parallel $=$ no VQ; parity
  $\neq$ semantics); needs a real corpus and present supervision.
- **`[4,-1,5]` IS-shape bug** (the 9 standing failures) -- the literal
  5-wide InputSpace reshape. **Deferred deliberately:** the GitHub plan
  `2026-06-08-analysis-synthesis-dual-input.md` reshapes the IS feed (IS $\to$
  PS as `[B,1,N]`, IS $\to$ SS as `[B,N,1]`), which subsumes the 6 IS-resize
  issues -- fix them there, not here.

## Suggested sequencing
0 (or fold into 1/3) $\to$ 1 $\to$ 2 $\to$ 3 $\to$ 4 $\to$ 5. Commit per task
for granularity; keep the suite at 9-failed after each.
