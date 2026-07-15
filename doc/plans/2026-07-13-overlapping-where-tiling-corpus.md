# Overlapping `.where` tilings in the subsymbolic pump — corpus validation and execution

> **Status:** opt-in implementation landed and curated gate passed,
> 2026-07-13; a default-cutover trial failed reconstruction gates, and the
> external held-out corpus gate remains open.  This is an
> experimental replacement for the row-wide `n_runs` passback heuristic.  It
> is accepted as the default mereological parser only after the corpus gates
> below pass.  The existing `mereologyRaise` path remains the rollback seam.

## Question and scope

Can the existing PS/WS↔CS subsymbolic loop find useful mental-object sizes by
feeding an overlapping lattice of part and whole `.where` candidates through
the callosum, instead of learning a merge logit at every possible boundary?

This plan tests **surface-object parsing**: discovering word-like, punctuation,
and enclosing sentence spans.  It does not claim to score a syntactic
constituency or dependency tree; that belongs to `syntacticOrder` and the
post-pump symbolic grammar.  The handoff criterion is that the surface tiling
is reliable enough for that grammar to receive stable sentence objects.

## Architecture under test

The fixed `subsymbolicOrder` loop remains the only refinement loop.

1. PS supplies bottom-up candidate parts with exact byte `.where` spans.
2. WS supplies overlapping whole candidates: typed runs, separator-bounded
   words, and the enclosing sentence.  These candidates coexist; they are not
   forced into a single segmentation before the callosum.
3. CS knits candidates by equality and immediate containment.  A part is
   assigned to its smallest strict containing whole, so a sentence candidate
   cannot steal bytes from an available word candidate.
4. Per local family, not per batch:
   - exact PS/WS span agreement is a settled identity and stays in the tiling;
   - several contiguous PS parts covering one WS whole request σ synthesis;
   - one PS part covered by several immediate WS children requests π analysis;
   - a gap/disconnected family raises order rather than inventing a boundary;
   - unsupported candidates remain null/unresolved.
5. Every local family is routed in parallel.  There is no batch `amax` and no
   single winning `.where`.  On later passes σ is applied only to the PS slots
   that requested synthesis; the ordinary WS pass-`t` π stack continues to
   analyse the whole stream.  Settled objects remain available to higher
   wholes and to the final feed-forward conceptual pyramid.
6. The callosum receives both valid candidate masks and carries the complete
   tiling observation on its CS output.  The symbolic FF pyramid still fires
   once after the continuous pump; it consumes the settled mixed field rather
   than becoming a second structural selector.

The structural decision is parameter-free.  Gradients train the candidate
representations, σ/π folds, callosum, and downstream task.  They do not have to
be converted into one merge decision per byte boundary.

## Implementation tasks

- [x] Add a fixed-shape `WhereTilingLayer` beside `RunStructureLayer`.
      Inputs are `[B,P,2]` part spans and `[B,W,2]` whole spans.  Outputs retain
      pairwise equality/containment, immediate parent/child incidence, local
      part and whole counts, coverage/run checks, settled/σ/π/raise masks, and
      valid candidate masks.
- [x] Record true per-percept spans in the radix/meronomy PS stem.  Keep the
      existing word-tile span record unchanged for reconstruction.
- [x] Under `<architecture><overlapWhereTiling>true</...>`, stage the WS union
      of typed, word, and sentence spans, deduplicated per row.
- [x] Build a `subsymbolicOrder`-length refinement schedule in the eager stem.
      The schedule is metadata over fixed slots; neural content continues
      through the existing compiled pump.
- [x] Replace the batch-collapsed structural passback with per-slot σ masks on
      the experimental path.  Preserve explicit reading-attention `.where`
      scopes as the higher-priority override and preserve the old action API
      for non-experimental configs/tests.
- [x] Carry the tiling observation through `ConceptualSpace.bind_streams`,
      zeroing invalid pad candidates before callosal glue and exposing the
      final accepted/unresolved tiling for evaluation.
- [x] Add XML schema/default documentation and a dedicated experiment config.
- [x] Add unit, integration, curated-corpus, and regression gates.  The
      external-corpus gradient/quality acceptance run remains deliberately
      open; the feature is not the default until that run passes.

## Corpus protocol

### Corpora

Use three increasingly realistic rungs; never tune on the held-out rung.

1. **Curated adversarial UTF-8 JSONL** checked into `test/fixtures`: short
   sentences covering repeated spaces, tabs, punctuation runs, apostrophes,
   hyphens, mixed letters/digits, non-ASCII text, emoji, empty rows, and rows
   longer than the candidate budget.  Each record carries gold UTF-8 byte
   spans and a split label.
2. **Repository text smoke:** deterministic samples from `data/sample.txt`
   and the locally available FineWeb parquet shards.  This rung checks
   invariants and throughput where no gold spans are available.
3. **External gold corpus:** a fixed train/dev/test release of a CoNLL-U
   corpus, converted once to JSONL byte spans.  Preserve document/sentence
   splits and record corpus name, version, checksum, conversion command, and
   tokenizer policy in the result artifact.  No network access is required by
   tests; the evaluator accepts a local JSONL path.

The evaluator consumes one JSON object per line:

```json
{"text":"The cat slept.","spans":[[0,3],[4,7],[8,13],[13,14]],"split":"test"}
```

Offsets are UTF-8 byte offsets, matching PS/WS `.where`.  Zero-width
multi-word-token rows and empty nodes from CoNLL-U are excluded; punctuation is
included.  The conversion policy must be frozen before measuring the test set.

### Metrics

Report by split and by diagnostic bucket:

- span precision, recall, and F1 over basic-level accepted spans;
- exact-sentence tiling rate;
- byte coverage, overlap error, and cross-whitespace error;
- convergence rate and pass-count histogram;
- unresolved/null, σ, π, raise, candidate-overflow, and truncation rates;
- hierarchy recall for gold word→sentence containment;
- reconstruction exactness and downstream sentence-task loss/accuracy;
- wall time, candidates/byte, and peak memory versus the existing
  `RunStructureLayer` route and a boundary-logit estimate.

All aggregate metrics must also be stratified by sentence byte length,
unknown-word status, Unicode/non-Unicode, punctuation, and mixed alphanumeric
content.  A high mean may not conceal a failed diagnostic bucket.

### Acceptance gates

The architecture may replace the old passback heuristic only when all are true:

1. Curated corpus: 100% exact tilings, no overflow except records explicitly
   labelled as budget probes.
2. External held-out corpus: span F1 ≥ 0.99, exact-sentence ≥ 0.97, byte
   coverage ≥ 0.999, and cross-whitespace error ≤ 1e-4.
3. ≥ 99% of sentences settle within configured `subsymbolicOrder`; unresolved
   cases are reported, never silently promoted to identities.
4. Every σ/π fold used by the experiment receives a finite nonzero gradient
   on a corpus training batch; pad/settled masking does not sever the head.
5. Existing reconstruction and sentence-task metrics do not regress by more
   than 1% relative, and all non-experimental tests remain green.
6. Median forward time is no worse than 1.15× the old route and materially
   below an explicit all-boundary scorer at the same maximum sentence length.

If a gate fails, keep the feature opt-in and retain its diagnostic artifact.
Do not weaken the gold policy or silently fall back to the lexer to make a
failed row pass.

## Tests and commands

- `pytest -q test/test_where_tiling.py`
- `pytest -q test/test_where_attention_handoff.py test/test_where_run_structure.py`
- `python bin/eval_where_tiling.py test/fixtures/where_tiling_corpus.jsonl`
- `python bin/eval_where_tiling.py /path/to/heldout.jsonl --json output/where_tiling.json`
- targeted model gate on the overlap config: forward twice, backward once,
  assert deterministic eval output, finite loss, σ/π/callosum gradients, and
  that the CS output carries the same accepted tiling staged by WS.
- broader regression: the mereology, dual-tower, reconstruction, symbolic
  pyramid, compile/export, and sentence parsing suites, followed by `make test`
  before default cutover.

## Result record

For every corpus run, store config and git revision; random seed; device and
compile mode; corpus identity/checksum; candidate budget; all metrics above;
the first 100 failed rows with predicted/gold spans and route trace; and the
baseline delta.  The checked-in plan is updated with the exact artifact paths
and the accept/reject decision.  A passing curated test is an implementation
gate, not evidence by itself that the external-corpus acceptance bar has been
met.

## Execution notes — 2026-07-13

Implementation landed behind `<overlapWhereTiling>` and requires
`<mereologyRaise>true</mereologyRaise>`.  `data/MM_overlap_tiling.xml` is the
acceptance vehicle; `MM_mereology.xml` and production grammar configs retain
their previous route.

- `WhereTilingLayer` emits per-candidate equality, immediate parent/child,
  coverage, run, settled, σ, π, raise, and diagnostic-route tensors.  Exact
  identities can simultaneously participate in a larger σ family.
- Radix PS records exact emitted-percept byte spans separately from the
  enclosing word-tile reconstruction record.
- WS stages word/separator, typed, and sentence candidates and builds the
  fixed refinement schedule in the eager stem.  Candidate overflow is an
  explicit tensor.
- Later PS passes blend learned σ feedback only into `sigma_part` slots.  The
  opt-in parallel WS path uses `pi_whole` to select learned π output; explicit
  reading-attention scopes retain priority.  Row-local `[B,2]` scopes are now
  handled without collapsing to row 0.
- Callosal bind masks only invalid pads and carries the full observation on
  the CS subspace, so accepted and unresolved levels coexist at the late FF
  cutover.
- Curated UTF-8 corpus (`test/fixtures/where_tiling_corpus.jsonl`): **106/106
  spans**, **26/26 exact sentences**, F1 **1.0**, zero overflow, via
  `bin/eval_where_tiling.py --passes 3` (matching the experiment config).
  One enclosing non-basic candidate remains unresolved on the mixed-
  alphanumeric diagnostic; all scored basic spans settle.
- Targeted/new tests: **39 passed**.  Broader tiling + passback + mereology +
  dual-input/tower + reconstruction sweep: **120 passed, 3 skipped**.  The
  skips and 25 warnings are pre-existing platform/config diagnostics.

This does **not** satisfy the external held-out acceptance gate.  Next action
before default cutover: convert/freeze the chosen CoNLL-U corpus policy, run
the train/dev/test evaluation and gradient/throughput probes, and append the
result artifact and accept/reject decision here.

### Default-cutover trial — rejected 2026-07-13

A trial made omitted `overlapWhereTiling` follow `mereologyRaise`, retaining
an explicit-false legacy opt-out.  Focused tiling/handoff tests passed after
updating the legacy no-attention expectation, but the stronger `make test_all`
gate found reconstruction regressions in `MM_20M_xor.xml` (which has
`mereologyRaise=true`):

- `test_recon_bench_blind_flag`: scaffold exact match was **0.25**, expected
  **0.5**.  An in-memory A/B run forcing the overlap path off restored this
  test exactly.
- `test_mm20m_xor_blind_roundtrip`: blind exact match was **0.0**, expected
  **1.0**, with `.where` recovery **0.6667**.  Forcing overlap off improved
  exact match to **0.5**, although the current dirty-worktree baseline still
  missed the independent 1.0 pin.

The full slow run was stopped after the causal failures: **376 passed, 4
skipped, 2 failed** out of 3,343 collected tests (11% reached), report
`output/20260713-143414_report.html`.  Since disabling overlap removed one
failure and materially improved the other, the default trial itself caused a
real regression even though a separate baseline regression also remains.

Decision: restore `model.xml` to `overlapWhereTiling=false`; keep
`MM_overlap_tiling.xml` as the explicit acceptance vehicle.  Do not retry the
default cutover until the overlap path preserves the short and long
reconstruction trajectories in addition to passing the external corpus,
gradient, and throughput gates.
