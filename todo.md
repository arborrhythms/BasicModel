# TODO

> **Keeping this file useful (Alec, 2026-09-19).** This file lists what is
> left; it is not a log. A landed item is one line with its commit hash under
> "Done"; its evidence and narrative live in the document the line links to
> ([Testing](doc/Testing.md) for receipts, the topic doc for the design). An
> item leaves the list when its exit criterion is met, not when work on it
> starts. When a spec or plan leaves residue, that residue is a line here or
> it is lost. Reconcile this file at every landing, before the commit.

## Countdown to the full training session

Items count **down**: the next item to do has the highest number and **item 0
is the full training session**. (They are bullets, not a numbered list, because
Markdown renumbers a numbered list upward.) The order is the intended order; an item may
be taken early when it does not depend on a higher-numbered one. Codex builds;
Claude writes the specs and reviews each landing (Alec, 2026-09-21). "Done"
lines below keep the numbers their items had when they landed.

Work follows
[2026-09-15-next-sentence-as-the-production-objective.md](doc/plans/2026-09-15-next-sentence-as-the-production-objective.md)
§10 and its completion gates. Current contracts, limits and receipts are in
[SelectedMeaning](doc/SelectedMeaning.md),
[KernelRetirement](doc/KernelRetirement.md),
[ExpectationRetention](doc/ExpectationRetention.md) and
[Testing](doc/Testing.md). The September 17 handoff and preserved candidates
are in [the checkpoint](doc/checkpoints/2026-09-17-production-spec/README.md).

**Publish rule, every item:** failing probe → fix → affected files → one
source-matched full receipt → BasicModel commit/push → WikiOracle bump/push,
with the co-author trailer; then stop and report for review. Natural word →
operator associations use the existing compose/generate grammars. Mechanism
probes do not satisfy learning gates, and a null result is recorded as null.
**No test pins a random seed in order to pass** (Alec, 2026-09-21): a fixed
seed may make a *measurement* reproducible, never an *assertion* true. A test
that fails at some seed has found a defect; fix the defect, or let it fail.
Do not remove unused reasoning methods without Alec's review.

- **11. A concept is sigma over pi** (Alec, 2026-09-21;
   [design](doc/Architecture.md#decided-in-direction-a-concept-is-sigma-over-pi-alec-2026-09-21)).
   **Not ready for Codex until point 6 in the design note carries Alec's
   confirmation.** Each rung of the concept pyramid stops being one
   additive hop and becomes **two scatter passes over the existing COO
   `(I, J, weight)` store, pi rows then sigma rows, the chart chosen per edge
   by the target row's stamp**: **pi rows** are terms, these members
   together; **sigma rows** are classes, any of these. With `<conceptualPi>`
   on (default off until measured) a co-active set mints a term *and* its
   class, so every symbol is read through a class of terms; off, one class
   row per set as today. Classes gain terms by substitution at the seal
   (with item 7) and by language (META, two-truths part rows), never from
   code geometry. On presence, exponents signed and
   initialised at zero (zero = the member is unknown to the term; a negative
   weight is the presence of the negated activation), no normalisation:
   `∏ lit^w` and `1 − ∏(1 − t)^v` computed as `−expm1(Σ v·log1p(−t))`, bounded
   by the top-K taper; the reverse is the same pass backwards with exponents
   normalised per row, many to one (a class implies each member only
   partly). Same kernel count per rung as the additive hop. Sequences are not the concept
   store's business: `conceptualize_chain`, `chain_idx` and the JOINT
   concept go when item 7's seal chaining lands. Exit: the unseeded XOR gates
   pass through the pyramid with the option on at four or more terms;
   activation holds across `symbolicOrder` rungs with one member of several
   active (finding 10's table, in the real layer); a class with many weak
   members does not exceed its strongest member by more than a stated bound;
   the transpose reverse recovers members within tolerance; the
   reconstruction baseline of `d4dc385` unchanged with the option off, and
   the change with it on recorded.
- **10. Evaluate bounded pi and sigma folds as normalized means** (Alec's
   proposal, 2026-09-21; [proposal and first evaluation](doc/Architecture.md#proposal-bounded-pi-and-sigma-folds-as-normalized-means-alec-2026-09-21)).
   A new item; the number is reused. Today Sigma and Pi are one map in the
   atanh chart, so a stack collapses to one affine map and one tanh and the
   only nonlinearity is the clamp at the rails. Proposed: sigma a weighted
   arithmetic mean in the raw chart, pi a weighted geometric mean in the log
   chart, non-negative `L D U` weights normalised per output, a convex bias,
   exact reverses, on `u` in `[0, 1]`. Claude's evaluation is in the doc: the
   algebra holds. As written it is the **monotone** form, which is what parts
   and wholes need and which cannot fit XOR (MSE .25). **XOR belongs to
   conceptual space with the monotonic flags off** (Alec): signed weights,
   where a negative weight is the complement. There the pair that works is a
   signed normalised sigma in the raw chart feeding today's pi unchanged — 8
   of 8 unseeded runs at four hidden units, 6 of 8 at two; today's same-chart
   pair 0 of 8; both folds normalised 0 of 8, since a mean has no gain.
   **Signed weights take the L2 norm, the monotone means the L1 sum** (Alec):
   over 20 signed layers L1 keeps none of the energy and its reverse gains
   10⁹, L2 keeps all of it with reverse gain 2.3; L2 is bounded on inputs of
   at most unit energy, and overshoots into pi's clamp on cube-valued ones,
   where both are needed. Codex
   evaluates in the real layers behind a `normalize` mode, **turned on
   selectively**: first `test_sigmapi` and the crisp-XOR gates of
   `test_explicit_dimensions`, in conceptual space without the monotonic flag
   and with at least four hidden units; then the membership folds of the
   perceptual towers with the monotone means; no production config until
   measured. **Not the concept pyramid as it stands:** where sigma alone builds
   higher orders and one member of several is active, a normalised mean gives
   .125 → .016 → .002 over three orders at 8 members, while max and the
   probabilistic sum hold 1 (finding 10); compare those union forms there.
   Answer the doc's findings 4–9 with measurements (init at
   width or per-node normalisation in butterfly mode; the slope at the log
   floor; the operating range of fold inputs; contraction with depth; kernel
   count and sentences/s). Exit: the measurements recorded; the unseeded XOR
   gates passing under the new mode, or the recorded reason they cannot; and a
   recommendation for Alec — adopt (then the tanh/atanh path is deleted: no
   two permanent modes), adopt for memberships only, or drop.
- **9. Expectation learning gates.** The negative-image mechanism and residual
   query credit are in (`7d7dc4f`,
   [measurements](doc/benchmarks/2026-09-21-item2/README.md)): the predictor
   learns in controlled settings and a related continuation leaves a smaller
   remainder than an unrelated one, but the native runs are seven optimizer
   steps on one seed, prediction does not beat its context-free control, and
   the reasoning comparison is null.
   **First, its own landing: diagnose the packed/single reconstruction
   gap that `d4dc385` recorded** (mean tied byte cost .78669 packed against
   .68387 single;
   [baseline](doc/benchmarks/2026-09-21-item10/README.md)). The evidence runs
   below and the full training session both train packed, so a 15% penalty for
   packing is not a number to carry. The clue is sharp: for the first sentence
   of a row the retained leaves and the sealed state are identical and the
   reconstruction still differs, so the difference enters the reverse path
   from something other than the sealed idea; and later sentences' sealed
   states differ, so composition is not independent across a pack. Name each
   input that differs between the two modes. Exit: parity within a stated
   tolerance after a fix, or a written account of why packing legitimately
   changes the result, for Alec to accept.
   Then the gates. Exit: on the packed native config, at
   least three seeds and a run length declared in advance — ordered prediction
   beats the shuffled and context-free controls at equal updates, with
   reconstruction and discrimination no worse than the reconstruction-only
   control; and thought work at matched answer error with the conceived
   remainder against the full observation as evidence (spec test 32, plan
   §10). A null that survives that run is recorded, and Alec decides.
- **8. Evidence: learned utility and structural preference.** Prefer
   understandable structural operators when they carry the meaning; any opaque
   operator must be an ordinary grammar-MLP choice, with the structural face
   preferred at equal fit. Measure the opaque routing share and its decrease
   as structural coverage grows on the same corpus. Held-out causal utility
   against direct-answer and equal-compute baselines across seeds;
   reconstruction/discrimination controls; warmed training throughput; the
   preserved arbitrary-symbol poison probes and renamed-vocabulary controls.
   Numerical values or symbol IDs never supply learner arithmetic or answer
   seeds. Learned utility stays explicitly unproven until these comparisons
   pass. The seed audit of `d4dc385` leaves the unseeded MM-grammar XOR gate
   failing (.21757 after 900 epochs, bar <.20), and the two `XOR_grammar.xml`
   CLI gates stop before training at unsupported W=6. Keep these failures
   visible; no passing-seed selection or expected-failure waiver
   ([audit](doc/benchmarks/2026-09-21-item10/README.md#validation-and-limits)).
- **7. Two truths** ([spec](doc/specs/2026-09-16-two-truths-ideas-and-relations.md)),
   new session. One S = one LTM row: an absolute S fuses to one point and
   writes an idea row with derivation and `refs`; a relative S (generic
   subject, or any S referencing a relation) stays three slots and writes a
   relation row of kind part / implies / operator over row references.
   Clause-level seal as grammar (`NP → S`, `NP → REF(S)`), one relation writer
   at the seal, `REL_OTHER` and the reducible/ineffable routing deleted, the
   WholeSpace META taxonomy retired for a concept-level index, luminosity
   restricted to idea rows, the sentence never setting its own trust; do not
   assume one word per META (§3.4). **Object permanence by reference** (§3.5):
   translating a word to its object may tie it to an earlier noun or sentence
   in the recency buffer; identity is imputed and carried by the predictor.
   Then the distributional context widens from the sentence to the
   **situation** the predictor anchors, under three `model.xml` variables
   (plan §8.4 point 2: situation weight, anchor bound, expectation weight);
   the frames that anticipatory `what`s already hand the predictor are its
   start. With the seal chaining rows, `conceptualize_chain`, `chain_idx` and
   the JOINT / sentence concept are deleted (item 11: sequences come from LTM
   references, not the concept store). Exit: the twenty-one §7 tests, the §8
   docs, the reconstruction baseline of `d4dc385`
   unchanged, and the `true` operator over the sealed clause declared in
   `<thought>` and executable.
- **6. Stored-idea generativity.** Forgetting's dropping of derivations depends
   on it. Item 1c's probe reports zero compound recovery
   ([measurements](doc/AccessibleMind.md#measured-limits)). It trains for 8
   small updates, so it mostly measures a split/stop policy that has not
   learned to split (it does within ~100). The deeper limit is the operator:
   with the correct split actions *forced*, the tied inverse of `lower` returns
   children about 65% from their codes at depth 1, they project to the wrong
   code, and 400 updates of the probe's training do not move that. Exit:
   recovery reported separately for forced actions (the operator's inverse)
   and free running (the policy, trained to convergence), by depth and chain
   length; clean-up decoding evaluated for lift / lower — bounded candidate
   search through the forward kernel (`_bounded_binary_reconstruction`) in
   place of the reference-free affine inverse. Until a recovery rate is
   measured, item 5 must not drop derivations. The `d4dc385` audit also records
   the existing MM_20M grammar free-derivation harness at 0/4 exact recovery
   after three epochs; its acceptance assertion now requires recovery rather
   than preserving that zero ([audit](doc/benchmarks/2026-09-21-item10/README.md#validation-and-limits)).
- **5. Forgetting** ([spec](doc/specs/2026-09-16-forgetting.md)), after item 7
   (needs `refs` and every S writing a row). Document-boundary pass from the
   high-water to the low-water mark deleting the lowest-value unprotected
   rows; value = `|trust|` + utility (1 − deducibility) + luminosity
   contribution; cascade, reference remap, dependents rebuilt, the human
   profile's age term; and **detail before rows** (§4a): wording, then
   subordinate rows, then the row, gradually, coarsening the referring row
   instead of cascading where the operand's point survives. Exit: the fourteen
   §7 tests, the accessible-mind spec's test 31 (retention by surprise), the
   §6 elements in schema/`model.xml`/Params.md, the §8 docs, and the `d4dc385`
   reconstruction measurements unchanged unless explicitly re-baselined.
- **4. Run harness and resume test.** One logger per interval: reconstruction
   loss; expectation discrepancy; LTM occupancy, forgetting passes, rows
   deleted per origin, value cut-off; luminosity of provisioned truths; the
   per-shared-operator gradient cosine and norm ratio — reconstruction against
   expectation, and against output where answers are supplied — with the
   operators in persistent opposition named; the held-out two-truths §7
   test-12 probe and a fixed reconstruction sample. And a resume test proving
   a mid-epoch checkpoint restores cursor, stream count, `refs` / surprise
   columns and forgetting counters with the next batch byte-identical. Exit:
   one command on a small config; the test in the suite.
- **3. Corpus at target size.** Raise `maxDocs`, exercise multi-shard if needed,
   measure sentence-list/address-table memory and loader time, confirm
   `resume_skip` with the run's stream count. Exit: the load recorded and
   `maxDocs`/`shardDir`/stream count in the run config.
- **2. Housekeeping.** Restore safe, faster bounded-test batching: the default
   256-case/16-file and 32-case/4-file workers exceeded the 8 GiB cap, and
   eight-case/one-file batches pass the same selection; do not raise caps or
   omit cases; reconsider [fixture reuse](doc/plans/2026-09-17-bounded-fixture-reuse.md)
   if useful ([evidence](doc/Testing.md#forward-owned-lexical-forms-september-19)).
   Delete the radix-backed non-word-major meronomy path (no legacy paths), and
   remove the stale `dispatch_per_row_reset` note in
   [the fold-ladder plan](doc/plans/2026-09-10-meronomy-fold-ladder.md#open-defects-found-on-the-way)
   (`taxonomy_parent_map` is now initialised).
- **1. Compiler work.** Forward chooser split/lift once per slot; backward's
   launch-bound kernel count per brick; B24 brick +25% against pre-ladder.
   Exit: sentences/s and peak memory at the run's batch and brick size against
   the July baseline.
- **0. The full training session**: the long FineWeb run, only with items 11–2
   done and item 1 measured. Expectation on in `model.xml`; the
   `BasicModel.xml` flip follows the plan's §10 gates. Stop on rising
   expectation discrepancy, a reconstruction regression against the `d4dc385`
   baseline, or a forgetting pass deleting protected rows.

Everything that is decided in direction but not on this path is in
[FutureWork](doc/FutureWork.md).

### Done (newest first)

- `d4dc385` Item 10 (the first of that number) records the reconstruction baseline and packed/single parity gap, fixes MentalModel compaction overflow, and removes selected passing seeds (taken early from item 2; [receipt](doc/Testing.md#item-10-reconstruction-baseline-and-seed-audit-september-21)).
- `f8aa23c` Item 11 completes generation catalogue ownership, scoped output dispatch, checkpoint/Adam migration and supervised gradient boundaries ([receipt](doc/Testing.md#item-11-generation-ownership-september-21)).
- `afdcdfa` Item 12 completes the expectation review corrections: role-normalized retention surprise, indexed pairs, reading purity, bounded cache recovery and gradient norm ratios ([receipt](doc/Testing.md#item-12-review-corrections-september-21)).
- `7d7dc4f` Item 2 implements negative-image expectation and residual credit; measured joint/useful-query learning remains open under item 9 ([design](doc/ExpectationRetention.md), [receipt](doc/Testing.md#negative-image-expectation-september-21)).
- `6bf211a` Item 1d restores rotation-owned concept codes and completes the 1b/1c review corrections ([GradientFlow](doc/GradientFlow.md), [receipt](doc/Testing.md#item-1d-review-corrections-september-21)).
- `101dc22` Item 1c complete in mechanism: checked subsystem effects, indexed cued retrieval and retained frames; compound recovery remains unproven ([AccessibleMind](doc/AccessibleMind.md), [receipt](doc/Testing.md#accessible-mind-effects-september-20)).
- `7c2fa5a` Item 1b complete: objective-local state gradients and measured shared operator/codebook credit ([GradientFlow](doc/GradientFlow.md), [receipt](doc/Testing.md#gradient-factorization-september-20)).
- `6a6bb21` Item 1 complete: grammar-owned wording on real parsed text, held-out complete wordings/nouns, converse/sense controls and full-vocabulary generation/recomposition ([evidence and receipt](doc/Testing.md#working-grammar-wording-gate-september-20)).
- `d1d8b5b` Ordered compose-MLP inputs and production generate wiring (item 1 increment; the wording gate remains open, [SelectedMeaning](doc/SelectedMeaning.md), [Testing](doc/Testing.md#grammar-owned-wording-architecture-september-20)).
- `02db7ef` Review corrections: removed the standalone codec and legacy policy; completed chooser context, bounded memory and nested-controller mechanisms ([SelectedMeaning](doc/SelectedMeaning.md), [Testing](doc/Testing.md)).
- `288b56b` Initial controller replacement; review reopened item 1's wording gate ([current design and limits](doc/SelectedMeaning.md), [original receipt and review corrections](doc/Testing.md#selected-meaning-and-one-controller-september-20), [test dispositions](doc/KernelRetirement.md)).
- `60b1497` Forward owns anchored lexical forms by WORD row; capture reads retained provenance ([Testing](doc/Testing.md#forward-owned-lexical-forms-september-19); item 1 increment).
- `a70c77d` Direct `concept`-unary selected meaning (`what(quantize(x))`)
  preserves its live leaf into the normal controller; description operands
  still require owned occurrences. Ordinary pytest failures now complete their
  selected receipt while resource/protocol failures remain bounded/fail-fast
  ([Testing](doc/Testing.md#complete-diagnostic-receipts-september-19)).
- `943a64f` Lexical converse forms: `whole` vs `part` provenance frozen at
  program capture ([Testing](doc/Testing.md#forward-owned-lexical-forms-september-19)
  records the later forward-ownership correction).
- `fb5611c` Normal thought resolution grammar-owned; `resolveAnswer()` no
  longer calls `answer_query()` from a raw prompt (item 1 increment).
- `ea92851` **Item 0 complete:** model-declared `<thought>` allow-list between
  `<compose>` and `<generate>`; `<Queries>`, the `is`-aliases and
  `query="false"` retired. [Spec](doc/specs/2026-09-18-thought-operations-in-compose.md),
  [plan](doc/plans/2026-09-18-thought-operator-unification.md),
  [receipts](doc/Testing.md#explicit-thought-catalogue-september-19).
- `573fe0c` Bounded tests fast: no background QoS, `cpu−4` one-thread pool,
  28 GiB aggregate cap; 4,649 cases in 197 s (was 4,165 s serial).
- `6a34bbf` Sentence expectations retained as `estimate` occurrences without
  LTM leakage ([ExpectationRetention](doc/ExpectationRetention.md); item 2 foundation).
- `c3918a2` `9dc5c58` Typed `ThoughtResult` and nested evidence persist through
  history checkpoints (sidecar v3; [ThoughtHistory](doc/ThoughtHistory.md)).
- `e8b2f43` Catalogue refinement preserves mode, polarity, bindings, scope.
- `b705a48` `69233d1` Selected truth and `arma` results become the row's answer
  seed; `arma` emits `[3, D]`, never a fact.
- `04b34a3` Normal What runs on the selected thought trace.
- `4518db0` Structural and thought operators unified (predecessor of `ea92851`).
- `7923845` Shared selected-query work meter ([QueryWork](doc/QueryWork.md)).
- `3ab3c4b` Nested grammatical occurrences retained ([NestedRetention](doc/NestedRetention.md)).
- `3a5703e` Checked query execution phases ([QueryPhases](doc/QueryPhases.md)).
- `985b594` Replayable ordinary thought history ([ThoughtHistory](doc/ThoughtHistory.md)).
- `3ffb465` Bounded runner foundations, 512 MiB VQ tile, MPS routing receipt.
- Expectation review (plan §11): implemented and validated on September 16 ([record](doc/plans/2026-09-15-next-sentence-as-the-production-objective.md#116-implementation-and-validation-september-16)).
- Xcode licence accepted; trust term sign `|trust|` (Alec, September 18).
