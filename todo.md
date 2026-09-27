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

**Current sequence (Alec, 2026-09-27):** 7 → the
conference freeze (with item 4's pulled-forward word-level evaluator) →
6.8 → 6.5 → 6 → 5 → 4 → 3 → 2 → 1 → 0. Items 9 and 8 are implemented and
reviewed; what remains of them is empirical and waits for the
million-sentence checkpoint that item 0's run provides, so they are not
in the implementation queue.

Work follows
[2026-09-15-next-sentence-as-the-production-objective.md](doc/plans/2026-09-15-next-sentence-as-the-production-objective.md)
§10 and its completion gates. Current contracts, limits and receipts are in
[SelectedMeaning](doc/SelectedMeaning.md),
[KernelRetirement](doc/KernelRetirement.md),
[ExpectationRetention](doc/ExpectationRetention.md) and
[Testing](doc/Testing.md). The September 17 handoff and preserved candidates
are in [the checkpoint](doc/checkpoints/2026-09-17-production-spec/README.md).

**Publish rule, every item (updated by Alec, September 24):** failing probe →
fix → affected files → one source-matched full receipt → **stop for Claude's
code review** → resolve review → BasicModel commit/push → WikiOracle bump/push,
with the co-author trailer. Do not commit or push before the review.
Natural word →
operator associations use the existing compose/generate grammars. Mechanism
probes do not satisfy learning gates, and a null result is recorded as null.
**No test pins a random seed in order to pass** (Alec, 2026-09-21): a fixed
seed may make a *measurement* reproducible, never an *assertion* true. A test
that fails at some seed has found a defect; fix the defect, or let it fail.
Do not remove unused reasoning methods without Alec's review.

**The grammatical operations and their inverses are still in development
(Alec, 2026-09-26).** The compose, thought and generate catalogs are a
fragment, their routing is learned with sparse credit, and their tied
inverses are exercised mostly by reconstruction. No item's gate, and no
conference number, should be read as a claim about the performance of the
grammar as a set yet; a poor result on wording, parse depth or free
generation is expected at this stage and is recorded, not tuned away.

- **9. Mature-checkpoint learning evidence.** The mechanism, checkpoint exposure
   counter and quality-evaluation machinery are reviewed and implemented
   ([receipt](doc/benchmarks/2026-09-26-item9-followup/README.md)). Learning
   acceptance remains deferred until a checkpoint has at least **one million
   completed FineWeb training sentences**; missing exposure skips, never passes,
   the quality evaluation. Correctness checks stay unconditional. No qualifying
   checkpoint has been evaluated. Then run the loaded-model wording, prediction
   and predictive-thought checks; the full causal gate still needs separately
   trained, equal-update controls across at least three seeds, with tasks, run
   length and thresholds declared first. Ordered prediction must beat shuffled
   and context-free controls without worsening reconstruction/discrimination
   against reconstruction-only; thought must improve error at matched work or
   reduce work at matched error. Same-checkpoint ablations do not establish this
   causal result, and fused AMP conservatively undercounts exposure. Preserve the
   tiny-run nulls, incomplete reverse programs and 5/56 wording failures as
   development diagnostics ([original receipt](doc/benchmarks/2026-09-26-item9/README.md));
   old context-free and assertion-thought scores are invalid controls. Item 7 is
   the next implementation task while this empirical gate waits for training.
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
   pass. The seed audit of `d4dc385` left the unseeded MM-grammar XOR gate
   failing (.21757 after 900 epochs, bar <.20), and the two `XOR_grammar.xml`
   CLI gates stopped before training at unsupported W=6. Keep these failures
   visible; no passing-seed selection or expected-failure waiver
   ([audit](doc/benchmarks/2026-09-21-item10/README.md#validation-and-limits)).
   *Compatibility:* `XOR_grammar.xml` composes `not` / `conjunction` /
   `disjunction` over symbols on the serial path and never used the deleted
   tower folds, so its gate stands as written. Reconstruction comparisons use
   the [reviewed 9b baseline](doc/benchmarks/2026-09-26-item9b-occurrence-fix/README.md):
   serial before/during/after .1005906649 / .0948241442 / .0928765051;
   packed/single byte reconstruction .6839025617 with exact parity. Learning
   quality gates follow the million-sentence prerequisite in item 9; mechanism
   and regression checks remain unconditional.
   The [item-8 implementation](doc/benchmarks/2026-09-26-item8-review/README.md) implements
   exact-score tie preference and committed-program routing measurements;
   its [protocol](doc/benchmarks/2026-09-26-item8/PROTOCOL.md) predeclares the
   qualified causal study. Learned utility and declining opaque routing remain
   unproven. W=6 scheduling now reaches the next unchanged
   CLI blocker: the six-row symbol inventory exhausts during the first epoch's
   reset/autobind, after batch execution. Keep both CLI assertions and the
   historical MM failing-seed result visible. All three declared diagnostic
   seeds were attempted; seed 1 hit the 8 GiB guard and remains incomplete.
   The initial 4,968-case full sweep exits 1 on an unchanged strict-XPASS
   reconstruction marker. Claude's [review corrections](doc/benchmarks/2026-09-26-item8-review/README.md)
   use the marginal uniformly and require complete reconstructed positions;
   the marker and threshold remain intact. The corrected source passes its
   single 4,980-case full sweep: 4,653 passed, 326 skipped and one expected
   failure. Reconstruction exactly matches reviewed 9b. The review corrections
   are validated for the requested commit; item 8's empirical gates remain open.
- **7.5. Deferred compose follow-ups (non-blocking).** Specify the nonzero
   training temperature and sentence parsimony/work term accepted for a later
   spec; evaluation remains deterministic. Restate the shared-operator report
   assertion around seal contributions or forced parametric selection, and
   rename the native context-read `exploration_trial` flag in housekeeping.
   The unchanged depth-three campaign remains red; retain its assertion and
   the XOR/MM evidence ([accepted landing](doc/benchmarks/2026-09-27-item7-5-landing/README.md)).
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
   **The taxonomy grows by language over those identities** (Alec,
   2026-09-22): "cats are animals" writes its part row between the *object*
   concepts the words resolve to, type or token as §3.5 decides, never
   between words — into the concept store's own hierarchy of higher-order
   concepts, which is the taxonomy (§3.4 as amended; item 11), trust on the
   LTM row.
   The seal must store `(c⁺, c⁻)` and preserve both separately from neither
   (§1.1); the current scalar `_collapse_trust` is replaced here, and §3.1's
   "scalar trust" row schema is amended in the same landing, gaining the
   sealed field's `.where` and `.when` (item 9b; the row's address is its
   `.when`, its `.where` is what it was looking at). Item 11's
   paired conceptual field and checkpoint do not implement this LTM seal.
   *Compatibility with 11c (Claude, 2026-09-25):* (a) the taxonomy is the
   concept store's sigma rows, and order is taxonomic depth: "cats are
   animals" is a sigma edge from the *animal* row to the *cat* symbol one
   order below, so the seal places a kind one order above what it
   subsumes and symbolizes when no row exists there; it never writes a
   conjunctive edge above order 0 or a same-order part row. (b) Testimony
   writes directly; the refine-before-raise gate governs only discovery
   from context. (c) 11c entry 11 lands here: a word form addresses a set
   of concept ids across orders (event, particular, kind); the particular
   is order-1 symbolization and *is* object permanence, and the seal binds
   the form to it — today production binds only the order-0 word and its
   object. (d) A pronoun ties to an order-1 particular, never a kind.
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
- **6.8. One attention: brackets, narrowing, and expectation at every bracket**
   ([plan](doc/plans/2026-09-27-item-6-8-one-attention.md); Alec, 2026-09-27;
   after 7.5 and 7, before 6.5; three non-blocking questions in plan §6).
   The simplification: attention is one mechanism, a bracket over the input.
   Open awareness is the bracket set to the whole input, read by the field,
   whose only operations are the order-independent ones — and, or, not —
   because a pooled reading admits nothing else; that is the criterion for
   what belongs to the field. Everything order-dependent (lift, lower, verb,
   preposition) is the grammar and acts *between* brackets, over the
   sequence of readings that narrowing produces. Mode exclusion stops being
   a rule and becomes a theorem. The word loop is the narrowing schedule:
   the four corners give the reading policy already decided in pieces —
   true or false, move on; *both*, divide the bracket; *neither*, look
   closer, descend, mint if nothing is there — so serial reading is
   "narrow until each bracket's encoding is pure". A known word reads
   purely at its word bracket and narrowing stops; an unknown word reads
   neither and narrowing continues to bytes, where `interpret` mints; a
   known multi-word unit reads purely at the wider bracket and is glossed
   (speed reading that slows at novelty). Parallel-first falls out: the
   open read is the first bracket. **Expectation** (Alec: the mathematical
   sense; applied with the opposite sign so that surprise is what is
   processed; held for any subject of attention — past, current or future
   frames, a whole sentence, the next concept) becomes one mechanism at
   every bracket level — next byte in a word, next word in a sentence,
   next sentence in a document, next row in the chain — the same ARMA
   machinery, negative image and surprise column with the level as an
   argument, giving a training target at every bracket instead of one per
   sentence, and making the pilot's next-word gate this predictor at the
   word bracket rather than a separate scorer.
   **Landing 6.8-1 (the fast loop is kept):** the stop is pinned at
   words, so the compiled per-word step keeps its shape; narrowing and
   glossing enter as candidates in the 7.5 softmax (the same chooser and
   straight-through learning, no separate policy), which is why this
   follows 7.5; the seal's sentence-bracket row with its `.where`/`.when`
   is the terminal encoding, which is why it follows 7; 6.5 then adds
   identity binding as further candidates in the same softmax, which is
   why it precedes 6.5. Deletions, per the no-legacy rule: `modeSchedule`,
   `serial` as a mode (the bracket schedule replaces it), the
   subsymbolic-versus-symbolic loop distinction and the two order budgets
   (one narrowing budget replaces them), and the boundary-only sentence
   predictor as a separate object (expectation at every bracket replaces
   it). Exit: the XOR gate and exact-zero controls unchanged; the serial
   reconstruction baseline within the reviewed-9b tolerance; the
   word-level predictor scoring the frozen NanoChat item manifest; the
   both-rate and categorical-discrimination fields logged per level (item
   4); item 9's prediction gates re-declared at the word bracket. Costs to
   measure, not assume: the open pass against item 1's per-word baseline;
   the reliability of the field's *both* that the policy turns on.
   **6.8-2, after the conference (FutureWork):** the dynamic stop —
   glossing above the word bracket and descending below it only at
   novelty — decided against item 1's throughput baseline.
   **Conference sequencing (Alec, 2026-09-27):** finish 7.5, land 7,
   freeze the demo checkpoint; before the freeze pull forward only the
   word-level predictor as the NanoChat gate's evaluator (recorded under
   item 4); start 6.8 after the freeze.
- **6.5. Independent components: identity as columns, verbs as change**
   ([spec](doc/specs/2026-09-26-independent-components.md)), after item 7
   (needs the seal writing every S as a row and §3.5's reference chain).
   Today identity is bookkeeping and prediction trains only the predictor:
   `interpret` binds a word to its object by set logic outside autograd,
   `resolve_word_concept` carries a referent by rule, the inter-sentence loss
   (weight .1) detaches both target and context although accessible-mind
   §2.6.4 requires live source ideas, and every policy, contrastive and
   trial knob is zero — so nothing pulls the encoder toward "same NP → same
   object" or "same VP → same encoding" (spec §1). **Decided (Alec,
   2026-09-26): ICA, not SIGReg**, is the model: the isotropic Gaussian is
   rotation-invariant and can only whiten; independence picks the axes. At
   the symbol level a frame already has the classical mixing form,
   `frame = A·a` with A's columns the codebook rows and `a` the sparse signed
   activations, so an individual is a column and identity across sentences
   is the fixedness of A over the LTM chain; verbs are the columns of a
   second matrix over the surprise residual `o − ê` in those coordinates,
   one-sparse per S (a learned codebook of change patterns). Gradient form
   only — maximum-likelihood / Infomax with a heavy-tailed prior,
   overcomplete so sparse coding, no fixed-point iteration (as 7.5) —
   over the LTM population, not the batch of two, on the content band only.
   Priors on the number of sources come from the grammar (roles per S)
   bounded by STM capacity; the total is nonparametric: mint on unexplained
   surprise under the 9b recurrence gate, prune by item 5's value with an
   automatic-relevance scale. Identity binding becomes a candidate in the
   7.5 softmax (bind to a column in the recency buffer or cued frames, or
   mint), credited by surprise through the seal; the inter-frame predictor's
   source ideas go live, target detached. ICA binds across frames and within
   an object; it does **not** bind roles within a sentence — the three slots
   do, and a test asserts the superposition catastrophe rather than hiding
   it. Exit: the ten §5 mechanism tests unconditional; the learning gates
   (held-out anaphora against the rule baseline, verb reuse across streams,
   prediction against the detached-target control, shuffled-order and
   renamed-vocabulary controls, seeds 0/1/2) follow item 9's
   million-sentence prerequisite and are recorded, never tuned.
   *Compatibility:* 7.5's exploit/explore derivations and tie rule are
   unchanged in form; item 5 consumes the relevance scale; item 6's recovery
   measurements are unaffected.
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
   *Compatibility:* this is the serial grammar's `lift`/`lower` with their
   tied chart inverses, unchanged by 11c and by item 10's drop. Recovery is
   measured from forward artifacts — the stored derivation and activations —
   never a saved input trace (Alec, 2026-09-25); and the 11c rule for
   descent applies: reverse sigma is a choice of case, reverse pi is
   attribution against the field, so clean-up decoding through the forward
   kernel is the intended form of that choice.
- **5. Forgetting** ([spec](doc/specs/2026-09-16-forgetting.md)), after item 7
   (needs `refs` and every S writing a row). Document-boundary pass from the
   high-water to the low-water mark deleting the lowest-value unprotected
   rows; value = `|trust|` + utility (1 − deducibility) + luminosity
   contribution; cascade, reference remap, dependents rebuilt, the human
   profile's age term; and **detail before rows** (§4a): wording, then
   subordinate rows, then the row, gradually, coarsening the referring row
   instead of cascading where the operand's point survives. Exit: the fourteen
   §7 tests, the accessible-mind spec's test 31 (retention by surprise), the
   §6 elements in schema/`model.xml`/Params.md, the §8 docs, and the reviewed 9b
   reconstruction measurements unchanged unless explicitly re-baselined.
   *Compatibility:* rows carry the pair `(c⁺, c⁻)` (§1.1); `|trust|` in
   the value is `|c⁺ − c⁻|` (11c entry 2's collapse) and the *both* corner
   `min(c⁺, c⁻)` counts as dissonance in the luminosity term, so a
   heterogeneous row is protected, not discarded. Open for Alec: forgetting
   of the concept inventory itself — order-0 definitions, alternatives and
   feature groups — is not in the spec; discovered rows are never recycled
   (item 11), so their retirement needs a rule here or in FutureWork.
- **4. Run harness and resume test.** *Pulled forward for the conference
   (Alec, 2026-09-27):* the word-level predictor as the NanoChat gate's
   evaluator — the model's own expectation of the next word scored on the
   frozen item manifest (top-1, reciprocal rank, shuffled-prefix control),
   living in `eval_nanochat_grammar.py` rather than the training loop; it
   is the first level of item 6.8's expectation at every bracket. Also logged
   per bracket level once 6.8 lands: the both-rate and the
   categorical-discrimination index. One logger per interval: reconstruction
   loss; expectation discrepancy; LTM occupancy and forgetting passes;
   [categorical discrimination](bin/CategoricalDiscrimination.py) from the
   fixed four XOR and 68 FineWeb probes, reporting CP and within/between
   distances without pass thresholds. Reuse captured readings and budget their
   collection cost; the three-seed experiment is archived in FutureWork.
   Also log rows deleted per origin, value cut-off; luminosity of provisioned truths; the
   per-shared-operator gradient cosine and norm ratio — reconstruction against
   expectation, and against output where answers are supplied — with the
   operators in persistent opposition named; the held-out two-truths §7
   test-12 probe and a fixed reconstruction sample; and from 11–11c: order-0
   inventory rows used against `nVectors`, provisional-pool occupancy and
   exhaustion warnings, raises per order with their run counts and stalled
   patience, refinement requests, lexical references left unknown per order.
   And a resume test proving
   a mid-epoch checkpoint restores cursor, stream count, `refs` / surprise
   columns and forgetting counters with the next batch byte-identical —
   including the concept sidecar (feature groups, located brackets, sparse
   context, refinement buffers), the radix part groups and pending groups,
   the word-form index and the understanding's conceptual field. Exit:
   one command on a small config; the test in the suite.
- **3. Corpus at target size.** Raise `maxDocs`, exercise multi-shard if needed,
   measure sentence-list/address-table memory and loader time, confirm
   `resume_skip` with the run's stream count. Measure admitted PartSpace rows
   on the target corpus and raise the current 32,768-row reserve before the
   million-sentence run. The required capacity depends on admitted parts,
   not one row per sentence; exhaustion now stops training. Record the chosen
   physical `nVectors`, occupancy and dictionary/optimizer memory cost.
   Target about 200,000 English word forms plus one associated object per
   word, with further room for non-verbal concepts: roughly one million
   physical ConceptualSpace rows (about 4.3 GB at width 1032 in float32,
   without Adam moments on the rotation-owned dictionary). Reserve a few
   hundred thousand PartSpace rows and measure WholeSpace admission for the
   same vocabulary and non-verbal coverage. The current CS 65,536 / PS 32,768
   settings are small-run values, not production capacity estimates.
   Set `SymbolSpace.ltmCapacity` for the intended million-row run; its default
   1,024 also sizes the shared `.when` ladder. Record retained-row memory, the
   actual subjective-clock range needed by the run and the ladder's resulting
   periods, so a long training history does not silently reuse temporal phases.
   Exit: the load and capacity measurement recorded, and
   `maxDocs`/`shardDir`/stream count/PS-CS-WS `nVectors`/`ltmCapacity` in the run config.
- **2. Housekeeping.** Restore safe, faster bounded-test batching: the default
   256-case/16-file and 32-case/4-file workers exceeded the 8 GiB cap, and
   eight-case/one-file batches pass the same selection; do not raise caps or
   omit cases; reconsider [fixture reuse](doc/plans/2026-09-17-bounded-fixture-reuse.md)
   if useful ([evidence](doc/Testing.md#forward-owned-lexical-forms-september-19)).
   Delete the radix-backed non-word-major meronomy path (no legacy paths), and
   remove the stale `dispatch_per_row_reset` note in
   [the fold-ladder plan](doc/plans/2026-09-10-meronomy-fold-ladder.md#open-defects-found-on-the-way)
   (`taxonomy_parent_map` is now initialised).
   From the 11–11c refactor, verify legacy before deleting (no-legacy rule;
   reasoning methods need Alec's review): `overlap_where_tiling` and
   `where_tiling_for_pass`, `intent_boosts`, WholeSpace `part_chain`,
   `_automatic_synthesize_higher_order` against the pool's discovery path,
   and the META `insert_meta` / `taxonomy_parent` bindings once item 7's
   concept-level taxonomy owns them. Move the 2.9 MB
   `evaluation-source.tar.gz` out of the repository (the receipt keeps its
   hash) unless Alec wants it in-tree.
- **1. Compiler work.** The throughput levers for the serial loop — batch
   across sentences, the known-word lookup concession under the serial
   flag, subsampled reconstruction, closing the host islands and moving
   the sparse stores to device CSR, single passes — are listed in
   [FutureWork](doc/FutureWork.md#throughput-levers-for-the-serial-loop-item-1-candidates)
   and belong here (Alec, 2026-09-25). Forward chooser split/lift once per slot; backward's
   launch-bound kernel count per brick; B24 brick +25% against pre-ladder.
   Exit: sentences/s and peak memory at the run's batch and brick size against
   the July baseline, re-taken on `99207a3` since the tower fold ladders are
   gone. New eager hot spots from 11–11c to measure and vectorise: the
   per-row loop in `cs_read_memberships`, the per-row composition in
   `_compose_order0`, the edge loops in `_reverse_field`, the per-reading
   descent in `refine_over_collected`, candidate growth in
   `_prepare_part_learning`, and the per-column loops in
   `promotion_observe`.
   The explicit slow reconstruction-cache probe fails its nonzero
   compose-gradient assertion on the current baseline; the cache claim remains
   unverified by that probe ([audit](doc/benchmarks/2026-09-25-item9-parity/README.md#validation)).
- **0. The full training session**: the long FineWeb run, only with items 11–2
   done and item 1 measured. Expectation on in `model.xml`; the
   `BasicModel.xml` flip follows the plan's §10 gates. Stop on rising
   expectation discrepancy, a reconstruction regression against the `99207a3`
   baseline, a forgetting pass deleting protected rows, a provisional-pool
   exhaustion warning, or order-0 inventory approaching `nVectors`.

Everything that is decided in direction but not on this path is in
[FutureWork](doc/FutureWork.md).

### Done (newest first)

- `6906727` Item 7.5: one-operation exploit/explore derivations trained at each sentence seal, reduction pressure/deadlines and seal-gradient reporting; accepted with the unchanged depth-three campaign red ([receipt](doc/benchmarks/2026-09-27-item7-5-landing/README.md)).
- `8bc710a` Item 9b: shared fields, parallel-first context, association-first interpretation, fixed capacities and corrected occurrence/time objectives; Claude accepted the source-matched 4,948-case sweep ([receipt](doc/benchmarks/2026-09-26-item9b-occurrence-fix/README.md)).
- `8bc710a` Item 9 evaluation machinery: verified FineWeb exposure and maturity-gated wording/prediction/predictive-thought checks; empirical acceptance remains open above ([receipt](doc/benchmarks/2026-09-26-item9-followup/README.md)).

- `0e70001` Item 9 parity landing: sentence-owned reconstruction candidates, final saved roots, first-sight bank checks without scalar host reads, and the historical-answer warning ([receipt](doc/benchmarks/2026-09-25-item9-bank-sync/README.md)); expectation-learning gates remain open.

- `99207a3` Item 10 decided: dropped; normalized means deleted, native XOR and forward-evidence reconstruction pass unseeded, with mode exclusions asserted ([receipt](doc/benchmarks/2026-09-25-item10-forward/README.md), [evaluation](doc/benchmarks/2026-09-24-item10/README.md)).
- `99207a3` Item 11c residue: accepted three-update refine-before-raise patience, reset by strict improvement and cleared by pure/unknown reads ([receipt](doc/benchmarks/2026-09-25-item10-forward/README.md)).
- `d5f2e1a` Item 11c: native perception, located XOR, per-pole witnessing, nonzero focus and grammatical reference across orders ([receipt](doc/benchmarks/2026-09-24-item11c/README.md)).
- `15c9bde` Item 11b extent correction: containment at the subject extent, observed raw spans and the restored positive-only `MM_sparse_concept` smoke ([receipt](doc/benchmarks/2026-09-24-item11b-extent/README.md)).
- `aa235d2` Item 11b accepted-review corrections: recurrent parts, containment over canonical ids, and order-0 fields bound per turn with persistent concept identity ([receipt](doc/benchmarks/2026-09-24-item11b-corrections/README.md)).
- `db73581` Item 11b: native membership definitions, located fused parts and pervading wholes, with alternatives preserved through refinement; normalization residue closed ([review receipt](doc/benchmarks/2026-09-24-item11b-review/README.md), [design](doc/Architecture.md#item-11b-membership-read-and-extent-truth-corners-september-23)).

- `4b097dc` Item 11a learns WholeSpace properties over byte primitives, retains grounded extent evidence and passes six unseeded native XOR runs ([receipt](doc/Testing.md#item-11a-primitive-properties-and-grounded-extents-2026-09-23)).
- `e917d9c` Item 11 review corrections carry paired evidence through scoped dual folds, symbols, thought and checkpoints; calibrated background rejection and the full receipt pass, learned XOR remains null for item 11a ([receipt](doc/Testing.md#item-11-review-corrections-paired-evidence-september-23)).
- `eb45ae4` Item 11 lands the union default, optional conjunction pass and context-backed provisional rows; unseeded XOR learning remains null ([receipt](doc/Testing.md#item-11-concept-parts-and-provisional-rows-september-22)).
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
