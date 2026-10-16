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

**Current:** Item 6 **part 1** is the reviewed mechanism landing after the
October 10 repair pass ([receipt](doc/benchmarks/2026-10-10-item6/part1/README.md)).
Item 6 **part 2** is in measurement and review: identity learned through the same unmixing,
with teaching corpus and measurements before rule retirement. Stored-idea
recovery beyond two words, order-lesson CE, MM grammar recovery and compiled
retention remain open below. Accepted 6.1, 6.2 and 6.5 mechanism landings make
no learning claim; their checkpoint gates remain at item 0.

**Current sequence (6.1 closed October 10; no conference freeze):**
6 → 5.5 → 5 → 4.5 → 4 → 3 → 2 → 1 → 0.
The word-level evaluator built in 6.8 waits for item 4's trained checkpoint;
acceptance does not authorize fresh-model bulk scoring. Items 9 and 8 are implemented and
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

**Publish rule (Alec, October 11):** repair → one regression per defect →
scoped measurement → candidate commit on a branch → **Claude's code review**
→ resolve review → one full sweep at landing → BasicModel push → WikiOracle
bump/push, with the co-author trailer. Candidate commits are authorized before
review; their commit hashes are the archive. No tarballs, manifests or
integrity checks. Receipts contain gate numbers and one table per question;
diagnostics stay in files.
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
   old context-free and assertion-thought scores are invalid controls.
   **6.5's learning gates** — held-out anaphora against `206a0146`, verb
   reuse, prediction control, shuffled order, renamed vocabulary and
   determiner control — remain pending the million-sentence checkpoint,
   across **seeds 0/1/2**, and are **not claimed** by its mechanism acceptance
   ([receipt](doc/benchmarks/2026-10-07-item6-5/README.md#learning-limits)).
   **6.2's learning measurement** (MM_math_chain: held-out binding 9 of 10,
   the chain bar, the expectation-only and zero-budget controls) also waits
   here, at item 0's checkpoint under [thinking spec §14.10–§14.13](doc/specs/2026-10-07-thinking.md):
   one-step problems first, worked steps as intermediate answers, binding a
   found candidate, and budget in steps with each query's record charge bounded
   separately ([protocol corrections](doc/benchmarks/2026-10-08-math-chain-repair-2/protocol-corrections-14-13.json)).
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
   The MM_xor bar stays as written: its bests sit within 10% of the bar in
   every campaign (plan §19), and the §20 rule — a miss whose path differs
   from the landing only in global-RNG consumption, shown by bisection, is
   not a regression of the round — is what handles that margin.
Item 7.5's non-blocking compose follow-ups (the training temperature and
parsimony term for a later spec) are in
[FutureWork](doc/FutureWork.md#compose-follow-ups-deferred-from-item-75-2026-09-27);
its 2026-09-29 costing finding is closed by 6.9's step 5, and the depth-three
campaign stays a recorded red in the
[accepted landing](doc/benchmarks/2026-09-27-item7-5-landing/README.md), not work.
Remaining operators work stays with its assigned items: surface, tense,
morphology, aspect, null and situation codes with 5.5; forgetting's truth-kind
bias with 5; trained-checkpoint evaluation with 4; the concept-face negative
image's learning measurement with 9 (its mechanism landed in round 2,
`3de37eef5`, and the complement it reads is populated since final-b); the
open-read/teacher-loss host island with 1; expectation across the
architecture with **4.5**; the bank's collision and merge rules in
[FutureWork](doc/FutureWork.md#one-bank-the-collision-rule-for-property-rows-and-a-merge-rule-decided-2026-10-06-unscheduled). The
[FutureWork list](doc/FutureWork.md#carried-from-the-68-1315-rounds-2026-10-04)
retains the dynamic stop, the remaining operator names and other deferred work. The
[6.8 plan](doc/plans/2026-09-27-item-6-8-one-attention.md) retains the reading
residue around forced `interpret` and digit boundaries. Historical operators
measurements and decisions are in the accepted receipt and its predecessors.
Deferred from 6.5 (non-blocking): the detached role view's retirement is in
item 2; learned-column pruning is in item 5.

Alongside item 6, retain these open findings, with the historical and current
outcomes separated in the [part-1 receipt](doc/benchmarks/2026-10-10-item6/part1/README.md):

- Stored-idea recovery beyond two words through trained A/B unmixing belongs
  to part 2. Initial-source list free recovery is 0/4 throughout; enabled
  two/four-word derivation recovery is 2/4 and 0/4. The order lesson's CE
  stays at .3465735912 / .7945134640 through 64 presentations, with zero
  scorer gradient. These inverse and credit findings remain active work.
  The valid part-2 two-word checkpoint reads its stored derivations at 4/4,
  but free readback is 0/4. A [read-only probe](doc/benchmarks/2026-10-10-item6/identity-first/validation/attention-readout-audit/result.json)
  finds every attention tanh unit saturated at ±1 and candidate logits tied;
  that study left saturation unchanged. The [R/S/T candidate](doc/benchmarks/2026-10-10-item6/identity-rst/README.md)
  bounds the shared context and removes saturation in the retained 64-step
  continuation. Fresh two-/four-word CE improves to .0320/.0166 but remains
  above .01; free recovery remains 0/4. New adverse result: the later relation
  trace falls from 4/4 to 2/4, swapping `than` and the final operand.
  Pair teaching also skips compound operands without identities in its word
  shortlist; the successful lexical-pair probe does not establish sufficient
  teaching credit for free reconstruction of longer constituents.
- MM grammar recovery: the 1,024-row LTM exhausts in epoch 2 before evaluation;
  there is no new recovery rate. Item 5 may not discard derivations yet.
- Compiled retention: the 64/312-element mismatch (max .2556) has no identified
  cause. Later passing runs do not close it. Also retain accepted 6.1's packed
  LTM retention and pending-premise order-`False` findings.
- c's retained supplied answer does not match `five`, and the eight-sentence
  fork certificate chooses only round 12. Keep both assertions unchanged.
  Question-sensitive answering also remains unproven in the current staged
  warm-up: noun–verb answers are 0/4 and relations 1/4; the earlier 4/4
  identity and noun-phrase scores match a last-word baseline
  ([receipt](doc/benchmarks/2026-10-10-item6/identity-first/README.md)).
- The [October 14 sweep](doc/benchmarks/2026-10-14-item6-gate-off/README.md)
  has three failures new relative to accepted R2's passing sweep. The
  supplied-answer failure and eight-sentence fork failure (now only round
  24) recur from 6.1. `test_forced_ordinary_bound_declaratives_open_no_episode`
  is new relative to 6.1 too: no open slots or episodes appear, but every
  recorded compose departure is 6, failing the diversity assertion.
  The [October 15 source-boundary replay](doc/benchmarks/2026-10-15-item6-corrected-warmup/README.md)
  finds no exemption effect: the shared fixture has an empty meaning slice,
  so both implementations charge zero and produce identical departures and
  keep decisions under matched initialization. The two diversity assertions
  demand more than the independent reservoir/walk sampler guarantees; on the
  recorded menus their failure probabilities are 2.15% and 26.21% respectively.
  Those are conditional probabilities, not full-test failure rates. c's
  immediate failure is different: its filled reference points to an order-one
  child associated with `the`, with no lexical path to `five`. All assertions
  remain unchanged; no repair or learning claim is made.
- b's initial binding-distribution bound remains flaky after construction-based
  coverage of 32 documents, histories 1–4 and four names. Development maximum
  1.153045 and the [first part-2 sweep's](doc/benchmarks/2026-10-10-item6/identity-first/README.md)
  .848586–1.179175 range violate the unchanged .9–1.1 band; the latter is a new
  failing node relative to accepted 6.1. No passing initialization closes it.
- The [R2 sweep](doc/benchmarks/2026-10-10-item6/identity-r2/full/summary.json)
  adds `test_review17_measurement_wiring_on_one_ordinary_batch` as a new
  failure relative to both 6.1 and R/S/T: its observer has no epoch-1 chooser
  logit entry (`KeyError: '1'`). The accepted inverse-domain repair resolves
  the cause: the observer dropped detached training calls on narrowing-only
  departures. It now counts those separately from gradient calls; the native
  regression and unchanged ordinary-batch certificate pass.
- Per-position cleared-cache overlap remains 0.000 against .800. The
  [follow-up bisection](doc/benchmarks/2026-10-10-item6/followup/README.md)
  separates the initial self-pair obstruction (repaired in part 1) from the
  current reversed pair. The first part-2 candidate removes the exact-fit
  override: learned weights give 1.000 overlap, against 0.000 with the old
  override on those same weights; the cold probe remains 0.000. Keep the order lesson, content
  coverage, walk order stamps and family-metadata use under part 2 (review J–L).
- The [identity corpus preparation](doc/benchmarks/2026-10-10-item6/identity-corpus/README.md)
  exposes a native exploratory trace/admission mismatch: an admitted leaf is
  filtered from reconstruction, which then underflows a binary operation.
  The first part-2 candidate repairs the trial/commit masks and verifies the
  preserved passive-input trace; it landed as preparation in `af634ed35`. The corpus drops
  unverified object fronting and reports the introduction-role/position confound.
  This probe has not been bisected against 6.1; do not call it an H regression.
- The first part-2 warm-up stopped on a non-concept `sym` cue entering the
  taxonomy order reader (four-word presentation 8; all experiment branches
  unreached). The candidate restricts that reader to positive concept handles;
  the original failed attempt and corrected-source run stay in the
  [same receipt](doc/benchmarks/2026-10-10-item6/identity-first/README.md).
  The corrected run passes that point and retains a negative address tagged
  `sym`; its old-reader replay raises and the corrected boundary unfolds the
  payload. The producer's typed-reference provenance remains open.
  Native checkpoint loading also changed eight shared-codebook tensor aliases
  in the smoke probe; the experiment restores frozen tensors for matched starts,
  without claiming a production retention repair.
- Supplied-answer warm-up adds 16 lazy comparison-reader weights that native
  strict checkpoint preflight rejects before the child's load hook can restore
  them. The [first part-2 receipt](doc/benchmarks/2026-10-10-item6/identity-first/README.md)
  preserves all eight failed branch starts. Its measurement adapter restores
  the child first and verifies all 874 saved tensors; production loading remains
  unchanged. This newly observed retention finding has not been bisected against
  6.1. The retained branch continuation reused the warm-up without extra exposure.
- The initial native identity experiment used a word-chunk fixture with no
  meaning complement. Its four completed zero-identification results are
  invalid admission measurements; the unfinished identity branches were stopped.
  The [receipt](doc/benchmarks/2026-10-10-item6/identity-first/README.md) retains
  the zero-projection audit. A fresh run with 64 meaning pairs and a verified
  nonzero warm-up signal completes recovery at 0/240 on all four branches:
  no new columns or merges, and every row selects mint, so the on/off gate
  changes no eligibility. All 192 added observations have zero meaning-band
  content despite nonzero full vectors. Four frozen replays locate the loss:
  word meanings are nonzero, but selected `non` clears them and `lower`
  repeatedly keeps the newer operand. Resolve why compose keeps these lossy
  readings before claiming an ICA recovery or admission-gate lesson.
  The [R/S/T replay](doc/benchmarks/2026-10-10-item6/identity-rst/validation/recovery-summary.json)
  still has zero closed meaning and `lower`–`non`–`non` in all 48 held-out
  rows despite a content charge of about 1.0. Four sampled paired walks
  charge both erased readings equally; this retention outcome remains unmet.
- The [R/S/T matched rerun](doc/benchmarks/2026-10-10-item6/identity-rst/README.md)
  completes all eight branches with the same seed, budgets and revision-3
  corpus. Recovery remains 0/240; all identity branches remain 0/146 bindings
  and 0/20 row pairs. Each now resolves 150/690 mentions to written individuals,
  but no probe has a complete candidate set or a valid selected reference.
  The other 540 mentions have named missing reasons; grading has no sentinels.
  Balanced unadmitted leaves rise from two in the old retained audit to 43.
  The balanced gate contrast is no longer vacuous: off mints 58 columns versus
  on's 57, without accuracy gain. Recovery and biased training remain vacuous;
  the admission lesson fails. The [retained observer audit](doc/benchmarks/2026-10-10-item6/identity-rst/validation/observer-summary.json)
  accounts for the old 690 unresolved mentions separately. Solve the missing
  individuals/requests before interpreting binding and recency controls as
  learned choices. Reviewed and accepted for landing by Alec (plan §8.4),
  with no learning claim; no rule is retired and part 2 remains open.
  [R2](doc/benchmarks/2026-10-10-item6/identity-r2/README.md) adds the forward
  discarded-meaning charge after the forced-reading diagnostic, with the
  scoped four-clause and two-branch recovery measurement (factorial/confounded,
  gate on), reusing the accepted R/S/T warm-up. Identity branches wait for
  nonzero closed meaning. Candidate `0cd221d74` on
  `codex/item6-r2-forward-discard` passes the 4/4 local keep regression, but
  both completed recovery branches have 0/48 nonzero closed meanings
  and identify 0/240 atoms. Factorial mints zero columns; confounded mints four;
  neither forms a measured kind/property merged pair. Confounded selects mint
  on 136/192 observed commits versus factorial's 192/192; the off contrast is
  unmeasured. All 96 kept readings still use `non`. No learning gain or rule
  retirement is claimed. This candidate was rejected in plan §9. The
  [inverse-domain repair](doc/benchmarks/2026-10-10-item6/identity-r2-inverse-domain/README.md)
  restores the accepted zero-cost certificates and scopes the forward charge
  to the local field inverse's undefined coordinates. The observer's missing
  epoch is repaired by counting detached training calls separately from
  gradient calls. Accepted `348fb86c7` + `90ef40a42` landed on main, pushed
  October 14 with WikiOracle bump `066fb5d`. It retains nonzero closed meaning on
  44/48 factorial and 4/48 confounded held-out rows (previously 0/48 each),
  but identification remains 0/240 on both. No new columns or merged pairs;
  all kept programs use `verb` and omit `non`. The retained four cat clauses
  still close at zero. Both branches select mint on 37/192 observed commits;
  the off contrast and identity lessons remain unmeasured. The one final
  sweep completes all 5,790 cases: 5,502 pass, 287 skip, one XFAIL, no failures.
  No new ordinary failures versus accepted 6.1 or rejected R2; cleared-cache
  overlap remains adverse to 6.1's XPASS. No rule is retired.
  The [scoped gate-off measurement](doc/benchmarks/2026-10-14-item6-gate-off/README.md)
  reuses the same R/S/T warm-up, seed and revision-3 corpus. Factorial off
  mints two columns (60 total); confounded off mints four (62 total), versus
  zero new columns on. Both still identify 0/240 atoms and 0/24 for each
  kind, with no measured merged pairs. Programs and norms match gate-on:
  `verb` only, 44/48 and 4/48 nonzero closed meanings. The admission contrast
  is no longer vacuous, but its retirement lesson fails; the gate stays.
  The frozen singleton audit finds 16/24 zero stored meanings despite
  nonzero word vectors, and 0/58 warm-up columns aligned at |cos| ≥ .9.
  The confounded trace loses nonzero word meaning at each `verb` fold from
  a zero `the` accumulator. The separately applied search-kind exemption
  makes exact field OR pay zero forward discard while keeping κ = 0 in
  the field reader; it does not change the preceding learning comparison.
  Candidate `c647e1506` completes the one full sweep: 5,503 pass, 287 skip,
  one XFAIL and three failures, with the new/recurring failures named above.
  Accepted zero-cost certificates, the observer and the 4/4 retained-clause
  keep regression pass. The parts-of-speech pass in plan §6 waits for a
  reading with NP structure; no rule is retired.
  The [fresh corrected-objective warm-up](doc/benchmarks/2026-10-15-item6-corrected-warmup/README.md)
  completes the same stages 1–4 budgets with seed 20261010: **8/24** singleton
  rows carry their atoms, below the required 20; 16 remain zero despite word
  meaning norms of .2. It mints **zero A columns**, confirmed in the saved
  checkpoint. The native `black` trace loses .2 → 0 at `non` before closing.
  All 76 singleton presentations are observation-only, with one greedy trial
  per batch, so the corrected keep charge never compares an alternative on
  those writes. Both order CE gates still fail (.032051/.016668 versus .01),
  and all four final stage-4 supplied-answer readbacks are 0/4. Under the
  identity-per-order amendment, balanced order-1 identity is not run because
  the singleton gate fails; kind recovery is deferred to order 2. No
  production change, new sweep, learning claim or rule retirement in this pass.
  The [positional-operands and paired-warm-up candidate](doc/benchmarks/2026-10-15-item6-orders-paired/README.md)
  (`6a14ba151` measured source) removes the name-to-order table and carries
  actual operand stamps through `argin1`, `argin2`, `argout1`; the order-3
  regression passes and operation-owned mint/bind/kind behavior stays.
  Its paired stages 1–2 retain **20/24** singleton atoms, so the balanced
  gate-on identity branch runs: 136 training and 166 held-out documents,
  **zero columns minted, 0/146 bindings and 0/20 row pairs resolved**.
  All 556 held-out clauses have zero closed meaning under `equal`-only
  programs; all 690 annotated references report `no_individual_request`.
  None reports an unadmitted leaf in this run; the earlier 43-leaf artifact
  is not retested. The bounded trained-artifact certificate covers **80/80**
  eligible rounds; the two batch-diversity tests leave the default suite.
  The one full sweep has ten failures, with subsequent targeted repairs
  leaving c's carried `five` structure failure; the paired zero-cost
  counterexample reproduces on accepted main. No clean final sweep or
  identity-learning claim is made, and no identity rule is retired.

The cleared-cache empty inverse was **new with the item-6 candidate**, not an
unchanged 6.1 failure. Part 1 repairs the candidate-search self-pair obstruction
and checks the unchanged drivability assertion. It also repairs the native
retrieval `meta`-reference error, scopes stored-read operator activations,
charges dropped support in the keep cost, and isolates captured walk inputs.
Their original failing evidence remains in the receipts.

- **6. Part 2 — stored-idea unmixing and identity from data.** Part 1 is under
   Done. Follow Alec's [full amendment, plan §2](doc/plans/2026-10-10-item-6-stored-idea-generativity.md#2-identity-from-data-amendment-alec-2026-10-10):
   one mechanism explains noun phrases and identifies occurrences, with
   dictionary columns reused for binding and recurring unexplained surprise
   for minting. Candidates are the eight-space plus cued frames; pronouns get
   prediction credit from what follows and determiners are learned cues.
   **Build the corpus and measurement before changing mechanisms**: factorial
   properties/objects/verbs, support 1→2→3, counterbalanced pronouns,
   imperfect determiner cues and conflicts, later distinguishing properties
   for same-kind pairs, and varying document context. Measure binding by
   candidate count, recency and role; one row versus two; determiner conflicts;
   neither-candidate and reversed-recency controls. Claude's
   [four-question numpy toy](doc/benchmarks/2026-10-10-identity-ica-toy/README.md)
   precedes retirement. Remove each declared mode, separate pronoun rule,
   positional bind path, mode/cosine resolver and mode-gated admission only
   when its lesson passes without it; otherwise retain it and report the failure.
   No parallel learned and declared replacement; no further declared modes.
   [Orders.md](doc/Orders.md) and plan §12 refine the measurement order:
   A's columns are order-1 individuals, and recurrence stays as their admission
   evidence. The paired stages 1–2 pass carries 20/24 singleton atoms and
   completes balanced gate-on identity, with zero minted columns and no
   resolved bindings or row pairs (candidate receipt above). The identity
   lesson remains open; it has not reached usable NP readings.
   Factorial/confounded kind recovery is deferred to order 2; whether its pi
   uses a second dictionary or taxonomy computation remains undecided.
   Word classes collect words by operand participation, never phrase heads;
   no new order ceiling or lexical count/mass declaration is introduced.
   The [corpus and measurement preparation](doc/benchmarks/2026-10-10-item6/identity-corpus/README.md)
   landed as `06f6e2b88`: 740 documents, separate grading labels,
   native reference observation and both control datasets. Its native outcomes
   and limitations are recorded separately from grading sanity controls;
   no identity rule is retired and no learned-control result is inferred from
   an unresolved reference or a failed run.
   The [first matched native experiment](doc/benchmarks/2026-10-10-item6/identity-first/README.md)
   is reviewed and accepted as preparation: all eight branches finish, but the recovery,
   binding and row-count lessons do not pass. The open findings above retain
   the failed prerequisites and distinguish column admission from identification.
   Measure forced raw/bounded and free stored recovery by depth and chain
   length on fixed forward artifacts, now with trained A/B columns; report
   stored dictionary separately from sentence-time live wholes. Which family
   comes first is taught by the corpus, never a family-to-side map. Carry the
   four inverse/CE/MM/compiled findings above. One landing, one measurement.
- **5.5. Where and when a sentence occurred; tense, aspect, the preposition
   and surface form** ([spec](doc/specs/2026-09-30-occurrence-tense-aspect.md)),
   after item 6 and needing item 6.5's verb columns; one check-in (Alec,
   2026-09-30). **Relative `.when` and address upsert moved to operators
   landing (accepted October 7; plan §45)**:
   source-relative `[i, i+1]`, stable document/content address keys,
   re-witnessing and fail-loud capacity. See its
   [receipt](doc/benchmarks/2026-10-07-operators-final-b/).
   The 10-03 amendment keeps `.when` in percepts and concepts; the absolute
   model clock is written to the `timestamp` column by 4a-0, which spec §7
   and test 6 retire — this item decides its fate with the recency use
   (Alec's 09-30 age ruling: the document index suffices). Remaining here:
   documents as situation codes in content; learned before/after; tense,
   aspect, prepositions and surface transformations. Interleaving is
   delivered by 6.8's open read (spec §0; `serial`/`modeSchedule` are
   rejected at load), so spec test 8 is met; tense and aspect are
   prepositions of the verb phrase with none written, the catalogue's
   compound selecting among the verb phrase's phases; the preposition is
   one operation with two attachments; markers are leaves, minted on
   recurrence, and `surface` is one operator with four suboperations,
   trained by reconstruction under parsimony. Removes the old tense, aspect
   and morphology code and its table of English forms. Dimensional
   `lift`/`lower` is not here (FutureWork; Architecture's "remains with
   5.5" is corrected in item 2). Exit: the remaining mechanism tests of
   spec §8; learning measurements after 6.5, recorded and not tuned.
   Replace the legacy `get_stm_chain` recency across batch rows and test
   lookup by the row's occurrence address (two truths §21). Open for Alec:
   spec §10.
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
   *Compatibility (corrected, Alec, 2026-09-28):* rows carry the evidence
   pair `(c⁺, c⁻)` (§1.1) **and, separately, a univalent trust scalar**.
   `|trust|` in the value is that scalar; it is **not** the collapse
   `|c⁺ − c⁻|`, which this line previously said and which was applying the
   evidence algebra to trust. The collapse `c⁺ − c⁻` and the *both* mass
   `min(c⁺, c⁻)` are the row's **evidence**, and the latter is the dissonance
   in the luminosity term, so a heterogeneous row is protected, not
   discarded. Open for Alec: forgetting
   of the concept inventory itself — order-0 definitions, alternatives and
   feature groups — is not in the spec; discovered rows are never recycled
   (item 11), so their retirement needs a rule here or in FutureWork; the
   same rule is the merge half of the bank's collision rule
   ([FutureWork](doc/FutureWork.md#one-bank-the-collision-rule-for-property-rows-and-a-merge-rule-decided-2026-10-06-unscheduled)).
   *Also here (deferred from 6.5):* pruning of learned identity/change
   columns by their relevance scales (ICA spec §10), in the exit criteria.
   *Age (Alec, 2026-09-30):* with no clock across documents (item 5.5), the
   age term "can probably just be an increasing document index: more
   important within a document will be its salience"
   ([5.5 spec §9](doc/specs/2026-09-30-occurrence-tense-aspect.md#9-what-it-touches)).
- **4.5. Expectation across the architecture** (Alec, 2026-10-08: "needs to
   go in sooner rather than later"; a spec of its own). **Spec (draft,
   2026-10-08 rev 3; two questions open in §8):**
   [expectation at every level](doc/specs/2026-10-08-expectation-at-every-level.md)
   — Alec, 2026-10-08: a predictor per conceptual level used as a negation
   that reduces signal to surprise (percepts are attended, not expected);
   exploit per level, exploration at least partly free of expectation; one
   departure, read at the end of the sentence, judged by `R + A`; a graded
   set in 4.5 (the Viterbi-like principle: verified in a toy, graded 0.96 vs
   root-only 0.04) and the age-appropriate corpus in item 3. The scheme confirmed on
   2026-10-06 ([operators plan §10.3, §12.2, §13](doc/plans/2026-10-05-operators-update.md#13-the-scheme-as-confirmed-alec-2026-10-06);
   [GradientFlow's target scheme](doc/GradientFlow.md#the-target-scheme-october-6-decided-scheduled-by-rounds)),
   never built: the row-level prediction inverted through the committed
   operations gives an expected operand at every round; the actual against
   it is the layer's surprise; the maps take exact targets by inversion
   (target propagation — no gradient through the choice, no
   straight-through, no mixed forward); the word-level expectation follows
   from the row-level predictor by inversion rather than by its own
   predictor; expectation's sources go live (the 2026-09-20 detachment
   existed against collapse, which an invertible path cannot do);
   reconstruction retires to an audit where the path is exactly invertible
   and stays a loss where it is not. The record until the spec exists is
   [FutureWork](doc/FutureWork.md#expectation-by-inversion-and-the-reconstruction-audit-decided-2026-10-06-unscheduled).
   Placed after 5 because the model has run without it so far; the spec
   says what the content width (item 3) must be for the audit to read zero,
   and whether that derivation moves here. Exit: the spec's gates.
- **4. Run harness and resume test.** *Once pulled forward for the
   conference (Alec, 2026-09-27; no freeze since 2026-10-03, so it is built
   in 6.8-1 and evaluated here):* the word-level predictor as the NanoChat gate's
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
   Also log rows deleted per origin, value cut-off; luminosity of provisioned truths;
   the held-out two-truths §7
   test-12 probe and a fixed reconstruction sample; and from 11–11c: order-0
   inventory rows used against `nVectors`, provisional-pool occupancy and
   exhaustion warnings, raises per order with their run counts and stalled
   patience, refinement requests, lexical references left unknown per order.
   (The per-shared-operator gradient cosine and "persistent opposition"
   logging is withdrawn: each parameter has one objective owner since 6.9,
   so there is no shared operator to measure — [GradientFlow](doc/GradientFlow.md);
   the dissonance diagnostic of the operators plan §10.3 guards E against O
   on the maps instead.)
   And a resume test proving
   a mid-epoch checkpoint restores cursor, stream count, `refs` / surprise
   columns and forgetting counters with the next batch byte-identical —
   including the concept sidecar (feature groups, located brackets, sparse
   context, refinement buffers), the radix part groups and pending groups,
   the word-form index and the understanding's conceptual field, and the
   state added since: the address index and witness counts (4a-0), the
   identity atoms and mints (3a), 6.5's identity/change columns and
   relevance scales, the comparison reader's copy of the answer path (2e)
   and the pending relational rows (6.2). Exit:
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
   *The content width* (operators plan §10.3, §11.6; never derived): exact
   clean-up of `k` bound items against `N` bank entries needs
   `D ≈ c·k·log N`; derive it for the run's vocabulary and run the R≡0
   identity audit (round 3a's) at that width on the run's config. Its
   reading decides item 0's condition below
   ([FutureWork](doc/FutureWork.md#expectation-by-inversion-and-the-reconstruction-audit-decided-2026-10-06-unscheduled)).
   Exit: the load and capacity measurement recorded, the width and the
   audit's reading, and
   `maxDocs`/`shardDir`/stream count/PS-CS-WS `nVectors`/`ltmCapacity` in the run config.
   *Added (Alec, 2026-10-08, from item 4.5):* the age-appropriate corpus
   ([gradual training](doc/FutureWork.md#gradual-training-an-age-ordered-corpus-read-in-stages-proposed-2026-09-29))
   is assembled here so item 0 reads it in order; the
   [Viterbi toy](doc/benchmarks/2026-10-08-viterbi-departure-toy/README.md)
   shows the order, not only the content, is what lets one departure escape
   multi-step minima ([4.5 spec §5.3–§5.4](doc/specs/2026-10-08-expectation-at-every-level.md)).
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
   reasoning methods need Alec's review): `intent_boosts`, WholeSpace `part_chain`,
   `_automatic_synthesize_higher_order` against the pool's discovery path,
   and the META `insert_meta` / `taxonomy_parent` bindings once item 7's
   concept-level taxonomy owns them (`overlap_where_tiling` and
   `where_tiling_for_pass` are already gone). Move the 2.9 MB
   `evaluation-source.tar.gz` out of the repository (the receipt keeps its
   hash) unless Alec wants it in-tree.
   *Added 2026-10-08 (the coverage audit):* delete the legacy typed STM
   `_buffer` that only grows, and the detached role view carried beside the
   native occurrence anchor (ICA spec §10); rename the native context-read
   `exploration_trial` flag; remove the one-release rename aliases that
   raise with the new name (`Queries.py` `_operation_form`'s retired map,
   `Language.py` `_check_operator_availability`) after the one-release
   compatibility period following 6.2's mechanism landing. Documents that still describe retired or superseded things,
   to correct in one pass: [Params](doc/Params.md) lists `thinkingBudget`
   as live and `selectedThoughtPolicyWeight` as a REINFORCE term (both
   rejected at load; `attentionBudget` is the one budget) and
   [QueryWork](doc/QueryWork.md) still meters `selectedThoughtBudget`;
   `complete.grammar`, `ladder.grammar` and the tied-reconstruction fixture
   still comment `chunk`/`lookup`/`quantize`/`arma`/`what`;
   [GradientFlow](doc/GradientFlow.md) still heads rounds 2d, 2e, 3a and
   the final update "review/uncommitted candidate … no commit or push"
   (all landed at `3de37eef5`, `cce3a4f7b`, `f4a68404e`), gives the serial
   leaf as "meaning poles exchanged for negative activation" (superseded by
   the two lanes, plan §42) and the round-3 scheme as the
   "narrowing-weighted ceiling" (superseded by the adjacent-word wholes,
   plan §40); [Architecture](doc/Architecture.md) still offers "each word's
   two symbols … either meaning-pole orientation" at pair search, "has-a …
   narrows (round 3)" in the scheme, and "dimensional lift remains with
   5.5" (FutureWork has it); the [catalogue](doc/specs/2026-09-29-operator-catalogue.md)
   §12 headings still say "(uncommitted)" and its table lists `bind` as
   open (decided, thought-loop plan §3–§4).
- **1. Compiler work.** *Added 2026-10-04:* the open read of 6.8-1 made the
   whole forward 2.6× slower (.141 s against .055 s on the XOR fixture;
   [profile](doc/benchmarks/2026-10-03-operators-attention/attention-profile-review/measurement.json));
   Alec: "the slowdown will have to be addressed." *2026-10-09:* the
   math-chain campaign ran at 1.5–2.5 s per sentence under two workers
   (episodes of 32 thought steps included), a week for thirty trainings;
   Alec deferred the thinking measurement "until we further optimize
   things" — this item's measurement now includes a thought episode's
   step and the per-sentence time at the math corpus. *2026-10-10:* profile
   the rising per-presentation runtime in the 6.1 curriculum (the final
   relation-answer stage took 1,001.95 seconds for 64 presentations plus
   its initial/final measurements). Its timing traces and active-autograd
   process sample are retained in the [6.1 receipt](doc/benchmarks/2026-10-09-item6-1/README.md);
   compare like-for-like work before assigning a cause. The decoder's two-word
   walk (6.8 plan §10.2) may pay part of it back. The throughput levers for the serial loop — batch
   across sentences, the known-word lookup concession, subsampled
   reconstruction, closing the host islands and moving
   the sparse stores to device CSR, single passes — are listed in
   [FutureWork](doc/FutureWork.md#throughput-levers-for-the-serial-loop-item-1-candidates)
   and belong here (Alec, 2026-09-25); the `serial` flag and the
   `subsymbolicOrder`/`symbolicOrder` knobs those levers named are rejected
   at load since 6.8, so the levers apply to the one interleaved loop.
   Forward chooser split/lift once per slot; backward's
   launch-bound kernel count per brick; B24 brick +25% against pre-ladder.
   The comparison reader of round 2e is a `functional_call` copy of the
   whole answer path (plan §23, observation 1); share its structure when
   the answer path grows.
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
   expectation discrepancy, a reconstruction regression against the
   re-taken baseline, a forgetting pass deleting protected rows, a
   provisional-pool exhaustion warning, or order-0 inventory approaching
   `nVectors`. *Baselines:* the `99207a3` and reviewed-9b reconstruction
   figures predate round 3a, after which identified-word reconstruction is
   an exact zero audit; re-baseline reconstruction on the accepted
   `f4a68404e` source before the run (items 8 and 5 cite the old figures
   until then). *Condition (operators plan §13, confirmed by Alec):* the
   run trains by expectation and output along one path, with expectation's
   sources live and reconstruction retired to an audit, only where item 3's
   audit reads zero at the run's width; where it does not, the run keeps
   reconstruction as a loss for that part. Expectation across the
   architecture itself is item 4.5. At the trained checkpoint, resume the
   deferred **MM_math_chain learning measurement** under all four corrections:
   one-step problems first; worked steps scored as intermediate answers;
   an episode that binds a found candidate; and budget in steps, with each
   query's record charge bounded separately. The stopped campaign supplies
   no learning claim; preserve it and declare the corrected measurement
   before running ([closing receipt](doc/benchmarks/2026-10-09-item6-2-closing/README.md),
   [protocol corrections](doc/benchmarks/2026-10-08-math-chain-repair-2/protocol-corrections-14-13.json)).

   **6.1's learning gates** also remain at item 0's checkpoint: supplied
   answers and asked parts; free reading in the lesson's order on permuted
   fields, with order-lesson CE near zero; and the context stages in
   [stream-state §3](doc/plans/2026-10-08-stream-state.md#3-required-learning-stages-the-replacement-for-explicit-context-management).
   The support mechanism and the first two generated-curriculum stages are
   accepted; these learning claims are not.

Everything that is decided in direction but not on this path is in
[FutureWork](doc/FutureWork.md).

### Done (newest first)

- Item 6 gate-off measurement and search-inverse exemption: `f1dbd12a2` (including `c647e1506`), pushed on main, WikiOracle `27a60da` ([receipt](doc/benchmarks/2026-10-14-item6-gate-off/README.md)); accepted in plan §11. No learning claim; the fresh corrected-objective warm-up and its singleton gate are measured separately.
- Item 6 part-2 M/N/O/I preparation and first eight-branch native experiment: `af634ed35` ([receipt](doc/benchmarks/2026-10-10-item6/identity-first/README.md)); reviewed in plan §7, no learning claim. R/S/T and all failed learning gates remain open.
- Item 6 R2 inverse-domain repair: `348fb86c7` + `90ef40a42`, pushed on main, WikiOracle `066fb5d` ([receipt](doc/benchmarks/2026-10-10-item6/identity-r2-inverse-domain/README.md)); accepted in plan §10. Reconstruction certificates and observer repaired; no learning claim, and part 2's lessons remain open.
- Item 6 corrected-objective warm-up: accepted `345734c38`…`d5d300aed` is pushed on main, with WikiOracle bump `7bf96de` ([receipt](doc/benchmarks/2026-10-15-item6-corrected-warmup/README.md)); accepted in plan §13 as a measurement, with no learning claim. The observation-only singleton gate was 8/24; paired training is measured separately in part 2.
- Item 6 R/S/T mechanism and matched measurement: `e38a19fbd`, WikiOracle `64a06d3` ([receipt](doc/benchmarks/2026-10-10-item6/identity-rst/README.md)); reviewed in plan §8.4, no learning claim. All failed learning gates remain open.
- Item 6 part-2 corpus and measurement preparation: `06f6e2b88` ([receipt](doc/benchmarks/2026-10-10-item6/identity-corpus/README.md)); M/N/O/I and the first experiment are accepted as preparation by plan §7, with no learning claim.

- **Item 6 H/I follow-up** (`9cc5f6e327`): unavailable reconstruction coverage is zero; I’s empty-inverse and remaining orientation causes are bisected, with learned order still open ([receipt](doc/benchmarks/2026-10-10-item6/followup/README.md)).

- **Item 6 part 1** (`324b84d67f`): shared stored/live inverse, dropped-support keep charge, cache recovery, captured-input isolation, family-order retirement and residual/greedy components; part 2 and measured failures stay open ([receipt](doc/benchmarks/2026-10-10-item6/part1/README.md)).

- **Item 6.1** (`4d9d9ca8a`): support masks, one read per placement, eight-space STM with witnesses, derivation reconstruction and the order lesson; curriculum harness with identity/permanence and word identification passing; learning gates at item 0 ([acceptance and receipt](doc/benchmarks/2026-10-09-item6-1/README.md)).
- **Item 6.2 closed** (initial mechanism `e43638a`; Alec accepted the closing October 9, thinking §14.13): mechanism complete, decomposition demonstrated; learning deferred to item 0 under the four protocol corrections—one-step problems first, worked steps as intermediate answers, binding a found candidate, and budget in steps with each query’s record charge bounded separately ([acceptance and receipt](doc/benchmarks/2026-10-09-item6-2-closing/README.md)).

- **Item 6.5 mechanism** (Alec accepted October 7; one landing from `f4a68404e`): definedness, native identity/change columns and global bind/mint; learning gates remain pending the million-sentence checkpoint ([acceptance](doc/benchmarks/2026-10-07-item6-5/acceptance.json)).
- `f4a68404e` **Operators update, final-b** (Alec accepted October 7): addressed rows, bipolar meanings, centroids, priming, catalogue and two-lane repair; all standing counts 10/10, R/E zero, green sweep ([acceptance](doc/benchmarks/2026-10-07-operators-final-b/acceptance.json)).
- `cce3a4f7b` Operators round 3a: identity by construction ([receipt](doc/benchmarks/2026-10-07-operators-round3a/README.md)).
- `3de37eef5` Operators round 2: accepted credit repair ([receipt](doc/benchmarks/2026-10-06-operators-round2e/README.md)).
- `73cd7b71b` Operators round 1 accepted and pushed, with WikiOracle bumped at `c9670b5` ([receipt](doc/benchmarks/2026-10-05-operators-update/README.md)).
- `42daf96f` Item 6.8 accepted: one attention, two spaces and one index; tag `6.8-landing`; class **7/10**, reconstruction **9/10**, joint **6/10**, MM_xor **10/10**, sum **10/10**, sweep **4,917 passed / 0 failed**, zero ownership conflicts ([receipt](doc/benchmarks/2026-10-03-operators-attention/README.md)); carried work: [FutureWork](doc/FutureWork.md#carried-from-the-68-1315-rounds-2026-10-04) and the [operators update plan](doc/plans/2026-10-05-operators-update.md).
- `bfe6a0d7` Item 6.9: the baseline of the composition mechanism (6.8 §12.1), one reconstruction/output decoder and one writer per weight; accepted class MSE .11475 and reconstruction 0/4 remain historical, with zero ownership conflicts ([closing receipt](doc/benchmarks/2026-10-03-item6-9-closing/README.md), [round history](doc/benchmarks/2026-10-03-item6-9-closing/todo-history.md)).
- `9810fc7` Item 7: two truths, indexed definitions, ended clause state and shared row-free predicates; accepted under review §25 after all three fixture ports and required checks ([landing receipt](doc/benchmarks/2026-09-30-item7-review-round5/landing/README.md)).
- `6906727` Item 7.5: one-operation exploit/explore derivations trained at each sentence closing, reduction pressure/deadlines and closing-gradient reporting; accepted with the unchanged depth-three campaign red ([receipt](doc/benchmarks/2026-09-27-item7-5-landing/README.md)).
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
