# Word-grain open fronts — trace filtering, checkpoint gap, lift/lower + G3, trained regime, seam re-evaluation

> Directive (Alec, 2026-07-13): after the v3 in-model word store, take
> care of the word-grain OPEN list (trace filtering, trained regime,
> lift/lower basis, G3 reading-scope priming, depth-3 parse) and the
> flagged pre-existing checkpoint gap for the RadixLayer extras; THEN
> re-evaluate the two blocked query-tools seams (§13 next-sentence via
> Method-2 exactness; depth-3 relation-endpoint parse fidelity).
> Parent plan: 2026-07-12-word-store-typed-reverse.md.

## Tasks

- [x] **A — RadixLayer extras ride checkpoints.** `BasicModel.save`
      collects `perceptualSpace.percept_store.vocab_extras()` under a
      new bundle key (emitted ONLY when a store exists and is
      non-empty — old blobs byte-identical); `_restore_vocab_extras`
      restores it (pre-state-dict, the existing envelope contract:
      `RadixLayer.load_vocab_extras` re-allocates capacity first so
      the shape pre-check sees matching dims). Words (trie +
      inverse_table + hash_map + hits) survive reload; `word_ids` /
      `word_text` / `get_id` round-trip.
- [x] **B — trace filtering to word-bearing folds.** STM slot-kind
      provenance, mirrored against the buffer discipline
      (newest-at-slot-0; reduce folds slots (1, 0) -> parent at 0,
      tail shifts down):
      * `push_step_masked` stamps kind `word` (the serial per-word
        commit is its only live caller); `_stm_shift_and_push` stamps
        `other`; kinds live on the STM as per-row host lists,
        eager-only, RECORDED only when `<PartSpace><wordStore>` is on
        (no host cost elsewhere); STM reset clears them.
      * `_stm_bounded_reduce_step` extends the reverse-reduce trace
        tuple to `(marg, can, left_word[B], right_word[B])` read from
        the kind stacks at fold time, then folds the kinds
        (word ∧ word -> word, else other).
      * `_reverse_reduce_unfold` emits `x1` only where the forward
        fold's LEFT was a word, and appends the final carry only
        where the first fold's RIGHT was a word (backward-order
        overwrite gives exactly that). Old 2-tuples = no filtering
        (back-compat; synthetic tests unchanged).
- [x] **C — lift/lower driver typed rows + G3 reading-scope priming.**
      The Language driver's Lift/Lower reverse (basis at the
      `subspace.what` pass site) gains, under the SAME
      `<PartSpace><wordStore>` gate, WS-side word-row restriction
      (the `_word_whole_ss` text->position registry resolved through
      `_ws_pos_to_row`) — the WS analogue of v3's PS rows. G3: widen
      the driver's isinstance guard to Lift/Lower so the EXISTING
      `<attention>` retrieval modes (`retrieval_candidates_for_slot`)
      can prime those slots; no new mechanism. The grammar exact-1.0
      bar is protected by the gates (no shipped config flips them).
- [x] **D — trained regime.** Free-derivation ladder on
      `MM_20M_grammar_wordstore` (E in {3, 38, 80}, seed 0,
      cpu/eager — fidelity comparisons stay on cpu per the device
      policy) with A+B+C landed; record exact / where / recon per
      rung; re-pin `test_mm20m_grammar_free_derivation_ceiling` ONLY
      if the bar moves.
- [x] **E — re-evaluate the blocked seams.** With A-D measured,
      re-state in todo.md: does Method-2 exactness now carry the §13
      next-sentence route, and what EXACTLY does depth-3 parse
      fidelity still need (grammar-routing training design)? The
      protect-depth machinery (`_sentence_relative_mask`,
      `protect_depth == 3`) already preserves relative end-states —
      the missing piece is measured, not guessed.

## EXECUTION NOTES (2026-07-13, cpu/eager, seed 0)

**A LANDED.** `_collect_vocab_extras` emits `"ps_percept_extras"`
(store present AND non-empty only — old/storeless blobs
byte-identical; the lexicon-less arm's emptiness early-return extended
to match); `_restore_vocab_extras` restores pre-state-dict (the
envelope contract — `RadixLayer.load_vocab_extras` re-allocates the
shared-basis capacity first). Pinned: roundtrip through a FRESH
RadixLayer on the same basis preserves size / word texts / ids;
storeless collect emits no key.

**B LANDED.** Slot-kind provenance: `ShortTermMemory.kinds_enable /
note_push_masked / note_push_all` (host stacks, newest kind at index
0; `ensure_batch` re-inits, `clear` empties; recording OFF by default
— zero cost). Enabled at the serial-loop start on
`<PartSpace><wordStore>` configs (carried STM content tagged
'other'); the per-word commit push tags 'word' (the masked push runs
on BOTH serialObjectMeta arms — commit gate falls back to the
word-active gate, so the grammar config tags correctly);
`_stm_shift_and_push` bookkeeping pushes tag 'other'. The reduce step
folds kinds at EVERY eager reduce (buffer sync, not just traced
sweeps; word ∧ word -> word) and the sweep trace rides 4-tuples
`(marg, can, left_word, right_word)`; mismatch rows degrade to
legacy True/True. `_reverse_reduce_unfold` emits x1 only where the
fold's LEFT was a word and appends the final carry only where the
first fold's RIGHT was (backward-order overwrite); legacy 2-tuples
stay unfiltered (synthetic pins unchanged).

**C LANDED.** The unreduce dispatch guard now includes Lift/Lower in
the G3 heat-retrieval family (their reverse routes through the
recommender since Track-1 G1; the stale "algebraic inverse" comment
corrected) — still dormant while `attention` stays off. NEW gated
block (`<PartSpace><wordStore>`, SS space_role, recommender-family
ops, heat rows take precedence): `left_rows`/`right_rows` = the WS
word-whole rows (`_word_whole_ss` resolved through
`_ws_pos_to_row`). Lift/Lower reverses already accept the kwargs.
Failures degrade to the plain reverse (never break generation).

**Tests:** test/test_word_store.py now 16 (A roundtrip + storeless
key-absence; kind-stack discipline; filtered-unfold counts incl.
legacy; live eval-forward records 4-tuple traces; WS registry row
resolution). Broad sweep (heat_reverse_wiring, primed_reverse ×3,
ps_reverse_e2e, boolean_reverse_recommender, radix_layer_reverse,
intra_sentence_layer_reverse, bounded_stm_fold, percept_store,
thinking_kernel, basicmodel): **367 passed / 2 skipped**.

**D MEASURED (free-derivation ladder, MM_20M_grammar_wordstore, seed
0, cpu/eager, A+B+C live):**

| E | exact | where_rec | recon |
|---|---|---|---|
| 3 | 0.0 | 0.0 | 0.048267 |
| 38 | 0.0 | 0.0 | 0.044975 |
| 80 | 0.0 | 0.0 | 0.040038 |

recon trains; the exact bar does NOT move with word-true candidates +
trace filtering in place, and `where_recovery` is 0.0 at every rung —
nothing can exact-match while the POSITIONAL decode recovers nothing.
The binding residual for Method-2 exactness is therefore NAMED: the
`where`-band decode on the free-derivation path (the same band
precision Gate-B left as Alec's open knob) + operand content quality;
candidates are no longer the front. Ceiling pin unchanged (no
re-pin).

**E MEASURED (depth-3 probe, MM_query_reasoning provisioning,
shipped AND wordStore+threshold-2 variants — probe in the session
scratchpad):**

- `_sentence_relative_mask`: 4 calls, **0 fired**, and
  `current_rules` is **None** at every call — the grammar dispatch
  never populates rules during the provisioning read, so the
  (already-built) depth-3 protect machinery has nothing to key on.
- Post-sweep depths **[1]** for both truth texts — the depth-1
  absolute collapse, exactly as the kernel notes recorded.
- Word substrate: **no words promote on either variant** —
  MM_query_reasoning is `synthesis=radix`, NOT meronomy: PS has no
  per-word tiling (lexer tokens are whole LINES; `observe_chunk`
  counts line recurrences), so word percepts / v3 recognition cannot
  fire there regardless of knobs.

## WHERE-decode correction (Alec's challenge, second pass 2026-07-13)

Alec: "the concept that is formed of parts and whole will always have
perfect position information (it gets stamped in)" — CONFIRMED, and
the first-pass "where-band decode precision" framing above was WRONG.
Measured mechanism: `where_recovery` reads per-slot `.where` claims
off the reconstructed event's band (`_decode_radix_meta`), and
`_reverse_reduce_unfold` ZEROED that band by construction — on the
free path no placement source ever engaged. Not lossy, not a training
gap (the recommender-recovered operands bypass the trained D3 reverse
entirely).

**Landed (second pass):**

1. **`_stamp_unfold_where`** — the fold order IS the position:
   emissions are earliest-first, each operand that snaps to a stored
   word row has a known surface length, so sequential byte offsets
   re-derive exactly and are stamped through the CS `whereEncoding`
   (the forward's idiom). Unmatched slots (sentinels / non-row
   operands) leave the band unwritten (the pre-existing
   below-noise-floor contract). Pinned by
   `test_unfold_stamps_sequential_where_offsets`.
2. **Sentence-scoped trace** — probing the stamp exposed the REAL
   horizon problem: the 2b-2 opportunistic reduces (Alec 2026-07-12)
   fold most of the sentence DURING reading, and the sweep-local
   trace saw only composite survivors (the un-fold emitted ONE slot
   per row). On wordStore configs the trace now resets at the
   serial-loop sentence start (`_stm_trace_sentence_scope`; the
   sweep's own reset stands down), so the backward walk unwinds ALL
   of the sentence's folds — measured: 23-fold unwind per row.

**Measured after both (E=1 probe):** the un-fold now reaches per-word
depth, but `where_recovery` stays 0.0 for two NAMED reasons that are
the true remaining front: (a) the mid-read folds' CHOSEN ops are the
relation hosts (the untrained DP chooser), whose reverses do not
return codebook rows — so operands are not word rows and the stamp's
row match never binds; (b) the kind filter dilutes (most steps pass
as word — mismatch fallbacks / kinds bookkeeping under the mid-read
interleave needs a dedicated probe). Both belong to the
FOLD-OP-CHOICE / trained-chooser front: until the reduce chooser
routes word folds through the recommender family, free-derivation
operands are not store rows, and neither content nor placement can
bind. This subsumes the earlier "content quality" phrasing.

**Regression state:** test_word_store 17 + bounded_stm_fold +
decode_exemplars green (26 passed) after both changes.

## mereologyRaise flip (Alec's decision, 2026-07-13 third pass)

Alec confirmed the knob's purpose ("governs the raise of the order of
the percepts... otherwise we cannot do subsymbolic raises") — verified
in code: `<architecture><mereologyRaise>` stamps `_mereology_raise`
onto the terminal WS + PS + stage-0 WS and gates the word-whole
σ-binding arm, `maybe_raise_order` (higher-order percept parts, the
subsymbolic raise ladder), the percept `.where` band fill, the
run-structure observation, and (with v3) in-model word recognition.
FLIPPED ON in `data/MM_20M_grammar.xml` + the wordstore matrix variant
(kept parent-verbatim + one flag).

**Measurements:** E=3 free-derivation recon is BYTE-IDENTICAL pre/post
flip (0.048267) — consistent with the site's own contract ("gates ONLY
the lattice behaviors"): the word arm is off-gradient bookkeeping at
this horizon (note: `insert_whole` does write WS virgin rows, so
longer trainings may shift once those rows are read — watch at the
next re-baseline). END-TO-END recognition verified on the flipped
config under the real loop discipline (forward + hard `Reset` at the
sentence boundary — `_commit_autobind_from_stash` is Reset-driven):
word-wholes bind {hello, world, there, loving} and `_recognized_words`
fills under the full strict criterion.

**Order-dependence finding (pre-existing, NOT the flip — proven by
control):** the 9-file sweep (compile_static_loop,
conceptual_recurrence, config_matrix, dimensional_governance,
dual_towers, per_word_capture_gate, per_word_ss_padding_noop,
rule_gate_isolates_side_effects, reconstruction_roundtrip) fails
`test_per_word_capture_gate`'s two fullgraph pins DETERMINISTICALLY in
that order, with the flip AND with it reverted; the file passes in
isolation and in (dual_towers, capture_gate) / (config_matrix,
capture_gate) pairs. Single-process cross-file contamination —
recorded in todo as a test-hygiene item.

## CLOSING SWEEP (Alec: "finish the todo.md in this thread", 2026-07-13)

**Test hygiene (todo §5) — CLOSED.** Order-bisect: ONE poisoning file,
`test_compile_static_loop` — its recompile-churn test deliberately
blows dynamo's recompile limit on `_per_word_body_step`, marking that
code object skip-forever; the capture-gate's fullgraph compile then
reports "no compiled frames. Compilation was not attempted."
Fix: an autouse `torch._dynamo.reset()` fixture in
test_per_word_capture_gate (self-isolation). The poisoned pair runs
8/8 green.

**Kind dilution (todo §1b) — CLOSED, two causes.** (i) The commit
gate is the always-true slab-activity mask on muxed events, so 6
padding pushes/row were tagged 'word' → the tag is now the per-slot
FILL TEST (`Models._push_kind_word_col`: pid's own `bytes_for` length
equals its word-tile span — the same test recognition uses); padding
pushes note 'other' (stacks must mirror the buffer). (ii) The
mismatch default flipped to STRICT (unknown ≠ word). Measured: push
tallies exactly {'word': 8, 'other': 24} on the 4×'2-word' batch;
un-fold emissions 23 → 3 slots/row.

**Fold-op constraint (todo §1a) — LANDED.** Word-bearing backward
steps dispatch the MAX-fold recommender regardless of the chooser's
op (`rec_gl`, disjunction/union preferred — the min-fold's ≥-filter
yields only sentinels against composite parents).

**THE NEW BINDING FRONT (supersedes "fold-op choice"):** with
provenance, candidates, dispatch, and placement all in place, the
operands STILL degenerate to sentinels because
`Ops._binary_op_recommend`'s elementwise ORDER FILTERS
(`(row <= parent).all(dim)`) are designed for NON-NEGATIVE lattice
activations — over SIGNED 1016-dim CS ideas no real word row is
elementwise comparable to the parent, so only ⊥/⊤ stay feasible. The
un-fold's operand recovery needs a SIGNED-SPACE SNAP — nearest word
row by overlap/cosine, exactly Track-1's
`snap(proposal, store, metric, candidates)` contract — in place of
the order-filter recommender at idea grain. DESIGN CALL (Alec): the
snap metric and the residual rule for the max-fold in signed space.

**MM_query_reasoning meronomy switch (todo §2b) — CLOSED.**
synthesis radix → meronomy + `chunkPromotionThreshold 2` (the tiny
truthSet corpus promotes its recurring words). Kernel suite 66/66
green; measured: the store now promotes ['human', 'partOf'] during
provisioning (radix promoted NOTHING — no per-word tiling existed).

**Provisioning rule dispatch (todo §2a) — PRECISE, design call.** On
the serial path `_chart_compose_at_C` (the router that populates
`current_rules`) fires AFTER the per-word STM accumulation, while
`_sentence_relative_mask` reads `current_rules` DURING the reduce
sweep — the mask structurally cannot see THIS sentence's rules.
Options for Alec: (i) a boundary chart fire BEFORE the sweep on the
serial path (touches the training-critical surface the per-word plan
deliberately reused verbatim); (ii) sweep twice (protect after the
post-hoc chart); (iii) provisioning-only: let the truth entry's
``kind`` tag drive ``protect_depth`` directly (the tag already forces
``rel_type``). Note: depth-3 ALSO needs in-grammar/vocab words — the
provisioning docstring's parse-fidelity caveat stands.

## SECOND CLOSING PASS (Alec's four directives, 2026-07-13)

**Item 1 — RADIAL (Alec: "Min should be a radial min to deal with
signed activations") — LANDED.** `Ops._binary_op_recommend` gains
``radial=True``: the ordering filters and the union residual use the
RADIAL order (per-dim magnitude, matching `_radmin`/`_radmax`), with
ZERO parent dims treated as ANNIHILATION WILDCARDS (a radmax/radmin
zero is a sign conflict, not a magnitude bound — without the wildcard
the deep-fold envelope collapses and only sentinels stay feasible;
measured). Threaded through `_binary_op_inverse_impl`,
`conjunction/disjunctionReverse`, and Conjunction/DisjunctionLayer
(`self.radial`, default False = byte-identical; their compose kernels
flip to `Ops.intersection/union(monotonic=False)` = RadMin/RadMax —
IDENTICAL on non-negative SS activations, signed-safe on CS ideas).
New gate `<architecture><radialStmReduce>` stamps the terminal WS
conjunction/disjunction hosts; flipped in the wordstore matrix
variant + MM_query_reasoning. Pinned:
`test_radial_recommender_recovers_signed_pair` (signed pair recovered
EXACTLY from its radmax fold, incl. an annihilated dim). MEASURED on
the bench (E=1): `where_recovery` moved 0.0 → 0.1667 — the first
nonzero placement on the free-derivation path; content partially
binds (word rows admitted again after the wildcard fix). The trained
regime remains the open dial.

**Item 2 — PER-WORD ROUTER (Alec: "the router should be firing as
every word is added, since the STM is not long enough to preserve the
sentence before parsing is initiated") — LANDED, and the finding is
sharp:** the documented `<routerWireSerial>` semantics ('per-word',
and the 'both' DEFAULT claiming per-word during training) were
VALIDATION-ONLY — no fire site ever existed. `_chart_compose_per_word`
now fires the signal router over the current STM after each word push
(eager-only; the full-router island is compiler-disabled). MEASURED on
MM_query_reasoning provisioning: `symbolSpace.current_rules` now
populates DURING the read (rule ids [0, 5, 6, 8] fire per word) — and
NONE is a relative rule, so the depth-3 residual is now EXACTLY the
routing choice (the untrained router never picks the relative rules
for 'socrates partOf human'): the training front, with every wiring
live. The earlier boundary-vs-sweep ordering design question is MOOT
(the per-word fire supersedes it). Fire guard: parsing initiates at
TWO constituents — the binary layer's degenerate N<=1 path returns a
routing dict without the rule-id keys (found by the accumulator /
capture-gate pins; the N>=2 guard fixed it; full compact sweep 121
passed / 1 xfailed). Alec's design notes recorded:
NP-R-NP (a relation over two propositions) is a START STATE exactly
like NP and NP-VP (the grammar already carries the relative_truth
starts); and UNIVERSALIZABILITY is to be expressed as an NP-R-NP
relative truth ("one person is better than another") rather than the
cumbersome S-V-O action form.

**Item 4 — XOR BARS (Alec: "you should be able to reason the answer")
— CLOSED by reasoning + a discovery:**
- The affine MM_xor head: converted from xfail-below-0.15 (which
  certified nothing — 0.15 is below the reachable floor) to the FLOOR
  TEST: best affine MSE on XOR is exactly 0.25 (the constant 1/2), so
  the bar is REACHING it (< 0.26). GREEN.
- The grammar convergence 0.15: annotated TWO-TIER — an empirical
  regression pin (certifies beating the affine floor ~2x; guards
  refactors), with the THEORY bar (<= 0.05, the XOR_exact crisp
  exemplar — the path is jointly nonlinear) recorded as the named
  frontier. Kept green and load-bearing.
- The MM_20M_xor roundtrip pins: the exact-roundtrip bar ALREADY
  asserts the theory value (1.0/1.0; nonlinear config + discrete
  vocab; xfail-annotated as the active reverse-path gap on the tree).
  The harness-budget E=3 pin: measured 0.5/1.0 = exactly its asserted
  values — it PASSES IN ISOLATION and in pairs; the "standing
  failures" were ORDER CONTAMINATION all along (annotated in the
  docstring). No long runs were needed — reasoning + isolation runs
  answered it.
- REMAINING hygiene: the composition-dependent dynamics leak (which
  earlier file shifts the E=3 dynamics in multi-file runs) is
  unlocated — the capture-gate dynamo leak (fixed) was a different
  mechanism.

**Item 3 — parked by Alec ("Lets come back to this").**

## THIRD CLOSING PASS (Alec's decisions executed, 2026-07-13)

**§3a — WORDS summary row (Alec: order-capped summary row) — LANDED.**
The face is the WELL-KNOWN 'words' ATOM (WS row 0 — the codebase's own
reserved words parent), carrying the RUNNING MEAN of member word-whole
rows with a FULL fold record (abstraction_order == max_order). Note:
`_sparse_active` requires parallel mode, so the CS-side populate path
is structurally dormant on the serial configs where WORDS lives — the
atom row is the serial-mode realization, unifying the row-plane atom,
the relation-plane concept, and the face. DEFERRED-WRITE discipline:
registration runs at the sentence-boundary Reset (between a forward
and its backward), where a Parameter write invalidates the pending
graph (measured "[1016] at version 1" crash) — updates STASH
(`_pending_words_summary`) and DRAIN at the next serial-loop start
(`apply_pending_words_summary`, pre-graph). Pinned:
`test_words_summary_row_running_mean`.

**§2 — SYNTACTIC ANCHORS (Alec: "lexical anchoring, and in fact
syntactic: NP-R-NP ... the 'is of definition' that does not reduce to
the 'is of predication'") — LANDED end-to-end:**

- `complete.grammar` gains `<Anchors>` — the closed-class relation
  surfaces (part ⇐ partOf/…, whole ⇐ wholeOf/…, equal ⇐ isEqual/…),
  parsed into `TheGrammar.surface_anchors` (casefolded surface →
  operator).
- The word-whole binding registers anchored pids
  (`ws._anchored_pids[pid] = op`); `_build_category_context`
  short-circuits an anchored slot to the operator's OUTPUT role
  one-hot (`<op>_O1`) — grammatical resolution, no learned centroid.
- `provision_ltm` now fires the sentence-boundary hard Reset per truth
  text (each text IS a sentence) — without it the Reset-driven seams
  (autobind, recognition, anchors) never ran during provisioning.
- MM_query_reasoning: WS `<codebook>none -> quantize` (the autobind
  documentedly no-ops without a WS Codebook — the `ws_cb: False` gate,
  found by instrumentation).
- Per-word router fire hardening (from the ladder's backward crash +
  the fullgraph pins): the fire feeds a DETACHED CLONE (snapshot() is
  a VIEW of the live STM buffer and the router's compose mutates its
  slab in place — bumping the version the pending backward saved), and
  fires only at N >= 2 (the binary layer's degenerate N<=1 path
  returns a keyless routing dict).

**MEASURED (provisioning, MM_query_reasoning):** word-wholes bind
({socrates, human, partOf, mortal}), 'partOf' ANCHORS to `part`
(pid 16), recognition fires ({human, partOf}) — the full chain is
live. The relative mask still reads False at provisioning end for a
STRUCTURAL reason: the anchor registers at the LAST text's boundary,
so no read has yet run WITH the anchor in force — engagement begins
with the next reads (training epochs), which is exactly where the §2
routing campaign starts, now with the syntactic anchor in place.

**GRAMMAR-FILE TRAP (second occurrence):** inserting at a
section-name substring hit `<Queries>` INSIDE a comment (line ~104) —
XML parse error. Anchor config/grammar insertions on the
LINE-ANCHORED form (`\\n  <Tag>\\n`), per the recorded
fixture-injection rule.

**Suites:** full compact sweep after everything —
**124 passed / 1 xfailed** (word_store 19, thinking_kernel 66 on the
codebook-flipped config, mm_xor with the floor test, static-loop +
capture-gate pair, bounded_stm_fold, mereology_word_binding,
autobind_from_cs, decode_exemplars). The radial E-ladder relaunch is
in flight; the first post-radial rung re-baselines recon (the radial
fold is a TRAINING change on this config: E=1 recon 4.21 vs the
lattice 0.048 — the suites are the arbiter, per the repair precedent).

**Seam re-evaluation (the deliverable):**

1. **§13 next-sentence via Method-2 exactness — still blocked; the
   blocker moved.** Candidates (v2/v3) and trace granularity (Task B)
   are now in place and verifiably engaged; the measured front is the
   free-derivation `where` decode (placement 0.0) + the trained
   content regime. Next increment: route the un-fold's emissions
   through the `where`-band placement (Gate-B's band-precision knob
   is the shared dependency).
2. **Depth-3 relation-endpoint parse fidelity — still blocked; the
   gap is two NAMED wirings, not (yet) training.** (a) The
   provisioning read must run the grammar dispatch so
   `current_rules` carries the relative rules when the truth text IS
   relative (today: None ⇒ mask can't fire ⇒ collapse to depth 1);
   (b) MM_query_reasoning needs the word-grain synthesis switch
   (meronomy, as MM_20M_grammar had in Task 6) before word percepts /
   recognition exist there at all. Only after (a)+(b) does
   "grammar-routing training" become the operative frontier.
