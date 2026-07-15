# Signed-space snap — design record and execution notes

> Directive (Alec, 2026-07-14): Decision A "is just dot-product", with
> the refinement that recovering N from ADJ(N) — an intersection over
> the two operands — benefits from a metric better than L2 that scores
> the match over the sparse support unaffected by the Adjective.
> Decision B: the JOINT PAIR SNAP, with the modeling principle that
> nouns are adjectives pre-applied to the domain $[1, 1, 1, \ldots]$ —
> black(cat(everything)) treats "black" as roughly the same kind of
> function as "cat" (the latter already mapped). Remaining calls
> delegated ("take your preferred actions for the rest, document them
> so we can revisit").

## The finding that reframed the problem

The "structural best-cos 0.8711" from the four-fronts chase
(doc/plans/2026-07-13-todo-four-fronts-execution.md §1) was measured to
be the SENTINEL DEGENERACY itself: every free-derivation operand emitted
by the radial recommender was the all-ones $\top$ sentinel — one unique
vector across all emissions, at every budget — and 0.8711 is simply
$\cos(\mathbb{1}, \text{nearest word row})$. The radial feasibility
filter (candidate radially contained in the parent on EVERY dim) admits
no real word row against trained composite parents, so the argmax always
lands on the sentinel. This is consistent with Alec's nouns-over-
$\mathbb{1}$ principle: word rows are attenuations of the ones domain,
so the sentinel sits close (but never equal) to all of them. The snap
therefore REPLACES the feasibility filter with similarity scoring —
nothing survives of the order-filter at idea grain.

## Decisions and their record

**Decision A (Alec) — the metric is the dot product.** Rationale: dot
ignores dims ABSENT from the parent (a modifier's attenuation costs the
true noun nothing, where L2 punishes every missing dim), and parent-zero
annihilation wildcards contribute nothing for free.

**My refinement (measured, documented for revisit):**
support-restricted normalization — ``score(c) = <p, c> / ||c||_supp(p)``
(``Ops._support_dot``). RAW dot was measured to collapse the decode:
without normalization the two largest-norm store rows won the argmax
for EVERY parent in every batch row (all rows decoded identically).
Restricting the norm to the PARENT's support keeps Alec's rationale
intact — the candidate's support outside the parent is still free — while
killing large-row dominance inside it. Known limit: a candidate that is
a uniform RESCALE of the parent's visible support ties the true word;
only pattern differences discriminate (pinned in
``test_word_side_snap_support_match_beats_l2``).

**Decision B (Alec) — joint pair snap** for word∧word folds:
``(x1, x2) = argmax over word-row pairs of <radmax(w_i, w_j), parent>``
(``Ops.word_pair_snap``; the store is tiny, so the $K^2$ pair sweep is
free). Nouns-as-preapplied-adjectives makes both operands the same KIND
of candidate — one store, no typed asymmetry.

**My calls for the rest (each revisitable):**

1. **Minimal residual** (``Ops.word_side_snap``): after snapping the
   word side of a word∧other fold, the residual carries the parent
   EXACTLY on the dims the word does not radially dominate and zero
   elsewhere — the next backward step scores against the UNEXPLAINED
   content, not the parent's echo (the maximal-residual alternative
   re-picks composite-like rows, the old one-dominant-word failure).
2. **No snap threshold.** The snap always returns the argmax pair; the
   below-noise-floor honesty lives in the METRICS (exact-match /
   where-recovery judge the outcome), not in a silent gate. The
   replaced behavior emitted sentinels, which no threshold improves.
3. **Pair-order ambiguity, recorded.** radmax is symmetric: the fold
   value cannot say which operand was the sentence-earlier word. The
   tie-break (higher one-sided dot first) is deterministic, not
   semantic. Disambiguation candidates for later: the store's recorded
   surface offsets, or discourse context.
4. **Atomic-word kind semantics.** The Task-B kind-fold rule
   (word ∧ word → word) re-tagged composites-of-words as words, so the
   un-fold emitted duplicate words for word-bearing composites
   (measured: 5 emissions on 2-word sentences). A fold parent is now
   always tagged 'other'; 'word' means ATOMIC word.
5. **Emission order — the head/pair/tail law.** The old
   emit-x1-if-LEFT-was-word contract dropped every right-side word
   (measured: empty decodes at trained budgets). Words now route by
   FOLD SIDE: an rw step peels the LAST word of the remaining span
   (the mid-read chain folds fold(composite, NEWEST word)) — those
   collect latest-first and reverse at the end; an lw-only step peels
   the FIRST word (the seal sweep's shape — after its first fold the
   parent sits at slot 0 and every later fold is fold(older word,
   composite)) — those emit in walk order at the head; the word∧word
   pair sits between them. A blanket latest-first reversal (the first
   cut) inverted sweep-shape sentences back-to-front — caught by the
   adversarial review, now pinned
   (``test_unfold_sweep_shape_emits_left_words_first``). Legacy
   2-tuple traces are byte-identical.
6. **Render-priority slab.** The trained CS reverse transport collapses
   the multi-slot un-fold event back to the root's single slot and
   re-stages its own render thunk (measured: render event ``[B, 1, D]``,
   LF band at noise) — the parent plan's §13 "route the un-fold's
   emissions through the where-band placement" increment. The un-fold
   stashes the stamped slab (``_unfold_recovered_slab``) on its
   WORD-ROWS path only (review finding: an ungated stash silently
   swapped the NON-wordstore ceiling's render source; the caller now
   just invalidates before the un-fold, and both RUN_SLOW grammar pins
   re-verified green); ``_materialize_recovered_input`` consumes it
   with priority over the transported thunk (consume-once; cleared by
   the Method-1 staging). Method-1's exact teacher path is untouched.

## Measured state after the snap (cpu/eager, seed 0, wordstore config)

* Operand binding: emitted operands are EXACT store rows — best-cos
  1.0000 at 100% (was: 100% sentinel, 0% $\ge 0.99$).
* End-to-end: words render through the radix render with stamped
  offsets (was: empty decodes at trained budgets).
* Ladder (post-snap; recon byte-identical to the pre-snap re-baseline —
  the snap is decode-only):

  | E | exact | where_rec | recon |
  |---|---|---|---|
  | 3 | 0.0 | 0.16667 | 0.337989 |
  | 38 | 0.0 | 0.16667 (was 0.0) | 0.094532 |
  | 80 | 0.0 | 0.16667 (was 0.0) | 0.089843 |

  Placement now HOLDS at trained budgets — the pre-snap unbinding was
  the sentinel degeneracy, not training.
* The exact bar did not move; no re-pin of
  ``test_mm20m_grammar_free_derivation_ceiling`` (it runs the
  non-wordstore config, which the snap does not touch).

## The two-space ontology (Alec, 2026-07-15 — the governing statement)

> Percepts are purely mereological and live on the UNSIGNED UNIT
> HYPERCUBE. Concepts live on the SIGNED HYPERSPHERE and are defined by
> the presence or absence of various percepts (in later orders, of
> various symbols, which exist 1:1 for every concept). There is no
> "cross-tower autobind" apart from the zero-th order concepts defined
> by parts and wholes occurring at the same location.

Consequences, recorded because they justify or correct several session
decisions:

1. **The snap's metric split is principled, not empirical.** The
   order-filter recommender (elementwise containment) is the CORRECT
   algebra on the percept hypercube; Decision A's dot metric is the
   correct one on the concept hypersphere. The sentinel degeneracy was
   a grain error (hypercube algebra applied at hypersphere grain), not
   a broken recommender.
2. **A concept's DEFINITION and its EMBEDDING never mix.** The
   definition is the signed membership pattern over constituents
   (presence $+$ / absence $-$), held as the sparse store's fuzzy
   set-membership edges ([N $\times$ N+1] ConceptualAttentionLayer —
   the structure that replaced the earlier two-column concept/percept
   table; a concept representing several percepts = several edges on
   its row). The embedding is a RANDOM row of the `similarity_codebook`
   (random init, dot-product/angular metric), shaped ONLY by
   SBOW/SGNS substitutability. Identity flows by INDEX: concept id
   $\to$ shared store row $\to$ codebook row (verified 1:1 — the
   codebook is sized `nVectors` and consumers index it by store row).
   No projection between percept and concept dimensionalities is ever
   needed, because vectors never meet across the tie.
3. **Order-0 concept formation IS the co-occurrence tie**: parts and
   wholes at the same `.where`/`.when` define the concept (the per-word
   A/B/C triple at the autobind seam is exactly this); higher orders
   recurse over symbols (1:1 with concepts).
4. **The collapse diagnosis, restated in these terms (CONFIRMED in
   code, 2026-07-15).** The per-word "idea" on the serial path is
   `cs.forward(PS_sub, WS_sub).materialize()[:, 0, :]`
   (`_per_word_body_step`, Models.py) — a COMPUTED percept-binding
   event, NOT a read of a random concept-table row. Percepts are
   mereological on the bounded unsigned hypercube and contract, so the
   computed ideas inherit that contraction; and the hypersphere's only
   shaping force (SBOW/SGNS) is parallel-only with
   `<conceptualSimilarityScale>` defaulting 0.0, so nothing holds the
   per-word ideas apart.

   **The ORDER of the fix matters — SBOW is downstream of an
   index-read that does not yet exist:**
   - (b) FIRST, the substrate: the per-word idea must READ THROUGH THE
     INDEX — percept (its `.where`/`.when`) $\to$ its order-0 concept id
     $\to$ that concept's RANDOM `similarity_codebook` row — so the
     value the fold consumes IS a concept-table row (signed
     hypersphere), not a recomputed binding. Only then is there a row
     for SGNS to rotate and for the fold to keep distinct.
   - (a) THEN the shaping: run SBOW/SGNS over the sentence's co-present
     per-word concept rows on the serial path
     (`conceptual_sbow_loss`, currently `not self.serial`-gated), so
     those rows spread on the sphere.
   With (b)+(a) the landed leaf distillation finally has separable
   material and Method-2's recovery becomes index-exact. A first
   serial-SBOW pass was attempted before (b) and correctly BACKED OUT
   (2026-07-15): it rotates rows the per-word path never reads, and the
   autobind window it needs never populates without the real
   forward+Reset loop — untestable and premature until the index-read
   lands.

   **(b) LANDED (2026-07-15): RESOLVE, don't mint — and the fold target
   is the OBJECT concept.** Alec's clarifications, in order:
   - *"We do not need to be minting on the serial path: we need to be
     lighting up word-concepts/object-concepts that are already known
     (learned in virtue of the parallel path, which executes both in
     parallel mode and before serial mode)."*
   - *"Wholes and parts form the SUPPORT for a concept when they co-occur
     at the same `.where`/`.when`. The concept, once formed, is
     INDEPENDENT of location."* (So the stable endpoint is the
     `create_word_object_meta` pair, not the transient `loc_sym`
     location-scaffold — location keys support-gathering, not the
     concept.)
   - *"Method-2 will unfold into OBJECT concepts, which are then
     translated into their corresponding WORD concepts (the exact reverse
     of the forward process)."* (So the forward folds B — the referent —
     and words are the decode-side translation.)
   The index-alignment trace found the concept-id → codebook-row half
   wired (`_csw_row_of` / `_csw_concept_row`, verified 1:1) and the
   percept → concept half MISSING. Landed:
   - the reverse tie percept → (A, B) recorded where
     `create_word_object_meta` calls `add_part(A, pid)` — on BOTH its
     mint arm and its idempotent reuse/accrue arm
     (`ConceptualSpace._record_percept_concept`, called unbound for the
     mock-namespace-self harnesses) — plus the B → A translation map;
   - resolvers: `concept_of_percept` (→ A, decode side),
     `object_concept_of_percept` (→ B, the fold target),
     `word_concept_of_object` (B → A, the Method-2 translation step),
     `concept_codebook_row_of_percept` (→ B's order-0 row via the
     idempotent allocating `_csw_concept_row` — allocating a slot for an
     existing concept, not minting), `concept_row_content` ([B] percept
     ids → re-normalized signed-hypersphere OBJECT rows + tie mask);
   - the gated serial read `BasicModel._maybe_concept_index_read` at the
     per-word idea site: sources the promoted percept id from the
     per-word PS index grid, replaces the idea's CONTENT slice with the
     OBJECT-concept row (keeping `.where`/`.when`), masked to tied
     percepts. Gate `<architecture><conceptIndexRead>` default off →
     byte-identical; eager-only; best-effort.
   Pinned: reverse tie + B-row resolution + B→A translation;
   `concept_row_content` resolve/gather/normalize/mask. Suites green
   (word_store + mereology 43); gate-on forward crash-free.

   **(a) RE-SCOPED — serial SGNS RETRACTED (architecture question,
   2026-07-15).** Two verified reasons: (i) random unit rows in 1016-D
   are near-orthogonal (measured mean $|\cos| = 0.024$, max $0.13$ over
   64 rows) — the index-read ALONE restores separability; SGNS adds
   substitutability structure (meaning), not separation; (ii) concept
   shaping is a PARALLEL-path job by design (`conceptual_sbow_loss` is
   deliberately `not self.serial`-gated) — the parallel path learns the
   concepts, the serial path reads them. Wanting semantic structure =
   turn up `<conceptualSimilarityScale>` on the parallel phase, not a
   serial SGNS.
   NEXT: the decode follow-through + re-ladder — under
   `conceptIndexRead` the un-fold's candidate set should be the order-0
   OBJECT rows (`similarity_codebook`), each recovered B translated
   B → A → A's parts (PS codes) → surface (the exact reverse); then
   re-ladder with `leafDistillWeight` re-enabled. Capacity note: the
   order-0 block must seat the word/object inventory (smoke caps0=1
   seats one; real configs need caps0 ≳ 2× distinct words).

   **PRE-FLIGHT (2026-07-15, MM_20M_grammar_wordstore, before the decode
   build) — WIRED + FIRES, but CAPACITY-BLOCKED and currently HARMFUL.**
   Two bug fixes first: (i) the reverse index populates on a PER-STAGE CS
   instance (the autobind runs there), not `m.conceptualSpace` — the read
   now resolves against the index-bearing CS; (ii) the part guard matches
   `add_part`'s `int(p)` (the corpus parts are ints, not the `isinstance`
   guard's type). Then E=38: the read fires 600/608 and hits — the
   mechanism is correct. BUT the per-stage CS's concept codebook is
   caps0=2 (nVectors=4), so the ~4 corpus words COLLIDE onto 2 object
   rows and the read makes the seal roots MORE similar (cross-row diff
   **0.00996**) than the gate-OFF baseline (**0.0896**) — a REGRESSION,
   not a win, purely from capacity starvation. `<ConceptualSpace>
   <nVectors>` 8→64 did NOT change it (the per-stage CS is sized by the
   word-subspace machinery, not that knob). Also: the "0.00017 collapse"
   is a DECODE-time number — the TRAIN seal root is healthy (0.0896), so
   the real arbiter is recon_bench `--free-derivation` exact/where, which
   only becomes meaningful post-decode-build.

   **CAPACITY RESOLVED (2026-07-15, same day).** Why caps0=2 when the
   corpus wants ≈8 (Alec: "1 whole (words), 4 distinct parts" — and the
   A/B/C mint takes TWO order-0 rows per word plus one pool META): two
   vocabulary-blind causes. (1) The serial arm sizes the per-stage CS
   inventory by the TOWER TILE COUNT (``stage_space_concept =
   [cs_out[0], ...]``, the FF-pyramid 8→4→2), bypassing the XML
   ``<ConceptualSpace><nVectors>`` entirely — why the 8→64 bump changed
   nothing. (2) ``_order_caps``' 50/50 snap/pool split halves that
   again. Fix (gated, byte-identical elsewhere): under
   ``<conceptIndexRead>`` an explicit ``<ConceptualSpace><nVectors>`` is
   honored as a FLOOR on the serial arm (``max(tile, explicit)``);
   pinned by ``test_concept_index_read_sizes_inventory_by_explicit_
   nvectors``. MEASURED (nVectors=16 → caps0=8 seats 4×(A+B)): E=38
   seal cross-row diff **0.0966** vs gate-off **0.0896** — parity; the
   collision regression (0.00996) is gone; read fires 600/608. The
   per-word ideas are now index-exact concept rows at NO separability
   cost.

   **DEFERRED (Alec: "reconstructing the XOR from the idea is a nice to
   have, but not a prerequisite"):** the decode follow-through — the
   un-fold's candidate set becoming the order-0 OBJECT rows with the
   B → A (``word_concept_of_object``) → parts → surface translation —
   and the eval-side free-derivation re-measurement that depends on it.
   Serial idea construction/deconstruction is expected to show its
   value over a larger, highly-trained corpus rather than the toy
   XOR/wordstore configs; the substrate (this section) is in place for
   that.

## THE NEW NAMED FRONT — root separability

With recovery, rendering and placement all working, the decode is
IDENTICAL across batch rows because the collapsed roots are identical:
pairwise max-diff across the four eval rows is **0.0117 at E=1** and
**0.00017 at E=38** — training COLLAPSES the sentence roots into a
common attractor (a $70\times$ contraction), with the per-fold
C$\to$S$\to$C roundtrip quantization as a structural contributor. No
decode-side mechanism can distinguish sentences whose roots coincide;
this is a TRAINING-OBJECTIVE gap (nothing preserves root separability)
and the next design call (e.g. a contrastive/identity term on the root,
or a quantization-aware fold). Secondary, behind it: rendered words
concatenate without separators, and pair-order ambiguity (call 3).

## Documented preferences for the other open design calls

* **Depth-3 determinism** (four-fronts §2 residual): escalate in order —
  (i) a config/learnable PRIOR GAIN on the anchored O1 term (smallest
  change, preserves the learned path), (ii) ``<transformChooser>mlp``
  (consumes cat\_ctx directly; new basin), (iii) the anchor-driven
  relative-mask short-circuit (hard guarantee; bypasses the chooser and
  forfeits its training signal). Not implemented — conditional on
  requiring determinism.
* **Attention priming's live surface** (four-fronts §3): decide the
  snap's home first — the snap now owns the un-fold's candidate
  scoring, so priming becomes a later additive bias on that scoring
  (hot rows breaking near-ties) rather than a separate decode route;
  ``attach_knowledge`` in production remains open.

## Adversarial review pass (multi-agent, post-implementation)

Five findings confirmed (two refuted), all addressed:

1. **(high) Sweep-shape emission inversion** — the blanket latest-first
   reversal was correct only for the mid-read chain shape; the seal
   sweep's fold(older word, composite) steps peel FRONT-to-back, so
   such sentences emitted back-to-front and every where-stamp
   mis-landed. Fixed with the head/pair/tail law (call 5); pinned.
2. **(medium) Ungated slab stash** — the render-source swap reached the
   non-wordstore free-derivation ceiling. Fixed: stash word-rows-gated
   inside the un-fold; both RUN_SLOW grammar pins re-run green.
3. **(medium, mutation-verified) minimal-residual pin was one-sided** —
   a parent-echo residual passed. Fixed: the one-line contract assert +
   a partial-overlap dim that separates minimal from subtractive.
4. **(medium, mutation-verified) the ``_support_dot`` denominator had
   zero coverage** — a raw-dot mutant passed every pin. Fixed: the
   big-norm decoy (raw dot 10.0 wins; support-dot 1.41 loses).
5. **(medium, mutation-verified) the pair-order tie-break was
   untested** — set-comparisons could not see it. Fixed:
   ``test_word_pair_snap_orders_by_side_dot``.

## Tests

``test_word_store.py`` (26): ``test_word_pair_snap_recovers_folded_
pair`` (exact pair from its radmax fold),
``test_word_pair_snap_orders_by_side_dot`` (x1 = higher one-sided dot),
``test_word_side_snap_support_match_beats_l2`` (the ADJ(N)
discrimination pin: dot-over-support beats L2 AND the big-norm raw-dot
decoy), ``test_word_side_snap_minimal_residual`` (the exact residual
contract, partial-overlap dim), ``test_unfold_word_word_step_snaps_to_
store_rows`` (dispatch: a word∧word trace step emits exact store rows),
``test_unfold_sweep_shape_emits_left_words_first`` (the head/pair/tail
emission law).

Suites: word_store 26 / bounded_stm_fold / reconstruction_roundtrip
(incl. both RUN_SLOW grammar pins re-run green) / stm_recon /
streaming_ar all green. Final gate: ``make test`` =
**1 failed / 3301 passed / 49 skipped / 6 xfailed** — the single
failure is the pre-existing ``space_equiv_selfcheck`` width mismatch
flagged for a separate session (four-fronts doc; baseline was 1/3295 —
the +6 are this arc's pins).

## Grammar-driven reverse — Alec's 3-step arc (2026-07-14 second pass)

> Directive (Alec): the snap short-circuits the GrammarLayers on decode.
> (1) op-respecting selection — conjunction may want DIFFERENT codebook
> selection from disjunction; recovery back inside the layer reverses.
> (2) the free-derivation reverse must find its OWN derivation via the
> ``reverse()`` GrammarLayer methods — NOT the forward record — with the
> reverse operation set DETERMINED FROM THE GRAMMAR FILE (``<generate>``),
> as the forward set comes from ``<compose>``. (3) Method-1 (exact
> teacher) should inform Method-2 (root separability).

Confirmed by a full reverse-machinery map (agent, cached): the forward
pipeline is ``<compose>`` rules $\to$ host layers $\to$ ``_stm_reducer``
(``BinaryStructuredReductionLayer`` over the arity-2 ops) $\to$ anchor-dot
chooser $\to$ recorded trace. The **reverse had the first two pieces and
nothing after** — Method-2 READ the op from the recorded trace
(``marg.argmax``). No reverse reducer, no reverse chooser.

**Step 1 — op-respecting selection, recovery inside the GrammarLayers
(LANDED).** ``Ops.word_pair_snap`` gains ``priming`` and, for
``op_name != 'union'``, a distinctness tie-break: the MEET is lossy
(``radmin(a,b)`` collapses to ``a`` when ``a`` is dominated — the fold
fit is uninformative, every candidate agrees on the parent's support), so
conjunction's selection LEANS ON PRIMING (which words are present) where
disjunction's join is fit-determined and needs none. ``snap=True`` on
``Ops.conjunctionReverse``/``disjunctionReverse`` (via a shared
``Ops._word_snap`` that indexes priming to the restricted rows) routes the
op's reverse through the snap; threaded through the four lattice layer
``reverse`` methods (``Union``/``Intersection`` CS, ``Conjunction``/
``Disjunction`` SS). Recovery now lives inside each layer, dispatched by
WHICH layer. Pins: ``test_intersection_snap_priming_recovers_present_
words`` (the asymmetry), ``test_union_snap_needs_no_priming``,
``test_grammar_layer_reverse_snap_is_op_respecting``.

**Step 2 — trace-free grammar-driven reverse derivation (LANDED).**
``_grammar_reverse_ops`` enumerates the arity-2 reverse ops from
``<generate>`` (``.reverse`` in canonical, host ``arity==2``,
snap-capable) — symmetric to ``_stm_reducer``'s ``<compose>``
enumeration; on this config the pair is ``disjunction`` (radmax) /
``conjunction`` (radmin), the SS lattice the reducer actually folds with.
``_reverse_choose_op`` is the REVERSE CHOOSER with NO forward record: for
each grammar reverse op it snap-reverses the parent, RE-FOLDS forward, and
scores the round-trip cosine to the parent — the op whose fold best
reconstructs the parent is the one that explains it (the trace-free
analogue of the forward anchor-dot chooser). ``_reverse_reduce_unfold``
now BRANCHES: with word rows + reverse ops it runs ``_reverse_derive_words``
(bounded frontier derivation, choosing the op per un-fold step, word-leaf
snapping); legacy configs (no word rows) keep the recorded-trace walk.
Measured E=38 decode (no trace): recovers the 2 words per sentence
(``lovingthere``), ``where`` 0.1667, recon byte-identical — a faithful
replacement for the trace-walk. Pins: ``test_grammar_reverse_ops_from_
generate_section``, ``test_reverse_chooser_picks_fold_op_by_roundtrip``,
``test_unfold_uses...``→ rewritten trace-free, ``test_unfold_trace_free_
recovers_two_word_root``. The trace-walk-internals pins
(``filters_non_word_folds``, ``sweep_shape_emits_left_words_first``) were
retired — the trace mechanism they pinned is gone.

**Bounded to 2 words — the honest scope.** The chooser's snap returns
store rows, so every un-fold operand is a word leaf; the flat/2-word
corpus bottoms out in one step. A deeper tree needs the root to un-fold
into a word + a SEPARABLE composite child, which the collapsed root
cannot yield — the same root-separability wall, now the sole blocker for
BOTH multi-word derivation and exactness. That is step 3.

**Step 3 — Method-1 $\to$ Method-2 leaf distillation (LANDED,
default-off; measured NULL at tested doses; diagnosis SHARPENED).**
Mechanism: ``<training><leafDistillWeight>`` (XSD-declared, default 0.0
= byte-identical) enables a ``LeafDecoderHead`` (shared tanh trunk +
per-slot embedding, $\approx$ 0.5M params — the anchors-over-MLPs
preference) trained by masked MSE to regenerate the EXACT Method-1
leaves (``_stm_pre_reduce_slab``) from the collapsed root
(``_stm_single_S``); lazily built, registered as a child, and handed to
the LIVE optimizer via ``add_param_group`` (the optimizer predates the
lazy build). Pins: ``test_leaf_distill_default_off`` (no term, no head)
+ ``test_leaf_distill_trains_root_toward_leaves`` (term fires, head
optimized). Verified live on the wordstore matrix config (term 0.0168,
root graph present).

**Measured:** E=38 at weight 0.5 AND weight 5.0 — root pairwise diff
0.00017 (byte-level unmoved from baseline), decode unchanged. Zero
dose-response $\Rightarrow$ structurally blocked, not under-tuned. The
decisive probe: **at the seal sweep the STM is already depth 1** (the
mid-read folds collapse everything during reading) and the cross-row
diff is ALREADY 0.000171 there, with the C$\to$S$\to$C roundtrip a
passthrough on quantize configs — so the collapse is complete before
any fold policy or root objective can act: **the per-word CS ideas
themselves contract to one attractor under training.** "Root
separability" is really PER-WORD IDEA separability. The distillation
gradient would have to reshape the whole encoder pathway against the
main objective — the wrong lever alone.

**The sharpened design call (Alec):** anchor idea separability at the
per-word grain — e.g. an idea $\leftrightarrow$ percept consistency
term (each word's CS idea tied to its DISTINCT PS percept row,
extending "a percept's vector position IS its identity" one level up),
or a contrastive term over per-word ideas; the landed distillation then
has separable material to work with (folds differ $\Rightarrow$ roots
differ $\Rightarrow$ the head's target becomes achievable and Method-1
$\to$ Method-2 transfer can bite).

**Arc gate:** word_store 33 + reconstruction_roundtrip +
bounded_stm_fold + primed/heat/boolean reverse suites green;
``make test`` = **1 failed / 3306 passed / 49 skipped / 6 xfailed** —
the single failure remains the pre-existing ``space_equiv_selfcheck``
width mismatch (re-confirmed in isolation; flagged for its own
session). The GrammarLayer forward/reverse INVENTORY (all ops, verified
from code) now lives in ``doc/Language.md`` §"GrammarLayer forward /
reverse inventory", with the ``RuleDef`` field list corrected and the
``<compose>``/``<generate>`` symmetry + generate-rule arity asymmetry
documented.
