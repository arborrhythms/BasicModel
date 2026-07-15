# Four-front todo execution — cross-batch graph severing, depth-3 flip, word-grain attention exercise, E=3 leak closure

> Directive (Alec, 2026-07-13): implement the four action items in
> todo.md. Parent context: 2026-07-13-word-grain-open-fronts.md.
> Environment: cpu/eager, seed 0 throughout (device policy).

## §4 Test hygiene — E=3 dynamics leak: NOT REPRODUCIBLE AT HEAD (closed)

Two independent full 9-file composition runs (compile_static_loop,
conceptual_recurrence, config_matrix, dimensional_governance, dual_towers,
per_word_capture_gate, per_word_ss_padding_noop,
rule_gate_isolates_side_effects, reconstruction_roundtrip) are GREEN at
head: **71 passed / 8 skipped** both times (253.6s / 251.9s), with
`test_mm20m_xor_roundtrip_at_harness_budget` passing INSIDE the
composition. The leak recorded at handoff does not reproduce on the
current tree — the third-pass changes (the capture-gate dynamo-reset
fixture plus the same-day test-file edits) removed or shifted whatever
carried it. The bisect harness (phase-1 confirm $\to$ binary search
$\to$ pair check $\to$ drop-one delta debug) is parked in the session
scratchpad (`bisect_e3_leak.py`) should it ever resurface.

## §1 Method-2 exactness — the ladder5 crash, the radial re-baseline, and the snap gap quantified

**The relaunch crash (why ladder5 never reported).** The in-flight
relaunch died on its FIRST batch of epoch 1 with the version-counter
crash (`[1016] at version 1; expected 0`). Root cause (anomaly +
version-ledger + graph-path probes): the serial arcs left PLAIN-ATTRIBUTE
tensors carrying live autograd graph across the batch boundary — the STM
`_live_buffer`, the router's `_last_output`/`_last_root_state`,
`routing_state.rule_probs`, `_stm_last_reduce_routing`/`_stm_single_S`,
`_intent_boosts`, `_stage0_recon_loss`, and the CS-bind
`_bind_context['slab']`. `_detach_persistent_state` (the established
severing discipline, run by `post_tick_compact` after every tick) only
covered registered buffers + a fixed attribute list, so epoch $N{+}1$'s
forward re-read epoch $N$'s graph; the optimizer step in between had
bumped the saved `d` (LDU diagonal) version $\to$ backward crash from the
second epoch on. **Fix:** step 4 of `_detach_persistent_state` severs the
serial-arc carriers (module walk + the host-layer registry). Pinned:
`test_word_store.py::test_two_epoch_training_severs_cross_batch_graph`
(two epochs on the wordstore config + a carrier audit; RED pre-fix at
exactly the epoch-1 crash).

**ladder5 (radial substrate, measured BEFORE the §2 conditioning chain
landed):**

| E | exact | where_rec | recon |
|---|---|---|---|
| 3 | 0.0 | 0.16667 | 0.337989 |
| 38 | 0.0 | 0.0 | 0.094532 |
| 80 | 0.0 | 0.0 | 0.089844 |

The radial fold is confirmed as a TRAINING change (recon re-baselines
from the lattice 0.048 family to 0.34 at E=3), and the E=1 bench's first
nonzero placement (0.1667) survives at E=3 but unbinds at trained budgets
ON THAT TREE.

**ladder6 (intermediate tree — measured BETWEEN the §2 conditioning
chain and the review's anchor-gating; SUPERSEDED):** E in {3, 38, 80}
read exact 0.0 / where 0.16667 / recon {0.001944, 0.001903, 0.001903} —
the UN-gated CS scan let chance-fired relative ids trigger mid-read
protection on this anchor-less corpus, a large (accidental) training
change. Kept only as evidence of what the depth-3 protection does to
training dynamics where it engages; NOT a valid re-baseline.

**Final-tree re-verification (2026-07-14, post anchor-gating): the
ladder is BYTE-IDENTICAL to ladder5** — E=3: 0.0 / 0.16667 / 0.337989;
E=38: 0.0 / 0.0 / 0.094532; E=80: 0.0 / 0.0 / 0.089843. The
anchor-gated chain is exactly conservative on anchor-less corpora (the
grammar bar's byte-identity contract), so **ladder5 IS the radial
re-baseline** and its trained-budget placement unbinding stands.

**The snap gap, quantified (the "chase").** Hooking the un-fold's
emissions and scoring each recovered operand against the stored word rows
(cosine over the `.what` slice): every emitted operand sits at best-cos
**0.8711** — IDENTICAL at E=1 and E=38, and RE-VERIFIED at 0.8711 on the
final anchor-gated tree (2026-07-14) whose training trajectory differs
by two orders of recon, i.e. a STRUCTURAL offset, not trained drift; 0%
reach $\ge 0.99$. This sharpens the parked design call
(the SIGNED-SPACE SNAP, parent plan §NEW BINDING FRONT): a
nearest-word-row cosine snap binds every operand at a $\approx 0.87$
threshold; the residual $\approx 13\%$ misalignment is the fold's own
contribution, constant across training. Exactness cannot move until the
snap (metric + max-fold residual rule in signed space — Alec's call)
replaces the order-filter recommender at idea grain.

**Ceiling pin:** the exact bar did NOT move (0.0 at every rung) — no
re-pin of `test_mm20m_grammar_free_derivation_ceiling`.

## §2 Depth-3 endpoint fidelity — the routing campaign FLIPPED

Campaign harness: build MM_query_reasoning, provision (anchors register
at each text's boundary Reset), run train epochs; per epoch record the
router's fired rule ids (per space_role), the relative mask by call site,
and the post-sweep depths (`campaign_depth3.py` in the session
scratchpad).

**Baseline (pre-change):** anchors registered ({16: 'part'}) but the
chain was dead end-to-end — relative rules NEVER fired, mask 0, every
sweep collapsed to depth 1. FOUR named defects, each found by
measurement and fixed test-first:

1. **Category conditioning structurally unreachable on word-grain
   configs.** `_build_category_context` returned None on EVERY call:
   its precondition `ws._category_last_pid` has a single writer at the
   TAIL of `_maybe_autobind_meta`, and the mereologyRaise word-whole arm
   RETURNED before reaching it — on every mereologyRaise config the
   chooser never saw any category context (anchored or otherwise).
   Fix: the category block extracted VERBATIM to module-level
   `_stash_category_roles` (the `_concept_rows_exist` duck-type
   convention) and called from BOTH arms.
2. **The relative mask scanned the wrong space_role.**
   `sentence_relative_mask` read only `current_rules['SS']` — but the
   relative producers (part/whole/equal, global ids 1/2/3) are CS-ROLE
   rules, so their fired ids land under `'CS'` while `'SS'` carries only
   the id-0 stem padding. Measured: relative id 3 fired under 'CS' with
   the mask still False. Fix: scan SS and CS, OR-ing per-key masks
   (`is_relative_rule` keeps the conservatism).
3. **The anchordot chooser was blind to the anchored one-hot.** The
   anchor short-circuit labels the closed-class slot with the operator's
   OUTPUT role (`<op>_O1`), but `_category_reduce_prior` read only
   I1/I2 columns — the anchored slot contributed exactly 0. Fix: an O1
   term credits pairs touching an anchored slot toward the operator's
   own rule.
4. **The mid-read folds ignored the protection.** The 2b-2 opportunistic
   reduces ran without `protect_depth`, so a relative sentence was
   already collapsed below depth 3 BEFORE the protected sweep (measured:
   sweep-site mask firing 3/3 with post-depths still 1). Fix: the
   per-word fire populates THIS sentence's rules, so the folds now
   compute the same per-row depth-3 floor eagerly (compile has no
   per-word rules to read — the fire itself is eager-only — so the
   compiled path is untouched).

**Measured post-chain (campaign, FINAL anchor-gated form):** the
provisioning texts themselves still collapse to depth 1 — anchors
register at their boundary Resets, so engagement starts from the NEXT
read, exactly the directive's framing — and the FIRST trained read then
ends **{3: 6}** (all six rows at the depth-3 relative end-state; the
sweep-site mask fires 1/1 with relative ids visible at the sweep).
Later epochs vary with the evolving chooser (some epochs fire no
relative rule and collapse to 1) — the TRAINED router's determinism is
the remaining refinement; the escalation candidates if it must be
deterministic: a config/learnable prior gain, `<transformChooser>mlp`
(consumes cat\_ctx directly), or the anchor-driven mask short-circuit.

Pins: `test_stm_relative_sentence_end_state.py::
test_relative_mask_sees_cs_role_rules` + `::test_category_prior_scores_
anchored_operator_slot`;
`test_thinking_kernel.py::TestDepth3RelativeEndState` (fresh-model
provisioning reaches a depth-3 end-state + the pid-grid stash engages).

**Known residuals recorded (not blocking, from the mapping pass):** the
cat\_ctx pid grid is stale by one sentence (single writer at the
boundary Reset) and indexed in word order against a newest-at-0
snapshot (ends swapped; the 3-word NP-R-NP anchor sits at the invariant
middle slot, which is why conditioning works on the syllogism corpus);
cat\_e threads into round 0 only.

## §3 Word-grain attention exercise — modes exercised, surface mapped

**Wiring pins** (`test/test_word_grain_attention.py`, the
heat-reverse-wiring harness idiom with a REAL LiftLayer at SS role over
the word-grain gates): primer fires `retrieval_candidates_for_slot` on
the Lift family (open-fronts Task C widening) and carries the typed
rows; attention off + wordStore on restricts the reverse to the WS
word-whole rows (THE Task-C SS-side pin); both-off = plain reverse; a
successful heat retrieval takes PRECEDENCE over the word rows (the
`'left_rows' not in reverse_kwargs` gate); a retrieval FAILURE degrades
to the word-rows restriction (generation never breaks); the REAL
LiftLayer.reverse completes under priming kwargs.

**E2E on the real config:** `<attention>primer</attention>` injected
into WholeSpace + ConceptualSpace of a scratch MM\_20M\_grammar\_wordstore
variant parses and stamps both spaces; the free-derivation bench runs
crash-free and BYTE-IDENTICAL to the off baseline (final tree,
re-verified 2026-07-14: exact 0.0 / where 0.16667 / recon 4.211482 on
both, zero retrieval calls) — the bar-protection contract holds.
**Measured routing finding:** the free-derivation decode routes through
the serial STM un-fold (`_reverse_reduce_unfold`, which reads the PS
word store and never consults `attention_mode`), and the boundary
generate fire (`ss.reverse(snap)`) also does not traverse
`LanguageLayer.unreduce` on this config — so the reverse-side priming
surface is dormant on the word-grain DECODE ROUTE, not just ungated.
Live E2E engagement needs (a) a decode route that traverses the
stack-mode grammar reverse, and (b) a production `attach_knowledge`
call (today `ss.knowledge` is always None, so retrieval returns `{}`
even when the mode is on).

## Adversarial review pass (multi-agent, post-implementation)

Confirmed findings, all addressed:

1. **(high) Unprotected back-pressure fold** — the capacity fold at the
   same per-word site ran without the depth-3 floor; the batch-global
   `_max_depth_host` mirror can pin at capacity and force-fold a
   FINISHED relative row on every later column. Fixed: the floor is
   computed once per word (from the rules the previous fires populated)
   and threaded into the back-pressure fold AND the two opportunistic
   reduces; depth $\le 3$ rows are never the rows actually at capacity,
   so protection cannot block a genuinely forced fold.
2. **(medium) CS-scan false positives** — fired rule ids are
   Viterbi-argmax picks at EVERY reduce site, so an untrained chooser
   fires a relative op on ABSOLUTE sentences at chance rate. Fixed: the
   CS scan is ANCHOR-GATED — a row flips only when a relative id fired
   AND the row's pid evidence (the category grid OR the current
   sentence's PS forward-stash indices, which covers the
   first-presentation promotion window) contains an anchored
   closed-class pid. No anchors / no grid $\Rightarrow$ the CS scan
   contributes False. This restores the documented conservatism
   (anchors are grammatical, not learned).
3. **(medium) eager-vs-compile divergence of the mid-read protect** —
   ACCEPTED + documented: the per-word fire itself is eager-only
   (compiler-disabled island), so under compile the per-word rule
   signal does not exist and mid-read protection cannot apply; the
   divergence is inherent to the fire's own semantics, not the floor.
4. **(high) the 2-epoch severing test repointed the process config
   mid-module** — moved to the END of test_word_store.py with the
   repoint documented; **(medium) vacuous graph-free asserts** — the
   key carriers must now EXIST before the graph-free checks.

## Pre-existing failures surfaced by the first full `make test` since the third pass

Proven pre-existing by runtime A/B (all session changes neutralized —
still red). Repaired in-session:

* **test_two_pass_driver (2)** — the SAME carrier crash class at the
  intra-tick pass boundary: pass A's optimizer step moves saved-tensor
  versions, pass B's forward re-read the carriers. Fixed:
  `_detach_persistent_state()` now also runs between pass A and pass B
  (the post_tick_compact discipline applied intra-tick). 6/6 green.
* **test_router_fires_per_word (2)** — STALE PINS: they still asserted
  the Task-3 (2026-06-05) "per-word compose fire is deleted" contract,
  superseded by the 2026-07-13 per-word fire directive. Re-pinned to
  the live contract (compose fires from the second word on under
  'per-word'/'both'; 'off'/'boundary' stay compose-free). 7/7 green.

Flagged for a separate session (NOT fixed here — outside the four
fronts, needs domain judgment):

* **test_space_equiv_selfcheck::test_identity_candidate_passes** —
  `MeronymicFoldAdapter._bfly_flatten: vector width 14 exceeds cascade
  M_total=8` (the harness appears to feed the muxed event where the
  production path folds the `.what` slice; other BPE paths are green).

## Suites

word_store 20 / mereology_word_binding 10 / thinking_kernel 66+1 /
stm_relative_sentence_end_state 17 (incl. the 2 new) /
relative_sentence_codebook_insertion / router_fires_per_word 7 /
two_pass_driver 6 / bounded_stm_fold / heat_reverse_wiring 8 +
word_grain_attention 6 / streaming_ar_training /
reconstruction_roundtrip — all green during the session. Final gate:
`make test` = **1 failed / 3295 passed / 49 skipped / 6 xfailed**
(15:50) — the single failure is the flagged pre-existing
space\_equiv\_selfcheck width mismatch above (was 5 failed / 3291
passed before the session's repairs).
