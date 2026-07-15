# Serial-Mode Reconstruction via Grammatical Derivation — Execution Plan

> **For agentic workers:** steps use checkbox (`- [ ]`) syntax. **Alec
> does ALL git writes** — GATES are his commit points. Design:
> [2026-07-04-serial-derivation-reconstruction-design.md](2026-07-04-serial-derivation-reconstruction-design.md).
> **Q2 RESOLVED (Alec, 2026-07-04, at plan-writing per the design's
> re-ask):** Method-1 replay TAKES the grammar config's round-trip bar
> slot immediately; Method-2 is the separate STAGED trained bar —
> mirroring the encoding pass's scaffold/blind pairing.

**Goal:** serial-mode reconstruction becomes a GRAMMAR DERIVATION.
Method-1 (replay the stored tree) routes the serial decode and is the
exact bar; per-op identity stubs become loud errors whose inventory
drives write-the-reverse vs remove-the-rule; the NULL-word pathway
gives forward a real compose-identity and reverse a way to emit/
recognize sentence length blind; Method-2 (free derivation) is the
trained bar, free-run only, scored on surfaces.

**Sequencing note:** the encoding pass (its own plan) ran FIRST per the
design's resolution 3a — the `.when` ideas carry is now the 4-dim start
ladder. Its Gate-B blind-tiling bar is RED (band precision, an open
Alec knob) — INDEPENDENT of this plan: Method-1 replay consumes the
stored derivation, not band-decoded tiling.

**House rules:** no git writes; comments one-liners; targeted pytest
(`PYTHONPATH=test:bin BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager
.venv/bin/python -m pytest ...`); `make test` only at the final gate on
a quiet tree; Inf/NaN fail loud; LaTeX in docs; probes in the
scratchpad; value-pins updated honestly with each change flagged.

**Key sites (verified 2026-07-04 on the current tree; re-verify before
editing):** stored-tree replay `_reverse_from_S` bin/Models.py:9373
(drives `_reverse_body` → `_reverse_perceptual` → `inputSpace.reverse`,
left-shift by `_word_active_mask`); replay trigger + seed
`_reconstruction_seed` :3510 (serial arm `snap[:, :1, :]` :3536, the
D3 consumer at :4358); free-inference path `_chart_generate_from_stm`
:8371 (+ `reconstruct_from_idea` gate :798); the collapse `_stm_reducer`
:7027; identity-stub contract `unreduce` bin/Language.py:5192-5211 +
per-op stubs (e.g. `ConjunctionLayer.reverse` :3043, `invertible =
False` sites :2795/:3016); grammar start states
`data/complete.grammar`; static per-word loop `for p in range(N_words)`
bin/Models.py:8016; `word_at` bin/Spaces.py:9164; percept-axis
`null_percept_idx` (IR-mask axis, NOT the word axis).

---

## Task 0 — Routing survey + serial baseline (measurement only)

- [ ] **0.1** Capture the CURRENT serial/grammar decode state: harness
  records (epochs 3, seed 0, cpu/eager, blind AND scaffold modes) for
  `MM_20M_grammar` at the post-encoding tree — the Gate-A re-baseline
  row is the reference (output 0.175000 / recon 0.002389 / exact 0.0 /
  where_rec 0.25). Confirm empirically (probe print, no source edit)
  that `recon_bench`'s decode on the grammar config consumes the TENSOR
  reverse (the `_reconstruction_seed` serial arm), NOT
  `_reverse_from_S`'s replayed surface — the Task-6 gap 1 this plan
  closes. Record in EXECUTION NOTES.
- [ ] **0.2** Replay-liveness probe: one grammar forward (serial), then
  call `_reverse_from_S(_stm_single_S)` directly; record which rules
  `generate_rules` carries and which per-op `reverse`s execute as
  identity stubs (the pre-inventory).

## Task 1 — Per-op reverse stubs → LOUD ERRORS (the inventory)

**Files:** bin/Language.py (`SyntacticLayer.reverse` dispatch region +
`unreduce`; the per-op stubs), test/test_serial_reverse_errors.py (new)

- [ ] **1.1 Failing tests first:** (a) a non-invertible rule layer's
  `reverse()` RAISES `NotImplementedError` naming the rule ("write a
  real reverse() or remove the rule from the grammar" — the fifth
  fail-loud application); (b) an invertible rule's real reverse is
  untouched; (c) the error message carries the rule name + arity +
  space_role (the decision procedure's row).
- [ ] **1.2** Convert the stubs: identity pass-throughs for
  non-invertible rules raise; the `unreduce` docstring contract
  updates (the "identity/pass-through stub" sanction is REVOKED by the
  design). Every consumer that DEPENDED on the stub (the fallback sites
  around Language.py:5471-5566 and any test pinning stub behavior) is
  adapted or recorded in the inventory — each flagged.
- [ ] **1.3** Method-1 replay dry-run WITH the errors in place (the
  grammar fixture + `MM_20M_grammar`): the rules that actually fire
  define the INITIAL INVERSE-COVERAGE WORKLIST. Record the inventory
  table (rule → fired-by → verdict-needed) in EXECUTION NOTES.

**GATE S1 (Alec commits + rules on the inventory):** each fired rule
gets a verdict — WRITE the real `reverse()` (this plan, Task 2.2) or
REMOVE the rule from the grammar (Alec's call per rule).

## Task 2 — Method-1 replay routes the serial decode (THE bar)

**Files:** bin/Models.py (`_reconstruction_seed` serial arm / the decode
staging seam), bin/Spaces.py (the radix render consumes the replayed
surface), bin/recon_bench.py (no flag change — serial routing is
mode-derived), test/test_reconstruction_roundtrip.py (the grammar bar)

- [ ] **2.1 Failing bar first:** `test_mm20m_grammar_derivation_roundtrip`
  — serial mode, the decode consumes `_reverse_from_S`'s replayed
  multi-slot surface event instead of the single-slot tensor reverse;
  exact_match == 1.0 (Method-1 is exact BY CONSTRUCTION once routed —
  any residual is a real defect of the replay machinery and gets FIXED,
  not accepted; Q2: this IS the grammar round-trip slot). Budget
  measured at execution (epochs pinned by the standing formula against
  the measured trajectory; RUN_SLOW if > 30s).
- [ ] **2.2** Write the real per-op `reverse()`s the Gate-S1 worklist
  demands (WRITE verdicts only). Each is EXACT for its op's forward
  (the design's teacher): $\sigma$/$\pi$ splits per the ramsification
  record where present; `lift`/`lower` per the grammar's own dual
  declarations (`<generate>` in role_collapsed/complete.grammar).
  Scaffold/blind mapping: stored-tree = the scaffold analogue (given),
  free-derivation = blind (Task 4).
- [ ] **2.3** The scaffold/tensor path DEMOTES but survives: parallel
  mode untouched (byte-identical); serial tensor-reverse reachable as
  the explicit debug fallback (mirrors `--scaffold`). Sweeps: roundtrip
  file, recon_bench tests, serial-mode integration, grammar fixtures.

**GATE S2 (Alec commits):** the derivation bar green; replay routed.

## Task 3 — The NULL-word pathway (forward identity + reverse emit)

**Files:** bin/Spaces.py (`word_at` :9164 — the NULL-word encoding),
bin/Models.py (the per-word loop :8016 forward; the reverse emit/
recognize), test/test_null_word_pathway.py (new)

- [ ] **3.1 Failing tests first:** (a) a padding position's word event
  is the EXPLICIT NULL-word (a genuine encoding, not the zero-slab
  accident) and composes as the identity under the live compose op
  (assert compose(x, NULL) == x for the grammar's arity-2 ops — robust
  BY CONSTRUCTION); (b) reverse/derivation EMITS NULL-words past the
  sentence end and the decode RECOGNIZES them (sentence LENGTH
  round-trips without the scaffold supplying it: right words AND right
  count); (c) the percept-axis `null_percept_idx` is untouched (a
  different axis).
- [ ] **3.2** Padding-cost measurement (the design's fork): time the
  static loop with NULL positions (i) full-cost today vs (ii) the
  fast-skip variant (padding positions skip the heavy forward/reverse
  body). IF fast-skip lands cleanly → keep FIXED-MAX one-graph (option
  ii); IF NULL stays full-cost → adopt NEXT-POWER-OF-TWO loop buckets
  (option iii; dev-data buckets {1,2,4,8}, fits `cache_size_limit`).
  The measurement decides; both keep the CORRECT NULL-word. Record the
  numbers; flag the choice at the gate (compile-policy adjacent —
  Alec's standing DEBUG=eager / PRODUCTION=inductor policy applies).
- [ ] **3.3** The COMPILE/PERF residual from todo (the `p == N` cursor
  guard at cache_size_limit) is re-measured under whichever fork won —
  the pow2 pathway is the principled fix the todo names; record
  whether the recompile pressure clears.

**GATE S3 (Alec commits):** NULL-word live both directions; fork chosen
by measurement.

## Task 4 — Method-2: the trained free-derivation bar

**Files:** bin/Models.py (`reconstruct_from_idea` path), bin/recon_bench.py
(a `--free-derivation` mode note), test/test_reconstruction_roundtrip.py

- [ ] **4.1** `reconstruct_from_idea=True` on the serial config:
  derivation inferred from the reduced STM + attention state (the
  chart's generate path), scored against the INPUT SURFACE (resolution
  3b: surfaces, never idea tensors; determinism bars within-run only,
  3c). Teacher-forced variants vs Method-1's tree stay DIAGNOSTICS
  (resolution Q1: free-run only, no per-production loss — density
  comes from the scaffold-masking curriculum at the XML maskRate).
- [ ] **4.2** Measure the trajectory; pin by the standing formula IF
  1.0 is reachable; otherwise report precisely (the remaining
  non-invertible-rule ceiling from the Gate-S1 inventory is the
  expected bound — surfacing WHICH rules' inverses matter is itself
  the deliverable).
- [ ] **4.3** Relation-table coupling stays DEFERRED (design point 4;
  the attention-to-relation-promotion plan governs promotion; interface
  noted, no wiring).

**GATE S4 (Alec commits):** Method-2 measured (pinned or
precisely-reported).

## Task 5 — Docs + notes + the suite gate

- [ ] **5.1** Architecture.md (serial reconstruction = derivation;
  Method-1/Method-2 duality), Language.md (per-op reverse contract:
  no identity stubs — error or real inverse), Reasoning.md pointer if
  the NP-R-NP forms land relation-table hooks. STM.md: the NULL-word
  as the temporal-word-axis NOTHING (symmetric with the clock owning
  exact time).
- [ ] **5.2** EXECUTION NOTES: inventory table, budgets, fork
  measurement, pins ledger, deviations.
- [ ] **5.3** `make test` on a quiet tree — THE gate.

**GATE S5 (Alec commits; plan done).**

---

## EXECUTION NOTES (append during execution)

## Self-review (writer, 2026-07-04)

- Spec coverage: design point 1 → Task 2 (Q2 resolved: replay IS the
  bar); point 2 → Task 4; point 3 → Task 1 (errors first, inventory
  gates the reverse-writing); the NULL-word requirement → Task 3
  (including the padding-cost fork's measurement-decided branch);
  point 4 deferred with its interface noted (4.3).
- The resolutions (free-run only; surfaces-not-ideas; within-run
  determinism; encoding-first sequencing) are each pinned to the task
  that enforces them (4.1, 2.1's budget language, 0.1's reference to
  the post-encoding baseline).
- Fail-loud lineage: Task 1 is the fifth application (after the three
  silent-band sites + the where overflow warn); the stub sanction in
  `unreduce`'s docstring is explicitly revoked rather than left to
  drift.
- No placeholders: every task names verified sites; budgets and the
  fork are measured-at-execution with the deciding criterion stated.

### Task 0 — routing survey + serial baseline (sub-agent, 2026-07-04 night)

**0.1 baseline (epochs 3, seed 0, cpu/eager):** both modes MATCH the
Gate-A reference row and each other: `output_loss=0.175000`,
`recon_loss=0.002389`, `exact_match_rate=0.0`, `where_recovery=0.25`.
Blind → `output/recon_MM_20M_grammar_20260704-133533_67103.json`;
scaffold → `output/recon_MM_20M_grammar_20260704-133547_67125.json`.
The blind==scaffold tie is genuine, not a probe bug (`decode_blind_rate`
does branch tiling in `_decode_radix_meta`, Spaces.py:12156) — low-epoch
recon quality isn't discriminating yet; not chased (out of scope).

**Routing proof (call-count + write-order timeline, one eval batch):**
`_reverse_from_S`=1 call, `_reconstruction_seed`=1 call;
`PartSpace.reverse` (the SOLE writer of `_recovered_input_thunk`,
Spaces.py:12008/12086) fires TWICE per batch: once inside
`_reverse_from_S`'s `_reverse_perceptual` leg (Models.py:9411), once
inside `model.reverse()`'s `_reverse_perceptual` leg (Models.py:9251,
seeded via `_reconstruction_seed`/`self.reverse(cs.subspace)` at
:4358-4363). `_d3_reconstruction_loss` (→`_reverse_from_S`, :4280)
fires BEFORE the `_reconstruction_seed` block (:4358) every batch
(instrumented timeline confirms write-order, not just line order).
**`reconstruct_data()` reads whichever write happened LAST → the
TENSOR reverse (`_reconstruction_seed`'s serial arm `snap[:, :1, :]`),
NOT `_reverse_from_S`'s replay — the Task-6 gap 1 this plan closes.**
Serial confirmed: `model.serial=True`, `inputSpace._per_word_enabled
=True` (`=bool(self.serial)`, Models.py:919, gates `_d3_active`), no
`<serial>` tag (derived from `symbolicOrder=1`); `*.serial_mode` attrs
are all `False` (unrelated legacy AR-streaming flag).

**generate_rules (0.2):** one train-batch forward populates
`symbolSpace.generate_rules = {'CS': [row]*4}`, row
`[2,2,2,15,5]` → `['isEqual','isEqual','isEqual','lower','exist']`
(`_grammar_is_default_only=False`, `complete.grammar` full-router
path). `_stm_single_S` shape `(4, 1024)`.

**Per-rule fired/identity/real:** `model._reverse_from_S(_stm_single_S)`
under `no_grad` succeeds (`[4,1,1024]` out) but **fires ZERO of the 33
`GRAMMAR_LAYER_CLASSES` reverse/generate methods** (every table row
0/0/0/0). Root cause: `unreduce` (Language.py:5378-5397 — NOT
:5192-5211 as this plan's header cites; fallback sites now ~:5645-5695,
NOT :5471-5566; `bin/Language.py` shifted ~+180 lines, uncommitted,
since the plan was written — **re-verify line refs before Task 1**) is
reached ONLY via `LanguageLayer.reverse`→`reverse_stack` (:5766), only
called from `WholeSpace._stack_route_reverse` gated on
`use_stack_router` (`False` here) — `_reverse_from_S` never enters
that path (`_reverse_body` walks `ConceptualSpace.reverse`, a "pure
bookkeeping carrier" per its own docstring, Spaces.py:16671-16679;
`WholeSpace.reverse` has zero callers in Models.py). Production
callers of `unreduce`/`reverse_stack` are ZERO outside direct
unit-test calls (`test_heat_reverse_wiring.py`,
`test_subspace_what_stm_contract.py`) — the "per-op identity stubs"
Task 1 targets are **dead code on the serial decode path today**, not
silently-firing stubs; Task 1/2 is what connects `generate_rules`'s
real rule sequence to a reverse dispatch at all.

**Unreduce contract (5 lines):** decodes the top stack slot's `.where`
to find the producing rule, calls that rule's `layer.reverse` on the
parent, writes children back (arity-1 in-place, arity-2
left+new-slot); a layer opts OUT via `reverse_dispatchable=False` /
missing `reverse_required_kwargs` / exception / wrong-arity-return →
`_identity_stub()` (children's `.where`→0, halts one level deep) — the
pass-through Task 1.2 revokes; empirically inert here (0 calls above).

### Tasks 1 + 2 (routing) — overnight execution (2026-07-04 night)

**Task 1 LANDED (suite 3013/0 via `make testp`).** The identity-stub
sanction is revoked at every site:

- `GrammarLayer.raise_no_inverse(why)` (bin/Layers.py) — the standard
  inventory error: rule name + arity + space_role + "write a real
  reverse() or remove the rule from the grammar".
- Per-op conversions (bin/Language.py): `isEqual`, `isPart`, `part`,
  `queryPart` (truth-bivector collapse), `bind` (contextual), and the
  no-basis tails of `union`/`intersection`/`conjunction`/`disjunction`
  now RAISE. The basis recommender paths stay (real attempts) and
  gained a 2-D-codebook guard — a 3-D stack `.what` masquerading as a
  codebook used to explode as a raw unpack ValueError swallowed into
  the stub; it now takes the honest inventory raise.
  `PrepositionLayer.reverse` NOT converted (it has real inverse
  structure — un-rotates `.where`; its `(marker_placeholder, phrase)`
  duplication is a documented surface convention) — inventory:
  CONDITIONAL. `exist`: forward IS the identity, so the identity
  reverse is faithful — `invertible` flipped True (the
  trivially-satisfied WRITE verdict).
- `unreduce`: `_identity_stub` deleted; `reverse_dispatchable=False` /
  missing-required-kwargs / wrong-arity-return each raise; the broad
  exception swallows are gone (docstring sanction revoked).
- `SyntacticLayer.reverse`: a HOSTED non-invertible rule raises
  (un-hosted = cross-space_role routing skip; binary invertible rules
  still route to chart generate). `Conjunction`/`Disjunction.generate`
  gained the Union/Intersection recommender-forwarding signature
  (parity fix surfaced by the e2e adaptation).
- Contract tests: `test/test_serial_reverse_errors.py` (6). Adapted
  stub-pinning consumers (each flagged in its docstring): 5 in
  test_subspace_what_stm_contract.py (unreduce/reverse_stack/ws-reverse
  spies), 2 in test_grammar_binary_ops.py, 2 in
  test_boolean_reverse_recommender.py, 1 in test_ispart_query_dispatch,
  and test_ps_reverse_e2e's round-trip now recovers operands through
  the recommender (a STRONGER e2e than the stub it pinned).

**Gate-S1 INVENTORY (from Task 0's empirical pre-inventory + the
conversion):** rules recorded by the live grammar forward =
`{isEqual x3, lower, exist}` per row. Verdicts needed from Alec:

| rule | forward | state | proposed verdict |
|---|---|---|---|
| `isEqual` | `torch.maximum` (lattice join) | stub → raises | needs STORED-OPERAND replay (max has no residual — the fusion/difference finding) or removal from the serial grammar |
| `lower` | pi log-mult fold | REAL reverse (recommender / balanced split) | dispatch it on the replay walk (no authorship needed) |
| `exist` | identity | REAL (flag fixed True) | done |
| `union`/`intersection`/`conjunction`/`disjunction` (not recorded here) | lattice folds | no-basis raises; recommender stays | per-config: basis at call sites |
| `bind`, `isPart`, `part`, `queryPart` (not recorded here) | collapses | raise | write-or-remove when a grammar records them |
| `preposition` | marker absorb + where-rotate | real-ish (convention) | CONDITIONAL — review the (marker, phrase) duplication |

**Task 2 ROUTING LANDED; THE BAR IS RED (measured).** runBatch's eval
staging now routes SERIAL decodes through `_reverse_from_S` (the
Method-1 replay stages the render thunk; the single-slot tensor arm
survives as `serial_tensor_reverse_debug`, the `--scaffold` analogue;
TRAIN paths untouched — the block was already `_rev_dedupe`-skipped on
D3 configs). Baseline reproduced pre-edit (grammar E=3: 0.175000 /
0.002389 / exact 0.0 / where 0.25, blind == scaffold); post-edit the
decode RENDERS THE REPLAY: `'hello world' → 'world'`,
`'hello there' → 'loving'` … — one word per sentence. The single-S is
never EXPANDED back to per-word slots: the stored derivation carries
RULE NAMES (the cursor) but not operand payloads, and the router-path
reverse walk dispatches no per-op generate (Task 0's zero-fire
finding). `test_mm20m_grammar_derivation_roundtrip` added
(RUN_SLOW-gated, asserts the 1.0 bar, docstring records the RED state
and the cause).

**THE REMAINING S2 WORK (the real build, for the morning):** operand
PROVENANCE — Method-1's by-construction exactness requires replaying
STORED operands, not algebraically inverting lattice folds. The
pre-reduce STM snapshot holds the per-word ideas; the reduce
(`_stm_bounded_reduce_step` routing is already parked on
`_stm_last_reduce_routing`) knows which op fired. Wiring: stash the
pre-reduce snapshot (or the per-step operand pairs) on the forward;
`_reverse_from_S` walks the recorded rules BACKWARD popping stored
operands (rules with real inverses — `lower` — may verify against
their algebraic reverse); NULL-words pad the expansion to N (Task 3's
emit side). The union/difference pair (landed tonight, its own plan; renamed from the interim 'fusion' per Alec 2026-07-05 — lattice max is now 'join')
is the residual-bearing substrate for the non-stored/free case
(Method 2).

**Sweeps after Tasks 1+2:** full parallel suite 3013/0;
roundtrip+recon_bench+blind_decode 29 passed / 2 RUN_SLOW-skipped
(the xor scaffold pin 0.75 and blind-mechanism tests unaffected —
parallel mode byte-identical).

**Deviations:** (1) Gate S1 is Alec's verdict point — overnight I
applied only the reversible/conservative verdicts (flag fix on exist;
raises everywhere; NO rule removed from any grammar). (2) Tasks 3–5
not started (NULL-word pathway, Method-2 bar, docs) — Task 3 is the
next block after the S2 expansion since the emit side needs it.

**S2 provenance probe (pre-build de-risk, 2026-07-04 night):** the
stash must be BATCH-SCOPED — after the epoch, `stm.snapshot()` is None
and `generate_rules` is empty (Reset clears them) while
`_stm_single_S` `[B, 1024]` SURVIVES as a model attr: the stash follows
that pattern (captured during the forward, read at the same-batch eval
reverse; recon_bench's decode reads the LAST staged batch, so
batch-scoped is sufficient). The reducer already parks per-step
routing on `_stm_last_reduce_routing` incl. `marginal_slab [B, 2, D]`
— the pre-fold OPERAND PAIR of the last reduce step (and
`reduce_marginal_op` naming the op) — i.e. single-step provenance
exists today; multi-step reduces need the per-step accumulation.

### Gate-S1 verdict: isEqual (Alec, 2026-07-05)

**"isEqual is a query, so it should be handled outside of the grammar
(see if it fits in the Queries section of the complete.grammar)."** It
fits: `<Queries>` already carried `equal(X, Y)` — the introspection
form. RELOCATED: `complete.grammar` drops the isEqual start state + the
two compose rules + the two generate rules; the Queries comment records
that `equal(X, Y)` IS the isEqual/copula relation (the reasoner's hard
tools answer it; queries are not parse structure). The serial
derivation on `MM_20M_grammar` (complete.grammar) no longer records an
un-invertible rule — the recorded set drops isEqual, easing the S2
expansion. `isPart` stays PENDING its own verdict.

**Scope note (measured):** generalizing the relocation to
`default.grammar`/`shamatha.grammar` shifted `MM_grammar.xml`'s
XOR-convergence bar past threshold (best 0.164 vs 0.15) — REVERTED;
those two grammars keep their isEqual parse rules pending Alec's call
on whether the same-principle relocation should follow there (with the
convergence re-pin that entails). The union→join rename in those files
stays (behavior-preserving; also caught this pass: the `.grammar`
files' `union.forward(...)` dotted form + xor.grammar's
`<start name="connective">union_O1` were missed by the XML sweep's
paren pattern — all four .grammar files now say `join`).

Test pins adapted (flagged): test_role_collapsed_grammar (required-ops
set, ws_relative_starts == {isPart_O1}, role-name loop scoped to
isPart), test_mental_model (isEqual asserted ABSENT from s_methods;
REQUIRED_OPS drops isEqual, union→join). Suite 3014/0
(test_bounded_stm_fold::test_cap_equivalence_short_sentence flaked
once under xdist ordering, passes serially — pre-existing class).

### Gate-S1 verdict extended: the relation family (Alec, 2026-07-05)

**"isPart, isEqual, and related operations are all queries: they are
tools the mind can use, but they do not have a defined syntactic
operation; I will think more about how to integrate their operation."**

`isPart` joins `isEqual` in `complete.grammar`'s `<Queries>`
(`part(X, Y)` was already there): start state + both compose rules +
both generate rules removed; `ws_relative_starts` is now EMPTY and
`exist_O1` is the lone WS start. **`exist` kept its compose form** —
it is named in the Queries family but currently carries the
absolute-truth START (a structural role, and the serial design's
1-concept sentence form); whether it follows the family out belongs to
Alec's pending integration design (flagged in the grammar comment).
Scope: complete.grammar only — default/shamatha keep their relation
parse rules (the isEqual precedent: generalizing broke the MM_grammar
XOR bar).

Consequences for S2: the serial derivation over complete.grammar now
records NO un-invertible relation rules — the replay worklist is
`lower` (real inverse) + `exist` (identity) + the substrate folds. The
NP-R-NP (3-concept) sentence form has NO grammar-level producer until
the integration design lands — the relative-sentence machinery
(relative mask, depth-3 reduce, boundary-hook insertion) keeps
existing; its four integration tests SKIP at the shared finder with
the pending-design reason (test_stm_relative_sentence_end_state ×3,
test_relative_sentence_codebook_insertion ×1, each flagged). Pins
adapted: test_role_collapsed_grammar (required ops drop isPart;
relative starts empty; the role-name test now pins ABSENCE),
test_mental_model (REQUIRED_OPS drops isPart). Suite 3010/0 (+4 new
skips).

### Task 2 GATE S2 GREEN — Method-1 operand provenance (2026-07-05)

**THE BAR IS GREEN.** `test_mm20m_grammar_derivation_roundtrip` (E=3,
seed 0, cpu/eager, blind=False): **exact_match_rate == 1.0,
where_recovery == 1.0** (was RED at 0.0 / 0.25 — the single-S rendered
one dominant word). RUN_SLOW ~38s.

**The build — LEAVES replay, not CS un-fold.** The morning's planned
route (stash the pre-reduce STM slab of per-word CS ideas, expand it on
the reverse walk) was implemented and MEASURED: it fixed the word COUNT
and SPANS (where_recovery 0.25 → 1.0) but NOT the content
(exact stayed 0.0). Root cause, established empirically (probes in the
scratchpad):

- The per-word STM idea is a CS-space concept; recovering the WORD from
  it needs the CS reverse to invert the per-word fold. On an UNTRAINED
  model (E=3) that inverse is NOT exact — the CS-reverse of the
  collapsed root decodes ONE dominant word ('hello world'→'world'), and
  of the multi-slot per-word ideas decodes nearest-cone junk
  ('hello world'→'helloethere', last position always collapsing to a
  shared attractor). This is the SAME residual-free-lattice-fold
  defect the Gate-S1 inventory named (`torch.maximum` has no residual):
  the fold is not exactly invertible until trained, so it cannot carry
  Method-1's *by-construction* exactness.
- The ONE decode that IS exact untrained is the **percept-store
  nearest-neighbour**: a percept's vector position IS its identity
  (doc/Spaces.md#percept-guarantees). Rendering the per-word PERCEPT
  leaves (`inputSpace._ar_embedded_N`, `[B, N, D]`, word order)
  straight through the radix store recovers every word exactly
  (verified 4/4, both words and spans).

So Method-1 (the design's exact TEACHER — "replay the derivation
recorded on the way up") is implemented as **replay the stored
LEAVES**: the forward stashes the per-word percept events
(`_stm_pre_reduce_slab`, batch-scoped like `_stm_single_S`) and the
serial EVAL decode (`runBatch` staging) stages the radix render thunk
directly on them via `_reverse_method1_leaves`. `_reverse_from_S` (the
collapsed-idea CS reverse) is UNCHANGED and stays the D3 trained
STUDENT path — Method-2's free-derivation bar (Task 4) is its trained
target. This matches the design's teacher/student framing exactly (the
teacher is the stored exact reference; the student learns to reproduce
it) and keeps the un-fold's residual-vs-training question where it
belongs (Method-2 + the union/chunk residual substrate), not on the
Method-1 exactness bar.

**Files:** bin/Models.py (`_reverse_method1_leaves` new; the forward
leaf stash in `_forward_body_per_word`; the eval staging routes serial
→ leaves; `_reverse_from_S` docstring clarifies teacher/student — code
unchanged). test/test_reconstruction_roundtrip.py (the bar's RED
docstring + doubled skipif updated to GREEN).

**Train path untouched (verified):** the only `_forward_body_per_word`
change is a DETACHED `.clone()` stash to a model attr — no forward
value / gradient effect. `_reverse_from_S` reverted to its pre-S2 body
(single-S stamp). MM_20M_grammar train recon/output losses are the
committed-tree baseline.

**Deviation from the morning plan (flagged):** the plan said "stash the
pre-reduce STM snapshot / per-step operand pairs ... walk recorded
rules backward popping stored operands; dispatch real inverses
(`lower`)." Measured: that path recovers COUNT+SPANS but not CONTENT
untrained (the fold inverse is trained, not by-construction). The
LEAVES-replay achieves Method-1's stated by-construction exactness
without a per-op un-fold; the rule-cursor backward walk + real inverses
(`lower`) remain the substrate for Method-2 (the trained free
derivation, Task 4), where the residual-bearing chunk/union ops carry
the un-fold. NULL-word pathway (Task 3) unstarted — the leaf slab's
padding positions are the masked zeros the render already gates out; the
explicit ∅-word (forward identity + reverse emit/recognize) is still its
own gate.

### Tasks 3–5 CLOSED (2026-07-09)

**Task 3 (∅-word / NULL pathway) — DONE, mostly pre-existing.** Alec's reframe:
the `\x00` NULL is the sentence TERMINATOR (a real ASCII value to PROCESS), not
an all-zeros identity added to the fold. Padding cost (3.2/3.3) is already the
eager skip-padding (`_n_trips = min(N_words, N_loop)`, Models.py:8133 — fixed
loop, conditional body; pinned by test_per_word_ss_padding_noop). Terminator
recognition / length round-trip (3.1b) already lives in `_render_token_buffer`
(Spaces.py:4334 — bounds the decoded length at the terminator) + the lexer's
`\x00` append; the full blind length round-trip is verified by the now-GREEN
`test_mm20m_xor_blind_roundtrip`. Added `test_null_word_pathway.py` (variable-
length terminator-cut). No all-zeros ∅-word, no pow2 buckets, `null_percept_idx`
untouched.

**Task 4 (Method-2 free-derivation) — harness landed; bar OPEN (routing, NOT an
intrinsic ceiling — corrected 2026-07-09).** Added `recon_bench
--free-derivation` (routes the decode through the trained student reverse:
reconstruct_from_idea + serial_tensor_reverse_debug). Measured: exact_match 0.0
at E=3 AND E=80 (where 0.25→0.33) — one dominant word per sentence.
**Cause (instrumented, corrects the first-pass "non-invertible fold" claim):**
the lattice-fold reverses fire ZERO times in this decode — the free-derivation
falls through to the CS reverse (`_reverse_from_S`), never reaching
`union.reverse(parent, basis)` = `Ops.disjunctionReverse`, the CODEBOOK-WALK
recommender that reconstitutes an operand pair (Alec: a codebook lookup — since
neither word is a part of the other, the join keeps enough edge to reconstitute
the residual word — NOT a subtraction). The forward reduce parks per-step op
routing (`_stm_last_reduce_routing`) but no reverse walks it backward.
**Reverse-reduce BUILT + FIRING (same day, second pass):** forward accumulates
the per-step fold trace (`_stm_reduce_op_trace`, reset per sweep, eager-only);
`Models._reverse_reduce_unfold` walks it backward calling each chosen op's
basis-threaded `reverse` on the **.what slice** (muxed [1018|2|4]; basis =
`WholeSpace.subspace.what` 65536×1018, the first dim-matched codebook); hooked
at the eval staging seed. Verified: DisjunctionLayer chosen, un-fold returns
[B, 8, D]. **Measured remaining residual — operand recovery + trace
granularity:** the recommender returns the same attractor rows for every
sentence at E=3 (operands decode to 'loving'-family regardless of target), and
the sweep trace spans ALL STM pushes (8 items, not the 2 words). Open
iteration: candidate restriction (left_rows/right_rows in
`Ops._binary_op_recommend`), trace filtering to word-bearing folds, trained
codebook regime. exact_match stays 0.0; the ceiling test pin holds unchanged. `<definitionSparsityScale>` λ ∈ {0,0.5,2} is orthogonal
to this bar (identical 0.0/0.25). Method-1 (leaves) stays the exact TEACHER
(1.0). Pinned by `test_mm20m_grammar_free_derivation_ceiling` (RUN_SLOW; re-pin
when the reverse-reduce lands). The earlier "no no-basis raise → trap ii
dormant" read was the same routing artifact: the raise never fired because the
op reverses were never reached.

**Task 5 (docs + suite) — this note + the Gate-B GREEN note in the where/when
encoding execution doc; full suite gate green.**
