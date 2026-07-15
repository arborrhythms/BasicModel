# The PS word store (type="words") for the reverse() recommenders

> Directive (Alec, 2026-07-12): "run the word grain: put all words in
> type=\"words\" that you store explicitly, so that we can use them in the
> reverse() methods." REVISED per Alec's review (2026-07-13): "I'm
> expecting the word store to be the collection of words in PS, which
> already existed (but which was not labelled as such)." The word store
> is NOT a new object — it IS the PS RadixLayer's promoted collection;
> this pass LABELS it (the type="words" view) and routes the Method-2
> free-derivation reverse's candidates through it. The measured Task-4
> residual (every sentence decoding to the 'loving' attractor family)
> is an unrestricted-candidate problem.

## REVISION (2026-07-13, supersedes the v1 design below)

**v1 (retired same-tree, never committed):** a parallel `WordStore`
class on WholeSpace holding CS-grain idea prototypes keyed by staged
surface texts. Wrong home: it shadowed a collection PS already stores.

**The facts that fix the design:**

- `RadixLayer` (the PS percept store) already IS the explicit word
  store: under meronomy synthesis it is `word_bounded` — promotion
  never crosses a word boundary, so promoted multi-byte entries ARE
  the words — with `inverse_table` (id $\to$ bytes) as the surface and
  the vectors stored as ROWS OF PS `subspace.what` (percept id ==
  codebook row; the embed gathers `what_event = W[pid_grid]`).
- Codebook rows are CONTENT-width: PS `.what` is `[65536, 1016]` and
  the STM idea `.what` slice (`d_what`) is 1016 on the same config —
  the percept rows are ALREADY dimensionally valid recommender
  candidates. No new vectors, no grain bridge, nothing new to persist.
- PS's radix embed already threads per-slot percept ids
  (`_forward_input['indices']`), word groups, and word surfaces
  (`word_texts`) — the v1 WholeSpace text staging duplicated this.

**v2 design (implemented):**

1. **Label, ungated (a view, no state):** `RadixLayer.WORDS_TYPE =
   "words"`; `word_ids(standalone_bytes=None)` — promoted MULTI-byte
   entries, plus single-byte entries only when their byte is in
   `standalone_bytes` (the same "appears standalone" attestation rule
   T3's within-whole division uses, sourced from the stage-0 WS's
   `_standalone_run_bytes`), so 'a'-as-word is admitted without
   admitting every seeded byte percept; `word_text(pid)` — the surface
   (realize direction; `bytes_for` decode).
2. **Reverse consumption, gated `<PartSpace><wordStore>` (default
   off = byte-identical):** `Models._reverse_reduce_unfold` prefers
   basis = PS `subspace.what` with `left_rows = right_rows =
   word_ids()` — the recommender's DESIGNED typed-restriction
   machinery (`_binary_op_recommend` row masks; ⊥/⊤ sentinels stay
   feasible). Row identity = percept id, so recovered operands map
   back to surfaces via the store (`RadixLayer.reverse` /
   `bytes_for`). Knob off, or no word rows, or a dim mismatch $\to$
   the existing first-dim-matched pick, unchanged.
3. **No population seam.** Promotion IS the storing. The v1
   serial-loop stamping, WS `_staged_word_texts` stash, WS
   `<wordStore>` knob, WS vocab_extras keys, and the `WordStore` class
   are all REMOVED.

**Persistence observation (pre-existing, out of scope):**
`RadixLayer.vocab_extras()` exists and roundtrips at class level
(test_percept_store), but `BasicModel.save`'s envelope does not
collect it — the trie/inverse_table do not currently ride model
checkpoints (the codebook rows do, via PS `.what`). The word VIEW
inherits this store-level story; flagged for Alec, not changed here.

## Design

**The store.** `Spaces.WordStore` — `store_type = "words"` (the literal
type tag), surface text $\to$ one prototype row. Rows live in a plain
(non-Parameter) tensor `W` `[cap, D]` grown by doubling — never
snapped, never trained, invisible to optimizers (the frozen
`type_subspace` storage idiom, but growing). Dedup by exact text;
re-sighting REFRESHES the row (last-write: the fold operands the
reverse must recover are the CURRENT regime's pushed ideas, so the
prototype tracks them; EMA is a recorded alternative, not built).
`getW()` returns the live slice `W[:count]` — the recommender's Basis
contract (`UnionLayer.reverse` accepts any object with `getW()`).

**Grain.** Prototypes are the per-word STM idea `.what` slice
(`d_what = D_c - nWhere - nWhen`, the same slice
`Models._reverse_reduce_unfold` un-folds) — NOT WS codebook rows. The
fold operands ARE pushed ideas, so candidates at that grain are exact
by construction.

**Population (training only).** The serial per-word loop
(`Models.py` static loop, `_per_word_body_step` site) has, per trip
`p`: the per-word idea `idea_bd [B, D_c]` and the word-active gate.
Surface texts come from the SAME cut that staged the word-wholes:
`WholeSpace.stage_analysis_spans` already holds the CPU byte grid and
the final (post within-whole division) spans — it now also stashes
`_staged_word_texts` (per-row decoded strings, the
`_staged_analysis_spans` idiom) when the store is enabled. The loop
body calls `Models._store_word_prototypes(p, idea_bd, gate_b_1)`:
host-eager only (`torch.compiler.is_compiling()` guard), training
mode only (eval decode must not read this batch's ideas — the bar
stays honest), active rows only. ALL identified words store (no
promotion threshold here — recurrence gating is the percept store's
job; an open knob if singleton pollution shows up at corpus scale).

**Consumption.** `Models._reverse_reduce_unfold` basis pick: PREFER
the word store when populated and dim-matched (`getW().shape ==
[count>0, d_what]`), falling back to the existing first-dim-matched
`.what` codebook unchanged. The recommender walk itself
(`Ops._binary_op_recommend`) is untouched — the store IS the
candidate restriction (⊥/⊤ sentinels still ride via the augmented
codebook). The lift/lower driver basis (`Language.py:5828`) is NOT
touched this pass — that path carries the grammar exact-1.0 bar;
wiring the word store there (and the reading-scope priming, Track-1
G3) is the recorded follow-on.

**Gate.** `<WholeSpace><wordStore>true</wordStore>` — absent/false =
OFF, byte-identical: no store object, no text staging, no population
call body, basis pick unchanged. Schema: `model.xsd` gains the
element beside `divideWithinWhole`.

**Persistence.** WS `vocab_extras()` emits `"word_store": {"texts":
[...], "W": cpu tensor}` ONLY when populated (flag-off blobs
byte-identical — the `type_property_kinds` precedent);
`load_vocab_extras` restores when the config's knob is on.

**Still open after this pass (named fronts, unchanged):** trace
filtering to word-bearing folds (the sweep trace spans all STM pushes,
so non-word folds still un-fold against word candidates); the trained
regime (prototypes are only as good as the epoch that wrote them);
relation-endpoint parse fidelity (grammar routing, not candidates).

## Tasks

- [x] **T1** `WordStore` (Spaces.py): store/dedup/refresh/getW/
      text_at/rows/ensure_device/extras/load_extras + unit tests
      (`test/test_word_store.py`).
- [x] **T2** WS knob read (`<wordStore>`, divideWithinWhole idiom) +
      `model.xsd` element + `stage_analysis_spans` text stash + tests.
- [x] **T3** `Models._store_word_prototypes` + serial-loop call +
      integration test (MM_meronomy_smoke-derived config, one train
      forward populates the phrase words exactly).
- [x] **T4** `_reverse_reduce_unfold` word-store basis preference +
      test (basis-seam spy: populated store IS the basis; empty store
      falls back).
- [x] **T5** vocab_extras roundtrip + test.
- [x] **T6** Targeted sweeps + execution notes + todo.md line.

## v3 — IN-MODEL recognition + META registration (Alec's calls, 2026-07-13)

> Alec: the only IN-MODEL way to recognize a word is the cross-tower
> correspondence — a string of letters (the WS whole's `.where`)
> corresponding 1:1 to the run of PS parts — and the label belongs at
> the higher-order META layer (the word/object/meta swap machinery),
> not on the RadixLayer. Design calls: **(Q1)** 1:1 means a SINGLE
> part and a word-whole; **(Q2)** with that criterion the FIRST
> sighting is already consistent (promotion encodes the recurrence),
> so register immediately; **(Q3)** explicit registration under a
> stored higher-order WORDS concept is preferred as long as it stays
> small (one concept + one part-edge per word — negligible). The
> storage-free alternative (membership DERIVED from A's structure:
> exactly one part-code beside the whole-code, roles distinguish the
> kinds) is recorded here as the fallback design.

**Mechanism (implemented):**

1. **Recognition** at the autobind seam
   (`ConceptualSpace._maybe_autobind_meta`, the `<mereologyRaise>`
   word-binding arm): a binding whose part-run has length 1
   (`len(w_parts) == 1` — one promoted percept spanning the
   word-whole; the grouping guarantees the span) registers on first
   sighting, right where `create_word_object_meta` mints A/B/C.
2. **Registration**: `_register_recognized_word(A, key, pid)` —
   lazy-mints THE higher-order WORDS concept (relation-only,
   `new_concept`), accrues `('sym', A)` into Parts(WORDS), and
   records `key -> pid` in `_recognized_words` (the host-registry
   idiom `_word_whole_ss` already uses). Parts(WORDS) is the in-model
   structure; the registry is the row projection. WORDS is
   deliberately NOT `_populate_concept_weights`-ed this pass (a
   many-part flat concept would fight the order caps; its codebook
   face is a follow-on).
3. **Consumption**: `recognized_word_rows()` (pids == PS `.what`
   rows) feeds the un-fold's `left_rows`/`right_rows` FIRST;
   `RadixLayer.word_ids` (synthesis-side recurrence evidence) remains
   the fallback when no recognition has fired (e.g. configs without
   `<mereologyRaise>`). Gate unchanged: `<PartSpace><wordStore>`.
4. **RadixLayer.WORDS_TYPE demotes** to evidence + the surface
   realize map (`bytes_for`/`word_text`); the authoritative in-model
   label is WORDS membership.

**v3 REFINEMENT (Alec, 2026-07-13 second pass):** "1:1 means
part-aggregations that are equal in span to the word-property span
from WholeSpace" — the first-cut `len(w_parts) == 1` tested only
PS's OWN tiling; the criterion is CROSS-TOWER. Landed:
`_agg_span_matches_ws_property` — the aggregation's span (PS's
per-slot tile record, threaded as `tile_spans` from
`_forward_input`) must EQUAL a staged WS analysis span (the
word-property `.where` claim; read from the autobind's ws, falling
back to `wholeSpace_ref`). STRICT: missing spans on either side, or
no equal span, or a still-unpromoted trie (multi-part run) ⇒ NO word
— and per Alec that is OK ("we may have no word on which some
algorithms will operate... if the radix trie has not built up
sufficiently"). Bindings now carry their slot indices (5-tuple; the
standalone word-binding tests ignore the return). Pinned: promoted
word with a WS span twin registers; multi-part run does not;
promoted word WITHOUT a WS property twin (span mismatch) does not.
Live-path tests confirm the PS tile frame and WS property frame agree
on the real forward (recognition still fires on the smoke).

**v3 REFINEMENT, third pass (Alec: "a word-whole may have a
letters-part which is not the entire word, and that is likely the
point of failure"):** correct — the tile record stamps the WORD-tile
span on EVERY slot of a piece, so a single promoted part covering
only a prefix (e.g. 'zebr' inside 'zebry', or an emission-budget
truncation) passed the span-equality test. The gate now ALSO requires
the parts' OWN byte extent (`percept_store.bytes_for` lengths,
threaded from `_forward_input['percept_store']`) to FILL the
aggregation span exactly. Pinned: a promoted 4-byte letters-part on a
5-wide property span does NOT register
(`test_one_to_one_recognition_registers_under_words`, the 'zebry'
case). Strictness unchanged: any missing ingredient ⇒ no word.

**v3 execution notes (2026-07-13, cpu/eager):** landed as designed;
test/test_word_store.py now 10 (added: the 1:1 recognition unit —
'zebra' promoted/single-pid registers on first sighting, Parts(WORDS)
carries ('sym', A_zebra), while unpromoted 'yak' (3-pid run) does
not; the unfold spy upgraded to expect the RECOGNIZED rows and to
fail loud if the recognition seam goes dark on the smoke config; the
fallback test clears BOTH sources). Autobind-adjacent sweeps green
(autobind_from_cs + mereology_word_binding + mereology_raise +
mereological + mereology + percept_store + bounded_stm_fold +
decode_exemplars: 97 passed). OBSERVATION for Alec:
`MM_20M_grammar` does NOT set `<mereologyRaise>`, so on the matrix
bench the recognition seam is dark and the un-fold exercised the
word_ids FALLBACK (the v2 measurement stands unchanged); lighting
recognition up there means flipping `<mereologyRaise>` on the grammar
config — a training-numerics change, Alec's call, not taken here.

## EXECUTION NOTES v2 (2026-07-13, cpu/eager, seed 0)

**The v1 pieces are fully reverted** (WordStore class, WS knob +
`_staged_word_texts` stash, WS vocab_extras keys, the serial-loop
population helper, the WholeSpace xsd element) and v2 is landed:

- `RadixLayer.WORDS_TYPE` / `word_ids(standalone_bytes=None)` /
  `word_text(pid)` (bin/Layers.py, beside `get_id`) — the LABEL over
  the existing promoted collection. Single-byte words attest via the
  caller-supplied standalone set (the T3 rule); seeded byte percepts
  alone never qualify.
- `<PartSpace><wordStore>` (model.xsd partSpaceType, beside the
  promotion knobs) $\to$ `PartSpace.word_store_reverse` (read beside
  `chunkPromotionMinLength`).
- `Models._reverse_reduce_unfold`: gated basis = PS `subspace.what` +
  `left_rows`/`right_rows` = `word_ids()` (threaded
  `wholeSpaces[0]._standalone_run_bytes`); kwargs degrade
  TypeError-stepwise (rows+basis $\to$ basis $\to$ bare). Knob off /
  no rows / dim mismatch = the prior pick, byte-identical.
- Tests REWRITTEN (test/test_word_store.py, 9): label predicates
  (multi-byte promoted; single-byte needs standalone attestation;
  id==row), recommender row-restriction recovery (decoy excluded),
  reading promotes the batch words, unfold basis+rows spy (on),
  fallback (empty store), knob-off identity.
- Sweeps: word_store 9 passed; type_run_spans + within_whole_division
  + null_word_pathway + percept_store + lexicon_ownership +
  bounded_stm_fold + decode_exemplars 104 passed / 1 RUN_SLOW-skipped.
- **Test-authoring trap (recorded):** MM_meronomy_smoke's HEADER
  COMMENT contains the literal `<synthesis>meronomy</synthesis>`, so a
  first-occurrence string replace lands knobs inside the comment —
  anchor config-fixture injections on the line-start indented element.
- Matrix variant `data/matrix/MM_20M_grammar_wordstore.xml` now flips
  `<PartSpace><wordStore>` (one flag; the parent already carries
  `chunkPromotionThreshold 2`).

**Free-derivation measurement v2 (E=3, seed 0, cpu/eager,
`recon_bench --free-derivation`, probe-instrumented):** the un-fold
fired once at the decode with knob TRUE and the word rows = exactly
the 4 promoted PS words `['hello', 'world', 'loving', 'there']` (the
xor vocab — the collection Alec named). exact_match 0.0 /
where_recovery 0.0 / recon 0.048267 — identical to the parent
baseline, so the same conclusion as v1 stands: candidate quality is
now in place and the residual exactness fronts are the TRACE
GRANULARITY (the sweep trace spans all STM pushes, so the walk
un-folds non-word steps too) and the `where` placement / trained
regime. The RUN_SLOW ceiling pin holds unchanged.

## EXECUTION NOTES v1 (2026-07-12, cpu/eager, seed 0 — RETIRED, kept for the record)

**All six tasks landed; test/test_word_store.py 12/12 green.**

**Ownership deviation (load-bearing, found by probe):** the model has
MULTIPLE WholeSpaces — `self.wholeSpace = wholeSpaces[-1]` (TERMINAL)
is the one whose `vocab_extras` ride checkpoints
(`_collect_vocab_extras`), while the stem stages analysis spans (and
so the word TEXTS) on `wholeSpaces[0]` (`Models.py` `_ws_list[0]`
site). The plan's single-owner language resolves as: the STORE's
canonical owner is the terminal WS (persistence for free);
`_store_word_prototypes` reads staged texts from `wholeSpaces[0]` and
writes the terminal store. Non-terminal stages carry an unused (lazy,
zero-cost) `WordStore` instance each — acceptable; noted here.

**T4 test deviation:** the first-pass test asserted exact operand
recovery through an ARBITRARY reducer op (first basis-accepting), and
the smoke grammar's first such op is not a lattice fold — the ≤/≥
ordering filters legitimately knocked the true rows out and returned
⊤. Recommender-level exact recovery is pinned at the unit seam
(`test_recommender_recovers_stored_pair_from_union_fold`, lattice
'union' semantics); the unfold tests pin the BASIS-PICK seam with a
reverse() spy (populated type="words" store IS the basis passed to the
chosen op; empty store falls back to the dim-matched codebook).

**Sweeps:** test_word_store 12 passed; type_run_spans +
within_whole_division + null_word_pathway 38 passed / 1 skipped;
bounded_stm_fold + decode_exemplars + reconstruction_roundtrip
27 passed / 2 failed / 2 RUN_SLOW-skipped — the 2 failures are
`test_mm20m_xor_roundtrip_at_harness_budget` +
`test_mm20m_xor_exact_roundtrip`, the DOCUMENTED pre-existing xor
exact-match family (pyramid exec notes item 37; no `<wordStore>` on
those configs, so this pass's paths are byte-identical there);
percept_store + lexicon_ownership 57 passed.

**Matrix variant:** `data/matrix/MM_20M_grammar_wordstore.xml` —
parent verbatim + `<wordStore>true</wordStore>` (the
MM_20M_grammar_reading one-flag pattern). Free-derivation measurement
recorded below.

**Free-derivation measurement (E=3, seed 0, cpu/eager,
`recon_bench --free-derivation`, same tree, same day):**

| config | exact | where_rec | recon | notes |
|---|---|---|---|---|
| MM_20M_grammar (parent, no store) | 0.0 | 0.0 | 0.048267 | baseline |
| MM_20M_grammar_wordstore | 0.0 | 0.0 | 0.048267 | store ENGAGED |

Training numerics are IDENTICAL by design (population is read-only on
the forward), and the bench decode was verified ENGAGED by probe: the
un-fold fired once at the free-derivation decode with the store
holding 4 words and the basis pick selecting the type="words" store.
`exact_match` honestly does not move at E=3 — the exec-notes'
remaining fronts bind exactly as recorded (the sweep trace spans ALL
STM pushes, not the word-bearing folds, and `where` placement is
noise at this regime): candidate quality was ONE of the three named
fronts, and it is now in place for the other two. The RUN_SLOW ceiling
pin `test_mm20m_grammar_free_derivation_ceiling` therefore holds
unchanged (no re-pin needed).

**Follow-ons (unchanged from the design):** trace filtering to
word-bearing folds; the lift/lower driver basis (Language.py:5828,
grammar-bar-carrying — untouched); reading-scope priming into
`left_rows`/`right_rows` (Track-1 G3); a promotion/recurrence knob if
singleton pollution shows at corpus scale; `realize` via
`WordStore.text_at` when the render path wants surface words.
