# Attention-to-Relation Promotion — Execution Plan + Notes

> **Executes:** [2026-07-04-attention-to-relation-promotion.md](2026-07-04-attention-to-relation-promotion.md)
> (0/7 built at start, per the 2026-07-06 concepts-formalization execution
> doc). Acceptance criterion pinned by todo.md:
> $\mathrm{learn\_score} = \mathrm{children\_in\_codebook} \cdot
> \mathrm{is\_truth\_obvious} \cdot \mathrm{resolves\_contradiction}$;
> accept iff $\mathrm{learn\_score} \ge \mathrm{truth\_criterion}$ AND
> $\mathrm{truth\_criterion} < 1$ — the SAME law the Task-6c sentence gate
> already enforces (`_maybe_learn_relation`, bin/Spaces.py $\approx$ 16703).

## 0. Architecture mapping (the plan predates the FF pyramid)

The 2026-07-04 plan speaks in wave terms. The 2026-07-10 dual-towers arc
retired the settling wave for the feedforward $\sigma$-pyramid, so the
plan's surfaces map as follows (all verified against current code):

| plan term | current surface |
|---|---|
| "bottom-up attention field" | the pyramid's per-rung ADMITTED rows + activations (`cs_forward_content` stashes `_cs_level_rows`; final signed `acts [N,B]`) |
| "enter the ConceptualAttentionLayer wave" | a row in the shared square store at its order block — admission is automatic once `assign_row` bumps `_row_next` for the region |
| "sparse edges with initial weights" | `add_concept_edge` values on the untyped store, set via `_set_concept_edge_value` (fixed here — see §2) |
| "relation-table higher-order mint path" | `synthesize_higher_order(part_refs)` with `("sym", cid)` part refs (order $\ge 1$, edges populated by `_populate_concept_weights`) |
| "route into masked recon / conceptual-SBOW loss" | automatic through the existing interfaces: admitted rows feed `acts` $\to$ `_concept_activations` (SS leg) and the content slab $\to$ `_cs_parallel_slab` (conceptual SBOW) |
| "attention-evidence collector at the post-pump symbolic cutover" | evidence STASHED at the cutover, CONSUMED at `ConceptualSpace.Reset(hard)` — the same compile-safety hoist `learn_relations_from_stm` uses (host-side dict mutation is untraceable) |

## 1. Gate + knobs

- `<architecture><attentionPromotion>` boolean, default false $\to$
  byte-identical (XSD element beside `<relevance>`; per-space
  `<ConceptualSpace>` override like `<truthCriterion>`). Parsed in
  `ConceptualSpace.__init__` via `TheXMLConfig.space(...)`.
- The promotion bar REUSES `self.truth_criterion` (default 1.0): even with
  the gate on, nothing promotes until the criterion is lowered below 1.
- Policy constants are CLASS ATTRIBUTES (test seams, the
  `_learn_children_dist_threshold` convention), not XML knobs:
  min-support, cache capacity, EWMA $\beta$, member fraction, activation
  $\varepsilon$, cosine merge bar, contrast bar, Jaccard whole-merge bar,
  stale age, decay factor, retire $\varepsilon$, intent top-C.

## 2. Pre-existing defect fixed in scope

`_set_concept_edge_value` (bin/Spaces.py $\approx$ 15602) resolved the
concept row via the RETIRED `("pool", cid)` namespace and constituents via
`("snap","pool")` only — on rev-2 per-order `o<k>` blocks it silently never
fired (the exact bug class execution-note item 30 fixed for
`_hebbian_strengthen`). Promotion initializes minted edges through it, so
it is rewritten over `_csw_row_of` + an all-namespace constituent scan that
keeps the documented "whichever namespace carries the live edge wins"
semantics. (Latent consequence before the fix: `assert_concept_relation(...,
weight=w)` was a silent no-op on taper stores.)

## 3. Mechanism (plan steps $\to$ implementation)

1. **Observe.** `cs_symbolic_phase` tail (gated) stashes detached
   `acts [N,B]`. `Reset(hard)` calls `promotion_observe()`: per batch row,
   the active set = admitted rows with $|a| \ge \varepsilon$; each active
   ORDER-0 row is a FOCAL member observation whose CONTEXT is the rest of
   the active set. Fail-loud on non-finite acts.
2. **Candidate cache.** Bounded dict keyed by the context-row frozenset;
   near-miss observations fold into the nearest entry by cosine over the
   entry's EWMA context vector ($\ge$ merge bar) — the plan's "signature =
   weighted neighborhood, not a single pair". Entry stats: support count,
   EWMA member weights `[N]`, EWMA context weights `[N]`, last-seen
   counter, committed cid. A global background EWMA over active rows is
   the negative-context baseline. Capacity eviction drops the weakest
   uncommitted entry first.
3. **Score.** Members = rows with $w \ge \mathrm{frac} \cdot \max w$;
   $\ge 2$ must resolve to concept ids (`_row_to_concept`). Contrast =
   mean member weight minus mean weight of co-observed non-members, must
   clear the bar. Then the learn-score gate:
   `_promotion_learn_score` = `_learn_score_members_in_codebook`
   (n-ary children factor: fraction of member content rows within
   `_learn_children_dist_threshold` of the terminal-WS codebook)
   $\times$ `_learn_score_is_truth_obvious(sig)` $\times$
   `_learn_score_resolves_contradiction(sig)` — the last two are the
   EXISTING Task-6c seams, passed the candidate's signature vector.
   Accept iff score $\ge$ `truth_criterion` and `truth_criterion` $< 1$.
4. **Commit.** `H = synthesize_higher_order(("sym", cid), ...)` — members
   become `Parts(H)` (the plan's membership edges; idempotent per member
   set via `relate_idx[("raise", frozenset)]`). Member edge VALUES
   initialize from normalized EWMA member weights. The shared intent =
   top-C context concepts, committed as weighted `sym_part` assertions via
   `assert_concept_relation` (this codebase's "a body has a leg" property
   channel). Entry marked committed.
5. **Wave/loss coupling.** Nothing extra to wire: H's row sits in its
   order block, so the next `cs_forward_content` composes it from member
   activations through its edges, top-K admission decides survival, and
   admitted activation flows to the SS leg + conceptual SBOW through the
   unchanged interfaces. The ablation test is the proof.
6. **Reuse, don't duplicate.** Re-supported committed entries
   Hebbian-strengthen H instead of re-minting; a new candidate whose
   member set overlaps a committed whole at $\ge$ the Jaccard bar folds
   into it.
7. **Prune / decay / retire.** Stale weak candidates drop from the cache.
   Committed wholes unsupported past the stale age decay their edge values
   (`SparseLayer.decay_row`); below the retire $\varepsilon$ the whole is
   retired (`retire_concept`; frozen never retire).

## 4. Deviations from the 2026-07-04 plan text (deliberate, recorded)

- **Collector site:** evidence is stashed at the cutover but consumed at
  `Reset(hard)` — sentence-boundary cadence, mirroring the
  `learn_relations_from_stm` hoist (fullgraph compile safety). The plan's
  "during parse/reconstruction" per-forward cadence would mutate host
  dicts inside the captured forward.
- **Role-aware evidence (subject/object role, predicate, STM position)**
  is NOT in v1 — the collector reads the pyramid's admitted field only.
  The plan marks role-awareness "when available"; the hooks (STM role
  masks, predicate positions) stay future work.
- **Prediction-gain bookkeeping** (loss-delta per candidate) is NOT in
  v1 — the promotion score uses support/contrast + the learn-score law
  pinned by todo.md. The ablation TEST measures the functional coupling
  the plan wanted from prediction-gain, without per-candidate loss
  attribution machinery.
- **Whole-merge** is fold-into-existing (Jaccard) rather than
  merge-two-existing-wholes; exact-duplicate minting is already impossible
  (`relate_idx` idempotency).

## 5. Verification

- Targeted: `test/test_attention_promotion.py` (default-off byte-identical;
  cache accumulation/merge/eviction; the criterion law incl. both
  endpoints; mint + reuse + strengthen; edge-value init; pyramid-coupling
  ablation — removing H's edges changes rung activations; decay/retire;
  fail-loud NaN) + a `_set_concept_edge_value` taper regression pin.
- Gate: `make test` green on the quiet tree.

## EXECUTION NOTES (append during execution)

1. **Executed 2026-07-12 (single session, cpu/eager).** All seven plan
   tasks landed as designed in §3. Files: bin/Spaces.py (gate parse,
   `_csw_rows_of` + `_csw_row_of` refactor, the §2 fix, the promotion
   block after `_hebbian_strengthen`, cutover stash, Reset wiring),
   bin/Layers.py (`SparseLayer.decay_row`), data/model.xsd (architecture
   element + ConceptualSpace override), doc/Architecture.md (cutover
   section paragraph), test/test_attention_promotion.py (17 pins).
2. **The §2 fix verified:** `_set_concept_edge_value` on a rev-2
   per-order store was a silent no-op pre-fix
   (`test_set_concept_edge_value_reaches_per_order_blocks` is the
   regression pin). `_csw_row_of` now derives from the new
   `_csw_rows_of` (all-namespace row enumeration).
3. **Collector fan-out is per-focal, by design:** every ACTIVE order-0
   row is focal for its own context entry, so one observation seeds
   sibling entries (the royalty probe: king-focal {crown, palace} PLUS
   crown-focal {king, palace}, ...). The shared-context entry is the one
   that recurs; siblings die by support/staleness/eviction. Two test
   premises were corrected during execution to match (fold + stale
   tests count sibling entries).
4. **Contrast is within-entry** (members vs co-observed non-member
   focals); the global background EWMA sketched in §3.2 was dropped as
   dead state in v1 — a ubiquitous-row discount (negative-context
   baseline) is future work alongside role-aware evidence.
5. **Verification:** test_attention_promotion.py 17/17; the 16
   neighboring suites (sparse-concept e2e, cs_symbol_table,
   conceptualize, truth-criterion cluster, frozen concepts, relevance
   bases, typed definition, definition sparsity, sparse layer,
   cs_sparse_weights, mereology word binding) 205 passed / 1 skipped.
   `make test` gate: see note 6.
6. **Full-suite gate (2026-07-12): 5 failed / 3145 passed / 53 skipped /
   34 xfailed — the 5 failures are PRE-EXISTING at HEAD 7f88069, not
   from this change.** Parity proven by running the identical node set on
   a pristine `git archive HEAD` checkout: the SAME five fail there
   (test_blind_decode::test_recon_bench_blind_flag, test_mm_xor
   convergence + learns_xor_signal, test_reconstruction_roundtrip
   mm20m_xor exact + harness-budget), same `reconstruction_zeroed`
   warning on MM_20M_xor. This is the xor item left OPEN AT HANDOFF by
   the dual-towers arc (2026-07-10 execution notes item 37) — the
   baseline this session inherited, byte-for-byte.
