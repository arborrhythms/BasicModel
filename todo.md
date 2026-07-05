================== SESSION SUMMARY 2026-07-04 → 2026-07-05 ==================

WHAT WE DID (all UNCOMMITTED in the working tree — Alec commits):

1. WHERE/WHEN ENCODING PASS (own plan, DONE through Gate B; Gate-B bar RED).
   .where period config-derived (<wherePeriod> 8192, warn-once raise-to-fit,
   decoupled from nObjects); .when v2 = 4-dim 2-rung start ladder
   (WhenStartDurationEncoding; <whenPeriod> 1e6 / <whenRungRatio> 32;
   duration left the band, exact long-int clock addresses LTM). Canonical
   band (2,2)→(2,4) plumbed via one construction seam; +2 when-band config
   ledger across model.xml + 8 configs; coordinated re-baseline (Gate A
   green); blind decode + scaffold-masking curriculum (Gate B — mechanism
   green, the E~80 full-model blind bar RED). 15 band-fixture tests adapted.

2. BLIND-BAR CHARACTERIZATION → led to Alec's union/difference idea. The
   lattice folds (join/intersection, join_from_bottom, conjunction/
   disjunction) saturate and carry NO residual; the [0,1] presence cube
   bunches cosine margins (xor store median 0.34, one positive cone) so
   with one symbol per word an inverse cannot recover constituents.

3. UNION/DIFFERENCE CONCEPT OPS (Alec's proposal; DONE).
   UnionLayer (a+b; reverse=(parent,∅); basis-reverse=peel step) +
   DifferenceLayer (w−a exact residual) + peel (matching pursuit) +
   registry/fixity + grammar-config integration + THE CONTRAST TEST.
   NAMING (Alec): pair = union/difference; the interim "fusion" was wrong
   (a different retired op); the lattice max renamed union→join (JoinLayer,
   matching Mereology.join_from_bottom) across 11 config grammars + 4
   .grammar files + tests. test_union_difference_ops.py 10/10.

4. SERIAL-DERIVATION PLAN Tasks 0–2 (Task 0 survey, Task 1 stubs→errors,
   Task 2 Method-1 routing) DONE; the grammar round-trip bar is RED (the
   single-S never expands — operand payloads not stored). Gate-S1 verdicts
   resolved by Alec (relation family → Queries; see below).

5. TOOLING: added pytest-xdist + `make testp` (opt-in parallel, ~3.6×;
   canonical `make test` stays serial/deterministic). Suite last green at
   3010 passed / 0 failed / 52 skipped (parallel).

OPEN QUESTIONS / FUTURE WORK (Alec's calls + next builds):

* [ALEC — DESIGN] Operational integration of the QUERY TOOLS (isEqual,
  isPart, exist, and the reasoner's equal/part/query/quantize/wholes/parts/
  arma). They are tools the mind uses but have NO defined syntactic
  operation — how do they get invoked during parse/reason? This is the
  gating design question for: (a) whether exist follows the family out of
  <compose> (it currently keeps the absolute-truth START role); (b) whether
  default.grammar/shamatha.grammar follow complete.grammar's relocation
  (generalizing there broke the MM_grammar XOR convergence bar — reverted);
  (c) reviving the NP-R-NP (3-concept) relative-sentence form, which now
  has no grammar-level producer (4 integration tests skip at the finder).

* [ALEC — KNOB] Gate-B blind round-trip bar is RED at E~80: full-model
  .where band precision (~4–7 byte claim error) can't separate xor's
  5-vs-6-byte tiles. Options: more where-band training pressure / longer
  budgets / accept scaffold-fed. Recorded in the encoding plan's Gate-B
  notes. (Mechanism is proven correct via synthetic-stamp tests.)

* [NEXT BUILD — Serial S2] Operand PROVENANCE for Method-1 exactness: stash
  the pre-reduce STM snapshot / per-step operand pairs on the forward
  (the reducer already parks marginal_slab [B,2,D] = the last step's
  operand pair on _stm_last_reduce_routing; multi-step needs per-step
  accumulation, batch-scoped like _stm_single_S). Walk recorded rules
  backward popping stored operands; dispatch real inverses (lower);
  then Task 3 NULL-word pathway (emit side; union's additive identity is
  the compose-identity), Task 4 Method-2 free-derivation bar, Task 5 docs.

* [POSSIBLE — Alec's call] Signed-domain concept storage (the deferred
  percept-cube signed-mixing decision): keep [0,1] presence as the symbolic
  READ (containment/truth), store+transport the ADDITIVE domain underneath
  where union/difference peel is exact — no nearest-neighbor guessing
  against a bunched cone. Couples to the blind-bar content path AND the
  serial peel decode (sigma.reverse's partition-blind split is the same
  missing-difference defect).

* [CLEANUP — deferred] Rename the space_role value pair to match: the
  lattice op is now "join" but its partner is still "intersection" (not
  "meet"); Alec did not ask for the meet sweep — left as-is.

============================================================================

* [DONE 2026-07-04 night; RENAMED 2026-07-05 per Alec] union/difference
  concept ops (doc/plans/2026-07-04-union-difference-concept-ops.md):
  UnionLayer (a+b; reverse = (parent, ∅); basis-reverse = peel step) +
  DifferenceLayer (w−a exact residual) + peel (matching pursuit) +
  registry/fixity + grammar-config integration + THE CONTRAST TEST
  (lattice pair provably destroys the residual; union/difference
  recovers) — test_union_difference_ops.py 10/10, suite green.
  NAMING RESOLVED (Alec 2026-07-05): pair = union/difference; the
  lattice max renamed join (11 config grammars swept union(→join().

* Execute doc/plans/2026-07-04-serial-derivation-reconstruction-execution.md.
  OVERNIGHT STATE (2026-07-04 night, notes in the plan): Task 0 DONE
  (sub-agent survey: decode consumed the TENSOR arm — _reverse_from_S's
  thunk was OVERWRITTEN; recorded rules {isEqual x3, lower, exist}; ZERO
  per-op reverses fire on the router path; unreduce = dead code there).
  Task 1 DONE (stub sanction revoked everywhere: raise_no_inverse
  inventory errors; unreduce + SyntacticLayer.reverse + 9 per-op stubs;
  exist invertible=True — identity forward; 2-D codebook guards on the
  four recommenders; 6 contract tests + 11 consumers adapted; suite
  3013/0). Task 2 ROUTING DONE, BAR RED: serial eval decode now consumes
  the Method-1 replay ('hello world'→'world' — the single-S never
  EXPANDS; rule names are stored but operand payloads are NOT);
  test_mm20m_grammar_derivation_roundtrip added (RUN_SLOW, RED).
  ⇒ NEXT (the real S2 build): operand PROVENANCE — stash the pre-reduce
  STM snapshot / per-step operand pairs on the forward; walk recorded
  rules backward popping stored operands; dispatch real inverses
  (lower); NULL-word pads the expansion (Task 3 emit side). GATE S1
  verdicts RESOLVED (Alec 2026-07-05): the relation family (isEqual,
  isPart, related) are QUERIES — tools, no defined syntactic operation —
  relocated to complete.grammar's <Queries> (equal/part forms); the
  OPERATIONAL INTEGRATION of query tools is Alec's open design question.
  exist KEPT its compose form (structural absolute-truth start; follows
  the family only if the integration design says so). default/shamatha
  keep relation parse rules (generalizing broke the MM_grammar XOR bar).
  S2 worklist now: lower (real inverse) + exist (identity) + substrate
  folds — no un-invertible rules recorded. NP-R-NP sentences have no
  grammar producer until the integration design; 4 relative-sentence
  integration tests skip at the finder (flagged). THEN Task 3 (NULL-word + padding
  fork), Task 4 (Method-2 bar), Task 5 (docs + suite). COUPLING: the
  union/difference pair is the residual substrate for the free
  (Method-2) case; NULL-word = union's additive identity.

* [DONE 2026-07-04] where/when encoding pass (doc/plans/2026-07-04-where-when-
  encoding-*.md): .where period decoupled (<wherePeriod> 8192, warn-once
  raise-to-fit); .when v2 = 4-dim 2-rung start ladder (<whenPeriod>/
  <whenRungRatio>; duration left the band, exact clock addresses LTM); band
  (2,2)→(2,4) plumbed + coordinated re-baseline (GATE A green); blind decode +
  scaffold-masking curriculum (GATE B — mechanism green, the E~80 blind bar is
  RED on full-model band precision, an OPEN Alec knob in the plan's Gate-B
  notes). Config ledger +2 when-band across model.xml + 8 configs; 15 band-
  fixture tests adapted.

* .where recovery is still a placeholder in [bin/recon_bench.py (line 206)](/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel/bin/recon_bench.py:206).

* The execution notes explicitly say residual mismatch is still slot alignment and pad-slot decode pollution, with .where stamp/decode untouched: [execution notes (line 696)](/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel/doc/plans/2026-07-03-reconstruction-fidelity-execution.md:696).

* Wave brightness and Task 11 nVectors wiring are explicitly parked out of scope: [design scope (line 36)](/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel/doc/plans/2026-07-03-reconstruction-fidelity-design.md:36).

* COMPILE/PERF residual (Task 8 census, doc/plans/2026-07-03-reconstruction-fidelity-execution.md). Two churns FIXED bit-identical: `len(out_slot)` guard -> pad-to-fixed-K (Models.py `_forward_body_per_word`); `vspace.errors['*']['count']` in-trace bump -> `is_compiling()` gate (Layers.py:13945). Inductor now amortizes ~2.6x vs eager (same host). REMAINING: the `p == N` per-word cursor guard (Models.py:7848 `out_slot[p]`, gaussian center Models.py:3020 `k=int(center_k)`) sits EXACTLY at `cache_size_limit=8` (reaches 8/7) — a longer sentence / larger config blows it. Principled fix = the NULL-word + power-of-two loop-bucket pathway captured in doc/plans/2026-07-04-serial-derivation-reconstruction-design.md; stopgap = small `recompile_limit` bump or tensorize the cursor int. Compile-default policy (Alec): DEBUG = none/eager (skip autotune); PRODUCTION = inductor past the amortization break-even.

* GPU/CUDA PORTABILITY (Task 8 GPU rung; blocks the borrowed CUDA server; GPU-first policy = everything on GPU by default). CPU-Generator-vs-device bug class SWEPT and fixed across bin/ (Language.py:2366, References.py:58-62; all other generator sites classified clean). REMAINING blockers before a full non-CPU grammar epoch: (1) MPS OOM on the 65536-row PS codebook — codebook/device-memory sizing, MPS-specific; (2) `AttributeError: 'NoneType'.is_empty` at bin/Spaces.py:21315 (`outputSpace.forward`, a subspace returns None) on the non-CPU forward path — device-independent-looking, needs root-cause.

* FULL-SUITE STATUS before relying on green: last `make test` was Gate 3 = 2971 passed / 0 failed, taken BEFORE the four post-Gate-3 fixes (len-churn pad, MPS/References generator sweep, count-churn gate). Each passed its targeted gate + the 24-test RUN_SLOW fidelity gate, but NOT a combined full-suite run — run a fresh `make test` on the committed tree to confirm the combined state (the Gate-4 close).

* The "Codebook.property_basis" is a hack that needs to be removed. Please summarize the WholeSpace property mechanism. You said properties "are" WholeSpace.what. But that codebook currently holds the symbol/truth prototypes wired into the codebook-snap machinery; making properties the live .what semantics would rip that out and move the basin. So I built the property capability as opt-in/additive (Codebook.property_basis) alongside the existing symbol codebook, not as a wholesale replacement. If you intended the live cutover, that's a separate deliberate step.
  * Codebook.property_basis = False still exists: [bin/Spaces.py (line 2599)](/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel/bin/Spaces.py:2599)
  * SubSpace.materialize(mode="property") only routes through property materialization when .what.property_basis is true: [bin/Spaces.py (line 6718)](/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel/bin/Spaces.py:6718)
  * The docs explicitly describe it as “additive and opt-in”: [doc/Spaces.md (line 41)](/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel/doc/Spaces.md:41)

* Implement user TruthSet-to-LTM integration.
  * WikiOracle parses the user-supplied XML <truthSet> into the /chat/completions request body as truth entries; bin/serve.py passes those to BaseModel.store_truths(entries). Today store_truths() still clears and writes SymbolSpace.truth_layer, while STM-derived memory and config-time <architecture><truthSet> provisioning write to symbolSpace.ltm_store when <ltmConsolidation>true.
  * Change store_truths(entries) so user-supplied truth entries append into the consolidated TernaryTruthStore LTM (symbolSpace.ltm_store) alongside STM-derived rows. Then make TruthLayer a compatibility/interface layer over the LTM tensor, or migrate its callers, so luminosity, falsity penalty, consistency, clarifications, and truth assessment read the same LTM-backed user truth data instead of a separate truth_layer.truths buffer.
  * Acceptance criteria: user XML TruthSet rows and STM-derived rows coexist in one persistent LTM store; TruthLayer no longer owns a separate canonical user-truth store; existing server truth request handling continues to surface clarifications/truth assessment; LTM consolidation tests plus runtime TruthSet ingestion tests cover the new path.


* Deliverable: create a dedicated masked semantic reconstruction training config.
  * Add a new XML training file derived from the masked IR config, but with the parser-side semantic context enabled. It should train masked-word reconstruction/prediction, not plain byte reconstruction: set a nonzero training.maskRate, use reconstruction-heavy loss (reconstructionScale high enough to dominate), keep sentencePrediction enabled, and enable the symbolic/category machinery needed for the parser to use word category evidence during reconstruction.
  * The config should also enable bottom-up concept attention: use the sparse conceptual loop path with symbolicOrder > 0, parallel symbolic execution, mereologyRaise, symbolTower, and the concept inventory sized so snap rows plus relation-pool rows cannot overflow. The goal is that a masked word is predicted from both its parser category/semantic-whole evidence and bottom-up attention over all active concepts.
  * Acceptance criteria: running this config performs masked IR training; masked positions contribute reconstruction loss; parser/category evidence is visible in the reconstruction path; concept attention/wave is live, not dark; and a targeted smoke test confirms masked-word predictions change when category evidence or concept-attention state is disabled.

* Make abstraction order canonical
  * the ramsification record has to become part of the normal codebook contract, not an optional sidecar. Every PS/WS codebook row that can represent a percept, word, type, or higher-order definition should carry fold provenance: for each subsymbolicOrder pass, whether that row was produced through Sigma, Pi, or neither. The scalar “order” should remain a derived readout, abstraction_order(row) = count(non-NEITHER folds), not separately stored state.
  * The main missing wiring is live stamping. Today the table exists and higher-order minting can stamp some rows, but the actual subsymbolic pump loop does not consistently call record_fold when a row is routed through PS synthesis or WS analysis. That needs to move into the canonical forward path: whenever a codebook row is created, selected, raised, or rewritten by a sigma/pi pass, the corresponding fold slot for that pass should be updated. This makes order provenance a normal consequence of processing rather than a special feature behind mereologyRaise.
  * The opt-in flag can be removed once ramsification allocation is cheap and universal. Allocate the table for relevant codebooks at creation time using max_order = max(1, architecture.subsymbolicOrder), grow it with the codebook, and reset nothing on ordinary document boundaries. Keep it non-gradient metadata, but decide whether it should persist. If explicit-constraint retraining depends on it after reload, it should ride checkpoint metadata or a sidecar serialization path; leaving it out of state_dict is only safe while it is reconstructible from deterministic build/mint history.
  * Explicit-constraint retraining should consume the fold provenance directly. A constraint should resolve the affected lexical/concept row, inspect its fold sequence, and route the update through the matching inverse path using invert_ramsified or an equivalent per-pass inversion. That lets “this word’s abstract definition changed” update the high-order semantic representation without clobbering the low-order perceptual form. It also lets constraints target the right layer: raw token identity, basic category, count/type noun, or higher-order definition.
  * Finally, tests should enforce the canonical contract. Build a model without any mereologyRaise flag and assert PS/WS codebooks have ramsification tables. Run a multi-pass forward and assert selected or minted rows receive expected Sigma/Pi fold stamps. Verify abstraction_order is stable after codebook growth and reload, and add a small explicit-constraint training test showing that an abstract definition update changes the intended high-order row while preserving low-order reconstruction.

* Turn on <subsymbolicStack>true</subsymbolicStack> and remove that parameter

* document the relation of this architecture to LLMs, Formal Concept Analysis, and DisCoCat

* My understanding of the relation table is that there is one concept index associated with one symbol index per row. 
* That is sufficient for a vine (ordering), since there is an order between the concept and the concept that symbol references.
* That is sufficient for set definition, since the table may contain multiple concept indices, thus allowing a single concept to accumulate multiple symbols.

* Implement doc/plans/2026-07-04-attention-to-relation-promotion.md
* Sentences create explicit knowledge if they pass a criterion:
  learn_score = children_in_codebook * is_truth_obvious * resolves_contradiction
  accept iff learn_score >= truth_criterion and truth_criterion < 1


================================== April 24 ==================================

### Ask Solid community for a simple file-getting interface
* if the user provides the server with an API key, we can query an LLM
* if the user provides the server with a SOLID key, we can retrieve a file
* if the user provides the server with a DSA key, we can decrypt a file
* is there a POD service that does simple free hosting?

### Ask EFF for a security review
* propose "Owning our Data"
* this entails taht marketers and AI are not allowed to lock us down karmically
with specifically-characterized information (concrete details)
* maybe it can learn from that data by removing or randomizing that information

### Send email proposal to Apertus 
* First develop boilerplate on WikiOracle that references wikipedia, eff, and solid

