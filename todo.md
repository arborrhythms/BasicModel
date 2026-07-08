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
  XFAILED 2026-07-06 (test_mm20m_xor_blind_roundtrip, strict=False) pending
  the knob call.

============================================================================

* Serial-derivation reconstruction (doc/plans/2026-07-04-serial-derivation-
  reconstruction-*.md): Tasks 0-2 + the S2 operand-provenance build DONE
  (2026-07-06, Method-1 leaves replay — test_mm20m_grammar_derivation_
  roundtrip GREEN). REMAINING from that plan: Task 3 (NULL-word pathway),
  Task 4 (Method-2 free-derivation bar). Its front half — the idea→typed
  conceptual definition (signed OMP peel + ConceptualSpace.typed_definition:
  head/modifiers/exclusions/residual) — LANDED with the concepts snap-contract
  work (doc/plans/2026-07-06-...-execution.md, T0–T6 full-suite gated); what
  remains for Task 4 is the grammar-surface COMPRESSION (definition → surface)
  scored against the input, plus λ-sweeping <definitionSparsityScale> against
  the fidelity bars on a real training config. Task 5 (docs + suite).
  Query-tool operational integration is the [ALEC — DESIGN] item above.

* .where recovery is still a placeholder in [bin/recon_bench.py (line 206)](/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel/bin/recon_bench.py:206).

* The execution notes explicitly say residual mismatch is still slot alignment and pad-slot decode pollution, with .where stamp/decode untouched: [execution notes (line 696)](/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel/doc/plans/2026-07-03-reconstruction-fidelity-execution.md:696).

* Wave brightness and Task 11 nVectors wiring are explicitly parked out of scope: [design scope (line 36)](/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel/doc/plans/2026-07-03-reconstruction-fidelity-design.md:36).

* COMPILE/PERF — fullgraph=True LANDED 2026-07-07 (was: Task 8 census residual). The serial per-word forward traces **fullgraph-clean (0 breaks / 1 graph, sentence protocol ON)** and the compile gate requests fullgraph=True for serial full-router configs on every device (the old MPS eager-skip is retired — its "inductor-MPS miscompiles (NaN)" premise was a test-harness device artifact, not a compile bug). What landed:
  - §6 mid-sentence reground REMOVED (Alec) — its per-word conflict-mass `.item()` reads were the only graph breaks, and it was dormant in training (the absolute truth store has no forward-path writers; sentence-START prelude kept, intent now sentence-scoped).
  - Per-vector sigma/pi mereological order-raise (fold width = percept_dim, not nOutput*D) — serial loop O(N²)→LINEAR in word count; XOR roundtrip pin IMPROVED 0.75→1.0 at E=3 (cross-word feature leak removed). data/MM_20M_fineweb.xml (N=64) processes WHOLE FineWeb sentences (~99% coverage vs 38% of words at N=8), ~3.9 sent/s MPS eager.
  - Host-stem hot spots: stage_analysis_spans/property_spans vectorized (37%→~0); embed_stem radix getW gather-then-STE via Codebook.lookup_rows (31%→2%); eager skip-padding loop (trip count = batch's longest sentence; static under trace).
  - Recompile churn KILLED (was a retrace per epoch, ~150-380s each on MPS): WS symbol codebook now grows ONCE to the XML nVectors budget (was cap-doubling 2→4→...→64, a Parameter re-registration each time) + pre-grown at compile-enlist; Optimizer.step restores the default device via canonical `str(TheDevice.get())` (a `get_default_device()` snapshot normalizes 'mps'→'mps:0' ≠ device('mps') and flipped the ambient guard); grad-off diagnostic passes (only `_reconstructionReport` — training verified 100% grad-on) route to the eager forward; `_prelude_pumps` pre-declared. Measured: 1 compile (~351s) + one 4.6s settle, then **0.17–0.40s/step vs ~1.0s eager (~2.6–6x)**.
  - Device coherence per util.TheDevice: whole-model `self.to(str(TheDevice.get()))` sweep at compile-enlist (lazy-built modules after the build-on-CPU→to(mps) dance); host-only reducer re-home gated by `is_compiling()`. util.compile MPS block: uint16→ushort DTYPE_TO_METAL patch + `max_fusion_unique_io_buffers` (env `BASICMODEL_MPS_IOBUF`, default 24; `BASICMODEL_MPS_FUSE` optional) for Metal's 31-buffer kernel-arg limit.
  REMAINING: (a) the N=64 whole-sentence graph still overflows 31 buffers on two fused reduction kernels at IOBUF=24 — tune down (12/8 run pending) or split reductions; (b) one `len(self._modules)` settle (a module registered during the first forward — one 4.6s retrace, cosmetic); (c) the `p == N` cursor guard / NULL-word + power-of-two loop-bucket pathway (doc/plans/2026-07-04-serial-derivation-reconstruction-design.md) for variable-length beyond the static-N contract. Compile-default policy (Alec): DEBUG = none/eager (skip the one-time compile); PRODUCTION = inductor past the amortization break-even.

* GPU/CUDA PORTABILITY (Task 8 GPU rung; GPU-first policy = everything on GPU by default; training target is now a cloud GPU — the GB10 was returned). CPU-Generator-vs-device bug class SWEPT and fixed across bin/ (Language.py:2366, References.py:58-62; all other generator sites classified clean). 2026-07-07: full MPS grammar epochs now run (65536-row PS codebook included — the old "MPS OOM on the PS codebook" blocker no longer reproduces), and the device-coherence class is handled per util.TheDevice (whole-model sweep at compile-enlist; Optimizer.step canonical restore; recon_bench `_build_model` remains device-incoherent for MPS — harness-only, produces NaN where the real ModelFactory path is clean). REMAINING: the `AttributeError: 'NoneType'.is_empty` at bin/Spaces.py:21315 (`outputSpace.forward`, a subspace returns None) on the non-CPU forward path — device-independent-looking, needs root-cause; and a CUDA smoke pass once a cloud GPU is borrowed (nothing MPS-specific should carry over: no 31-buffer limit, uint16 supported).

* The "Codebook.property_basis" is a hack that needs to be removed. Please summarize the WholeSpace property mechanism. You said properties "are" WholeSpace.what. But that codebook currently holds the whole/truth prototypes wired into the codebook-snap machinery; making properties the live .what semantics would rip that out and move the basin. So I built the property capability as opt-in/additive (Codebook.property_basis) alongside the existing whole codebook, not as a wholesale replacement. If you intended the live cutover, that's a separate deliberate step.
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
  * NOTE (2026-07-06): `invert_ramsified` now HAS a consumer — the order-k
    unfold (`Codebook.unfolded_prototypes` → the decode peel's `prototypes=`,
    exercised by `ConceptualSpace.typed_definition`). So the live stamping
    below is no longer dark scaffolding: stamped fold provenance feeds the
    idea→definition decode. The remaining gap is exactly the live stamping.
  * the ramsification record has to become part of the normal codebook contract, not an optional sidecar. Every PS/WS codebook row that can represent a percept, word, type, or higher-order definition should carry fold provenance: for each subsymbolicOrder pass, whether that row was produced through Sigma, Pi, or neither. The scalar “order” should remain a derived readout, abstraction_order(row) = count(non-NEITHER folds), not separately stored state.
  * The main missing wiring is live stamping. Today the table exists and higher-order minting can stamp some rows, but the actual subsymbolic pump loop does not consistently call record_fold when a row is routed through PS synthesis or WS analysis. That needs to move into the canonical forward path: whenever a codebook row is created, selected, raised, or rewritten by a sigma/pi pass, the corresponding fold slot for that pass should be updated. This makes order provenance a normal consequence of processing rather than a special feature behind mereologyRaise.
  * The opt-in flag can be removed once ramsification allocation is cheap and universal. Allocate the table for relevant codebooks at creation time using max_order = max(1, architecture.subsymbolicOrder), grow it with the codebook, and reset nothing on ordinary document boundaries. Keep it non-gradient metadata, but decide whether it should persist. If explicit-constraint retraining depends on it after reload, it should ride checkpoint metadata or a sidecar serialization path; leaving it out of state_dict is only safe while it is reconstructible from deterministic build/mint history.
  * Explicit-constraint retraining should consume the fold provenance directly. A constraint should resolve the affected lexical/concept row, inspect its fold sequence, and route the update through the matching inverse path using invert_ramsified or an equivalent per-pass inversion. That lets “this word’s abstract definition changed” update the high-order semantic representation without clobbering the low-order perceptual form. It also lets constraints target the right layer: raw token identity, basic category, count/type noun, or higher-order definition.
  * Finally, tests should enforce the canonical contract. Build a model without any mereologyRaise flag and assert PS/WS codebooks have ramsification tables. Run a multi-pass forward and assert selected or minted rows receive expected Sigma/Pi fold stamps. Verify abstraction_order is stable after codebook growth and reload, and add a small explicit-constraint training test showing that an abstract definition update changes the intended high-order row while preserving low-order reconstruction.

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
