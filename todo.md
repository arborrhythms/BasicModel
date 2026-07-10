OPEN QUESTIONS / FUTURE WORK (Alec's calls + next builds):

* Wave brightness and Task 11 nVectors wiring are explicitly parked out of scope: [design scope (line 36)](/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel/doc/plans/2026-07-03-reconstruction-fidelity-design.md:36).

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
  * This work satarted as thinking_kernel_spec


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
