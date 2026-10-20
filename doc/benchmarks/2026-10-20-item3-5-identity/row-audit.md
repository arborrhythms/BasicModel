# Step 1b row audit (§6.3)

This is the required field-by-field disposition for the **step-2 LTM1/LTM3 conversion**. Step 1b precedes that conversion: it retires identity dictionaries/count tables, makes STM an address index, and changes new row timestamps now. It does not claim the existing TernaryTruthStore has already become the two decided stores. The transitional reference and metadata columns below remain explicit step-2 work, not an alternate identity architecture.

The new `.when`/orders amendment in the working tree is included unchanged. At concept order the stored clock cannot enter composition; occurrence lowering is the place to reattach it. Step 2 must carry the symbol's situated `.where`/`.when`, the concept's spatial extent and the document/world whole through the final two-store representation.

## Every tensor field and store buffer

| Field | Disposition | Reason / execution boundary |
|---|---|---|
| `slots` | keep | LTM1 one concept; LTM3 three symbol references. Physical split is step 2. |
| `refs` | move | The three LTM3 slots own references; LTM1 has no identity column. Step 2 removes the transitional NP identity column used by this migration. |
| `when` | keep | Model processing time, with a two-rung sinusoidal ladder. RowTime stamps new rows; situated_point never passes this column to composition. |
| `c_plus` | keep / rename | truth_plus at the step-2 layout conversion; preserve independent positive evidence. |
| `c_minus` | keep / rename | truth_minus at the step-2 layout conversion; preserve independent negative evidence. |
| `witness_count` | keep / rename | Access count at the step-2 conversion; supplies retention and recurrence. |
| `surprise` | keep | The attended residual at write cannot be reconstructed later; item 5 reads it. |
| `rel_type` | delete | LTM3 kind is its middle reference; derive arity from the native store form at step 2. |
| `stream` | move | Reader history plus a context whole; no stream column in the decided row. |
| `order` | delete | Read the concept order from the shared conceptual index. |
| `source_trust` | move | Attribution in LTM3, or fold the supplied authority into lanes at write. |
| `origin` | move | Attribution/provenance owner; no identity feature. |
| `row_ids` | delete | LTM3 references symbols; native store indexes own physical locations. |
| `where` | delete column | The concept retains its spatial extent; occurrence readers obtain the situated symbol band. The physical column conversion is step 2. |
| `role_mask` | delete | Derive occupied slots from the LTM1/LTM3 form and open references. |
| `record_kind` | move | Predictor owns estimates and thought/query state owns pending questions; evidence belongs in the lanes. |
| `grammatical_mode` | delete | A question is an open reference. |
| `polarity` | delete | Negative stance exchanges lanes; no independent polarity column. |
| `address_keys` | delete | Context whole plus native row/symbol index replaces the source-hash column. |
| `document_keys` | move | Document and world are context wholes, as decided in occurrence §12–§14. |
| `sentence_content_keys` | delete | Content equality uses native regions and references, not a second hash column. |
| `sentence_index` | delete | Read order from model time and the reader history. |
| `metadata_required` | delete | A complete native row must be interpretable without a required arbitrary sidecar. |
| `semantic_fingerprint` | cache | Rebuildable validation index outside the row. |
| `leaf_complete` | cache | Derived decoding/index completeness; recompute from retained native structure. |
| `posting_codes` | cache | Inverted code index over native rows. |
| `posting_roles` | cache | Role portion of the inverted index. |
| `posting_rows` | cache | Native row targets in the inverted index. |
| `adjacent_left` | cache | Left member of an inverted adjacency index. |
| `adjacent_right` | cache | Right member of an inverted adjacency index. |
| `adjacent_address` | cache | Native adjacency occurrence target. |
| `adjacent_position` | cache | Position component of the adjacency index; never a retained forward parse. |
| `count` | keep owner state | Store occupancy scalar, not a field of a row. |
| `model_clock` | keep owner state | Monotone writer fallback for isolated stores; live model writes read present(). Not a row column. |
| `forget_passes` | keep owner state | Item 5 scheduler counter, not a row field. |
| `forget_sentences` | keep owner state | Item 5 observation counter, not a row field. |
| `forget_last_sentence` | keep owner state | Item 5 schedule cursor, not a row field. |
| `forget_cutoff` | keep owner state | Item 5 policy value, not a row field. |
| `truth_schema` | keep transport state | Checkpoint schema marker, not conceptual row content. |

## Sidecar keys, envelopes and derived indexes

| Key | Disposition | Reason |
|---|---|---|
| `_texts` | delete | No source-text copy in the row; generating/querying uses native values and references. |
| `_semantic_rows` | cache | Rebuildable native semantic index after the step-2 representation is authoritative. |
| `_definition_rows` | cache | Rebuildable DEF index over LTM3 references. |
| `_expectation_rows` | cache | Rebuildable predictor index over retained estimates/observations; predictor owns the target. |
| `role_refs` | delete sidecar | LTM3 native slots are authoritative; open references derive from them. |
| `bindings` | split / move | Only individually justified keys below remain with their owners. |
| `scope` | move | Attribution/world in the native context structure, not an arbitrary row envelope. |
| `_open_references` | derive | Native reference slots determine which roles remain open. |
| `_bound_roles` | derive | Native filled slots determine bound roles. |
| `_forward_references` | derive | Forward handles are reference state owned by bind/native slots. |
| `_reference_sites` | move | Bind training/operation state; no second candidate-region store. |
| `_reference_mentions` | move | Bind owns detached mention-ranking sufficient statistics and their labels. |
| `_individual_context` | move | Bind owns role/derivation training context; native rows and concept seats own content. |
| `_individual_occurrences` | cache | Rebuildable occurrence index over the native binding relation. |
| `_expected_symbols` | move | Predictor owns forecasts. |
| `_observed_symbol_activations` | move | Predictor owns its detached observation target. |
| `_withheld_symbols` | move into concept content | The per-dimension withheld mark is content beside the lanes, per the October 20 amendment; it is neither a row column nor negative content. Step 2 removes this sidecar representation. |
| `_omitted_symbol` | move | Predictor's omitted-identity query state; it is not descriptive content. |
| `_symbol_handles` | cache | Exact native identity handles from retained references, never a closest-code assignment. |
| `_evidence_pair` | delete duplicate | Truth lanes are authoritative. |
| `_source_evidence` | move | Writer/attribution input; fold into lanes at commit. |
| `_equality` | derive | Native relation kind and operands determine equality. |
| `_direct_binding` | derive | Native references determine direct binding. |
| `_formation_records` | move | Bind/chooser training provenance; excluded from choice features. |
| `_formation_reason` | move | Bind/chooser explanation, not row identity. |
| `_point_constructor` | derive | Native concept structure determines the constructor. |
| `_producing_operation` | move | Thought/derivation owner; no separate identity. |
| `_thought_witnesses` | cache | Query retained native proof references. |
| `_original_bindings` | delete | No parallel old binding state after a rewrite. |
| `_query_occurrence` | move | Ephemeral reader/query cursor. |
| `_query_relation` | move | Ephemeral query operation state. |
| `_query_exclude` | move | Ephemeral query budget/scope. |
| `_write_address` | delete | Step-2 context whole and native reference replace the source-address cursor. |
| `_pending` | delete | Open references express pending state. |
| `_record_revision` | delete | Native cache invalidation has no row identity meaning. |
| `_current_gate` | delete | Execution-local state, not a row field. |
| `_live_*` | delete | Execution-local state, not a durable sidecar. |
| `_leaf_postings` | cache | Derived inverted index, not a forward sentence trace. |
| `_adjacent_postings` | cache | Derived adjacency index. |
| `_index_occurrences` | cache | Native address/index lookup. |
| `_slot_index_source` | cache | Rebuild native open-slot index. |
| `_forward_index_source` | cache | Rebuild forward-reference index. |
| `_native_predicate_index` | cache | Rebuild native relation-kind index. |
| `_native_reference_token` | cache | Reference transport, not descriptive content. |
| `_native_row_index` | cache | Rebuild native row lookup. |
| `namespace` | keep transport state | Checkpoint/native-reference domain, not a content column. |
| `version` | keep transport state | Sidecar migration version, not a row field. |
| `records[].id` | move | Native cache key; redundant content-address columns disappear at step 2. |
| `records[].text` | delete | Same disposition as _texts. |
| `records[].context` | split / move | Same individually enumerated native semantics and binding state above. |
| `records[].expectation` | move | Predictor-owned estimate/observation pairing and target. |
| `records[].definition` | cache | Rebuild the native DEF index. |

Nested binding and prediction records have the following individual dispositions;
moving their envelope does not grant permission to keep copied identity regions.

| Nested key | Disposition | Reason |
|---|---|---|
| `_reference_sites[].role` | move | Bind site role; native slot occupancy remains authoritative. |
| `_reference_sites[].leaves` | move | Open bind's reverse-face input, released with the operation state. |
| `_reference_sites[].cue` | move / reference | Bind points to the native form; no durable copied form vector after conversion. |
| `_reference_sites[].content` | delete copy | Read the immutable NP occurrence slot. |
| `_reference_sites[].predicate` | move / reference | Read the native predicate/verb reference. |
| `_reference_sites[].occurrence` | move | Bind site address, not an identity region. |
| `_reference_mentions[].candidates` | move | Bind's detached supervised menu. |
| `_reference_mentions[].features` | move | Training sufficient statistics only; never thought-choice inputs. |
| `_reference_mentions[].cue` | move / reference | Training form identity, with the native form as owner. |
| `_individual_context.role` | move | Bind's recorded derivation role. |
| `_individual_context.position` | derive | Reader order and model time replace the duplicate cursor. |
| `_individual_context.document` | move | Native context whole replaces the document hash. |
| `_individual_context.predicate` | move / reference | Native LTM3 predicate reference; participant cache reads it. |
| `_individual_context.mint` | move | Bind's formation provenance for revision credit. |
| `_individual_context.initial` | move | Bind's original membership provenance, excluded from scoring. |
| `_formation_records[].choice` | move | Chooser training provenance; not a content feature. |
| `_formation_records[].probability` | move | Chooser's detached credit record. |
| `_formation_records[].reference` | move | Choice provenance, with native slots authoritative. |
| `_formation_records[].role` | move | Choice-site address for credit. |
| `expectation.kind` | move | Predictor distinguishes its estimate and observed target. |
| `expectation.source_occurrences` | move | Predictor's causal inputs, held as native addresses. |
| `expectation.stream` | move | Predictor/reader stream owner. |
| `expectation.document` | move | Reader context whole, referenced by the predictor. |
| `expectation.intended_occurrence` | move | Predictor target address; preserve pairing. |
| `expectation.presence_logits` | move | Predictor's forecast, not observed row content. |
| `expectation.estimate_occurrence` | move | Predictor's inverse pairing address. |
| `expectation.previous_estimates` | move | Predictor owns prior forecasts that targeted the occurrence. |

## What changes now

`model_clock` is store-owner state. New native writes stamp the live model's `present()` counter; isolated mechanism stores use their writer counter. The ladder declares 0 through 2^32−1 ticks (over 136 years at one batch per second), asserts its horizon and is checked at both ends and interior run-scale values. Re-witnessing retains the original occurrence time. A pre-1b checkpoint did not retain model processing times: its old source-position phasors become explicitly unknown zero phasors in one load migration. Its addresses and content remain unchanged; no processing times are fabricated.

`IndependentComponents.situated_point` no longer injects `frame.when` into a compose operand. The regression changes that column while requiring the complete returned compose operand to remain bitwise identical. Pending predictor inputs remain detached forecast inputs; committed STM entries contain only typed occurrence addresses, and every region read follows the current retained row.

## Required next landing

Step 2 performs the LTM1/LTM3 physical split, the field removals/moves/renames in this table, recurrence minting, document/world context wholes and occurrence lowering with the symbol's location and model time. The current step-1b transitional NP reference column is explicitly removed there. These layout obligations are separate from the closed step-0 invariants and from the carried Item 0 / 5.5 learning gates.

## Added audit note: the surviving intra-sentence predictor

The working-tree October 20 amendment also calls out
`IntraSentenceLayer` / `intraLossWeight`. Inspection confirms that they
remain live: `Layers.IntraSentenceLayer` predicts a concept-width working
slot with PI/Sigma and a routing bias; `ConceptualSpace` accumulates its
loss; `Models` consumes it under the `intra` objective. It is not the
shared symbol-bank activation predictor. There is no architectural
justification in this receipt for retaining that extra predictor.

Its retirement (parameters, loss plumbing, configuration and checkpoint
migration), or an explicit accepted justification, remains required
before the one-predictor claim can be made. This held candidate records
the ownership and the unresolved disposition; passing the current
reference gates does not settle it. It is carried explicitly in todo,
separate from the Item 0 and 5.5 learning gates.
