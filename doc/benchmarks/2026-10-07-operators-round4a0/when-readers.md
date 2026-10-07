# Round 4a-0 `.when` reader inventory

Input sentence positions are **one-based**, following the October 3 amendment
in the occurrence specification §0. Each word and its committed sentence use
`0.5 × (Q(i) + Q(i+1))`, where Q uses the existing coarse/fine spatial ladder
frequencies and scale. Its fixed period exceeds the loaded document bound.
The absolute `when_time` buffer advances as before for `runBatch`; raw forwards
also tick it. Only the row timestamp receives that clock.

The live `.where` implementation remains 3a's onset ladder. This change uses
its frequency ladder and scale for `.when`'s requested endpoint bracket; it
does not change the form-space or `.where` transform.

The [machine-readable fixture](../../../test/fixtures/when-readers-round4a0.json)
covers all named band and band-width attribute/getattr readers in `bin/`,
plus the dynamic tail helpers and timestamp readers below. The focused test
fails if the named reader set changes. “No interpretation” means the function
carries, selects, validates, encodes or decodes a supplied band, or uses its
width; it never interprets a model clock. Its supplied band is now relative.
Teacher's `when` is a different, unchanged objective/source metadata API.

Dictionary poles and DEF rows are unlocated and have zero bands. Located
percept/concept occurrences carry the source bracket. Recency and expectation
selection continue to read timestamps/source metadata, respectively. Legacy
standalone codecs remain available; the live model uses DocumentWhenEncoding.
Conceptual tense/lift/lower remain 3a's opaque operations; before/after and the
situation code remain in 5.5. Reconstruction excludes the shared time tail
where it did in 3a; general event comparisons still transport/score it.

| Reader | Previously used absolute band time? | Now reads |
| --- | --- | --- |
| `bin/ClauseJournal.py:finish_clause` | Yes | Closing inherits its first leaf band; the sentence writer fixes the source bracket. |
| `bin/ClauseRow.py:Clause.__post_init__` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Interpret.py:InterpretLayer.define` | Yes | Definitions are unlocated; their when band is zero. |
| `bin/Language.py:_WhenOpMixin._when_encoding` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Language.py:_WhenOpMixin._split_when` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Language.py:SymbolSubSpace.forwardSymbols` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Language.py:SymbolSpace._publish_symbol_snapshot` | Yes | Dictionary poles have zero when; located occurrences retain their source bracket. |
| `bin/Layers.py:TernaryTruthStore.append_meaning` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Layers.py:TernaryTruthStore.row` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Layers.py:TernaryTruthStore.reset` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Layers.py:ModelLoss.compute` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Layers.py:ModelLoss.compute_masked` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Layers.py:ModelLoss.register` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Layers.py:ModelLoss.compute_piecewise` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Models.py:BasicModel._reverse_event_loss` | Yes | Sentence reconstruction excludes the shared when tail; general event comparison may score relative brackets. The scoring rule is unchanged. |
| `bin/Models.py:BasicModel._masked_event_loss` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Models.py:BasicModel._publish_reading_symbols` | Yes | Location matching and concept/percept binding read the relative interval. |
| `bin/Models.py:BasicModel._stage_serial_concept_rows` | Yes | Each word is stamped at its containing source sentence position. |
| `bin/Models.py:BasicModel._program_entries` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Models.py:BasicModel._commit_sentence` | Yes | The selected closing writes the source unit bracket; timestamp alone receives model time. |
| `bin/Models.py:BasicModel._leaf_distill_loss` | Yes | Sentence reconstruction excludes the shared when tail; general event comparison may score relative brackets. The scoring rule is unchanged. |
| `bin/PerceptField.py:PerceptField.select` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:EventEncoding.__init__` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:EventEncoding.set_band` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:Embedding.decode_reverse_meta` | Yes | Returns encoded address bands with recovered lexical metadata, without interpreting absolute time. |
| `bin/Spaces.py:SubSpace.__init__` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.storage_versions` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.carrier_like` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.set_demuxed` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace._compute_active` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace._coerce_basis` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.Start` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.End` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.set_when` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.demux` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.set_activation_from_event` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.mux` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.materialize` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.select` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.put` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.encode` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.decode` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.pad` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:SubSpace.slice` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:Space.reverseBegin` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:Space.set_sigma` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:InputSpace._lex_and_embed` | Yes | Stamps source-relative sentence positions; no model-clock input. |
| `bin/Spaces.py:PartSpace.__init__` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:PartSpace._radix_part_events` | Yes | Stamps source-relative sentence positions; no model-clock input. |
| `bin/Spaces.py:PartSpace._embed_radix` | Yes | Stamps source-relative sentence positions; no model-clock input. |
| `bin/Spaces.py:PartSpace.reverse` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:ModalSpace.__init__` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:ModalSpace.forward` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:ModalSpace.reverse` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:ConceptualSpace._clear_percept_field` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:ConceptualSpace._commit_autobind_from_stash` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:ConceptualSpace.cs_read_memberships` | Yes | Percept fields carry the current sentence unit interval for located binding. |
| `bin/Spaces.py:ConceptualSpace.decode_sparse_concept_rows` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:ConceptualSpace.decode_prior_stm_tensors` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:WholeSpace.compute_stage0_carrier` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:WholeSpace.compute_stage0_unity_event` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:WholeSpace.compute_word_property_event` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Spaces.py:WholeSpace._stage0_unity_forward` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Teacher.py:Teacher.What` | No interpretation | Teacher.when is corpus/source metadata, not the event band; unchanged. |
| `bin/Teacher.py:Teacher._resolve_staged_sources` | No interpretation | Teacher.when is corpus/source metadata, not the event band; unchanged. |
| `bin/Understanding.py:SentenceEndState.detached` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/WhereRegistry.py:install_where_registry` | No interpretation | Selects a fixed coarse/fine period above both LTM capacity and the loaded document bound. |
| `bin/pipeline.py:SubSpaceSchema.__post_init__` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/pipeline.py:SubSpaceSchema.event_width` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/pipeline.py:_parts_for_factored` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/pipeline.py:SubSpace._selected_parts` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/pipeline.py:SubSpace.materialize` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/pipeline.py:_payload_bytes` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/pipeline.py:_payload_transfer_bytes` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/pipeline.py:BoundaryAdapter.adapt` | No interpretation | Carries, slices, copies, validates or transports the relative bracket without interpreting its value. |
| `bin/Language.py:_event_split` | Yes | Separates when tail; its value is now relative. |
| `bin/Language.py:_lift_when` | Yes | The standalone symbolic helper shifts a relative bracket by one; conceptual lift is opaque. |
| `bin/Language.py:_lower_when` | Yes | The standalone symbolic helper shifts a relative bracket by minus one; conceptual lower is opaque. |
| `bin/Spaces.py:DocumentWhenEncoding` | Yes | Encodes/decodes [i,i+1] with two endpoint-sum rungs; shift_time is a relative rotation. |
| `bin/Spaces.py:WhenRangeEncoding / WhenStartDurationEncoding` | Yes | Legacy standalone codecs retained for explicit old encodings; the model uses DocumentWhenEncoding. |
| `bin/Layers.py:TernaryTruthStore.recent` | No interpretation | Reads the timestamp column (absolute model time), never the when band. |
| `bin/MereologicalCodes.py:MereologicalCodes.occurrence_terms` | No interpretation | Recency still reads timestamp; meanings remain empty in the 4a-0 gate configuration. |

## Runtime field-name dispatch

These additional paths access a band or its carrier by a variable field name. They are included in the same fixture; explicit conceptual scope is distinguished from the event band.

| File and function | Earlier absolute-time interpretation | Current read |
|---|---|---|
| `bin/AccessibleMind.py: apply_thought_effect` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/ClauseRow.py: ClauseRows.write_clause` | No | Copies the completed clause band and writes its address; external input closings supply the document-relative bracket. |
| `bin/Language.py: LanguageLayer.reduce` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Language.py: SymbolSpace.forward_concept_to_symbol` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Layers.py: TernaryTruthStore.__init__` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Layers.py: TernaryTruthStore._load_from_state_dict` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Layers.py: TernaryTruthStore.clear_origin` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/MemoryIndex.py: LeafCodeIndex._scope_contains` | No | Compares explicit conceptual scope intervals named when; this is separate from the event band and remains unchanged. |
| `bin/Models.py: BaseModel._collect_structural_extras` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Models.py: BaseModel._restore_structural_extras` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Models.py: BaseModel.load_weights` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Models.py: BasicModel._synthesis_state_objects` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Models.py: BasicModel._forward_body` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Occurrence.py: OccurrenceRows._resize_checkpoint_rows` | Yes, migration only | Migration/resize reconstructs unit brackets from saved relative positions or explicit legacy ordinals; timestamps are preserved separately. |
| `bin/Occurrence.py: OccurrenceRows._migrate_address_tensors` | Yes, migration only | Migration/resize reconstructs unit brackets from saved relative positions or explicit legacy ordinals; timestamps are preserved separately. |
| `bin/PerceptField.py: PerceptField.clone` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Spaces.py: SubSpace.normalize` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Spaces.py: SubSpace._apply_normalization` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Spaces.py: SubSpace._apply_reverse_normalization` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Spaces.py: SubSpace.denormalize` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Spaces.py: Space.__init__` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Spaces.py: Space._adopt_subspace_modules` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Spaces.py: ConceptualSpace.forward` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/Understanding.py: SentenceEndState.__post_init__` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |
| `bin/space_carrier.py: SpaceCarrierMixin._carrier_basis` | No | Selects, validates, copies or restores the band through a runtime field name. No clock interpretation; event bands now carry the source-relative bracket. |

Inventory: 75 named entries and 32 dynamic entries.
