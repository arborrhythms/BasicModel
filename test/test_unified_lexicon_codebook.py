"""Stages 1.D + 1.B (combined): Unified lexicon codebook on WholeSpace.

==========================================================================
PHASE 1 MIGRATION PROPOSAL  --  STATUS: NEEDS_CONTEXT (paused for review)
==========================================================================

This file is a placeholder for the Phase 2 TDD gate.  Phase 1 is recon
only -- nothing in this file yet executes; the contents below are the
written proposal the controller is asked to approve before Phase 2
implementation begins.

--------------------------------------------------------------------------
RECON FINDINGS (the six areas the dispatch enumerates)
--------------------------------------------------------------------------

1. CURRENT SS.codebook STRUCTURE
   - File: ``bin/Spaces.py`` ``class WholeSpace(Space)`` at line 9807.
   - Constructed in ``WholeSpace.__init__`` (line 9915). The codebook
     lives on ``self.subspace.what`` and is built by ``_build_what_basis``
     (line 10389).  Modes: ``"none"`` -> Tensor passthrough,
     ``"project"`` -> ``ProjectionBasis`` (LDU-factored), ``"quantize"``
     -> ``Codebook`` (the default in MM_xor, MM_xor_loopback, MM_20M,
     MM_grammar -- every config we touch).
   - What it stores: symbol prototypes -- one ``[symbol_dim]`` row per
     symbol slot.  Width = ``self.nDim`` (post-2026-05-07 rollback; the
     leading [pos,neg] bivector was dropped).  Per-row metadata buffers
     allocated in ``Codebook.create(..., category=True)`` (lines 1820-1830):
       * ``category_ids: [V] long`` -- per-row POS/category tag (0 = '?').
       * ``part_parents: [V] long``  -- per-row meronymic parent (-1 root).
       * ``category_logits: [V, C] float`` -- lazy POS-prior buffer.
   - "Well-known atoms" registry (line 10077): name -> codebook-row dict.
     Row 0 is reserved as the ``"words"`` meronymic parent.  Saved/loaded
     in the .ckpt bundle as ``vocab_extras["well_known_atoms"]``.
   - Indexing:  ``Codebook.lookup(indices)`` gathers rows by integer
     index; ``set_part_parent(idx, parent_idx)`` writes the meronymic
     parent.  Nearest-row search runs through ``self.vq`` (VectorQuantize).
   - Dim of entries: ``self.nDim`` (symbol_dim, validated equal to
     ``concept_dim`` at config-load).

2. CURRENT InputSpace TEXT-MODE CODEBOOK
   - File: ``bin/Spaces.py`` ``class InputSpace(Space)`` at line 6354.
   - ``InputSpace._build_what_basis`` (line 6407) explicitly returns
     ``None`` when ``model_type == "embedding"``:  the Embedding (lexicon)
     is OWNED BY PartSpace.  IS only carries non-lexical bases
     (Codebook / Tensor for numeric modes).
   - InputSpace's text-mode contribution to the lexicon is the LEXER:
     ``_lex_batch`` (line 6621).  It takes raw bytes, calls the peer's
     ``vocabulary._token_stream`` (line 6688) -- i.e. the tokenizer
     currently lives on the lexicon (the ``Embedding`` instance).
     IS produces ``(what_buf, where_idx, when_idx, host_tokens)``:
     null-terminated UTF-8 byte buffer + offsets, no vector lookup yet.
   - The MIGRATION TARGET: the BPE / Embedding / WordVectors structure
     that today resolves a surface byte buffer to a vector
     (PartSpace._embed / _embed_bpe / _embed_mphf -- see #3 below)
     moves OFF the input-pipeline side and onto SS.  IS's lexer is
     pure-byte and that's where it stays; the LOOKUP (bytes -> vector)
     is what migrates.
   - "InputSpace text-mode codebook" is largely a misnomer in current code:
     IS never owned a text codebook directly post-2026-05.  What IS owns
     today (and is in scope to retire / relocate) is the WIRING:
     ``self._peer_perceptual`` (line 6473), the ``vocab =
     self._peer_perceptual.vocabulary`` traversal in ``_lex_batch``
     line 6640, and the assumption that the lexer is reachable through
     ``perceptualSpace.vocabulary``.  After the migration the lexer
     reaches the lexicon through ``wholeSpace.codebook`` (or
     equivalent SS-side accessor).

3. CURRENT MPHF INFRASTRUCTURE
   - File: ``bin/Layers.py`` ``class MPHFGpuLayer(Layer)`` at line 7464.
     Pure static (poly-hash + searchsorted + byte-verify); zero
     host-sync; non-invertible (reverse is table lookup, never inverse
     hash).
   - Build-once cache lives on PartSpace: ``self._mphf_gpu_layer``
     (line 7439), ``self._mphf_static_tables`` (line 7448).
   - Access path: ``PartSpace._mphf_codebook`` (line 8360) returns
     ``self.subspace.what`` when it is an Embedding (today's lexicon
     location).  ``_mphf_tables`` (line 8369) builds the static tables
     against that codebook.  ``mphf_index`` (line 8388) does the
     byte-slots -> row resolve.  ``mphf_table_rows`` (line 8403)
     gathers the rows.
   - Runtime entry: ``PartSpace._embed_mphf`` (line 8540) is the
     ``<synthesis>mphf</synthesis>`` path; ``_embed_bpe`` (line 7776) is
     the BPE / trie path that also resolves through ``self.subspace.what``.
   - MIGRATION: PS keeps the MPHF *algorithm* (``_mphf_gpu_layer``,
     ``_mphf_static_tables``) but the *target codebook* shifts from
     ``self.subspace.what`` to a reference at ``self.codebook`` (the
     SS-owned Embedding via the new lookup contract).

4. CURRENT MEREOLOGY APIs (read-only ``bin/Mereology.py``)
   - The ``Mereology`` mixin is back-projection machinery; it provides
     measure operators (Contiguous, Continuous, Area, Luminosity,
     Peaceful, isIsomorphic) plus the reverse walker (hoc_shape,
     _walk_reverse, _derivation_path, _leaf_path_trust).  It is NOT
     where parthood / intersection / equality live.
   - The actual mereological PRIMITIVES live in ``bin/Layers.py``:
       * ``Ops._part_kernel`` (line 9692): part(x, y) scalar /
         vector form.  Scalar form is clipped cosine in [0,1].
       * ``Ops._equal_kernel`` (line 9731): mutual parthood ==
         intersection of part(x,y) and part(y,x).
       * ``Ops.overlap`` (line 9744): elementwise min of part forms.
       * ``Ops.intersection`` (line 9018), ``Ops.union`` (line 9036),
         ``Ops.whole`` (line 9720), ``Ops.partOf`` (line 9816),
         ``Ops.wholeOf`` (line 9828), ``Ops.overlapOf`` (line 9839).
       * Per-row meronymic links on Codebook itself via
         ``set_part_parent`` (line 1871) / ``get_part_parent`` /
         ``part_parents`` buffer (lines 1820-1830).
   - The LOOKUP CHAIN proposal (see step 4 of the user prompt) uses:
       (a) ``part_parents`` to navigate orthographic -> meta-symbol
           ("words" or future per-word meta-symbols).
       (b) ``Ops.partOf`` / ``Ops.intersection`` to isolate the
           semantic child within the meta-symbol's children set.
   - No new mereology API is needed; the migration consumes the
     existing surface.

5. CURRENT TEXT-MODE BPE / ChunkLayer (read-only ``bin/Layers.py``)
   - ``class ChunkLayer(Layer)`` at line 6692.  Holds:
       * ``self.merges: list[(int, int)]`` (BPE merge table).
       * ``self.vocab: dict[tuple[int,...], int]`` (chunk byte-tuple ->
         chunk-id).
       * ``self.id_to_bytes: dict[int, tuple[int,...]]``.
       * Boundary set ``BOUNDARY_BYTES``.
       * Growth counters / pair / unigram Counters.
   - Lives on ``PartSpace.chunk_layer`` (built at line 7413).
     Save/load through ``embed.bpe_section_from_chunk_layer`` /
     ``embed.save_artifact`` (lines 6786, 6818); ckpt bundle stashes
     pure-Python state as ``bpe_extras`` (Models.py line 1142).
   - "Pre-trained merges + per-chunk vector lookup" semantics are
     currently split:
       * ChunkLayer emits chunk-byte-tuple keys.
       * Embedding (= Lexicon) stores the per-key vector.
   - MIGRATION: ChunkLayer itself is read-only ``bin/Layers.py``; it
     STAYS THERE.  The PartSpace-side wiring of where ChunkLayer
     emits its keys (currently: ``self.subspace.what`` Embedding via
     PartSpace._embed_bpe_trie line 7847) is what changes -- the
     keys land in the SS-owned Embedding instead.

6. EXISTING TESTS THAT AFFIRM CURRENT OWNERSHIP
   Grep for ``perceptualSpace.vocabulary`` / ``perceptualSpace.subspace.what``
   surfaced (this is "what existing tests will break" -- not all need
   modification, but the contract-on-ownership ones do):

     * ``test/test_lexicon_ownership.py``:
         test_embedding_lives_on_perceptual_space  -- asserts
         ``isinstance(m.perceptualSpace.vocabulary, Embedding)``.
         test_get_embedding_returns_perceptual     -- asserts
         ``m._get_embedding() is m.perceptualSpace.vocabulary``.
         test_output_space_text_mode_reads_perceptual -- asserts
         ``m.outputSpace._vocabulary is m.perceptualSpace.vocabulary``.
         test_optimizer_includes_emb_params_when_trainable / _excludes_*
         -- ptr-equality on ``m.perceptualSpace.vocabulary.wv._vectors``.
         (These tests are the contract that today's Embedding lives on PS;
         post-migration they need to be updated to "physical OR logical
         on SS" assertions, or relocated to the equivalent SS surface.)
     * ``test/test_basicmodel.py`` lines 2050-2061: ``assertIsInstance(
         m.perceptualSpace.vocabulary, Models.Embedding)`` and
         ``emb_weight = m.perceptualSpace.vocabulary.wv._vectors``.
     * ``test/test_xor_spaces.py`` lines 173, 181: ``emb =
         self.model.perceptualSpace.vocabulary``.
     * ``test/test_perceptual_loopback.py`` lines 268-275: asserts
         ``sym.perceptualSpace_ref`` is wired (structural ref).  POST-
         migration this ref stays (PS keeps lexer wiring it does not need
         to know who owns the lexicon -- the back-reference becomes the
         API hook).
     * ``test/test_perceptualspace_bpe_forward.py``,
       ``test/test_perceptual_chunking.py``,
       ``test/test_serial_mode_perceptual.py``:  exercise the embed_bpe /
       embed_mphf paths through ``self.subspace.what`` directly.  These
       will need to read from the new MPHF target (PS.codebook ->
       SS.codebook).
     * ``test/test_lexicon_unit_ball.py``,
       ``test/test_chunk_layer_bpe.py``,
       ``test/test_perceptualspace_bpe_forward.py``: ChunkLayer-only
       tests; unaffected (ChunkLayer lives on PS still).

--------------------------------------------------------------------------
MIGRATION PROPOSAL  --  CONCRETE CHANGES PER FILE
--------------------------------------------------------------------------

A.  ``bin/Spaces.py``  --  WholeSpace
    A1. Extend ``WholeSpace.__init__`` to BUILD the Embedding (the
        orthographic + semantic + meta-symbol unified codebook) on
        ``self.codebook`` when ``model_type == "embedding"`` AND the
        InputSpace lexer is text-mode.  This is the "SS owns the
        unified codebook structure" line item.  The existing
        ``self.subspace.what`` Codebook (symbol-prototype) STAYS:
        the orthographic + semantic + meta-symbol Embedding is a
        SEPARATE attribute (``self.codebook``) sized to the lexicon
        capacity (~65k+).
    A2. Add ``WholeSpace.codebook`` (the new physical Embedding
        attribute).  The existing ``self.codebook`` boolean property
        on ``Space`` (line 6113) is a NAME COLLISION: that property
        means "is a codebook present" and is consulted by ~25 sites
        across the codebase (see grep above).  Resolution: rename the
        new attribute to avoid the collision.  Two options:
          (a) ``WholeSpace.word_codebook``  -- explicit, narrow scope.
          (b) ``WholeSpace.lexicon``       -- aligns with Lexicon.md
                                                  doc terminology and
                                                  ``WholeSpace.vocabulary``
                                                  already exists as the
                                                  read-side property.
        Recommendation: use ``self.lexicon`` (the Embedding instance) on
        SS, keep ``self.vocabulary`` as the property that returns it.
        ``PartSpace`` then exposes ``self.codebook`` as the
        *reference back to* ``wholeSpace.lexicon`` (resolving the
        spec's "PS.self.codebook" requirement without colliding with
        ``Space.codebook`` -- on PS the attribute is set as a direct
        ``object.__setattr__`` from Models.py, so it shadows the
        property without breaking inheritance for other Spaces).
        NOTE TO REVIEWER: please confirm naming preference before
        Phase 2.  If you prefer the spec text literally ("PS gains
        self.codebook"), we can shadow on PS only and use
        ``WholeSpace.codebook`` on SS too -- but that does collide
        with the ``Space.codebook`` bool property on SS (which is
        consumed at lines 10153 / 10689 / 10755 inside WholeSpace
        itself).  Renaming SS's attribute to ``lexicon`` keeps that
        boolean working while still giving SS ownership.
    A3. Add the orthographic-vs-semantic vs meta-symbol partitioning:
        the Embedding row metadata already supports ``part_parents``,
        ``category_ids`` (line 1820-1830 in Codebook).  Convention:
          * ``part_parents[i] == words_atom_id``  -> orthographic entry.
          * ``part_parents[i] == meaning_atom_id`` -> semantic entry.
          * ``part_parents[i] == -1`` (root)       -> meta-symbol.
        Add a new well-known atom ``"meanings"`` alongside the existing
        ``"words"`` (line 10077).  The meta-symbol-per-word
        construction (pair of orthographic-row + semantic-row sharing
        a parent meta-row) is the artifact this stage creates and
        tags via ``set_part_parent`` at insert time.
    A4. Migrate the input-pipeline-side methods: ``train_embeddings``,
        ``sbow_loss``, ``set_embedding_sigma``, ``reconstruct_*``,
        ``get_recovered_word`` are ALREADY DEFINED on SS (lines
        10337-10383) and forward through ``perceptualSpace_ref``.
        Post-migration they read from ``self.lexicon`` directly; the
        forwarding becomes vestigial (delete it).
    A5. Add the lookup chain implementation as ``self.resolve_surface``
        (or similar) on SS.  Signature:
            resolve_surface(byte_slots) -> semantic_vectors
        Body (per the controller's enumeration):
          step1 = PS-side ``_mphf_gpu_layer.mphf_index(byte_slots,
                  self.lexicon_mphf_tables)``  -> orthographic-row idx.
          step2 = self.lexicon.part_parents[step1]  -> meta-symbol idx.
          step3 = the meta-symbol's children list = the row indices
                  whose ``part_parents`` equals the meta-symbol idx.
                  INTERSECT against the "meanings" set (= rows whose
                  meta-symbol parent has the meanings tag) -> the
                  semantic-row idx.  This uses ``Ops.partOf`` /
                  ``Ops.intersection`` (scalar form) on the
                  part_parents long buffer (cast to floats) plus a
                  meaning-mask buffer.
          step4 = self.lexicon.lookup(semantic_idx) -> semantic
                  embedding vector.
        Per-batch fast path uses a precomputed
        ``orthographic_to_semantic_idx`` long buffer (one-to-one map
        built at lexicon insert time) so the per-slot cost is one
        gather, not a runtime intersection.  The intersection logic
        runs ONCE per insert (at codebook construction / OOV
        promotion), not per forward.

B.  ``bin/Spaces.py``  --  InputSpace
    B1. ``InputSpace._lex_batch`` (line 6621): replace
        ``vocab = self._peer_perceptual.vocabulary`` with
        ``vocab = self._wholeSpace_ref.lexicon`` (new structural
        ref wired in Models.py).  IS keeps the lexer step; it just
        reaches the tokenizer-bearing Embedding through SS now.
    B2. Retire ``self._peer_perceptual`` (line 6473) as the lexicon
        anchor.  PS still has lexer-side coordination (the Embedding's
        tokenizer is invoked from IS; PS doesn't own the lookup
        target anymore).  Replace with ``self._wholeSpace_ref``.
    B3. ``InputSpace._build_what_basis`` (line 6407) already returns
        ``None`` for embedding mode -- no change.  IS stays free of
        the lexicon.

C.  ``bin/Spaces.py``  --  PartSpace
    C1. ``PartSpace._build_what_basis`` (line 7535): for
        ``model_type == "embedding"``, return ``None`` instead of
        building an Embedding.  PS subspace.what becomes empty (or a
        passthrough Tensor) for embedding mode; the Embedding lives
        on SS.
    C2. PS gains ``self.codebook`` -- a non-property attribute set
        post-construction in Models.py to ``wholeSpace.lexicon``.
        Set via ``object.__setattr__(ps, 'codebook',
        wholeSpace.lexicon)`` (the property shadow pattern --
        breaks ``ps.codebook`` *as a bool* on PS only; PS uses it as
        an Embedding reference after the migration).
    C3. ``_mphf_codebook`` (line 8360): return ``self.codebook``
        instead of ``self.subspace.what``.
    C4. ``_embed`` (line 7650), ``_embed_bpe_trie`` (line 7847),
        ``_embed_mphf`` (line 8540), ``_embed_byte`` (line 8449):
        replace each ``codebook = self.subspace.what`` with
        ``codebook = self.codebook``.  These methods become pure
        lookup-through-SS resolvers.
    C5. PS retains ``self.chunk_layer`` (the BPE merge table -- the
        keyspace), ``self._mphf_gpu_layer`` (the algorithm), and
        ``self._mphf_static_tables`` (built against SS.lexicon now).
        These are PS's MPHF infrastructure -- they tokenize and
        index, and call into SS for the row vector.

D.  ``bin/Models.py``
    D1. ``BasicModel._create_per_stage`` (line 3919, around 4076-4280):
        WIRE the new refs in this order:
          (a) Build IS, PS, CS, SS as today.
          (b) After SS is built, build ``ws.lexicon`` (the Embedding)
              -- this is a refactor of the eager call in
              ``PartSpace.__init__`` that today builds
              ``self.subspace.what`` Embedding.  Move that build into
              the SS init or a post-init Models.py step.
              Recommendation: build inside ``WholeSpace.__init__``
              so SS is the structural owner, but SS init reads the
              configured PS knobs (embedding_path, byte_mode, etc.)
              -- supplied via the existing
              TheXMLConfig.space("PartSpace", ...) reads (the
              XML schema doesn't change; SS init reads from a
              different XML section).
          (c) Wire ``object.__setattr__(inputSpace, '_wholeSpace_ref',
              ws)``.  ``object.__setattr__`` so it is not registered
              as an nn.Module child (no double-counting params).
          (d) Wire ``object.__setattr__(perceptualSpace, 'codebook',
              ws.lexicon)`` (note: PS.codebook is the bool property
              today; this shadowing makes it the Embedding instance
              on the PS instance level).
          (e) Wire ``object.__setattr__(ws, 'perceptualSpace_ref',
              perceptualSpace)``  -- this already exists (line 4256);
              keep.  The structural pair survives -- PS owns lexer
              wiring, SS owns the lexicon.
    D2. ``BaseModel._get_embedding`` (line 921): change to read from
        ``self.wholeSpace.lexicon`` first, fall back to
        ``self.perceptualSpace.vocabulary`` for non-text models.
    D3. ``BaseModel.save_weights`` / ``_collect_vocab_extras`` (line
        1120): swap the source from
        ``emb = self._get_embedding()`` (already returns SS's via D2;
        no change needed beyond D2).
    D4. OutputSpace construction (line 4275): the
        ``vectors=self.perceptualSpace.vocabulary`` arg becomes
        ``vectors=self.wholeSpace.lexicon``.  Or, since
        ``perceptualSpace.vocabulary`` already resolves correctly via
        the existing property chain (line 6121), this is no-op IF
        the property is updated to read through the codebook ref.
        Cleaner: update OutputSpace to read from SS directly.
    D5. ``optimize_embedding`` plumbing at lines 674-680: change
        ``self.perceptualSpace.vocabulary`` reads to
        ``self.wholeSpace.lexicon``.  Optimizer-rebuild paths at
        2422 likewise.

E.  ``data/*.xml``
    E1. The XSD schema doesn't change for the migration.  No new XML
        knob is needed -- the lexicon location is structural, not
        configurable.
    E2. New invariant: ``IS.outputSize == CS.outputSize ==
        PS.inputSize``.  The XML knobs that control these:
          * IS.outputSize  ==  ``<InputSpace><nOutput>`` (slot count)
            * ``<nOutputDim>`` (per-slot dim).
            For "size" we interpret per the spec as the FLAT vector
            size: ``nOutput * nOutputDim``.
          * PS.inputSize   ==  ``<PartSpace><nInput>`` *
            ``<nInputDim>``.
          * CS.outputSize  ==  ``<ConceptualSpace><nOutput>`` *
            ``<nOutputDim>``.
        The invariant ENFORCES that the IS->PS handoff is
        size-preserving (the slab carries the same flat width) AND
        the C->P feedback (CS->PS) is size-preserving (the
        cross-recurrence handoff is dim-aligned).
    E3. Audit table for the configs we update (XML files in scope):
          MM_xor.xml:   IS.nOutput=8 * nOutputDim=10 = 80;
                        PS.nInput=8 * nInputDim=10 = 80;
                        CS.nOutput=8 * nOutputDim=10 = 80.
                        PASSES the invariant; no edit needed.
          MM_xor_loopback.xml: 8*4=32; 8*4=32; 8*4=32.  PASSES.
          MM_grammar.xml: missing nInputDim / nOutputDim everywhere;
                          defaults fill in.  Audit needed: verify the
                          defaults yield the invariant.
          MM_20M.xml: IS.nOutput=1024 * nDim=6 = 6144;
                     PS.nInput=1024 * nDim=6 = 6144;
                     CS.nOutput=8 * nDim=1024 = 8192.  FAILS (6144 vs
                     8192).  This config is the progressive-bottleneck
                     case; the invariant may be intentionally violated
                     here.  Open question to controller: should the
                     invariant be EQUAL or DIVIDES (CS.outputSize a
                     divisor of IS.outputSize, etc.)?  The spec's
                     "PS.inputSize" is the per-slot dim ``nInputDim``
                     in PS terminology (line 7385 -- ``percept_dim =
                     int(self.subspace.getEncodedInputSize())``); if
                     the spec means "per-slot dims align" rather than
                     "total flat slab equal", then the MM_20M case
                     reduces to:
                       IS.outputSize (per-slot) = 6 (IS.nOutputDim);
                       PS.inputSize  (per-slot) = 6 (PS.nInputDim);
                       CS.outputSize (per-slot) = 1024 (CS.nOutputDim).
                       Still mismatched -- but this is the
                       progressive-bottleneck on purpose.
                     RECOMMENDATION: enforce
                       ``IS.outputSize per-slot dim ==
                        PS.inputSize per-slot dim``
                     (the IS->PS handoff is the unambiguous one).  The
                     CS->PS feedback path is more nuanced and already
                     has its own validate_config invariants (line 6551:
                     ``effective_concept_dim == symbol_dim``).  Open
                     question for the controller.
    E4. New ``ModelFactory.validate_config`` rule (Models.py line
        6470): add a ``TheXMLConfig.require`` predicate that asserts
        the chosen interpretation of the invariant.  Loud-error on
        mismatch (consistent with the existing rules at lines
        6551-6577).

--------------------------------------------------------------------------
LOOKUP CHAIN IMPLEMENTATION  (the architectural target step-by-step)
--------------------------------------------------------------------------

A surface byte buffer arrives at IS: ``[B, N, W]`` long, null-terminated
UTF-8 per slot.

Step 1 (PS-side, lexer-adjacent):  PS resolves bytes -> orthographic row.
       ``ortho_idx, verified = self._mphf_gpu_layer.mphf_index(
           byte_slots, self._mphf_static_tables, return_verified=True)``
       where ``self._mphf_static_tables`` is built against
       ``self.codebook`` (which IS ``wholeSpace.lexicon``).  Result:
       ``ortho_idx: [B, N]`` long, indices into ``ws.lexicon.wv``.

Step 2 (SS-side, mereological):  PS hands ortho_idx to SS for
       resolution.  ``ortho_idx`` indexes orthographic-tagged rows;
       lookup ``meta_idx = ws.lexicon.part_parents[ortho_idx]``: the
       meta-symbol parent row.

Step 3 (intersect with "meanings" set):  for each ``meta_idx`` we
       want the SEMANTIC child of the meta-symbol.  The meta-symbol's
       children are rows whose part_parents == meta_idx; the semantic
       child is the one ALSO tagged with the ``meanings`` well-known
       atom (via a parallel ``meaning_mask: [V] bool`` buffer set at
       insert time, or via category_ids = the meanings well-known atom
       id).  Per-forward fast path: a precomputed
       ``orthographic_to_semantic_idx: [V] long`` buffer (a one-time
       index built when the meta-symbol is constructed at insert) lets
       Step 3 collapse to ONE gather:
       ``sem_idx = ws.orthographic_to_semantic_idx[ortho_idx]``.
       The intersection runs once (at insert) to BUILD this index;
       runtime is O(1) per slot.

Step 4 (semantic vector):  ``sem_vec = ws.lexicon.lookup(sem_idx)``
       -> ``[B, N, sym_D]``.

Step 5 (cast to conceptual space):  the existing
       ``WholeSpace.decode_to_concept`` (line 10445) does the pass-
       through cast (validate_config enforces ``sym_dim == concept_dim``
       so no learned remap is needed).  Result feeds downstream into
       ``ConceptualSpace.stm.push`` per the Stage 1.C contract.

Reverse direction (S -> C -> P -> I -> bytes):
       Reverse uses the existing ``ws.lexicon`` API
       (``reconstruct_data``, ``reconstruct_to_buffer``,
       ``get_recovered_word``).  The reverse is non-invertible
       (nearest-row table lookup); the surface half of the table is
       ``ws.lexicon.wv.index_to_key`` -- the ASCII-prefilled
       Embedding mapping that already stores the literal surface
       string per row.  See ``MPHFGpuLayer.reverse_map_rows`` line
       7651.

--------------------------------------------------------------------------
BACKWARD-INCOMPATIBLE CHANGES (what existing tests will break)
--------------------------------------------------------------------------

1. ``test/test_lexicon_ownership.py``: ~12 tests asserting the
   physical Embedding lives on PS.  Update to check
   ``m.wholeSpace.lexicon`` (the new owner); the
   ``m.perceptualSpace.vocabulary`` property still works (it now
   reads from SS through the codebook ref) so those individual
   reads can stay -- but the OWNERSHIP semantics (which class
   "has-a") change.  Specific test changes:
     - test_embedding_lives_on_perceptual_space  -> rename and assert
       on ``wholeSpace.lexicon``.
     - test_get_embedding_returns_perceptual     -> rename; assert
       ``m._get_embedding() is m.wholeSpace.lexicon``.

2. ``test/test_basicmodel.py`` lines 2050-2061: parameter-pointer
   tests on ``m.perceptualSpace.vocabulary.wv._vectors``.  These
   still work *value-wise* (the property still returns the same
   Embedding), but if the property is dropped the .vocabulary
   chain breaks.  RECOMMENDATION: KEEP the
   ``PartSpace.vocabulary`` property as a back-compat view of
   ``ws.lexicon``.  Tests need no change.

3. ``test/test_perceptual_loopback.py`` test_symbolic_perceptualSpace_ref_wired
   (line 268) -- the structural ref persists (PS still wraps the lexer
   pipeline); ASSERT REMAINS TRUE.

4. ``test/test_perceptualspace_bpe_forward.py``,
   ``test/test_perceptual_chunking.py``,
   ``test/test_serial_mode_perceptual.py``: these dispatch into
   ``PartSpace._embed*`` paths.  They will continue to work if
   the embed methods are updated to read from ``self.codebook`` instead
   of ``self.subspace.what`` (a one-line attribute swap inside each
   method).  No test-level changes expected.

5. Save/load (``test_lexicon_ownership.test_save_and_load_embeddings_round_trip``,
   ``test_ckpt_includes_embedding_vectors``,
   ``test_ckpt_includes_vocab_and_bpe_extras``): the .ckpt format
   doesn't change.  The Embedding's nn.Parameter
   ``wv._vectors`` continues to land in ``state_dict`` under whatever
   module-path name SS gives it.  THE STATE_DICT KEYS CHANGE.  An old
   .ckpt cannot reload into the new model (clean break per dispatch
   constraint 6).  Tests that save+load in-memory work without change;
   tests that load a checkpoint from disk built pre-migration would
   break -- none in scope.

6. ``test/test_pi_sigma_ownership.py``, ``test/test_ps_single_arg_refactor.py``,
   ``test/test_cs_stm_bookkeeping.py``: structural ownership of
   PI/SIGMA/STM -- UNAFFECTED.  All 27 tests in these three files
   should pass unchanged.

7. ``test/test_perceptual_loopback.py``: structural assertions about
   PS not having Embedding ownership for the LEXICON were a Phase-1.A
   pre-condition for this migration; we re-affirm here.

--------------------------------------------------------------------------
ESTIMATED IMPLEMENTATION EFFORT
--------------------------------------------------------------------------

Tests to write (Phase 2, this file):

  TestSSOwnsLexicon (5 tests):
    test_ws_has_lexicon_attribute
    test_ws_lexicon_is_embedding_instance
    test_ws_lexicon_replaces_ps_subspace_what
    test_ws_vocabulary_property_returns_lexicon
    test_ps_subspace_what_is_none_in_embedding_mode

  TestPSHasCodebookRef (4 tests):
    test_ps_has_codebook_attribute_post_wiring
    test_ps_codebook_is_ws_lexicon
    test_ps_mphf_codebook_resolves_to_ss
    test_ps_embed_paths_read_from_codebook_ref

  TestInputSpaceHasSSRef (3 tests):
    test_is_has_symbolicspace_ref
    test_is_lex_batch_uses_ws_lexicon
    test_is_no_peer_perceptual_lexicon_dependency

  TestMereologicalLookupChain (5 tests):
    test_well_known_atoms_includes_words_and_meanings
    test_orthographic_row_has_words_parent
    test_semantic_row_has_meanings_parent
    test_lookup_resolves_bytes_to_semantic_vector
    test_intersect_isolates_semantic_from_meta_symbol

  TestConfigInvariant (4 tests):
    test_is_output_eq_ps_input_dim
    test_cs_output_eq_ps_input_dim
    test_invariant_raises_on_mismatch
    test_invariant_passes_on_mm_xor_configs

  TestForwardReverseSmoke (2 tests):
    test_forward_pipeline_smoke_no_crash
    test_reverse_pipeline_smoke_no_crash

  Total: ~23 new tests.

XML files to update:
  MM_xor.xml             -- audit only (no edit expected -- already
                            satisfies the invariant).
  MM_xor_loopback.xml    -- audit only.
  MM_20M.xml              -- OPEN: needs controller decision on invariant
                            interpretation (per-slot vs flat slab) before
                            edit.
  MM_grammar.xml         -- audit; potentially add explicit
                            nInputDim / nOutputDim so the invariant
                            check is unambiguous.
  Other embedding-mode configs in scope (BasicModel.xml, LM_5M.xml,
  LM_5M_IR.xml, MM_400M.xml, MM_5M_AR.xml, MM_5M_IR.xml, MM_bpe.xml,
  MM_boolean.xml, MM_shamatha.xml, MM_xor_step{1..4}.xml,
  MentalModel.xml, RamsifiedModel.xml, HeadEmission.xml,
  POS_smoke.xml, stream_smoke.xml, tomatoes.xml, XOR_*.xml,
  HeadEmission.xml, model.xml): audit; only modify if the chosen
  invariant interpretation requires it.

Lines changed (rough estimate):
  bin/Spaces.py    -- ~400 lines (SS lexicon build + helpers, IS ref
                      rewire, PS attribute swap across 4 embed methods).
  bin/Models.py    -- ~80 lines (wiring + validate_config rule).
  data/*.xml       -- 0 lines if per-slot invariant; up to ~30 if
                      flat-slab invariant forces MM_20M-style configs to
                      be redesigned (open question, see E3).
  test/test_unified_lexicon_codebook.py -- ~500 lines (~23 tests + the
                      cheap-model harness following the
                      MM_xor_loopback.xml pattern from the
                      Stage-1.A/1.C tests).
  Existing tests (lexicon_ownership.py rename / reassert): ~30 lines.

--------------------------------------------------------------------------
RISKS / OPEN QUESTIONS FOR THE CONTROLLER
--------------------------------------------------------------------------

R1. NAMING.  ``Space.codebook`` is a bool property today; the new
    SS-side ATTRIBUTE that holds the Embedding cannot share that
    name on SS without breaking ``if self.codebook:`` consumers
    INSIDE WholeSpace itself (lines 10153, 10689, 10755).
    Recommendation: rename SS attribute to ``self.lexicon``; on PS the
    new reference ``self.codebook`` shadows the bool property via
    instance-level set (a known pattern in this codebase; see
    ``object.__setattr__`` at Models.py line 4085 already used for
    this same kind of breakage avoidance).  PLEASE CONFIRM.

R2. INVARIANT INTERPRETATION.  ``IS.outputSize ==
    CS.outputSize == PS.inputSize`` -- per-slot dim equality
    (``nOutputDim`` / ``nInputDim``), or flat slab equality
    (``nOutput * nOutputDim``)?  The per-slot interpretation lets
    MM_20M's progressive-bottleneck stand; the flat-slab forces
    MM_20M to be redesigned.  Recommendation: PER-SLOT (the IS->PS
    handoff and C->P feedback already operate per-slot in
    ``PartSpace`` line 7385; the slab equality is enforced
    SEPARATELY by ``_register_requirements`` line 7523 for muxed
    width divisibility).  PLEASE CONFIRM.

R3. INSERT-TIME META-SYMBOL CONSTRUCTION.  When PS encounters an
    OOV surface word (today: ``codebook.insert(word)`` at line 7719),
    the new pipeline must:
      (a) insert the orthographic row (existing call path).
      (b) insert a NEW semantic row (with its own
          ``Embedding.insert`` call).
      (c) ensure a meta-symbol row exists (or create one) and tag
          BOTH orthographic and semantic rows with that
          ``part_parents`` link.
      (d) update ``orthographic_to_semantic_idx`` mapping.
    This is a NEW insert flow.  Open question: should the semantic
    row start as a copy of the orthographic embedding (the legacy
    "lexicon entry IS the word vector" semantics), or all-zeros
    (no semantic content until learned)?  Recommendation: copy
    initially -- preserves existing training behavior and the
    semantic row drifts via gradient over time.  PLEASE CONFIRM.

R4. ConfigA (MM_20M) WILL CHANGE STATE_DICT KEYS.  The dispatch
    explicitly grants the clean break ("existing trained checkpoints
    are not preserved").  Confirming: MM_20M.ckpt becomes unusable
    after this migration.  PLEASE CONFIRM you're willing to retrain
    -- if not, a one-time conversion script becomes in scope.

R5. WORD-LEARNING (Embedding.insert) IS A WRITE PATH.  The Stage-1.B
    spec line (".bn/Spaces.py" "WholeSpace owns the unified word
    lexicon codebook and hosts grammar ops needing codebook write
    access") suggests SS-side insert plumbing.  Today's insert path
    runs through ``PartSpace._embed`` line 7717.  Migration
    moves the insert call to an SS-side method
    (``ws.insert_oov(word)`` or similar) called from PS.  No
    behavioral change in this stage; just relocates the call site.

R6. ``_peer_perceptual`` retirement: today ``InputSpace._lex_batch``
    needs the TOKENIZER (lives on the Embedding's _token_stream method).
    Post-migration the Embedding is on SS, but the tokenizer is a
    METHOD on the Embedding instance -- same code, different owner.
    No semantic change.  Confirming the rewire of one structural
    reference (IS._peer_perceptual -> IS._wholeSpace_ref) is in scope.

R7. SYM_LEARN / KNOWLEDGE ARTIFACT: ``WholeSpace.attach_knowledge``
    (line 10234) and ``PartSpace.attach_knowledge`` (line 7583)
    both write metadata onto the lexicon.  Today they write at
    different levels (SS writes reference codebook; PS writes
    word_table.ref_ids onto wv).  Post-migration both write through
    ``ws.lexicon``.  The change is local and additive (no breakage
    expected); test_knowledge_artifact regression should be re-run.

--------------------------------------------------------------------------
END PROPOSAL  --  AWAITING CONTROLLER GO-AHEAD FOR PHASE 2
--------------------------------------------------------------------------
"""

# ==========================================================================
# PHASE 2 TDD GATE  --  Stages 1.D + 1.B (combined, narrower)
# ==========================================================================
#
# Locked design (controller revised the recon proposal):
#
# - PS keeps its existing Lexicon (Embedding), MPHF, and BPE machinery.
#   No migration of these.
# - PS's per-word representation is already CS-space-dimensioned
#   (flat-slab invariant validated below).
# - ``WholeSpace.codebook`` (= ``WholeSpace.subspace.what`` Codebook)
#   gains PAIRED ROWS per word at insert time:
#     * an orthographic row = a copy of PS's per-word vector
#     * a semantic row      = a fresh random CS-space vector (trainable)
#   The two are DIRECTLY parented: orthographic row -> semantic row,
#   via ``Codebook.set_part_parent``. No meta-symbol.
# - Flat-slab invariant: IS.nOutput * IS.nDim == CS.nOutput * CS.nDim ==
#   PS.nOutput * PS.nDim. Loud error on mismatch.
#
# This module exercises:
#   * SS.codebook paired insertion API (orth + semantic + parenthood).
#   * The orth row matches PS's per-word vector (copy contract).
#   * The semantic row is random (different from orth at init).
#   * Config validator rejects flat-slab mismatches.
#   * Existing MM_xor / MM_xor_loopback configs satisfy the invariant.

import os
import sys
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_DATA_DIR = os.path.join(_PROJECT, "data")
_CONFIG = os.path.join(_DATA_DIR, "MM_xor_loopback.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")

import Models  # noqa: E402
import Language  # noqa: E402
from Spaces import Codebook, Embedding  # noqa: E402
from util import init_config  # noqa: E402


def _make_plain_model():
    """Cheap-boot MM_xor_loopback model; mirrors the Stage-1.A / 1.C
    test fixtures.
    """
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    model.eval()
    return model


class TestSSCodebookPairedInsertRetired(unittest.TestCase):
    """Step 3 (2026-06-10 symbolic-iteration plan): the Stage-1.B
    paired-row contract (orth copy + random semantic partner per lexicon
    word on SS.codebook, orth parented to semantic) is RETIRED. The
    CS-leg symbol codebook captures the code-as-written vs
    code-for-the-concept correspondence in place (codebook row = concept
    code; row id = the written symbol); the lexicon stays PS-local.
    """

    def test_paired_insert_api_is_gone(self):
        model = _make_plain_model()
        ws = model.wholeSpace
        self.assertFalse(
            hasattr(ws, "insert_paired_word"),
            "WholeSpace.insert_paired_word must be RETIRED (Step 3 of "
            "the 2026-06-10 symbolic-iteration plan).")
        self.assertFalse(
            hasattr(ws, "mark_word_atom"),
            "the mark_word_atom autobind fallback retires with the "
            "paired-row machinery.")

    def test_lexicon_insert_leaves_ws_codebook_untouched(self):
        model = _make_plain_model()
        ws = model.wholeSpace
        emb = model.perceptualSpace.vocabulary
        cb = ws.subspace.what
        self.assertIsInstance(cb, Codebook)
        W_before = cb.getW().detach().clone()
        vec = torch.zeros(int(emb.wv._vectors.shape[1]))
        vec[0] = 0.5
        emb.insert("novelword", vector=vec)
        self.assertTrue(
            torch.equal(W_before, cb.getW().detach()),
            "a PS-side lexicon insert must leave the SS codebook "
            "prototype bit-identical (no orth/semantic reach-across).")


class TestFlatSlabInvariant(unittest.TestCase):
    """Config validator must enforce IS.nOutput*IS.nDim ==
    CS.nOutput*CS.nDim == PS.nOutput*PS.nDim. Loud error on mismatch.
    """

    def _build_config_dict(self, is_out, is_dim, ps_out, ps_dim,
                           cs_out, cs_dim):
        """Build a minimal cfg dict the validator can chew on.

        ``is_dim`` / ``ps_dim`` / ``cs_dim`` are the per-tier CONTENT
        widths. Under "6+2+2" the (2,2) muxed tiers (IS/PS/CS) carry an
        EVENT ``nDim`` = content + .where/.when band, while the (0,0)
        content tiers (SS/OS) carry the bare content width. Both the
        flat-slab invariant and the SS==CS pass-through compare CONTENT,
        so adding the band uniformly to the muxed tiers preserves the
        slab-match / mismatch semantics these tests exercise while keeping
        the call sites expressed in content widths.
        """
        from architecture import canonical_shape
        is_event = is_dim + sum(canonical_shape("InputSpace"))
        ps_event = ps_dim + sum(canonical_shape("PartSpace"))
        cs_event = cs_dim + sum(canonical_shape("ConceptualSpace"))
        return {
            "architecture": {
                "modelType": "embedding",
                "monotonic": False,
                "naive": False,
            },
            "InputSpace": {
                "nOutput": is_out, "nDim": is_event,
                "nInputDim": 0, "flatten": False, "hasAttention": False,
            },
            "PartSpace": {
                "nInput": is_out, "nInputDim": is_event,
                "nOutput": ps_out, "nDim": ps_event,
                "invertible": True, "hasAttention": False,
                "codebook": "quantize",
            },
            "ConceptualSpace": {
                "nInput": ps_out, "nInputDim": ps_event,
                "nOutput": cs_out, "nDim": cs_event,
                "invertible": True, "hasAttention": False,
                "codebook": "quantize",
            },
            "WholeSpace": {
                "nInput": cs_out, "nInputDim": cs_event,
                "nOutput": cs_out, "nDim": cs_dim,
                "codebook": "quantize",
            },
            "OutputSpace": {
                "nInput": cs_out, "nInputDim": cs_dim,
                "nOutput": 1,
            },
        }

    def test_validator_passes_when_flat_slab_matches(self):
        cfg = self._build_config_dict(
            is_out=8, is_dim=4, ps_out=8, ps_dim=4, cs_out=8, cs_dim=4)
        # 8*4 == 8*4 == 8*4: passes.
        Models.ModelFactory.validate_config(cfg)  # no raise expected

    def test_validator_raises_on_is_ps_mismatch(self):
        # IS slab=8*4=32, PS slab=8*5=40: mismatch.
        cfg = self._build_config_dict(
            is_out=8, is_dim=4, ps_out=8, ps_dim=5, cs_out=8, cs_dim=4)
        with self.assertRaises((ValueError, AssertionError, KeyError)) as ctx:
            Models.ModelFactory.validate_config(cfg)
        msg = str(ctx.exception)
        self.assertIn("flat-slab", msg.lower(),
                      f"Mismatch error must mention flat-slab; got: {msg}")

    def test_validator_raises_on_cs_mismatch(self):
        # IS slab=8*4=32, PS slab=8*4=32, CS slab=8*5=40: mismatch.
        cfg = self._build_config_dict(
            is_out=8, is_dim=4, ps_out=8, ps_dim=4, cs_out=8, cs_dim=5)
        with self.assertRaises((ValueError, AssertionError, KeyError)) as ctx:
            Models.ModelFactory.validate_config(cfg)
        msg = str(ctx.exception)
        self.assertIn("flat-slab", msg.lower())


class TestExistingConfigsSatisfyFlatSlab(unittest.TestCase):
    """The configs in scope (MM_xor, MM_xor_loopback, MM_grammar, MM_20M)
    must satisfy the flat-slab invariant after the Stage 1.D
    re-architecture. They build through ``BasicModel.from_config``
    without validate_config raising.
    """

    def _load_and_validate(self, config_path):
        """Validate one XML by parsing into the cfg dict and running
        validate_config. Returns the cfg on success, else lets the
        ValueError propagate.
        """
        init_config(path=config_path, defaults_path=_DEFAULTS)
        Language.TheGrammar._configured = False
        from util import TheXMLConfig
        cfg = TheXMLConfig.data
        Models.ModelFactory.validate_config(cfg)
        return cfg

    def test_mm_xor_satisfies_flat_slab(self):
        self._load_and_validate(os.path.join(_DATA_DIR, "MM_xor.xml"))

    def test_mm_xor_loopback_satisfies_flat_slab(self):
        self._load_and_validate(os.path.join(_DATA_DIR, "MM_xor_loopback.xml"))

    def test_mm_5m_satisfies_flat_slab(self):
        self._load_and_validate(os.path.join(_DATA_DIR, "MM_20M.xml"))

    def test_mm_grammar_satisfies_flat_slab(self):
        self._load_and_validate(os.path.join(_DATA_DIR, "MM_grammar.xml"))


# ==========================================================================
# Stage 8: structural META-taxonomy replacement of the orth-row-copy contract
# (doc/plans/2026-05-27-perceptstore-meta-taxonomy-reentrancy.md §Stage 8).
#
# The Stage 1.B "orth row in SS.codebook = COPY of PS-side vector" contract
# is retired under the radix-mode pipeline. The replacement is structural:
#   * a percept is inserted into PS.percept_store (PS-side bytes + vector);
#   * an SS row is allocated for the meaning;
#   * a META node binds (ps_idx, ws_idx) and records cross-codebook signed
#     references in SS.taxonomy / SS.taxonomy_parent_map.
# These tests assert the structural shape, in addition to the legacy
# orth-row-copy assertions which remain valid for the legacy lexicon
# chunking path (MM_xor_loopback.xml uses the lexicon path; MM_xor.xml
# uses radix).
# ==========================================================================


class TestStage8MetaTaxonomyStructural(unittest.TestCase):
    """Stage 8 structural assertions over (insert_percept, insert_symbol,
    insert_meta) on the radix-mode model.

    The legacy ``insert_paired_word`` tests above continue to assert the
    orth-row-copy contract under the lexicon path; this class is the
    Stage-8 replacement: in radix mode the percept lives on
    ``PS.percept_store`` and the SS-side row for the meaning is freshly
    allocated, bound via a META node that records cross-codebook signed
    references rather than copying the PS vector into SS.
    """

    @classmethod
    def setUpClass(cls):
        import warnings
        import Models as _Models
        import Language as _Language
        from util import init_config as _init_config
        # 2026-05-29: use MM_xor_fixture.xml (dedicated test fixture
        # with chunking=radix AND SS codebook=quantize) so runtime
        # experiments on MM_xor.xml don't break the META taxonomy
        # tests, which inherently require an SS Codebook for
        # ``insert_symbol`` to allocate rows.
        cfg = os.path.join(_DATA_DIR, "MM_xor_fixture.xml")
        _init_config(path=cfg, defaults_path=_DEFAULTS)
        _Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cls.model, _ = _Models.BasicModel.from_config(cfg)
        _Models.TheData.load("xor")

    def test_radix_mode_supplies_percept_store(self):
        ps_space = self.model.perceptualSpace
        self.assertIsNotNone(
            ps_space.percept_store,
            "MM_xor radix-mode model must expose percept_store on PS")

    def test_insert_meta_records_positive_int_cross_codebook_references(self):
        """The META node's children list contains a PS-tagged position
        and an SS-tagged position. PS-side bytes are recoverable via
        ``PerceptStore.bytes_for(ps_row)`` where ``ps_row`` resolves
        through ``WholeSpace._ps_pos_to_row``.
        """
        ws = self.model.wholeSpace
        ps_store = self.model.perceptualSpace.percept_store
        # Insert PS-side bytes + SS-side meaning row; bind via META.
        ps_pos = ws.insert_percept(b"structural_meta")
        ws_pos = ws.insert_symbol()
        meta_pos = ws.insert_meta(ps_pos, ws_pos)
        # All three are positive positions; kinds are tagged accordingly.
        self.assertGreater(ps_pos, 0,
                           "insert_percept must return a positive position")
        self.assertGreater(ws_pos, 0,
                           "insert_symbol must return a positive position")
        self.assertGreater(meta_pos, 0,
                           "insert_meta must return a positive position")
        self.assertEqual(ws._pos_kind.get(ps_pos), "ps")
        self.assertEqual(ws._pos_kind.get(ws_pos), "ws")
        self.assertEqual(ws._pos_kind.get(meta_pos), "meta")
        # Children list contains both PS and SS positions.
        children = ws.taxonomy_children(meta_pos)
        ps_children = [c for c in children if ws._pos_kind.get(c) == "ps"]
        ws_children = [c for c in children if ws._pos_kind.get(c) == "ws"]
        self.assertTrue(
            ps_children and ws_children,
            f"META children must contain both PS- and SS-tagged "
            f"positions; got {children!r}")
        # PS-side bytes recoverable via the position -> row lookup.
        ps_row = ws._ps_pos_to_row[ps_children[0]]
        self.assertEqual(ps_store.bytes_for(ps_row), b"structural_meta")

    def test_ws_row_not_copied_from_ps_vector(self):
        """The retired contract: the META node's SS row is NOT a copy of
        the PS vector. (Under the new structural binding, SS gets a fresh
        symbolic row + a separately allocated META row; neither is a
        copy of the percept_store row.)
        """
        ws = self.model.wholeSpace
        ps_store = self.model.perceptualSpace.percept_store
        ps_pos = ws.insert_percept(b"distinct_storage")
        ps_row = ws._ps_pos_to_row[ps_pos]
        # Pin the PS row to a known vector so we can prove the SS-side
        # rows are not duplicates of it.
        D = int(ws.nDim)
        marker = torch.zeros(D)
        marker[0] = 1.0  # distinctive shape
        with torch.no_grad():
            ps_store.codebook.data[ps_row].copy_(
                marker.to(ps_store.codebook.device, ps_store.codebook.dtype))
        # Allocate a *random* SS row and bind via META using a custom
        # fused vec also distinct from the pinned PS row.
        sym_init = torch.zeros(D)
        sym_init[1] = 1.0
        ws_pos = ws.insert_symbol(init_vec=sym_init)
        fused_init = torch.zeros(D)
        fused_init[2] = 1.0
        meta_pos = ws.insert_meta(ps_pos, ws_pos, fused_vec=fused_init)
        # SS row for ws_pos must not equal the PS row.
        ws_row = ws._ws_pos_to_row[ws_pos]
        W = ws.subspace.what.getW()
        ps_vec = ps_store.codebook[ps_row].detach()
        ws_vec = W[ws_row].detach()
        self.assertFalse(
            torch.allclose(ws_vec.to(ps_vec.device, ps_vec.dtype),
                           ps_vec, atol=1e-5),
            "SS row must NOT be a copy of the PS vector under the Stage 8 "
            "structural-binding contract")
        # Same for the META row.
        meta_row = ws._ws_pos_to_row[meta_pos]
        meta_vec = W[meta_row].detach()
        self.assertFalse(
            torch.allclose(meta_vec.to(ps_vec.device, ps_vec.dtype),
                           ps_vec, atol=1e-5),
            "META row must NOT be a copy of the PS vector under the "
            "Stage 8 structural-binding contract")


if __name__ == "__main__":
    unittest.main()
