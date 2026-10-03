def _run_batch_once(self, train=True, batchNum=0, batchSize=10, split="train",
             optimizer=None, batch_override=None, progress=None,
             exploration_trial=False,
             trial_mode="reconstruct", questions=None,
             source_rows=None, attach_outputs=False):
    """Run a single batch: forward pass, loss, and (if training) backward + step.

    Sentence trials are completed by the eager closing driver during forward.
    ``exploration_trial`` is the non-recording mode-schedule context read;
    it does not run a second batch derivation.

    Args:
        train: whether to compute gradients and update parameters.
        batchNum: opaque cursor for the next batch position.
        batchSize: number of examples per batch.
        split: "train", "test", or "validation".
        optimizer: pre-built optimizer (required when train=True).
        batch_override: optional ``(inputTensor, outputTensor)`` pair;
            the primary dispatch path used by the DataLoader streaming
            path in ``runEpoch``.
        progress: optional fraction in ``[0.0, 1.0]`` indicating how
            far through the current split's data the cursor has
            advanced. When set, the per-batch timing line includes
            a percentage so long runs report visible progress.
            ``runEpoch`` populates this from
            ``SentenceStreamDataset.progress()``; callers that drive
            ``runBatch`` directly leave it ``None``.
        questions: optional one-per-row ``WhatQuestion`` batch. When
            omitted, cursor coordinates select present reconstruction,
            supervised output, runtime inference, or future prediction.
        source_rows: stable zero-based presentation rows emitted by the
            data cursor (nested for packed sentence batches).
        attach_outputs: on runtime inference, retain each generated
            response on the presentation's output side with model
            provenance. Supplied labels remain unchanged.


    Returns:
        (BatchResult, nextBatchNum) on success, or (None, batchNum) when
        the dataset is exhausted.
    """
    self._install_unit_span_fn()
    record_trial = not exploration_trial
    self._packed_prediction_drained = False
    self._compose_complete = None
    # Completion belongs to this derivation, including serial/parallel
    # interleaving whose context batch can have a different row count.
    self._stm_post_depth = None
    language = getattr(getattr(self, 'symbolSpace', None), 'languageLayer', None)
    if language is not None:
        language._last_derivation = None
    self.train(bool(train))
    _ensure_grad_anchors(TheDevice.get())
    _disc = getattr(getattr(self, "symbolSpace", None), "discourse", None)
    if _disc is not None:
        _disc.detach_prediction_context()
    _stm = getattr(getattr(self, "conceptualSpace", None), "stm", None)
    if _stm is not None and hasattr(_stm, "detach_live"):
        _stm.detach_live()                  # no graph crosses a brick
    _trace = getattr(self, "_reconstruction_stack", None)
    _trace = _trace() if callable(_trace) else None
    if _trace is not None and hasattr(_trace, "detach_live"):
        _trace.detach_live()
    _release_loop_checkpoints(tuple(        # the previous brick's own loops
        getattr(self, name, None) for name in (
            "_stm_single_S", "_recon_cost", "_recon_idea_cost",
            "_tensor_sentence_roots_live", "_tensor_pushed_ideas",
            "_packed_sentence_roots", "_output_policy_cost")))
    object.__setattr__(self, "_output_policy_cost", None)
    object.__setattr__(self, "_recon_cost", None)
    object.__setattr__(self, "_recon_completed", False)
    object.__setattr__(self, "_reconstruction_product", None)
    object.__setattr__(self, "_recon_ideas", None)
    object.__setattr__(self, "_tensor_pushed_ideas", None)
    object.__setattr__(self, "_tensor_sentence_roots_depth", None)
    object.__setattr__(self, "_tensor_final_end_slots", None)
    object.__setattr__(self, "_tensor_final_end_depth", None)
    object.__setattr__(self, "_tensor_sentence_roots_live", None)
    # A caller may supply an optimizer built outside ``getOptimizer``.
    # Make the live owner visible to the PartSpace admission boundary.
    if optimizer is not None and getattr(
            self, "perceptualSpace", None) is not None:
        object.__setattr__(
            self.perceptualSpace, "_radix_optimizer", optimizer)
    # Enabling a previously absent expectation head changes the graph.
    if getattr(self, "_compiled_step_needs_rebuild", False):
        self.enable_compiled_step()
        self._compiled_step_needs_rebuild = False

    # First-call hook: try to enable §7 torch.compile reduce-overhead
    # mode on CUDA targets. Idempotent; safe on non-CUDA hosts.
    self._maybe_compile_brick()
    sentenceIdx = batchNum  # sentence index before batchNum increments
    if batch_override is None:
        raise RuntimeError(
            "runBatch: no batch_override supplied. Callers must pass "
            "batch_override=(inputTensor, outputTensor) — training "
            "path via the DataLoader in runEpoch, inference path via "
            "InputSpace.prepInput (or, for sentence-level generation, "
            "BasicModel.generate_sentence)."
        )
    batch = batch_override
    inputTensor, outputTensor = batch
    if train and trial_mode != 'predict':
        self._sentence_supplied_answers = outputTensor
    self._stage_expectation_documents(
        split, source_rows, int(inputTensor.shape[0])
        if isinstance(inputTensor, torch.Tensor) else len(inputTensor))
    inference_only = not train and split == "runtime"
    what_questions = self._questions_for_batch(
        split=split,
        batch_size=(int(inputTensor.shape[0])
                    if isinstance(inputTensor, torch.Tensor)
                    else int(len(inputTensor))),
        sentence_index=sentenceIdx,
        source_rows=source_rows,
        questions=questions,
        trial_mode=trial_mode,
    )
    self._sentence_answer_questions = what_questions
    _objective_probe.opened(self, locals())

    # Advance the serialized model clock once per PROCESSED batch (this
    # point is past the no-batch early raise, so it ticks exactly once on
    # BOTH the train and inference paths regardless of which return fires
    # below). Done here, in eager Python before the forward, so the live
    # WhenRangeEncoding(s) the forward stamps already carry the advanced
    # absolute time (``.t == present()``). See ``_advance_when_time``.
    # The mode schedule's internal context read does not consume an
    # external batch. Compose's sentence trials run inside this one tick.
    if not exploration_trial:
        self._advance_when_time()

    # Pre-allocate per-batch state OUTSIDE the compiled forward.
    # ``SymbolSpace.ensure_microbatch`` allocates completion state /
    # ``_last_svo`` / ``_svo_valid`` / ``_recent_count`` etc. on
    # first call (and on shape changes); when those allocations
    # happen INSIDE a torch.compile region, CUDAGraph capture
    # takes ownership of the underlying memory, so the next
    # replay's allocation overwrites the previous step's state
    # and the next attribute read raises ``RuntimeError: Error:
    # accessing tensor output of CUDAGraphs that has been
    # overwritten by a subsequent run.`` Hoisting the call up
    # here keeps the resulting tensors Python-owned.
    if self.symbolSpace is not None and not inference_only:
        ss = self.symbolSpace
        try:
            if isinstance(inputTensor, torch.Tensor):
                B_pre = int(inputTensor.shape[0])
            else:
                B_pre = int(len(inputTensor))
        except Exception:
            B_pre = None
        if B_pre is not None:
            ss.ensure_microbatch(B_pre, 1)

    if train:
        optimizer.zero_grad()
    if record_trial:
        with torch.set_grad_enabled(bool(train)):
            self._stage_expectation_queries(training=bool(train))

    # Teacher owns the lesson, clean complete-input target, and error-term
    # registration window. Objective corpus addresses staged by runEpoch
    # are resolved here; the model's subjective .where/.when state is not
    # an input to Teacher and remains untouched.
    self._open_batch(
        split=split,
        batch_size=(
            int(inputTensor.shape[0])
            if isinstance(inputTensor, torch.Tensor)
            else int(len(inputTensor))
        ),
        training=train,
        clean_input=inputTensor,
    )

    # Per-batch Space.Start cascade (moved out of forward() so sliding
    # buffers can persist across forward() calls within a stream).
    self._start_spaces_for_forward()

    # AMP: torch.autocast wrapper from util.amp_context() honors
    # MODEL_AMP env var (hydrated from XML <architecture><amp> in
    # ModelFactory.run). bf16 returns scaler=None; fp16+CUDA returns
    # the process-wide GradScaler used in the backward path below.
    amp_cm, amp_scaler = amp_context()
    self._sentence_amp_scaler = amp_scaler
    with torch.set_grad_enabled(bool(train)), amp_cm:
        # Forward pass returns a 4-tuple.  IR-only contract:
        # ``predictions`` is ``[B, N, predDim]`` (one head emission
        # per P-slot) and ``forwardInput`` is the inputSpace event
        # ``[B, N, D]``.  ``reconstruction`` is always None after
        # the reverse pipeline retirement (2026-05-14); the slot
        # is kept in the tuple for downstream code that pattern-
        # matches on the legacy shape.
        #
        # ``cudagraph_mark_step_begin`` (only meaningful under modes
        # that capture CUDAGraphs -- "reduce-overhead", "max-
        # autotune") tells the runtime to release the previous
        # step's CUDAGraph outputs so the memory pool can be reused
        # for this step.  No-op under "default" mode; idempotent on
        # non-CUDA hosts.
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except (AttributeError, RuntimeError):
            pass
        # O1: route the per-batch compute through the compiled
        # callable when enabled (else eager). runEpoch/runBatch stay
        # eager Python; only this forward+loss+backward unit is
        # torch.compiled. See doc/plans/2026-05-16-compiled-step-
        # boundary-design.md.
        # Keep eager staging immediately beside the compiled invocation:
        # the traced forward only reads these parked tensors.  The eager
        # path intentionally remains unchanged and lexes inline.
        if self._compiled_step is not None:
            if isinstance(inputTensor, torch.Tensor):
                inputTensor = inputTensor.to(TheDevice.get())
            self._staged_in_sub = self._lex_embed_stem(inputTensor)
            # Sentence traces are staged after the cold-start reset and
            # before every compiled callable.  This keeps dense detached
            # surface copies and all Python lifecycle mutation outside the
            # graph while preserving the trace through its reverse/loss
            # consumer until the outer sentence-boundary reset.
            _ss_trace = getattr(self, "symbolSpace", None)
            if (getattr(self, "serial", False)
                    and _ss_trace is not None
                    and not getattr(
                        _ss_trace, "_per_sentence_initialized", False)):
                _ss_trace.soft_reset()
                _ss_trace._per_sentence_initialized = True
            self._stage_reconstruction_teacher()
            _trace_slab = getattr(
                self.inputSpace, "_ar_embedded_N", None)
            if (getattr(self, "serial", False)
                    and torch.is_tensor(_trace_slab)):
                self._prepare_reconstruction_choices(
                    int(_trace_slab.shape[0]),
                    int(_trace_slab.shape[1]),
                    _trace_slab.device)
            _requires_sparse_bank = self._aligned_serial_sparse_bank_mode()
            if _requires_sparse_bank:
                _bank_rows = getattr(
                    self.inputSpace, "_ar_concept_lookup_rows", None)
                _bank_atoms = getattr(
                    self.inputSpace, "_ar_concept_lookup_atoms", None)
                if (not torch.is_tensor(_bank_rows) or _bank_rows.dim() != 2
                        or not torch.is_tensor(_bank_atoms)
                        or _bank_atoms.dim() != 3
                        or tuple(_bank_rows.shape)
                        != tuple(_bank_atoms.shape[:2])):
                    raise RuntimeError(
                        "compiled aligned serial forward requires the eager "
                        "sparse concept row bank; lexical staging did not "
                        "produce a shape-aligned rows/atoms pair")
            # A full W-loop owns one fixed, identity-masked residual-part
            # layout.  Do not give MPS a symbolic P range or compile a
            # separate full loop for every radix spelling.  The older
            # word-cell path retains its broad dynamic contract below.
            _fullgraph_word_loop = bool(getattr(
                self, "_compiled_word_loop_fullgraph", False))
            if _fullgraph_word_loop:
                # Cold-start sentence state must be materialized before
                # Dynamo observes the word-loop module.  Creating this
                # attribute inside the captured prelude changes an
                # ``hasattr`` guard after the first forward and forces a
                # duplicate W specialization.
                _ss = getattr(self, "symbolSpace", None)
                if (_ss is not None
                        and not getattr(_ss, "_per_sentence_initialized", False)):
                    _ss.soft_reset()
                    _ss._per_sentence_initialized = True
                if _ss is not None:
                    # Establish every scalar/carrier attribute the
                    # captured recurrence writes before its first guard
                    # census.
                    object.__setattr__(
                        _ss, "_target_cursor_length", int(getattr(
                            self.inputSpace, "_active_word_bucket", 0) or 0))
                _cs_owner = getattr(self, "conceptualSpace", None)
                if _cs_owner is not None:
                    object.__setattr__(
                        _cs_owner, "_peer_prev_sub", getattr(
                            _cs_owner, "_peer_empty_sub", None))
                    object.__setattr__(
                        _cs_owner, "_peer_prev_sym", getattr(
                            _cs_owner, "_peer_empty_sym", None))
                # ``_per_word_prelude`` creates a fresh STM working slab
                # inside the graph.  Establish the same MPS-backed state
                # before the first guard census; otherwise the
                # constructor's CPU placeholder is observed on call one
                # and the MPS replacement forces call two to compile
                # again.
                _staged_event = self._staged_in_sub.materialize()
                _stm = getattr(
                    getattr(self, "conceptualSpace", None), "stm", None)
                if torch.is_tensor(_staged_event) and _stm is not None:
                    _stm.begin_forward(
                        int(_staged_event.shape[0]),
                        device=_staged_event.device,
                        dtype=(_staged_event.dtype
                               if _staged_event.is_floating_point() else None))
                self._stage_fixed_residual_part_capacity()
            elif self.serial:
                # Only the serial word carrier has a variable part axis.
                # Parallel perception uses its configured fixed slot count.
                # W is deliberately static (one of 16/32/64/128); the
                # number of residual radix constituents inside one
                # complete word is not.  Mark only that P axis dynamic so
                # promotion (e.g. five pieces -> one stored percept) does
                # not compile another W graph. P>=3 excludes the
                # contradictory Inductor guard ``P - 1 != 1`` while
                # masked identity padding preserves words with one or two
                # real constituents.
                _part_min = 3
                _part_max = max(_part_min, int(getattr(
                    self.inputSpace, "outputShape", (8192,))[0]))
                for _part_tensor in (
                        getattr(self.inputSpace, "_ar_word_part_ids", None),
                        getattr(self.inputSpace, "_ar_word_part_mask", None),
                        getattr(self.inputSpace, "_ar_word_part_offsets", None),
                        getattr(self.inputSpace, "_ar_target_word_bytes", None),
                        getattr(self.inputSpace, "_ar_target_word_mask", None)):
                    if torch.is_tensor(_part_tensor) and _part_tensor.dim() == 3:
                        torch._dynamo.mark_dynamic(
                            _part_tensor, 2, min=_part_min, max=_part_max)
                _word_width = max(1, int(getattr(
                    self.inputSpace, "_active_word_bucket", 1) or 1))
                _flat_min = _part_min * _word_width
                _flat_max = _part_max * _word_width
                _ps_forward = getattr(
                    self.perceptualSpace, "_forward_input", None)
                if isinstance(_ps_forward, dict):
                    for _name in ("indices", "seed_event", "word_groups",
                                  "part_spans", "percept_where"):
                        _value = _ps_forward.get(_name)
                        if torch.is_tensor(_value) and _value.dim() >= 2:
                            torch._dynamo.mark_dynamic(
                                _value, 1, min=_flat_min, max=_flat_max)
                    for _name in ("word_part_indices", "word_part_mask"):
                        _value = _ps_forward.get(_name)
                        if torch.is_tensor(_value) and _value.dim() == 3:
                            torch._dynamo.mark_dynamic(
                                _value, 2, min=_part_min, max=_part_max)
            for _cs in (getattr(self, "conceptualSpaces", None) or ()):
                _validate_routing = getattr(_cs, "validate_intra_routing", None)
                if callable(_validate_routing):
                    _validate_routing()
            _active_bucket = int(getattr(
                self.inputSpace, "_active_word_bucket", 0) or 0)
            if _fullgraph_word_loop:
                self._active_compiled_step = (
                    self._ensure_lazy_fullgraph_word_loop(_active_bucket))
            else:
                self._active_compiled_step = self._compiled_word_steps.get(
                    _active_bucket, self._compiled_step)
            disc = (self.symbolSpace.discourse
                    if self.symbolSpace is not None else None)
            self._stage_legacy_discourse_prediction(
                disc, fullgraph_word_loop=_fullgraph_word_loop)
        _fwd = (self.forward if exploration_trial or getattr(self, '_sentence_ends', False) else
                (self._active_compiled_step if torch.is_grad_enabled()
                 and self._active_compiled_step is not None else self.forward))
        self._exploration_trial = bool(exploration_trial)
        try:
            # ``what()`` is a thin delegation over this exact executor.
            # It installs only target-free question/LTM chooser context;
            # the returned execution is the ordinary forward tuple.
            _what_answers = self._what_or_think(
                what_questions, inputTensor, executor=_fwd,
                record=record_trial)
            _forward_result = _what_answers[0].execution
            (forwardInput, symbols, predictions,
             _) = self._publish_compiled_sentence_state(_forward_result)
            self._last_what_actual = tuple(_what_answers)
            # Answer-path modules built during this what() join the live
            # optimizer now (they post-date getOptimizer).
            _fresh_synth = getattr(self, "_fresh_synthesis_params", None)
            if (train and optimizer is not None and _fresh_synth
                    and hasattr(optimizer, "add_param_group")):
                _present = {id(p) for g in optimizer.param_groups
                            for p in g.get("params", ())}
                _new = [p for p in _fresh_synth if id(p) not in _present]
                if _new:
                    optimizer.add_param_group({"params": _new})
                _fresh_synth.clear()
            # This value is already a public graph output.  Publish its
            # detached discourse view eagerly as well, avoiding the same
            # attribute-only escape that PyTorch 2.14 eliminates for S.
            self._current_discourse_s = (
                symbols.detach() if torch.is_tensor(symbols) else None)
        finally:
            self._exploration_trial = False
        self._end_step()
        # Unconditional SEEN priming: the rows this forward fired bump
        # their surfaces (no-grad bookkeeping; consumers read next batch).
        self._prime_seen_step()
        outputDataPred = predictions
        _construction = getattr(self, "_last_answer_construction", None)
        if (getattr(self, "answer_synthesis", False)
                and _construction is not None
                and torch.is_tensor(_construction.actual)):
            outputDataPred = _construction.actual

        # Resolve desired answers only AFTER the model response is fixed.
        # The resulting values are loss-side metadata and never enter the
        # forward call, grammar context, STM, LTM, or runtime cache.
        _what_data = getattr(self.inputSpace, "data", None)
        if _what_data is not None and hasattr(_what_data, "what"):
            self._last_what_desired = tuple(
                _what_data.what(question) for question in what_questions)
        else:
            self._last_what_desired = tuple()
        if (inference_only and attach_outputs
                and _what_data is not None
                and hasattr(_what_data, "attach_output")):
            for question, answer in zip(
                    what_questions, self._last_what_actual):
                _what_data.attach_output(question, answer)

        # ε-growing codebook hook (Phase 4 follow-up): when any
        # codebook-bearing space carries ``codebookGrowthEpsilon > 0``
        # in its XSD, invoke ``VectorQuantize.grow_on_novelty`` on
        # the encoder output to insert novel inputs into empty
        # slots before the EMA path locks in assignments.  No-op
        # when ``growth_epsilon`` is 0 (default) or every slot is
        # already populated.
        if train:
            for _sp_attr in ("perceptualSpace", "conceptualSpace",
                             "wholeSpace"):
                _sp = getattr(self, _sp_attr, None)
                _cb = getattr(getattr(_sp, "subspace", None),
                              "what", None)
                _vq = getattr(_cb, "vq", None)
                if _vq is None:
                    continue
                _eps = float(getattr(_vq, "growth_epsilon", 0.0)
                             or 0.0)
                if _eps <= 0.0:
                    continue
                _ev = (_sp.subspace.materialize()
                       if _sp.subspace is not None else None)
                if _ev is None:
                    continue
                try:
                    _vq.grow_on_novelty(_ev, _eps)
                except Exception:
                    # Growth is best-effort -- never let an
                    # insertion glitch stop the training step.
                    pass

        if inference_only:
            # Inference path: forward only, no loss.
            result = self.BatchResult(
                outputPred=outputDataPred, symbols=symbols,
                lossOut=None, lossIn=None,
                inputPred=None, forwardInput=forwardInput,
            )
            self.End()
            return result, batchNum

        if outputTensor is None:
            raise RuntimeError(
                f"runBatch: missing output targets for split='{split}'. "
                "For inference use split='runtime'."
            )

        # 2026-05-28: restore supervised output-head loss for
        # tasks that need it (e.g. XOR_exact with binary labels).
        # The prior IR-only regime hardcoded output_weight=0 and
        # routed all gradient through the masked-LM ``lossIn``;
        # that path is preserved (when ``outputTensor`` is absent
        # or the labels are zero-width, the supervised branch
        # degenerates to the original "side channel" semantics).
        # When ``outputTensor`` IS supplied, compare the head
        # prediction against the labels and apply a non-zero
        # weight so the head receives gradient. Mirrors the
        # ``reconstruction_reverse`` pattern (try-guarded so a
        # shape edge case degrades to zero contribution rather
        # than crashing the step).
        lossOut = torch.zeros((), device=TheDevice.get())
        output_errors, input_errors, reverse_errors = Error(), Error(), Error()
        output_weight = 0.0
        output_policy_loss = None
        selected_answer_credit = None
        # ``Data.what()`` is the answer-loss authority (What spec Step 5):
        # the question's desired answer replaces the loader's incidental
        # ``outputTensor``; unavailable rows are masked out of the term.
        _answer_target, _answer_mask = self._what_answer_target(
            what_questions, outputTensor,
            supervised_only=bool(train and getattr(self, "answer_synthesis", False)))
        self._last_answer_target = _answer_target
        self._last_answer_mask = _answer_mask
        _answer_authority = _answer_target is not None
        _scored_target = _answer_target if _answer_authority else (
            None if _answer_mask is not None else outputTensor)
        _surface_target = getattr(self, "_answer_surface_target", None)
        _surface = getattr(getattr(self, "_last_answer_construction", None),
                           "surface", None)
        try:
            if (_surface_target is not None and torch.is_tensor(_surface)
                    and _answer_mask is not None and bool(_answer_mask.any())):
                # Text answer: constructed surface vs embedded
                # target sentence, band-aware, masked to answerable rows.
                lossOut = self._reverse_event_loss(
                    _surface[_answer_mask], _surface_target[_answer_mask],
                    registry=output_errors, name='answer', objective='output')
                output_weight = 1.0
                if train and float(getattr(self, "selected_thought_policy_weight", 0.0)) > 0:
                    selected_answer_credit = self._selected_thought_answer_errors(
                        _surface, _surface_target, _answer_mask, what_questions, surface=True)
                if (train and trial_mode != "predict"
                        and float(getattr(self, "output_policy_weight", 0.0)) > 0.0):
                    output_policy_loss = self._output_action_credit(
                        _surface, _surface_target, _answer_mask, what_questions,
                        surface=True)
            elif (getattr(self.inputSpace.data,
                        "has_supervised_outputs", True)
                    and _scored_target is not None
                    and torch.is_tensor(_scored_target)
                    and _scored_target.numel() > 0
                    and outputDataPred is not None
                    and torch.is_tensor(outputDataPred)
                    and outputDataPred.numel() > 0):
                # shape reconciliation lives in _align_output_pred; irreconcilable warns once and zeroes the term.
                _pred = self._align_output_pred(outputDataPred,
                                                _scored_target)
                if _pred is not None:
                    if train and float(getattr(self, "selected_thought_policy_weight", 0.0)) > 0:
                        selected_answer_credit = self._selected_thought_answer_errors(
                            _pred, _scored_target, _answer_mask, what_questions)
                    if (train and trial_mode != "predict" and _answer_authority
                            and float(getattr(self, "output_policy_weight", 0.0)) > 0.0):
                        output_policy_loss = self._output_action_credit(
                            _pred, _scored_target, _answer_mask, what_questions)
                    if (_answer_authority
                            and not bool(_answer_mask.all())):
                        _pred = _pred[_answer_mask]
                        _scored_target = _scored_target[_answer_mask]
                    lossOut = self.loss.compute(_pred, _scored_target)
                    self.loss.register(output_errors, 'answer', _pred, _scored_target,
                        category='prediction', objective='output')
                    output_weight = 1.0
        except Exception as _out_exc:
            # Best-effort degrade to zero, but never SILENTLY (5b fail-loud).
            lossOut = torch.zeros((), device=TheDevice.get())
            output_weight = 0.0
            output_policy_loss = None
            selected_answer_credit = None
            self._warn_zeroed_channel(
                "output_loss_exception",
                f"supervised output loss zeroed by "
                f"{type(_out_exc).__name__}: {_out_exc}")
        self.record_loss(
            "output", lossOut,
            weight=output_weight,
            space="OutputSpace", category="prediction", trained=False,
        )
        self.errors.merge(output_errors)

        # IR masked-LM loss: compare the post-body perceptual event
        # at masked positions against the pre-mask embedding the
        # forward snapshotted in ``_ir_pre_mask_input``.  No reverse
        # pipeline involved -- the body / head ran masked, the
        # masked positions carry the prediction target, the head
        # plays no role in the loss.  This is the BERT-style
        # masked-LM contract.
        inputDataPred = None
        inputPred = None
        mask_pos = getattr(self, "_ir_mask_positions", None)
        pre_mask = getattr(self, "_ir_pre_mask_input", None)
        pred_full = None
        if hasattr(self.perceptualSpace.subspace, 'materialize'):
            pred_full = self.perceptualSpace.subspace.materialize()
        # None checks stay Python (no sync). The masked reconstruction
        # loss is computed densely via `compute_masked` instead of a
        # boolean-mask gather `pred[mask]` -- the gather is
        # data-dependent (its row count is read to host: an implicit
        # cudaMemcpyDtoH that breaks CUDA-graph capture).
        # `compute_masked` is the sync-free, value-equivalent form
        # (masked-sum / masked-count) and returns 0.0 on an empty
        # mask with no NaN (replacing the old nan_to_num gate). See
        # doc/BrickHostSyncStatus.md residual D.
        # Serial grammar readings consume the owned byte reconstruction.
        # Paths without a grammar keep perceptual reconstruction.
        if self.reconstruct_in_loop:
            # Completion belongs to understanding, before reasoning and
            # output can change staging. Training and evaluation consume
            # precisely the same owned byte objective, once.
            owned = self._last_understanding.input_reconstruction
            if owned is None or not torch.is_tensor(owned.byte_cost):
                raise RuntimeError("tied reconstruction completed without its byte objective")
            lossIn = owned.byte_cost.mean()
            input_errors.error('reconstruction.bytes', owned.byte_cost, math.log(256),
                mask=self.inputSpace._reconstruction_sentence_available.any(-1),
                category='reconstruction')
            if not train:
                inputDataPred = owned.event.detach()
        elif (mask_pos is not None and pre_mask is not None
                and pred_full is not None):
            # Band-aware seam: percept-layout where/when scales (silent-band wiring fix, 2026-07-04).
            lossIn = self._masked_event_loss(pred_full, pre_mask,
                                             mask_pos, registry=input_errors)
        else:
            lossIn = torch.zeros((), device=TheDevice.get())
            # 5b fail-loud: a dead reconstruction channel must announce itself (warn-once, names the gate).
            _missing = ", ".join(
                n for n, v in (("mask", mask_pos), ("target", pre_mask),
                               ("pred", pred_full)) if v is None)
            _cb = getattr(getattr(self.perceptualSpace, 'subspace',
                                  None), 'what', None)
            self._warn_zeroed_channel(
                "reconstruction_zeroed",
                f"perceptual reconstruction inputs missing "
                f"({_missing}); percept .what={type(_cb).__name__}, "
                f"pred_full shape="
                f"{tuple(pred_full.shape) if torch.is_tensor(pred_full) else None}")
        self.record_loss(
            "reconstruction", lossIn,
            weight=1.0,
            space="InputSpace", category="reconstruction", trained=False,
        )
        self.errors.merge(input_errors, weight=self.loss.reconstruction_scale)
        missing = getattr(self.inputSpace, '_reconstruction_missing_sentence_count', None)
        self.record_loss('reconstruction_unavailable_sentences', missing, trained=False, category='count')

        # A grammar-free reading retains its perceptual event inverse.
        # The grammar path has already supplied its single byte objective.
        lossRev = torch.zeros((), device=TheDevice.get())
        _rev_dedupe = self.reconstruct_in_loop
        try:
            if forwardInput is not None and not _rev_dedupe:
                # reverseReconstruct owns the input inverse and its seed;
                # runBatch consumes its result and diagnostic score.
                # One understanding for both downward paths (Step 6):
                # reuse the one ``what()`` captured for this execution.
                understanding = getattr(self, "_last_understanding", None)
                if (understanding is None
                        or understanding.execution is not (
                            tuple(_forward_result)
                            if isinstance(_forward_result, list)
                            else _forward_result)
                        and understanding.symbolic_state is not symbols):
                    understanding = self._capture_understanding(
                        _forward_result)
                self._last_understanding = understanding
                rev_ev, lossRev_rec = self.reverseReconstruct(
                    understanding, target=forwardInput, train=train, registry=reverse_errors)
                if (not train and rev_ev is not None
                        and torch.is_tensor(rev_ev)):
                    inputDataPred = rev_ev.detach()
                if lossRev_rec is not None:
                    lossRev = lossRev_rec
        except Exception:
            # Reverse round-trip is approximate through averaged
            # loops; never let a reconstruction edge case stop the
            # training step.
            lossRev = torch.zeros((), device=TheDevice.get())
        if not _rev_dedupe:
            self.record_loss(
                "reconstruction_reverse", lossRev,
                weight=float(getattr(self.loss, 'reconstruction_scale',
                                     0.0) or 0.0),
                space="InputSpace", category="reconstruction", trained=False,
            )
            self.errors.merge(reverse_errors, weight=self.loss.reconstruction_scale)

        # Raw diagnostic costs, recorded before backward. The Error
        # registry owns the trained total; tied byte reconstruction has
        # no second reverse-event training term.
        self._last_primary_costs = {
            "input_reconstruction": lossIn,
            "input_reconstruction_reverse": lossRev,
            "answer_construction": lossOut,
        }
        self._record_what_batch(
            what_questions, lossOut,
            lossIn if (torch.is_tensor(lossIn) and float(lossIn.detach()) != 0.0)
            else lossRev,
            sentences=len(what_questions))

        # JOINT mode: compute SBOW embedding loss
        sbow = None
        if train:
            embedding_errors = Error()
            sbow = self.trainEmbeddings(('JOINT'), sentenceIdx, split, errors=embedding_errors)
            # Perceptual SBOW: when lexer=byte, train percept vectors
            # via leave-one-out centroid prediction
            if getattr(self, 'lexer', None) in ('byte', 'bytes'):
                psbow = self.perceptual_sbow_loss(errors=embedding_errors)
                if psbow is not None:
                    sbow = psbow if sbow is None else sbow + psbow
            if sbow is not None:
                self.record_loss(
                    "embedding_sbow", sbow,
                    weight=self.loss.embedding_scale,
                    space="SymbolSpace", category="embedding", trained=False,
                )
                self.errors.merge(embedding_errors, prefix='embedding.', weight=self.loss.embedding_scale)

        # Inter-sentence ARMA(p, q) loss term -- model-owned
        # (``_discourse_arma_loss``): predicts ``s_hat_t`` from the
        # lagged reps/residuals, returns the per-batch MSE, and
        # commits the new rep + residual into the rings (vectorized,
        # sync-free). Cold-start rows return ``None``. Computed here,
        # post-body / pre-backward, so the term trains the predictor.
        # Each paired trial starts from the same prior runtime. Its
        # private observation and prediction context survive only if that
        # trial wins; both trials are scored under the same objective.
        legacy_prediction = bool(getattr(self, "legacy_prediction_enabled", False))
        arma_loss = (self._discourse_arma_loss()
                     if (train and legacy_prediction
                         and record_trial) else None)

        # Intra-sentence prediction loss term (Task 3, STM serial/
        # parallel modes) -- ``ConceptualSpace.forward`` ran the
        # in-STM predictor predict-then-perceive over the per-word
        # steps and accumulated ``L_intra = MSE(prediction, perceived)``
        # live (grad-bearing) on the conceptual space. Consume the
        # per-batch mean here, post-body / pre-backward (mirroring the
        # ARMA term), so the term trains the intra-sentence predictor.
        # ``consume_intra_loss`` resets the accumulator; it returns
        # ``None`` when nothing was accumulated (eval, weight off, or
        # an all-degenerate sentence).
        intra_loss = (self.conceptualSpace.consume_intra_loss()
                      if train and legacy_prediction else None)

        # Inter-sentence end-state prediction loss term (Task 8, plan
        # §9) -- the sentence-boundary hook ran the inter-level
        # predictor + scored it against the arriving end-state,
        # accumulating ``L_inter`` live on the discourse layer. Consume
        # the per-sentence mean here, post-body / pre-backward (mirroring
        # the ARMA + intra terms). ``None`` when the discourse layer is
        # absent (absolute-only no-op), eval-time, weight off, or no
        # scored sentence this batch.
        inter_loss = (
            self._discourse_inter_loss()
            if train and record_trial else None
        )
        expectation_policy_loss = self._expectation_policy_loss() if train and record_trial else None
        # InfoNCE next-idea contrastive term (the discourse layer's second
        # accumulator; populated during the forward boundary observe).
        inter_contrastive = None
        if train and record_trial and self.symbolSpace is not None:
            _disc = getattr(self.symbolSpace, "discourse", None)
            if _disc is not None and hasattr(
                    _disc, "consume_inter_contrastive_loss"):
                inter_contrastive = _disc.consume_inter_contrastive_loss()

        # Trial-split: on a PURE next-idea PREDICTION trial, zero the
        # reconstruction + auxiliary terms so the next-idea signal (inter
        # MSE + InfoNCE contrastive) is the SOLE gradient. prediction_trial_
        # ratio == 0 -> trial_mode is always "reconstruct" -> this never
        # fires -> byte-identical. Done post-forward so the forward stays
        # mode-independent (no Dynamo recompiles).
        if train and trial_mode == "predict":
            _z = torch.zeros((), device=TheDevice.get())
            lossOut, lossIn = _z, _z
            sbow = None
            lossRev = None
            arma_loss = None
            intra_loss = None

        # The registry owns the trained sum; raw gate metrics stay separate.
        # Keep the actual weighted objectives distinct for diagnostics.
        # A missing answer is not substituted with reconstruction credit.
        gradient_objectives = None
        _rr = float(self.loss.reconstruction_scale)
        if train:
            gradient_objectives = {
                "reconstruction": _rr * lossIn,
                "output": (1.0 - _rr) * lossOut,
                "expectation": lossIn.new_zeros(()),
            }
        lesson_reports = getattr(self, '_reading_lesson_reports', ())
        if train and trial_mode != 'predict' and lesson_reports:
            grammar_lesson = torch.stack(lesson_reports).mean()
            # Already trained inside the sentence; this detached term
            # keeps the batch's reported total complete without replay.
            self.record_loss('grammar_lesson', grammar_lesson,
                weight=self.grammar_lesson_weight, space='LanguageSpace', category='grammar', trained=False)
        if train and trial_mode != "predict" and output_policy_loss is not None:
            if gradient_objectives is not None:
                gradient_objectives["output"] = (gradient_objectives["output"]
                    + self.output_policy_weight * output_policy_loss)
            self.record_loss(
                "output_policy", output_policy_loss,
                weight=self.output_policy_weight,
                space="LanguageSpace", category="policy")
        _cscale = float(getattr(
            self.loss, "conceptual_similarity_scale", 0.0) or 0.0)
        if train and not self.serial and _cscale > 0.0:
            concept_errors = Error()
            csbow = self.conceptual_sbow_loss(errors=concept_errors)
            if csbow is not None:
                self.record_loss(
                    "conceptual_sbow", csbow, weight=_cscale,
                    space="ConceptualSpace", category="embedding", trained=False,
                )
                self.errors.merge(concept_errors, prefix='conceptual_sbow.', weight=_cscale)
        # Rank-ordered soft-L0 that keeps concept definitions compact
        # (snap contract sec 5). Default lambda 0.0 -- byte-identical off.
        _dss = float(getattr(
            self.loss, "definition_sparsity_scale", 0.0) or 0.0)
        if train and _dss > 0.0 and self.conceptualSpace is not None:
            defsp = self.conceptualSpace.definition_sparsity_loss(lam=1.)
            if defsp is not None:
                self.record_loss(
                    "definition_sparsity", defsp, weight=_dss,
                    space="ConceptualSpace", category="reg")
        if lossRev is not None:
            if gradient_objectives is not None:
                gradient_objectives["reconstruction"] = (
                    gradient_objectives["reconstruction"] + _rr * lossRev)
        if arma_loss is not None:
            if gradient_objectives is not None:
                gradient_objectives["expectation"] += self.arma_scale * arma_loss
            self.errors.merge(self.symbolSpace.discourse._arma_errors,
                              prefix='expectation.', weight=self.arma_scale)
        if intra_loss is not None:
            self.errors.merge(self.conceptualSpace._consumed_intra_errors,
                              prefix='expectation.', weight=self.conceptualSpace.intra_loss_weight)
        if inter_loss is not None and self.inter_loss_weight > 0.0:
            if gradient_objectives is not None:
                gradient_objectives["expectation"] += self.inter_loss_weight * inter_loss
            self.errors.merge(_disc._consumed_inter_errors, prefix='expectation.',
                              weight=self.inter_loss_weight)
        if inter_contrastive is not None and self.inter_contrastive_weight > 0.0:
            if gradient_objectives is not None:
                gradient_objectives["expectation"] += self.inter_contrastive_weight * inter_contrastive
            self.errors.merge(_disc._consumed_contrastive_errors, prefix='expectation.',
                              weight=self.inter_contrastive_weight)
        if hasattr(self, 'outputSpace'):
            pipeline_errors = self.outputSpace.subspace.errors
            self.errors.merge(pipeline_errors)
            pipeline_errors.clear()

        if expectation_policy_loss is not None and self.expectation_policy_weight > 0:
            self.record_loss("expectation_policy", expectation_policy_loss,
                weight=self.expectation_policy_weight, space="SymbolSpace", category="policy")
            if gradient_objectives is not None:
                gradient_objectives["expectation"] = (gradient_objectives["expectation"]
                    + self.expectation_policy_weight * expectation_policy_loss)
        # One hard-choice credit contract for completed grammatical
        # episodes, using the shared meter and the later answer error.
        if train and selected_answer_credit is not None and float(
                getattr(self, "selected_thought_policy_weight", 0.0) or 0.0) > 0.0:
            selected_pol_loss = self._selected_thought_policy_loss(
                selected_answer_credit[0], mask=selected_answer_credit[1])
            if selected_pol_loss is not None:
                self.record_loss(
                    "selected_thought_policy", selected_pol_loss,
                    weight=self.selected_thought_policy_weight,
                    space="SymbolSpace", category="policy")


        # Method-1 -> Method-2 leaf distillation (snap design doc step
        # 3): the exact-leaf teacher supervises a from-root decoder so
        # the collapsed root stays SEPARABLE per sentence. Default
        # weight 0.0 -> skipped -> byte-identical.
        if train and float(
                getattr(self, "leaf_distill_weight", 0.0) or 0.0) > 0.0 \
                and not self.reconstruct_in_loop:
            try:
                leaf_errors = Error()
                ld_loss = self._leaf_distill_loss(registry=leaf_errors)
            except Exception:
                ld_loss = None     # distillation must not abort training
            if ld_loss is not None:
                if gradient_objectives is not None:
                    gradient_objectives["reconstruction"] = (
                        gradient_objectives["reconstruction"]
                        + self.leaf_distill_weight * ld_loss)
                self.errors.merge(leaf_errors, weight=self.leaf_distill_weight)
                # The head is built lazily on first use — hand its
                # params to the LIVE optimizer once (getOptimizer ran
                # before the head existed).
                if (optimizer is not None
                        and getattr(self, "_leaf_distill_head_fresh",
                                    False)):
                    self._leaf_distill_head_fresh = False
                    optimizer.add_param_group({"params": list(
                        self._leaf_distill_head_module.parameters())})

        # Truth-modulated loss: delegated to SymbolSpace since the
        # TruthLayer lives there.  SymbolSpace handles the empty-store
        # guard internally; we only gate on ``train``.  The falsity
        # penalty operand is the last cached symbol activation --
        # stored truths are also recorded from symbol space, so both
        # sides of the disjunction live in the basis's native space.
        if train and self.symbolSpace is not None:
            symbol_acts = None
            if hasattr(self, 'symbol_states') and self.symbol_states:
                symbol_acts = self.symbol_states[-1]
            truth_base = self.errors.total()
            self.symbolSpace.truth_modulated_loss(
                lossOut*0 if truth_base is None else truth_base,
                symbolic_space=self.conceptualSpace,
                symbol_acts=symbol_acts,
                universality_score=getattr(self, '_universality_score', None),
                luminosity_weight=getattr(self, 'luminosity_weight', 0.1),
                universality_weight=getattr(self, 'universality_weight', 0.1),
                truth_loss_weight=getattr(self, 'truth_loss_weight', 0.0),
                allow_excluded_middle=getattr(self, 'allow_excluded_middle', 1),
                allow_contradiction=getattr(self, 'allow_contradiction', 0),
                model=self,
                gradient_objectives=gradient_objectives,
            )
            # Gate-L1 sparsity penalty on LiftLayer / LowerLayer
            # raw_gate parameters. Pulls unused singular-component
            # multipliers toward zero so each rule converges to a
            # low-rank slice of its host operator. Default lambda
            # is 0.0 -- no penalty unless a config opts in.
            gate_l1 = self.symbolSpace.gate_l1_loss(
                lam=1. if getattr(self, 'gate_l1_lambda', 0.0) > 0 else 0.)
            if gate_l1 is not None:
                self.record_loss(
                    "gate_l1", gate_l1,
                    weight=getattr(self, 'gate_l1_lambda', 0.0),
                    space="SymbolSpace", category="reg")

            # Stage 3 cleanup: the chart's sparse-MoE load-balance
            # bookkeeping retired with the chart itself. The
            # <loadBalanceWeight> knob stands by for any future
            # signal-router rule load-balancing; no consumer yet.

        if train:
            batch_objective = self.errors.total()
            readout_l1 = self._configure_concept_readout_l1(optimizer,
                stage=batch_objective is not None and batch_objective.requires_grad)
            if readout_l1 is not None:
                # Reporting only: the single optimizer step handles the
                # non-smooth L1 term in Adam's diagonal metric. Adding its
                # subgradient here would penalize every connection twice.
                self.record_loss(
                    "concept_readout_l1", readout_l1 / self._concept_readout_l1_strength,
                    weight=self._concept_readout_l1_strength,
                    space="ConceptualSpace", category="reg", trained=False)

        if train and trial_mode == 'predict':
            for term in self.errors._terms.values():
                if term['category'] in ('reconstruction', 'prediction', 'embedding', 'grammar'):
                    term['trained'] = False
        totalLoss = self.errors.total()
        if totalLoss is None:
            totalLoss = lossOut + lossIn * 0
        gradient_objectives = {name: self.errors.total(objective=name)
            for name in ('reconstruction', 'output', 'expectation')} if train else None

        _objective_probe.batch(self, locals())
        # Snapshot the breakdown before the backward pass so later
        # calls to TheError.covariance() can see it in the history
        # even if the step is aborted by a non-finite detector below.
        # snapshot()'s per-term .item() is a cudaMemcpyDtoH (breaks
        # the brick CUDA-graph-capture contract, test_brick_no_sync)
        # and only feeds the diagnostic covariance() API (no
        # training-loop consumer). Gate behind MODEL_DEBUG, like the
        # finite-loss guard just below.
        if _util.MODEL_DEBUG:
            self.errors.snapshot()

    # Per-batch finite-loss guard is a GPU sync (.all() materializes).
    # Gate it behind MODEL_DEBUG so production training pays no per-batch
    # sync; failures still surface via NaN gradients downstream.
    if _util.MODEL_DEBUG and not torch.isfinite(totalLoss).all():
        def _loss_value(name, value):
            if value is None:
                return f"{name}=None"
            if isinstance(value, torch.Tensor):
                finite = torch.isfinite(value).all().item()
                if value.numel() == 1:
                    return f"{name}={value.detach().item()} finite={finite}"
                return f"{name}=shape{tuple(value.shape)} finite={finite}"
            return f"{name}={value}"

        details = ", ".join([
            _loss_value("lossOut", lossOut),
            _loss_value("lossIn", lossIn),
            _loss_value("sbow", sbow),
            _loss_value("arma", arma_loss),
            _loss_value("symbol", symbol_loss),
            _loss_value("total", totalLoss),
        ])
        raise FloatingPointError(
            f"Non-finite total loss in {self.name}.runBatch("
            f"split={split}, batch={batchNum}): {details}"
        )

    # Plain f-string without totalLoss -- interpolating a tensor's
    # __format__ forces a device sync every batch.  The epoch summary
    # still carries per-epoch losses; per-batch loss is available via
    # MODEL_DEBUG if needed.
    #
    # Per-batch wall-clock: time.perf_counter() reads the host
    # monotonic clock (no GPU sync). The first batch (before
    # ``self._last_batch_time`` is set) prints without a delta --
    # that "compile + warm-up" tick is heavily front-loaded and
    # measuring it as "delta from epoch start" would be misleading.
    # Subsequent batches print ``(Δ=X.XXXs)`` where X is the
    # wall-clock elapsed since the last batch's report.
    import time as _time
    now = _time.perf_counter()
    last = getattr(self, '_last_batch_time', None)
    # Optional percent-complete suffix when the cursor populated
    # ``progress`` (set from ``SentenceStreamDataset.progress()``
    # in ``runEpoch``). Direct ``runBatch`` callers (tests,
    # inference) pass progress=None and get the bare timing line.
    pct = ("" if progress is None
           else f", {min(progress, 1.0) * 100.0:.2f}%")
    if last is None:
        TheMessage(f"batch = {batchNum} (warm-up{pct})")
    else:
        TheMessage(f"batch = {batchNum} (Δ={now - last:.3f}s{pct})")
    self._last_batch_time = now

    # Inductor / Dynamo recompile detector. Per-shape recompiles can
    # show up as wall-clock variance on otherwise identical batches;
    # this prints a one-line delta whenever Dynamo records new
    # frame events (compiles, recompiles, graph breaks). Defensive
    # try/except: torch._dynamo counters API has changed across
    # versions, and on MODEL_COMPILE=none there's nothing to count.
    try:
        from torch._dynamo.utils import counters as _dyn_counters
        frames = _dyn_counters.get('frames', {})
        rc_now = sum(int(v) for v in frames.values())
        rc_last = getattr(self, '_last_recompile_count', None)
        if rc_last is None:
            self._last_recompile_count = rc_now
        elif rc_now > rc_last:
            delta = rc_now - rc_last
            # Show the breakdown (e.g. ok=N, recompile=M) so a
            # stable steady-state with occasional recompiles is
            # visible at a glance.
            detail = ", ".join(f"{k}={v}" for k, v in sorted(frames.items()))
            TheMessage(f"  [compile] dynamo +{delta} frame events (total {rc_now}; {detail})")
            self._last_recompile_count = rc_now
    except Exception:
        pass

    if train:
        _every = int(getattr(self, "branch_diagnostics_every", 0) or 0)
        if (_every > 0 and int(getattr(
                self, "_training_step_count", 0) or 0) % _every == 0):
            self._sample_gradient_diagnostics(gradient_objectives, optimizer)
        if not totalLoss.requires_grad:
            # An input-only batch already trained at its sentence endings.
            # With no batch-end teacher term there is no third backward.
            _fineweb_step_performed = bool(self._sentence_trial_costs)
        elif amp_scaler is not None:
            # fp16 on CUDA: scale grads to avoid underflow, then unscale
            # inside scaler.step() before the actual optimizer update.
            self._backward_training_loss(
                totalLoss, amp_scaler, optimizer=optimizer)
            self._assert_finite_train_state("after backward")
            if self.ergodic:
                self.paramUpdate()
            # CUDA fp16 GradScaler owns its fused unscale/non-finite
            # check. In auto mode the optimizer wrapper's guard is also
            # disabled, preserving the brick's zero-D2H contract.
            from LearningEvaluation import scaler_step_performed
            _fineweb_step_performed = scaler_step_performed(amp_scaler, optimizer)
        else:
            self._backward_training_loss(
                totalLoss, optimizer=optimizer)
            self._assert_finite_train_state("after backward")
            preflight_finite_gradients(
                optimizer, self.named_parameters(),
                cache_for_step=hasattr(
                    optimizer, "_step_without_finite_preflight"))
            if self.ergodic:
                self.paramUpdate()
            optimizer.step()
            _fineweb_step_performed = True
        self._project_sentence_parameters()
        self._assert_finite_train_state("after optimizer.step")
        # The episode credit boundary (spec 8.2): durable LTM detaches
        # here, after the one optimizer step of the episode.
        self._end_what_episodes()
        self._flush_partspace_promotions(optimizer=optimizer)
        self._clamp_symbolic_codebook()
        # Contextual learning consumes only committed sentence evidence.
        if record_trial:
            self._update_contextual_concept_codebooks()
        self._normalize_conceptual_codebooks()
        # 2026-05-28: enforce the |W| <= 1 invariant on the
        # Embedding (Lexicon) by re-projecting rows onto the unit
        # ball after each optimizer step. Matches the SBOW
        # pre-training pattern at bin/embed.py:1976. Without this,
        # JOINT training drifts Embedding rows beyond [-1, 1]
        # (measured: |W|.max ~ 1.54 after 200 epochs on XOR_exact),
        # which breaks the nearest-Embedding reverse decode -- the
        # bounded recon vector from pi.reverse cannot reach the
        # unbounded target rows.
        self._normalize_perceptual_embedding()
        self._advance_codebook_parameter_versions()
        # Sentence updates never advance this public batch counter.
        if record_trial:
            self._training_step_count = (
                int(getattr(self, "_training_step_count", 0) or 0) + 1
            )
            from LearningEvaluation import record_fineweb_training
            if _fineweb_step_performed:
                record_fineweb_training(self, split=split, source_rows=source_rows)
    else:
        # The eager lexical stem may discover promotions during no-grad
        # inference too. The completed forward is its graph-safe boundary;
        # install now so the next request can use the promoted row.
        self._flush_partspace_promotions(optimizer=None)

    result = self.BatchResult(
        outputPred=outputDataPred,
        symbols=symbols,
        lossOut=lossOut,
        lossIn=lossIn,
        inputPred=inputDataPred,
        forwardInput=forwardInput,
    )
    # Spec names for the two primary costs (``primary_costs()``); the
    # legacy lossOut/lossIn fields stay for existing callers.
    self._last_primary_costs = {
        **(getattr(self, "_last_primary_costs", None) or {}),
        "input_reconstruction": lossIn,
        "answer_construction": lossOut,
    }
    # Pure compute brick: no Reset, no truth-layer compact, no host
    # sync inside runBatch. The outer doc-streaming loop in runEpoch
    # (or any per-tick driver) is responsible for:
    #   * Hard reset (per-row, on document boundary) via
    #     ``BasicModel.dispatch_per_row_reset(hard_eos_list)``.
    #   * Soft reset (per-row, on grammar sentence completion) via
    #     ``symbolSpace.drain_sentence_completed()`` →
    #     ``symbolSpace.soft_reset(b)``.
    #   * ``truth_layer.compact()`` (one host sync per tick, kept
    #     outside the brick).
    # See doc/plans/2026-04-26-rolling-cursor-doc-streaming-handoff.md.

    # Memory-leak diagnostics (perf-notes/08-*). Three independently
    # gated probes; each is a no-op without its env var.
    if os.environ.get("BASIC_PROFILE_DIAG"):
        try:
            ss_diag = self.symbolSpace
            tl_diag = getattr(ss_diag, 'truth_layer', None) if ss_diag is not None else None
            if tl_diag is not None and hasattr(tl_diag, 'count'):
                tl_count_diag = int(tl_diag.count.item())
                tl_pending_diag = int(getattr(tl_diag, '_pending_count', 0))
            else:
                tl_count_diag = 0
                tl_pending_diag = 0
            if torch.cuda.is_available():
                cuda_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                cuda_max_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                cuda_alloc_mb = 0.0
                cuda_max_mb = 0.0
            opt_state_numel = 0
            if optimizer is not None:
                for st in optimizer.state.values():
                    for v in st.values():
                        if torch.is_tensor(v):
                            opt_state_numel += v.numel()
            word_lens_diag = {}
            for sp in self.spaces:
                sub = getattr(sp, 'subspace', None)
                if sub is not None and hasattr(sub, 'word'):
                    word_lens_diag[sp.__class__.__name__] = len(sub.word)
            pair_counts_diag = {}
            for sp in self.spaces:
                cl = getattr(sp, 'chunkLayer', None) or getattr(sp, 'chunk_layer', None)
                if cl is not None:
                    pair_counts_diag[sp.__class__.__name__] = (
                        len(getattr(cl, '_pair_counts', {})),
                        len(getattr(cl, '_unigram_counts', {})),
                    )
            TheMessage(
                f"[diag] batch={batchNum} tl={tl_count_diag}/{tl_pending_diag} "
                f"cuda_alloc_mb={cuda_alloc_mb:.1f} cuda_max_mb={cuda_max_mb:.1f} "
                f"opt_state_numel={opt_state_numel} word_lens={word_lens_diag} "
                f"chunk_pair_uni={pair_counts_diag}"
            )
        except Exception as _diag_exc:
            TheMessage(f"[diag] batch={batchNum} diag_error={_diag_exc!r}")

    if os.environ.get("BASIC_PROFILE_LEAK") and torch.cuda.is_available():
        if batchNum >= 96 and not getattr(self, "_leak_recording", False):
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python",
                max_entries=400_000)
            self._leak_recording = True
            TheMessage(f"[leak] start recording at batch {batchNum}")
        elif batchNum >= 160 and getattr(self, "_leak_recording", False):
            _leak_dir = os.path.expanduser("~/WikiOracle/basicmodel/perf-notes")
            os.makedirs(_leak_dir, exist_ok=True)
            out = os.path.join(_leak_dir, "08-leak-snapshot.pkl")
            try:
                torch.cuda.memory._dump_snapshot(out)
                TheMessage(f"[leak] dumped snapshot to {out} at batch {batchNum}")
            except Exception as _leak_exc:
                TheMessage(f"[leak] dump failed: {_leak_exc!r}")
            torch.cuda.memory._record_memory_history(enabled=None)
            self._leak_recording = False

    if os.environ.get("BASIC_PROFILE_TENSORS") and torch.cuda.is_available():
        import gc as _gc, collections as _coll
        counts = _coll.Counter()
        for obj in _gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.device.type == "cuda":
                    counts[(tuple(obj.shape), str(obj.dtype))] += 1
            except Exception:
                continue
        prev = getattr(self, "_leak_prev_counts", None)
        if prev is not None:
            keys = set(counts) | set(prev)
            deltas = sorted(
                ((k, counts.get(k, 0) - prev.get(k, 0)) for k in keys
                 if counts.get(k, 0) != prev.get(k, 0)),
                key=lambda kv: -abs(kv[1]))
            top = ", ".join(f"{k[0]}/{k[1]}:{d:+d}" for k, d in deltas[:8])
            TheMessage(f"[tensors] batch={batchNum} top deltas: {top}")
        self._leak_prev_counts = dict(counts)

    # Clear per-batch IR scratch so the next batch's
    # create_ir_mask starts from a clean slate (no stale mask /
    # pre-mask tensor pinned in GPU memory).
    self._ir_mask_positions = None
    self._ir_pre_mask_input = None

    self.End()
    return result, batchNum
