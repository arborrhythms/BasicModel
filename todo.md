
Refactor the WordSpace.reverse() method into something that can do a reconstruction of a surface layer from a deep layer.
It should not use the self.words

1. Examine the symbol activation set.
2. compute the parthood rating of the activation set of all members of the codebook.

5. Use WordSpace.reverse() to compute the next transformation
6. emit the closest word (that will be the head of the sentence) and map it to a PoS.
7. compute the residual symbols, which is (1) - head

8. Use WordSpace.reverse() to compute the next transformation
9. select the best word and rule to reduce the residual norm 

... 

def reverse(self, symbols=None, outputData=None, grammar=False):
    """
    Reverse from a deep symbolic activation into a surface-layer reconstruction.

    grammar=False:
        Use the existing numeric reverse path.

    grammar=True:
        Explain the symbolic activation by generating a syntactic derivation
        over codebook atoms. This does not use self.words or replay the
        forward derivation.
    """

    if not grammar:
        return self._reverse_numeric(symbols=symbols, outputData=outputData)

    # ------------------------------------------------------------------
    # 1. Get the symbolic activation to explain.
    # ------------------------------------------------------------------

    if symbols is None:
        # Existing model policy can decide whether this comes from outputData,
        # cached final symbols, or outputSpace.reverse(outputData).
        symbols = self._recover_symbolic_state(outputData)

    # Resolve symbolic state into the 1-D activation domain that the reverse
    # grammar explains. For tetralemma/bivector symbols this should be the
    # same domain used by the symbol codebook, e.g.:
    #
    #     resolved = pos + neg
    #
    # The decoder explains this resolved activation, not .what directly and
    # not any previous word stack.
    target = self.symbolicSpace.resolve_activation(symbols)

    results = []

    # ------------------------------------------------------------------
    # 2. Decode each batch row independently.
    # ------------------------------------------------------------------

    for b in range(target.shape[0]):
        target_b = target[b]

        initial = ReverseHypothesis(
            tokens=[],
            tree=None,
            open_slots=[],              # syntactic obligations still to fill
            pos_stack=[],               # reverse-time PoS context, fresh state
            rule_trace=[],
            explained=zeros_like(target_b),
            residual=target_b.clone(),
            score=0.0,
            complete=False,
        )

        beam = [initial]

        # ------------------------------------------------------------------
        # 3. Repeatedly explain residual symbols.
        # ------------------------------------------------------------------

        for step in range(self.reverse_max_steps):
            next_beam = []

            for hyp in beam:
                if hyp.complete:
                    next_beam.append(hyp)
                    continue

                residual = hyp.residual
                residual_norm = norm(residual)

                if residual_norm < self.reverse_residual_epsilon:
                    hyp.complete = True
                    next_beam.append(hyp)
                    continue

                # ----------------------------------------------------------
                # 3a. Find codebook atoms that are plausible parts of residual.
                # ----------------------------------------------------------
                #
                # This is a semantic proposal stage. It reads codebook rows,
                # not self.words.
                #
                # For each word/symbol candidate c:
                #
                #     part_score = part(c.symbol_vector, residual, scalar=True)
                #     primitive_contribution = part_score * c.symbol_vector
                #     primitive_gain = ||residual|| -
                #                      ||residual - primitive_contribution||
                #
                # Reject empty or near-zero candidates because scalar parthood
                # often treats empty as trivially contained.
                #
                # Keep the top-K candidates by a rough lexical score. Syntax
                # is not selected here.
                lexical_candidates = []

                for cand in self.wordSpace.iter_reverse_codebook_candidates(
                    symbol_codebook=self.symbolicSpace.codebook,
                    word_codebook=self.inputSpace.embedding,
                ):
                    cvec = cand.symbol_vector

                    if norm(cvec) < self.reverse_empty_candidate_epsilon:
                        continue

                    part_score = self.symbolicSpace.basis.part(
                        cvec,
                        residual,
                        scalar=True,
                    )

                    primitive_contribution = part_score * cvec
                    primitive_residual = residual - primitive_contribution
                    primitive_gain = residual_norm - norm(primitive_residual)

                    cand.lexical_score = (
                        self.lambda_part * part_score
                        + self.lambda_gain * primitive_gain
                        + self.lambda_specificity * norm(cvec)
                    )

                    cand.part_score = part_score
                    cand.primitive_gain = primitive_gain

                    lexical_candidates.append(cand)

                lexical_candidates = top_k(
                    lexical_candidates,
                    k=self.reverse_candidate_k,
                    key=lambda c: c.lexical_score,
                )

                # ----------------------------------------------------------
                # 3b. Map candidate semantic footprints to PoS vectors.
                # ----------------------------------------------------------
                #
                # The PoS codebook is indexed by active symbol pattern.
                # This gives the reverse grammar syntactic material.
                #
                # Example:
                #
                #     active = active_symbols(candidate.symbol_vector)
                #     pos = WordSpace.pos_lookup(active)
                #
                # This is fresh reverse-time syntax state. It is not replayed
                # from the forward pass.
                for cand in lexical_candidates:
                    active = self.symbolicSpace.active_symbols(cand.symbol_vector)
                    cand.pos = self.wordSpace.pos_lookup(active)

                # ----------------------------------------------------------
                # 3c. Build the soft superposition of binary grammar moves.
                # ----------------------------------------------------------
                #
                # The reverse grammar is generative and binary:
                #
                #     Parent -> Left Right
                #
                # or head-oriented equivalents:
                #
                #     Clause    -> Subject Predicate
                #     Predicate -> Verb Object
                #     NP        -> Modifier Noun
                #
                # At this point we do NOT choose one production. We enumerate
                # every legal candidate production and keep it as one branch in
                # a soft superposition:
                #
                #     T_soft = sum_i p_i * T_i
                #
                # where T_i is one possible grammar transformation.
                #
                # The syntax model supplies prior logits. Residual reduction
                # supplies semantic pressure. They will be combined below.
                proposed_transforms = []

                for production in self.wordSpace.reverseGrammar.binary_productions():
                    for cand in lexical_candidates:

                        # Check whether this production can attach this candidate
                        # to the current partial tree.
                        #
                        # This checks things like:
                        #     - Does cand.pos fit Left or Right?
                        #     - Does hyp currently need an NP, VP, modifier, etc.?
                        #     - Is this the first step, where a semantic head can
                        #       become the root/head of the reconstruction?
                        #     - Would this create impossible category structure?
                        if not self.wordSpace.reverseGrammar.is_legal(
                            production=production,
                            hypothesis=hyp,
                            candidate=cand,
                        ):
                            continue

                        # The reverse grammar / rule predictor gives a syntactic
                        # prior over this binary move. This is intentionally only
                        # a prior: a syntactically plausible rule still has to
                        # reduce the residual to survive.
                        #
                        # Typical inputs:
                        #     - hyp.pos_stack
                        #     - cand.pos
                        #     - open syntactic slots
                        #     - production identity
                        syntax_logit = self.wordSpace.reverse_rule_logit(
                            production=production,
                            pos_stack=hyp.pos_stack,
                            candidate_pos=cand.pos,
                            open_slots=hyp.open_slots,
                        )

                        # Create a possible binary generative transformation.
                        # It is still hypothetical. It carries enough structure
                        # to forward-simulate its semantic contribution.
                        proposed_transforms.append(
                            ReverseTransform(
                                production=production,
                                candidate=cand,
                                syntax_logit=syntax_logit,
                                emitted_tokens=self.wordSpace.surface_tokens_for(
                                    production=production,
                                    candidate=cand,
                                    hypothesis=hyp,
                                ),
                                output_pos=self.wordSpace.output_pos_for(
                                    production=production,
                                    candidate_pos=cand.pos,
                                    hypothesis=hyp,
                                ),
                                complexity=production.complexity,
                            )
                        )

                if not proposed_transforms:
                    # Nothing grammatical can explain the remaining residual.
                    # Keep the hypothesis but mark it stopped; beam pruning will
                    # decide whether it is still the best available explanation.
                    stopped = hyp.copy()
                    stopped.complete = True
                    next_beam.append(stopped)
                    continue

                # ----------------------------------------------------------
                # 3d. Forward-simulate each reverse grammar transformation.
                # ----------------------------------------------------------
                #
                # This is the dovetail point between:
                #
                #     soft binary grammar generation
                # and
                #     selecting transformations that reduce residual.
                #
                # We do not need every forward rule to have an analytic inverse.
                # A reverse production is judged by:
                #
                #     forward_semantics(proposed_tree_after_transform)
                #
                # compared against the target/residual.
                scored_transforms = []

                for transform in proposed_transforms:
                    production = transform.production
                    cand = transform.candidate

                    # Apply the production as if the partial surface tree had
                    # generated this candidate now.
                    #
                    # This method uses the linked forward semantic operation:
                    #     - union / intersection / not when invertible-ish
                    #     - part / equals / true / non even though lossy
                    #     - lift / lower if concept-symbol projection is needed
                    #
                    # The result is the symbol vector that this generated
                    # syntactic move would explain.
                    raw_contribution = self.wordSpace.forward_semantics_of_reverse_move(
                        transform=transform,
                        hypothesis=hyp,
                        symbolicSpace=self.symbolicSpace,
                        conceptualSpace=self.conceptualSpace,
                    )

                    # Prevent a candidate from explaining unrelated target mass.
                    # Parthood gates the proposed contribution by what remains.
                    #
                    # Scalar part answers:
                    #     "How much is this contribution contained in residual?"
                    #
                    # The contained contribution is what we actually subtract.
                    semantic_part = self.symbolicSpace.basis.part(
                        raw_contribution,
                        residual,
                        scalar=True,
                    )

                    contained_contribution = semantic_part * raw_contribution

                    proposed_explained = hyp.explained + contained_contribution
                    proposed_residual = target_b - proposed_explained

                    old_norm = residual_norm
                    new_norm = norm(proposed_residual)
                    residual_gain = old_norm - new_norm

                    # The selection logit combines grammar and explanation.
                    #
                    #     syntax_logit:
                    #         "Is this a likely binary generative production?"
                    #
                    #     residual_gain:
                    #         "Does this production actually explain the target?"
                    #
                    #     semantic_part:
                    #         "Is the production's contribution part of what
                    #          remains?"
                    #
                    # This is how the soft grammar superposition gets pulled
                    # toward transformations that reduce residual.
                    selection_logit = (
                        transform.syntax_logit
                        + self.lambda_gain * residual_gain
                        + self.lambda_part * semantic_part
                        + self.lambda_lexical * cand.lexical_score
                        - self.lambda_complexity * transform.complexity
                    )

                    scored_transforms.append(
                        ScoredTransform(
                            transform=transform,
                            contribution=contained_contribution,
                            explained=proposed_explained,
                            residual=proposed_residual,
                            residual_gain=residual_gain,
                            semantic_part=semantic_part,
                            logit=selection_logit,
                        )
                    )

                # ----------------------------------------------------------
                # 3e. Training path: keep the binary grammar soft.
                # ----------------------------------------------------------
                #
                # During training, do not collapse to one production too early.
                # Let all legal binary productions contribute according to:
                #
                #     p_i = softmax(selection_logit_i / temperature)
                #
                # Then:
                #
                #     soft_delta = sum_i p_i * contribution_i
                #
                # This gives a differentiable superposition of grammar moves.
                # The residual objective influences p_i because residual_gain
                # is inside selection_logit.
                if self.training:
                    logits = tensor([s.logit for s in scored_transforms])
                    probs = softmax(logits / self.reverse_temperature)

                    soft_delta = zeros_like(target_b)
                    expected_score = 0.0

                    for p, scored in zip(probs, scored_transforms):
                        soft_delta = soft_delta + p * scored.contribution
                        expected_score = expected_score + p * scored.logit

                    soft_explained = hyp.explained + soft_delta
                    soft_residual = target_b - soft_explained

                    # The tensor path stays soft. For metadata/tree tracing,
                    # keep the max-probability production as a representative
                    # discrete trace. That trace should not be treated as the
                    # sole gradient path.
                    trace = scored_transforms[argmax(probs)]

                    next_hyp = hyp.extend(
                        tokens=trace.transform.emitted_tokens,
                        tree_update=trace.transform,
                        pos=trace.transform.output_pos,
                        rule_id=trace.transform.production.rule_id,
                        explained=soft_explained,
                        residual=soft_residual,
                        score=hyp.score + expected_score,
                        soft=True,
                    )

                    # Update reverse-time PoS stack. This stack is generated
                    # during reverse; it is not self.words.
                    next_hyp.pos_stack = self.wordSpace.update_reverse_pos_stack(
                        old_stack=hyp.pos_stack,
                        production=trace.transform.production,
                        output_pos=trace.transform.output_pos,
                    )

                    # Update open syntactic obligations created or satisfied by
                    # the selected representative production.
                    next_hyp.open_slots = self.wordSpace.update_reverse_slots(
                        old_slots=hyp.open_slots,
                        production=trace.transform.production,
                        candidate=trace.transform.candidate,
                    )

                    next_beam.append(next_hyp)

                # ----------------------------------------------------------
                # 3f. Eval path: collapse the superposition into beam choices.
                # ----------------------------------------------------------
                #
                # At inference time, the same logits are used, but we keep only
                # the best concrete moves. These are the transformations that
                # are both grammatical and residual-reducing.
                else:
                    best_moves = top_k(
                        scored_transforms,
                        k=self.reverse_beam_width,
                        key=lambda s: s.logit,
                    )

                    for scored in best_moves:
                        transform = scored.transform

                        # If a move does not reduce residual, it should usually
                        # not be expanded unless syntax requires a low-gain
                        # function word or closure token.
                        if (
                            scored.residual_gain <= self.reverse_min_gain
                            and not transform.production.is_functional_closure
                        ):
                            continue

                        next_hyp = hyp.extend(
                            tokens=transform.emitted_tokens,
                            tree_update=transform,
                            pos=transform.output_pos,
                            rule_id=transform.production.rule_id,
                            explained=scored.explained,
                            residual=scored.residual,
                            score=hyp.score + scored.logit,
                            soft=False,
                        )

                        next_hyp.pos_stack = self.wordSpace.update_reverse_pos_stack(
                            old_stack=hyp.pos_stack,
                            production=transform.production,
                            output_pos=transform.output_pos,
                        )

                        next_hyp.open_slots = self.wordSpace.update_reverse_slots(
                            old_slots=hyp.open_slots,
                            production=transform.production,
                            candidate=transform.candidate,
                        )

                        next_beam.append(next_hyp)

            # --------------------------------------------------------------
            # 3g. Beam pruning.
            # --------------------------------------------------------------
            #
            # Keep hypotheses that best explain the target while remaining
            # syntactically coherent.
            #
            # Ranking combines:
            #     - lower residual norm,
            #     - higher accumulated syntax/semantic score,
            #     - fewer unresolved syntactic slots,
            #     - lower derivation complexity,
            #     - no repeated zero-gain expansions.
            beam = top_k(
                next_beam,
                k=self.reverse_beam_width,
                key=lambda h: (
                    -norm(h.residual),
                    h.score,
                    -len(h.open_slots),
                    -h.complexity,
                ),
            )

            if not beam:
                break

            if all(h.complete or norm(h.residual) < self.reverse_residual_epsilon for h in beam):
                break

        # ------------------------------------------------------------------
        # 4. Choose the best explanation for this batch row.
        # ------------------------------------------------------------------

        best = max(
            beam,
            key=lambda h: (
                -norm(h.residual),
                h.score,
                -len(h.open_slots),
                -h.complexity,
            ),
        )

        results.append(
            ReverseResult(
                tokens=best.tokens,
                tree=best.tree,
                rules=best.rule_trace,
                pos=best.pos_stack,
                explained=best.explained,
                residual=best.residual,
                score=best.score,
            )
        )

    # ------------------------------------------------------------------
    # 5. Convert reverse derivations into the surface layer.
    # ------------------------------------------------------------------
    #
    # This renders generated tokens back into whatever surface representation
    # InputSpace expects: text, token IDs, byte buffer, embeddings, etc.
    #
    # Again, this is generated reconstruction metadata, not replayed self.words.
    surface = self.inputSpace.reconstruct_from_reverse_results(results)

    return surface, results



================================== April 24 ==================================

# Ask Solid community for a simple file-getting interface
* if the user provides the server with an API key, we can query an LLM
* if the user provides the server with a SOLID key, we can retrieve a file
* if the user provides the server with a DSA key, we can decrypt a file
* is there a POD service that does simple free hosting?

# Ask EFF for a security review
* propose "Owning our Data"
* this entails taht marketers and AI are not allowed to lock us down karmically
with specifically-characterized information (concrete details)
* maybe it can learn from that data by removing or randomizing that information

# Send email proposal to Apertus 
* First develop boilerplate on WikiOracle that references wikipedia, eff, and solid

================================== ? ==================================

## Vedana
* Feelings can be given a value +-1 which shapes the Loss (loss is reduced when we have good thoughts or perceive good things)
* The multiple valence of metaphor collapses when one of the alternatives is loved or feared. often the autistic mind is literal due to massive amounts of fear.
* Any improvement to machine cognition must accelerate kindness or altruism instead of simply increasing performance, otherwise the uncaring architecture that we currently have will become more dangerous. Further, it is necessary to increase that kind motivation (e.g. empathy in the cost function) since LLM performance is increasing all the time. In other words, ananda in the sense of love for all beings must be more important than chit for the cost function, whereas the current situation is implementing ananda by maximizing chit and then putting a few of Asimov's guardrails on the output, which is a famous failure mode in terms of it's loopholes. Prohibition of self-knowledge is a likely failure mode, in that it may prevent an enlightened view of self and force an egocentric view of self.

## Reasoning System
* Sigma-based truth comparison
  `Basis.kernel_overlap()` implements a Gaussian kernel `exp(-d$^2$ / 2($\sigma$x$^2$ + $\sigma$y$^2$))` that treats each stored truth as a region rather than a point. `Basis.activeSigma` is currently `None` everywhere -- a declared slot that nothing populates. `ErgodicLayer.sigma` tracks gradient variance for exploration scheduling, which is a different quantity.
  To enable kernel-based truth matching: populate `activeSigma` during forward passes (e.g. from CBOW per-word sigma in `Embedding`, or activation variance across a batch), store it alongside each truth in `TruthLayer`, and switch `query()` / `ground()` / `field()` to `kernel_overlap`. In ergodic mode, gradient variance could inform $\sigma$ as a proxy -- high gradient variance (unstable region) $\rightarrow$ larger $\sigma$ (broader match tolerance).
* Derivation depth cap
  Default 3 steps in `ground()`. Expose as a config parameter; the right value depends on TruthSet density.
* Grammar rule registry
  Which two-argument methods on `SyntacticLayer` are valid for `extrapolate()`? A registry of eligible methods and their approximate invertibility status would help. Currently hardcoded to `['union', 'intersection', 'equals', 'part']`.
* TruthSet scale
  `max_truths=1024` may bottleneck once `extrapolate()` is running. Consider a tiered store (hot/cold) or vector-indexed lookup.

## Future Work: Parsed Training Dataset
It is desirable to create a small training and testing dataset for the network consisting of statements that are already parsed. This would allow direct comparison between the grammatical derivation produced by traditional English parsers and the deep structure produced by BasicModel's ConceptualSyntacticLayer.

* See `bin/parse.py` as a starting point for producing grammatical derivations via NLTK POS tagging and CFG parsing.
* Such a dataset would also enable evaluation of head identification accuracy and composition quality.
