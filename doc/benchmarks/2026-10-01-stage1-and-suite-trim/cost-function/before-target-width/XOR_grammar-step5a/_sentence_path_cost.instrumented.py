def _sentence_path_cost(self, state, sid, active):
    """The same reconstruction and prediction objective for either trial."""
    self._publish_sentence_scratch(state)
    current_stm, lang, _feedback = state
    B, W, D = self._tensor_pushed_ideas.shape
    slots = lang[13][:, sid].reshape(B, 3, D)
    depths = lang[14][:, sid]
    root = lang[9][:, sid]
    if getattr(self, '_sentence_reconstruction', self.reconstruct_in_loop):
        reconstruction = self._compiled_reconstruct()(
            root.clone(), self._tensor_pushed_ideas, lang[13], lang[14], slots.clone(), depths.clone(),
            torch.tensor(sid, device=root.device))
        cost = self.loss.reconstruction_scale * reconstruction[2]
    else:
        # Configurations without the tied traversal retain their existing
        # batch-end objectives. Their sentence cost has no reconstruction
        # contribution; the zero edge keeps the two training calls valid.
        cost = root.sum(-1) * 0
        reconstruction = (root.new_zeros(B, W, D), cost, cost,
                          torch.zeros(B, dtype=torch.bool, device=root.device),
                          lang[14].to(root) * 0)
    reconstruction_cost, expectation_cost, output_cost = cost, None, None
    observation = self._sentence_observation(state, sid, active)
    if getattr(self, '_reading_lesson_enabled', False) and self.grammar_lesson_weight > 0:
        programs, sources = [], []
        source_rows = self._reading_lesson_sources
        for b, program in enumerate(observation['entries']):
            if program is None or not bool(active[b]):
                continue
            programs.append(program)
            source = None if source_rows is None else source_rows[b]
            if isinstance(source, (tuple, list)):
                source = source[sid] if sid < len(source) else None
            sources.append(source)
        objectives = self._grammar_lesson_objectives(programs,
            split=self._reading_lesson_split, source_rows=sources)
        if objectives:
            lesson = torch.stack(tuple(objectives.values())).sum()
            cost = cost + self.grammar_lesson_weight * lesson
            if 'generate' in objectives:
                output_cost = self.grammar_lesson_weight * objectives['generate']
            self._reading_lesson_reports.append(lesson.detach())
    disc = getattr(self.symbolSpace, 'discourse', None)
    pending = None
    if disc is not None:
        inter, contrast, pending = disc.sentence_prediction_cost(
            observation['observed_depths'], observation['observed'], observation['mask'],
            documents=self._expectation_documents_for_slot(sid, B),
            layout=observation['layout'], role_masks=observation['roles'],
            sentence_kinds=[None if m is None else m.sentence_kind for m in observation['meanings']])
        if self.inter_loss_weight > 0:
            expectation_cost = self.inter_loss_weight * inter
            cost = cost + expectation_cost
        if self.inter_contrastive_weight > 0:
            contrast_cost = self.inter_contrastive_weight * contrast
            expectation_cost = (contrast_cost if expectation_cost is None else
                                expectation_cost + contrast_cost)
            cost = cost + contrast_cost
    answer_error = BasicModel._sentence_answer_error(self, state, sid, active, observation)
    if answer_error is not None:
        cost = cost + answer_error
        output_cost = answer_error if output_cost is None else output_cost + answer_error
    if getattr(self, '_sentence_operator_gradients', None) is not None:
        self._sentence_gradient_objectives = dict(
            reconstruction=reconstruction_cost, expectation=expectation_cost,
            output=output_cost)
    _objective_probe.trial(self, locals())
    return cost, reconstruction, observation, pending
