def _sentence_path_cost(self, state, sid, active):
    """Both trials use the registry's named, relative objectives."""
    self._publish_sentence_scratch(state)
    current_stm, lang, _feedback = state
    B, W, D = self._tensor_pushed_ideas.shape
    slots = lang[13][:, sid].reshape(B, 3, D)
    depths, root = lang[14][:, sid], lang[9][:, sid]
    errors = Error(row_mask=active)
    record = self._trial_understanding(state, sid, active)
    if getattr(self, '_sentence_reconstruction', self.reconstruct_in_loop):
        reconstruction = self._reconstruct_trial(record)
        available = self.inputSpace._reconstruction_sentence_available[:, sid] & active
        errors.error('reconstruction.free_bytes', reconstruction[2], math.log(256), mask=available,
            weight=self.loss.reconstruction_scale, category='reconstruction')
    else:
        zero = root.sum(-1) * 0
        reconstruction = (root.new_zeros(B, W, D), zero, zero,
                          torch.zeros(B, dtype=torch.bool, device=root.device), lang[14].to(root) * 0)
    observation = self._sentence_observation(state, sid, active)
    observation['record'] = record
    observation['decomposition_loss'] = (self._decomposition_teacher_loss(observation, record)
        if getattr(self, '_sentence_training', False) and torch.is_grad_enabled() else None)
    observation['walk_loss'] = (self._decomposition_walk_teacher_loss(observation, record)
        if getattr(self, '_sentence_training', False) and torch.is_grad_enabled() else None)
    if getattr(self, '_reading_lesson_enabled', False) and self.grammar_lesson_weight > 0:
        source_rows = self._reading_lesson_sources
        for b, program in enumerate(observation['entries']):
            if program is None or not bool(active[b]):
                continue
            source = None if source_rows is None else source_rows[b]
            if isinstance(source, (tuple, list)):
                source = source[sid] if sid < len(source) else None
            objectives = self._grammar_lesson_objectives([program],
                split=self._reading_lesson_split, source_rows=[source])
            if objectives:
                for name, value in objectives.items():
                    errors.merge(self._grammar_lesson_errors[name], prefix='grammar.',
                                 weight=self.grammar_lesson_weight, row=b)
                self._reading_lesson_reports.append(torch.stack(tuple(objectives.values())).sum().detach())
    disc = getattr(self.symbolSpace, 'expectation', None)
    pending = None
    self._sentence_expectation_comparison = None
    if disc is not None:
        inter, contrast, pending = disc.sentence_prediction_cost(
            observation['observed_depths'], observation['observed'], observation['mask'],
            documents=self._expectation_documents_for_slot(sid, B),
            layout=observation['layout'], role_masks=observation['roles'],
            sentence_kinds=[None if m is None else m.sentence_kind for m in observation['meanings']],
            gain=getattr(self, 'expectation_gain', 1.))
        gated = disc._sentence_comparison_errors.total()
        if gated is not None:
            self._sentence_expectation_comparison = gated * self.inter_loss_weight
        if self.inter_loss_weight > 0:
            errors.merge(disc._sentence_prediction_errors[0],
                         prefix='expectation.', weight=self.inter_loss_weight)
        if self.inter_contrastive_weight > 0:
            errors.merge(disc._sentence_prediction_errors[1],
                         prefix='expectation.', weight=self.inter_contrastive_weight)
    if getattr(self, '_component_reading', False):
        components = self._concept_owner().components
        estimates = (None if pending is None else [
            None if item is None else item.prediction for item in pending[0]])
        component_cost = components.cost(observation['meanings'], estimates, active,
                                          clauses=observation['clauses'])
        errors.add('expectation.components', component_cost, weight=components.weight,
                   category='expectation')
        cost = component_cost * components.weight
        self._sentence_expectation_comparison = (cost if self._sentence_expectation_comparison is None
            else self._sentence_expectation_comparison + cost)
    word_inputs = getattr(self, '_word_expectation_input', None)
    word_expected = (None if word_inputs is None or disc is None else disc.expect('word', *word_inputs,
        teacher_forcing=bool(self.training), gain=self.word_expectation_gain,
        active=self._word_expectation_mask))
    if word_expected is not None and self.word_expectation_weight > 0:
        mask = self._word_expectation_mask & (self.inputSpace._packed_sentence_ids == sid)
        loss = (word_expected.loss * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        errors.error('expectation.word', loss, 1., mask=active & mask.any(-1),
                     weight=self.word_expectation_weight, category='expectation')
    BasicModel._sentence_answer_error(self, state, sid, active, observation, registry=errors)
    if getattr(self, '_sentence_training', False):
        configure_l1 = getattr(self, '_configure_concept_readout_l1', None)
        penalty = None if configure_l1 is None else configure_l1(None, stage=False)
        if penalty is not None:
            strength = self._concept_readout_l1_strength
            errors.add('concept_readout_l1', penalty / strength, weight=strength,
                       category='reg', trained=False)
    self._sentence_cost_registry = errors
    self._sentence_gradient_objectives = {name: errors.total(objective=name)
        for name in ('reconstruction', 'expectation', 'output')}
    cost = errors.total()
    if cost is None:
        cost = root.sum(-1) * 0
    elif cost.ndim == 0:
        cost = cost.expand(B)
    _objective_probe.trial(self, locals())
    return cost, reconstruction, observation, pending
