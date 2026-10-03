def sentence_prediction_cost(self, depths, payloads, mask, *,
                             documents=None, layout="stm", role_masks=None, sentence_kinds=None):
    """Preview one sentence per row without observing either candidate.

    Only the pending predictions are scratch. Context, occurrences, LTM,
    counters and policy outcomes are written once, when the winner arrives.
    The returned pending records retain the estimate belonging to that trial.
    """
    before = (list(self._inter_last_meaning), list(self._inter_last_pred_root))
    like = next((p for p in payloads if p is not None), self._s_history)
    if self._external_observations_suspended or not self.expectation_enabled:
        zero = like.new_zeros(len(payloads))
        return zero, zero.clone(), before
    self._prepare_expectation_documents(documents, mask, len(payloads))
    before = (list(self._inter_last_meaning), list(self._inter_last_pred_root))
    costs, contrastive = [], []
    try:
        for b, payload in enumerate(payloads):
            cost, contrast = like.new_zeros(()), like.new_zeros(())
            if payload is not None and bool(mask[b]) and self._inter_predictor is not None:
                self.predict_next_end_state(b)
                negatives = []
                pred = target = None
                if self.expectation_scope == "structured":
                    roles, occupied = self._canonical_meaning(
                        payload, depths[b], layout,
                        None if role_masks is None else role_masks[b])
                    pending = self._inter_last_meaning[b]
                    prediction = (pending.prediction if isinstance(
                        pending, _PendingMeaningExpectation) else pending)
                    if prediction is not None:
                        pred, logits = prediction.roles, prediction.presence_logits
                        kind_logit = prediction.kind_logit
                        if (isinstance(pending, _PendingMeaningExpectation)
                                and pending.inputs is not None):
                            # Equal-parameter trials still need separate
                            # prediction graphs: the first backward frees
                            # its graph before the second trial trains.
                            # Replay only the stored prior inputs, keeping
                            # the pending record and its estimate intact.
                            replay, presence, kinds = self._inter_predictor(
                                *(v.detach() for v in pending.inputs))
                            pred, logits = replay[0], presence[0]
                            kind_logit = kinds[0]
                        target = roles.detach().to(pred)
                        cost = (pred - target).square().mean() + F.binary_cross_entropy_with_logits(
                            logits, occupied.to(logits))
                        cost = cost + self._kind_loss(kind_logit,
                            None if sentence_kinds is None else sentence_kinds[b], cost)
                        _objective_probe.expectation_parts(self, locals())
                        negatives = [p.detach().to(target).flatten()
                                     for _, p, _ in self._inter_context[b]]
                else:
                    pred = self._inter_last_pred_root[b]
                    pd = payload.detach()
                    if self._ltm_store is not None and pd.dim() == 2 and pd.shape[0] >= 2:
                        pd = pd[pd.shape[0] - 2].unsqueeze(0)
                    target = self._reduce_end_state_to_root(pd)
                    if pred is not None and target is not None:
                        target = target.to(pred)
                        cost = F.mse_loss(pred, target)
                        negatives = [root.detach().to(pred) for _, p, _ in self._inter_context[b]
                                     if (root := self._reduce_end_state_to_root(p)) is not None]
                if (self._inter_contrastive_weight > 0 and pred is not None
                        and target is not None and negatives):
                    q = F.normalize(pred.reshape(-1), dim=0)
                    candidates = torch.stack([target.flatten()] + [n.flatten() for n in negatives])
                    logits = (F.normalize(candidates, dim=-1) @ q) / self._inter_contrastive_temp
                    contrast = F.cross_entropy(logits[None], logits.new_zeros(1, dtype=torch.long))
            if not bool(torch.isfinite(cost) and torch.isfinite(contrast)):
                raise FloatingPointError('non-finite sentence prediction cost')
            costs.append(cost)
            contrastive.append(contrast)
        pending = (list(self._inter_last_meaning), list(self._inter_last_pred_root))
        return torch.stack(costs), torch.stack(contrastive), pending
    finally:
        self._inter_last_meaning, self._inter_last_pred_root = before
