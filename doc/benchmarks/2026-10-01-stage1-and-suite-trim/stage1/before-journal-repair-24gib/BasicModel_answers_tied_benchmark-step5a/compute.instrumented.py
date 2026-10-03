def compute(self, pred, target, nWhere=None, nWhen=None):
    """Per-slot MSE with what/where/when weighting (legacy).

    The constructor band defaults to canonical_shape("OutputSpace")
    == (0, 0) -- right for the lossOut call (output events carry no
    band), wrong for event comparisons: call sites comparing muxed
    ``[what|where|when]`` events pass their own layout's widths
    (the 2026-07-04 nWhere=0 lossRev wiring fix).
    """
    embSize = pred.shape[-1]
    nWhere = self.nWhere if nWhere is None else int(nWhere)
    nWhen = self.nWhen if nWhen is None else int(nWhen)
    nWhat = embSize - nWhere - nWhen

    loss = pred.new_tensor(0.0)
    if nWhat > 0:
        loss = loss + self.what_scale * F.mse_loss(
            pred[..., :nWhat], target[..., :nWhat])
    if nWhere > 0:
        loss = loss + self.where_scale * F.mse_loss(
            pred[..., nWhat:nWhat + nWhere], target[..., nWhat:nWhat + nWhere])
    if nWhen > 0:
        loss = loss + self.when_scale * F.mse_loss(
            pred[..., nWhat + nWhere:], target[..., nWhat + nWhere:])
    if not torch.compiler.is_compiling():
        _objective_probe.compute(self, locals())
    return loss
