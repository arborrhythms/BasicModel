def truth_modulated_loss(self, total_loss, symbolic_space,
                         symbol_acts=None, universality_score=None,
                         luminosity_weight=0.1, universality_weight=0.1,
                         truth_loss_weight=0.0,
                         allow_excluded_middle=1,
                         allow_contradiction=0,
                         balance_weight=0.1,
                         model=None, gradient_objectives=None):
    """Apply the SymbolSpace-owned TruthLayer modulation to a loss.

    The transform has two parts:

    1. **Multiplicative modulation** -- penalize irrational and
       unkind propositions by scaling ``total_loss`` by
       ``(1 + lum_w * (1 - lum_norm) + u_w * (1 - u_norm))``,
       where ``lum_norm = luminosity(symbolic_space.sigma).clamp(0, 1)``
       and ``u_norm = universality_score.detach().clamp(-1, 1)`` (or 0
       when the caller has no universality score cached yet).

    2. **Additive falsity penalty** -- when
       ``truth_loss_weight > 0`` and the caller provides
       committed symbol activations, add
       ``truth_loss_weight * falsity_penalty(symbol_acts, basis)``
       using ``symbolic_space.subspace.basis``.  ``symbol_acts``
       should be the last entry of the model's ``symbol_states``
       cache -- the post-pi activations from the final Sigma-Pi
       iteration.  Both operands of the
       disjunction then live in symbol space by construction
       (stored truths were also recorded from symbol-space
       activations in ``WholeSpace.forwardEnd``).

    Returns ``total_loss`` unchanged when the TruthLayer is
    absent or empty (bootstrap case with no truths recorded
    yet).  The caller is responsible for only invoking this in
    train mode -- the method itself has no ``train`` flag.

    All inputs that reach outside SymbolSpace (``symbolic_space``,
    ``symbol_acts``, ``universality_score``, the three weights)
    are passed explicitly so SymbolSpace never needs a back-
    reference to the model.

    When supplied, ``gradient_objectives`` contains weighted primary
    losses used for per-operator diagnostics. Apply the same detached
    contextual multiplier to each branch. A context-dependent scale must
    not reopen the concluded-state boundary. Additive truth/balance
    penalties remain independent auxiliary objectives.
    """
    if self.truth_layer is None or self.truth_layer.is_empty():
        _objective_probe.truth_return(locals())
        return total_loss

    # Luminosity is now a Mereology measure on the model itself.
    # When the caller supplies a `model` reference we delegate to
    # `model.Luminosity(truth_layer=...)`; otherwise (legacy path
    # without a model handle) we fall back to a neutral 0.0 score
    # so the multiplicative modulation degenerates to the
    # universality-only term -- preserving training stability for
    # callers that haven't been migrated yet.
    if model is not None and hasattr(model, 'Luminosity'):
        lum_val = float(model.Luminosity(truth_layer=self.truth_layer))
    else:
        lum_val = 0.0
    lum = torch.tensor(lum_val, device=total_loss.device,
                       dtype=total_loss.dtype)
    lum_norm = lum.clamp(0, 1)
    if universality_score is not None:
        u_norm = universality_score.detach().clamp(-1, 1)
    else:
        u_norm = torch.tensor(0.0, device=total_loss.device)

    multiplier = (1 + luminosity_weight * (1 - lum_norm)
                  + universality_weight * (1 - u_norm))
    total_loss = total_loss * multiplier
    registry = getattr(model, 'errors', None) if model is not None else None
    if registry is not None:
        registry.scale(multiplier)
    if gradient_objectives is not None:
        for name, objective in gradient_objectives.items():
            gradient_objectives[name] = objective * multiplier

    if truth_loss_weight > 0 and symbol_acts is not None:
        basis = getattr(
            getattr(symbolic_space, 'subspace', None), 'basis', None)
        if basis is not None:
            truth_penalty = self.truth_layer.falsity_penalty(
                symbol_acts, basis)
            total_loss = total_loss + truth_loss_weight * truth_penalty
            if registry is not None:
                registry.add('truth.falsity', truth_penalty, weight=truth_loss_weight,
                             category='truth', kind='penalty')

    # Quaternary-corner balance penalty: discourages forbidden
    # corners (N, B). The bivector substrate was retired (Phase 3):
    # ``symbol_acts`` is now a single signed scalar, so the old
    # ``symbol_acts[..., :2]`` pole slice is gone. The corner policy
    # instead reads the TruthLayer-internal accumulator -- the only
    # legitimate remaining bivector surface. ``tetralemma_balance_
    # penalty`` is a kept op that returns 0 for a non-paired
    # accumulator, so the term is inert until a paired/bivector
    # accumulator is configured; the Phase 5 client assessment
    # builds on this same accumulator read.
    wants_balance = (int(allow_excluded_middle) == -1
                     or int(allow_contradiction) == 0)
    if (balance_weight > 0 and wants_balance
            and not self.truth_layer.is_empty()):
        n = self.truth_layer.count.item()
        accumulator = self.truth_layer.truths[:n]
        balance = self.truth_layer.tetralemma_balance_penalty(
            accumulator,
            allow_excluded_middle=int(allow_excluded_middle),
            allow_contradiction=int(allow_contradiction))
        total_loss = total_loss + balance_weight * balance
        if registry is not None:
            registry.add('truth.balance', balance, weight=balance_weight,
                         category='truth', kind='penalty')

    _objective_probe.truth_return(locals())
    return total_loss
