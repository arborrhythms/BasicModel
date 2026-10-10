"""Loss-side credit for one departure across the sentence's two walks."""
import torch


def expectation_terms(registry, pred, logits, kind_logit, target, occupied,
                      kind, *, row, gain=1.):
    """Gate each role's error, retaining the Error registry's baseline.

    Gating the denominator too would cancel confidence. Prediction training
    uses its separate, unchanged registry and detached observation targets.
    """
    import math
    from torch.nn import functional as F
    target = target.detach().to(pred)
    gate = torch.as_tensor(gain, device=pred.device, dtype=pred.dtype) * logits.detach().sigmoid()
    registry.error('roles', (pred-target).square()*gate[:,None], target.square(),
                   row=row, category='expectation')
    registry.error('presence', F.binary_cross_entropy_with_logits(logits,
        occupied.to(logits), reduction='none')*gate, math.log(2), row=row, category='expectation')
    if kind is not None:
        error = F.binary_cross_entropy_with_logits(kind_logit,
            kind_logit.new_tensor(float(kind == 'relation')), reduction='none')
        registry.error('kind', error*gate.mean(), math.log(2), row=row, category='expectation')


def departure(narrowing, compose, *, active, sentence_ids=None, sentence=0,
              compose_round=None, candidates=None):
    """Uniform walk, then uniform eligible round; the chooser draws the action."""
    from WalkTrials import departure_at
    if narrowing is None:
        attention = compose.new_zeros(compose.shape[0], 0)
    else:
        attention = narrowing.alternatives.clone()
        if sentence_ids is not None:
            owners = sentence_ids.gather(1, narrowing.round_words.clamp_min(0))
            attention &= (owners == sentence) & (narrowing.round_words >= 0)
    groups = [attention.sum(-1), compose.sum(-1)]
    if candidates is not None:
        groups.append(candidates.sum(-1))
    counts = torch.stack(groups, -1) * active[:, None]
    available = counts > 0
    walk = departure_at(available)
    walk_count = available.sum(-1)
    walk_rounds = torch.where(walk >= 0,
        counts.gather(1, walk.clamp_min(0)[:, None]).squeeze(1), 0)
    width = attention.shape[1]
    attention_round = (departure_at(attention) if width else torch.full_like(walk, -1))
    if compose_round is None:
        if bool(compose.any()):
            raise ValueError('compose departure requires its reservoir snapshot round')
        compose_round = torch.full_like(walk, -1)
    chosen = torch.where(walk == 0, attention_round,
                         torch.where(walk == 1, width + compose_round, -1))
    at_attention, at_compose = walk == 0, walk == 1
    candidate_round = (departure_at(candidates) if candidates is not None and candidates.shape[1]
                       else torch.full_like(walk, -1))
    return dict(round=chosen, rounds=counts.sum(-1), walk=walk,
        walk_count=walk_count, walk_rounds=walk_rounds, narrowing=at_attention,
        attention_round=torch.where(at_attention, chosen, -1),
        compose_round=torch.where(at_compose, chosen-width, -1),
        candidate_round=torch.where(walk == 2, candidate_round, -1))


def components(registry, expectation, like, *, answer=None):
    """R and A use the training registry; E is its gated comparison view."""
    def value(v):
        return torch.zeros_like(like) if v is None else v.detach().expand_as(like)
    return torch.stack((value(registry.total(objective='reconstruction')),
        value(expectation), value(answer)), -1)


def comparison(parts, active):
    """Reconstruction keeps the derivation; R + A credits its departure."""
    costs = (parts[..., 0] + parts[..., 2]).detach()
    delta = parts[:, 1] - parts[:, 0]
    keep_costs = parts[:, :, 0].detach()
    wins = active & (keep_costs[:, 1] < keep_costs[:, 0])
    direction = torch.sign(costs[:, 1]-costs[:, 0])
    supports = delta * direction[:, None] > 0
    supports[:, 1] = False  # E trains predictors, never the departure.
    names = ('reconstruction', 'expectation', 'answer')
    deciding = [('+'.join(name for name, yes in zip(names, row) if yes)
                 if sign else 'tie:greedy') for row, sign in
                zip(supports.tolist(), direction.tolist())]
    keep_decision = [('explore' if d < 0 else 'greedy' if d > 0 else 'tie:greedy')
                     for d in delta[:, 0].tolist()]
    # Negative advantage rewards explore. A positive advantage rewards greedy.
    against_keep = active & direction.ne(0) & ((direction < 0) != wins)
    without_answer = torch.sign(delta[:, 0])
    answer_against_keep = against_keep & (without_answer != direction) & delta[:, 2].ne(0)
    return dict(costs=costs, keep_costs=keep_costs, components=parts.detach(), delta=delta.detach(),
                wins=wins, deciding=deciding, keep_decision=keep_decision,
                advantage=(costs[:, 1]-costs[:, 0]), advantage_sign=direction,
                policy_against_keep=against_keep, answer_against_keep=answer_against_keep)


def score_function(probabilities, scale, mask, costs):
    """The proposal-corrected p(a) ΔC surrogate and exact-tie disconnection."""
    advantage = (costs[:, 1]-costs[:, 0]).detach()
    scale = scale.detach().to(probabilities)
    trained = mask & advantage.ne(0)[:, None]
    result = (torch.where(trained, probabilities * scale * advantage[:, None], 0.).sum(-1)
              if bool(trained.any()) else probabilities.new_zeros(len(costs)))
    return result, dict(costs=costs.detach(), advantage=advantage, mask=mask.detach(),
                        probabilities=probabilities.detach(), scale=scale)


def reader_weights(parts, active, draw):
    """Comparison exposure with a draw; presented kept-only exposure without one."""
    wins = active & (parts[:, 1, 0] < parts[:, 0, 0])
    compose = (torch.zeros_like(active) if draw is None else
               active & (draw['compose_round'] >= 0))
    explore = torch.where(compose, .5, wins.to(parts.dtype)) * active
    return torch.stack((active.to(parts.dtype) - explore, explore), -1)


def reader_costs(registries, weights):
    """Combine the existing output-owned losses without changing their owners."""
    from ObjectiveOwnership import registry_costs
    result = {}
    for registry, rows in zip(registries, weights.unbind(-1), strict=True):
        for name, value in registry_costs(registry, reader_rows=rows).items():
            if name in ('output', 'penalty.output'):
                result[name] = result.get(name, 0) + value
    return result


def expectation_costs(registries, weights):
    """One predictor update against only the reconstruction-kept trial rows."""
    from ObjectiveOwnership import registry_costs
    result = {}
    for registry, rows in zip(registries, weights.unbind(-1), strict=True):
        for name, value in registry_costs(registry, expectation_rows=rows).items():
            if name in ('expectation', 'penalty.expectation'):
                result[name] = result.get(name, 0) + value
    return result
