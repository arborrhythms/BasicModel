"""Loss-side credit for one departure across the sentence's two walks."""

import torch


WALKS = ('attention', 'compose', 'bind', 'thought', 'output')


def departure(narrowing, compose, *, active, sentence_ids=None, sentence=0,
              compose_round=None, interpretations=None, binds=None,
              thought=None, output=None, used=()):
    """Draw a distinct walk/round, uniformly at each level of the remaining menu."""
    from WalkTrials import departure_at
    empty = compose.new_zeros(compose.shape[0], 0)
    attention = empty if narrowing is None else narrowing.alternatives.clone()
    if narrowing is not None and sentence_ids is not None:
        owners = sentence_ids.gather(1, narrowing.actions.clamp_min(0))
        attention &= (owners == sentence) & (narrowing.actions >= 0)
    interpretation = empty if interpretations is None else interpretations.clone().bool()
    for prior in used:
        if interpretation.shape[1]:
            at = prior.get('interpret_round', torch.full_like(active, -1, dtype=torch.long))
            interpretation &= torch.arange(interpretation.shape[1], device=active.device)[None] != at[:, None]
    menus = [attention, compose, binds, thought, output]
    menus = [empty if menu is None else menu.clone().bool() for menu in menus]
    for prior in used:
        for name, menu in zip(WALKS, menus):
            if menu.shape[1]:
                at = prior[name + '_round']
                menu &= torch.arange(menu.shape[1], device=menu.device)[None] != at[:, None]
    counts = torch.stack([menu.sum(-1) for menu in menus], -1) * active[:, None]
    counts[:, 1] += interpretation.sum(-1) * active
    walk = departure_at(counts > 0)
    result = dict(walk=walk, rounds=counts.sum(-1), walk_count=(counts > 0).sum(-1),
        walk_rounds=torch.where(walk >= 0, counts.gather(1, walk.clamp_min(0)[:, None]).squeeze(1), 0),
        narrowing=walk == 0)
    for i, (name, menu) in enumerate(zip(WALKS, menus)):
        if name == 'compose' and compose_round is not None:
            at = compose_round
        else:
            at = departure_at(menu) if menu.shape[1] else torch.full_like(walk, -1)
        result[name + '_round'] = torch.where(walk == i, at, -1)
    # Interpretation is a unary grammar choice in the compose walk. The
    # existing reservoir samples its other rounds uniformly; select between
    # those and the interpretation rounds in proportion to their counts.
    result['interpret_round'] = torch.full_like(walk, -1)
    if interpretation.shape[1]:
        pick = torch.rand(len(walk), device=walk.device) * counts[:, 1]
        chosen = (walk == 1) & (pick < interpretation.sum(-1))
        result['interpret_round'] = torch.where(chosen, departure_at(interpretation), -1)
        result['compose_round'] = torch.where(chosen, -1, result['compose_round'])
    result['round'] = torch.where(walk == 0, result['attention_round'],
        torch.where(walk == 1, attention.shape[1] + result['compose_round'], -1))
    return result


def components(registry, expectation, like, *, answer=None):
    """Only a supplied answer contributes A; bind menus carry no price."""
    def value(v):
        return torch.zeros_like(like) if v is None else v.detach().expand_as(like)
    return torch.stack((value(registry.total(objective='reconstruction')),
        value(expectation), value(answer)), -1)


def costs_and_winner(parts, active):
    """Select a complete path per stream. The exploit wins a tie."""
    costs = (parts[..., 0] + parts[..., 2]).detach()
    return costs, torch.where(active, costs.argmin(-1), 0)


def path_comparison(parts, active):
    """Retain the numerical comparison used by the sentence and its report."""
    costs, winner = costs_and_winner(parts, active)
    delta = parts[:, 1:] - parts[:, :1]
    advantage = costs[:, 1:] - costs[:, :1]
    direction = advantage.sign()
    pair = costs.shape[1] == 2
    return dict(costs=costs, keep_costs=costs, components=parts.detach(), winner=winner,
        wins=winner.ne(0), delta=delta[:, 0].detach() if pair else delta.detach(),
        advantage=advantage[:, 0] if pair else advantage,
        advantage_sign=direction[:, 0] if pair else direction,
        policy_against_keep=torch.zeros_like(active), answer_against_keep=torch.zeros_like(active))


def credit_schedule(draws):
    """Which policies each exploration used, plus its host thought record per stream."""
    if not draws:
        return ()
    if torch.compiler.is_compiling():
        return [dict(reading=True, bind=True, interpret=True, attention=True, output=True,
                     thoughts=[-1] * len(draw['thought_round'])) for draw in draws]
    flags = []
    for draw in draws:
        walk = draw['walk']
        flags.append(torch.stack(((walk >= 0) & (walk < 3), draw['bind_round'] >= 0,
                                  draw['interpret_round'] >= 0, draw['attention_round'] >= 0,
                                  draw['output_round'] >= 0), 1).any(0))
    thoughts = torch.stack([draw['thought_round'] for draw in draws])
    rows = torch.cat((torch.stack(flags).long(), thoughts), 1).tolist()
    return [dict(reading=reading, bind=bind, interpret=interpret, attention=attention,
                 output=output, thoughts=choices)
            for reading, bind, interpret, attention, output, *choices in rows]


def describe_comparison(audit):
    """Materialize human-readable reasons only when the audit is inspected."""
    if 'deciding' not in audit:
        direction = audit['advantage_sign'].reshape(-1)
        supports = audit['delta'].reshape(-1, 3) * direction[:, None] > 0
        supports[:, 1] = False
        names = ('reconstruction', 'expectation', 'answer')
        audit['deciding'] = [('+'.join(name for name, yes in zip(names, row) if yes)
                             if sign else 'tie:greedy') for row, sign in
                             zip(supports.tolist(), direction.tolist(), strict=True)]
        audit['keep_decision'] = ['greedy' if i == 0 else 'explore' for i in audit['winner'].tolist()]
    return audit


def comparison(parts, active):
    return describe_comparison(path_comparison(parts, active))


def score_function(probabilities, scale, mask, costs):
    """The proposal-corrected p(a) ΔC surrogate and exact-tie disconnection."""
    advantage = (costs[:, 1]-costs[:, 0]).detach()
    scale = scale.detach().to(probabilities)
    trained = mask & advantage.ne(0)[:, None]
    torch._assert_async((~trained | (torch.isfinite(scale) & (scale > 0))).all(),
                        'credited departures require a positive finite proposal correction')
    torch._assert_async((~trained | torch.isfinite(advantage[:, None])).all(),
                        'credited departures require a finite paired advantage')
    # An unused proposal can have zero probability and an infinite inverse
    # correction. Mask its factors before multiplication: masking the product
    # afterwards still differentiates 0 * inf and poisons the scorer's gradient.
    probability = torch.where(trained, probabilities, 0.)
    correction = torch.where(trained, scale, 0.)
    difference = torch.where(trained, advantage[:, None], 0.)
    result = ((probability * correction * difference).sum(-1)
              if torch.compiler.is_compiling() or bool(trained.any())
              else probabilities.new_zeros(len(costs)))
    return result, dict(costs=costs.detach(), advantage=advantage, mask=mask.detach(),
                        probabilities=probabilities.detach(), scale=scale)


def reader_weights(parts, active, draw, *, winner=None):
    """One answer exposure, shared with every compose departure and its exploit."""
    if draw is None:
        raise ValueError('answer exposure requires the whole-path departure draw')
    draws = (draw,) if isinstance(draw, dict) else tuple(draw)
    if len(draws) != parts.shape[1] - 1:
        raise ValueError('one departure draw is required for every explore path')
    if winner is None:
        _, winner = costs_and_winner(parts, active)
    exposure = torch.nn.functional.one_hot(winner, parts.shape[1]).bool()
    compose = torch.stack([item['compose_round'] >= 0 for item in draws], -1) if draws else active[:, None][:, :0]
    exposure[:, 1:] |= compose
    exposure[:, 0] |= compose.any(-1)
    exposure &= active[:, None]
    return exposure.to(parts) / exposure.sum(-1, keepdim=True).clamp_min(1)


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
