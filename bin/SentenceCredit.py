"""Loss-side credit for one departure across the sentence's two walks."""
import torch


def forward_discard(language, programs, meaning_start, like, *, case_bank=None):
    """Pay for meaning lost where a local inverse does not exist.

    The denominator is the step's whole detached input meaning mass, never
    the masked subset. Binary overlap counts once. Domain masks are evaluated
    at each step's actual output, without inherited undefinedness. Defined
    coordinates retain the inverse's existing error and receive no surcharge.
    Search-kind inverses are recoverable by bank search and are exempt even
    though the field derivation reader leaves their coordinates undefined.
    """
    from DerivationReconstruction import inverse_step
    costs = []
    for b,program in enumerate(programs):
        if program is None or program.operation_values is None:
            costs.append(like.new_zeros(()))
            continue
        steps = []
        for index, (kind, operation, _word) in enumerate(program.actions.tolist()):
            if kind not in (1, 2):
                continue
            tree = language._tree_layer(1 if kind == 2 else 2)
            wrapped = (tree.unary_ops if kind == 2 else tree.ops)[operation]
            op = getattr(wrapped, 'gl', wrapped)
            if getattr(op, 'inverse_kind', None) == 'search':
                continue
            frame = program.operation_values[index]
            with torch.no_grad():
                from FormCentroids import inverse_context
                from ChunkJournal import action_context
                children = inverse_step(language, kind, operation, frame[2:3].detach(),
                    frame.detach(), case_bank=None if case_bank is None else case_bank.for_stream(b),
                    action_context=action_context(program, index),
                    **inverse_context(program, index))
                undefined = torch.cat([~domain[:, meaning_start:] for _, domain in children])
                inputs = frame[:len(children), meaning_start:].detach().abs()
                whole_mass = inputs.amax(0).sum()
                uncovered = torch.where(undefined, inputs, 0.).amax(0)
            returned = torch.where(undefined.any(0), frame[2, meaning_start:].abs(), 0.)
            removed = (uncovered - returned).relu().sum()
            steps.append(removed / torch.where(whole_mass > 0, whole_mass, 1.))
        costs.append(torch.stack(steps).sum() if steps else like.new_zeros(()))
    return torch.stack(costs)




WALKS = ('attention', 'compose', 'candidate', 'bind', 'reconstruction', 'thought', 'output')


def departure(narrowing, compose, *, active, sentence_ids=None, sentence=0,
              compose_round=None, candidates=None, binds=None, reconstruction=None,
              thought=None, output=None, used=()):
    """Draw a distinct walk/round, uniformly at each level of the remaining menu."""
    from WalkTrials import departure_at
    empty = compose.new_zeros(compose.shape[0], 0)
    attention = empty if narrowing is None else narrowing.alternatives.clone()
    if narrowing is not None and sentence_ids is not None:
        owners = sentence_ids.gather(1, narrowing.round_words.clamp_min(0))
        attention &= (owners == sentence) & (narrowing.round_words >= 0)
    menus = [attention, compose, candidates, binds, reconstruction, thought, output]
    menus = [empty if menu is None else menu.clone().bool() for menu in menus]
    for prior in used:
        for name, menu in zip(WALKS, menus):
            if menu.shape[1]:
                at = prior[name + '_round']
                menu &= torch.arange(menu.shape[1], device=menu.device)[None] != at[:, None]
    counts = torch.stack([menu.sum(-1) for menu in menus], -1) * active[:, None]
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
    result['round'] = torch.where(walk == 0, result['attention_round'],
        torch.where(walk == 1, attention.shape[1] + result['compose_round'], -1))
    return result


def components(registry, expectation, like, *, answer=None):
    """Only a supplied answer contributes A; bind menus carry no price."""
    def value(v):
        return torch.zeros_like(like) if v is None else v.detach().expand_as(like)
    return torch.stack((value(registry.total(objective='reconstruction')),
        value(expectation), value(answer)), -1)


def comparison(parts, active):
    """Keep the cheapest complete R + A path per row; the exploit wins ties."""
    costs = (parts[..., 0] + parts[..., 2]).detach()
    winner = torch.where(active, costs.argmin(-1), 0)
    delta = parts[:, 1:] - parts[:, :1]
    advantage = costs[:, 1:] - costs[:, :1]
    direction = advantage.sign()
    supports = delta * direction[..., None] > 0
    supports[..., 1] = False
    names = ('reconstruction', 'expectation', 'answer')
    deciding = [('+'.join(name for name, yes in zip(names, row) if yes)
                 if sign else 'tie:greedy') for row, sign in
                zip(supports.reshape(-1, 3).tolist(), direction.reshape(-1).tolist())]
    pair = costs.shape[1] == 2
    return dict(costs=costs, keep_costs=costs, components=parts.detach(), winner=winner,
        wins=winner.ne(0), delta=delta[:, 0].detach() if pair else delta.detach(),
        deciding=deciding, keep_decision=['greedy' if i == 0 else 'explore' for i in winner.tolist()],
        advantage=advantage[:, 0] if pair else advantage,
        advantage_sign=direction[:, 0] if pair else direction,
        policy_against_keep=torch.zeros_like(active), answer_against_keep=torch.zeros_like(active))


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
              if bool(trained.any()) else probabilities.new_zeros(len(costs)))
    return result, dict(costs=costs.detach(), advantage=advantage, mask=mask.detach(),
                        probabilities=probabilities.detach(), scale=scale)


def reader_weights(parts, active, draw):
    """One answer exposure, shared with every compose departure and its exploit."""
    if draw is None:
        raise ValueError('answer exposure requires the whole-path departure draw')
    draws = (draw,) if isinstance(draw, dict) else tuple(draw)
    if len(draws) != parts.shape[1] - 1:
        raise ValueError('one departure draw is required for every explore path')
    winner = comparison(parts, active)['winner']
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
