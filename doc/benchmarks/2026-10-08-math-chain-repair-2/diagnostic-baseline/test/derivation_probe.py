"""Name diagnostic derivations from the model's held catalogue, never a new parse.

This module observes existing tensors only. It neither draws random numbers nor
performs a forward, changes model state, or retains a training graph.
"""


def compose_catalogue(language):
    result = {}
    for label in ('binary', 'unary'):
        ids = getattr(language, f'_cs_{label}_rule_ids').detach().cpu().tolist()
        rules = getattr(language, f'_compose_{label}_rules')
        if len(ids) != len(rules):
            raise ValueError('compose audit requires the held rule catalogue')
        for rule_id, rule in zip(ids, rules):
            result[rule_id] = dict(rule_id=rule_id, rule_name=rule.method_name,
                                   surface_name=rule.surface_name)
    return result


def named_compose_sequence(language, rule_ids, arities, positions):
    catalogue = compose_catalogue(language)
    result = []
    for rule_id, arity, position in zip(rule_ids, arities, positions, strict=True):
        if rule_id not in catalogue:
            raise ValueError(f'unknown compose rule id {rule_id}')
        result.append(dict(arity=arity, **catalogue[rule_id], position=position))
    return result


def compose_derivations(model, state, sid, active, *, record=None):
    lang = state[1]
    owners, _, _ = model._compose_round_owners(lang[4])
    if record is None:
        record = model._last_sentence_understanding
    result = []
    for row in active.nonzero().flatten().tolist():
        valid = lang[6][row] & (owners[row] == sid)
        sequence = named_compose_sequence(model.languageSpace,
            lang[4][row][valid].detach().cpu().tolist(),
            lang[5][row][valid].detach().cpu().tolist(),
            lang[17][row][valid].detach().cpu().tolist())
        result.append(dict(batch_row=row, sentence_slot=int(sid),
            word_rows=record.word_rows[row][record.word_valid[row]].detach().cpu().tolist(),
            sequence=sequence))
    return result


def named_decoder_sequence(language, actions):
    catalogue = []
    for label in ('binary', 'unary'):
        ids = getattr(language, f'_generate_{label}_rule_ids').detach().cpu().tolist()
        names = getattr(language, f'_generate_{label}_names')
        catalogue.extend(dict(rule_id=rule_id, rule_name=name)
                         for rule_id, name in zip(ids, names, strict=True))
    catalogue.append(dict(rule_id=None, rule_name='STOP'))
    return [dict(round=round, action_id=action, **catalogue[action])
            for round, action in enumerate(actions) if action >= 0]


def named_walk_observations(model):
    """Name grammar rule IDs without mistaking placement actions for rule IDs."""
    catalogue = compose_catalogue(model.languageSpace)
    result = []
    for value in getattr(model, '_walk_observations', ()):
        row = dict(value)
        if row['kind'] == 'compose':
            # The stability trace uses flattened placement candidates (and
            # STOP), not grammar IDs. Full named rule/position sequences are
            # recorded from lang[4:7], lang[17] at the trial and commit boundary.
            row['action_encoding'] = 'flattened placement candidates, not rule IDs'
            row['rule_catalogue'] = list(catalogue.values())
        elif row['kind'] in ('generate.decoder', 'generate'):
            row['derivation'] = named_decoder_sequence(model.languageSpace, row['actions'])
        result.append(row)
    return result
