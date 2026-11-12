"""The concluded binding and its live numerical sentence view."""
from dataclasses import replace
import torch


def accepted(meaning):
    from ThoughtReferences import bindings, open_slots
    data = bindings(meaning)
    return (not open_slots(meaning) and data.get('_evidence_kind') != 'intuition'
            and (data.get('_producing_operation') in ('infer', 'query', 'isTrue')
                 or data.get('_evidence_kind') in ('pervasion', 'retrieval', 'truth')))


def bound_view(state, observation, sentence, row, meaning):
    """Substitute only this row's completed content into the existing reader."""
    language = list(state[1])
    for index in (6, 10, 11):
        language[index] = language[index].clone()
    count = int(meaning.role_mask.sum())
    from ThoughtReferences import open_slots
    if count not in (1, 3) or not accepted(meaning):
        raise ValueError('the answer reader requires a completed one- or three-role meaning')
    slots = meaning.roles[meaning.role_mask]
    # Sentence physical order is NP2, NP1, VP for a ternary row.
    if count == 3:
        slots = slots[[2,0,1]]
    packed = torch.zeros_like(meaning.roles)
    packed[:count] = slots
    language[6][row,sentence] = slots[-1]
    language[10][row,sentence] = packed.flatten()
    language[11][row,sentence] = count
    view = dict(observation)
    view['meanings'] = list(observation['meanings'])
    view['meanings'][row] = meaning
    view['clauses'] = list(observation['clauses'])
    clause = view['clauses'][row]
    if clause is not None:
        view['clauses'][row] = replace(clause, meaning=meaning,
            point=slots[0] if count == 1 else None,
            relation=None if count == 1 else (clause.relation or 'operator'))
    record = observation['record']
    root, end, depth = record.root.clone(),record.end_slots.clone(),record.end_depth.clone()
    root[row],end[row],depth[row] = slots[-1],packed,count
    view['record'] = replace(record,root=root,end_slots=end,end_depth=depth)
    return (state[0],tuple(language),state[2]),view
