"""Serial ideas addressed by their thought-step .when in constructed STM.

Sentence and thought ideas share the model clock and RowTime ladder. A new
deposit advances it once; reads and updates keep the original address.
"""
from dataclasses import replace
import torch
from architecture import WHERE_WIRES

from Memory import STMChunks as chunks
from ThoughtReferences import bindings, evidence_pair, open_slots, with_slots


def _next_when(stm, row):
    model = stm._time_owner()
    if model is None:
        raise RuntimeError('thought time requires its live model clock')
    model._advance_when_time()
    return tuple(model.present_code().tolist())


def attach(memory, stm):
    if stm is None:
        raise ValueError('thinking requires the constructed STM owner')
    memory._thought_stm = stm


def _index(stm, when, row):
    state = stm.state
    target = torch.as_tensor(when, device=state[chunks.WHEN].device,
                             dtype=state[chunks.WHEN].dtype).expand(state[chunks.WHEN].shape[0], 4)
    matches = (state[chunks.WHEN][row] == target[row]).all(-1) & (state[chunks.READ][row] > 0)
    if int(matches.sum()) > 1:
        raise ValueError('multiple thought chunks occupy the same .when address')
    index = chunks.locate(state, when, band=chunks.WHEN)
    if int(index[row]) < 0:
        raise ValueError('thought chunk is absent at its .when address')
    return index


def held(stm, row):
    """Enumerating a menu neither reads nor retains an evicted chunk."""
    state = stm.state
    result = []
    for index, item in enumerate(stm._thought_contents[row]):
        if item is None:
            continue
        when, view = item
        stamp = state[chunks.WHEN].new_tensor(when)
        if bool(state[chunks.READ][row, index] > 0) and torch.equal(state[chunks.WHEN][row, index], stamp):
            meaning = view.read()
            result.append((when, with_slots(meaning, open_slots(meaning),
                pair=state[chunks.TRUTH][row, index].tolist())))
    return tuple(result)


def recalled(stm, row):
    """Recalled records are read from their live slot, never the trace copy."""
    result = []
    for when, meaning in held(stm, row):
        data = bindings(meaning)
        occurrence = data.get('_rule_row')
        if data.get('_evidence_kind') != 'retrieval' or occurrence is None:
            continue
        kind = data.get('_rule_kind')
        readings = ('equality',) if kind == 'equal' else ('conditional',) if kind == 'implies' else ('part',)
        result.append(dict(occurrence=occurrence, meaning=meaning,
            evidence=evidence_pair(meaning), trust=evidence_pair(meaning)[0]-evidence_pair(meaning)[1],
            time=meaning.roles.new_tensor(when),
            refs=tuple(meaning.role_refs[slot] for slot in (0, 2)
                       if meaning.role_refs[slot] is not None),
            readings=readings, temporal_witnesses=tuple(ref for ref in
                data.get('_thought_witnesses', ()) if ref != occurrence)))
    return tuple(result)


def read(stm, when, row):
    index = _index(stm, when, row)
    item = stm._thought_contents[row][int(index[row])]
    if item is None or item[0] != tuple(when):
        raise ValueError('thought chunk has no retained semantic reference')
    state = stm.state
    gate = torch.arange(len(index), device=index.device) == row
    pair = state[chunks.TRUTH][row, index[row]].tolist()
    stm.set_state(chunks.read(state, index, gate))
    meaning = item[1].read()
    return with_slots(meaning, open_slots(meaning), pair=pair)


def put(stm, meaning, when=None, row=0, *, pin=False, existing=False):
    state = stm.state
    if meaning.roles.shape[-1] != state[0].shape[-1]:
        raise ValueError('thought content differs from the constructed STM width')
    gate = torch.arange(state[0].shape[0], device=state[0].device) == row
    pair = (0., 0.) if open_slots(meaning) else evidence_pair(meaning)
    # Working content is the occupied ideas' union. Relations remain references.
    value = meaning.roles[meaning.role_mask].amax(0).detach().to(state[0])
    if existing:
        index = _index(stm, when, row)
        result = list(state)
        for field, member in ((chunks.BUFFER, value), (chunks.TRUTH, state[0].new_tensor(pair))):
            result[field] = chunks.write(state[field], index,
                member.expand(state[0].shape[0], *member.shape), gate)
        state = chunks.read(tuple(result), index, gate)
    else:
        if when is not None:
            raise ValueError('a new thought idea takes its .when from model time')
        _, available = chunks.allocate_chunk(state, gate)
        if int(available[row]) < 0:
            raise OverflowError('thought focus cannot exceed the constructed STM capacity')
        when = _next_when(stm, row)
        if int(chunks.locate(state, when, band=chunks.WHEN)[row]) >= 0:
            raise ValueError('thought .when address is already occupied; use existing=True to update')
        state, index = chunks.deposit(state, value.expand(state[0].shape[0], -1), gate,
            where=(-1.,) * WHERE_WIRES, when=when, truth=pair, stack=pin)
        if int(index[row]) < 0:
            raise OverflowError('thought focus cannot exceed the constructed STM capacity')
    meaning = retain(stm, meaning, when, row, int(index[row]), pair=pair)
    stm.set_state(state)
    return meaning


def retain(stm, meaning, when, row, index, *, pair):
    """Keep the semantic view at a deposited slot's existing time and address."""
    data = bindings(meaning)
    data['_chunk_when'] = tuple(when)
    meaning = with_slots(replace(meaning, bindings=data), open_slots(meaning), pair=pair)
    rows = list(stm._thought_contents)
    contents = list(rows[row])
    from ThoughtStrata import write
    contents[index] = (tuple(when), write(stm._thought_stores[row][index], meaning, key=when))
    rows[row] = tuple(contents)
    stm._thought_contents = tuple(rows)
    return meaning


def begin(memory, meaning, row):
    stm = getattr(memory, '_thought_stm', None)
    if stm is None:
        return None
    # The incoming request has selected these resident operands. Read them
    # before allocating its focus, which can replace the least-recently read.
    for reference in dict.fromkeys(meaning.role_refs):
        if reference is not None and reference[0] == 'chunk':
            read(stm, reference[1], row)
    return bindings(put(stm, meaning, row=row, pin=True))['_chunk_when']


def release(memory, when, row):
    stm = getattr(memory, '_thought_stm', None)
    if stm is None:
        return
    index = _index(stm, when, row)
    state = stm.state
    gate = torch.arange(len(index), device=index.device) == row
    position = state[chunks.STACK][row, index[row]]
    result = list(state)
    result[chunks.STACK] = torch.where(gate[:, None] & (state[chunks.STACK] > position),
                                      state[chunks.STACK] - 1, state[chunks.STACK])
    result[chunks.STACK] = chunks.write(result[chunks.STACK], index, torch.full_like(index, -1), gate)
    result[chunks.DEPTH] = (result[chunks.STACK] >= 0).sum(-1)
    stm.set_state(tuple(result))


def _write_serial(context, value):
    """Write the selected content into the episode's pinned focus."""
    stm, when, row = context.stm, context.focus_when, context.row
    if stm is None or when is None:
        raise ValueError('serial write requires an open addressed focus chunk')
    if bindings(value).get('_producing_operation') == 'infer' and not open_slots(value):
        alternatives = []
        matching = []
        for other_when, other in held(stm, row):
            if (other.role_refs == value.role_refs and torch.equal(other.role_mask, value.role_mask)
                    and bindings(other).get('_evidence_kind') != 'intuition'):
                alternatives.append((other_when, evidence_pair(other)))
                pair = tuple(max(a, b) for a, b in zip(evidence_pair(other), evidence_pair(value)))
                value = with_slots(value, (), pair=pair)
                if other_when != when:
                    matching.append(other_when)
        value = replace(value, bindings=dict(bindings(value), _alternative_pairs=tuple(alternatives)))
        for other_when in matching:
            put(stm, value, other_when, row, existing=True)
    return put(stm, value, when, row, existing=True)


def stage(memory, result, row):
    """Recall retains a premise beside the focus; inference updates the focus."""
    from types import SimpleNamespace
    from Queries import _freeze_boundary_value, THOUGHT_EXECUTORS
    from AccessibleMind import Subsystem
    descriptor = THOUGHT_EXECUTORS[result.semantic_id]
    if descriptor.write_target is not Subsystem.SERIAL:
        return result
    value = result.evidence.get('meaning')
    if value is None:
        return result
    state = memory.thought_state(b=row)
    context = SimpleNamespace(stm=memory._thought_stm, focus_when=state.focus_when, row=row)
    evidence = dict(result.evidence)
    data = bindings(value)
    data['_evidence_kind'] = result.evidence_kind
    value = _write_serial(context, replace(value, bindings=data))
    evidence['meaning'] = value
    evidence['support_true'], evidence['support_false'] = evidence_pair(value)
    if result.semantic_id == 'infer' and ('derivation' in evidence or 'intuition_derivation' in evidence):
        evidence['alternatives'] = bindings(value).get('_alternative_pairs', ())
    return replace(result, evidence=_freeze_boundary_value(evidence))
