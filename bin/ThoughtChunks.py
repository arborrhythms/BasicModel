"""Serial ideas addressed by their thought-step .when in constructed STM.

Sentence and thought ideas share the owner passed to attach. Thought time uses
four exact 16-bit digits above the percept unit band, disjoint from model time.
Each stream advances once per deposited idea; reads and updates keep its stamp.
"""
from dataclasses import replace
import torch

from Memory import STMChunks as chunks
from ThoughtReferences import bindings, evidence_pair, open_slots, with_slots


def _next_when(stm, row):
    tick = stm._thought_clock[row] + 1
    if tick >= 2**64:
        raise OverflowError('thought time exceeds four-coordinate capacity')
    return tuple(float(2 + ((tick >> shift) & 65535)) for shift in (0, 16, 32, 48))


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
    for when, meaning in (item for item in stm._thought_contents[row] if item is not None):
        index = int(chunks.locate(state, when, band=chunks.WHEN)[row])
        if index >= 0:
            result.append((when, with_slots(meaning, open_slots(meaning),
                pair=state[chunks.TRUTH][row, index].tolist())))
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
    return with_slots(item[1], open_slots(item[1]), pair=pair)


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
            raise ValueError('a new thought idea takes its .when from the thought clock')
        when = _next_when(stm, row)
        if int(chunks.locate(state, when, band=chunks.WHEN)[row]) >= 0:
            raise ValueError('thought .when address is already occupied; use existing=True to update')
        state, index = chunks.deposit(state, value.expand(state[0].shape[0], -1), gate,
            where=(-1.,) * 4, when=when, truth=pair, stack=pin)
        if int(index[row]) < 0:
            raise OverflowError('thought focus cannot exceed the constructed STM capacity')
    data = bindings(meaning)
    data['_chunk_when'] = tuple(when)
    meaning = with_slots(replace(meaning, bindings=data), open_slots(meaning), pair=pair)
    rows = list(stm._thought_contents)
    contents = list(rows[row])
    contents[int(index[row])] = (tuple(when), meaning.detached())
    rows[row] = tuple(contents)
    stm._thought_contents = tuple(rows)
    stm.set_state(state)
    if not existing:
        clocks = list(stm._thought_clock)
        clocks[row] += 1
        stm._thought_clock = tuple(clocks)
    return meaning


def begin(memory, meaning, row):
    stm = getattr(memory, '_thought_stm', None)
    if stm is None:
        return None
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
    frames = evidence.get('frames', ())
    if result.semantic_id in ('query', 'ask') and frames:
        previous = read(context.stm, context.focus_when, row)
        if (previous.role_refs != value.role_refs
                and not any(item.role_refs == previous.role_refs for other_when, item in held(context.stm, row)
                            if other_when != context.focus_when)):
            put(context.stm, previous, row=row)
        frame = frames[0]
        data['_rule_row'] = frame['occurrence']
        readings = frame.get('readings', ())
        requested = (bindings(result.request.constituents[0]).get('_query_relation')
                     if result.request.constituents else None)
        data['_rule_kind'] = (requested if requested in ('part', 'equal', 'implies') else
            'equal' if 'equality' in readings else 'implies' if 'conditional' in readings else 'part')
        value = replace(value, bindings=data)
        value = put(context.stm, value, row=row)
        _write_serial(context, value)
    else:
        value = _write_serial(context, replace(value, bindings=data))
    evidence['meaning'] = value
    evidence['support_true'], evidence['support_false'] = evidence_pair(value)
    if result.semantic_id == 'infer' and ('derivation' in evidence or 'intuition_derivation' in evidence):
        evidence['alternatives'] = bindings(value).get('_alternative_pairs', ())
    return replace(result, evidence=_freeze_boundary_value(evidence))
