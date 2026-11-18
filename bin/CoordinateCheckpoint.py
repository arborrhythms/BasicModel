"""One-way import of centered location/time bands; empty bands stay empty."""
import torch
from collections import deque
from dataclasses import fields, is_dataclass, replace


def band(value):
    from Spaces import unsigned_phase
    return torch.where(value.ne(0).any(-1, keepdim=True), unsigned_phase(value), 0.)


def point(value, percept_width):
    if percept_width < 8 or value.shape[-1] < percept_width:
        return value
    start = percept_width - 8
    bands = value[..., start:percept_width].reshape(*value.shape[:-1], 2, 4)
    return torch.cat((value[..., :start], band(bands).flatten(-2), value[..., percept_width:]), -1)


def observations(value, percept_width, name=''):
    """Convert typed saved observations, never a learned weight or its moments."""
    if torch.is_tensor(value):
        if not value.is_floating_point() or not value.ndim:
            return value
        if name in ('where', 'when', 'symbol_where', 'symbol_when', 'percept_where',
                    'percept_when', '_live_where', '_live_when', 'chunk_where', 'chunk_when',
                    '_live_chunk_where', '_live_chunk_when') and value.shape[-1] == 4:
            return band(value)
        if name == 'percept' and value.shape[-1] == 8:
            return band(value.reshape(*value.shape[:-1], 2, 4)).flatten(-2)
        if name in ('point', 'value', 'values', 'before', 'after', 'slots', 'codes',
                    'buffer', '_live_buffer', 'roles', 'end_state', 'focus',
                    'case_value', 'verb_meaning', 'modifiers', 'basis', '_inter_context'):
            return point(value, percept_width)
        return value
    if isinstance(value, dict):
        # Allocator sparse layers store learned edge weights under `values`;
        # those are not located observations, regardless of their length.
        if 'values_requires_grad' in value and 'nInput' in value:
            return value
        result = {key: observations(item, percept_width, str(key)) for key,item in value.items()}
        if 'singleton_members' in value:
            result['singleton_members'] = {key: (member, token, observations(point_value, percept_width, 'value'))
                for key, (member, token, point_value) in value['singleton_members'].items()}
        # MTM events encode their typed fields as tuples rather than mappings.
        if 'candidate_counter' in value and 'entries' in value and 'version' in value:
            entries = []
            for original, entry in zip(value['entries'], result['entries']):
                timelines = []
                for timeline in original['timelines']:
                    events = timeline if value['version'] >= 12 else timeline[0]
                    converted = []
                    for event in events:
                        event = list(event)
                        for at in (4, 5):
                            event[at] = observations(event[at], percept_width, 'value')
                        if value['version'] >= 12:
                            for at in (7, 9, 10):
                                if at < len(event):
                                    event[at] = observations(event[at], percept_width, 'value')
                        else:
                            for at in (7, 8, 10, 11):
                                if at < len(event):
                                    event[at] = observations(event[at], percept_width, 'value')
                        converted.append(tuple(event))
                    timelines.append(tuple(converted) if value['version'] >= 12 else
                                     (tuple(converted), observations(timeline[1], percept_width, 'value')))
                entry['timelines'] = tuple(timelines)
                entries.append(entry)
            result['entries'] = tuple(entries)
        return result
    if isinstance(value, deque):
        return deque((observations(v, percept_width, name) for v in value), maxlen=value.maxlen)
    if isinstance(value, (list, tuple)):
        return type(value)(observations(v, percept_width, name) for v in value)
    if is_dataclass(value) and not isinstance(value,type):
        return replace(value, **{f.name:observations(getattr(value,f.name), percept_width, f.name)
                                 for f in fields(value) if f.init})
    return value


def migrate(checkpoint, percept_width):
    """The bundle marker governs observation sidecars; idea tables own their schema."""
    if checkpoint.get('coordinate_encoding') == 1:
        return checkpoint
    if checkpoint.get('coordinate_encoding') not in (None, 0):
        raise ValueError('unsupported location/time coordinate encoding')
    import warnings
    result = dict(checkpoint)
    if 'state_dict' in result:
        result['state_dict'] = {key: point(value, percept_width)
            if key.endswith(('.truths', '._pending_truths')) else value
            for key, value in result['state_dict'].items()}
    for key in ('vocab_extras', 'structural_extras', 'training_state'):
        if key in result:
            result[key] = observations(result[key], percept_width)
    result['coordinate_encoding'] = 1
    warnings.warn('checkpoint: converted saved location/time observation bands to [0, 1]; learned parameters are unchanged')
    return result
