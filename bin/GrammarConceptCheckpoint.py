"""Import added grammar declarations without renaming learned concepts.

Grammar reserves seats in addition to the configured concept inventory. An
added declaration occupies an added seat; every saved identity and row stays
where it was. This is a checkpoint transformation, not runtime inventory growth.
"""
from dataclasses import replace
import copy
import warnings

import torch


def prepare(model, state, structural, vocab):
    from IndependentComponents import _pack_state, _unpack_state
    from PartsOfSpeech import PartsOfSpeech
    entries = (structural or {}).get('conceptual_spaces', {})
    readers = [(name, module) for name, module in model.named_modules(remove_duplicate=False)
               if isinstance(module, PartsOfSpeech)]
    if not entries or not readers:
        return {}, {}
    declarations = readers[0][1]
    arguments, operations = declarations.arguments, declarations.operations
    plans = []
    for index, entry in entries.items():
        allocator = entry.get('allocator')
        if not allocator or not allocator.get('layers'):
            continue
        space = model.conceptualSpaces[int(index)]
        old_arguments = allocator.get('formal_arguments', {})
        old_operations = allocator.get('operator_concepts', {})
        if not old_arguments and not old_operations:
            continue  # This stage does not own grammar declarations.
        if not set(old_arguments) <= set(arguments) or not set(old_operations) <= set(operations):
            raise ValueError('removed argument concepts require an explicit identity migration')
        additions = [('formal_arguments', name, 1, 'formal-argument') for name in arguments
                     if name not in old_arguments]
        additions += [('operator_concepts', name, 2, 'operator') for name in operations
                      if name not in old_operations]
        layer = next(iter(allocator['layers'].values()))
        old_size, new_size = int(layer['nOutput']), int(space.nVectors)
        if additions and new_size - old_size != len(additions):
            raise ValueError('added grammar concepts differ from the constructed reserved capacity')
        plans.append((space, entry, allocator, layer, old_size, new_size, additions))
    drift, transforms = {}, {}
    grown = {}

    def expand(key, axis, indices):
        value = state[key]
        cache_key = (value.data_ptr(), tuple(value.shape), axis, tuple(indices))
        target = grown.get(cache_key)
        if target is None:
            shape = list(value.shape); shape[axis] = len(indices)
            target = value.new_zeros(shape)
            valid = [(dst, src) for dst, src in enumerate(indices) if src >= 0]
            if valid:
                dst, src = zip(*valid)
                target.index_copy_(axis, torch.tensor(dst, device=value.device),
                                   value.index_select(axis, torch.tensor(src, device=value.device)))
            grown[cache_key] = target
        state[key] = target
        drift[key] = dict(action='added grammar concept seats', axis=axis)
        transforms[key] = (key, axis, tuple(indices))

    for space, entry, allocator, layer, old_size, new_size, additions in plans:
        if not additions:
            continue
        # Append both identities and physical rows. In particular, sorted
        # formal-argument names must not shift the identities learned before.
        for offset, (registry, name, order, kind) in enumerate(additions):
            concept = allocator['next_id']; allocator['next_id'] += 1
            allocator[registry][name] = concept
            allocator['placement'][concept] = order
            allocator['reference_orders'][concept] = order
            allocator['symbol_types'][concept] = {'concept', 'class', kind}
            row = old_size + offset
            layer['tensor_rows']['shared', concept] = row
            layer['row_next'][row] = 1
            layer['constituents'][concept] = []
        def columns(values):
            return [(new_size if col % (old_size+1) == old_size else col % (old_size+1))
                    + (col // (old_size+1)) * (new_size+1) for col in values]
        layer['cols'] = columns(layer['cols'])
        layer['nOutput'], layer['nInput'] = new_size, 2*(new_size+1)
        parts = layer['parts']
        conjunction = parts['matrices']['conjunctive']
        conjunction['cols'] = columns(conjunction['cols'])
        conjunction['nInput'] = 2*(new_size+1)
        for key, value in tuple(parts['buffers'].items()):
            if value.ndim and value.shape[0] == old_size:
                parts['buffers'][key] = torch.cat((value, value.new_zeros(new_size-old_size)), 0)
        shared_spaces = [candidate for candidate in model.conceptualSpaces
                         if candidate.similarity_codebook is space.similarity_codebook]
        placement_blobs = [record.get('conceptual_structure', {}).get('placement')
                           for index, record in entries.items()
                           if model.conceptualSpaces[int(index)] in shared_spaces]
        contexts = [record.get('conceptual_structure', {}) for index, record in entries.items()
                    if model.conceptualSpaces[int(index)] in shared_spaces]
        contexts += [record for index, record in entries.items()
                     if model.conceptualSpaces[int(index)] in shared_spaces]
        if model.conceptualSpace in shared_spaces:
            placement_blobs.append((vocab or {}).get('conceptual_structure', {}).get('placement'))
            contexts.append((vocab or {}).get('conceptual_structure', {}))
        for context in contexts:
            pool = context.get('mtm')
            if pool is None or int(pool['transient_ids'][0]) == new_size:
                continue
            if int(pool['transient_ids'][0]) != old_size:
                raise ValueError('grammar checkpoint has a foreign transient inventory')
            pool['transient_ids'] = pool['transient_ids'] + (new_size-old_size)
            for record in pool['entries']:
                if record.get('transient') is not None:
                    record['transient'] += new_size-old_size
        for blob in placement_blobs:
            if blob is None or blob['placement'].shape[0] == new_size:
                continue
            for key, value in tuple(blob.items()):
                if torch.is_tensor(value):
                    tail = value.new_zeros(new_size-old_size, *value.shape[1:])
                    if key == 'derived': tail.fill_(True)
                    blob[key] = torch.cat((value, tail), 0)
        modules = {id(space.similarity_codebook),
                   *(id(candidate.concepts_from_percepts) for candidate in shared_spaces)}
        prefixes = [name+'.' for name, module in model.named_modules(remove_duplicate=False)
                    if id(module) in modules]
        for key, value in tuple(state.items()):
            if (key.startswith(tuple(prefixes)) and torch.is_tensor(value) and value.ndim
                    and value.shape[0] == old_size):
                expand(key, 0, (*range(old_size), *([-1]*(new_size-old_size))))
                if key.endswith(('.trust.allocated', '.trust.known', '.mereology.placement_derived')):
                    state[key][old_size:] = True
        verb_ids = {id(candidate.components.verbs) for candidate in shared_spaces}
        component_ids = {id(candidate.components) for candidate in shared_spaces}
        for prefix, module in model.named_modules(remove_duplicate=False):
            if id(module) in component_ids:
                key = prefix+'._extra_state'
                metadata = _unpack_state(state[key])
                for identity, value in metadata.get('innovations', {}).items():
                    if value.numel() == 3*old_size:
                        metadata['innovations'][identity] = torch.nn.functional.pad(
                            value.reshape(3,old_size),(0,new_size-old_size)).flatten()
                state[key] = _pack_state(metadata)
            if id(module) not in verb_ids:
                continue
            prefix += '.'
            key = prefix+'_extra_state'
            metadata = _unpack_state(state[key])
            if metadata['dimension'] == 3*old_size:
                metadata['dimension'] = 3*new_size
                for pending in metadata.get('pending', ()):
                    value = pending['direction'].reshape(3, old_size)
                    pending['direction'] = torch.nn.functional.pad(value, (0, new_size-old_size)).flatten()
                state[key] = _pack_state(metadata)
            for key in tuple(state):
                if key.startswith(prefix+'directions.'):
                    indices = [index for block in range(3)
                               for index in (*range(block*old_size, (block+1)*old_size),
                                             *([-1]*(new_size-old_size)))]
                    expand(key, 0, indices)
        if model.conceptualSpace in shared_spaces:
            for key, value in tuple(state.items()):
                if key.endswith('.field_readout.weight') and value.shape[-1] == 2*old_size:
                    expand(key, 1, (*range(2*old_size), *([-1]*(2*(new_size-old_size)))))
            key = 'percept_expectation.known'
            if key in state and len(state[key]) == old_size:
                expand(key, 0, (*range(old_size), *([-1]*(new_size-old_size))))
            key = 'candidate_attention.hidden.0.weight'
            if key in state and getattr(model, 'candidate_attention', None) is not None:
                count = sum(model._candidate_geometry[0])
                old_count = count - 2*(new_size-old_size)
                indices, offset = [], 0
                for width in (2, 1, 2, 2):
                    indices.extend(range(offset, offset+width*old_count))
                    indices.extend([-1]*(width*(count-old_count)))
                    offset += width*old_count
                indices.extend(range(offset, state[key].shape[1]))
                if len(indices) != model.candidate_attention.hidden[0].weight.shape[1]:
                    raise ValueError('grammar inventory differs from attention checkpoint geometry')
                expand(key, 1, indices)
        warnings.warn(f'checkpoint: added {len(additions)} grammar concept seats; preserved all saved concept identities')
    owner = declarations.owner
    allocator = next(allocator for space, _, allocator, *_ in plans if space is owner)
    seen = set()
    for name, module in readers:
        key = name+'._extra_state'
        if key not in state:
            raise ValueError('grammar checkpoint lacks its argument declarations')
        saved = _unpack_state(state[key])
        if saved['version'] != module.VERSION:
            raise ValueError('argument concept checkpoint requires replay of its producer')
        saved = dict(saved, arguments=arguments, operations=operations,
                     concepts=copy.deepcopy(allocator['formal_arguments']))
        state[key] = _pack_state(saved)
        if id(module) in seen:
            continue
        seen.add(id(module))
        module.concepts = dict(allocator['formal_arguments'])
        module.seats.by_name = {name: replace(seat, concept=allocator['operator_concepts'][name])
                                for name, seat in module.seats.by_name.items()}
        module.seats.by_concept = {seat.concept: seat for seat in module.seats.by_name.values()}
    return drift, transforms
