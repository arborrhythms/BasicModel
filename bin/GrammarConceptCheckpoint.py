"""Import declarations and place saved concepts in fixed, contiguous blocks.

Concept identities do not change. Row-indexed values and optimizer moments
follow the row permutation, once at checkpoint import, before live loading.
"""
from dataclasses import replace
import copy
import warnings

import torch


def row_plan(space, allocator, layer, additions):
    """A complete permutation, including unallocated rows and added seats."""
    from Spaces import ConceptRowCapacityError
    old_size, new_size = int(layer['nOutput']), int(space.nVectors)
    if new_size - old_size != len(additions):
        raise ValueError('grammar declarations differ from the constructed concept capacity')
    caps = allocator.get('construction', {}).get('order_caps')
    old_orders = allocator.get('reference_orders', allocator.get('placement', {}))
    def order_of(concept, row):
        if caps is None:
            if concept not in old_orders:
                raise ValueError('checkpoint concept has no recoverable order')
            return int(old_orders[concept])
        end = 0
        for order, capacity in enumerate(caps):
            end += capacity
            if row < end:
                return order
        raise ValueError('checkpoint concept lies outside its order blocks')

    occupied, seats = set(), {}
    rows = sorted(layer['tensor_rows'].items(), key=lambda pair: pair[1])
    def reserve(order, preferred=None):
        low, high = space.order_slice(order)
        row = preferred if preferred is not None and low <= preferred < high and preferred not in occupied else next(
            (row for row in range(low, high) if row not in occupied), None)
        if row is None:
            raise ConceptRowCapacityError(f'concept order {order} capacity exhausted importing checkpoint')
        occupied.add(row)
        return row

    # Preserve already valid addresses before relocating any out-of-block row.
    def concept_id(key):
        return int(key[1] if isinstance(key, tuple) else key)
    if len({concept_id(key) for key, _ in rows}) != len(rows):
        raise ValueError('checkpoint has several physical rows for one concept')
    orders = {key: order_of(concept_id(key), row) for key, row in rows}
    for key, row in rows:
        low, high = space.order_slice(orders[key])
        if low <= row < high:
            seats[key] = reserve(orders[key], row)
    for key, row in rows:
        if key not in seats:
            seats[key] = reserve(orders[key])
    added_rows = [reserve(order) for _, _, order, _ in additions]
    indices = [-1] * new_size
    for key, old in rows:
        indices[seats[key]] = old
    used = set(layer['tensor_rows'].values())
    # Unallocated weights are still learned parameters; retain them as well.
    remaining = iter(row for row in range(new_size) if row not in occupied)
    for old in range(old_size):
        if old in used:
            continue
        if old < new_size and old not in occupied:
            indices[old] = old
            occupied.add(old)
        else:
            row = next(row for row in remaining if row not in occupied)
            indices[row] = old
            occupied.add(row)
    return indices, seats, added_rows


def prepare(model, state, structural, vocab, training=None):
    from IndependentComponents import _pack_state, _unpack_state
    from PartsOfSpeech import PartsOfSpeech
    from GrammarCheckpoint import imported_name
    entries = (structural or {}).get('conceptual_spaces', {})
    readers = [(name, module) for name, module in model.named_modules(remove_duplicate=False)
               if isinstance(module, PartsOfSpeech)]
    if not entries:
        return {}, {}
    declarations = readers[0][1] if readers else None
    arguments = declarations.arguments if declarations is not None else ()
    operations = declarations.operations if declarations is not None else ()
    plans = []
    for index, entry in entries.items():
        allocator = entry.get('allocator')
        if not allocator:
            continue
        # The live concept ids were saved as 'symbols' before symbols had a
        # table of their own; that field is read here once, as 'concepts'.
        if 'symbols' in allocator and 'concepts' not in allocator:
            allocator['concepts'] = allocator.pop('symbols')
        space = model.conceptualSpaces[int(index)]
        if not allocator.get('layers'):
            # A shared codebook's other stages can have empty local allocators.
            # They still declare the same newly constructed physical blocks.
            allocator.setdefault('construction', {})['order_caps'] = list(space._order_caps())
            if 'placement' in allocator:
                allocator['concepts'] = dict.fromkeys(allocator.pop('placement'))
            allocator.pop('reference_orders', None)
            continue
        for registry in ('formal_arguments', 'operator_concepts'):
            previous = allocator.get(registry, {})
            renamed = {imported_name(name): identity for name, identity in previous.items()}
            if len(renamed) != len(previous):
                raise ValueError('checkpoint has conflicting renamed grammar identities')
            if renamed != previous:
                warnings.warn(f'checkpoint: renamed {registry}; retained their concept identities')
                allocator[registry] = renamed
        old_arguments = allocator.get('formal_arguments', {})
        old_operations = allocator.get('operator_concepts', {})
        if not set(old_arguments) <= set(arguments) or not set(old_operations) <= set(operations):
            removed = sorted((set(old_arguments) - set(arguments)) | (set(old_operations) - set(operations)))
            raise ValueError(f'checkpoint retains removed grammar declarations {removed}; '
                             'regenerate this model from its training configuration')
        additions = [('formal_arguments', name, 1, 'formal-argument') for name in arguments
                     if name not in old_arguments]
        additions += [('operator_concepts', name, 2, 'operator') for name in operations
                      if name not in old_operations]
        if not old_arguments and not old_operations:
            additions = []  # This stage does not own grammar declarations.
        layer = next(iter(allocator['layers'].values()))
        old_size, new_size = int(layer['nOutput']), int(space.nVectors)
        if additions and new_size - old_size != len(additions):
            raise ValueError('added grammar concepts differ from the constructed reserved capacity')
        indices, seats, added_rows = row_plan(space, allocator, layer, additions)
        plans.append((space, entry, allocator, layer, old_size, new_size, additions,
                      indices, seats, added_rows))
    drift, transforms = {}, {}
    grown, migrated_owners = {}, set()

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

    for space, entry, allocator, layer, old_size, new_size, additions, indices, seats, added_rows in plans:
        old_symbols = allocator.pop('placement', None)
        old_orders = allocator.pop('reference_orders', None)
        if old_symbols is not None:
            allocator['concepts'] = dict.fromkeys(old_symbols)
        allocator.setdefault('construction', {})['order_caps'] = list(space._order_caps())
        if old_symbols is not None or old_orders is not None:
            warnings.warn('checkpoint: concept orders now read from their fixed row blocks')
        layer['tensor_rows'] = {('concept', int(key[1] if isinstance(key, tuple) else key)): row
                                for key, row in seats.items()}
        if not additions and indices == list(range(old_size)):
            continue
        inverse = {old: new for new, old in enumerate(indices) if old >= 0}
        def remap(value, axis=0, *, fill=0):
            shape = list(value.shape); shape[axis] = new_size
            target = value.new_full(shape, fill)
            dst, src = zip(*inverse.items())
            target.index_copy_(axis, torch.tensor(src, device=value.device),
                               value.index_select(axis, torch.tensor(dst, device=value.device)))
            return target
        layer['row_next'] = {}
        for offset, (registry, name, order, kind) in enumerate(additions):
            concept = allocator['next_id']; allocator['next_id'] += 1
            allocator[registry][name] = concept
            allocator['concepts'][concept] = None
            allocator['symbol_types'][concept] = {'concept', 'class', kind}
            row = added_rows[offset]
            layer['tensor_rows']['concept', concept] = row
            layer['constituents'][concept] = []
        def columns(values):
            if layer.get('columns') == 'symbols':
                # A column is a symbol, which the concept rows do not move.
                return list(values)
            # An older checkpoint's columns are concept rows, which move
            # with the rows; they are read as things when it is restored.
            return [(new_size if col % (old_size+1) == old_size else inverse[col % (old_size+1)])
                    + (col // (old_size+1)) * (new_size+1) for col in values]
        layer['rows'] = [inverse[row] for row in layer['rows']]
        layer['cols'] = columns(layer['cols'])
        layer['nOutput'] = new_size
        if layer.get('columns') != 'symbols':
            layer['nInput'] = 2*(new_size+1)
        parts = layer['parts']
        for matrix in parts.get('matrices', {}).values():
            matrix['rows'] = [inverse[row] for row in matrix['rows']]
        if 'brought' in allocator:
            allocator['brought'] = {percept: inverse[row] for percept, row in allocator['brought'].items()}
        if 'order_zero_members' in allocator:
            allocator['order_zero_members'] = {(inverse[row], percept, negated): weight
                for (row, percept, negated), weight in allocator['order_zero_members'].items()}
        for key, value in tuple(parts['buffers'].items()):
            if value.ndim and value.shape[0] == old_size:
                parts['buffers'][key] = remap(value, fill={'refinement_best': float('inf'),
                                                         'refinement_step': -1}.get(key, 0))
        if id(space.similarity_codebook) in migrated_owners:
            continue
        migrated_owners.add(id(space.similarity_codebook))
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
        seen_contexts = set()
        for context in contexts:
            if id(context) in seen_contexts:
                continue
            seen_contexts.add(id(context))
            readout = context.get('percept_readout_references')
            if readout is not None:
                readout['references'] = {inverse[row]: refs
                                         for row, refs in readout['references'].items()}
        runtime = (training or {}).get('corpus_runtime')
        if runtime is not None:
            contexts += [dict(mtm=runtime['mtm'][index])
                         for index, candidate in enumerate(model.conceptualSpaces)
                         if candidate in shared_spaces]
            names = {name for name, module in model.named_modules() if module in shared_spaces}
            for name, surfaces in runtime.get('surfaces', {}).items():
                if name in names:
                    for key, value in surfaces.items():
                        if value.shape[1] == old_size:
                            surfaces[key] = remap(value, 1)
            stm_names = {name for name, module in model.named_modules()
                         if any(module is getattr(candidate, 'stm', None) for candidate in shared_spaces)}
            addresses = torch.tensor([inverse[row] for row in range(old_size)])
            for name in stm_names.intersection(runtime.get('stm', {})):
                fields = runtime['stm'][name]['fields']
                rows = fields['_live_concept_rows']
                valid = (rows >= 0) & (rows < old_size)
                fields['_live_concept_rows'] = torch.where(valid,
                    addresses.to(rows.device)[rows.clamp(0, old_size-1)], rows)
        seen_pools = set()
        for context in contexts:
            pool = context.get('mtm')
            if pool is None or id(pool) in seen_pools:
                continue
            seen_pools.add(id(pool))
            if int(pool['transient_ids'][0]) != old_size:
                raise ValueError('grammar checkpoint has a foreign transient inventory')
            pool['transient_ids'] = pool['transient_ids'] + (new_size-old_size)
            for record in pool['entries']:
                record['support'] = {inverse[row]: value for row, value in record.get('support', {}).items()}
                refinement = record.get('refinement', {})
                if 'contexts' in refinement:
                    refinement['contexts'] = {inverse[focal]: {inverse[row]: value for row, value in support.items()}
                                              for focal, support in refinement['contexts'].items()}
                if record.get('transient') is not None:
                    record['transient'] += new_size-old_size
        seen_blobs = set()
        for blob in placement_blobs:
            if blob is None or id(blob) in seen_blobs:
                continue
            seen_blobs.add(id(blob))
            for key, value in tuple(blob.items()):
                if torch.is_tensor(value) and value.ndim and value.shape[0] == old_size:
                    blob[key] = remap(value, fill=key == 'derived')
        modules = {id(space.similarity_codebook)}
        prefixes = [name+'.' for name, module in model.named_modules(remove_duplicate=False)
                    if id(module) in modules]
        for key, value in tuple(state.items()):
            if (key.startswith(tuple(prefixes)) and torch.is_tensor(value) and value.ndim
                    and value.shape[0] == old_size):
                expand(key, 0, indices)
                if key.endswith(('.trust.allocated', '.trust.known')):
                    state[key][added_rows] = True
                elif key.endswith('.vq.active_mask'):
                    # This is a compiled search extent, not concept occupancy.
                    # Reordering a prefix can leave holes; reveal the smallest
                    # prefix covering its old active rows and all native seats.
                    active = state[key].nonzero().flatten()
                    end = max(int(active[-1])+1 if len(active) else 1,
                              max(layer['tensor_rows'].values(), default=0)+1)
                    state[key] = torch.arange(new_size, device=value.device) < end
        verb_ids = {id(candidate.components.verbs) for candidate in shared_spaces}
        component_ids = {id(candidate.components) for candidate in shared_spaces}
        for prefix, module in model.named_modules(remove_duplicate=False):
            if id(module) in component_ids:
                key = prefix+'._extra_state'
                metadata = _unpack_state(state[key])
                for identity, value in metadata.get('innovations', {}).items():
                    if value.numel() == 3*old_size:
                        metadata['innovations'][identity] = remap(value.reshape(3, old_size), 1).flatten()
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
                    pending['direction'] = remap(value, 1).flatten()
                state[key] = _pack_state(metadata)
            for key in tuple(state):
                if key.startswith(prefix+'directions.'):
                    directions = [old + block*old_size if old >= 0 else -1
                                  for block in range(3) for old in indices]
                    expand(key, 0, directions)
        if model.conceptualSpace in shared_spaces:
            for key, value in tuple(state.items()):
                if key.endswith('.field_readout.weight') and value.shape[-1] == 2*old_size:
                    expand(key, 1, [2*old+lane if old >= 0 else -1
                                    for old in indices for lane in range(2)])
            key = 'percept_expectation.known'
            if key in state and len(state[key]) == old_size:
                expand(key, 0, indices)
        warnings.warn(f'checkpoint: placed concepts in fixed order blocks and added {len(additions)} grammar seats; preserved all saved concept identities')
    if declarations is None:
        return drift, transforms
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
                     concepts=copy.deepcopy(allocator['formal_arguments']),
                     uses={imported_name(name): value for name, value in saved['uses'].items()})
        state[key] = _pack_state(saved)
        if id(module) in seen:
            continue
        seen.add(id(module))
        module.concepts = dict(allocator['formal_arguments'])
        module.seats.by_name = {name: replace(seat, concept=allocator['operator_concepts'][name])
                                for name, seat in module.seats.by_name.items()}
        module.seats.by_concept = {seat.concept: seat for seat in module.seats.by_name.values()}
    return drift, transforms
