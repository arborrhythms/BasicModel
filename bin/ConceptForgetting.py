"""Conservative capacity recovery without reusing a concept's identity."""
import warnings
from dataclasses import fields, is_dataclass

import torch


def warn_capacity(owner):
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(owner)
    layer = alloc.layer()
    capacity = getattr(owner, 'nVectors', None)
    if capacity is None:
        return False
    _, reserved, limit = owner._concept_capacity_window(0)
    full = (len(layer._tensor_row_keys) >= int(capacity)
            or (limit is not None and reserved >= limit))
    if full and not getattr(owner, '_concept_capacity_warned', False):
        warnings.warn(f'Concept inventory reached maximum preallocation ({owner.nVectors}); '
                      'a redundancy review is required at a closed document boundary', RuntimeWarning,
                      stacklevel=3)
    owner._concept_capacity_warned = full
    return full


def _integers(value):
    """Conservatively retain identities appearing in native owner metadata."""
    pending = [value]
    seen = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if type(item) is int:
            yield item
        elif torch.is_tensor(item) and item.dtype in (torch.int32, torch.int64):
            pending.extend(item.detach().flatten().tolist())
        elif is_dataclass(item) and not isinstance(item, type):
            pending.extend(getattr(item, field.name) for field in fields(item))
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list, set, frozenset)):
            pending.extend(item)


def protected_concepts(owner, store, retained=()):
    from Spaces import _concept_alloc_of
    alloc, layer = _concept_alloc_of(owner), _concept_alloc_of(owner).layer()
    protected = set(retained) | set(getattr(owner, '_frozen_concepts', ()))
    protected.update(_integers(alloc.word_forms))
    edges = alloc.parts.indices()
    protected.update(edges.tolist())
    protected.update(alloc.parts.refs[edges].flatten().tolist())
    protected.update(alloc.definitions.word_ids)
    protected.update(alloc.definitions.object_ids)
    protected.update(_integers(tuple(alloc.relate_idx)))
    for name in ('_recognized_words', '_words_concept_id', '_priming_bridge', '_frozen_named'):
        protected.update(_integers(getattr(owner, name, None)))
    for cid in alloc.placement:
        for _, ref in alloc.records(cid):
            if isinstance(ref, tuple) and ref[0] == 'sym':
                protected.add(int(ref[1]))
    if store is not None:
        protected.update(_integers(store._semantic_rows))
    interpretation = owner.__dict__.get('_interpret_operator')
    if interpretation is not None:
        protected.update(_integers(interpretation._pending))
    components = getattr(owner, 'components', None)
    if components is not None:
        vectors = [(3, value) for value in components.verbs.directions.values()]
        vectors += [(3, value) for value in components._innovations.values()]
        for lanes, vector in vectors:
            used = vector.detach().reshape(lanes, int(owner.nVectors)).abs().sum(0).nonzero().flatten()
            for row in used.tolist():
                cid = owner.concept_id_at_row(row)
                if cid is not None:
                    protected.add(cid)
    for matrix in layer.part_matrices():
        if matrix.values is None:
            continue
        for (row, column), position in matrix._index.items():
            if float(matrix.values[position]) != 0:
                source = column % (layer.nOutput + 1) if matrix._paired_sources else column
                cid = owner.concept_id_at_row(source)
                if cid is not None:
                    protected.add(cid)
    # Situation and borrowed identity ties are additional live references.
    protected.update(_integers(alloc.identity))
    if layer.where._nnz():
        for row in layer.where.coalesce().indices().flatten().tolist():
            cid = owner.concept_id_at_row(row)
            if cid is not None:
                protected.add(cid)
    return protected & set(alloc.placement)


def _definition_rows(layer):
    matrices = []
    for matrix in layer.definition_matrices():
        rows = {}
        values = [] if matrix.values is None else matrix.values.detach().tolist()
        for (row, column), position in matrix._index.items():
            rows.setdefault(row, []).append((column, values[position]))
        matrices.append({row: tuple(sorted(items)) for row, items in rows.items()})
    groups = {}
    for key, value in layer.feature_groups.items():
        groups.setdefault(key[0], []).append((repr(key[1:]), repr(value)))
    return matrices, {row: tuple(sorted(items)) for row, items in groups.items()}


def _signature(owner, cid, row, definitions):
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(owner)
    records = tuple(alloc.records(cid))
    # Empty meanings are ignorance, not a certificate of equivalence.
    if not records or alloc.order_of(cid) == 1:
        return None
    matrices = tuple(rows.get(row, ()) for rows in definitions[0])
    if not any(value for matrix in matrices for _, value in matrix):
        return None
    groups = definitions[1].get(row, ())
    return alloc.order_of(cid), repr(records), matrices, groups


def _magnitude(owner, row):
    codebook = owner.similarity_codebook
    meaning = getattr(codebook, 'mereology', None)
    if meaning is not None:
        code = meaning.derive([row])[0, meaning.percept_event_width:]
    else:
        code = codebook.getW()[row]
    return float(code.detach().norm())


def _unused_mint(alloc, layer, cid, row, definitions, sparse_rows):
    """No definition, reference history, posting or sparse placement to lose.

    This is independent of equivalence: two unused mints need not match.
    A zero-valued edge still counts as placement, not as an unused mint.
    """
    if not 0 < cid < alloc.next_id or alloc.records(cid) or row in sparse_rows:
        return False
    if (cid in alloc.raised or cid in alloc.singletons or cid in alloc.reference_orders
            or cid in alloc.testimony_seen or cid in alloc.testimony_current
            or cid in alloc.relate_idx.values()):
        return False
    if any(row in rows for rows in definitions[0]) or row in definitions[1]:
        return False
    return not any(bool(getattr(layer, name)[row]) for name in
        ('assigned', 'managed', 'witnessed', 'witness_strength', 'context_seen'))


def _forget_parameters(model, parameters):
    if model is None:
        return
    ids = {id(p) for p in parameters}
    seen = set()
    for name in ('_optimizer', '_sentence_optimizer'):
        optimizer = getattr(model, name, None)
        if optimizer is None or id(optimizer) in seen:
            continue
        seen.add(id(optimizer))
        for leaf in model._optimizer_leaves(optimizer):
            for group in leaf.param_groups:
                group['params'] = [p for p in group['params'] if id(p) not in ids]
            for parameter in parameters:
                leaf.state.pop(parameter, None)


def _clear_code_moments(model, parameter, row):
    """A reused seat must not inherit its previous occupant's gradient/moments."""
    if parameter.grad is not None:
        parameter.grad[row].zero_()
    if model is None:
        return
    for name in ('_optimizer', '_sentence_optimizer'):
        optimizer = getattr(model, name, None)
        if optimizer is not None:
            for leaf in model._optimizer_leaves(optimizer):
                for value in leaf.state.get(parameter, {}).values():
                    if torch.is_tensor(value) and value.shape == parameter.shape:
                        value[row].zero_()


def model_concept_roots(model):
    """Native handles held by retained thoughts and the returned understanding."""
    found = set(_integers(getattr(model, '_last_understanding', None)))
    memory = getattr(getattr(model, 'symbolSpace', None), 'what_memory', None)
    if memory is not None:
        for stream in range(memory.batch):
            for record in memory.thought_history(b=stream):
                found.update(_integers(record))
    return found


@torch.no_grad()
def reclaim_concepts(owner, store=None, *, retained=(), model=None):
    """Erase exact redundancies, unused mints or zero-relevance columns.

    A stronger representative survives every duplicate group. Physical seats
    can be used again, but the retired handles can never name those occupants.
    Call only after all streams close and the last backward has completed.
    """
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(owner)
    layer = alloc.layer()
    if store is None:
        store = owner._closed_clause_store()
    protected = protected_concepts(owner, store, retained)
    components = getattr(owner, 'components', None)
    dictionaries = [] if components is None else [components.verbs]
    individual_ids = set(alloc.individuals) | set(alloc.form_regions.values())
    individual_ids.update(int(cid) for dictionary in dictionaries for cid in dictionary.directions)
    groups, rows, magnitudes = {}, {}, {}
    definitions = _definition_rows(layer)
    # Unlike duplicate matching, unused-mint eligibility excludes every
    # structural edge, including zero coefficients and both endpoints.
    sparse_rows = set()
    for matrix in layer.part_matrices():
        for row, column in matrix._index:
            sparse_rows.add(row)
            sparse_rows.add(column % (layer.nOutput + 1) if matrix._paired_sources else column)
    for cid in sorted(alloc.placement):
        seats = owner._csw_rows_of(cid)
        if len(seats) != 1:
            continue
        row = seats[0]
        rows[cid] = row
        signature = None if cid in individual_ids else _signature(owner, cid, row, definitions)
        if signature is not None:
            groups.setdefault(signature, []).append(cid)
    remove = {cid for cid, row in rows.items() if cid not in protected | individual_ids
              and _unused_mint(alloc, layer, cid, row, definitions, sparse_rows)}
    for group in groups.values():
        if len(group) < 2:
            continue
        for cid in group:
            magnitudes[cid] = _magnitude(owner, rows[cid])
        keeper = max(group, key=lambda cid: (cid in protected, magnitudes[cid], -cid))
        remove.update(cid for cid in group if cid != keeper and cid not in protected)
    for dictionary in dictionaries:
        remove.update(cid for cid in dictionary.pruning_candidates() if cid in rows and cid not in protected)
    for cid in remove:
        magnitudes[cid] = _magnitude(owner, rows[cid])
    ordered = sorted(remove, key=lambda cid: (magnitudes[cid], cid))
    before = len(layer._tensor_row_keys)
    for cid in ordered:
        row = rows[cid]
        for dictionary in dictionaries:
            if str(cid) in dictionary.directions:
                parameters = (dictionary.directions[str(cid)], dictionary.relevance[str(cid)])
                del dictionary.directions[str(cid)], dictionary.relevance[str(cid)]
                dictionary._witnesses.pop(cid, None)
                _forget_parameters(model, parameters)
        for matrix in layer.definition_matrices():
            matrix.remove_edges([(r, c) for r, c in matrix._index if r == row])
        layer.feature_groups = {key: value for key, value in layer.feature_groups.items() if key[0] != row}
        layer.clear_context(row)
        for name in ('participation', 'provisional', 'assigned', 'managed', 'witnessed',
                     'witness_strength', 'refinement_best', 'refinement_stale', 'refinement_step', 'context_seen'):
            getattr(layer, name)[row] = (1 if name == 'participation' else
                float('inf') if name == 'refinement_best' else -1 if name == 'refinement_step' else 0)
        key = layer._tensor_row_keys.pop(row)
        del layer._tensor_rows[key]
        for base, next_row in list(layer._row_next.items()):
            if base <= row < base + next_row:
                layer._row_next[base] = row - base
        alloc.drop(cid)
        alloc.individuals.discard(cid)
        alloc.form_regions = {key: value for key, value in alloc.form_regions.items() if cid not in (key, value)}
        alloc.raised.discard(cid)
        alloc.singletons.discard(cid)
        alloc.reference_orders.pop(cid, None)
        alloc.testimony_seen.pop(cid, None)
        alloc.testimony_current.pop(cid, None)
        if cid < alloc.next_id:
            alloc.reclaimed_seats += 1
        _clear_code_moments(model, owner.similarity_codebook.W, row)
        owner.similarity_codebook.W[row].zero_()
    meaning = getattr(owner.similarity_codebook, 'mereology', None)
    if remove and meaning is not None:
        meaning.begin_forward()
    if remove and components is not None:
        components._chart = None
    owner._concept_capacity_warned = False if remove else getattr(owner, '_concept_capacity_warned', False)
    report = dict(before=before, after=len(layer._tensor_row_keys), reclaimed=before-len(layer._tensor_row_keys),
                  protected=len(protected), retired=tuple(ordered), next_id=alloc.next_id)
    owner._last_concept_forgetting = report
    from util import TheMessage
    TheMessage(f'[Concept forgetting] {report}')
    if not remove:
        warnings.warn('Concept inventory has no safely redundant unreferenced candidate; no seat was reclaimed',
                      RuntimeWarning, stacklevel=2)
    return report
