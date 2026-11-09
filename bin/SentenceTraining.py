"""Sentence objective assembly, one backward, and read-only gradient receipts."""
import math
import time
import torch
from ObjectiveOwnership import registry_costs


def separate(model):
    return bool(getattr(model, 'separate_chooser', False))


def strength(model, like):
    jitter = float(getattr(model, 'reconstruction_jitter', 0.))
    if not math.isfinite(jitter) or not 0 <= jitter <= 1:
        raise ValueError('reconstructionJitter must be finite and in [0, 1]')
    return like.new_ones(()) if jitter == 0 else 1. - jitter * torch.rand((), device=like.device, dtype=like.dtype)


def current_costs(model, registry, *, sentence):
    readers = getattr(model, '_sentence_reader_costs', None) if sentence else None
    rows = getattr(model, '_sentence_reader_rows', None) if sentence else None
    result = registry_costs(registry,
        reader_rows=torch.zeros_like(registry.row_mask) if readers is not None else rows,
        expectation_rows=torch.zeros_like(registry.row_mask) if readers is not None else None,
        definition_rows=getattr(model, '_sentence_definition_rows', None) if sentence else None)
    if readers is not None:
        combine(result, readers)
    return result


def combine(destination, costs):
    for name, cost in costs.items():
        destination[name] = destination.get(name, 0) + cost
    return destination


def surrogate(costs, reconstruction_strength):
    """Scale reconstruction alone; each shared regularizer enters once."""
    terms = []
    penalty = False
    for name, cost in costs.items():
        if not torch.is_tensor(cost) or not cost.requires_grad:
            continue
        if name.startswith('penalty.'):
            # registry_costs exposes this same term once per owner for the
            # restricted backward. The ordinary graph has all those edges.
            if penalty:
                continue
            penalty = True
        terms.append(cost * reconstruction_strength if name == 'reconstruction' else cost)
    if not terms:
        raise RuntimeError('single sentence backward has no differentiable objective')
    return sum(terms)


def operator_groups(model, optimizer):
    available = {id(p) for group in optimizer.param_groups for p in group['params']}
    groups = {}
    language = getattr(model, 'languageSpace', None)
    tree = language._tree_layer(2) if language is not None else None
    families = (
        ('compose_binary_ops', getattr(tree, 'ops', ())),
        ('compose_unary_ops', getattr(tree, 'unary_ops', ())),
        ('generate_binary_ops', getattr(language, '_generate_binary_ops', ())),
        ('generate_unary_ops', getattr(language, '_generate_unary_ops', ())),
    )
    for family, operators in families:
        for index, wrapper in enumerate(operators):
            operator = getattr(wrapper, 'gl', wrapper)
            parameters = tuple(p for p in operator.parameters() if id(p) in available)
            if parameters:
                groups[f'{family}.{index}.{type(operator).__name__}'] = parameters
    for index, space in enumerate(getattr(model, 'conceptualSpaces', ())):
        allocator = getattr(space, '_concept_allocator', None)
        if allocator is not None:
            for number, matrix in enumerate(allocator.layer(0).definition_matrices()):
                group = tuple(p for p in matrix.parameters() if id(p) in available)
                if group:
                    groups[f'definitions.{index}.{number}'] = group
        for name in ('intuition', 'concept_predictor'):
            module = getattr(space, name, None)
            if module is not None:
                groups[f'{name}.{index}'] = tuple(p for p in module.parameters() if id(p) in available)
    owners = model.objective_parameter_groups(optimizer)
    for name in ('reconstruction', 'output', 'expectation', 'attention'):
        # Owner totals are useful beside the individual operator cosines.
        groups['owner.' + name] = owners[name]
    # These are the understanding's trainable coordinates, excluding all
    # band readers and policy parameters. Buffers (including placed concepts)
    # have no optimizer coordinate and therefore contribute no gradient.
    from Spaces import Codebook, Embedding
    understanding = [p for name, group in groups.items()
                     if name.startswith(('compose_', 'generate_', 'definitions.'))
                     for p in group]
    for module in model.modules() if isinstance(model, torch.nn.Module) else ():
        if isinstance(module, (Codebook, Embedding)):
            understanding.extend(module.parameters())
    reconstruction = set(owners['reconstruction'])
    groups['understanding'] = tuple(dict.fromkeys(
        p for p in understanding if p in reconstruction and id(p) in available))
    return groups


def _record_understanding(model, parameters):
    """Index the existing epoch vectors by durable name and sparse allocation coordinates."""
    if not parameters:
        return
    names = {p: (name, None) for name, p in model.named_parameters()}
    for i, space in enumerate(getattr(model, 'conceptualSpaces', ())):
        alloc = getattr(space, '_concept_allocator', None)
        if alloc is not None:
            for j, matrix in enumerate(alloc.layer(0).definition_matrices()):
                for name, p in matrix.named_parameters():
                    ids = matrix._parameter_edge_ids if name == 'values' else None
                    names[p] = (f'definitions.{i}.{j}.{name}', ids)
    by_name = model.__dict__.setdefault('_understanding_epoch', {})
    for p in parameters:
        name, ids = names[p]
        # A leaf's epoch vector is already summed by the diagnostic. Keep
        # each incarnation under its name, including leaves no longer live.
        by_name.setdefault(name, {})[p] = ids


def _understanding_norm(named, gradients):
    from GradientDiagnostics import _norm, _sum_gradients
    norms = []
    for versions in named.values():
        active = [(ids, gradients[p]) for p, ids in versions.items() if p in gradients]
        if not active:
            continue
        if len(active) == 1:
            norms.append(_norm(active[0][1]))
        elif active[0][0] is None:
            total = None
            for _, vector in active:
                total = _sum_gradients(total, vector)
            norms.append(_norm(total))
        else:
            # Growth, removal and recycling preserve only surviving edge IDs.
            # The union has one coordinate per edge seen during this epoch.
            vectors = torch.cat([v.to_dense() if v.is_sparse else v for _, v in active])
            ids = torch.tensor([i for coordinates, _ in active for i in coordinates],
                               device=vectors.device, dtype=torch.long)
            unique, positions = ids.unique(return_inverse=True)
            norms.append(_norm(vectors.new_zeros(len(unique)).index_add_(0, positions, vectors)))
    return math.hypot(*norms)


def begin_sentence(model):
    if getattr(model, '_measure_sentence_gradients', False):
        model._sentence_chooser_gradients = {}


def measure(model, costs, optimizer, *, pullback=None, reconstruction_strength=1.):
    """Read actual weighted directions without an update or .grad writes."""
    if not getattr(model, '_measure_sentence_gradients', False):
        return
    started = time.monotonic()
    from GradientDiagnostics import accumulate_objective_gradients, _sum_gradients
    groups = operator_groups(model, optimizer)
    model._gradient_operator_groups = groups
    weighted = dict(costs)
    if 'reconstruction' in weighted:
        weighted['reconstruction'] = weighted['reconstruction'] * reconstruction_strength
    gradients = {}
    accumulate_objective_gradients(gradients, weighted, groups,
        gradient_fn=None if pullback is None else pullback.gradients)
    epoch = model.__dict__.setdefault('_epoch_objective_gradients', {})
    _record_understanding(model, groups['understanding'])
    for objective, current in gradients.items():
        accumulated = epoch.setdefault(objective, {})
        for parameter, gradient in current.items():
            accumulated[parameter] = _sum_gradients(accumulated.get(parameter), gradient)
    current = model.__dict__.setdefault('_sentence_chooser_gradients', {})
    for parameter, gradient in gradients.get('chooser', {}).items():
        current[parameter] = _sum_gradients(current.get(parameter), gradient)
    model._gradient_diagnostic_seconds = getattr(model, '_gradient_diagnostic_seconds', 0.) + time.monotonic() - started


def finish_sentence(model):
    """One variance sample is the sum of a sentence's chooser cotangents."""
    if not getattr(model, '_measure_sentence_gradients', False):
        return
    started = time.monotonic()
    from GradientDiagnostics import _norm, _sum_gradients
    groups = getattr(model, '_gradient_operator_groups', {})
    stats = model.__dict__.setdefault('_chooser_gradient_moments', {})
    current = model.__dict__.pop('_sentence_chooser_gradients', {})
    for name, group in groups.items():
        record = stats.setdefault(name, dict(count=0, sum_square=0., total={}))
        record['count'] += 1
        for parameter in dict.fromkeys(group):
            gradient = current.get(parameter)
            if gradient is None:
                continue
            record['sum_square'] += _norm(gradient) ** 2
            record['total'][parameter] = _sum_gradients(record['total'].get(parameter), gradient)
    model._gradient_diagnostic_seconds = getattr(model, '_gradient_diagnostic_seconds', 0.) + time.monotonic() - started


def gradient_receipt(model, *, clear=False):
    from GradientDiagnostics import _norm, objective_agreement
    stats = getattr(model, '_chooser_gradient_moments', {})
    variance = {}
    for name, record in stats.items():
        count = record['count']
        mean_square = record['sum_square'] / max(count, 1)
        square_mean = sum(_norm(value) ** 2 for value in record['total'].values()) / max(count, 1) ** 2
        variance[name] = dict(samples=count, mean_square_norm=mean_square,
            mean_gradient_square_norm=square_mean, variance=max(0., mean_square-square_mean))
    groups = getattr(model, '_gradient_operator_groups', {})
    dissonance = objective_agreement({}, groups,
        accumulated=getattr(model, '_epoch_objective_gradients', {}))
    accumulated = getattr(model, '_epoch_objective_gradients', {})
    understanding = groups.get('understanding', ())
    named = getattr(model, '_understanding_epoch', {})
    norms = {name: _understanding_norm(named, accumulated.get(name, {}))
             for name in ('reconstruction', 'output')}
    # The existing collection also records hypothetical cross-owner vectors
    # for dissonance. The actual separate backward sends no A cotangent here.
    if separate(model):
        norms['output'] = 0.
    sent = dict(reconstruction_norm=norms['reconstruction'], output_norm=norms['output'],
        separate_chooser=separate(model), parameters=len(understanding),
        elements=sum(p.numel() for p in understanding),
        coordinates='codes, grammar operators and inverses, conceptual definitions',
        convention='norm of the weighted gradient vector summed across the epoch; actual ownership applied')
    # Record where the nonzero A direction landed. This distinguishes a
    # missing compose operator from a trained reader outside the understanding.
    output = accumulated.get('output', {})
    understanding_set = set(understanding)
    reconstruction_set = set(groups.get('owner.reconstruction', ()))
    output_set = set(groups.get('owner.output', ()))
    named_parameters = model.named_parameters() if isinstance(model, torch.nn.Module) else ()
    destinations = {}
    for name, parameter in named_parameters:
        if parameter not in output or (separate(model) and parameter not in output_set):
            continue
        norm = _norm(output[parameter])
        if norm:
            destinations[name] = dict(norm=norm, understanding=parameter in understanding_set,
                                      reconstruction_owned=parameter in reconstruction_set)
    sent['output_destinations'] = destinations
    from GradientProjection import receipt as projection_receipt
    from ActionLearning import receipt as step_receipt
    result = dict(realized_steps=step_receipt(model), projection=projection_receipt(model), dissonance=dissonance, chooser_variance=variance, understanding=sent,
        sample='one sentence, summing its trial and close gradients',
        cosine='weighted gradient vectors summed across the epoch in parameter coordinates')
    if clear:
        model._epoch_objective_gradients = {}
        model._understanding_epoch = {}
        model._chooser_gradient_moments = {}
        model._projection_epoch = {}
        model._realized_steps_epoch = {}
    return result
