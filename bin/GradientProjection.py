"""Two objective gradients, projected in each optimizer unit's metric."""
import torch
from GradientDiagnostics import _sum_gradients


def sides(costs, strength):
    """R owns reconstruction and lessons; A owns answers and choice rewards."""
    result = dict(R=[], A=[])
    penalty = False
    for name, value in costs.items():
        if not torch.is_tensor(value) or not value.requires_grad:
            continue
        if name.startswith('penalty.'):
            if penalty:
                continue
            penalty = True
        side = 'A' if (name in ('output', 'generate_lesson', 'chooser') or name.endswith('.A')) else 'R'
        result[side].append(value * strength if name == 'reconstruction' else value)
    return {key: sum(values) if values else None for key, values in result.items()}


def units(model):
    """Stable concept rows and durable definition edges, otherwise tensors."""
    names = {p: (name, 'tensor', None) for name, p in model.named_parameters()}
    for name, module in model.named_modules():
        book = getattr(module, 'similarity_codebook', None)
        if book is not None and isinstance(getattr(book, 'W', None), torch.nn.Parameter):
            names[book.W] = (name + '.similarity_codebook.W', 'row', None)
    for i, space in enumerate(model.conceptualSpaces):
        allocator = getattr(space, '_concept_allocator', None)
        if allocator is not None:
            matrix = allocator.layer(0)
            if isinstance(getattr(matrix, 'values', None), torch.nn.Parameter):
                names[matrix.values] = (f'definitions.{i}.values', 'edge', matrix._parameter_edge_ids)
    return names


def metric(optimizer, parameter, like, rows=None):
    """Read Adam's bias-corrected second moment without creating state."""
    group = next(g for g in optimizer.param_groups if any(p is parameter for p in g['params']))
    state = optimizer.state.get(parameter, {})
    step = state.get('step', 0)
    if 'betas' not in group or not step or 'exp_avg_sq' not in state:
        return torch.ones_like(like)
    variance = state['exp_avg_sq']
    if rows is not None:
        selected = torch.zeros_like(like)
        live = rows < variance.shape[0]
        selected[live] = variance[rows[live]].to(selected)
        variance = selected
    elif variance.shape != like.shape:
        selected = torch.zeros_like(like)
        selected[:variance.shape[0]] = variance.to(selected)
        variance = selected
    return ((variance.to(like) / (1 - group['betas'][1] ** float(step))).sqrt() + group['eps']).reciprocal()


def project(reconstruction, answer, weight, *, per_row=False):
    """Leave R intact and remove only A's opposing component."""
    shape = reconstruction.shape
    r = reconstruction.reshape(shape[0] if per_row else 1, -1)
    a = answer.reshape_as(r)
    w = weight.reshape_as(r)
    dot = (r * a * w).sum(-1)
    rr = (r.square() * w).sum(-1)
    aa = (a.square() * w).sum(-1)
    fraction = torch.where((dot < 0) & (rr > 0), dot / rr.clamp_min(torch.finfo(rr.dtype).tiny), 0.)
    removed = fraction[:, None] * r
    adjusted = a - removed
    cosine = torch.where(rr * aa > 0, dot / (rr * aa).sqrt().clamp_min(torch.finfo(rr.dtype).tiny), 0.)
    removed_norm = (removed.square() * w).sum(-1).sqrt()
    fraction_removed = torch.where(aa > 0, removed_norm / aa.sqrt().clamp_min(torch.finfo(aa.dtype).tiny), 0.)
    return adjusted.reshape(shape), torch.stack((cosine, dot.lt(0).to(dot), fraction_removed), -1)


def _aligned(gradient, rows, parameter):
    if gradient is None:
        return parameter.new_zeros((len(rows), *parameter.shape[1:]))
    if not gradient.is_sparse:
        return gradient[rows]
    gradient = gradient.coalesce()
    result = gradient.new_zeros((len(rows), *parameter.shape[1:]), layout=torch.strided)
    return result.index_add_(0, torch.searchsorted(rows, gradient.indices()[0]), gradient.values())


def backward(model, costs, optimizer, strength, *, scale=1.):
    """Two reverse traversals and one accumulated gradient for the existing step."""
    parameters = tuple(dict.fromkeys(p for group in optimizer.param_groups for p in group['params'] if p.requires_grad))
    objectives = sides(costs, strength)
    def gradients(value, retain):
        return ((None,) * len(parameters) if value is None else
                torch.autograd.grad(value, parameters, retain_graph=retain, allow_unused=True))
    r_gradients = gradients(objectives['R'], True)
    a_gradients = gradients(objectives['A'], True)
    metadata = units(model)
    receipt = model.__dict__.setdefault('_projection_epoch', {})
    writers = model.__dict__.setdefault('_projection_writers', {})
    for index, (p, r, a) in enumerate(zip(parameters, r_gradients, a_gradients)):
        if r is None and a is None:
            continue
        writers.setdefault(id(p), set()).update(name for name, gradient in (('R', r), ('A', a)) if gradient is not None)
        name, kind, ids = metadata.get(p, (f'parameter.{index}', 'tensor', None))
        sparse = [g.coalesce() for g in (r, a) if g is not None and g.is_sparse]
        rows = None
        if sparse:
            if any(g.sparse_dim() != 1 for g in sparse):
                raise RuntimeError('projection requires sparse row gradients')
            # Dense partners have support on the whole table; ordinary sparse
            # readers need only the union of the two touched row sets.
            rows = (torch.arange(p.shape[0], device=p.device) if any(g is not None and not g.is_sparse for g in (r, a))
                    else torch.unique(torch.cat([g.indices()[0] for g in sparse]), sorted=True))
            rv, av = _aligned(r, rows, p), _aligned(a, rows, p)
        else:
            rv = torch.zeros_like(a) if r is None else r
            av = torch.zeros_like(r) if a is None else a
        adjusted, stats = project(rv, av, metric(optimizer, p, rv, rows), per_row=kind != 'tensor')
        if a is not None and r is not None:
            if rows is not None:
                adjusted = torch.sparse_coo_tensor(rows[None], adjusted, p.shape, device=p.device).coalesce()
                if not a.is_sparse:
                    adjusted = adjusted.to_dense()
            a = adjusted
        combined = _sum_gradients(r, a)
        p.grad = _sum_gradients(p.grad, combined.detach() * scale)
        coordinates = (rows.tolist() if rows is not None else range(len(stats))) if kind != 'tensor' else (None,)
        for coordinate, values in zip(coordinates, stats.detach().cpu().tolist()):
            suffix = '' if coordinate is None else f'.{kind}.{coordinate if ids is None else ids[coordinate]}'
            record = receipt.setdefault(name + suffix, [0., 0., 0., 0])
            for i, value in enumerate(values):
                record[i] += value
            record[3] += 1


def receipt(model):
    return {name: dict(cosine_before=values[0]/values[3], fraction_projected=values[1]/values[3],
                      fraction_answer_removed=values[2]/values[3], steps=values[3])
            for name, values in getattr(model, '_projection_epoch', {}).items()}
