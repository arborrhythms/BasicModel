"""The current sentence context that `attend` reads at each look."""
import torch
from torch import nn
from torch.nn import functional as F


class RequestStart(nn.Module):
    """The request's start: one learned value for each concept row, added to
    the for lane of what the context's map asks (``attend``); one on each of
    ``rows`` and nothing elsewhere at the start."""
    def __init__(self, size, rows=(), device=None):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(int(size), device=device))
        with torch.no_grad():
            for row in rows:
                self.value[int(row)] = 1.


def install(model):
    from util import TheXMLConfig
    from AttentionObjective import tolerate_heterogeneity
    model.het_tolerance = float(TheXMLConfig.get('architecture.hetTolerance', default=1.))
    tolerate_heterogeneity(torch.zeros(1, 2), model.het_tolerance)
    model.attention_context = None
    model.request_start = None
    owner = model._concept_owner() if hasattr(model, '_concept_owner') else None
    if owner is not None and hasattr(owner, 'similarity_codebook'):
        from Spaces import _concept_alloc_of
        words = _concept_alloc_of(owner).declared_classes.get('words')
        row = None if words is None else owner._concept_row_of(words)
        model.request_start = RequestStart(int(owner.nVectors), () if row is None else (row,),
                                           device=owner.similarity_codebook.W.device)
    if not model.attention_budget:
        return
    registry, cs = model.where_registry, model.conceptualSpace
    # The heat of the percepts' two codebooks by their where slices, and of
    # the concepts by the concept inventory (priming reads concept rows).
    sizes = (*(registry.slices[name][1] - registry.slices[name][0] for name in ('parts', 'wholes')),
             int(model._concept_owner()._priming_dim()))
    geometry = tuple((int(space.outputShape[0]), int(space.subspace.muxedSize))
                     for space in (model.perceptualSpace, model.wholeSpace))
    width = sum(sizes) + sum(n*d for n,d in geometry) + geometry[0][0]*geometry[0][1]
    dim = int(cs.stm.concept_dim)
    width += int(cs.stm.capacity)*(dim+28) + (2+len(model._concept_owner()._order_caps()))*dim
    model._attention_geometry = sizes, geometry
    with torch.random.fork_rng(devices=[]):
        projection = nn.Linear(width + 2, dim, bias=False, device='cpu')
    nn.init.zeros_(projection.weight)
    model.attention_context = projection.to(cs.stm._buffer)


def _fit(value, batch, count, width, like):
    if value is None:
        return like.new_zeros(batch, count, width)
    if value.ndim == 2:
        value = value[:, None]
    if value.ndim != 3 or value.shape[0] != batch or value.shape[1] > count:
        raise ValueError('attention field does not match its native batch/slot geometry')
    # Content carriers may contain a wider positional tail than a need.
    value = value[..., :width].to(like)
    return F.pad(value, (0, width - value.shape[-1], 0, count - value.shape[1]))


def _heat(space, batch, width, like):
    method = getattr(space, 'priming_weights', None)
    # A prefix view of the constructed slab neither resizes nor republishes it.
    value = method(batch=batch) if callable(method) else None
    if value is None:
        return like.new_ones(batch, width)
    if value.ndim != 2 or value.shape[1] > width:
        raise ValueError('attention priming does not match its native codebook')
    if value.shape[0] != batch:
        neutral = (value == 1).all()
        if torch.compiler.is_compiling():
            torch._assert_async(neutral, 'attention priming differs from the current batch')
        elif not bool(neutral):
            raise ValueError('attention priming differs from the current batch')
        return like.new_ones(batch, width)
    return F.pad(value.detach().clone().to(like), (0, width - value.shape[1]), value=1.)


def _needs(model, batch, width, like, opened=None):
    bank = getattr(model.languageSpace, '_reference_bank', None)
    lesson = getattr(model, '_attention_part_lesson', None)
    if opened is None:
        opened = getattr(model, '_attention_prompt_need', None)
    if opened is None and lesson is not None:
        opened = lesson.need
    if opened is None:
        opened = getattr(bank, 'query', None)
    gist = getattr(model, '_last_gist', None)
    from RowTime import compared
    return torch.cat([compared(model, _fit(value, batch, count, width, like)).flatten(1)
                      for value, count in ((opened, 1), (gist, 1))], -1)


def context(model, stm, field, surprise=None):
    """Summarize the understood prefix before the next attention decision."""
    from RowTime import encoder, how_far_back
    values = stm[0]
    B = len(values)
    registry = model.where_registry
    sizes, geometry = model._attention_geometry
    parts = None if field is None else field.parts
    wholes = None if field is None else field.wholes
    placements = None if field is None else field.placements()
    heat = torch.cat([_heat(space, B, width, values) for space, width in zip(
        (model.perceptualSpace, model.wholeSpace, model._concept_owner()), sizes)], -1)
    supplied = (parts, wholes)
    keys = []
    for space, shape, value in zip((model.perceptualSpace, model.wholeSpace), geometry, supplied):
        # A serial carrier can hold a whole sentence's unprocessed words.
        # Only the explicit native working field belongs to this read.
        key = _fit(value, B, *shape, values)
        if space.subspace.nWhen:
            key = torch.cat((key[..., :-4], how_far_back(model.present_code(), key[..., -4:])), -1)
        keys.append(key.flatten(1))
    cs = model.conceptualSpace
    buffer, reference_rows = stm[0], stm[4]
    K, D = int(cs.stm.capacity), int(cs.stm.concept_dim)
    occupied = buffer.ne(0).any(-1)
    from RowTime import compared
    contents = compared(model, _fit(buffer, B, K, D, values)) * occupied[..., None]
    shared = stm
    spatial = shared[11].to(values)
    temporal = how_far_back(model.present_code(), shared[12].to(values))
    symbolic_heat = heat[:, sum(sizes[:2]):]
    salience = symbolic_heat.gather(1, reference_rows.clamp(0, sizes[2] - 1))
    salience = torch.where(occupied & (reference_rows >= 0), salience, 0.)
    # A slot's interval of places and of times, each end as a code: the
    # places by the where encoder, the times by how far back each end lies.
    if placements is None:
        stamps = values.new_zeros(B, K, 18)
    else:
        held = occupied[..., None, None]
        places = registry.encoding.encode(placements['where'].to(values.device)).to(values)
        times = how_far_back(model.present_code(), encoder().encode(placements['when']).to(values))
        stamps = torch.cat((torch.where(held, places, 0.).flatten(-2),
                            torch.where(held, times, 0.).flatten(-2),
                            placements['lanes'].to(values)), -1)
    conceptual = torch.cat((contents, spatial, temporal,
                            occupied[..., None].to(values), salience[..., None], stamps), -1)
    prime = getattr(model._stm_reducer(), '_expectation_prime', None)
    orders = len(model._concept_owner()._order_caps())
    prime = (values.new_zeros(B, orders * D) if prime is None or not model.expectation_prime_enabled else
             torch.cat([prime.mean(order) for order in range(orders)], -1))
    native = _fit(parts, B, *geometry[0], values)
    residual = model.percept_expectation.attention_residual(native) * model.percept_expectation_gain
    if model.perceptualSpace.subspace.nWhen:
        residual = torch.cat((residual[..., :-4], torch.zeros_like(residual[..., -4:])), -1)
    inputs = torch.cat((heat, prime, residual.flatten(1), *keys, conceptual.flatten(1),
                        _needs(model, B, D, values)), -1)
    return project(model.attention_context, inputs, surprise)


def project(mapping, inputs, surprise=None):
    """Apply one map to shared sentence features and each look's surprise pair."""
    weights = mapping.weight
    prefix = F.linear(inputs.detach(), weights[:, :-2])
    if surprise is None:
        return prefix
    if surprise.ndim != 3 or surprise.shape[0] != inputs.shape[0] or surprise.shape[-1] != 2:
        raise ValueError('surprise requires one pair per stream and look')
    return prefix[:, None] + F.linear(surprise.detach().to(inputs), weights[:, -2:])
