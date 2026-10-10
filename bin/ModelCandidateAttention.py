"""Native inputs to the candidate scorer, with the existing reader boundaries.

The input layout is fixed by the existing codebooks and working fields. It
contains their activation after spreading, their current content keys, the
eight conceptual positions, and three needs. Candidate support is supplied by the field; this module assembles reader inputs.
"""
import torch
from torch.nn import functional as F

from CandidateAttention import CandidateAttention


def install(model):
    from util import TheXMLConfig
    from AttentionObjective import validate_floor, tolerate_heterogeneity
    model.attention_floor = float(TheXMLConfig.get('architecture.attentionFloor', default=.1))
    model.het_tolerance = float(TheXMLConfig.get('architecture.hetTolerance', default=1.))
    validate_floor(model.attention_floor)
    tolerate_heterogeneity(torch.zeros(1, 2), model.het_tolerance)
    if not getattr(model, 'attention_budget', 0):
        model.candidate_attention = None
        return
    registry = model.where_registry
    cs = model.conceptualSpace
    # Codebook activation is a dense vector over the fixed native addresses;
    # content uses the existing PS/WS working geometry and conceptual slots.
    sizes = tuple(registry.slices[name][1] - registry.slices[name][0]
                  for name in ('parts', 'wholes', 'symbols'))
    geometry = tuple((int(space.outputShape[0]), int(space.subspace.muxedSize))
                     for space in (model.perceptualSpace, model.wholeSpace))
    width = 5 * sum(sizes) + sum(n * d for n, d in geometry)
    width += int(cs.stm.capacity) * (int(cs.stm.concept_dim) + 16)
    # One open slot, the complete three-role prediction, and one gist.
    width += 5 * int(cs.stm.concept_dim)
    # Adding an attention reader must not change the ambient initialization
    # of the pre-existing composition/control architecture.
    with torch.random.fork_rng(devices=[]):
        model.candidate_attention = CandidateAttention(width, int(cs.stm.concept_dim) + 8).to(
            device=cs.stm._buffer.device, dtype=cs.stm._buffer.dtype)
    model._candidate_geometry = (sizes, geometry)


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
    # A read inside a compiled loop must not resize or republish owner state.
    value = method() if callable(method) else None
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
    # The prediction is a need, never evidence added to a field.
    prediction = getattr(model.conceptualSpace, '_stm_predicted_idea', None)
    discourse = getattr(model.symbolSpace, 'expectation', None)
    pending = getattr(discourse, '_inter_last_meaning', ())
    if pending:
        images = [getattr(value, 'prediction', None) for value in pending]
        if len(pending) != batch and any(value is not None for value in images):
            raise ValueError('attention expectation history differs from the current batch')
        if len(pending) == batch:
            prediction = torch.stack([like.new_zeros(3, width) if value is None else
                                      value.roles.detach().clone().to(like) for value in images])
    gist = getattr(model, '_last_gist', None)
    return torch.cat([_fit(value, batch, count, width, like).flatten(1)
                      for value, count in ((opened, 1), (prediction, 3), (gist, 1))], -1)


def context(model, values, addresses, valid, *, when=None, parts=None, wholes=None,
            stm=None, need=None, placements=None):
    """Read a located activation slab [B,I,2], independent of its item count.

    Addresses are native codebook indices in the shared .where ladder, not
    document/row identities. Scatter preserves every addressed activation;
    neither the candidate count nor the STM width truncates this input.
    """
    B = values.shape[0]
    sizes, geometry = model._candidate_geometry
    registry = model.where_registry
    offset = registry.slices['parts'][0]
    count = sum(sizes)
    indices = addresses - offset
    eligible = valid & (indices >= 0) & (indices < count)
    source = torch.where(eligible[..., None], values, 0.)
    activation = values.new_zeros(B, count, 2).scatter_reduce(
        1, indices.clamp(0, count - 1)[..., None].expand(-1, -1, 2),
        source, reduce='amax', include_self=True)
    if when is None:
        from Occurrence import event_positions
        onset = event_positions(model.conceptualSpace, values.shape[:2], device=values.device)
        when = torch.stack((onset, onset + 1), -1)
    if when.shape != (*values.shape[:2], 2):
        raise ValueError('attention time coordinates must align with the native activation')
    time = when.to(values) / float(model.when_encoding.maxVal)
    observed = eligible & values.ne(0).any(-1)
    index = indices.clamp(0, count - 1)
    first = values.new_ones(B, count).scatter_reduce(1, index,
        torch.where(observed, time[..., 0], 1.), reduce='amin', include_self=True)
    last = values.new_zeros(B, count).scatter_reduce(1, index,
        torch.where(observed, time[..., 1], 0.), reduce='amax', include_self=True)
    time = torch.stack((torch.where(last > 0, first, 0.), last), -1)
    whole_heat, seen = [], set()
    for space in model.wholeSpaces:
        basis = space.subspace.what
        if id(basis) not in seen:
            seen.add(id(basis))
            capacity = int(getattr(basis, 'lexicon_capacity', basis.nVectors))
            whole_heat.append(_heat(space, B, capacity, values))
    heat = torch.cat((_heat(model.perceptualSpace, B, sizes[0], values),
                      *whole_heat, _heat(model._concept_owner(), B, sizes[2], values)), -1)
    # Neutral priming is a prior of one, not a positive observation. Retain
    # the two evidence lanes; heat affects the MLP's input, never their truth.
    spread = activation * heat[..., None]
    supplied = (parts, wholes)
    keys = []
    for shape, value in zip(geometry, supplied):
        # A serial carrier can hold a whole sentence's unprocessed words.
        # Only the explicit native working field belongs to this read.
        keys.append(_fit(value, B, *shape, values).flatten(1))
    cs = model.conceptualSpace
    if stm is None:
        memory = cs.stm
        if memory._buffer.shape[0] == B:
            buffer, depth = memory._buffer, memory._depth
            reference_rows = memory._concept_rows
        else:
            buffer = values.new_zeros(B, memory.capacity, memory.concept_dim)
            depth = torch.zeros(B, device=values.device, dtype=torch.long)
            reference_rows = torch.full((B, memory.capacity), -1, device=values.device, dtype=torch.long)
    else:
        buffer, depth, _, _, reference_rows, _ = stm
    K, D = int(cs.stm.capacity), int(cs.stm.concept_dim)
    occupied = torch.arange(K, device=values.device)[None] < depth[:, None]
    contents = _fit(buffer, B, K, D, values) * occupied[..., None]
    spatial = cs.where.to(values)[None].expand(B, -1, -1)
    temporal = (buffer[..., -4:] if int(cs.nWhen) else values.new_zeros(B, K, 4))
    symbolic_heat = heat[:, sum(sizes[:2]):]
    salience = symbolic_heat.gather(1, reference_rows.clamp(0, sizes[2] - 1))
    salience = torch.where(occupied & (reference_rows >= 0), salience, 0.)
    stamps = (values.new_zeros(B, K, 6) if placements is None else torch.cat((
        placements['where'] / registry.capacity,
        placements['when'] / float(model.when_encoding.maxVal), placements['lanes']), -1))
    conceptual = torch.cat((contents, spatial, temporal,
                            occupied[..., None].to(values), salience[..., None], stamps), -1)
    inputs = torch.cat((spread.flatten(1), heat, time.flatten(1), *keys, conceptual.flatten(1),
                        _needs(model, B, D, values, need)), -1).detach()
    return inputs
