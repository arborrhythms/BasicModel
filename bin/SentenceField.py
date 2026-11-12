"""The live parallel field shared by sentence trials and the common decoder."""
import torch
from torch.nn import functional as F
from Understanding import ConceptualField, InputReconstruction


def inventory(owner, field):
    ids = field.concept_ids
    return torch.tensor([(-1 if int(cid) < 0 else owner._csw_row_of(int(cid)))
                         for cid in ids.reshape(-1).tolist()],
                        dtype=torch.long, device=ids.device).reshape(ids.shape)


def publish(model, field):
    owner, carrier = model._concept_owner(), model.conceptualSpace.subspace
    addresses = inventory(owner, field)
    carrier._concept_activations = field.evidence
    carrier._concept_ids = field.concept_ids
    carrier._concept_inventory_rows = addresses
    carrier._concept_where, carrier._concept_when = field.where, field.when
    carrier._concept_code_owner = next(i for i, space in enumerate(model.conceptualSpaces) if space is owner)
    carrier._concept_codes = owner._field_codes(owner.similarity_codebook.getW(), addresses).detach()


def descended(model, field):
    """Give the common decoder only the order-zero face of knowing."""
    owner = model._concept_owner()
    addresses = inventory(owner, field)
    evidence = owner.cs_reverse_presence(field.evidence, inventory_rows=addresses)
    zero = torch.tensor([int(row) >= 0 and owner._row_order(int(row)) == 0
                         for row in addresses.reshape(-1).tolist()],
                        device=addresses.device).reshape(addresses.shape)
    zero = zero[:, None, None, None] if zero.ndim == 1 else zero[..., None, None]
    return ConceptualField(field.concept_ids, evidence * zero, field.where, field.when, None)


def knowing(model):
    carrier = model.conceptualSpace.subspace
    return ConceptualField(carrier._concept_ids, carrier._concept_activations,
        carrier._concept_where, carrier._concept_when, None)


def open_field(model, sid, a0):
    owner = model._concept_owner()
    field = ConceptualField(owner._cs_field_concept_ids,
        F.pad(a0, (0, 0, 0, 0, 0, 0, 0, sum(owner._field_caps()) - len(a0))),
        owner._cs_field_where, owner._cs_field_when, None)
    model._sentence_open_fields[int(sid)] = field
    publish(model, field)
    return field


def sentence_generated(model, record, **walk_options):
    """Unfold this sentence's untransformed end slots through shared generate."""
    from Generative import walk_bank
    from Output import PreparedGeneration
    positions = torch.arange(record.end_slots.shape[1], device=record.root.device)[None]
    occupied = positions < record.end_depth[:, None]
    slots = torch.where(occupied[..., None], record.end_slots, 0.)
    slots = torch.where(((record.end_depth == 1)[:, None] & (positions == 0))[..., None],
                        record.root[:, None], slots)
    bank = walk_bank(model, record.end_slots, record.primed)
    prepared = PreparedGeneration((), slots, bank, None, torch.is_grad_enabled())
    return model.generate(slots, structure=record, occupied=occupied, infix_rows=None,
                          walk_options=walk_options, prepared=prepared,
                          when=record.open_field.when[:, :1])


def reconstruct_generation(model, record, target, active):
    """One grammar inverse; only the whole path chooses or credits a departure."""
    from OutputOccurrences import realize
    options = getattr(model, '_path_reconstruction_options', None)
    generation = sentence_generated(model, record, **dict(options or {}, return_candidates=True))
    reverse = realize(model, generation)
    model._sentence_generation = generation
    model._sentence_walk_residuals.append(generation.trace[3])
    model._last_decoder_comparison = None
    return generation, reverse


def selected_field(first, second, wins):
    """Publish each stream's kept generated field, with one union of its rows."""
    return select_fields((first, second), wins.long())


def select_fields(fields, winners):
    """Select streams by concept identity, including differently sized reviews."""
    picked = [fields[int(index)] for index in winners]
    ids = sorted({int(cid) for b, field in enumerate(picked) for cid in
        (field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:, b])})
    width = max(field.evidence.shape[2] for field in picked)
    evidence, where, when = [], [], []
    for b, field in enumerate(picked):
        column = field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:, b]
        lookup = {int(cid): i for i, cid in enumerate(column)}
        zero = field.evidence.new_zeros(1, field.evidence.shape[2], 2)
        values = torch.cat((field.evidence[:, b], zero), 0)
        indices = column.new_tensor([lookup.get(cid, len(column)) for cid in ids])
        evidence.append(F.pad(values[indices], (0, 0, 0, width - values.shape[1])))
        where.append(F.pad(field.where[b], (0, 0, 0, width - field.where.shape[1])))
        when.append(F.pad(field.when[b], (0, 0, 0, width - field.when.shape[1])))
    return ConceptualField(fields[0].concept_ids.new_tensor(ids), torch.stack(evidence, 1),
                           torch.stack(where), torch.stack(when), None)


def native_target(model, field):
    """Observed per-byte native events are targets, never decoder inputs."""
    ps = model.perceptualSpace
    from Omission import target as omission_target
    targets = omission_target(model)
    native = ps._forward_input if targets is None else targets['native']
    ids, spans = native['native_indices'], native['native_part_spans']
    # A promoted native part can span several bytes on the parallel path.
    # The target still has the same per-byte layout as the common decoder.
    width = max(1, int(spans[..., 1].max()) if spans.shape[1] else 0)
    byte_ids = ids.new_full((len(ids), width), -1)
    for batch, (parts, extents) in enumerate(zip(ids.tolist(), spans.tolist())):
        for part, (begin, end) in zip(parts, extents):
            if part < 0 or end <= begin:
                continue
            raw = ps.percept_store.bytes_for(part)
            if len(raw) != end - begin:
                raise ValueError('observed native part disagrees with its byte bracket')
            rows = [ps.percept_store.get_id(bytes([byte])) for byte in raw]
            if any(row is None for row in rows):
                raise ValueError('observed native part has an unadmitted byte')
            byte_ids[batch, begin:end] = ids.new_tensor(rows)
    positions = torch.arange(width, device=ids.device).expand(len(ids), -1)
    event = ps._radix_part_events(byte_ids.clamp_min(0), positions).detach()
    owned = ((positions[:, :, None] >= field.where[:, None, :, 0]) &
             (positions[:, :, None] < field.where[:, None, :, 1]) &
             (field.where[:, None, :, 1] > field.where[:, None, :, 0])).any(-1)
    return event * ((byte_ids >= 0) & owned)[..., None]


def word_percepts(reverse, field, width):
    """A reporting/forecast view of decoded native rows at field brackets."""
    event = reverse._reconstruction_percepts
    positions = torch.arange(event.shape[1], device=event.device)
    values = []
    for word in range(field.where.shape[1]):
        begin, end = field.where[:, word].unbind(-1)
        mask = (positions[None] >= begin[:, None]) & (positions[None] < end[:, None])
        values.append(torch.where(mask[..., None], event, 0.).amax(1))
    result = torch.stack(values, 1)
    return F.pad(result, (0, width - result.shape[-1])) if result.shape[-1] < width else result[..., :width]


def reconstruct(model, record, errors, active, *, previous=None):
    """Input and order-zero reconstructions share exactly one reverse chain."""
    field = record.open_field
    if previous is None:
        first = model.reverse(model.conceptualSpace.subspace.carrier_like(), field=field)
        target = native_target(model, field)
    else:
        first, target = previous
    model._sentence_reverse = (first, target)
    from Omission import visible_events
    visible, observed = visible_events(model, first._reconstruction_percepts, target)
    model._reverse_event_loss(visible, observed,
        registry=errors, name='reconstruction_reverse', weight=model.loss.reconstruction_scale)
    generation, second = reconstruct_generation(model, record, target, active)
    model._sentence_order_zero_field = generation.field
    publish(model, generation.field)
    model._reverse_event_loss(second._reconstruction_percepts, target,
        registry=errors, name='reconstruction_order_zero', weight=model.loss.reconstruction_scale)
    B, W, D = record.word_values.shape
    def rows(prefix):
        return sum(term['weight'] * term.get('multiplier', 1.) * errors._value(term)
                   for name, term in errors._terms.items() if name.startswith(prefix + '.'))
    input_cost, zero_cost = rows('reconstruction_reverse'), rows('reconstruction_order_zero')
    ideas = word_percepts(first, field, D)
    from Omission import target as omission_target
    targets = omission_target(model)
    raw = (model._staged_concepts_in if targets is None else targets['raw'])[:, 0]
    decoded = first.materialize()[:, 0]
    length = min(raw.shape[-1], decoded.shape[-1])
    positions = torch.arange(length, device=raw.device)
    scope = ((positions[None, None] >= field.where[..., :1]) &
             (positions[None, None] < field.where[..., 1:])).any(1)
    byte_error = ((raw[:, :length] != decoded[:, :length]) & scope).sum(-1) / scope.sum(-1).clamp_min(1)
    count = record.word_valid.sum(-1)
    model._last_decoder_byte_identity = byte_error.detach()
    errors.error('reconstruction.free_bytes', byte_error.detach(), 1., mask=active,
        weight=model.loss.reconstruction_scale, category='reconstruction', trained=False)
    missing = (decoded[:, :length].eq(0) & scope).sum(-1)
    errors.error('reconstruction.coverage', missing.to(input_cost).detach(), scope.sum(-1), mask=active,
        weight=model.loss.reconstruction_scale, category='reconstruction', trained=False)
    prior = record.percept_expected
    if prior is not None:
        from Expectation import percept_reconstruction
        correction, target_residual, rebuilt, squared = percept_reconstruction(ideas[..., :prior.shape[-1]],
            record.word_values[..., :prior.shape[-1]], prior)
        model._last_percept_reconstruction = (dict(
            correction=correction.detach(), target=target_residual.detach(),
            rebuilt=rebuilt.detach(), expected=prior.detach(),
            observed=record.word_values[..., :prior.shape[-1]].detach(),
            residual_loss=squared.detach(),
            full_loss=(rebuilt - record.word_values[..., :prior.shape[-1]].detach()).square().detach()),)
        residual = (squared * record.word_valid[..., None]).sum((1, 2)) / (record.word_valid.sum(-1) * prior.shape[-1]).clamp_min(1)
        errors.error('reconstruction.free_percept', residual, 1., mask=active,
            weight=model.loss.reconstruction_scale, category='reconstruction')
    truncated = missing > 0
    model._sentence_order_zero_reverse = second
    trace = generation.trace
    model._last_decoder_trace = (generation.words, generation.counts, generation.truncated,
        trace[3], trace[0], trace[4], trace[-2], trace[-1])
    sid = int(record.sentence)
    sentence_costs = ideas.new_zeros(B, record.depths.shape[1])
    sentence_costs[:, sid] = input_cost + zero_cost
    input_sentence_costs = torch.zeros_like(sentence_costs)
    input_sentence_costs[:, sid] = input_cost
    result = (ideas, ideas.new_zeros(B), input_cost + zero_cost, truncated, sentence_costs)
    model._input_reconstructions[sid] = InputReconstruction(ideas, first.materialize(), result[1],
        input_cost, truncated, input_sentence_costs,
        first._reconstruction_percepts)
    return result


def completed(model):
    values = tuple(model._input_reconstructions.values())
    if not values:
        raise RuntimeError('input reconstruction requires a completed parallel field')
    width = max(value.event.shape[-1] for value in values)
    event = torch.stack([F.pad(value.event, (0, width - value.event.shape[-1])) for value in values]).amax(0)
    percepts = torch.stack([F.pad(value.percepts, (0, 0, 0, width - value.percepts.shape[1])) for value in values]).sum(0)
    return InputReconstruction(torch.stack([v.ideas for v in values]).sum(0), event,
        torch.stack([v.idea_cost for v in values]).sum(0), torch.stack([v.byte_cost for v in values]).sum(0),
        torch.stack([v.truncated for v in values]).any(0), torch.stack([v.sentence_costs for v in values]).sum(0), percepts)


def selected(first, second, wins):
    """Keep each stream's input result from its selected sentence trial."""
    from dataclasses import fields
    def choose(a, b):
        if a is None and b is None:
            return None
        return torch.where(wins.reshape((-1,) + (1,) * (a.ndim - 1)), b, a)
    return InputReconstruction(**{item.name: choose(getattr(first, item.name), getattr(second, item.name))
                                 for item in fields(InputReconstruction)})
