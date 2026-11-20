"""The generate walk's emitted terminals become an independently located field."""
from dataclasses import dataclass
import torch
from Understanding import ConceptualField


@dataclass(frozen=True)
class Generated:
    words: torch.Tensor
    counts: torch.Tensor
    truncated: torch.Tensor
    bank: object
    trace: tuple
    field: ConceptualField
    references: tuple = ()
    realization_field: ConceptualField | None = None
    extension: tuple = ()
    reconstruction_field: ConceptualField | None = None

    def texts(self, surface):
        decoded = surface[:, 0].detach().long().cpu().tolist()
        spans = (self.field if self.realization_field is None else self.realization_field).where.cpu().tolist()
        return tuple(b' '.join(bytes(row[lo:hi]) for lo, hi in extents if hi > lo).decode('utf8', 'replace')
                     for row, extents in zip(decoded, spans))


def realize(model, generation):
    """Read spelling and the generated meaning through the common inverse."""
    with model.languageSpace.generation_scope(), model._synthesis_guard():
        result = model.reverse(model.conceptualSpace.subspace.carrier_like(),
            field=generation.field if generation.realization_field is None else generation.realization_field)
        if generation.reconstruction_field is not None:
            scored = model.reverse(model.conceptualSpace.subspace.carrier_like(),
                                   field=generation.reconstruction_field)
            result._reconstruction_percepts = scored._reconstruction_percepts
        return result


def object_identity(owner, reference):
    """Follow an owned reference to its object; coordinates never name it."""
    seen = set()
    store = owner._closed_clause_store()
    while reference is not None and reference not in seen:
        seen.add(reference)
        if reference[0] == 'sym':
            return reference[1]
        if reference[0] == 'bp':
            return None  # A reading is not its class; the field supplies that.
        token = owner.mtm.token_of(reference)
        entry = owner.mtm.entries.get(token)
        if entry is not None and entry.symbol is not None and not entry.pending:
            return entry.symbol
        location = store.location(reference)
        if location is None:
            return None
        meaning = store.meaning_of(location)
        if int(meaning.role_mask.sum()) != 1:
            return None
        reference = meaning.role_refs[int(meaning.role_mask.nonzero()[0])]
    return None


def generated(model, words, counts, truncated, bank, trace, *, when=None, references=(), depths=None):
    from SentenceUnderstanding import readback_scores
    owner = model._concept_owner()
    B, W, D = words.shape
    scores = torch.stack([readback_scores(words[:, w], bank.codes, torch.ones_like(bank.weights),
        percept_width=bank.percept_width, meaning_start=bank.meaning_start) for w in range(W)], 1)
    scores = scores.masked_fill(~bank.terminal_valid[:, None], -torch.inf)
    best = bank.terminal_valid[:, None] & scores.eq(scores.amax(-1, keepdim=True))
    chosen = bank.weights[:, None].expand_as(scores).masked_fill(~best, -torch.inf).argmax(-1)
    selected = bank.rows.gather(1, chosen)
    active = (torch.arange(W, device=words.device)[None] < counts[:, None]) & bank.terminal_valid.any(-1)[:, None]
    selected = torch.where(active, selected, -1)
    from FieldProjection import stimulate
    orders = selected.new_zeros(B, W)
    for b, terminals in enumerate(references):
        for w, reference in enumerate(terminals[:W]):
            if reference is not None:
                orders[b,w] = max(0, owner._concept_source_order(reference))
    field = stimulate(owner, words, order=orders)
    scalar = torch.zeros_like(active)
    if hasattr(owner, 'basic_percepts'):
        from ScalarPercepts import bare
        scalar = bare(owner, words)
    # Only full percept membership names a bare reading. Partial support
    # outside every collection supplies no word; ties keep inventory order.
    lexical = selected.new_full(selected.shape, -1)
    strength = words.new_ones(B, W)
    said = [[None] * W for _ in range(B)]
    for b in range(B):
        for w in range(min(W, int(counts[b]))):
            reference = references[b][w] if references else None
            identity = object_identity(owner, reference) if reference is not None else None
            if reference is None and bool(scalar[b, w]):
                eligible = owner.interpret.recognized_objects(field, batch=b, position=w)
                identity = eligible[0] if eligible else None
            ordinary = reference is None and not bool(scalar[b, w])
            row = (int(selected[b, w]) if ordinary and bool(active[b, w]) else
                   None if identity is None else owner._concept_row_of(identity))
            if ordinary and row is not None:
                identity = owner.concept_id_at_row(row)
                # Identification presence belongs to form. The claim's two
                # meaning lanes still belong to the independently scored field.
                width = bank.meaning_start or bank.codes.shape[-1]
                form = bank.codes[b, chosen[b, w], :width]
                energy = form.square().sum()
                strength[b, w] = (words[b, w, :width] * form).sum() / energy.clamp_min(
                    torch.finfo(form.dtype).tiny)
            raw = None if row is None else owner.word_surface_for_row(row)
            active[b, w] = raw is not None
            selected[b, w] = -1 if raw is None else row
            if raw is None:
                continue
            word = (identity if identity in owner.definitions.word_ids or owner.interpret.no_object(identity)
                    else owner.interpret.reverse(identity))
            lexical[b, w] = word
            if not ordinary:
                said[b][w] = reference if reference is not None else ('sym', identity)
    where = selected.new_zeros(B, W, 2)
    for b in range(B):
        cursor = 0
        for w in range(min(W, int(counts[b]))):
            if not bool(active[b, w]):
                continue
            raw = owner.word_surface_for_row(int(selected[b, w]))
            if raw is None:
                raise ValueError('a generated terminal has no owned native surface')
            where[b, w] = where.new_tensor((cursor, cursor + len(raw)))
            cursor += len(raw) + 1
    if when is None:
        tick = model.present()
        when = selected.new_tensor((tick, tick + 1))[None, None]
    when = when.to(device=words.device).expand(B, W, 2)
    emitted = torch.arange(W, device=words.device)[None] < counts[:, None]
    field = stimulate(owner, words * emitted[...,None], order=orders, where=where, when=when)
    # A terminal has selected one spelling. Re-stimulating its word code and
    # expanding its types would add sibling words to that spelling. Read the
    # selected lexical identity's own percepts through the common inverse;
    # the stimulated semantic field above still measures what was said.
    presence = field.concept_ids[:, None, None].eq(lexical[None]) & active[None]
    realization = ConceptualField(field.concept_ids,
        torch.stack((presence * strength[None], torch.zeros_like(presence).to(words)), -1),
        where, when, None)
    from Extension import generated as extend_generated, stored_field
    # A spelling identifies a word; it cannot replace the generated point's
    # live meaning in the percept reconstruction cost. Both reads use the
    # same native inverse, with no further answer derivation or grammar walk.
    reconstruction = stored_field(owner, field)
    width = bank.meaning_start or bank.percept_width or D
    has_spelling = words[..., :width].detach().ne(0).any(-1) | scalar
    if references:
        has_spelling = has_spelling | torch.tensor(
            [[ref is not None for ref in row] for row in references], device=words.device)
    # A point without a form or an object reference is said through the
    # ordinary inverse of its extension, including its declared members.
    ids = torch.cat((realization.concept_ids, reconstruction.concept_ids)).unique(sorted=True)
    def align(value):
        evidence = value.evidence.new_zeros(len(ids), *value.evidence.shape[1:])
        evidence[torch.searchsorted(ids, value.concept_ids)] = value.evidence
        return evidence
    realization = ConceptualField(ids,
        torch.where(has_spelling[None, ..., None], align(realization), align(reconstruction)),
        where, when, None)
    if depths is None:
        raise ValueError('an emitted field requires the native generate walk root count')
    emitted_references = tuple(tuple(said[b][w] if said[b][w] is not None else
        references[b][w] if references else None for w in range(W)) for b in range(B))
    extension, field = extend_generated(model,trace[0],counts,depths,field,emitted_references)
    return Generated(words, counts, truncated, bank, trace, field, tuple(map(tuple, said)),
                     realization, extension, reconstruction)
