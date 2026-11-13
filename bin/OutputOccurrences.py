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

    def texts(self, surface):
        decoded = surface[:, 0].detach().long().cpu().tolist()
        spans = self.field.where.cpu().tolist()
        return tuple(b' '.join(bytes(row[lo:hi]) for lo, hi in extents if hi > lo).decode('utf8', 'replace')
                     for row, extents in zip(decoded, spans))


def realize(model, generation):
    """The same system-1 inverse realizes output and reconstructed grammar."""
    with model.languageSpace.generation_scope(), model._synthesis_guard():
        return model.reverse(model.conceptualSpace.subspace.carrier_like(),
            field=generation.field if generation.realization_field is None else generation.realization_field)


def object_identity(owner, reference):
    """Follow an owned reference to its object; coordinates never name it."""
    seen = set()
    store = owner._closed_clause_store()
    while reference is not None and reference not in seen:
        seen.add(reference)
        if reference[0] == 'sym':
            return reference[1]
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


def generated(model, words, counts, truncated, bank, trace, *, when=None, references=()):
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
    field = stimulate(owner, words)
    scalar = torch.zeros_like(active)
    memberships = {}
    if hasattr(owner, 'basic_percepts'):
        from ScalarPercepts import bare, concepts
        scalar = bare(owner, words)
        memberships = {owner.concept_id_at_row(row): None for row in concepts(owner)}
    # Only actual box membership names a bare reading. Partial support
    # outside every box supplies no word; ties keep inventory order.
    ids = field.concept_ids.tolist()
    candidates = [cid for cid in ids if cid in memberships and owner.word_surface_for_row(owner._csw_row_of(cid))]
    native = words.clone()
    said = [[None] * W for _ in range(B)]
    for b in range(B):
        for w in range(min(W, int(counts[b]))):
            reference = references[b][w] if references else None
            identity = object_identity(owner, reference) if reference is not None else None
            if reference is None and bool(scalar[b, w]):
                eligible = [cid for cid in candidates if float(field.evidence[ids.index(cid), b, w, 0].detach()) >= 1.]
                identity = eligible[0] if eligible else None
            if reference is None and not bool(scalar[b, w]):
                continue
            row = None if identity is None else owner._csw_row_of(identity)
            raw = None if row is None else owner.word_surface_for_row(row)
            active[b, w] = raw is not None
            selected[b, w] = -1 if raw is None else row
            if raw is None:
                native[b, w] = 0.
                continue
            word = (identity if identity in owner.definitions.word_ids or owner.interpret.no_object(identity)
                    else owner.interpret.reverse(identity))
            # The learned unique definition supplies the interpretation
            # inverse. The semantic point remains unchanged for scoring.
            native[b, w] = owner.similarity_codebook.lookup_rows([owner._csw_row_of(word)])[0].to(words)
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
    field = stimulate(owner, words * emitted[...,None], where=where, when=when)
    realization = stimulate(owner, native * active[...,None], where=where, when=when)
    return Generated(words, counts, truncated, bank, trace, field, tuple(map(tuple, said)), realization)
