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

    def texts(self, surface):
        decoded = surface[:, 0].detach().long().cpu().tolist()
        spans = self.field.where.cpu().tolist()
        return tuple(b' '.join(bytes(row[lo:hi]) for lo, hi in extents if hi > lo).decode('utf8', 'replace')
                     for row, extents in zip(decoded, spans))


def realize(model, generation):
    """The same system-1 inverse realizes output and reconstructed grammar."""
    with model.languageSpace.generation_scope(), model._synthesis_guard():
        return model.reverse(model.conceptualSpace.subspace.carrier_like(), field=generation.field)


def generated(model, words, counts, truncated, bank, trace, *, when=None):
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
    # Generate already emitted a terminal. Its form owns identity; contextual
    # priming breaks identity ties but cannot rename that terminal. The leaf
    # still supplies its live amplitude, including the opposing pole.
    codes = bank.codes.gather(1, chosen[..., None].expand(B, W, D))
    from ConceptActivation import poles as activation_poles
    poles = activation_poles(words, codes, D if bank.meaning_start is None else bank.meaning_start)
    poles = poles * active[..., None]
    addresses = sorted(set(int(row) for row in selected.reshape(-1).tolist() if row >= 0),
                       key=lambda row: (owner._row_order(row), row))
    # Include the higher symbols' dependencies before descent; basic rows
    # have their own feature definitions and need no perception witness.
    store = owner._concept_allocator.layer(0)
    reached = set(addresses)
    while True:
        parents = {int(source) % (store.nOutput + 1) for target, source in zip(store._rows, store._cols)
                   if target in reached and int(source) % (store.nOutput + 1) < store.nOutput}
        if parents <= reached:
            break
        reached |= parents
    addresses = sorted(reached, key=lambda row: (owner._row_order(row), row))
    if not addresses:
        addresses = [-1]
    rows = selected.new_tensor(addresses)
    ids = selected.new_tensor([owner.concept_id_at_row(row) if row >= 0 else -1 for row in addresses])
    evidence = (rows[:, None, None] == selected[None])[..., None] * poles[None]
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
    evidence = owner.cs_reverse_presence(evidence, inventory_rows=rows)
    base = rows.new_tensor([row >= 0 and owner._row_order(row) == 0 for row in addresses], dtype=torch.bool)
    evidence = evidence * base[:, None, None, None]
    return Generated(words, counts, truncated, bank, trace, ConceptualField(ids, evidence, where, when, None))
