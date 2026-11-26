"""MTM's word nominations, addressed by rows of the table of forms.

The caller reserves symbol names before entering a sentence. Reading chooses
among those names on the device; no spelling, allocator, or property code is
an identity key. A percept can therefore change its visible cover without
changing the word it names. Installing a nominated symbol's perceptual and
semantic rows belongs to the caller, using the returned arrivals.
"""
import torch
from torch import nn


class WordConceptPool(nn.Module):
    """One entry per stored form, and a finite supply of reserved symbol names.

    ``read`` is called once per presented reading, before exploration forks.
    A row of its input is one stream's reading: repetitions within it count
    once; distinct streams count separately, in stream order. Unknown and
    hidden forms are -1. Promotion of a perceptual form is independent of
    nomination of its word concept.
    """

    def __init__(self, form_capacity, symbols, *, threshold=3):
        super().__init__()
        if form_capacity < 1 or threshold < 1:
            raise ValueError('word pool capacities and threshold must be positive')
        if symbols.ndim != 1 or symbols.dtype != torch.long:
            raise ValueError('reserved word symbols must be a vector of integer names')
        if bool((symbols <= 0).any()) or len(symbols.unique()) != len(symbols):
            raise ValueError('reserved word symbols must be distinct and positive')
        self.threshold = int(threshold)
        self.register_buffer('symbols', symbols.detach().clone())
        self.register_buffer('words', symbols.new_full((form_capacity,), -1))
        self.register_buffer('counts', symbols.new_zeros(form_capacity))
        self.register_buffer('used', symbols.new_zeros(()))

    @torch.no_grad()
    def read(self, forms, *, learn=True, available=None):
        """Return word names and new (form row, symbol) pairs at fixed shape.

        Names become visible at the occurrence that reaches the pool's count,
        including later occurrences in the same reading. Earlier streams do
        not acquire a word retroactively. Arrivals follow reading order, with
        (-1, -1) in all other positions; exhaustion leaves a form pending.
        A held-out read neither counts forms nor consumes symbol names.
        """
        from Meaning import require_sentence_tensor
        require_sentence_tensor(((forms >= -1) & (forms < len(self.words))).all(),
                                'word pool form row outside the table', IndexError)
        rows = forms.flatten().long()
        live = rows >= 0
        at = rows.clamp_min(0)
        known = self.words[at].clone()
        known = torch.where(live, known, -1)
        arrivals = rows.new_full((*forms.shape, 2), -1)
        if not learn or not rows.numel():
            return known.reshape_as(forms), arrivals

        width = forms.shape[1]
        # Stable order puts a form's occurrences together, retaining stream
        # and word order. Segment sums avoid a square over batch * word count.
        order = rows.argsort(stable=True)
        ordered = rows[order]
        stream = order // width
        boundary = torch.cat((live.new_ones(1), ordered[1:] != ordered[:-1]))
        reading = torch.cat((live.new_ones(1), stream[1:] != stream[:-1]))
        first_sorted = (boundary | reading) & live[order] & (known[order] < 0)
        totals = first_sorted.long().cumsum(0)
        offsets = torch.where(boundary, totals - first_sorted.long(), 0).cummax(0).values
        first = torch.empty_like(first_sorted).scatter(0, order, first_sorted)
        # Counts are evaluated in presentation order, before publishing any
        # new mapping. This matters when a later stream nominates a word.
        ready_sorted = (live[order] & (known[order] < 0)
                        & (self.counts[at[order]] + totals - offsets >= self.threshold))
        previous = torch.cat((live.new_zeros(1), ready_sorted[:-1]))
        nominate_sorted = ready_sorted & (boundary | ~previous)
        ready = torch.empty_like(ready_sorted).scatter(0, order, ready_sorted)
        nominate = torch.empty_like(nominate_sorted).scatter(0, order, nominate_sorted)
        rank = nominate.long().cumsum(0) - 1
        slot = self.used + rank
        take = nominate & (slot < len(self.symbols))
        if available is not None:
            take &= rank < available
        # The sentinel also handles a pool constructed with no spare symbols.
        supply = torch.cat((self.symbols, self.symbols.new_full((1,), -1)))
        names = supply[slot.clamp(0, len(self.symbols))]
        names = torch.where(take, names, -1)
        mapping = self.words.clone().scatter_reduce(0, at, names, 'amax')
        result = torch.where(ready, mapping[at], known)
        self.words.copy_(mapping)
        self.counts.index_add_(0, at, first.long())
        self.used.add_(take.sum())
        arrivals = torch.stack((torch.where(take, rows, -1), names), -1)
        return result.reshape_as(forms), arrivals.reshape(*forms.shape, 2)


def prepare(model):
    """Reserve MTM's names and attach existing definitions at the host boundary.

    This visits existing definitions only, never the incoming words. Inserting
    an already defined form does not count it as a perceptual reading.
    """
    from WordIdentity import bytes_of
    ps, owner = model.perceptualSpace, model._concept_owner()
    table = getattr(ps, 'form_links', None)
    if table is None or not model.word_brackets:
        return
    alloc = owner._concept_allocator
    pool = getattr(owner.mtm, 'word_pool', None)
    if pool is None:
        start, end = owner.order_slice(0)
        matrix = alloc.layer()
        capacity = sum(row not in matrix._tensor_row_keys for row in range(start, end))
        if getattr(owner, '_online_learning_frozen', False):
            capacity = 0
        names = alloc.reserve_word_symbols(capacity)
        pool = owner.mtm.word_pool = WordConceptPool(table.capacity,
            table.left.new_tensor(names, dtype=torch.long), threshold=owner.mtm.mint_count)
        # Migrate a legacy checkpoint's pending readings once, preserving the
        # naming reading. New word-major readings use only the pool's counts.
        pending = []
        for key, readings in owner.mtm.sightings.items():
            if len(key) != 2 or key[0] != 'form' or not isinstance(key[1], (str, bytes)):
                continue
            raw = bytes_of(key[1]) if isinstance(key[1], str) else key[1]
            if raw:
                pending.append((key, raw, len(readings)))
        if pending and not getattr(owner, '_online_learning_frozen', False):
            width = max(len(raw) for _, raw, _ in pending)
            data = table.left.new_tensor([list(raw) + [0] * (width-len(raw)) for _, raw, _ in pending])
            rows = table.insert(data, table.left.new_tensor([len(raw) for _, raw, _ in pending]))
            pool.counts.index_add_(0, rows.long(), pool.counts.new_tensor([count for _, _, count in pending]))
            for key, _, _ in pending:
                del owner.mtm.sightings[key]
    pool.to(device=table.left.device)
    known = set(owner.definitions.word_ids) | set(owner.interpret.objectless())
    stamp = (owner.interpret._description_revision, ps.percept_store.identity.revision,
             tuple(sorted(known)))
    if getattr(pool, '_definitions_stamp', None) == stamp:
        return
    raw, words = [], []
    for word, description in alloc.word_descriptions.items():
        if word not in known:
            continue
        for form in description['forms']:
            if not isinstance(form, (str, bytes)):
                continue
            value = bytes_of(form) if isinstance(form, str) else form
            if value:
                raw.append(value)
                words.append(word)
    mapping = torch.full_like(pool.words, -1)
    if raw:
        width = max(map(len, raw))
        data = table.left.new_tensor([list(value) + [0] * (width-len(value)) for value in raw])
        lengths = table.left.new_tensor(list(map(len, raw)))
        frozen = getattr(owner, '_online_learning_frozen', False)
        rows = table.read(data, lengths, admit=False) if frozen else table.insert(data, lengths)
        # A restored frozen model can have definitions predating its form
        # table. Its missing form remains unknown; inference admits nothing.
        valid = rows >= 0
        mapping.scatter_reduce_(0, rows.clamp_min(0),
            torch.where(valid, rows.new_tensor(words), -1), 'amax')
    pool.counts.masked_fill_((pool.words >= 0) & (mapping < 0), 0)
    pool.words.copy_(mapping)
    pool._definitions_stamp = stamp


def stage(model, fi):
    """Let the pool decide before the legacy host record installation.

    The nomination itself is fixed-shape device work. Host installation is
    still a sentence capture break until the concept/definition consumers
    read the pool's rows directly.
    """
    from WordSurfaces import bytes_at, trimmed_spans
    owner, ps = model._concept_owner(), model.perceptualSpace
    pool = getattr(owner.mtm, 'word_pool', None)
    if pool is None or 'form_rows' not in fi:
        return False
    table = ps.form_links
    surface, units = fi['sentence_bytes'], fi['unit_spans']
    longest = min(fi['unit_longest'], surface.shape[1])
    trimmed = trimmed_spans(surface, units)
    lengths = trimmed[..., 1] - trimmed[..., 0]
    active = lengths > 0
    data, valid = bytes_at(surface, trimmed, active, longest)
    active &= ((data != 0) | ~valid).all(-1)
    forms = table.read(data.flatten(0, 1), torch.where(active, lengths, 0).flatten(),
                       admit=False).reshape_as(lengths)
    start, end = owner.order_slice(0)
    available = (~owner._concept_allocator.layer().assigned[start:end]).sum()
    words, arrivals = pool.read(forms, learn=not getattr(owner, '_online_learning_frozen', False),
                                available=available)
    # The old metadata consumers are ported separately. They no longer decide
    # whether the word exists: that decision and its name came from MTM above.
    fi['pool_word_ids'] = words
    fi['pool_form_rows'] = forms
    fi['pool_spans'] = trimmed
    fi['pool_word_host'] = words.tolist()
    host_spans = trimmed.tolist()
    for b, row in enumerate(arrivals.tolist()):
        for w, (form, word) in enumerate(row):
            if form < 0:
                continue
            spelling, size = table.spell(forms[b, w].reshape(1), longest=longest)
            raw = bytes(spelling[0, :int(size[0])].tolist())
            parts = [int(ps.byte_store_rows[byte]) for byte in raw]
            wholes = model.wholeSpace.property_rows_for_bytes(raw)
            admitted = owner.interpret.lookup_word(parts, wholes, form=raw,
                word_reading=True, reading=owner.reading(b), pool_word=word)
            if admitted is None:
                raise RuntimeError('MTM nominated a word without a free concept row')
    # Text labels are compatibility metadata of already named concepts.
    # Unnamed units have no label here; reporting and reset can still spell
    # the input at their boundary. Attention never spells an input to find it.
    descriptions = owner._concept_allocator.word_descriptions
    fi['pool_by_extent'] = {(b, lo, hi): word
        for b, (row, extents) in enumerate(zip(fi['pool_word_host'], host_spans))
        for word, (lo, hi) in zip(row, extents) if hi > lo}
    fi['pool_word_texts'] = [[next(iter(descriptions.get(word, {}).get('forms', ())), '')
                             for word in row] for row in fi['pool_word_host']]
    fi['pool_word_intervals'] = [[tuple(span) for span in row if span[1] > span[0]]
                                  for row in units.tolist()]
    return True
