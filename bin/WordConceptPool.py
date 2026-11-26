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
        self.register_buffer('free_rows', symbols.new_empty(0), persistent=False)
        self.register_buffer('form_seats', symbols.new_full((form_capacity,), -1), persistent=False)
        self.register_buffer('seats_used', symbols.new_zeros(()), persistent=False)
        self.register_buffer('seat_forms', symbols.new_empty(0), persistent=False)
        self.register_buffer('initial_places', None, persistent=False)

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

    def read_seated(self, forms, *, learn):
        """Assign the next free order-zero seat alongside each nominated name."""
        words, arrivals = self.read(forms, learn=learn, available=len(self.free_rows) - self.seats_used)
        if learn:
            new_forms = arrivals[..., 0].flatten()
            live = new_forms >= 0
            rank = live.long().cumsum(0) - 1
            seats = torch.nn.functional.pad(self.free_rows, (0, 1), value=-1)
            assigned = torch.where(live, seats[(self.seats_used + rank).clamp(0, len(self.free_rows))], -1)
            self.seats_used.add_(live.sum())
            self.form_seats.scatter_reduce_(0, new_forms.clamp_min(0), assigned, 'amax')
            if self.seat_forms.numel():
                self.seat_forms.scatter_reduce_(0, assigned.clamp_min(0),
                    torch.where(live, new_forms, -1), 'amax')
        rows = torch.where(words >= 0, self.form_seats[forms.clamp_min(0)], -1)
        return words, rows, arrivals


def candidate_bytes(pool, table, seats, active, capacity, longest):
    """Spell nominated word seats from stored form links with one spare EOS byte."""
    from Meaning import require_sentence_tensor
    valid = active & (seats >= 0) & (seats < len(pool.seat_forms))
    forms = torch.nn.functional.pad(pool.seat_forms, (0, 1), value=-1)
    rows = forms[torch.where(valid, seats, len(pool.seat_forms))].flatten()
    width = min(longest, capacity)
    data, lengths = table.spell(rows, longest=width)
    require_sentence_tensor((lengths <= capacity).all(),
                            f'word surface exceeds wordSurfaceCapacity {capacity}')
    require_sentence_tensor((lengths <= width).all(), 'stored word exceeds its delivered unit bound')
    data = torch.nn.functional.pad(data, (0, capacity + 1 - width), value=-1)
    data = data.reshape(*seats.shape, capacity + 1)
    return data.clamp_min(0).to(torch.uint8), data >= 0


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
    frozen = getattr(owner, '_online_learning_frozen', False)
    matrix = alloc.layer()
    start, end = owner.order_slice(0)
    first = start + matrix._row_next.get(start, 0)
    free_rows = [row for row in range(first, end) if row not in matrix._tensor_row_keys]
    if pool is None:
        pool = owner.mtm.word_pool = WordConceptPool(table.capacity,
            table.left.new_empty(0, dtype=torch.long), threshold=owner.mtm.mint_count)
    pool.to(device=table.left.device)
    if not frozen and not len(pool.symbols):
        pool.symbols = table.left.new_tensor(alloc.reserve_word_symbols(len(free_rows)), dtype=torch.long)
        # Migrate a legacy checkpoint's pending readings once, preserving the
        # naming reading. New word-major readings use only the pool's counts.
        pending = []
        for key, readings in owner.mtm.sightings.items():
            if len(key) != 2 or key[0] != 'form' or not isinstance(key[1], (str, bytes)):
                continue
            raw = bytes_of(key[1]) if isinstance(key[1], str) else key[1]
            if raw:
                pending.append((key, raw, len(readings)))
        if pending:
            width = max(len(raw) for _, raw, _ in pending)
            data = table.left.new_tensor([list(raw) + [0] * (width-len(raw)) for _, raw, _ in pending])
            rows = table.insert(data, table.left.new_tensor([len(raw) for _, raw, _ in pending]))
            pool.counts.index_add_(0, rows.long(), pool.counts.new_tensor([count for _, _, count in pending]))
            for key, _, _ in pending:
                del owner.mtm.sightings[key]
    # Concept placement uses a private CPU generator seeded by the symbol's
    # name. Reserve exactly those starting positions along with the names.
    # They acquire physical rows only when the device nominates a word.
    from ConceptPlacement import positions, reserved_seeds
    places = positions(owner)
    if (pool.initial_places is None or len(pool.initial_places) != len(pool.symbols)
            or pool.initial_places.dtype != places.dtype):
        pool.initial_places = reserved_seeds(pool.symbols.tolist(), places[0])
    pool.free_rows = table.left.new_tensor(free_rows, dtype=torch.long)
    pool.seats_used.zero_()
    known = set(owner.definitions.word_ids) | set(owner.interpret.objectless())
    stamp = (frozen, owner.interpret._description_revision, ps.percept_store.identity.revision,
             tuple(sorted(known)))
    if getattr(pool, '_definitions_stamp', None) == stamp:
        return
    pool.labels = {word: next(iter(value['forms']), '')
                   for word, value in alloc.word_descriptions.items() if word in known}
    raw, words, unique = [], [], []
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
                unique.append(len(description['forms']) == 1)
    pool.longest_spelling = max(map(len, raw), default=1)
    pool.stored_spellings = True
    mapping = torch.full_like(pool.words, -1)
    # The inverse is only for the spelling of a word seat. A word with
    # several forms identifies input but cannot spell one unique candidate.
    pool.seat_forms = pool.words.new_full((owner.nVectors,), -1)
    if raw:
        width = max(map(len, raw))
        data = table.left.new_tensor([list(value) + [0] * (width-len(value)) for value in raw])
        lengths = table.left.new_tensor(list(map(len, raw)))
        rows = table.read(data, lengths, admit=False) if frozen else table.insert(data, lengths)
        # A restored frozen model can have definitions predating its form
        # table. Its missing form remains unknown; inference admits nothing.
        valid = rows >= 0
        word_ids = rows.new_tensor(words)
        mapping.scatter_reduce_(0, rows.clamp_min(0), torch.where(valid, word_ids, -1), 'amax')
        seats = matrix.concept_rows(word_ids)
        if frozen:
            # Older checkpoints can carry a definition whose spelling was
            # never in FormLinks. Its inverse still reads that definition.
            pool.stored_spellings = bool((valid & (seats >= 0)).all())
        spelled = valid & (seats >= 0) & rows.new_tensor(unique, dtype=torch.bool)
        pool.seat_forms.scatter_reduce_(0, seats.clamp_min(0), torch.where(spelled, rows, -1), 'amax')
    pool.counts.masked_fill_((pool.words >= 0) & (mapping < 0), 0)
    pool.words.copy_(mapping)
    pool.form_seats.copy_(matrix.concept_rows(mapping))
    pool._definitions_stamp = stamp


def read_units(pool, table, surface, units, longest, chart, discard, *, learn):
    """Read native word names, physical seats and their property witnesses together."""
    from WordSurfaces import bytes_at, reading_property_signatures, trimmed_spans
    trimmed = trimmed_spans(surface, units)
    lengths = trimmed[..., 1] - trimmed[..., 0]
    data, valid = bytes_at(surface, trimmed, lengths > 0, longest)
    visible = (lengths > 0) & ((data != 0) | ~valid).all(-1)
    forms = table.read(data.flatten(0, 1), torch.where(visible, lengths, 0).flatten(),
                       admit=False).reshape_as(lengths)
    first = pool.used.clone()
    words, seats, arrivals = pool.read_seated(forms, learn=learn)
    live = arrivals[..., 0] >= 0
    ordinal = first + live.flatten().long().cumsum(0).reshape_as(forms) - 1
    # The final row supplies the absent value even for an empty frozen pool.
    reserved = torch.nn.functional.pad(pool.initial_places, (0, 0, 0, 1))
    initial = reserved[torch.where(live, ordinal, len(pool.symbols))]
    properties = reading_property_signatures(surface, trimmed, chart, discard, latin1_text=False)
    # Host consumers of new definitions need bytes only for arrivals. Keep
    # the source seat so they pick the corresponding private initial place.
    columns = torch.arange(words.numel(), device=words.device).reshape_as(words)
    records = torch.cat((arrivals, seats[..., None], columns[..., None], lengths[..., None],
                         properties[..., None], data.to(words.dtype)), -1)
    return words, seats, forms, trimmed, visible, properties, records, initial


def registration_witnesses(table, roots, data, lengths):
    """Character rows in encounter order, and all contained form addresses.

    The low bit of a contained address marks a complete character span.
    For long units collect one address per table row to bound live storage.
    """
    from BasicPercepts import utf8_boundaries
    width = data.shape[1]
    remaining = lengths[:, None] - torch.arange(width, device=data.device)
    cuts, stride = utf8_boundaries(data, remaining, with_stride=True)
    if not width:
        return roots.new_empty((len(data), 0, 2)), roots.new_empty((len(data), 0))
    dense = width * (width + 1) // 2 > table.capacity
    if dense:
        found = lengths.new_full((len(data), table.capacity + 1), -1)
    else:
        levels = []
    elements = roots.new_full(data.shape, -1)
    current = roots.new_full((len(roots), 1), -1)
    for size in reversed(range(1, width + 1)):
        span = width - size + 1
        # Install each root at its actual length. Its left and right links
        # cover every substring one byte shorter, including overlapping runs.
        first = torch.where(lengths == size, roots, current[:, 0])
        current = torch.cat((first[:, None], current[:, 1:]), 1)
        aligned = cuts[:, :span] & cuts[:, size:]
        encoded = torch.where(current >= 0, 2 * current + aligned, -1)
        if size <= 4:
            match = aligned & (stride[:, :span] == size) & (current >= 0)
            elements[:, :span] = torch.where(match, current, elements[:, :span])
        if dense:
            found.scatter_reduce_(1, torch.where(current >= 0, current, table.capacity),
                                  encoded, reduce='amax')
        else:
            levels.append(encoded)
        if size > 1:
            index = current.clamp_min(0)
            left = torch.where(current >= 0, table.left[index], -1)
            right = torch.where(current >= 0, table.right[index], -1)
            current = torch.maximum(torch.nn.functional.pad(left, (0, 1), value=-1),
                                    torch.nn.functional.pad(right, (1, 0), value=-1)).long()
    ordered, order = elements.sort(dim=-1, stable=True)
    first = torch.nn.functional.pad(ordered[:, 1:] != ordered[:, :-1], (1, 0), value=True)
    keep = torch.empty_like(first).scatter(1, order, first & (ordered >= 0))
    characters = torch.stack((torch.where(keep, elements, -1), torch.where(keep, stride, 0)), -1)
    return characters, found[:, :-1] if dense else torch.cat(levels, dim=1)


def basic_word_rows(words, names, rows):
    """Each nominated word's live basic percept, with -1 for absent words."""
    at = torch.searchsorted(names, words.contiguous())
    keys = torch.nn.functional.pad(names, (0, 1), value=-1)
    values = torch.nn.functional.pad(rows, (0, 1), value=-1)
    return torch.where((words >= 0) & (keys[at] == words), values[at], -1)


class WordLabels:
    """Display labels of the nominated words, materialized by a text reader."""
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels
        self._rows = None

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        if self._rows is None:
            self._rows = [[self.labels.get(word, '') for word in stream]
                          for stream in self.words.tolist()]
        return self._rows[index]


def stage(model, fi):
    """Let the pool decide before the legacy host record installation.

    The nomination itself is fixed-shape device work. Host installation is
    still a sentence capture break until the concept/definition consumers
    read the pool's rows directly.
    """
    owner, ps = model._concept_owner(), model.perceptualSpace
    pool = getattr(owner.mtm, 'word_pool', None)
    if pool is None or 'form_rows' not in fi:
        return False
    table = ps.form_links
    surface, units = fi['sentence_bytes'], fi['unit_spans']
    longest = min(fi['unit_longest'], surface.shape[1])
    from Spaces import _analysis_property_signature
    chart, discard = _analysis_property_signature(model.wholeSpace, device=surface.device)
    learn = not getattr(owner, '_online_learning_frozen', False)
    words, seats, forms, trimmed, visible, properties, records, initial = read_units(
        pool, table, surface, units, longest, chart, discard, learn=learn)
    fi.update(pool_visible=visible, pool_concept_rows=seats, pool_word_ids=words,
              pool_form_rows=forms, pool_spans=trimmed, pool_properties=properties)
    if learn:
        # Sparse catalog updates retain the order of the actual nominations.
        # Ordinary named occurrences and padding have no record to install.
        new = records[records[..., 0] >= 0]
        if len(new):
            width = new.shape[1] - 6
            labels = pool.labels.copy()
            places = initial.flatten(0, 1)
            property_count = min(int(model.wholeSpace.nVectors), 63)
            # Bound temporary storage for wide batches of long nominations.
            columns = min(width * (width + 1) // 2, table.capacity) + 3 * width + 6
            for group in new.split(max(1, (1 << 20) // columns)):
                elements, parts = registration_witnesses(table, group[:, 0], group[:, 6:], group[:, 4])
                publication = torch.cat((group, elements.flatten(1), parts), 1).tolist()
                for form, word, seat, source, length, bits, *payload in publication:
                    raw = bytes(payload[:length])
                    addresses = zip(payload[width:3 * width:2], payload[width + 1:3 * width:2], strict=True)
                    characters = [(raw[pos:pos + size], row)
                                  for pos, (row, size) in enumerate(addresses) if size]
                    wholes = tuple(i for i in range(property_count) if bits & (1 << i))
                    installed = owner.interpret.install_nomination(
                        word, seat, form, raw, wholes, places[source],
                        witnesses=(characters, payload[3 * width:]))
                    if installed is None:
                        raise RuntimeError('MTM nominated a word without a free concept row')
                    description = owner._concept_allocator.word_descriptions[word]
                    labels[word] = next(iter(description['forms']), '')
            pool.labels = labels
    fi['pool_word_texts'] = WordLabels(words, pool.labels)
    return True
