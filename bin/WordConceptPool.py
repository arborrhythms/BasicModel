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
        ps.percept_store.identity.registered_roots()
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
    pool.form_labels = {}
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
        pool.form_labels = {row: spelling.decode('latin1') for row, spelling in zip(rows.tolist(), raw)
                            if row >= 0}
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
    ps.percept_store.identity.registered_roots()


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
    properties = reading_property_signatures(surface, trimmed, chart, discard, latin1_text=False)
    # The form row owns its spelling, and the reserved ordinal owns its
    # initial geometry. Neither needs duplicating at every input occurrence.
    records = torch.stack((arrivals[..., 0], words, seats, ordinal, properties), -1)
    return words, seats, forms, trimmed, visible, properties, records


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


def form_holders(table, roots, wanted, longest):
    """Which stored roots strictly contain each wanted form, by link address."""
    matches = roots.new_zeros((len(wanted), len(roots)), dtype=torch.int32)
    level = roots[:, None]
    for _ in range(1, longest):
        # At a given depth the left child at each start and the last
        # right child enumerate every shorter run, including overlaps.
        level = torch.cat((table.left[level.clamp_min(0)], table.right[level[:, -1:].clamp_min(0)]), 1).long()
        # Row zero is a leaf. Its links, like those of an absent row, are -1.
        matched = (wanted[:, None, None] == level).int().amax(-1)
        matches = torch.maximum(matches, matched)
    return (matches > 0) & (wanted[:, None] >= 0) & (roots >= 0)


def registration_holders(table, known, longest, nominations, width, *, batch_bound):
    """Existing and simultaneously nominated roots, resolved in bounded slabs."""
    # A static bound also works when a captured nomination count is zero.
    group = max(1, (1 << 20) // max(1, longest * batch_bound))
    held = []
    for slab in known.split(group) if len(known) else ():
        links = form_holders(table, slab, nominations, longest)
        held.append(torch.where(links, slab, -1))
    links = form_holders(table, nominations, nominations, width)
    held.append(torch.where(links, nominations, -1))
    return torch.cat(held, 1)


def registration_parts(parts, roots, lengths, characters, known, runs):
    """The stored words and character-aligned runs held at each actual nomination."""
    encoded = parts.sort(dim=1).values
    forms = encoded // 2
    # The last occurrence of a form preserves a character-aligned witness
    # whenever it had one, even if it also occurred inside a UTF-8 character.
    following = torch.nn.functional.pad(forms, (0, 1), value=-1)[:, 1:]
    last = (encoded >= 0) & (forms != following)
    sorted_roots, order = roots.sort()
    earlier = basic_word_rows(forms, sorted_roots, order)
    page = torch.arange(len(roots), device=roots.device)[:, None]
    old = basic_word_rows(forms, known, known)
    inside = last & (forms != roots[:, None]) & ((old >= 0) | ((earlier >= 0) & (earlier < page)))

    names, ranks = runs.unbind()
    rank = basic_word_rows(forms, names, ranks)
    # A multi-character root becomes a run on its own admission. Taught
    # runs retain their original rank when they later acquire a word.
    first_size = torch.nn.functional.pad(characters[..., 1], (0, 1))[:, 0]
    several = torch.nn.functional.pad(lengths > first_size, (0, 1))
    added = (earlier >= 0) & (earlier <= page) & several[torch.where(earlier >= 0, earlier, len(roots))]
    next_rank = torch.nn.functional.pad(ranks, (0, 1), value=-1).amax() + 1
    rank = torch.where((rank < 0) & added, next_rank + earlier, rank)
    held = last & (encoded % 2 == 1) & (rank >= 0)
    by_rank = torch.where(held, rank, torch.iinfo(rank.dtype).max).argsort(dim=1)
    counts = torch.stack((inside.sum(1), held.sum(1)), 1)
    return counts, forms[inside], forms.gather(1, by_rank)[held.gather(1, by_rank)]


def nomination_records(table, known, longest, records, width, runs):
    """Read each new word's directly usable catalog witnesses in encounter order."""
    new = records[records[:, 0] >= 0]
    forms = new[:, 0].contiguous()
    seated = basic_word_rows(forms, known, known) >= 0
    roots = forms[~seated]
    headers = torch.cat((new, seated[:, None]), 1)
    data, lengths = table.spell(roots, longest=width)
    elements, parts = registration_witnesses(table, roots, data, lengths)
    counts, contained, added = registration_parts(parts, roots, lengths, elements, known, runs)
    holders = registration_holders(table, known, longest, roots, width, batch_bound=len(records))
    registered = torch.cat((lengths[:, None], counts, data, elements.flatten(1), holders), 1)
    return headers, registered, contained, added


def basic_word_rows(words, names, rows):
    """Each nominated word's live basic percept, with -1 for absent words."""
    at = torch.searchsorted(names, words.contiguous())
    keys = torch.nn.functional.pad(names, (0, 1), value=-1)
    values = torch.nn.functional.pad(rows, (0, 1), value=-1)
    return torch.where((words >= 0) & (keys[at] == words), values[at], -1)


def supported_parts(parts, seats, entries, capacity):
    """Test every observed part against the stored membership of its own word."""
    keys = seats[..., None] * capacity + parts
    at = torch.searchsorted(entries, keys.contiguous())
    stored = torch.nn.functional.pad(entries, (0, 1), value=-1)
    valid = (seats >= 0)[..., None] & (parts < capacity) & (stored[at] == keys)
    return ((parts < 0) | valid).all(-1)


def raw_part_ids(refs):
    raw = set()
    for ref in refs:
        if not isinstance(ref, tuple):
            try:
                raw.add(int(ref))
            except (ValueError, TypeError):
                continue
    return raw


def word_membership(owner, store, like):
    """The sorted (word seat, canonical native part) index of the live schema."""
    matrix = owner._concept_allocator.layer(0)
    basic = owner.basic_percepts
    capacity = store.capacity
    stamp = (getattr(matrix, '_record_revision', 0), id(matrix._constituents), len(matrix._tensor_row_keys),
             id(basic.words), len(basic.words), capacity, like.device)
    cached = vars(matrix).get('_word_membership')
    groups = store._canonical_group_cache
    if cached is None or cached[0] != stamp or cached[1] is not groups:
        entries = []
        for name in basic.words:
            seat = owner._concept_row_of(name)
            if seat is None:
                continue
            # Added structural relations have tagged endpoints. Only the
            # native records supply raw percept ids. Preserve concept_parts'
            # ordering before it hands those ids to the recurrent retile.
            refs = {ref for role, ref in matrix.iter_constituents(name) if role == 'part'}
            raw = raw_part_ids(sorted(refs, key=repr))
            canonical = set(store.canonical_parts(raw))
            entries.extend(seat * capacity + part for part in canonical if 0 <= part < capacity)
        cached = stamp, groups, like.new_tensor(sorted(entries), dtype=torch.long)
        matrix._word_membership = cached
    return cached[2]


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
    words, seats, forms, trimmed, visible, properties, records = read_units(
        pool, table, surface, units, longest, chart, discard, learn=learn)
    fi.update(pool_visible=visible, pool_concept_rows=seats, pool_word_ids=words,
              pool_form_rows=forms, pool_spans=trimmed, pool_properties=properties)
    if learn:
        # Sparse catalog updates retain the order of the actual nominations.
        # Ordinary named occurrences and padding have no record to install.
        records = records.flatten(0, 1)
        if not torch.compiler.is_compiling() and not bool((records[:, 0] >= 0).any()):
            fi['pool_word_texts'] = WordLabels(words, pool.labels)
            return True
        width = longest
        labels = pool.labels
        property_count = min(int(model.wholeSpace.nVectors), 63)
        part_width = min(width * (width + 1) // 2, table.capacity)
        end = 3 * width + part_width
        identity = ps.percept_store.identity
        from math import isqrt
        from itertools import islice
        from WordIdentity import StoredParts
        # At most one new catalog word per occurrence, including groups
        # installed earlier in this call. The simultaneously nominated
        # holders also have a [group, group, width] link frontier.
        columns = end + 6 + len(identity.words) + len(records)
        group_size = max(1, min((1 << 20) // columns, isqrt((1 << 20) // max(1, width))))
        # The width of the input bounds the scratch. Only actual nominations
        # cross to the catalog, with their physical addresses already resolved.
        for group in records.split(group_size):
            if not torch.compiler.is_compiling() and not bool((group[:, 0] >= 0).any()):
                continue
            known, longest_stored, runs = identity.nomination_inventory()
            headers, registrations, inside, added = nomination_records(
                table, known, longest_stored, group, width, runs)
            # The remaining host catalog reads one transfer. The two compact
            # membership columns have one entry per actual stored relation.
            packet = torch.cat((headers.flatten(), registrations.flatten(), inside, added)).tolist()
            values = iter(packet)
            head_rows = [tuple(islice(values, headers.shape[1])) for _ in range(len(headers))]
            witnesses = iter([tuple(islice(values, registrations.shape[1])) for _ in range(len(registrations))])
            contained = iter(tuple(islice(values, len(inside))))
            for form, word, seat, ordinal, bits, seated in head_rows:
                if labels is pool.labels:
                    labels = labels.copy()
                parts = holders = None
                if seated:
                    raw = identity._row_spellings[form]
                else:
                    length, n_words, n_runs, *payload = next(witnesses)
                    raw = bytes(payload[:length])
                    addresses = zip(payload[width:3 * width:2], payload[width + 1:3 * width:2], strict=True)
                    characters = [(raw[pos:pos + size], row)
                                  for pos, (row, size) in enumerate(addresses) if size]
                    stored = StoredParts(tuple(islice(contained, n_words)), tuple(islice(values, n_runs)))
                    parts, holders = (characters, stored), payload[3 * width:]
                wholes = tuple(i for i in range(property_count) if bits & (1 << i))
                installed = owner.interpret.install_nomination(
                    word, seat, form, raw, wholes, pool.initial_places[ordinal],
                    witnesses=parts, holder_rows=holders)
                if installed is None:
                    raise RuntimeError('MTM nominated a word without a free concept row')
                pool.form_labels[form] = raw.decode('latin1')
                description = owner._concept_allocator.word_descriptions[word]
                labels[word] = next(iter(description['forms']), '')
        pool.labels = labels
    fi['pool_word_texts'] = WordLabels(words, pool.labels)
    return True
