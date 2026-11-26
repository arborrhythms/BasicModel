"""Lexical form admission and constructed byte capacity for surface tables."""
from collections.abc import Mapping
from typing import NamedTuple

import torch
from torch.nn import functional as F


def occupied_width(active):
    """End of the last occupied column across the batch, or zero."""
    if not active.numel():
        return active.new_zeros((), dtype=torch.long)
    slots = torch.arange(1, active.shape[1] + 1, device=active.device)
    return torch.where(active.any(0), slots, 0).amax()


def is_lexical_form(value):
    """Known text needs a character other than whitespace: all content of
    the input is a word, marks included.

    Bytes use the definition index's decoding. An opaque native part key
    carries no text to classify and retains its existing identity contract.
    """
    from Definitions import form_key
    value = form_key(value)
    if isinstance(value, str) and '\0' in value:
        # Zero content is the out-of-band omission, never an admitted word.
        return False
    return not isinstance(value, str) or any(not c.isspace() for c in value)


def capacity(value):
    """A raw byte bound; the candidate EOS reserve is separate."""
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'<wordSurfaceCapacity> must be a positive integer; got {value!r}')
    if isinstance(value, bool) or result < 1 or str(result) != str(value).strip():
        raise ValueError(f'<wordSurfaceCapacity> must be a positive integer; got {value!r}')
    return result


def stage(surfaces, raw_capacity, *, device, eos=False, retain_bytes=True, with_support=True):
    """Read each surface once; never clip or infer a width from current words.

    Native walks use only presence and primitive membership. Their compact
    tables therefore do not construct a second copy of the lexical bytes.
    """
    B = len(surfaces)
    K = len(surfaces[0]) if B else 0
    width = raw_capacity + int(eos)
    # These strings come from the old checkpoint/lesson interface. Assemble
    # its bytes before transfer so each word does not launch separate writes.
    octets = bytearray(B * K * width) if retain_bytes else None
    lengths = []
    members = [] if with_support and not retain_bytes else None
    for row in surfaces:
        if len(row) != K:
            raise ValueError('surface rows must have the constructed candidate count')
        for raw in row:
            if raw is None:
                raw = b''
            if not isinstance(raw, bytes):
                raise TypeError(f'word surface must be bytes, got {type(raw).__name__}')
            if len(raw) > raw_capacity:
                raise ValueError(f'word surface {raw!r} has {len(raw)} bytes; '
                                 f'wordSurfaceCapacity is {raw_capacity}')
            seat = len(lengths)
            lengths.append(len(raw))
            if raw:
                if octets is not None:
                    start = seat * width
                    octets[start:start + len(raw)] = raw
                if members is not None:
                    members.extend(seat * 256 + byte for byte in set(raw))
    lengths = torch.tensor(lengths, device=device, dtype=torch.long).reshape(B, K)
    data = valid = support = None
    if octets is not None:
        data = (torch.frombuffer(octets, dtype=torch.uint8).to(device) if octets else
                torch.empty(0, dtype=torch.uint8, device=device)).reshape(B, K, width)
        valid = torch.arange(width, device=device) < lengths[..., None]
        if with_support:
            support = primitive_support(data, valid)
    elif members is not None:
        support = torch.zeros(B * K * 256, device=device, dtype=torch.bool)
        support.index_fill_(0, torch.tensor(members, device=device, dtype=torch.long), True)
        support = support.reshape(B, K, 256)
    return data, valid, lengths > 0, support


def bytes_at(surface, spans, active, capacity):
    """Gather the observed spelling of each active interval for byte supervision."""
    from Meaning import require_sentence_tensor
    lo, hi = spans.unbind(-1)
    lengths = torch.where(active, hi - lo, 0)
    if surface.shape[1] > capacity:
        require_sentence_tensor((lengths <= capacity).all(),
                                f'word surface exceeds wordSurfaceCapacity {capacity}')
    offsets = torch.arange(capacity, device=surface.device)
    valid = offsets < lengths[..., None]
    source = torch.nn.functional.pad(surface, (0, 1))
    at = torch.where(valid, lo[..., None] + offsets, surface.shape[1])
    data = source.gather(1, at.flatten(1)).reshape_as(at).to(torch.uint8)
    return data, valid


def trimmed_spans(surface, spans):
    """The visible extent after bytes.strip(), including vertical tab and form feed."""
    length = surface.shape[1]
    if not length:
        return torch.zeros_like(spans)
    lo, hi = spans.unbind(-1)
    pos = torch.arange(length, device=surface.device)
    visible = (surface != 32) & ((surface < 9) | (surface > 13))
    first = torch.where(visible, pos, length).flip((1,)).cummin(1).values.flip((1,))
    last = torch.where(visible, pos + 1, 0).cummax(1).values
    start = torch.nn.functional.pad(first, (0, 1), value=length).gather(1, lo)
    end = torch.nn.functional.pad(last, (1, 0)).gather(1, hi)
    return torch.where((start < end)[..., None], torch.stack((start, end), -1), 0)


def spelling_sources(surface, bounds, present, *, across_pages=False):
    """The first observed seat with the same length and every byte in the same order."""
    width = surface.shape[1]
    if not width or not present.numel():
        return torch.full_like(bounds[..., 0], -1)
    lo, hi = bounds.unbind(-1)
    sizes = torch.where(present, hi - lo, 0)
    # Four consecutive bytes per int32, at every possible starting offset.
    # Length distinguishes a real trailing NUL from the zeros beyond an end.
    octets = F.pad(surface.to(torch.int32), (0, 3))
    words = octets[:, :width]
    for i in range(1, 4):
        words = words | (octets[:, i:i+width] << (8*i))
    offsets = torch.arange(0, width, 4, device=surface.device)
    remaining = (sizes[..., None] - offsets).clamp(0, 4)
    at = (lo[..., None] + offsets).clamp(0, width)
    chunks = F.pad(words, (0, 1)).gather(1, at.flatten(1)).reshape_as(at)
    masks = words.new_tensor([0, 255, 65535, 16777215, -1])
    spellings = chunks & masks[remaining]
    if across_pages:
        spellings = spellings.flatten(0, 1)[None]
        sizes = sizes.reshape(1, -1)
    B, W, columns = spellings.shape
    if B * W * W * columns <= 1 << 20:
        same = (spellings[:, :, None] == spellings[:, None]).all(-1)
        same = same & (sizes[:, :, None] == sizes[:, None])
        first = same.int().argmax(-1)
    else:
        # Equal runs stay in presentation order under each stable sort.
        # Compare only neighbours once all columns have been ordered.
        seats = torch.arange(W, device=surface.device).expand(B, -1)
        order = seats
        for column in reversed(spellings.unbind(-1)):
            order = order.gather(1, column.gather(1, order).argsort(stable=True))
        sorted_sizes, by_size = sizes.gather(1, order).sort(stable=True)
        order = order.gather(1, by_size)
        sorted_bytes = spellings.gather(1, order[..., None].expand_as(spellings))
        boundary = ((sorted_sizes[:, 1:] != sorted_sizes[:, :-1])
                    | (sorted_bytes[:, 1:] != sorted_bytes[:, :-1]).any(-1))
        begin = torch.where(F.pad(boundary, (1, 0), value=True), seats, 0).cummax(1).values
        first = torch.empty_like(order).scatter(1, order, order.gather(1, begin))
    return torch.where(present, first.reshape_as(present), -1)


def reading_forms(surface, spans, active, *, trimmed=None):
    """Name each ordered spelling by its first live occurrence on the same page."""
    if not surface.shape[1] or not active.shape[1]:
        return torch.zeros_like(spans), torch.full_like(spans[..., 0], -1)
    if trimmed is None:
        trimmed = trimmed_spans(surface, spans)
    present = active & (trimmed[..., 1] > trimmed[..., 0])
    bounds = torch.where(present[..., None], trimmed, 0)
    return bounds, spelling_sources(surface, bounds, present)


def held_properties(surface, spans, memberships):
    """Property rows held by any non-padding byte within each unit."""
    lo, hi = spans.unbind(-1)
    width = memberships.shape[0]
    if not width:
        return torch.zeros((*lo.shape, 0), device=surface.device, dtype=torch.bool)
    # Bound scratch for a long page or a large property inventory. The
    # reduction is over bytes in the interval and retains repeated letters.
    group = max(1, (1 << 20) // max(1, surface.numel()))
    blocks = []
    for rows in memberships.split(group, dim=0):
        held = (rows.T[surface.long()] > .5) & (surface != 0)[..., None]
        prefix = torch.nn.functional.pad(held.cumsum(1, dtype=torch.int32), (0, 0, 1, 0))
        ends = prefix.gather(1, hi[..., None].expand(-1, -1, len(rows)))
        starts = prefix.gather(1, lo[..., None].expand(-1, -1, len(rows)))
        blocks.append(ends > starts)
    return torch.cat(blocks, -1)


def reading_property_signatures(surface, spans, signatures, discard, *, latin1_text=True):
    """Intersect primitive memberships over each observed reading's spelling.

    The existing nomination interface passes Latin-1 word text to the UTF-8
    property reader. Read both UTF-8 chart entries for each high byte here.
    """
    width = surface.shape[1]
    start, end = spans.unbind(-1)
    length = (end - start).clamp_min(0)
    result = torch.full_like(start, -1)
    byte = surface.long()
    table = signatures[byte]
    if latin1_text:
        table = torch.where(byte < 128, table,
                            signatures[192 | (byte >> 6)] & signatures[128 | (byte & 63)])
    # Read each requested length as adjacent power-of-two intervals.
    for level in range(width.bit_length()):
        size = 1 << level
        take = (length & size) != 0
        segment = torch.nn.functional.pad(table, (0, 1), value=-1).gather(1, start)
        result = torch.where(take, result & segment, result)
        start = torch.where(take, start + size, start)
        if 2 * size <= width:
            table = table & torch.nn.functional.pad(table[:, size:], (0, size), value=-1)
    return torch.where(length > 0, result & ~discard, 0)


def reading_witnesses(surface, ids, events, signatures, discard, clipped, attention=None):
    """Locate complete primitive runs and assign their native part witnesses.

    The mixing reader ends at the first NUL. Each native event contributes
    only to a run that contains both endpoints. Overlapping and incomplete
    inputs keep the original sum-of-extents coverage rule.
    """
    from Meronomy import sentence_unit_spans
    surface = surface.masked_fill((surface == 0).long().cumsum(1) != 0, 0)
    units = sentence_unit_spans(surface, events[:, :0], include_gaps=False)
    start, end = units.unbind(-1)
    present = end > start
    width = surface.shape[1]
    lo, hi = events.unbind(-1)
    seat = torch.searchsorted(torch.where(present, start, width).contiguous(), lo.contiguous(), right=True) - 1
    index = seat.clamp_min(0)
    selected = ((ids >= 0) & (lo < hi) & (seat >= 0) & present.gather(1, index)
                & (hi <= end.gather(1, index)))
    total = torch.zeros_like(start).scatter_add(1, index, torch.where(selected, hi - lo, 0))
    present = present & (total == end - start)
    present = present & ~(clipped[:, None] & (end == width) & (start + 1 == end))
    if attention is not None:
        attended, accepted = attention
        # Each disjoint run has one start. A fixation must cover that exact run.
        at = attended[..., 0].contiguous()
        target = torch.searchsorted(torch.where(end > start, start, width).contiguous(), at, right=True) - 1
        index_at = target.clamp_min(0)
        same = ((target >= 0) & (start.gather(1, index_at) == at)
                & (end.gather(1, index_at) == attended[..., 1]))
        visited = torch.zeros_like(start).scatter_add(1, index_at, (accepted & same).long())
        present = present & (visited > 0)
    first = torch.nn.functional.pad(surface, (0, 1)).gather(1, start)
    lexical = ((65 <= first) & (first <= 90)) | ((97 <= first) & (first <= 122))
    bits = reading_property_signatures(surface, units, signatures, discard, latin1_text=False)
    return (torch.stack((start, torch.where(present, end, start), lexical.long(), bits), -1),
            torch.where(selected & present.gather(1, index), seat, -1))


def witnessed_columns(witnesses, starts, active, width):
    """Address a disjoint native part by its first byte in the delivered page."""
    at = torch.where(active & (starts >= 0) & (starts < width), starts, width)
    seats = torch.arange(starts.shape[1], device=starts.device).expand_as(at)
    columns = starts.new_full((len(starts), width + 1), -1)
    columns.scatter_reduce_(1, at, seats, 'amax')
    lo, hi = witnesses[..., 0], witnesses[..., 1]
    return torch.where(hi > lo, columns.gather(1, lo.clamp(0, width)), -1)


class MixingWords(NamedTuple):
    bounds: torch.Tensor
    columns: torch.Tensor
    names: torch.Tensor
    forms: torch.Tensor


def mixing_columns(units, seats, spans, active, witnessed):
    """The unique grammar location ending at each completed reading."""
    batch, count = units.shape[:2]
    width = active.shape[1]
    if witnessed:
        return units[..., 4]
    if spans is None:
        positions = torch.arange(seats.shape[1], device=seats.device).expand_as(seats)
        last = seats.new_full((batch, count + 1), -1).scatter_reduce_(
            1, torch.where(seats >= 0, seats, count), positions, 'amax')[:, :-1]
        return torch.where(last < width, last, -1)
    spans = spans[:, :width]
    n = spans.shape[1]
    # The units are disjoint, ordered runs. Exactly one active native span
    # must finish at the right boundary, with its left end inside the run.
    lo, hi = units[..., 0], units[..., 1]
    starts, stops = spans.unbind(-1)
    # Incomplete runs have hi == lo. Their starts remain in source order.
    # Padded spans end at zero and cannot match a nonempty native event.
    ordered = torch.where((hi > lo), hi, torch.iinfo(hi.dtype).max)
    ends, index = ordered.sort(dim=1, stable=True)
    at = torch.searchsorted(ends, stops.contiguous())
    index = F.pad(index, (0, 1), value=count).gather(1, at)
    first = F.pad(lo, (0, 1)).gather(1, index)
    valid = ((at < count) & (F.pad(ends, (0, 1)).gather(1, at) == stops)
             & (first <= starts) & (starts < stops) & active[:, :n])
    target = torch.where(valid, index, count)
    counts = torch.zeros_like(F.pad(lo, (0, 1))).scatter_add_(1, target, valid.long())
    locations = torch.arange(n, device=seats.device).expand_as(target)
    last = torch.full_like(counts, -1).scatter_reduce_(1, target, locations, 'amax')
    return torch.where(counts[:, :-1] == 1, last[:, :-1], -1)


class WitnessedForms(Mapping):
    """The actual alias read at each word seat during one presentation."""
    def __init__(self, seats, forms, valid, capacity, labels):
        words = seats.flatten()
        target = torch.where(valid.flatten(), words, capacity)
        position = torch.arange(len(words), device=words.device) + 1
        latest = words.new_zeros(capacity + 1).scatter_reduce_(0, target, position, 'amax')
        self.forms = F.pad(forms.flatten(), (1, 0), value=-1)[latest[:capacity]]
        self.labels = labels
        self._materialized = None

    def read(self, rows):
        valid = (rows >= 0) & (rows < len(self.forms))
        forms = F.pad(self.forms, (0, 1), value=-1)[torch.where(valid, rows, len(self.forms))]
        return [tuple(self.labels.get(form) for form in stream) for stream in forms.tolist()]

    def _mapping(self):
        if self._materialized is None:
            self._materialized = {seat: self.labels[form] for seat, form in enumerate(self.forms.tolist()) if form >= 0}
        return self._materialized

    def __len__(self):
        return len(self._mapping())

    def __iter__(self):
        return iter(self._mapping())

    def __getitem__(self, key):
        return self._mapping()[key]


def mixing_reference_rows(owner, words, width):
    """Place word and referent identities in their witnessed grammar columns."""
    from Meaning import require_sentence_tensor
    from WordConceptPool import basic_word_rows
    names = words.names
    objects, object_seats, orders = owner.interpret.sentence_objects(names)
    matrix = owner._concept_allocator.layer(0)
    word_seats = matrix.concept_rows(names)
    keys, rows = owner.basic_percepts.word_row_index(None, names.device)
    basic = basic_word_rows(names, keys, rows)
    ends = names.new_tensor(owner._order_caps()).cumsum(0)[:-1]
    word_order = torch.bucketize(word_seats, ends, right=True)
    valid = (names >= 0) & (words.columns >= 0) & (words.columns < width)
    require_sentence_tensor((~valid | (basic >= 0)).all(), 'a witnessed word has no basic percept')
    # Multiple witnesses at a seat were written in source order.
    target = torch.where(valid, words.columns, width)
    positions = torch.arange(names.shape[1], device=names.device).expand_as(names)
    source = names.new_full((len(names), width + 1), -1).scatter_reduce_(1, target, positions, 'amax')[:, :width]
    def placed(column, present):
        values = F.pad(torch.where(present, column, -1), (0, 1), value=-1)
        return values.gather(1, torch.where(source >= 0, source, names.shape[1])).contiguous()
    word_live, object_live = word_seats >= 0, object_seats >= 0
    word_ids = placed(names, word_live | object_live)
    word_rows = placed(word_seats, word_live)
    referents = placed(objects, object_live)
    # A grammar leaf's historical lexical form is the witnessed alias. Keep
    # the last occurrence for each physical WORD seat, as in the host writer.
    forms = WitnessedForms(word_seats, words.forms, valid & word_live, owner.nVectors,
                           owner.mtm.word_pool.form_labels)
    return ((placed(basic, valid), word_ids, word_rows, placed(word_order, word_live),
             placed(object_seats, object_live), referents.clone(), referents, placed(orders, object_live)), forms)


def mixing_word_witnesses(model, raw, ids, spans, attention, *, grammar_starts=None):
    """Translate located byte evidence once for the existing MTM word writer."""
    from Spaces import _analysis_property_signature
    rows = raw.reshape(len(raw), -1)
    capacity = rows.shape[1]
    delivered = getattr(model.inputSpace, '_delivered_length', None)
    if delivered is not None:
        rows = rows[:, :delivered]
    surface = rows.long()
    sentences = getattr(model.inputSpace, '_last_sentences', None)
    clipped = torch.tensor([surface.shape[1] == capacity and
                            (sentences is None or b >= len(sentences) or
                             len(sentences[b].encode('utf-8')) > capacity)
                            for b in range(len(rows))], device=rows.device, dtype=torch.bool)
    attended = getattr(model, '_attention_spans', None)
    fixations = (attended, attention.accepted) if attention is not None and torch.is_tensor(attended) else None
    signatures, discard = _analysis_property_signature(model.wholeSpace, device=rows.device)
    units, seats = reading_witnesses(surface, ids, spans, signatures, discard, clipped, fixations)
    # Unit records preserve source order, including several events covering
    # the same span. The byte column belongs to the input, not to a tiling.
    if grammar_starts is not None:
        columns = witnessed_columns(units, grammar_starts, model.inputSpace._word_active_mask, surface.shape[1])
        # A column holds the witness only when its unit spans it with the same bytes.
        stem = model.perceptualSpace._forward_input
        spans_at = stem['unit_spans'].gather(1, columns.clamp_min(0)[..., None].expand(-1, -1, 2))
        read = stem['sentence_bytes'].long()
        width = max(read.shape[1], surface.shape[1])
        differ = F.pad(read, (0, width - read.shape[1])) != F.pad(surface, (0, width - surface.shape[1]))
        prefix = F.pad(differ.long().cumsum(1), (1, 0))
        lo, hi = units[..., 0].clamp(0, width), units[..., 1].clamp(0, width)
        same = (spans_at[..., 0] == units[..., 0]) & (spans_at[..., 1] == units[..., 1])
        same = same & (prefix.gather(1, hi) == prefix.gather(1, lo))
        units = torch.cat((units, torch.where(same, columns, -1)[..., None]), dim=-1)
    # Resolve a mixing witness by the pool occurrence at these exact byte
    # endpoints. A partial overlap or an unadmitted word carries no name.
    fi = model.perceptualSpace._forward_input
    words = fi.get('pool_word_ids')
    if words is not None:
        limits = fi['pool_spans']
        if limits.shape[1]:
            lo, hi = limits.unbind(-1)
            column = witnessed_columns(units, lo, hi > lo, surface.shape[1])
            at = column.clamp_min(0)
            exact = (column >= 0) & (hi.gather(1, at) == units[..., 1])
            names = torch.where(exact, words.gather(1, at), -1)
        else:
            names = torch.full_like(units[..., 0], -1)
        forms = (torch.where(exact, fi['pool_form_rows'].gather(1, at), -1)
                 if limits.shape[1] else torch.full_like(names, -1))
        columns = mixing_columns(units, seats, fi.get('part_spans'), model.inputSpace._word_active_mask,
                                 grammar_starts is not None)
        names = torch.where((units[..., 1] > units[..., 0]) & units[..., 2].bool(), names, -1)
        return MixingWords(units[..., :2], columns, names, forms)
    events = torch.stack((seats, ids), -1)
    fields = (units.flatten(1), events.flatten(1), surface)
    records = torch.cat(fields, 1).tolist()
    stride = units.shape[-1]
    event_offset = units.shape[1] * stride
    byte_offset = event_offset + events.shape[1] * 2
    extents, observations = [], []
    for b, record in enumerate(records):
        grouped = {}
        for offset in range(event_offset, byte_offset, 2):
            seat, part = record[offset:offset + 2]
            if seat >= 0:
                grouped.setdefault(seat, []).append(((offset - event_offset) // 2, part))
        data = bytes(record[byte_offset:])
        located = []
        for seat, offset in enumerate(range(0, event_offset, stride)):
            lo, hi, lexical, bits = record[offset:offset + 4]
            if lo >= hi:
                continue
            located.append((lo, hi))
            if lexical:
                members = grouped[seat]
                wholes = tuple(row for row in range(min(int(model.wholeSpace.nVectors), 63)) if bits & (1 << row))
                column = members[-1][0] if grammar_starts is None else record[offset + 4]
                observations.append((b, lo, hi, data[lo:hi], [part for _, part in members], column, wholes, None))
        extents.append(located)
    return extents, observations


def unit_texts(surface, units):
    """Host spellings and extents of the present units, in one read; for readers keyed by spelling."""
    width = surface.shape[1]
    texts, intervals = [], []
    for row in torch.cat((surface, units.flatten(1)), 1).tolist():
        raw, ends = bytes(row[:width]), row[width:]
        spans = [(s, e) for s, e in zip(ends[0::2], ends[1::2]) if e > s]
        intervals.append(spans)
        texts.append([raw[s:e].decode("latin1") for s, e in spans])
    return texts, intervals


def grammar_leaves(surface, spans, active):
    """Mask all-whitespace Latin-1 units in the native byte presentation."""
    # Serial word surfaces use lossless Latin-1. Python str.isspace()
    # includes the four information separators, NEL and NBSP on this chart.
    space = (((surface >= 9) & (surface <= 13)) | ((surface >= 28) & (surface <= 32))
             | (surface == 133) | (surface == 160))
    prefix = torch.nn.functional.pad(space.long().cumsum(-1), (1, 0))
    lo, hi = spans.unbind(-1)
    spaces = prefix.gather(1, hi) - prefix.gather(1, lo)
    return active & ~((hi > lo) & (spaces == hi - lo))


def primitive_support(data, valid):
    """Compatibility for explicit banks; only scatter indices become int64."""
    return torch.zeros(*data.shape[:2], 256, dtype=torch.int32, device=data.device).scatter_add(
        -1, data.long(), valid.to(torch.int32)).bool()
