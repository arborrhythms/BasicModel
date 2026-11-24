"""Lexical form admission and constructed byte capacity for surface tables."""
import torch


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
    data = torch.zeros(B, K, width, dtype=torch.uint8) if retain_bytes else None
    valid = torch.zeros(B, K, width, dtype=torch.bool) if retain_bytes else None
    present = torch.zeros(B, K, dtype=torch.bool)
    support = torch.zeros(B, K, 256, dtype=torch.bool) if with_support else None
    for b, row in enumerate(surfaces):
        if len(row) != K:
            raise ValueError('surface rows must have the constructed candidate count')
        for k, raw in enumerate(row):
            if raw is None:
                continue
            if not isinstance(raw, bytes):
                raise TypeError(f'word surface must be bytes, got {type(raw).__name__}')
            if len(raw) > raw_capacity:
                raise ValueError(f'word surface {raw!r} has {len(raw)} bytes; '
                                 f'wordSurfaceCapacity is {raw_capacity}')
            if raw:
                present[b, k] = True
                if support is not None:
                    support[b, k, sorted(set(raw))] = True
                if retain_bytes:
                    data[b, k, :len(raw)] = torch.tensor(list(raw), dtype=torch.uint8)
                    valid[b, k, :len(raw)] = True
    return (None if data is None else data.to(device),
            None if valid is None else valid.to(device),
            present.to(device), None if support is None else support.to(device))


def bytes_at(surface, spans, active, capacity):
    """Gather the observed spelling of each active interval for byte supervision."""
    lo, hi = spans.unbind(-1)
    lengths = torch.where(active, hi - lo, 0)
    if surface.shape[1] > capacity and bool((lengths > capacity).any()):
        longest = int(lengths.max())
        raise ValueError(f'word surface has {longest} bytes; wordSurfaceCapacity is {capacity}')
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


def reading_forms(surface, spans, active, *, trimmed=None):
    """Name each ordered spelling by its first live occurrence on the same page."""
    B, length = surface.shape
    W = active.shape[1]
    if not length or not W:
        return torch.zeros_like(spans), torch.full_like(spans[..., 0], -1)
    if trimmed is None:
        trimmed = trimmed_spans(surface, spans)
    present = active & (trimmed[..., 1] > trimmed[..., 0])
    bounds = torch.where(present[..., None], trimmed, 0)
    # Bound transient pairwise comparison for larger constructed capacities.
    spellings, _ = bytes_at(surface, bounds, present, length)
    sizes = bounds[..., 1] - bounds[..., 0]
    block = max(1, 1048576 // max(1, B * W * length))
    keys = []
    for i in range(0, W, block):
        same = (spellings[:, i:i+block, None] == spellings[:, None]).all(-1)
        same = same & (sizes[:, i:i+block, None] == sizes[:, None])
        keys.append(same.long().argmax(-1))
    return bounds, torch.where(present, torch.cat(keys, 1), -1)


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


def mixing_word_witnesses(model, raw, ids, spans, attention):
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
    events = torch.stack((seats, ids), -1)
    fields = (units.flatten(1), events.flatten(1), surface)
    records = torch.cat(fields, 1).tolist()
    event_offset = units.shape[1] * 4
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
        for seat, offset in enumerate(range(0, event_offset, 4)):
            lo, hi, lexical, bits = record[offset:offset + 4]
            if lo >= hi:
                continue
            located.append((lo, hi))
            if lexical:
                members = grouped[seat]
                wholes = tuple(row for row in range(min(int(model.wholeSpace.nVectors), 63)) if bits & (1 << row))
                observations.append((b, lo, hi, data[lo:hi], [part for _, part in members], members[-1][0], wholes))
        extents.append(located)
    return extents, observations


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
