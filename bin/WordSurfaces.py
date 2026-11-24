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


def reading_forms(surface, spans, active):
    """Trim byte whitespace and identify repeated ordered spellings within each page.

    A name is the first live place with the same bytes on this page. It is
    only an equality key for this reading. Identical multisets in different
    orders, and repetitions with different lengths, remain distinct.
    """
    B, length = surface.shape
    W = active.shape[1]
    lo, hi = spans.unbind(-1)
    if not length or not W:
        return torch.zeros_like(spans), torch.full_like(lo, -1)
    pos = torch.arange(length, device=surface.device)
    # Match bytes.strip(), including vertical tab and form feed.
    visible = (surface != 32) & ((surface < 9) | (surface > 13))
    first = torch.where(visible, pos, length).flip((1,)).cummin(1).values.flip((1,))
    last = torch.where(visible, pos + 1, 0).cummax(1).values
    start = torch.nn.functional.pad(first, (0, 1), value=length).gather(1, lo)
    end = torch.nn.functional.pad(last, (1, 0)).gather(1, hi)
    present = active & (start < end)
    bounds = torch.where(present[..., None], torch.stack((start, end), -1), 0)
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


def primitive_support(data, valid):
    """Compatibility for explicit banks; only scatter indices become int64."""
    return torch.zeros(*data.shape[:2], 256, dtype=torch.int32, device=data.device).scatter_add(
        -1, data.long(), valid.to(torch.int32)).bool()
