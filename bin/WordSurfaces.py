"""Constructed byte capacity shared by sentence and lexical surface tables."""
import torch


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


def primitive_support(data, valid):
    """Compatibility for explicit banks; only scatter indices become int64."""
    return torch.zeros(*data.shape[:2], 256, dtype=torch.int32, device=data.device).scatter_add(
        -1, data.long(), valid.to(torch.int32)).bool()
