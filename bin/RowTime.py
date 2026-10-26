"""Model time in the percept band, never a separate durable time column."""
import torch

# Four coordinates form the existing two-rung ladder: over 136 years at
# one batch per second, without wrap or an increase in storage width.
HORIZON = (1 << 32) - 1


def encoder():
    from Spaces import WhenEncoding
    return WhenEncoding(n_when=4).set_capacity(HORIZON)


def stamp(store):
    source = getattr(store, '_model_present', None)
    tick = int(store.model_clock) if source is None else int(source())
    if not 0 <= tick <= HORIZON:
        raise OverflowError('model time exceeds the declared row-time ladder horizon')
    store.model_clock.fill_(max(int(store.model_clock), tick + 1))
    return encoder().encode(tick).to(store.slots)


def read_stamp(store, index):
    if isinstance(index, tuple):
        return store.parts.percept[int(index[1]), 4:]
    if store.when_start is None:
        return store.slots.new_zeros(4)
    return store.slots[int(index), store.when_start:store.when_start+4]


def write_stamp(store, index, value):
    band = read_stamp(store, index)
    if value is False or value is None:
        band.zero_()
    else:
        value = torch.as_tensor(value, device=band.device, dtype=band.dtype)
        if value.shape != (4,) or not bool(torch.isfinite(value).all()):
            raise ValueError('percept time requires four finite coordinates')
        band.copy_(value)
