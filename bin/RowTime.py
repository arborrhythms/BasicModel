"""Model time in the percept band, never a separate durable time column."""
import math
import torch
from functools import wraps

# Four coordinates form the existing two-rung ladder: over 136 years at
# one sentence/thought per second, without increasing storage width.
HORIZON = (1 << 32) - 1


from Spaces import WhenEncoding, signed_phase, unsigned_phase

_ENCODER = None


def encoder():
    """The one encoder of model time: every code of a time is made by it."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = WhenEncoding(n_when=4).set_capacity(HORIZON)
    return _ENCODER


def produced(method):
    """The output trials share the tick of the one sentence they produce."""
    @wraps(method)
    def sentence(model, *args, **kwargs):
        if not model.answer_synthesis or '_produced_when' in model.__dict__:
            return method(model, *args, **kwargs)
        model._produced_when = model._advance_when_time()
        try:
            return method(model, *args, **kwargs)
        finally:
            del model._produced_when
    return sentence


def how_far_back(present, stamp):
    """The comparator: the code of how far ``stamp`` lies behind ``present``.

    Both are codes of the one encoder, each a tick's unit bracket on two
    rungs. On each rung the stamp's phase is taken from the present's, and
    the bracket is laid again from the difference, so the result is the
    encoder's own code of the distance in ticks. A band of zeros (no stamp)
    gives zeros.
    """
    clock = encoder()
    # A present per stream reads every stamp of that stream.
    while present.ndim < stamp.ndim:
        present = present.unsqueeze(-2)
    # The rotation is taken in double precision, so a distance decodes to
    # its exact tick count over the whole ladder.
    present = signed_phase(present.to(device=stamp.device, dtype=torch.float64))
    signed = signed_phase(stamp.to(torch.float64))
    pairs = []
    for rung, step in zip(range(0, stamp.shape[-1], 2), (clock.div_lf, clock.div_hf)):
        ps, pc = present[..., rung], present[..., rung + 1]
        ss, sc = signed[..., rung], signed[..., rung + 1]
        sin, cos = ps * sc - pc * ss, pc * sc + ps * ss
        norm = torch.sqrt(sin * sin + cos * cos).clamp_min(1e-300)
        sin, cos = sin / norm, cos / norm
        turn_sin, turn_cos = math.sin(step), math.cos(step)
        pairs += [.5 * (sin + sin * turn_cos + cos * turn_sin),
                  .5 * (cos + cos * turn_cos - sin * turn_sin)]
    code = unsigned_phase(torch.stack(pairs, -1)).to(stamp.dtype)
    return torch.where(stamp.ne(0).any(-1, keepdim=True), code, 0.)


def compared(model, idea):
    """A reader's full-width idea, its stamp read through the comparator
    against the present on the model."""
    store = model.symbolSpace.ltm_store
    start = None if store is None else store.when_start
    if start is None or idea.shape[-1] <= start:
        return idea
    return torch.cat((idea[..., :start], how_far_back(model.present_code(), idea[..., start:start+4]),
                      idea[..., start+4:]), -1)


def subintervals(onset, rank, count):
    """Exact word order within a tick, even at the ladder's last tick.

    Metal has no float64 storage. Its small coordinate table stays on the
    host; content, evidence and their gradients stay on their native device.
    """
    device = 'cpu' if onset.device.type == 'mps' else onset.device
    onset = onset.to(device=device).to(dtype=torch.float64)
    rank = rank.to(device=device).to(dtype=torch.float64)
    count = count.to(device=device).to(dtype=torch.float64)
    return torch.stack((onset + rank / count, onset + (rank + 1) / count), -1)


def present_stamp(store):
    """The present's code for a row written now, from the model that owns
    the store (or the one encoder's code of a present it was given); zeros
    (no stamp) for a store with no present."""
    source = getattr(store, '_model_present_code', None)
    band = store.slots.new_zeros(4)
    if source is not None:
        return source().to(band)
    tick = getattr(store, '_model_present', None)
    return band if tick is None else encoder().encode(int(tick())).to(band)


def present_tick(store):
    """The model's present as an integer, for what is not learned (the
    order of mentions); zero for a store with no model."""
    source = getattr(store, '_model_present', None)
    return 0 if source is None else int(source())


def name_row(store):
    """The next row name of ``store``: the count of the names it has issued,
    with this one. Names begin at one; zero names nothing."""
    name = int(store.names_issued) + 1
    store.names_issued.fill_(name)
    return name


def read_time(store, index):
    if isinstance(index, tuple):
        return store.slots.new_zeros(4)
    if store.when_start is None:
        return store.slots.new_zeros(4)
    return store.slots[int(index), store.when_start:store.when_start+4]


def write_time(store, index, value):
    band = read_time(store, index)
    if value is False or value is None:
        band.zero_()
    else:
        value = torch.as_tensor(value, device=band.device, dtype=band.dtype)
        if value.shape != (4,) or not bool(torch.isfinite(value).all()):
            raise ValueError('entry time requires four finite coordinates')
        band.copy_(value)
