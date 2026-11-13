"""Model time in the percept band, never a separate durable time column."""
import torch
from functools import wraps

# Four coordinates form the existing two-rung ladder: over 136 years at
# one sentence/thought per second, without increasing storage width.
HORIZON = (1 << 32) - 1


from Spaces import WhenEncoding


class _ModelTimeEncoding(WhenEncoding):
    """A transient tensor clock, so compiled reads do not guard its value.

    BasicModel.when_time remains the saved clock. This encoder's existing
    current-time view follows assignments in place and adds no checkpoint key.
    """
    @property
    def t(self):
        return self._model_tick

    @t.setter
    def t(self, value):
        if '_model_tick' not in self._buffers:
            self.register_buffer('_model_tick', torch.as_tensor(value, dtype=torch.long),
                                 persistent=False)
        else:
            self._model_tick.copy_(torch.as_tensor(value, device=self._model_tick.device,
                                                  dtype=torch.long))


def encoder():
    return _ModelTimeEncoding(n_when=4).set_capacity(HORIZON)


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


def age_interval(model, interval):
    """Age reverses the ends of a historical interval, including age zero."""
    return model.when_time.to(interval.device) - interval.flip(-1)


def subintervals(onset, rank, count):
    """Exact word order within a tick, even at the ladder's last tick.

    Metal has no float64 storage. Its small coordinate table stays on the
    host; readers subtract the clock there before moving ages to the device.
    Content, evidence and their gradients stay on their native device.
    """
    device = 'cpu' if onset.device.type == 'mps' else onset.device
    onset = onset.to(device=device).to(dtype=torch.float64)
    rank = rank.to(device=device).to(dtype=torch.float64)
    count = count.to(device=device).to(dtype=torch.float64)
    return torch.stack((onset + rank / count, onset + (rank + 1) / count), -1)


def age_band(model, band):
    """A learned reader sees elapsed time, never the phase of a stored time."""
    occupied = band.ne(0).any(-1)
    age = model.when_time.to(band.device) - model.when_encoding.decode_index(band)
    return torch.where(occupied[..., None], model.when_encoding.encode(age).to(band), 0.)


def relative_to(band, now):
    """Express a stored band relative to the supplied observation tick."""
    clock = encoder()
    tick = clock.decode_index(now)
    while tick.ndim < band.ndim - 1:
        tick = tick.unsqueeze(-1)
    age = tick - clock.decode_index(band)
    return torch.where(band.ne(0).any(-1, keepdim=True), clock.encode(age).to(band), 0.)


def difference(clock, band):
    """A composition reads elapsed time as a departure from the present.

    Times stay on the shared ladder. The learned content mixer receives
    differences from its zero-age band, so the current tick contributes no
    constant content and changing the clock's origin cannot change a read.
    """
    age = clock.t - clock.decode_index(band)
    delta = clock.encode(age).to(band) - clock.encode(0).to(band)
    return torch.where(band.ne(0).any(-1, keepdim=True), delta, 0.)


def relative_idea(model, idea):
    """A reader's full-width idea, with its located form expressed as age."""
    store = model.symbolSpace.ltm_store
    start = None if store is None else store.when_start
    if start is None or idea.shape[-1] <= start:
        return idea
    return torch.cat((idea[..., :start], age_band(model, idea[..., start:start+4]),
                      idea[..., start+4:]), -1)


def next_time(store):
    source = getattr(store, '_model_present', None)
    tick = int(store.model_clock) if source is None else int(source())
    if not 0 <= tick <= HORIZON:
        raise OverflowError('model time exceeds the declared row-time ladder horizon')
    store.model_clock.fill_(max(int(store.model_clock), tick + 1))
    return encoder().encode(tick).to(store.slots)


def read_time(store, index):
    if isinstance(index, tuple):
        row = store.index_of_row(int(store.parts.sentence_reference[int(index[1])]))
        return store.slots.new_zeros(4) if row is None else read_time(store,row)
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
