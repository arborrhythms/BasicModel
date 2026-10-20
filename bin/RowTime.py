"""Model-time provenance is a column, never an operand of composition."""
import torch

# More than 136 years at one batch per second. The two-rung ladder retains
# integer resolution throughout this declared run horizon; it never wraps.
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
    return encoder().encode(tick).to(store.when)


def migrate(store, state, prefix):
    """Old row clocks were source positions, not recoverable model times.

    Preserve addresses and content; explicitly mark historical times unknown
    (the zero phasor), rather than inventing processing-time evidence.
    """
    key = prefix + 'model_clock'
    if key in state:
        return False
    state[key] = torch.zeros_like(store.model_clock)
    if prefix + 'when' in state:
        state[prefix + 'when'] = torch.zeros_like(state[prefix + 'when'])
    return True
