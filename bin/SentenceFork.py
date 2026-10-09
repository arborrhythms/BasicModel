"""A detached reservoir snapshot, joined independently by each batch row."""
import torch
from SentenceCompose import select_rows


def detached(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(detached(item) for item in value)
    if isinstance(value, dict):
        return {key: detached(item) for key, item in value.items()}
    raise TypeError(f'unsupported fork value: {type(value).__name__}')


class SentenceFork:
    """Reservoir sampling costs no prefix replay and retains no prefix graph.

    The greedy walk replaces a row's saved state with probability 1/k at its
    kth eligible decision. The explore walk joins that state at exactly that
    word and round. Before joining, a row performs no composition or deposit.
    Other rows may already be completing their suffix in the same batch.
    """
    def __init__(self, active):
        self.count = torch.zeros_like(active, dtype=torch.long)
        self.word = torch.full_like(self.count, -1)
        self.slot = torch.full_like(self.count, -1)
        self.action = torch.full_like(self.count, -1)
        self.state = self.latches = None
        self.explore = False
        self.selected = torch.zeros_like(active)
        self.joined = torch.zeros_like(active)
        self.taken = torch.zeros_like(active)

    def record(self, choice, slot, state, *, word, latches):
        if self.explore:
            at = self.selected & (self.slot == slot)
            self.taken |= at & choice.valid & (choice.action != self.action)
            return
        eligible = choice.valid & (choice.alternative_count > 0)
        if not bool(eligible.any()):
            return
        self.count += eligible.long()
        take = eligible & (torch.rand_like(self.count, dtype=torch.float32)
                           < self.count.clamp_min(1).reciprocal())
        if not bool(take.any()):
            return
        state, latches = detached(state), detached(tuple(latches))
        self.state = state if self.state is None else select_rows(self.state, state, take)
        self.latches = latches if self.latches is None else select_rows(self.latches, latches, take)
        self.word = torch.where(take, word, self.word)
        self.slot = torch.where(take, slot, self.slot)
        self.action = torch.where(take, choice.action, self.action)

    def start(self, draw):
        self.explore = True
        self.selected = draw['compose_round'] >= 0
        self.joined = draw['narrowing'].clone()
        self.taken.zero_()

    def begin_word(self, word, latches):
        if not self.explore:
            return latches
        take = self.selected & (self.word == word)
        return (latches if self.latches is None else
                select_rows(tuple(latches), self.latches, take))

    def resume(self, slot, state):
        if not self.explore or self.state is None:
            return state
        take = self.selected & ~self.joined & (self.slot == slot)
        self.joined |= take
        return select_rows(state, self.state, take) if bool(take.any()) else state

    def mask(self, slot):
        if not self.explore:
            return None
        return torch.where(self.selected & (self.slot == slot), self.action, -1)

    def pending(self, word, *, closing, width):
        if not self.explore:
            return False
        return bool((self.selected & ~self.joined & (self.word == word)
                     & ((self.slot >= 3 * width) == closing)).any())
