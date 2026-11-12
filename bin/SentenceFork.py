"""A reservoir snapshot, joined independently by each batch row."""
import torch
from SentenceCompose import select_rows


def snapshot(value, *, live=False):
    if torch.is_tensor(value):
        return value.clone() if live else value.detach().clone()
    if isinstance(value, tuple):
        return tuple(snapshot(item, live=live) for item in value)
    if isinstance(value, dict):
        return {key: snapshot(item, live=live) for key, item in value.items()}
    raise TypeError(f'unsupported fork value: {type(value).__name__}')


def detached(value):
    return snapshot(value)


class SentenceFork:
    """Reservoir sampling costs no prefix replay.

    The single-backward route retains its prefix graph until the sentence's
    update; the separate route snapshots detached values.

    The greedy walk replaces a row's saved state with probability 1/k at its
    kth eligible decision. The explore walk joins that state at exactly that
    word and round. Before joining, a row performs no composition or deposit.
    Other rows may already be completing their suffix in the same batch.
    """
    def __init__(self, active, *, live=False, capacity=1):
        self.live = bool(live)
        self.count = torch.zeros_like(active, dtype=torch.long)
        self.word = torch.full_like(self.count, -1)
        self.slot = torch.full_like(self.count, -1)
        self.action = torch.full_like(self.count, -1)
        self.state = self.latches = None
        self.inverse_candidates = self.inverse_prefix = None
        self.explore = False
        self.selected = torch.zeros_like(active)
        self.joined = torch.zeros_like(active)
        self.taken = torch.zeros_like(active)
        self.current_rows = active.clone()
        self.phase = None
        self.reservoir = [None] * max(1, int(capacity))

    def record(self, choice, slot, state, *, word, latches):
        if self.explore:
            at = self.selected & self.current_rows & (self.slot == slot)
            self.taken |= at & choice.valid & (choice.action != self.action)
            return
        eligible = self.current_rows & choice.valid & (choice.alternative_count > 0)
        if not bool(eligible.any()):
            return
        self.count += eligible.long()
        # Algorithm R retains K distinct rounds. A later whole-path draw can
        # choose any remaining member uniformly without replaying a prefix.
        rank = (torch.rand_like(self.count, dtype=torch.float32) * self.count).long()
        rank = torch.where(self.count <= len(self.reservoir), self.count - 1, rank)
        take = eligible & (rank < len(self.reservoir))
        if not bool(take.any()):
            return
        state, latches = snapshot(state, live=self.live), snapshot(tuple(latches), live=self.live)
        prefix = None if self.inverse_candidates is None else self.inverse_candidates.snapshot()
        for i, prior in enumerate(self.reservoir):
            selected = take & (rank == i)
            if not bool(selected.any()):
                continue
            saved = dict(state=state, latches=latches, inverse_prefix=prefix,
                word=torch.zeros_like(self.word) + word,
                slot=torch.zeros_like(self.slot) + slot, action=choice.action,
                valid=selected)
            if prior is not None:
                saved = {key: (None if value is None else select_rows(prior[key], value, selected))
                         for key, value in saved.items()}
                saved['valid'] = prior['valid'] | selected
            self.reservoir[i] = saved
        self.choose(())

    def choose(self, used):
        """Select an unused saved round; its marginal proposal is uniform."""
        from WalkTrials import departure_at
        eligible = torch.stack([torch.zeros_like(self.selected) if item is None else item['valid']
                                for item in self.reservoir], -1)
        for previous in used:
            for i, item in enumerate(self.reservoir):
                if item is not None:
                    eligible[:, i] &= item['slot'] != previous['compose_round']
        chosen = departure_at(eligible)
        self.slot.fill_(-1)
        for i, item in enumerate(self.reservoir):
            if item is None:
                continue
            rows = chosen == i
            for key in ('state', 'latches', 'inverse_prefix', 'word', 'slot', 'action'):
                value, old = item[key], getattr(self, key)
                if value is not None:
                    setattr(self, key, value if old is None else select_rows(old, value, rows))
        return self.slot

    def start(self, draw):
        self.explore = True
        self.selected = draw['compose_round'] >= 0
        self.joined = draw['narrowing'].clone()
        self.taken.zero_()

    def begin_word(self, word, latches):
        if not self.explore:
            return latches
        take = self.selected & self.current_rows & (self.word == word)
        return (latches if self.latches is None else
                select_rows(tuple(latches), self.latches, take))

    def resume(self, slot, state):
        if not self.explore or self.state is None:
            return state
        take = self.selected & self.current_rows & ~self.joined & (self.slot == slot)
        self.joined |= take
        return select_rows(state, self.state, take) if bool(take.any()) else state

    def mask(self, slot):
        if not self.explore:
            return None
        return torch.where(self.selected & (self.slot == slot), self.action, -1)

    def pending(self, word, *, closing, width):
        if not self.explore:
            return False
        if self.phase is not None and closing != (self.phase == 'closing'):
            return False
        return bool((self.selected & self.current_rows & ~self.joined & (self.word == word)
                     & ((self.slot >= 3 * width) == closing)).any())
