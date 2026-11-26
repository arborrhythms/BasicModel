"""A reservoir snapshot, joined independently by each batch row."""
import torch
from SentenceCompose import select_paths, select_rows


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
        if not torch.compiler.is_compiling() and not bool(eligible.any()):
            return
        self.count += eligible.long()
        # Algorithm R retains K distinct rounds. A later whole-path draw can
        # choose any remaining member uniformly without replaying a prefix.
        rank = (torch.rand_like(self.count, dtype=torch.float32) * self.count).long()
        rank = torch.where(self.count <= len(self.reservoir), self.count - 1, rank)
        places = torch.arange(len(self.reservoir), device=rank.device)
        take = eligible[:, None] & (rank[:, None] == places)
        changed = None
        if not torch.compiler.is_compiling():
            changed = take.any(0).tolist()
            if not any(changed):
                return
        state, latches = snapshot(state, live=self.live), snapshot(tuple(latches), live=self.live)
        prefix = None if self.inverse_candidates is None else self.inverse_candidates.snapshot()
        for i, prior in enumerate(self.reservoir):
            if changed is not None and not changed[i]:
                continue
            selected = take[:, i]
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
        slots = torch.stack([torch.full_like(self.slot, -1) if item is None else item['slot']
                             for item in self.reservoir], -1)
        eligible = torch.stack([torch.zeros_like(self.selected) if item is None else item['valid']
                                for item in self.reservoir], -1)
        if used:
            previous = torch.stack([draw['compose_round'] for draw in used], -1)
            eligible &= (slots[:, :, None] != previous[:, None, :]).all(-1)
        chosen = departure_at(eligible)
        self.slot = torch.where(chosen >= 0, slots.gather(1, chosen.clamp_min(0)[:, None])[:, 0], -1)
        occupied = [i for i, item in enumerate(self.reservoir) if item is not None]
        if not occupied:
            return self.slot
        for key in ('state', 'latches', 'inverse_prefix', 'word', 'action'):
            old = getattr(self, key)
            available = [i for i in occupied if self.reservoir[i][key] is not None]
            if not available:
                continue
            first = self.reservoir[available[0]][key]
            if len(available) == 1:
                value = first if old is None else select_rows(old, first, chosen == available[0])
            else:
                baseline = first if old is None else old
                values = [baseline] + [baseline if item is None or item[key] is None else item[key]
                                       for item in self.reservoir]
                value = select_paths(values, chosen + 1)
            setattr(self, key, value)
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

    def round_visits(self, word, slots, *, closing, width):
        """Read the retained departures for one phase in one transfer.

        A stream joins at its first matching round. Subsequent rounds no
        longer wait for it, even when two streams have different close slots.
        A policy graph belongs to every round containing a selected slot.
        """
        selected = self.selected[:, None] & (self.slot[:, None] == slots)
        waiting = self.selected & self.current_rows & ~self.joined
        joins = selected & waiting[:, None]
        if self.state is None:
            joins = torch.zeros_like(joins)
        visited = joins.long().cumsum(-1) > 0
        before = torch.nn.functional.pad(visited, (1, 0))[:, :-1]
        joins = joins & ~before
        pending = waiting & (self.word == word) & ((self.slot >= 3 * width) == closing)
        if self.phase is not None and closing != (self.phase == 'closing'):
            pending = torch.zeros_like(pending)
        waiting_here = (pending[:, None] & ~before).any(0)
        return torch.stack((waiting_here, joins.any(0), selected.any(0)), -1)

    def resume(self, slot, state, *, joining=None):
        if not self.explore or self.state is None or joining is False:
            return state
        take = self.selected & self.current_rows & ~self.joined & (self.slot == slot)
        self.joined |= take
        return select_rows(state, self.state, take)

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
