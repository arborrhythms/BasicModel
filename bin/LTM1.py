"""LTM1: the single durable Part(x, y) store over native symbol handles.

Argument types belong to the concept bank. The relation has no kind column,
predicate slot, vectors, occurrence addresses, or stored reading. Indexes are
rebuilt from the two references after load or compaction.
"""
import math

import torch
from torch import nn


class LTM1(nn.Module):
    columns = ('refs', 'truth_plus', 'truth_minus', 'access_count', 'surprise', 'when')

    def __init__(self, capacity=8192):
        super().__init__()
        self.capacity = int(capacity)
        self.allocate = None
        self.register_buffer('refs', torch.full((self.capacity, 2), -1, dtype=torch.long))
        self.register_buffer('truth_plus', torch.zeros(self.capacity))
        self.register_buffer('truth_minus', torch.zeros(self.capacity))
        self.register_buffer('access_count', torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer('surprise', torch.full((self.capacity,), -1.))
        self.register_buffer('when', torch.zeros(self.capacity, 4))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        self._index_version = None
        self._pairs, self._parts, self._wholes = {}, {}, {}

    def __len__(self):
        return int(self.count)

    @property
    def revision(self):
        return (id(self.refs), self.refs._version, self.truth_plus._version,
                self.truth_minus._version, self.access_count._version, len(self))

    def _index(self):
        version = (id(self.refs), self.refs._version, len(self))
        if self._index_version == version:
            return
        pairs, parts, wholes = {}, {}, {}
        for row in self.indices().tolist():
            part, whole = self.refs[row].tolist()
            if min(part, whole) <= 0:
                raise ValueError('Part arguments must be positive native symbol handles')
            if (part, whole) in pairs:
                raise ValueError('duplicate Part edge')
            pairs[part, whole] = row
            parts.setdefault(whole, []).append(part)
            wholes.setdefault(part, []).append(whole)
        self._pairs, self._parts, self._wholes = pairs, parts, wholes
        self._index_version = version

    def indices(self):
        return (self.refs[:, 0] > 0).nonzero().flatten()

    def contains(self, symbol):
        return 0 <= int(symbol) < self.capacity and bool((self.refs[int(symbol)] > 0).all())

    def find(self, part, whole):
        self._index()
        return self._pairs.get((int(part), int(whole)))

    def parts(self, whole):
        self._index()
        return tuple(self._parts.get(int(whole), ()))

    def wholes(self, part):
        self._index()
        return tuple(self._wholes.get(int(part), ()))

    @torch.no_grad()
    def add(self, part, whole, *, truth_plus=0., truth_minus=0., when=None,
            surprise=-1., witness=True, symbol=None):
        part, whole = int(part), int(whole)
        if min(part, whole) <= 0:
            raise ValueError('Part arguments must be positive native symbol handles')
        if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1
               for x in (truth_plus, truth_minus)):
            raise ValueError('Part evidence has two independent poles in [0, 1]')
        if not math.isfinite(float(surprise)):
            raise ValueError('Part surprise must be finite')
        if when is not None:
            when = torch.as_tensor(when, device=self.when.device, dtype=self.when.dtype)
            if when.shape != (4,) or not bool(torch.isfinite(when).all()):
                raise ValueError('Part occurrence time requires four finite coordinates')
        row = self.find(part, whole)
        if row is None:
            if symbol is not None:
                row = int(symbol)
            elif self.allocate is None:
                free = (self.refs[:, 0] < 0).nonzero().flatten()
                free = free[free > 0]
                if not len(free):
                    raise OverflowError('LTM2 capacity exhausted; forgetting is required')
                row = int(free[0])
            else:
                row = int(self.allocate())
            if not 0 < row < self.capacity:
                raise OverflowError('LTM2 symbol index exceeds its capacity')
            if self.contains(row):
                raise RuntimeError('a live Part symbol cannot be reassigned')
            self.refs[row] = self.refs.new_tensor((part, whole))
            self.count += 1
        elif symbol is not None and int(symbol) != row:
            raise ValueError('a Part pair already has a different native symbol')
        self.truth_plus[row] = max(float(self.truth_plus[row]), float(truth_plus))
        self.truth_minus[row] = max(float(self.truth_minus[row]), float(truth_minus))
        self.access_count[row] += int(witness)
        if surprise >= 0:
            self.surprise[row] = float(surprise)
        if when is not None:
            self.when[row].copy_(when)
        return row

    @torch.no_grad()
    def retain(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.refs.device)
        count = len(indices)
        if count and (int(indices.min()) < 0 or int(indices.max()) >= self.capacity
                      or len(indices.unique()) != count
                      or not bool((self.refs[indices] > 0).all())):
            raise ValueError('invalid LTM2 retention selection')
        # Row addresses are native symbol handles. Retention never renumbers
        # them: a retained outer edge must keep the same inner-edge argument.
        erase = torch.ones(self.capacity, dtype=torch.bool, device=self.refs.device)
        erase[indices] = False
        for name in self.columns:
            getattr(self, name)[erase] = -1 if name in ('refs', 'surprise') else 0
        self.count.fill_(count)

    def remove(self, part, whole):
        row = self.find(part, whole)
        if row is not None:
            self.retain([i for i in self.indices().tolist() if i != row])

    def remove_symbol(self, symbol):
        indices = self.indices()
        self.retain(indices[(self.refs[indices] != int(symbol)).all(-1) & (indices != int(symbol))])

    def row(self, index):
        if not self.contains(index):
            raise IndexError('unavailable Part edge')
        return {name: getattr(self, name)[int(index)] for name in self.columns}

    def readings(self, index, types):
        """Permitted readings from argument types; context may choose among them."""
        part, whole = self.refs[int(index)].tolist()
        left, right = set(types(part)), set(types(whole))
        readings = []
        if 'situation' in left and 'situation' in right:
            readings.append('conditional')
        if 'situation' in left and 'speech_individual' in right:
            readings.append('attribution')
        if part != whole and 'object' in left and 'word' in right:
            readings.append('definition')
        if 'kind' in left and 'kind' in right:
            readings.append('subsumption')
        # A form may itself be a singleton. Its DEF edge names an object;
        # that does not make the object identical to the word naming it.
        if 'singleton' in right and 'definition' not in readings:
            readings.append('equality')
        return tuple(readings)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        refs = state_dict.get(prefix + 'refs')
        if refs is not None and refs.shape != self.refs.shape:
            if refs.ndim != 2 or refs.shape[1] != 2:
                raise ValueError('LTM2 requires exactly two symbol references')
            self.capacity = int(refs.shape[0])
            for name in self.columns:
                value = state_dict[prefix + name]
                self._buffers[name] = torch.empty_like(value, device=self.refs.device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._index_version = None
        self._index()
        indices = self.indices()
        if len(indices) != len(self):
            raise ValueError('Part count disagrees with its occupied symbol rows')
        values = torch.stack((self.truth_plus[indices], self.truth_minus[indices]))
        if not bool(torch.isfinite(values).all() & (values >= 0).all() & (values <= 1).all()):
            raise ValueError('invalid restored Part evidence')
