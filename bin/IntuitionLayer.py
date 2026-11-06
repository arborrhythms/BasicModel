"""Output-owned sparse Sigma-over-Pi banks with fixed storage and native keys."""
import torch
from torch import nn
from torch.nn import functional as F


class IntuitionLayer(nn.Module):
    def __init__(self, capacity):
        super().__init__()
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('intuitionCapacity must be a positive integer')
        self.capacity = capacity
        self.pi = nn.Parameter(torch.ones(capacity, 1))
        self.sigma = nn.Parameter(torch.ones(capacity, 1))
        self._indices = {'pi': {}, 'sigma': {}}

    def seed(self, rule, literals):
        """Reserve a whole term atomically; never replace a parameter tensor."""
        keys = {'pi': tuple((rule, literal) for literal in literals), 'sigma': (rule,)}
        for bank, values in keys.items():
            if len(self._indices[bank]) + len(set(values) - self._indices[bank].keys()) > self.capacity:
                raise RuntimeError('intuition sparse bank exceeds intuitionCapacity')
        result = {}
        for bank, values in keys.items():
            index = self._indices[bank]
            for key in values:
                if key not in index:
                    index[key] = len(index)
            result[bank] = tuple(index[key] for key in values)
        return result['pi'], result['sigma'][0]

    def weight(self, bank, index):
        parameter = getattr(self, bank)
        ids = torch.tensor(index, device=parameter.device, dtype=torch.long)
        return F.embedding(ids, parameter, sparse=True).squeeze(-1).clamp_min(0)

    def get_extra_state(self):
        from IndependentComponents import _pack_state
        return _pack_state({'capacity': self.capacity,
                'indices': {bank: dict(index) for bank, index in self._indices.items()}})

    def validate_extra_state(self, state):
        from IndependentComponents import _unpack_state
        state = _unpack_state(state)
        if (not isinstance(state, dict) or state.get('capacity') != self.capacity
                or set(state.get('indices', ())) != {'pi', 'sigma'}):
            raise ValueError('intuition checkpoint differs from constructed capacity or banks')
        for index in state['indices'].values():
            if (not isinstance(index, dict) or len(index) > self.capacity
                    or any(type(value) is not int for value in index.values())
                    or set(index.values()) != set(range(len(index)))):
                raise ValueError('intuition checkpoint has invalid sparse addresses')
        return state

    def set_extra_state(self, state):
        state = self.validate_extra_state(state)
        self._indices = {bank: dict(index) for bank, index in state['indices'].items()}


def power(value, weight, *, sigma=False):
    """Unit exponents are exactly identity, including the two cube rails."""
    eps = torch.finfo(value.dtype).eps
    clipped = value.clamp(eps, 1-eps)
    smooth = -torch.expm1(weight * torch.log1p(-clipped)) if sigma else clipped.pow(weight)
    exact = torch.where(value == 0, 0., torch.where(value == 1, 1., smooth))
    # Identity at construction is bit-exact, without discarding its tangent.
    exact = torch.where(weight.detach() == 1, value, exact)
    return smooth + (exact - smooth).detach()
