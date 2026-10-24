"""Lane-preserving grammar maps and identity actions on the meaning block.

The form carrier has no parameters. Counts belong to a composition, not to
a word or a conceptual seat. Numerical methods here never inspect memory.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def centroid(values, counts=None, *, whole=None):
    if len(values) == 1:
        parts = values[0]
    elif counts is None:
        counts = tuple(torch.ones_like(value[..., :1]) for value in values)
        parts = sum(values) / len(values)
    else:
        counts = tuple(torch.as_tensor(n, device=value.device, dtype=value.dtype)
                       for value, n in zip(values, counts))
        counts = tuple(n[..., None] if n.ndim == value.ndim-1 else n
                       for n, value in zip(counts, values))
        parts = sum(value * n for value, n in zip(values, counts)) / sum(counts).clamp_min(1)
    if whole is None:
        return parts
    mean, count = whole
    return torch.where(count > 0, (parts + mean) * .5, parts)


def split_centroid(parent, known, counts, *, known_right=True, whole=None):
    if whole is not None:
        mean, count = whole
        parent = torch.where(count > 0, parent * 2 - mean, parent)
    left, right = (torch.as_tensor(n, device=parent.device, dtype=parent.dtype) for n in counts)
    if left.ndim == parent.ndim-1:
        left, right = left[..., None], right[..., None]
    a, b = (left, right) if known_right else (right, left)
    return (parent * (a + b) - known * b) / a.clamp_min(torch.finfo(parent.dtype).tiny)


def exact_initial(value, transformed, neutral):
    """Exact identity value with the map's derivative at its neutral element."""
    return torch.where(neutral, transformed + (value - transformed).detach(), transformed)


class LaneKernel(nn.Module):
    """One chart-affine kernel, shared by two independent evidence lanes.

    The head's diagonal is positive, so a known context has an exact inverse.
    For a determiner this is a multiplicative (pi) fold in chart coordinates;
    the marker begins at the multiplicative identity.
    """
    def __init__(self, width):
        super().__init__()
        if width <= 0 or width % 2:
            raise ValueError('a meaning kernel requires two equal evidence lanes')
        size = width // 2
        self.log_scale = nn.Parameter(torch.zeros(size))
        self.context = nn.Parameter(torch.zeros(size, size))
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, head, other=None, *, inverse=False):
        from Layers import cube_chart, cube_unchart
        lanes = head.unflatten(-1, (2, -1))
        scale = self.log_scale.clamp(-4, 4).exp()
        shift = self.bias
        if other is not None:
            shift = shift + F.linear(cube_chart(other.unflatten(-1, (2, -1))), self.context)
        result = cube_unchart((cube_chart(lanes) - shift) / scale if inverse else
                             cube_chart(lanes) * scale + shift).flatten(-2)
        neutral = (self.log_scale == 0).all() & (self.bias == 0).all() & (self.context == 0).all()
        return exact_initial(head, result, neutral)


def householder(vectors):
    """Orthogonal product; a zero reflection vector is the identity."""
    size = vectors.shape[-1]
    eye = torch.eye(size, dtype=vectors.dtype, device=vectors.device)
    result = eye.expand(*vectors.shape[:-2], size, size)
    for vector in vectors.unbind(-2):
        norm = vector.square().sum(-1, keepdim=True)
        projected = (result @ vector[..., None])[..., 0]
        result = result - 2 * projected[..., :, None] * vector[..., None, :] / torch.where(norm > 0, norm, 1.)[..., None]
    return result


class VerbGroups(nn.Module):
    """Learned bases with host-owned fit witnesses and verb assignments."""
    def __init__(self, size, capacity=24, reflections=4, fit=.1):
        super().__init__()
        if capacity < 1 or reflections < 2 or reflections % 2 or not math.isfinite(fit) or fit < 0:
            raise ValueError('verb bases require positive capacity, paired reflections and nonnegative fit')
        vectors = torch.zeros(capacity, reflections, size)
        for i in range(reflections):
            vectors[:, i, (i // 2) % size] = 1
        self.vectors = nn.Parameter(vectors)
        self.capacity, self.fit = int(capacity), float(fit)
        self.assignments, self.witnesses, self.reports = {}, {}, []
        self.used = 0

    def forward(self, group):
        eye = torch.eye(self.vectors.shape[-1], device=self.vectors.device, dtype=self.vectors.dtype)
        selected = householder(self.vectors[group.clamp(0, self.capacity-1)])
        return torch.where((group >= 0)[..., None, None], selected, eye)

    def observe(self, verb, occurrence, residuals, current):
        """A distinct third failed fit chooses a retained basis or opens one."""
        residuals = tuple(float(x) for x in residuals)
        if (len(residuals) != self.capacity or not math.isfinite(current) or current < 0
                or not all(math.isfinite(x) and x >= 0 for x in residuals)):
            raise ValueError('one finite residual is required for each verb basis')
        report = dict(verb=verb, occurrence=occurrence, residual=float(current),
                      group=self.assignments.get(verb, -1), capacity=False)
        self.reports.append(report)
        if current <= self.fit:
            self.witnesses.pop(verb, None)
            return report
        witnesses = self.witnesses.setdefault(verb, [])
        if occurrence not in witnesses:
            witnesses.append(occurrence)
        if len(witnesses) < 3:
            return report
        best = min(range(self.used), key=lambda i: residuals[i]) if self.used else None
        if best is None or residuals[best] > self.fit:
            if self.used < self.capacity:
                best, self.used = self.used, self.used + 1
            else:
                report['capacity'] = True
        self.assignments[verb] = report['group'] = best
        self.witnesses.pop(verb, None)
        return report

    def get_extra_state(self):
        from IndependentComponents import _pack_state
        return _pack_state(dict(version=1, used=self.used, assignments=dict(self.assignments),
                    witnesses={key: tuple(value) for key, value in self.witnesses.items()}))

    def prepare_checkpoint(self, state, prefix):
        # The first swap freeze wrote the host dictionary directly. Keep
        # that readable while honoring the model's tensor-only envelope.
        from IndependentComponents import _pack_state
        key = prefix + '_extra_state'
        if key in state and not torch.is_tensor(state[key]):
            state[key] = _pack_state(state[key])

    def set_extra_state(self, state):
        from IndependentComponents import _unpack_state
        state = _unpack_state(state)
        if (state.get('version') != 1 or type(state.get('used')) is not int
                or not 0 <= state['used'] <= self.capacity):
            raise ValueError('invalid verb-group checkpoint')
        assignments = dict(state['assignments'])
        if any(type(v) is not int or not 0 <= v < state['used'] for v in assignments.values()):
            raise ValueError('invalid verb-group assignment')
        witnesses = {key: list(value) for key, value in state['witnesses'].items()}
        if any(len(value) >= 3 or len(value) != len(set(value)) for value in witnesses.values()):
            raise ValueError('invalid verb-group recurrence witnesses')
        self.used, self.assignments, self.witnesses = state['used'], assignments, witnesses
        self.reports = []


class MeaningAction(nn.Module):
    """A diagonal group action in an object or group basis, on each lane."""
    def __init__(self, width, *, capacity=24, reflections=4, fit=.1, modifier=False):
        super().__init__()
        if width <= 0 or width % 2:
            raise ValueError('a meaning action requires two equal lanes')
        size = width // 2
        self.readout = nn.Parameter(torch.zeros(size, width))
        self.groups = VerbGroups(size, 1 if modifier else capacity, reflections, fit)
        # Paired readouts initialize Q(object) to I without consuming RNG.
        vectors = torch.zeros(reflections, size, width)
        for i in range(reflections):
            vectors[i, (i // 2) % size, (i // 2) % size] = 1
            vectors[i, (i // 2) % size, size + (i // 2) % size] = 1
        self.object_vectors = nn.Parameter(vectors)
        self.modifier = modifier

    def spectrum(self, verb):
        raw = F.linear(verb, self.readout)
        sparse = raw.sign() * (raw.abs() - .1).clamp_min(0)
        return (raw + (sparse-raw).detach()).clamp(-4, 4)

    def basis(self, value, *, object_value=None, group=None, object_present=None):
        size = self.readout.shape[0]
        eye = torch.eye(size, dtype=value.dtype, device=value.device)
        q = eye if group is None else self.groups(group)
        if self.modifier:
            q = self.groups(torch.zeros((), dtype=torch.long, device=value.device))
        support = torch.ones_like(value[..., :size], dtype=torch.bool)
        if object_value is not None:
            support = object_value.unflatten(-1, (2, size)).ne(0).any(-2)
            vectors = torch.einsum('...d,knd->...kn', object_value, self.object_vectors)
            object_q = householder(vectors * support[..., None, :])
            if object_present is None:
                q = object_q
            else:
                q = torch.where(object_present[..., None, None], object_q, q)
                support = torch.where(object_present[..., None], support, True)
        return q, support

    def forward(self, value, verb, *, object_value=None, group=None,
                object_present=None, inverse=False):
        from Layers import cube_chart, cube_unchart
        # Unknown groups and absent objects have exactly the identity basis.
        # Eager evaluation need not build a dense matrix for that masked path.
        # The compiled path retains the tensor-only general expression.
        neutral_basis = object_value is None and group is None
        if not torch.compiler.is_compiling() and not self.modifier:
            neutral_basis = ((object_value is None or
                (object_present is not None and not bool(object_present.any())))
                and (group is None or not bool((group >= 0).any())))
        if not self.modifier and neutral_basis:
            w = self.spectrum(verb)
            lanes = value.unflatten(-1, (2, -1))
            chart = cube_chart(lanes, bounded=False)
            result = torch.where(lanes != 0, cube_unchart(chart * ((-w if inverse else w).exp()[..., None, :])), 0.).flatten(-2)
            return exact_initial(value, result, (w == 0).all(-1, keepdim=True))
        q, allowed = self.basis(value, object_value=object_value, group=group,
                                object_present=object_present)
        w = self.spectrum(verb) * allowed
        size = w.shape[-1]
        lanes = value.unflatten(-1, (2, size))
        present = lanes.ne(0)
        chart = torch.where(present, cube_chart(lanes, bounded=False), 0.)
        if inverse:
            # SPD restriction preserves invertibility on each fixed support.
            action = (q * w.exp()[..., None, :]) @ q.transpose(-1, -2)
            mask = present[..., :, None] & present[..., None, :]
            restricted = torch.where(mask, action[..., None, :, :], 0.) + torch.diag_embed((~present).to(value))
            changed = torch.linalg.solve(restricted, chart[..., None])[..., 0]
        else:
            projected = (q.transpose(-1, -2)[..., None, :, :] @ chart[..., None])[..., 0]
            changed = (q[..., None, :, :] @ (projected*w.exp()[..., None, :])[..., None])[..., 0]
        result = torch.where(present, cube_unchart(changed), 0.).flatten(-2)
        result = torch.where(allowed[..., None, :].expand_as(lanes).flatten(-2), result, value)
        return exact_initial(value, result, (w == 0).all(-1, keepdim=True))

    def teacher(self, before, observed, verb, **basis):
        from Layers import cube_chart
        before, observed, verb = before.detach(), observed.detach(), verb.detach()
        predicted = self(before, verb, **basis)
        defined = before.ne(0) & observed.ne(0)
        error = torch.where(defined, (cube_chart(predicted)-cube_chart(observed)).square(), 0.)
        return error.sum(-1) / defined.sum(-1).clamp_min(1)
