"""Native WholeSpace witnesses for a composed concept's form centroid.

This transient bank contains existing property rows and input primitive
counts. It owns no run types, lexical read, identity, or durable membership.
"""
from typing import NamedTuple

import torch


def _code_columns(codes, weights):
    """Broadcast a shared or per-stream, fixed-capacity code bank."""
    shape = (*codes.shape[:-2], *((1,) * (weights.ndim-codes.ndim+1)), codes.shape[-1])
    return (code.reshape(shape) for code in codes.unbind(-2))


# AOT can trace this pure tensor operation with its explicit forward-mode rule.
@torch.compiler.allow_in_graph
class _WholeMean(torch.autograd.Function):
    """Convex interpolation forward, derivative of the complete weighted mean."""

    generate_vmap_rule = True

    @staticmethod
    def forward(weights, codes):
        mean = codes.new_zeros(*weights.shape[:-1], codes.shape[-1])
        total = weights.new_zeros(*weights.shape[:-1], 1)
        for index, code in enumerate(_code_columns(codes, weights)):
            mass = weights[..., index:index+1]
            total = total + mass
            fraction = mass / torch.where(total > 0, total, 1.)
            mean = torch.lerp(mean, code, fraction)
        return mean

    @staticmethod
    def setup_context(ctx, inputs, output):
        weights, codes = inputs
        ctx.save_for_backward(weights, codes, output)
        ctx.save_for_forward(weights, codes, output)

    @staticmethod
    def jvp(ctx, weight_tangent, code_tangent):
        weights, codes, mean = ctx.saved_tensors
        total = weights.sum(-1, keepdim=True)
        divisor = torch.where(total > 0, total, 1.)
        tangent = torch.zeros_like(mean)
        tangents = None if code_tangent is None else tuple(_code_columns(code_tangent, weights))
        for index, code in enumerate(_code_columns(codes, weights)):
            if weight_tangent is not None:
                tangent = tangent + (code - mean) * weight_tangent[..., index:index+1] / divisor
            if code_tangent is not None:
                tangent = tangent + (weights[..., index:index+1] / divisor) * tangents[index]
        return tangent

    @staticmethod
    def backward(ctx, gradient):
        weights, codes, mean = ctx.saved_tensors
        total = weights.sum(-1, keepdim=True)
        divisor = torch.where(total > 0, total, 1.)
        # A zero prefix in the interpolation is not a singularity of a
        # nonempty whole. Use its final mean and mass in both pullbacks.
        # For an entirely empty whole, the zero forward and unit divisor
        # retain the existing local convention for a first membership.
        weight_grad = torch.zeros_like(weights)
        if codes.shape[-2]:
            numerators = [((code - mean) * gradient).sum(-1) for code in _code_columns(codes, weights)]
            weight_grad = torch.stack(numerators, dim=-1) / divisor
        if codes.ndim == weights.ndim + 1:
            code_grad = (weights / divisor)[..., None] * gradient.unsqueeze(-2)
        elif codes.ndim == 2:
            code_grad = torch.einsum('...r,...d->rd', weights / divisor, gradient)
        else:
            code_grad = torch.einsum('bnr,bnd->brd',
                (weights / divisor).reshape(codes.shape[0], -1, codes.shape[-2]),
                gradient.reshape(codes.shape[0], -1, codes.shape[-1]))
        return weight_grad, code_grad


def convex_mean(weights, codes):
    """Evidence-weighted interpolation preserving common input endpoints."""
    torch._assert_async((torch.isfinite(weights) & (weights >= 0)).all(),
                        'whole centroid weights must be finite and nonnegative')
    return _WholeMean.apply(weights, codes), weights.sum(-1, keepdim=True)


def weighted_mean(weights, codes):
    """Average native property codes; common cube rails remain exact."""
    from Layers import assert_cube
    assert_cube(codes)
    return convex_mean(weights, codes)


class WholeRead(NamedTuple):
    """A rail-exact mean and its directly weighted presence carrier.

    The carrier owns the consumer's derivative. Multiplying a differentiated
    normalized mean by its mass would retain a reciprocal-small-mass
    intermediate in its pullback.
    """
    mean: torch.Tensor
    total: torch.Tensor
    carrier: torch.Tensor


def presence_read(weights, codes):
    # The exact mean supplies only the lerp's forward rails. Its normalized
    # derivative is not part of a presence read, in backward or forward AD.
    mean, _ = weighted_mean(weights.detach(), codes.detach())
    total = weights.sum(-1, keepdim=True)
    divisor = torch.where(total >= 1, total, torch.ones_like(total))
    carrier = (((weights / divisor)[..., None] * codes).sum(-2)
        if codes.ndim == weights.ndim + 1 else
        (weights / divisor) @ codes if codes.ndim == 2 else
        torch.einsum('bnr,brd->bnd',
            (weights / divisor).reshape(codes.shape[0], -1, codes.shape[-2]), codes)
        .reshape(*weights.shape[:-1], codes.shape[-1]))
    return WholeRead(mean, total, carrier)


def as_whole(whole):
    """Accept literal mean/mass witnesses as well as native presence reads."""
    if len(whole) == 3:
        return WholeRead(*whole)
    mean, total = whole
    presence = torch.where(total < 1, total, torch.ones_like(total))
    return WholeRead(mean, total, mean * presence)


def pad_whole(whole, width):
    whole = as_whole(whole)
    padding = (0, width-whole.mean.shape[-1])
    return WholeRead(torch.nn.functional.pad(whole.mean, padding), whole.total,
                     torch.nn.functional.pad(whole.carrier, padding))


class SpanWholes(NamedTuple):
    prefix: torch.Tensor
    properties: torch.Tensor
    codes: torch.Tensor

    def for_stream(self, stream):
        return self._replace(prefix=self.prefix[stream:stream+1])

    def _weights(self, spans):
        lo, hi = spans.unbind(-1)
        width = self.prefix.shape[-1]
        lo, hi = lo.clamp(0, self.prefix.shape[1]-1), hi.clamp(0, self.prefix.shape[1]-1)
        counts = (self.prefix.gather(1, hi[..., None].expand(*hi.shape, width))
                  - self.prefix.gather(1, lo[..., None].expand(*lo.shape, width)))
        support = counts > 0
        weights = torch.where(support[..., None, :], self.properties, 1.).amin(-1)
        weights = weights * (support.any(-1) & (spans[..., 0] >= 0) & (hi > lo))[..., None]
        return weights

    def read(self, spans):
        """Read the mean of properties active over a half-open input span."""
        return weighted_mean(self._weights(spans), self.codes)

    def read_presence(self, spans):
        return presence_read(self._weights(spans), self.codes)


class BasicSpanWholes(NamedTuple):
    prefix: torch.Tensor
    codes: torch.Tensor

    def for_stream(self, stream):
        return self._replace(prefix=self.prefix[stream:stream+1],
                             codes=self.codes[stream:stream+1])

    def _weights(self, spans):
        if (self.codes.ndim != 3 or self.codes.shape[0] != self.prefix.shape[0]
                or self.codes.shape[-2] != self.prefix.shape[-1]):
            raise ValueError('basic span prefix and codes must select the same streams and words')
        lo, hi = spans.unbind(-1)
        width = self.prefix.shape[-1]
        lo, hi = lo.clamp(0, self.prefix.shape[1]-1), hi.clamp(0, self.prefix.shape[1]-1)
        weights = (self.prefix.gather(1, hi[..., None].expand(*hi.shape, width))
                   - self.prefix.gather(1, lo[..., None].expand(*lo.shape, width)))
        return weights.clamp_min(0) * ((spans[..., 0] >= 0) & (hi > lo))[..., None]

    def read(self, spans):
        return weighted_mean(self._weights(spans), self.codes)

    def read_presence(self, spans):
        return presence_read(self._weights(spans), self.codes)


class CandidateWholes(NamedTuple):
    forms: torch.Tensor
    support: torch.Tensor
    valid: torch.Tensor
    properties: torch.Tensor
    codes: torch.Tensor

    def _weights(self, *parts):
        """Hypothesize wholes from dictionary leaves, never the input's span.

        An unspelled candidate has no known primitive set here. It therefore
        cannot supply a WholeSpace witness to the free inverse. That inverse
        remains approximate; it receives neither a target nor a saved tree.
        """
        found, complete = [], []
        for value in parts:
            shape = value.shape[:-1]
            form = value[..., :self.forms.shape[-1]].flatten(1, -2) if value.ndim > 2 else value[:, None, :self.forms.shape[-1]]
            error = (form.square().sum(-1, keepdim=True) + self.forms.square().sum(-1)[:, None]
                     - 2 * form @ self.forms.transpose(-1, -2)).clamp_min(0)
            error = error.masked_fill(~self.valid[:, None], torch.inf)
            best, index = error.min(-1)
            tolerance = (8 * torch.finfo(form.dtype).eps) * form.square().sum(-1).clamp_min(1)
            complete.append((best <= tolerance).reshape(shape))
            found.append(self.support.gather(1, index[..., None].expand(*index.shape, 256)).reshape(*shape, 256))
        support = torch.stack(found).any(0)
        available = torch.stack(complete).all(0) & support.any(-1)
        # Property rows are independent minima. Bound the temporary's property
        # axis while preserving the exact dense read at every coordinate.
        properties = self.properties.shape[-2]
        weights = (torch.cat([
            torch.where(support[..., None, :], self.properties[..., start:start + 256, :], 1.).amin(-1)
            for start in range(0, properties, 256)], -1) if properties else
            torch.where(support[..., None, :], self.properties, 1.).amin(-1))
        weights = weights * available[..., None]
        return weights

    def read(self, *parts):
        return weighted_mean(self._weights(*parts), self.codes)

    def read_presence(self, *parts):
        return presence_read(self._weights(*parts), self.codes)

    def for_stream(self, stream):
        return self._replace(forms=self.forms[stream:stream+1], support=self.support[stream:stream+1],
                             valid=self.valid[stream:stream+1])


class BasicCandidateWholes(NamedTuple):
    forms: torch.Tensor
    weights: torch.Tensor
    valid: torch.Tensor
    codes: torch.Tensor

    def _selected(self, *parts):
        weights, codes, available = [], [], []
        for value in parts:
            shape = value.shape[:-1]
            form = value[..., :self.forms.shape[-1]].reshape(value.shape[0], -1, self.forms.shape[-1])
            error = (form.square().sum(-1, keepdim=True) + self.forms.square().sum(-1)[:, None]
                     - 2 * form @ self.forms.transpose(-1, -2)).clamp_min(0)
            best, index = error.masked_fill(~self.valid[:, None], torch.inf).min(-1)
            tolerance = 8 * torch.finfo(form.dtype).eps * form.square().sum(-1).clamp_min(1)
            available.append((best <= tolerance).reshape(shape))
            weights.append(self.weights.gather(1, index).reshape(shape))
            codes.append(self.codes.gather(1, index[..., None].expand(*index.shape, self.codes.shape[-1]))
                         .reshape(*shape, self.codes.shape[-1]))
        return (torch.stack(weights, -1) * torch.stack(available).all(0)[..., None],
                torch.stack(codes, -2))

    def read(self, *parts):
        return weighted_mean(*self._selected(*parts))

    def read_presence(self, *parts):
        return presence_read(*self._selected(*parts))

    def for_stream(self, stream):
        return self._replace(forms=self.forms[stream:stream+1], weights=self.weights[stream:stream+1],
                             valid=self.valid[stream:stream+1], codes=self.codes[stream:stream+1])


class PreparedCandidateWholes:
    """An eager inverse walk's invariant, detached dictionary-whole read."""
    def __init__(self, bank, dictionary):
        self.bank, self.dictionary = bank, dictionary

    def __getattr__(self, name):
        return getattr(self.bank, name)

    def read(self, *parts):
        if len(parts) == 1 and parts[0] is self.bank.forms:
            return self.dictionary.mean, self.dictionary.total
        return self.bank.read(*parts)

    def read_presence(self, *parts):
        if len(parts) == 1 and parts[0] is self.bank.forms:
            return self.dictionary
        return self.bank.read_presence(*parts)

    def for_stream(self, stream):
        return type(self)(self.bank.for_stream(stream),
                          WholeRead(*(value[stream:stream+1] for value in self.dictionary)))


def prepare_inverse(bank):
    """Reuse dictionary-only work; argument-dependent reads and graphs stay live."""
    if bank is None or any(value.requires_grad for value in bank):
        return bank
    return PreparedCandidateWholes(bank, bank.read_presence(bank.forms))


def candidates(whole_space, bank, *, owner=None):
    """Snapshot the same native properties for the free dictionary search."""
    basis = getattr(getattr(whole_space, 'subspace', None), 'what', None)
    properties = getattr(basis, 'primitive_properties', None)
    if properties is None or bank.meaning_start is None or whole_space.held_at_zero:
        return None
    support = getattr(bank, 'primitive_support', None)
    if support is None:
        from WordSurfaces import primitive_support
        support = primitive_support(bank.bytes, bank.byte_valid)
    if owner is not None:
        basic = owner.basic_percepts
        rows = bank.rows.new_tensor([[basic.words.get(owner.concept_id_at_row(int(row)), -1)
            for row in stream] for stream in bank.rows.cpu().tolist()])
        weights, codes = basic.word_weights(support.to(bank.codes), rows)
        return BasicCandidateWholes(bank.codes[..., :bank.meaning_start].detach(),
            weights.detach(), bank.terminal_valid, codes.detach())
    return CandidateWholes(bank.codes[..., :bank.meaning_start].detach(), support,
        bank.terminal_valid, properties.coefficients().detach(), basis.getW().detach())


def stage(whole_space, *, basic=None, word_rows=None):
    """Read the existing WholeSpace table once outside the compose loop."""
    counts = getattr(whole_space, '_staged_word_primitive_counts', None)
    basis = getattr(getattr(whole_space, 'subspace', None), 'what', None)
    properties = getattr(basis, 'primitive_properties', None)
    if not torch.is_tensor(counts) or properties is None or whole_space.held_at_zero:
        return None
    counts = counts.sum(-2)
    if basic is not None:
        weights, codes = basic.word_weights(counts, word_rows)
        weights = torch.diag_embed(weights)
        prefix = torch.cat((weights.new_zeros(weights.shape[0], 1, weights.shape[-1]), weights.cumsum(1)), 1)
        return BasicSpanWholes(prefix, codes)
    prefix = torch.cat((counts.new_zeros(counts.shape[0], 1, counts.shape[-1]), counts.cumsum(1)), 1)
    return SpanWholes(prefix, properties.coefficients(), basis.getW())


def selected(context, arity, like, rule):
    """Expand one span witness over a candidate's reference alternatives."""
    if context is None:
        return None
    *whole, orders = context[arity]
    mean, count, carrier = as_whole(whole)
    from ReferenceContext import denoted_order
    order = context.get((arity, getattr(rule, 'predicate_identity', None)))
    if order is None:
        order = denoted_order(rule, orders, context.get('references', {}).get(arity, ()), None)
    count = count * (order == 0)[..., None]
    if mean.ndim == like.ndim and mean.shape[1] != like.shape[1]:
        copies = like.shape[1] // mean.shape[1]
        mean, count = mean.repeat_interleave(copies, 1), count.repeat_interleave(copies, 1)
        carrier = carrier.repeat_interleave(copies, 1)
    return WholeRead(mean.to(like), count.to(like), carrier.to(like))


def capture(program, bank, batch, positions, language):
    """Retain the same span witnesses for this open derivation's inverse."""
    from dataclasses import replace
    from ReferenceContext import denoted_order
    stack, stamps, spans, counts, active = [], [], [], [], []
    for index, (kind, operation, word) in enumerate(program.actions.tolist()):
        pair = (1, 1)
        if kind == 0:
            stack.append((int(positions[word]),))
            stamps.append(-1 if program.leaf_orders is None else int(program.leaf_orders[word]))
            span = (-1, -1)
            order = -1
        elif kind == 3:
            stack[-2:] = reversed(stack[-2:])
            stamps[-2:] = reversed(stamps[-2:])
            span = (-1, -1)
            order = -1
        elif kind == 2 or (kind == 1 and
                language._compose_binary_rules[operation].method_name == 'bind'):
            field_bind = kind == 1
            at = len(stack)-1-(0 if field_bind else max(0, word))
            part = stack[at]
            pair = (len(part), 0)
            span = (min(part), max(part)+1)
            order = (stamps[at] if field_bind else denoted_order(
                language._compose_unary_rules[operation], (stamps[at],), (), None))
            if program.timeline_actions is not None:
                order = int(program.timeline_actions[index, 0])
            stamps[at] = order
        else:
            right, left = stack.pop(), stack.pop()
            part = left+right
            stack.append(part)
            right_order, left_order = stamps.pop(), stamps.pop()
            order = denoted_order(language._compose_binary_rules[operation], (left_order, right_order), (), None)
            if program.timeline_actions is not None:
                order = int(program.timeline_actions[index, 0])
            stamps.append(order)
            pair = (len(left), len(right))
            span = (min(part), max(part)+1)
        spans.append(span)
        counts.append(pair)
        active.append(order == 0)
    local = bank.for_stream(batch)
    values, number, carrier = local.read_presence(program.actions.new_tensor(spans)[None])
    return replace(program, operand_counts=program.leaves.new_tensor(counts),
                   form_wholes=values[0], form_whole_counts=number[0] *
                       number.new_tensor(active)[:, None], form_whole_presence=carrier[0])


def inverse_context(program, index):
    counts = getattr(program, 'operand_counts', None)
    whole = getattr(program, 'form_wholes', None)
    carrier = getattr(program, 'form_whole_presence', None)
    witness = None if whole is None else (whole[index:index+1],
                                          program.form_whole_counts[index:index+1])
    if witness is not None and carrier is not None:
        witness = WholeRead(*witness, carrier[index:index+1])
    return dict(counts=(1, 1) if counts is None else counts[index].unbind(),
                whole=witness)


def unary(parent, bank):
    """Undo a free unary's form center from candidate words, without a stamp.

    Compare the plain and centered form equations over the same dictionary
    as binary search. Unknown constituent spellings supply no whole witness.
    """
    if bank is None:
        return parent
    from MeaningKernels import centroid, uncenter
    width = bank.forms.shape[-1]
    whole = pad_whole(bank.read_presence(bank.forms), width)
    form = parent[..., :width]
    plain = (bank.forms-form[:, None]).square().mean(-1).masked_fill(~bank.valid, torch.inf)
    centered = (centroid((bank.forms,), whole=whole)-form[:, None]).square().mean(-1)
    recovered = uncenter(form[:, None], whole)
    in_chart = ((recovered >= 0) & (recovered <= 1) & torch.isfinite(recovered)).all(-1)
    centered = centered.masked_fill(~bank.valid | (whole.total[..., 0] <= 0) | ~in_chart, torch.inf)
    cost, index = centered.min(-1)
    use = cost < plain.min(-1).values
    decoded = recovered.gather(1, index[:, None, None].expand(-1, 1, width))[:, 0]
    decoded = torch.where(use[:, None], decoded, form)
    return torch.cat((decoded, parent[..., width:]), -1)


def fit(parent_form, left, right, bank=None):
    """Fit a free pair's count ratio, without a parent stamp or input trace.

    The unknown result may be order zero or higher. Compare the two form
    equations using the candidates' native wholes; no extra parse action,
    target, or saved correspondence enters the candidate search.
    """
    width = parent_form.shape[-1]
    left, right = left[..., :width], right[..., :width]
    delta = left-right
    length = delta.square().sum(-1)

    def ratio(target):
        result = ((target-right)*delta).sum(-1) / torch.where(length > 0, length, 1.)
        return torch.where(length > 0, result, .5).clamp(0, 1)

    plain = ratio(parent_form)
    if bank is None:
        return plain, None
    from MeaningKernels import centroid, uncenter
    whole = pad_whole(bank.read_presence(left, right), width)
    centered = ratio(uncenter(parent_form, whole))
    simple_error = (right+plain[..., None]*delta-parent_form).square().mean(-1)
    whole_error = (centroid((right+centered[..., None]*delta,), whole=whole)-parent_form).square().mean(-1)
    simple_error = simple_error.masked_fill((plain <= 0) | (plain >= 1), torch.inf)
    use = ((whole.total[..., 0] > 0) & (centered > 0) & (centered < 1)
           & (whole_error < simple_error))
    return torch.where(use, centered, plain), whole._replace(total=whole.total*use[..., None])
