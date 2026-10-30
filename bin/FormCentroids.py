"""Native WholeSpace witnesses for a composed concept's form centroid.

This transient bank contains existing property rows and input primitive
counts. It owns no run types, lexical read, identity, or durable membership.
"""
from typing import NamedTuple

import torch


def weighted_mean(weights, codes):
    """A convex mean without different dot/sum rounding at the cube rails.

    WholeSpace has a small fixed property inventory. Successive convex
    interpolation keeps common zero/one coordinates exact in float32,
    including on devices without float64, and preserves both gradients.
    """
    from Layers import assert_cube
    assert_cube(codes)
    torch._assert_async((torch.isfinite(weights) & (weights >= 0)).all(),
                        'whole centroid weights must be finite and nonnegative')
    mean = codes.new_zeros(*weights.shape[:-1], codes.shape[-1])
    total = weights.new_zeros(*weights.shape[:-1], 1)
    for index in range(codes.shape[0]):
        mass = weights[..., index:index+1]
        total = total + mass
        fraction = mass / torch.where(total > 0, total, 1.)
        mean = torch.lerp(mean, codes[index], fraction)
    return mean, weights.sum(-1, keepdim=True)


class SpanWholes(NamedTuple):
    prefix: torch.Tensor
    properties: torch.Tensor
    codes: torch.Tensor

    def read(self, spans):
        """Read properties active over this half-open span of input words."""
        lo, hi = spans.unbind(-1)
        width = self.prefix.shape[-1]
        lo, hi = lo.clamp(0, self.prefix.shape[1]-1), hi.clamp(0, self.prefix.shape[1]-1)
        counts = (self.prefix.gather(1, hi[..., None].expand(*hi.shape, width))
                  - self.prefix.gather(1, lo[..., None].expand(*lo.shape, width)))
        support = counts > 0
        weights = torch.where(support[..., None, :], self.properties, 1.).amin(-1)
        weights = weights * (support.any(-1) & (spans[..., 0] >= 0) & (hi > lo))[..., None]
        return weighted_mean(weights, self.codes)


class CandidateWholes(NamedTuple):
    forms: torch.Tensor
    support: torch.Tensor
    valid: torch.Tensor
    properties: torch.Tensor
    codes: torch.Tensor

    def read(self, *parts):
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
        weights = torch.where(support[..., None, :], self.properties, 1.).amin(-1)
        weights = weights * available[..., None]
        return weighted_mean(weights, self.codes)

    def for_stream(self, stream):
        return self._replace(forms=self.forms[stream:stream+1], support=self.support[stream:stream+1],
                             valid=self.valid[stream:stream+1])


class PreparedCandidateWholes:
    """An eager inverse walk's invariant, detached dictionary-whole read."""
    def __init__(self, bank, dictionary):
        self.bank, self.dictionary = bank, dictionary

    def __getattr__(self, name):
        return getattr(self.bank, name)

    def read(self, *parts):
        if len(parts) == 1 and parts[0] is self.bank.forms:
            return self.dictionary
        return self.bank.read(*parts)

    def for_stream(self, stream):
        return type(self)(self.bank.for_stream(stream),
                          tuple(value[stream:stream+1] for value in self.dictionary))


def prepare_inverse(bank):
    """Reuse dictionary-only work; argument-dependent reads and graphs stay live."""
    if bank is None or any(value.requires_grad for value in bank):
        return bank
    return PreparedCandidateWholes(bank, bank.read(bank.forms))


def candidates(whole_space, bank):
    """Snapshot the same native properties for the free dictionary search."""
    basis = getattr(getattr(whole_space, 'subspace', None), 'what', None)
    properties = getattr(basis, 'primitive_properties', None)
    if properties is None or bank.meaning_start is None or whole_space.held_at_zero:
        return None
    support = bank.bytes.new_zeros(*bank.bytes.shape[:2], 256).scatter_add(
        -1, bank.bytes.clamp(0, 255), bank.byte_valid.long()).bool()
    return CandidateWholes(bank.codes[..., :bank.meaning_start].detach(), support,
        bank.terminal_valid, properties.coefficients().detach(), basis.getW().detach())


def stage(whole_space):
    """Read the existing WholeSpace table once outside the compose loop."""
    counts = getattr(whole_space, '_staged_word_primitive_counts', None)
    basis = getattr(getattr(whole_space, 'subspace', None), 'what', None)
    properties = getattr(basis, 'primitive_properties', None)
    if not torch.is_tensor(counts) or properties is None or whole_space.held_at_zero:
        return None
    counts = counts.sum(-2)
    prefix = torch.cat((counts.new_zeros(counts.shape[0], 1, counts.shape[-1]), counts.cumsum(1)), 1)
    return SpanWholes(prefix, properties.coefficients(), basis.getW())


def selected(context, arity, like, rule):
    """Expand one span witness over a candidate's reference alternatives."""
    if context is None:
        return None
    mean, count, orders = context[arity]
    from ReferenceContext import denoted_order
    order = context.get((arity, getattr(rule, 'predicate_identity', None)))
    if order is None:
        order = denoted_order(rule, orders, context.get('references', {}).get(arity, ()), None)
    count = count * (order == 0)[..., None]
    if mean.ndim == like.ndim and mean.shape[1] != like.shape[1]:
        copies = like.shape[1] // mean.shape[1]
        mean, count = mean.repeat_interleave(copies, 1), count.repeat_interleave(copies, 1)
    return mean.to(like), count.to(like)


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
    local = SpanWholes(bank.prefix[batch:batch+1], bank.properties, bank.codes)
    values, number = local.read(program.actions.new_tensor(spans)[None])
    return replace(program, operand_counts=program.leaves.new_tensor(counts),
                   form_wholes=values[0], form_whole_counts=number[0] *
                       number.new_tensor(active)[:, None])


def inverse_context(program, index):
    counts = getattr(program, 'operand_counts', None)
    whole = getattr(program, 'form_wholes', None)
    return dict(counts=(1, 1) if counts is None else counts[index].unbind(),
                whole=None if whole is None else (whole[index:index+1],
                    program.form_whole_counts[index:index+1]))


def unary(parent, bank):
    """Undo a free unary's form center from candidate words, without a stamp.

    Compare the plain and centered form equations over the same dictionary
    as binary search. Unknown constituent spellings supply no whole witness.
    """
    if bank is None:
        return parent
    width = bank.forms.shape[-1]
    mean, count = bank.read(bank.forms)
    mean = torch.nn.functional.pad(mean, (0, width-mean.shape[-1]))
    form = parent[..., :width]
    plain = (bank.forms-form[:, None]).square().mean(-1).masked_fill(~bank.valid, torch.inf)
    centered = ((bank.forms+mean)*.5-form[:, None]).square().mean(-1)
    recovered = form[:, None]*2-mean
    in_chart = ((recovered >= 0) & (recovered <= 1) & torch.isfinite(recovered)).all(-1)
    centered = centered.masked_fill(~bank.valid | (count[..., 0] <= 0) | ~in_chart, torch.inf)
    cost, index = centered.min(-1)
    use = cost < plain.min(-1).values
    whole = mean.gather(1, index[:, None, None].expand(-1, 1, width))[:, 0]
    decoded = torch.where(use[:, None], form*2-whole, form)
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
    mean, count = bank.read(left, right)
    padded = torch.nn.functional.pad(mean, (0, width-mean.shape[-1]))
    centered = ratio(parent_form*2-padded)
    simple_error = (right+plain[..., None]*delta-parent_form).square().mean(-1)
    whole_error = ((right+centered[..., None]*delta+padded)/2-parent_form).square().mean(-1)
    simple_error = simple_error.masked_fill((plain <= 0) | (plain >= 1), torch.inf)
    use = ((count[..., 0] > 0) & (centered > 0) & (centered < 1)
           & (whole_error < simple_error))
    return torch.where(use, centered, plain), (mean, count*use[..., None])
