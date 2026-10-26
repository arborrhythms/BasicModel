"""Read-time numerical isomorphisms, authenticated by MTM event values.

A checked equality supplies the input identities. This module asks whether
one declared kernel and context explain both outputs. Inverse hypotheses
are ephemeral; no composition trace, lexical membership or target is read.
"""
from math import sqrt
from types import SimpleNamespace
import torch

from Generative import preserve_operator_activations


def _reconstructs(value, expected):
    # The native decoder's relative 1e-4 code tolerance (Generative and
    # Language.decoder_eligibility), including finite-chart inverse error.
    return bool(torch.isfinite(value).all() and
                (value - expected).norm() <= 1e-4 * max(1., float(expected.norm())))


def _inverse(op, after, before, *, checked=True):
    inner = getattr(op, '_sigma', None) or getattr(op, '_pi', None)
    yes = torch.ones(1, dtype=torch.bool, device=before.device)
    left, right = inner.generate_functional(after[None], reference=before[None],
                                            reference_side=(~yes, yes))
    if not bool(torch.isfinite(right).all() & (right >= 0).all() & (right <= 1).all()):
        return None
    return right[0] if not checked or _reconstructs(op.compose(left, right)[0], after) else None


def _common_context(op, first, second, left, right):
    # Clipping loses different coordinates in the two inverses. Their meet
    # and join are hypotheses, not evidence: both complete forward outputs
    # must agree before either can witness a common context.
    contexts = torch.stack((torch.minimum(first, second), torch.maximum(first, second)))
    a = op.compose(left['before'][None].expand(2, -1), contexts)
    b = op.compose(right['before'][None].expand(2, -1), contexts)
    error = ((a-left['after']).norm(dim=1) / max(1., float(left['after'].norm())) +
             (b-right['after']).norm(dim=1) / max(1., float(right['after'].norm())))
    valid = any(_reconstructs(x, left['after']) and _reconstructs(y, right['after']) for x, y in zip(a, b))
    return float(error.min()), valid


def _lexical_context(op, event, native, work):
    """Find one lexical activation whose inverse context explains both outputs.

    Rail clipping can erase coordinates in the event's inverse. Recover the
    context from both outputs. Golden-section search spans the
    reader's entire activation interval; the work meter bounds every inverse
    and both forward checks. No starting value is selected from a fixture.
    """
    if not work.consume('unfold'):
        return False
    first = _inverse(op, event['after'], event['before'], checked=False)
    if first is None:
        return False
    def evaluate(activation):
        # One inverse and two forward checks for each of two hypotheses.
        if not work.consume('unfold', 5):
            return float('inf'), False
        candidate = dict(native, before=activation * native['before'])
        context = _inverse(op, native['after'], candidate['before'], checked=False)
        if context is None:
            return float('inf'), False
        return _common_context(op, first, context, event, candidate)
    lower, upper = torch.finfo(native['before'].dtype).eps, 1.
    # A four-bit coarse scan brackets a basin before local refinement:
    # clipped inverses need not have a unimodal residual on the whole cube.
    grid = [lower + (upper-lower) * i / 16 for i in range(17)]
    scores = []
    for activation in grid:
        if work.remaining < 5:
            return False
        error, valid = evaluate(activation)
        if valid:
            return True
        scores.append(error)
    best = min(range(len(scores)), key=scores.__getitem__)
    lower, upper = grid[max(0, best-1)], grid[min(16, best+1)]
    ratio = (sqrt(5.) - 1.) / 2.
    a, b = upper - ratio * (upper-lower), lower + ratio * (upper-lower)
    fa, valid = evaluate(a)
    if valid:
        return True
    fb, valid = evaluate(b)
    if valid:
        return True
    while work.remaining >= 5 and upper-lower > torch.finfo(native['before'].dtype).eps:
        if fa <= fb:
            upper, b, fb = b, a, fa
            a = upper - ratio * (upper-lower)
            fa, valid = evaluate(a)
        else:
            lower, a, fa = a, b, fb
            b = lower + ratio * (upper-lower)
            fb, valid = evaluate(b)
        if valid:
            return True
    return False


@torch.no_grad()
def isomorphic(kernels, left, right, work):
    if work is None:
        return False
    language = SimpleNamespace(_generate_binary_ops=kernels, _generate_unary_ops=())
    return _isomorphic(language, left, right, work)


@preserve_operator_activations
def _isomorphic(language, left, right, work):
    for known, other in ((left, right), (right, left)):
        if known['native']:
            continue  # a prototype alone supplies no witnessed event
        for wrapped in language._generate_binary_ops:
            op = getattr(wrapped, 'gl', wrapped)
            inner = getattr(op, '_sigma', None) or getattr(op, '_pi', None)
            if (getattr(op, 'inverse_kind', None) != 'fold' or inner is None
                    or getattr(op, 'meaning_layout', None) is not None
                    or not hasattr(inner, 'generate_functional')):
                continue
            if other['native']:
                if _lexical_context(op, known, other, work):
                    return True
            elif work.consume('unfold', 6):
                a = _inverse(op, known['after'], known['before'], checked=False)
                b = _inverse(op, other['after'], other['before'], checked=False)
                if a is not None and b is not None and _common_context(op, a, b, known, other)[1]:
                    return True
    return False
