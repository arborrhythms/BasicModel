"""Interpretability is a tie order, never an answer signal or a second policy."""
import torch


def operator_is_structural(operator):
    """Adapters preserve their implementation's contract; names prove nothing."""
    implementation = getattr(operator, 'gl', operator)
    kind = getattr(implementation, 'routing_kind', 'opaque')
    if kind not in ('structural', 'opaque'):
        raise ValueError(f'invalid grammar routing kind: {kind!r}')
    return kind == 'structural'


def structural_argmax(scores, structural=None):
    """Highest score first, structural face on EXACT ties, then catalog order.

    No epsilon can overturn a better score. All-structural legacy catalogs
    use the original argmax without changing tensors or checkpoint state.
    Soft probabilities/gradients and training exploration are unchanged.
    """
    if structural is None:
        return scores.argmax(dim=-1)
    flags = tuple(structural)
    if len(flags) != scores.shape[-1] or any(type(x) is not bool for x in flags):
        raise ValueError('one structural flag is required per grammar candidate')
    if not any(flags) or all(flags):
        return scores.argmax(dim=-1)
    tied = scores == scores.amax(dim=-1, keepdim=True)
    preferred = tied & torch.tensor(flags, device=scores.device, dtype=torch.bool)
    eligible = torch.where(preferred.any(dim=-1, keepdim=True), preferred, tied)
    return eligible.to(torch.long).argmax(dim=-1)


def require_opaque_mlp(ops, chooser):
    """Explicit opaque extensions cannot bypass the ordinary grammar MLP.

    Unlabelled standalone tensor fixtures predate the catalog and retain their
    compatibility scorer. Production registration checks every implementation.
    """
    if any(getattr(getattr(op, 'gl', op), 'routing_kind', None) == 'opaque'
           for op in ops) and chooser != 'mlp':
        raise ValueError('opaque grammar operators require the ordinary mlp chooser')
