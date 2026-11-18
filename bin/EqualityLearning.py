"""The conceptual equality residual of an asserted, selected reading.

Only actual equality operations in comprehension contribute.
No lexical spelling, arithmetic label, reference order or desired program is
consulted. The loss follows the current operands back to their producing VP.
"""
import torch


def observation(rule, operands):
    """Read the live operands before their numerical stack entry is reduced."""
    if getattr(rule, 'method_name', None) == 'equal':
        left, right = operands
        return (left - right).square().mean()
    return None


def cost(language, comprehensions, active, like):
    terms, counts = [], []
    for row, result in enumerate(comprehensions):
        local = () if result is None or not bool(active[row]) else result.equality_terms
        terms.append(torch.stack(local).mean() if local else like[row].sum() * 0)
        counts.append(len(local))
    return torch.stack(terms), counts
