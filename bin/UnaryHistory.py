"""Last unary and repetition count carried with each existing operand slot."""
import torch

# One int64 lane retains the existing rule/swap state and the bounded count.
# First applications retain their historical rule-id encoding.
RADIX = 1 << 32


def count(history):
    return torch.where(history >= 0, history // RADIX + 1,
        torch.where(history <= -2, (-history-2) // RADIX, 0))


def last(history):
    return torch.where(history >= 0, history % RADIX, torch.where(history <= -2, -2, -1))


def applied(history, operation):
    return count(history) * RADIX + operation


def transposed(history):
    return -2 - count(history) * RADIX


def penalty(history, like):
    """First application costs zero, then one, four, nine, ... logit units."""
    return count(history).to(like).square()
