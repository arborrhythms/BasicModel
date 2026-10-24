"""Trial-local case selections beside the conceptual stack.

The owner supplies the case bank before parsing. This carrier retains an
intersection through headed composition; it neither owns identities nor
reads memory. The journal is consumed by the kept close.
"""
import torch

from CaseSelection import CaseView


def empty(like, *, cases, journal):
    batch, slots, width = like.shape
    return (like.new_zeros(batch, slots, cases, 2),
            like.new_ones(batch, slots, width),
            torch.zeros(batch, slots, dtype=torch.bool, device=like.device),
            like.new_zeros(batch, journal, width),
            torch.zeros(batch, journal, dtype=torch.bool, device=like.device),
            torch.zeros_like(like))


def gather(value, positions):
    index = positions.reshape(*positions.shape, *([1] * (value.ndim-2)))
    return value.gather(1, index.expand(*positions.shape, *value.shape[2:]))


def view(state, bank, positions):
    return CaseView(bank.codes, gather(state[0], positions),
                    gather(state[1], positions), gather(state[2], positions), gather(state[5], positions))


def local(state, columns):
    return (*state[:3], gather(state[3], columns), gather(state[4], columns), state[5])


def publish(previous, current, columns):
    """Publish one word's bounded journal once, after its local choices."""
    result = list(current[:3])
    for index in (3, 4):
        old, value = previous[index], current[index]
        at = columns.reshape(*columns.shape, *([1]*(old.ndim-2)))
        result.append(old.scatter(1, at.expand_as(value), value))
    return (*result, current[5])


def push(state, bank, head, active):
    incoming = bank.view(head)
    result = list(state)
    for index, value in zip((0, 1, 2, 5), (incoming.weights, incoming.test, incoming.selected, incoming.reference)):
        mask = active.reshape(-1, *([1] * (state[index].ndim-1)))
        result[index] = torch.where(mask, torch.cat((value[:, None], state[index][:, :-1]), 1), state[index])
    return tuple(result)


def apply(state, bank, choice, operands, rules, unary_rules, operations, *, column, unary_ops=()):
    """Carry the selected head and record the same learned case test once."""
    batch, slots = state[0].shape[:2]
    binary = choice.kind == 1
    unary = choice.kind == 2
    position = choice.position.clamp(0, slots-1)[:, None]
    left = tuple(torch.where(binary.reshape(batch, *([1]*(value.ndim-2))),
                             value[:, min(1, slots-1)], gather(value, position)[:, 0])
                 for value in (*state[:3], state[5]))
    right = tuple(value[:, 0] for value in (*state[:3], state[5]))
    weights, test, selected, reference = left
    for kind, catalogue in ((1, rules), (2, unary_rules)):
        for index, rule in enumerate(catalogue):
            mask = choice.applied & (choice.kind == kind) & (choice.local_op == index)
            head = 2 if getattr(rule, 'clause_form', None) == 'S' else getattr(rule, 'head_role', 0)
            source = right if head == 2 and kind == 1 else left
            source_weights, source_test, source_selected, source_reference = source
            if head == 0:
                source_weights = torch.zeros_like(source_weights)
                source_test = torch.ones_like(source_test)
                source_selected = torch.zeros_like(source_selected)
                source_reference = torch.zeros_like(source_reference)
            if getattr(rule, 'case_head_role', 0):
                selected_ops = operations if kind == 1 else unary_ops
                op = getattr(selected_ops[index], 'gl', selected_ops[index])
                if getattr(op, 'meaning_layout', None) is not None:
                    chosen = op.select_view(operands[:, 0], CaseView(bank.codes, *source))
                    source_weights, source_test, source_selected = chosen.weights, chosen.test, chosen.selected
            weights = torch.where(mask[:, None, None], source_weights, weights)
            test = torch.where(mask[:, None], source_test, test)
            selected = torch.where(mask, source_selected, selected)
            reference = torch.where(mask[:, None], source_reference, reference)
    result = list(state)
    width = next(getattr(op, 'gl', op).meaning_layout[0] for op in (*operations, *unary_ops)
                 if getattr(getattr(op, 'gl', op), 'case_head_role', 0)
                 and getattr(getattr(op, 'gl', op), 'meaning_layout', None) is not None)
    case_value = torch.nn.functional.pad(
        CaseView(bank.codes, weights, test, selected).fold(meaning_start=width), (width, 0))
    for index, value in zip((0, 1, 2, 5), (weights, test, selected, reference)):
        old = state[index]
        folded = torch.cat((value[:, None], old[:, 2:],
                            (torch.ones_like(old[:, :1]) if index == 1 else torch.zeros_like(old[:, :1]))), 1)[:, :slots]
        bmask = (choice.applied & binary).reshape(batch, *([1]*(old.ndim-1)))
        umask = (choice.applied & unary)[:, None] & (torch.arange(slots, device=old.device)[None] == position)
        umask = umask.reshape(batch, slots, *([1]*(old.ndim-2)))
        changed = torch.where(bmask, folded, torch.where(umask, value[:, None], old))
        swap = torch.cat((old[:, 1:2], old[:, :1], old[:, 2:]), 1)
        tmask = (choice.applied & (choice.kind == 3)).reshape(batch, *([1]*(old.ndim-1)))
        result[index] = torch.where(tmask, swap, changed)
        if index in (0, 5):
            continue
        journal = state[index+2]
        write = choice.applied[:, None] & (torch.arange(journal.shape[1], device=old.device)[None] == column[:, None])
        write = write.reshape(batch, journal.shape[1], *([1]*(journal.ndim-2)))
        result[index+2] = torch.where(write, (case_value if index == 1 else value)[:, None], journal)
    return tuple(result)


def reset(state, mask):
    return tuple(torch.where(mask.reshape(-1, *([1]*(value.ndim-1))),
        torch.ones_like(value) if index == 1 else torch.zeros_like(value), value)
        for index, value in enumerate(state))
