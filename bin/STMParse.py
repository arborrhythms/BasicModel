"""The parse written by the chooser, with static names in each trial's carry.

Heads use Memory's native typed references. Occurrences name input columns;
the close translates them to located STM0 names. Each structural column owns
two STM1 rows, so an application needs neither allocation nor a host read.
"""
import torch

from Memory import OCCURRENCE, SYMBOL, RELATION, STMChunks


def push(state, word, gate):
    index = STMChunks.at(state)
    word = torch.as_tensor(word, device=index.device).expand_as(index)
    head = torch.stack((torch.full_like(index, OCCURRENCE), word), -1)
    result = list(state)
    result[STMChunks.HEADS] = STMChunks.write(state[STMChunks.HEADS], index, head, gate.reshape(-1))
    return tuple(result)


def apply(before, after, choice, column, pairs, order, symbols, binary_operators, unary_operators):
    """Write op(L,R) or op(L); transposition only exchanges operand order.

    STMChunks.apply_choice has already exchanged stack positions for a
    transpose, and its heads follow those positions. It creates no parse node.
    """
    active = choice.applied & ((choice.kind == 1) | (choice.kind == 2))
    bind = torch.zeros_like(active) if choice.stack_arity is None else choice.stack_arity == 1
    binary = (choice.kind == 1) & ~bind
    position = torch.where(choice.kind == 2, choice.position, 0)
    left_at = STMChunks.at(before, torch.where(binary, 1, position))
    right_at = STMChunks.at(before)
    left = STMChunks.gather(before[STMChunks.HEADS], left_at.clamp_min(0))
    right = STMChunks.gather(before[STMChunks.HEADS], right_at.clamp_min(0))
    column = torch.as_tensor(column, device=pairs.device).expand_as(left_at)
    # Both catalog reads are legal even when only the other kind applies.
    def operator(table):
        if table.numel() == 0:
            return torch.zeros_like(left_at)
        return table[choice.local_op.clamp(0, table.numel() - 1)]
    op = symbols[torch.where(choice.kind == 2, operator(unary_operators), operator(binary_operators))]
    children = torch.stack((left, right), -2)
    pairs = STMChunks.write(pairs, 2 * column, children, active & binary)
    operand = torch.where(binary[:, None], torch.stack((torch.full_like(column, RELATION), 2 * column), -1), left)
    parent = torch.stack((torch.stack((torch.full_like(op, SYMBOL), op), -1), operand), -2)
    pairs = STMChunks.write(pairs, 2 * column + 1, parent, active)
    order = STMChunks.write(order, column, order.amax(-1) + 1, active)
    target = STMChunks.at(after, position)
    head = torch.stack((torch.full_like(column, RELATION), 2 * column + 1), -1)
    result = list(after)
    result[STMChunks.HEADS] = STMChunks.write(after[STMChunks.HEADS], target, head, active)
    return tuple(result), pairs, order


def roots(state):
    """Oldest-to-newest remaining operands, padded with empty references."""
    positions = state[STMChunks.DEPTH][:, None] - 1 - torch.arange(
        state[STMChunks.STACK].shape[1], device=state[STMChunks.STACK].device)[None]
    match = (state[STMChunks.STACK][:, None] == positions[:, :, None]) & (positions[:, :, None] >= 0)
    indices = match.long().argmax(-1)
    heads = STMChunks.gather(state[STMChunks.HEADS], indices)
    return torch.where(match.any(-1)[:, :, None], heads, 0)


def copy_pairs(pairs, roots, words, names, order):
    """One host close: translate reachable, already-written pairs into STM1.

    Compact in column order, preserving the native allocator's write order.
    This walks references; it does not replay choices or a reduction stack.
    """
    records = pairs.tolist()
    turns = order.tolist()
    heads = [tuple(head) for head in roots.tolist() if head[0]]
    occurrences = dict(zip(words.tolist(), names, strict=True))
    reachable = set()
    todo = list(heads)
    while todo:
        tag, value = todo.pop()
        if tag == RELATION and value not in reachable:
            if not 0 <= value < len(records):
                raise ValueError('parse reference is outside its trial')
            reachable.add(value)
            todo.extend(records[value])
    compact = []
    translated = {}
    def reference(head):
        tag, value = head
        if tag == OCCURRENCE:
            return tag, occurrences[value]
        if tag == RELATION:
            return tag, translated[value]
        if tag != SYMBOL:
            raise ValueError('parse contains an empty operand')
        return tag, value
    for row in sorted(reachable, key=lambda row: (turns[row // 2], row % 2)):
        compact.append(tuple(reference(head) for head in records[row]))
        translated[row] = len(compact) - 1
    head = None
    for root in reversed(heads):
        root = reference(root)
        if head is None:
            head = root
        else:
            compact.append((root, head))
            head = RELATION, len(compact) - 1
    return compact, head
