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
    order = STMChunks.write(order, column, order.amax(-1) + 1, choice.applied)
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
    n_pairs, n_turns, n_roots = pairs.numel(), order.numel(), roots.numel()
    flat = torch.cat((pairs.reshape(-1), order.reshape(-1), roots.reshape(-1), words.reshape(-1))).tolist()
    i0 = n_pairs
    i1 = i0 + n_turns
    i2 = i1 + n_roots
    p_flat = flat[:i0]
    records = [((p_flat[k], p_flat[k + 1]), (p_flat[k + 2], p_flat[k + 3])) for k in range(0, n_pairs, 4)]
    turns = flat[i0:i1]
    r_flat = flat[i1:i2]
    heads = [(r_flat[k], r_flat[k + 1]) for k in range(0, n_roots, 2) if r_flat[k]]
    occurrences = dict(zip(flat[i2:], names, strict=True))
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


def program(word_events, order, kind, local, offsets, word_owners, owners, sentence,
            binary_count, stop, target_width):
    """Pack the carry's existing events; topology remains exclusively in heads.

    A word records the next application ordinal at its arrival and its own
    fixation rank. Sorting these immutable stamps interleaves arrivals and
    choices without a reduction-stack replay or a guess from source columns.
    """
    batch, width = word_events.shape[:2]
    wanted = sentence if torch.is_tensor(sentence) else torch.full_like(order[:, :1], sentence)
    words_live = (word_events[..., 0] >= 0) & (word_owners == wanted)
    ops_live = (order >= 0) & (owners == wanted)
    ranks = torch.where(words_live, word_events[..., 1], torch.iinfo(torch.long).max)
    positions = ranks.argsort(dim=1, stable=True)
    words_live = words_live.gather(1, positions)
    positions = torch.where(words_live, positions, -1)
    stamps = word_events[..., 0].gather(1, positions.clamp_min(0))
    word_actions = torch.stack((torch.zeros_like(positions), torch.full_like(positions, -1),
        torch.arange(width, device=order.device)[None].expand(batch, -1)), -1)
    op_actions = torch.stack((kind, local, torch.where(kind == 2, offsets, -1)), -1)
    actions = torch.cat((word_actions, op_actions), 1)
    live = torch.cat((words_live, ops_live), 1)
    stamps = torch.cat((2 * stamps, 2 * order + 1), 1)
    slots = torch.cat((torch.full_like(positions, -1),
        torch.arange(order.shape[1], device=order.device)[None].expand(batch, -1)), 1)
    take = torch.where(live, stamps, torch.iinfo(torch.long).max).argsort(dim=1, stable=True)
    actions = torch.where(live[..., None], actions, -1).gather(1, take[..., None].expand(-1, -1, 3))
    columns = torch.where(live, slots, -1).gather(1, take)
    count = live.sum(1, keepdim=True)
    from Meaning import require_sentence_tensor
    require_sentence_tensor((count <= target_width).all(), 'parse events exceed their constructed capacity')
    targets = torch.where(actions[..., 0] == 0, stop,
        torch.where(actions[..., 0] == 2, binary_count + actions[..., 1], actions[..., 1]))
    reverse = count - 1 - torch.arange(target_width, device=order.device)
    targets = torch.where(reverse >= 0, targets.gather(1, reverse.clamp(0, targets.shape[1] - 1)), -1)
    return positions, actions[:, :target_width], targets, columns[:, :target_width]
