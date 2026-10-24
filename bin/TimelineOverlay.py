"""Numerical branch mirrors for the open parse; Token owns every kept timeline.

The extra transaction fields are alternative values/validity, stack scope,
stack alternative evaluations and prior actual values, latest actual writers,
and an order/scope journal. They neither enumerate nor select identities.
"""
import torch


VALUES, VALID, SCOPE, SHADOW, BEFORE, WRITER, JOURNAL = range(5, 12)


def empty(state):
    values, valid, stack, origins, journal = state
    return (*state, torch.zeros_like(values), torch.zeros_like(valid),
        stack.new_zeros(*stack.shape[:2], 5), values.new_zeros(*stack.shape[:2], values.shape[-1]),
        values.new_zeros(*stack.shape[:2], values.shape[-1]), valid.long()*0,
        journal.new_zeros(*journal.shape[:2], 4))


def push(state, active, value):
    extra = list(state[5:])
    for column in (SCOPE, SHADOW, BEFORE):
        old = state[column]
        top = value[:, None] if column == SHADOW else torch.zeros_like(old[:, :1])
        extra[column-5] = torch.where(active[:, None, None], torch.cat((top, old[:, :-1]), 1), old)
    return tuple(extra)


def gather(value, positions):
    return value.gather(1, positions[..., None].expand(*positions.shape, value.shape[-1]))


def meaning(value, replacement, start):
    return torch.cat((value[..., :start], replacement[..., start:]), -1)


def branch(state, handles, fallback):
    from MTMOverlay import MTMOverlay
    available = state[VALID].gather(1, (handles-1).clamp_min(0)) & (handles > 0)
    return torch.where(available[..., None], MTMOverlay.read(state[VALUES], handles), fallback)


def prepare(references, state, source, rules, *, meaning_start):
    """Change numerical operands before scoring, never reference eligibility."""
    if references is None or len(state) <= VALUES:
        return references
    scopes, stack = gather(state[SCOPE], source), gather(state[2], source)
    shadows = gather(state[SHADOW], source)
    left = references['left']
    replacements, masks = [], []
    for index in references['binary_ops']:
        rule = rules[index]
        if rule.method_name == 'lift':
            replacements.append(branch(state, stack[:, :-1, 0], left[:, :, len(replacements)]))
            masks.append(scopes[:, 1:, 0].bool() & (stack[:, :-1, 0] > 0))
        elif rule.method_name == 'adverb':
            replacements.append(shadows[:, :-1])
            masks.append(scopes[:, :-1, 1] > 0)
        else:
            replacements.append(left[:, :, len(replacements)])
            masks.append(torch.zeros_like(scopes[:, :-1, 0], dtype=torch.bool))
    if not replacements:
        return references
    replacement = torch.stack(replacements, 2)
    changed = meaning(left, replacement, meaning_start)
    return dict(references, left=torch.where(torch.stack(masks, 2)[..., None], changed, left))


def choice(overlay, state, selected, operands, orders, evaluate, *, meaning_start):
    """Carry the selected face's branch evaluation for a later S attachment."""
    if len(state) <= VALUES:
        return selected
    from MTMOverlay import MTMOverlay
    binary = selected.kind == 1
    rule = overlay.rule(selected)
    size, batch = state[2].shape[1], len(operands)
    positions = selected.position.clamp(0, size-1)[:, None]
    def pair(array):
        unary = gather(array, positions)[:, 0]
        return (torch.where(binary[:, None], array[:, min(1,size-1)], unary),
                torch.where(binary[:, None], array[:, 0], unary))
    left, right = pair(state[2])
    ls, rs = pair(state[SCOPE])
    lh, rh = pair(state[SHADOW])
    lb, rb = pair(state[BEFORE])
    head = torch.where((rule[:, 0] == 2)[:, None], rs, ls)
    head_before = torch.where((rule[:, 0] == 2)[:, None], rb, lb)
    event = rule[:, 4].bool() | rule[:, 5].bool() | rule[:, 7].bool()
    post = rule[:, 8].bool() & (ls[:, 1] > 0)
    lift = rule[:, 4].bool()
    modal = torch.where(event, rs[:, 0].bool() & lift, head[:, 0].bool()) | rule[:, 8].bool()
    prior = MTMOverlay.read(state[0], left[:, 0:1])[:, 0]
    prior = torch.where((left[:, 0] > 0)[:, None], prior, operands[:, 0])
    alternative = branch(state, left[:, 0:1], prior[:, None])[:, 0]
    first = meaning(operands[:, 0], alternative, meaning_start)
    shadow_left = torch.where(event[:, None], first,
        torch.where((ls[:, 1] > 0)[:, None], meaning(operands[:, 0], lh, meaning_start), operands[:, 0]))
    shadow_right = torch.where((rs[:, 1] > 0)[:, None], meaning(operands[:, 1], rh, meaning_start), operands[:, 1])
    exists = state[VALID].gather(1, (left[:, 0:1]-1).clamp_min(0))[:, 0] & (left[:, 0] > 0)
    inherited = (head[:, 1] > 0) & ((rule[:, 0] > 0) | ~binary)
    evaluate_mask = selected.applied & ((event & exists & ~modal) | (inherited & ~post & ~event))
    transformed = evaluate(shadow_left, shadow_right, evaluate_mask)
    shadow = torch.where(evaluate_mask[:, None], transformed, selected.candidate)
    shadow = torch.where((event & rule[:, 5].bool())[:, None], operands[:, 1], shadow)
    shadow = torch.where((event & rule[:, 7].bool())[:, None], torch.maximum(first, operands[:, 1]), shadow)
    # A determination or bind begins a new NP scope. It does not turn an
    # embedded abstraction into a modal modifier of an enclosing predication.
    reset = rule[:, 1].bool() | rule[:, 2].bool()
    scope = torch.where(inherited[:, None], head, torch.zeros_like(head))
    scope = torch.stack((modal.long(), scope[:, 1], scope[:, 2], scope[:, 3], scope[:, 4]), -1)
    left_order = torch.where(binary, orders[:, min(1, size-1)], orders.gather(1, positions)[:, 0])
    unlowered = torch.where(lift, left_order.clamp_min(1), selected.order.clamp_min(1))
    unlowered = torch.where(post, ls[:, 3], unlowered)
    prior_resolved = MTMOverlay.read(state[3], left[:, 0:1])[:, 0, 2]
    scope = torch.where((event | post)[:, None], torch.stack((modal.long(),
        torch.full_like(left_order, -1), torch.where(post, ls[:, 2], -1), unlowered,
        torch.where(post, ls[:, 4], prior_resolved)), -1), scope)
    scope = torch.where(reset[:, None], 0, scope)
    before = torch.where(event[:, None], prior, head_before)
    order = torch.where(lift, torch.where(modal, unlowered, (left_order-1).clamp_min(0)), selected.order)
    order = torch.where(post, unlowered, order)
    return selected._replace(order=order, timeline=(shadow, before, scope))


def apply(overlay, old, core, selected, operands, *, column):
    from MTMOverlay import MTMOverlay
    values, valid, stack, origins, events = core
    alternatives, available, scopes, shadows, befores, writers, journal = old[5:]
    batch, size = stack.shape[:2]
    binary, unary = selected.kind == 1, selected.kind == 2
    active = selected.applied & (binary | unary)
    rule = overlay.rule(selected)
    position = selected.position.clamp(0,size-1)[:, None]
    def pair(array):
        member = gather(array, position)[:, 0]
        return (torch.where(binary[:, None], array[:, min(1,size-1)], member),
                torch.where(binary[:, None], array[:, 0], member))
    left, right = pair(old[2]); ls, rs = pair(scopes)
    subject = left[:, 0]
    event = active & (rule[:, 4].bool() | rule[:, 5].bool() | rule[:, 7].bool())
    post = active & rule[:, 8].bool() & (ls[:, 1] > 0)
    if selected.timeline is None:
        raise ValueError('a timeline transaction requires the selected numerical correspondence')
    shadow, before, scope = selected.timeline
    modal = scope[:, 0].bool()
    current = column.expand(batch) + 1
    scope = torch.stack((scope[:, 0], torch.where(event | post, current, scope[:, 1]),
        torch.where(event, current, scope[:, 2]), scope[:, 3], scope[:, 4]), -1)
    wrote = events.gather(1, (current-1)[:, None, None].expand(batch,1,6))[:, 0,5].bool()
    shifted = event & modal & wrote & (subject > 0)
    result = MTMOverlay.read(values, subject[:, None])[:, 0]
    values = MTMOverlay.write(values, subject, MTMOverlay.read(old[0], subject[:, None])[:, 0], shifted)
    origins = MTMOverlay.write(origins, subject, MTMOverlay.read(old[3], subject[:, None])[:, 0], shifted)
    alternatives = MTMOverlay.write(alternatives, subject, result, shifted)
    # Only the covered event's own provisional actual write can be undone.
    previous_writer = writers.gather(1, (subject-1).clamp_min(0)[:, None])[:, 0]
    restore = post & (subject > 0) & (previous_writer == ls[:, 1])
    values = MTMOverlay.write(values, subject, before, restore)
    prior_origin = MTMOverlay.read(origins, subject[:, None])[:, 0]
    prior_origin = torch.cat((prior_origin[:, :2], ls[:, 4:5]), -1)
    origins = MTMOverlay.write(origins, subject, prior_origin, restore)
    alternatives = MTMOverlay.write(alternatives, subject, selected.candidate, post & (subject > 0))
    at = torch.arange(alternatives.shape[1], device=values.device)[None]
    available = available | ((shifted | post)[:, None] & (subject > 0)[:, None] & (at == subject[:, None]-1))
    parent = stack[:, 0] if size == 1 else torch.where(binary[:, None], stack[:, 0], gather(stack, position)[:, 0])
    wrapped = active & modal & (scope[:, 1] > 0) & ~event & ~post & (parent[:, 0] > 0)
    alternatives = MTMOverlay.write(alternatives, parent[:, 0], selected.candidate, wrapped)
    opened = active & (rule[:, 1].bool() | (rule[:, 2].bool() & (selected.order == 1)))
    actual_write = (event & ~modal & wrote) | opened
    writer_handle = torch.where(opened, parent[:, 0], subject)
    writers = torch.where(actual_write[:, None] & (at == writer_handle[:, None]-1), current[:, None], writers)
    writers = torch.where(restore[:, None] & (at == subject[:, None]-1), 0, writers)
    # Post-S attachment owns one event; its covered provisional event remains
    # only in the temporary derivation, not as an independently written row.
    old_event = events.gather(1, (ls[:, 1]-1).clamp_min(0)[:, None, None].expand(batch,1,6))[:, 0]
    events = MTMOverlay.write(events, current, old_event, post)
    covered = torch.arange(journal.shape[1],device=values.device)[None] == ls[:, 1:2]-1
    journal = torch.cat((journal[..., :3], torch.where(post[:,None] & covered, 1, journal[...,3])[...,None]), -1)
    entry = torch.stack((selected.order, scope[:,0], scope[:,2], torch.zeros_like(current)), -1)
    journal = MTMOverlay.write(journal, current, entry, active)
    def move(old_value, new_value):
        rewritten = torch.where((active & unary)[:, None, None] &
            (torch.arange(size,device=values.device)[None,:,None] == position[...,None]), new_value[:,None], old_value)
        folded = torch.cat((new_value[:,None], old_value[:,2:], torch.zeros_like(old_value[:,:1])),1) if size>1 else new_value[:,None]
        result = torch.where((active & binary)[:,None,None], folded, rewritten)
        if size > 1:
            result = torch.where((selected.applied & (selected.kind==3))[:,None,None],
                torch.cat((result[:,1:2],result[:,:1],result[:,2:]),1), result)
        return result
    return (values, valid, stack, origins, events, alternatives, available,
            move(scopes,scope), move(shadows,shadow), move(befores,before), writers, journal)
