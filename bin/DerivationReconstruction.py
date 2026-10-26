"""Invert the actual compose forest using its occurrence co-operands.

The free decoder owns a different problem: choosing a tree and its children.
This reader follows the already committed tree; no word bank or target is
used to choose a child. A retained co-operand is a detached inverse witness.
"""
import torch


def inverse_step(language, kind, operation, parent, frame, *, coordinates=None, case_bank=None,
                 counts=(1, 1), whole=None):
    """Invert one step and return each child's value and defined coordinates.

    Omit coordinates to measure this operation's own inverse domain at its
    actual output. Reconstruction instead carries the domain inherited from
    the parent. A later undefined inverse must not make an earlier operation
    responsible for loss it did not cause.
    """
    coordinates = (torch.isfinite(parent) if coordinates is None else
                   coordinates & torch.isfinite(parent))
    # Missing coordinates carry no value into a later inverse. Their mask
    # remains false; a defined invalid coordinate still rejects in its map.
    parent = torch.where(coordinates, parent, 0.)
    op_index = torch.tensor([operation], device=parent.device)
    live = torch.ones(1, dtype=torch.bool, device=parent.device)
    if kind == 2:
        wrapped = language._tree_layer(1).unary_ops[operation]
        op = getattr(wrapped, 'gl', wrapped)
        if (getattr(op, 'meaning_layout', None) is not None
                and callable(getattr(op, 'compose_case_meaning', None))):
            width = op.meaning_layout[0]
            # Selection discards case content. The single child's form is
            # preserved; no affine inverse invents the discarded meaning.
            if whole is not None:
                mean, number = whole
                mean = torch.nn.functional.pad(mean, (0, width-mean.shape[-1]))
                form = torch.where(number > 0, parent[..., :width]*2-mean, parent[..., :width])
            else:
                form = parent[..., :width]
            return ((torch.cat((form, torch.zeros_like(parent[..., width:])), -1),
                     torch.cat((coordinates[..., :width], torch.zeros_like(coordinates[..., width:])), -1)),)
        if getattr(op, 'retains_operand', False):
            # A withholding operation explicitly retains its operand in the
            # committed journal. This is an occurrence witness, not a free
            # inverse of exclusion or a prediction of a missing operand.
            witness = frame[:1].detach()
            return ((witness, torch.isfinite(witness)),)
        if whole is not None and getattr(op, 'meaning_layout', None) is not None:
            width = op.meaning_layout[0]
            mean, number = whole
            mean = torch.nn.functional.pad(mean, (0, width-mean.shape[-1]))
            form = torch.where(number > 0, parent[..., :width]*2-mean, parent[..., :width])
            parent = torch.cat((form, parent[..., width:]), -1)
        child, bad = language.reverse_unary_step(parent, op_index, live, return_status=True)
        if getattr(op, 'inverse_kind', None) == 'unary' and not hasattr(op, 'marker_base'):
            coordinates = op.reverse(coordinates.to(parent)).bool()
        return ((child, coordinates & ~bad[:, None]),)
    wrapped = language._tree_layer(2).ops[operation]
    op = getattr(wrapped, 'gl', wrapped)
    if getattr(op, 'field_operand', None) == 2:
        # The second field position is an identity, not a presented word.
        # Binding retains the mention as an occurrence witness in the journal.
        witness = frame[:1].detach()
        return ((witness, torch.isfinite(witness)),)
    if (getattr(op, 'meaning_layout', None) is not None
                and callable(getattr(op, 'compose_case_meaning', None))):
        from MeaningKernels import split_centroid
        width = op.meaning_layout[0]
        witness = frame[0:1].detach()
        if whole is not None:
            whole = (torch.nn.functional.pad(whole[0], (0, width-whole[0].shape[-1])), whole[1])
        form = split_centroid(parent[..., :width], witness[..., :width], counts,
                              known_right=False, whole=whole)
        missing = torch.cat((form, torch.zeros_like(parent[..., width:])), -1)
        domain = torch.cat((coordinates[..., :width], torch.zeros_like(coordinates[..., width:])), -1)
        return ((witness, coordinates), (missing, domain))
    if getattr(op, 'meaning_layout', None) is not None and (
            getattr(op, 'meaning_kernel', None) is not None or
            getattr(op, 'meaning_action', None) is not None or op.meaning_rule == 'sum'):
        right_known = getattr(op, 'meaning_action', None) is not None or op.meaning_head_role == 0
        witness = frame[1 if right_known else 0].detach()[None]
        left, right = op.reverse_blocks(parent, witness, counts=counts,
                                        reference_right=right_known, whole=whole)
        valid = ((left >= -1e-6) & (left <= 1+1e-6) & (right >= -1e-6) & (right <= 1+1e-6)).all(-1)
        domain = coordinates & valid[:, None]
        return ((left, domain), (right, domain))
    inverse = getattr(op, 'inverse_kind', None)
    side = 'left' if inverse in ('right', 'surface') else 'right'
    witness = frame[0 if side == 'left' else 1].detach()[None]
    if inverse == 'search':
        # No derivation inverse: a free search or occurrence target must not
        # invent an expectation. Free-byte reconstruction is a separate path.
        child = torch.zeros_like(parent)
        defined = torch.zeros_like(coordinates)
        return ((child, defined), (child, defined))
    left, right, bad = language.reverse_binary_step(parent, op_index, live,
        reference=witness, reference_side=side, return_status=True, case_bank=case_bank)
    if inverse == 'product':
        remainder_defined = coordinates & (witness.abs() > 1e-8)
        left_defined, right_defined = ((remainder_defined, coordinates)
            if side == 'right' else (coordinates, remainder_defined))
    else:
        if inverse == 'fold':
            inner = getattr(op, '_sigma', None)
            if inner is None:
                inner = getattr(op, '_pi', None)
            if bool(getattr(inner, 'nonlinear', False)):
                coordinates = coordinates & parent.abs().lt(1) & witness.abs().lt(1)
        # Dense maps mix coordinates; upstream undefinedness propagates.
        if inverse not in ('residual', 'left', 'right'):
            coordinates = coordinates.all(-1, keepdim=True).expand_as(coordinates)
        left_defined = right_defined = coordinates & ~bad[:, None]
    return ((left, left_defined), (right, right_defined))


def reconstruct(language, record, programs, positions):
    B, W, D = record.word_values.shape
    recovered = torch.zeros_like(record.word_values)
    defined = torch.zeros_like(recovered, dtype=torch.bool)
    for b, program in enumerate(programs):
        if program is None:
            continue
        source = positions(b)
        forest = []
        for index, (kind, operation, word) in enumerate(program.actions.tolist()):
            if kind == 0:
                forest.append((kind, word))
            elif kind == 1:
                if language._compose_binary_rules[operation].method_name == 'bind':
                    forest[-1] = (kind, operation, index, forest[-1])
                else:
                    right, left = forest.pop(), forest.pop()
                    forest.append((kind, operation, index, left, right))
            elif kind == 2:
                position = len(forest)-1-max(0,word)
                forest[position] = (kind, operation, index, forest[position])
            elif kind == 3:
                forest[-2:] = reversed(forest[-2:])
        depth = int(record.end_depth[b])
        if depth != len(forest):
            raise ValueError('the recorded derivation must have the completed forest depth')
        values = record.end_slots[b, :depth]
        # The compose forest is oldest first; STM's occupied slots are newest first.
        pending = [(node, values[depth-1-i:depth-i],
                    torch.ones_like(values[depth-1-i:depth-i], dtype=torch.bool))
                   for i, node in enumerate(forest)]
        while pending:
            node, parent, coordinates = pending.pop()
            coordinates = coordinates & torch.isfinite(parent)
            if node[0] == 0:
                position = source[node[1]]
                recovered[b, position] = parent[0]
                defined[b, position] = coordinates[0]
                continue
            kind, operation, index = node[:3]
            rules = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
            if getattr(rules[operation], 'relation_kind', None):
                # ClauseScope publishes a relative result as a journal-local
                # address in STM. Dereference that name to the operation's
                # live result before applying its numerical inverse. The
                # address code itself is not a concept value to invert.
                parent = program.operation_values[index, 2:3]
            bank = getattr(getattr(record, 'primed', None), 'case_bank', None)
            from FormCentroids import inverse_context
            children = inverse_step(language, kind, operation, parent,
                program.operation_values[index], coordinates=coordinates,
                case_bank=None if bank is None else bank.for_stream(b),
                **inverse_context(program, index))
            pending.extend((child_node, value, domain)
                for child_node, (value, domain) in zip(node[3:], children, strict=True))
    unavailable = record.word_valid & ~defined.all(-1)
    return recovered, unavailable, defined
