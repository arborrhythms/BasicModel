"""Invert the actual compose forest using its occurrence co-operands.

The free decoder owns a different problem: choosing a tree and its children.
This reader follows the already committed tree; no word bank or target is
used to choose a child. A retained co-operand is a detached inverse witness.
"""
import torch


def reconstruct(language, record, programs, positions):
    B, W, D = record.word_values.shape
    recovered = torch.zeros_like(record.word_values)
    defined = torch.zeros_like(recovered, dtype=torch.bool)
    binary = list(language._tree_layer(2).ops)
    for b, program in enumerate(programs):
        if program is None:
            continue
        source = positions(b)
        forest = []
        for index, (kind, operation, word) in enumerate(program.actions.tolist()):
            if kind == 0:
                forest.append((kind, word))
            elif kind == 1:
                right, left = forest.pop(), forest.pop()
                forest.append((kind, operation, index, left, right))
            elif kind == 2:
                forest.append((kind, operation, index, forest.pop()))
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
            op_index = torch.tensor([operation], device=parent.device)
            live = torch.ones(1, dtype=torch.bool, device=parent.device)
            if kind == 2:
                child, bad = language.reverse_unary_step(parent, op_index, live, return_status=True)
                unary = language._tree_layer(1)
                wrapped = unary.unary_ops[operation]
                op = getattr(wrapped, 'gl', wrapped)
                if getattr(op, 'inverse_kind', None) == 'unary':
                    # The native unary inverse is the evidence-pole exchange.
                    coordinates = op.reverse(coordinates.to(parent)).bool()
                pending.append((node[3], child, coordinates & ~bad[:, None]))
                continue
            op = getattr(binary[operation], 'gl', binary[operation])
            side = 'left' if getattr(op, 'inverse_kind', None) == 'right' else 'right'
            frame = program.operation_values[index]
            witness = frame[0 if side == 'left' else 1].detach()[None]
            inverse = getattr(op, 'inverse_kind', None)
            if inverse == 'search':
                # There is no derivation inverse here. A free bank search or
                # an occurrence target must not invent an expectation.
                left = right = torch.zeros_like(parent)
                left_defined = right_defined = torch.zeros_like(coordinates)
            else:
                left, right, bad = language.reverse_binary_step(parent, op_index, live,
                    reference=witness, reference_side=side, return_status=True,
                    case_bank=record.primed.case_bank)
                if inverse == 'product':
                    # Match the native inverse's domain coordinate by
                    # coordinate; its row-wide status loses this information.
                    remainder_defined = coordinates & (witness.abs() > 1e-8)
                    left_defined, right_defined = ((remainder_defined, coordinates)
                        if side == 'right' else (coordinates, remainder_defined))
                else:
                    if inverse == 'fold':
                        inner = getattr(op, '_sigma', None)
                        if inner is None:
                            inner = getattr(op, '_pi', None)
                        if bool(getattr(inner, 'nonlinear', False)):
                            # Clamping atanh makes its implementation finite,
                            # but cannot define an inverse at a saturated edge.
                            coordinates = coordinates & parent.abs().lt(1) & witness.abs().lt(1)
                    # Dense maps mix coordinates. If a parent coordinate was
                    # undefined upstream, the mixed child has no expectation.
                    if inverse not in ('residual', 'left', 'right'):
                        coordinates = coordinates.all(-1, keepdim=True).expand_as(coordinates)
                    left_defined = right_defined = coordinates & ~bad[:, None]
            pending.extend(((node[3], left, left_defined), (node[4], right, right_defined)))
    unavailable = record.word_valid & ~defined.all(-1)
    return recovered, unavailable, defined
