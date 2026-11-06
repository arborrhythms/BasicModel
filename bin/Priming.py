"""Directed native class-membership cache, independent of concept spread."""
import torch


@torch.no_grad()
def part_edges(owner):
    """Cache (member, raised class, net positive evidence), never definitions."""
    allocator = owner._concept_allocator
    parts = allocator.parts
    classes = {key[1] for key in allocator.relate_idx if key[0] == 'part-base'}
    parts._class_rows = classes
    revision = parts.symbol_revision
    if owner._priming_edge_revision != revision:
        src, dst, weights = (owner._priming_edge_src, owner._priming_edge_dst,
                             owner._priming_edge_weight)
        count = 0
        rows = parts.ref_kinds[:, 1].eq(parts.SYMBOL).nonzero().flatten().tolist()
        for relation in rows:
            member, parent = parts.refs[relation].tolist()
            if parent not in classes:
                continue
            from FamiliarTier import member_row
            left = member_row(owner, int(parts.ref_kinds[relation, 0]), member)
            right = owner._csw_row_of(parent)
            if left is None or right is None or left == right:
                continue
            if count >= len(src):
                raise RuntimeError('class membership exceeds the constructed edge capacity')
            src[count], dst[count] = left, right
            weights[count] = (parts.truth_plus[relation] - parts.truth_minus[relation]).clamp_min(0)
            count += 1
        # An order-zero located conjunction records its same-order form
        # members as field constituents, rather than a next-order raise.
        for parent in classes:
            right = owner._csw_row_of(parent)
            if right is None:
                continue
            for ref in allocator.refs(parent, 'part') if hasattr(allocator, 'refs') else ():
                if not isinstance(ref, tuple) or ref[0] != 'field':
                    continue
                left = owner._csw_row_of(ref[1])
                if left is None or left == right:
                    continue
                if count >= len(src):
                    raise RuntimeError('class membership exceeds the constructed edge capacity')
                src[count], dst[count], weights[count] = left, right, ref[2]
                count += 1
        owner._priming_edge_count = count
        owner._priming_edge_revision = revision
        owner._priming_edge_rebuilds += 1
    count = owner._priming_edge_count
    return (owner._priming_edge_src[:count], owner._priming_edge_dst[:count],
            owner._priming_edge_weight[:count]) if count else None
