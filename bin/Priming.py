"""Directed native class-membership cache, independent of concept spread."""
import torch


@torch.no_grad()
def part_edges(owner, *, authority=True):
    """Cache (member, raised class, net positive evidence), never definitions."""
    allocator = owner._concept_allocator
    parts = allocator.parts
    from ConceptHierarchy import classes as hierarchy_classes
    classes = hierarchy_classes(owner)
    parts._class_rows = classes
    revision = (parts.symbol_revision, getattr(allocator.layer(), '_record_revision', 0) if hasattr(allocator, 'layer') else 0, tuple(sorted(classes)))
    if owner._priming_edge_revision != revision:
        src, dst, weights = (owner._priming_edge_src, owner._priming_edge_dst,
                             owner._priming_edge_weight)
        count = 0
        relation_rows = []
        enumerated = set()
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
            if int(parts.ref_kinds[relation, 0]) == parts.SYMBOL:
                enumerated.add((member, parent))
            relation_rows.append(relation)
            count += 1
        # An order-zero located conjunction records its same-order form
        # members as field constituents, rather than a next-order raise.
        for parent in classes:
            right = owner._csw_row_of(parent)
            if right is None:
                continue
            for ref in allocator.refs(parent, 'part') if hasattr(allocator, 'refs') else ():
                if not isinstance(ref, tuple) or ref[0] not in ('field', 'sym'):
                    continue
                if ref[0] == 'sym' and (ref[1], parent) in enumerated:
                    continue
                if ref[0] == 'sym' and not allocator.class_member(ref[1], parent):
                    continue
                left = owner._csw_row_of(ref[1])
                if left is None or left == right:
                    continue
                if count >= len(src):
                    raise RuntimeError('class membership exceeds the constructed edge capacity')
                src[count], dst[count], weights[count] = left, right, ref[2] if ref[0] == 'field' else 1.
                relation_rows.append(-1)
                count += 1
        owner._priming_edge_relations = tuple(relation_rows)
        owner._priming_edge_count = count
        owner._priming_edge_revision = revision
        owner._priming_edge_rebuilds += 1
    count = owner._priming_edge_count
    if not count:
        return None
    weights = owner._priming_edge_weight[:count].clone()
    if not authority:
        from ConceptHierarchy import direction, fit
        for i, row in enumerate(owner._priming_edge_relations):
            if row == -1:
                member, parent = (owner.concept_id_at_row(int(column[i])) for column in
                    (owner._priming_edge_src, owner._priming_edge_dst))
                if bool(direction(owner, parent).any()):
                    weights[i] *= fit(owner, member, parent)
    if authority:
        # Geometry uses stored membership to construct a radius-bearing row.
        # Query authority follows current effective evidence, without expiring
        # or overwriting the structural cache when only a radius changes.
        weights = weights.clone()
        for i, row in enumerate(owner._priming_edge_relations):
            if row != -1:
                plus, minus = parts.evidence(row)
                weights[i] = (plus-minus).clamp_min(0)
            else:
                from ConceptTrust import positive_length
                identities = (owner.concept_id_at_row(int(column[i])) for column in
                              (owner._priming_edge_src, owner._priming_edge_dst))
                weights[i] *= min(positive_length(owner, cid) for cid in identities)
    return owner._priming_edge_src[:count], owner._priming_edge_dst[:count], weights
