"""Read-only certificates for the identity and property form subspaces."""
import torch


def _report(comparisons, *, ties=False):
    result = dict(pairs=0, violating_pairs=0, coordinates=0, largest=0.)
    if ties:
        result['non_strict_pairs'] = 0
        result['equal_set_ties'] = 0
    for a, b, different in comparisons:
        # MPS has no double tensor support. Scalar receipts are always CPU
        # double; the actual forms retain their native device and dtype.
        a, b = a.detach().cpu().double(), b.detach().cpu().double()
        delta = (a-b).clamp_min(0)
        count = int(delta.gt(0).sum())
        result['pairs'] += 1
        result['violating_pairs'] += int(count > 0)
        result['coordinates'] += count
        result['largest'] = max(result['largest'], float(delta.max()) if delta.numel() else 0.)
        if ties:
            result['non_strict_pairs'] += int(different and not bool((a < b).any()))
            result['equal_set_ties'] += int(not different and torch.equal(a, b))
    return result


def _containers(parts):
    """Intersect sparse postings instead of constructing a pairwise tensor."""
    postings = {}
    for row, members in parts.items():
        for member in members:
            postings.setdefault(member, set()).add(row)
    for row, members in parts.items():
        lists = sorted((postings[p] for p in members), key=len)
        containers = set.intersection(*lists) if lists else set(parts)
        for other in sorted(containers - {row}):
            yield row, other


@torch.no_grad()
def audit(derived, *, before=None, after=None):
    from Spaces import _concept_alloc_of
    owner, width = derived.owner, derived.percept_width
    rows = sorted(row for row in _concept_alloc_of(owner).layer()._tensor_row_keys
                  if owner._order0_inventory_row(row))
    definitions = derived._definitions(rows)
    parts = {row: {pid for tower, pid, net in edges if tower == 0 and net > 0}
             for row, edges in definitions.items()}
    wholes = {row: {pid: min(1., net) for tower, pid, net in edges if tower == 1 and net > 0}
              for row, edges in definitions.items()}
    values = dict(zip(rows, derived.derive(rows)[..., :width]))
    candidates = [(a, b) for a, b in _containers(parts) if wholes[b].keys() <= wholes[a].keys()]
    # A soft meet is monotone in the evidence as well as in the selected
    # types. Matching positive supports alone do not imply an order.
    pairs = [(a, b) for a, b in candidates
             if not derived.property_width or all(wholes[a][pid] >= net for pid, net in wholes[b].items())]
    model = getattr(owner, '_model', None)
    ps = getattr(model, 'perceptualSpace', None)
    identity = getattr(getattr(ps, 'percept_store', None), 'identity', None)
    ws = getattr(getattr(getattr(model, 'wholeSpace', None), 'subspace', None), 'what', None)
    properties = getattr(ws, 'primitive_properties', None)

    tower0_pairs, distances = [], {}
    if identity is not None:
        basis = ps.subspace.what
        word_forms = {raw: basis.lookup_rows(row)[:derived.identity_width]
                      for raw, row in identity.word_rows.items()}
        atoms = {raw: identity.part_atoms(raw) for raw in identity.words}
        for raw, form in word_forms.items():
            for atom in atoms[raw]:
                tower0_pairs.append((basis.lookup_rows(identity.atoms[atom])[:derived.identity_width], form, False))
            for part, _lo, _hi in identity.nested.get(raw, ()):
                tower0_pairs.append((word_forms[part], form, False))
            distances[raw.hex()] = float((1-form.detach().cpu().double()).norm())
        tower0_pairs.extend((word_forms[a], word_forms[b], False) for a, b in _containers(atoms))
    tower0 = _report(tower0_pairs)
    tower0['distance_to_ones'] = distances

    tower1_pairs, support_only = [], 0
    if properties is not None and derived.property_width:
        coefficients = properties.coefficients().detach().cpu()
        observed = properties.observed.detach().cpu()
        active = (observed.any(-1) | coefficients.ne(0).any(-1)).nonzero().flatten().tolist()
        codes = properties.codes().detach().cpu()
        for a in active:
            for b in active:
                if a != b:
                    if bool((coefficients[a] <= coefficients[b]).all()):
                        tower1_pairs.append((codes[a], codes[b], False))
                    elif bool(((coefficients[a] > 0) <= (coefficients[b] > 0)).all()):
                        support_only += 1
    tower1 = _report(tower1_pairs)
    tower1['support_pairs_without_weight_order'] = support_only

    def reports(snapshot):
        snapshot = values if snapshot is None else snapshot
        order0 = _report(((snapshot[a][:width], snapshot[b][:width],
                           parts[a] != parts[b] or wholes[a] != wholes[b])
                          for a, b in pairs), ties=True)
        # Alec, 2026-10-27: parts and types need not be comparable. Keep
        # the report key for callers, with no cross-level inequality claim.
        cross = dict(_report(()), applicable=False)
        return dict(tower0=tower0, tower1=tower1, cross=cross, order0=order0)

    previous, current = reports(before), reports(after)
    return dict(**current, before=previous, after=current, rows=len(rows), self_pairs=False,
                support_pairs_without_weight_order=len(candidates)-len(pairs))
