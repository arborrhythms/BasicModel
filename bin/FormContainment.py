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
def audit(basic, *, before=None, after=None):
    """Containment belongs to the basic percept's identity/property boundaries."""
    rows = tuple(basic.index.values())
    parts = {row: set(basic.occurrences[row][0]) for row in rows}
    wholes = {row: set(basic.occurrences[row][1]) for row in rows}
    values = {row: basic.code(row) for row in rows}
    pairs = [(a, b) for a, b in _containers(parts) if wholes[b] <= wholes[a]]
    native, identity_width = basic.native._basis, basic.identity_width
    part_comparisons = []
    whole_comparisons = []
    for row in rows:
        for part in parts[row]:
            part_comparisons.append((basic.part_codes([part])[0], values[row][:identity_width], False))
        for whole in wholes[row]:
            whole_comparisons.append((values[row][identity_width:],
                basic.whole_codes([whole])[0], False))
    def reports(snapshot):
        snapshot = values if snapshot is None else snapshot
        return dict(tower0=_report(part_comparisons), tower1=_report(whole_comparisons),
            cross=dict(_report(()), applicable=False),
            order0=_report(((snapshot[a], snapshot[b], parts[a] != parts[b] or wholes[a] != wholes[b])
                           for a, b in pairs), ties=True))
    previous, current = reports(before), reports(after)
    return dict(**current, before=previous, after=current, rows=len(rows), self_pairs=False,
                support_pairs_without_weight_order=0, premises='basic_percepts')
