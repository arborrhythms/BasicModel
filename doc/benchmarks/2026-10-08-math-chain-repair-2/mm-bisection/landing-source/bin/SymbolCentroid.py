"""Detached placement between a word's parts and its adjacent wholes."""
import torch


@torch.no_grad()
def place(forms, parts, adjacency, *, enabled=True):
    """Return c and an exhaustive identity/containment certificate.

    adjacency maps (word, neighbour) to distinct (address, pair-position)
    witnesses. Equal part sets are projected together, then strict subsets
    are capped in decreasing part count. The lower bound is never changed.
    """
    rows = sorted(forms)
    neighbours = {row: [] for row in rows}
    weights = dict.fromkeys(rows, 0)
    for (left, right), witnesses in adjacency.items():
        for word, other in set(((left, right), (right, left))):
            if word in forms and other in forms and witnesses:
                neighbours[word].append(other)
                weights[word] += len(witnesses)
    before = {}
    for row, lower in forms.items():
        upper = lower
        if neighbours[row]:
            shared = torch.stack([forms[n] for n in neighbours[row]]).amin(0)
            upper = torch.maximum(lower, shared)
        wp, wu = len(parts[row]), weights[row]
        alpha = wu / (wp + wu) if enabled and wp + wu else 0.
        before[row] = lower + alpha * (upper - lower)

    postings, groups = {}, {}
    for row in rows:
        groups.setdefault(frozenset(parts[row]), []).append(row)
        for atom in parts[row]:
            postings.setdefault(atom, set()).add(row)
    containers = {}
    for row in rows:
        extents = sorted((postings[p] for p in parts[row]), key=len)
        containers[row] = (set.intersection(*extents) if extents else set(rows)) - {row}
    after = {row: value.clone() for row, value in before.items()}
    for atoms, group in sorted(groups.items(), key=lambda item: -len(item[0])):
        bounds = set(group)
        for row in group:
            bounds.update(containers[row])
        cap = torch.stack([after[r] for r in sorted(bounds)]).amin(0)
        for row in group:
            after[row] = torch.maximum(forms[row], torch.minimum(after[row], cap))

    def violations(values):
        count = coordinates = 0
        largest = 0.
        for row in rows:
            for other in containers[row]:
                delta = (values[row] - values[other]).clamp_min(0)
                count += int(bool(delta.gt(0).any()))
                coordinates += int(delta.gt(0).sum())
                largest = max(largest, float(delta.max()))
        return dict(pairs=sum(map(len, containers.values())),
                    violating_pairs=count, coordinates=coordinates, largest=largest)

    def cosine(values):
        if len(rows) < 2:
            return None
        x = torch.stack([values[row] for row in rows]).double()
        x = torch.nn.functional.normalize(x, dim=-1)
        # Sum of every off-diagonal dot product, without a V by V allocation.
        return float((x.sum(0).square().sum() - x.square().sum()) / (len(rows)*(len(rows)-1)))

    moved = sum(not torch.equal(forms[r], after[r]) for r in rows)
    report = dict(words=len(rows), moved=moved, share_moved=moved/len(rows) if rows else 0.,
                  cosine_before=cosine(forms), cosine_after=cosine(after),
                  identity_recovered=all(torch.equal(after[r].eq(1), forms[r].eq(1)) for r in rows),
                  below_lower=sum(bool((after[r] < forms[r]).any()) for r in rows),
                  before=violations(before), after=violations(after),
                  adjacent_witnesses=sum(len(w) for w in adjacency.values()), enabled=enabled)
    return after, report
