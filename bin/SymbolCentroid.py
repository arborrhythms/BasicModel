"""Audit native word forms without moving their perceptual coordinates."""
import torch


@torch.no_grad()
def place(forms, parts):
    """Return unchanged forms and their identity/containment certificate."""
    rows = sorted(forms)
    postings = {}
    for row in rows:
        for atom in parts[row]:
            postings.setdefault(atom, set()).add(row)
    containers = {}
    for row in rows:
        extents = sorted((postings[p] for p in parts[row]), key=len)
        containers[row] = (set.intersection(*extents) if extents else set(rows)) - {row}
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
        # This scalar audit uses double precision; MPS has no float64 dtype.
        # Keep the native forms on their device and compute the report on CPU.
        x = torch.stack([values[row] for row in rows]).cpu().double()
        x = torch.nn.functional.normalize(x, dim=-1)
        # Sum of every off-diagonal dot product, without a V by V allocation.
        return float((x.sum(0).square().sum() - x.square().sum()) / (len(rows)*(len(rows)-1)))

    placed = {row: value.clone() for row, value in forms.items()}
    report = dict(words=len(rows), moved=0, share_moved=0.,
                  cosine_before=cosine(forms), cosine_after=cosine(placed),
                  identity_recovered=all(torch.equal(placed[r].eq(1), forms[r].eq(1)) for r in rows),
                  below_lower=0, before=violations(forms), after=violations(placed))
    return placed, report
