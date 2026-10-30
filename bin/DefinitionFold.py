"""Witnessed two-pole coverage of the shared sparse sigma definitions.

Naming records provide provenance, never membership. The teacher is detached;
only the sigma read is live, including prospective zero edges on either pole.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Example:
    concept: int
    row: int
    member: tuple
    source: int
    witness: int
    names: tuple
    activation: torch.Tensor
    evidence: torch.Tensor


def examples(owner, field):
    from FamiliarTier import member_row
    from LTM1 import SYMBOL
    if field.ndim != 4 or field.shape[-1] != 2:
        raise ValueError('definition fold requires paired occurrence evidence')
    alloc = owner._concept_allocator
    parts, sigma = alloc.parts, alloc.layer(0)
    addresses = owner._field_inventory_rows(field.device)[:len(field)]
    addresses = addresses[:, None].expand(-1, field.shape[1]) if addresses.ndim == 1 else addresses
    result = []
    for witness in parts.indices().tolist():
        member, whole = parts.references(witness)
        if whole[0] != SYMBOL:
            continue
        cid = whole[1]
        row = owner._csw_row_of(cid)
        # A DEF destination is a word, not a class. A raised membership row
        # owns the same witness flag whether it has a name or not.
        if row is None or cid in alloc.retired or not bool(sigma.witnessed[row]):
            continue
        if cid in alloc.word_descriptions:
            continue
        source = member_row(owner, *member)
        if source is None:
            continue
        activation = torch.where((addresses == source)[..., None], field[..., 0], 0.).amax(0).detach()
        evidence = torch.stack((parts.truth_plus[witness], parts.truth_minus[witness])).detach().to(field)
        if not bool(activation.any()) or not bool(evidence.any()):
            continue
        result.append(Example(cid, row, member, source, witness,
                              tuple(owner.definitions.words(cid)), activation, evidence))
    return tuple(result)


def prepare(owner, field):
    """Stage additions before either trial retains a parameter graph."""
    records = examples(owner, field)
    owner._prepare_part_learning(witnesses=tuple((e.row, e.source, e.evidence) for e in records))
    owner._maybe_rebuild_optimizer_for_csw()
    return field.detach().clone(), owner._field_inventory_rows(field.device).detach().clone(), records


def category(model):
    """Sigma is output-owned only in the existing named-output partition."""
    return 'output' if getattr(getattr(model,'outputSpace',None), 'concept_ids', ()) else 'reconstruction'


def loss(owner, staged):
    field, addresses, records = staged
    B, L = field.shape[1:3]
    costs, reports = [], []
    for b in range(B):
        binding = addresses[:len(field)] if addresses.ndim == 1 else addresses[:len(field), b]
        classes = sorted({e.row for e in records if bool(e.activation[b].any())})
        if not classes:
            costs.append(field.new_zeros(()))
            continue
        source_rows = sorted(set(binding[binding >= 0].tolist()) - set(classes))
        rows = field.new_tensor(source_rows + classes, dtype=torch.long)
        # This is the attended field plus its witnessed classes, never the
        # inventory squared. Code values and placement have no role here.
        compact = field.new_zeros(len(rows), 1, L, 2)
        for i, row in enumerate(rows.tolist()):
            at = binding == row
            if bool(at.any()):
                compact[i, 0] = field[at, b].amax(0)
        q = owner._concept_rung_presence(compact.detach(), len(source_rows), len(rows),
                                         inventory_rows=rows)[:, 0]
        teacher, mask = torch.zeros_like(q), torch.zeros_like(q, dtype=torch.bool)
        for e in records:
            if e.row not in classes:
                continue
            j = classes.index(e.row)
            value = torch.minimum(e.activation[b, :, None], e.evidence[None])
            teacher[j] = torch.maximum(teacher[j], value)
            mask[j] |= (e.activation[b, :, None] > 0) & (e.evidence[None] > 0)
        target = torch.maximum(q.detach(), teacher)
        costs.append(torch.where(mask, (q-target).square(), 0.).sum()/mask.sum().clamp_min(1))
        present = mask.any(-1)
        both = (q.detach().amin(-1) > owner.concept_use_floor) & present
        reports.append(dict(stream=b, classes=classes, readings=q.detach(), teacher=teacher,
                            witnessed=mask, both=int(both.sum()), count=int(present.sum())))
    owner._last_definition_fold = dict(examples=records, readings=reports,
        both=sum(r['both'] for r in reports), count=sum(r['count'] for r in reports))
    return torch.stack(costs)
