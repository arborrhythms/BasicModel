"""Nonnegative numbers as magnitudes of fixed perceptual axes."""
import torch
from Understanding import ConceptualField


def feeling(numbers):
    if bool((torch.isnan(numbers) | (numbers < 0)).any()):
        raise ValueError('numeric percepts require numbers from zero to infinity')
    g = torch.log1p(numbers.double())
    return torch.where(torch.isinf(g), 1., g/(1.+g)).to(numbers.dtype)


def number(values):
    values = values.double()
    return torch.expm1(values/(1.-values))


class NumericPercepts:
    """An adapter into the existing basic percept and concept inventories."""
    def __init__(self, model, dimensions):
        self.model, self.dimensions = model, int(dimensions)
        self.percepts, self.rows = [], []
        owner = model._concept_owner()
        alloc = owner._concept_allocator
        for axis in range(self.dimensions):
            percept = owner.basic_percepts.numeric_dimension(axis)
            cid = alloc.new_concept()
            alloc.add(cid, 'part', -axis-1)
            alloc.reference_orders[cid] = 0
            alloc.settle(cid)
            row = owner._csw_concept_row(0, cid)
            owner.add_concept_feature(row, 'bp', percept, 1.)
            with torch.no_grad():
                code = owner.basic_percepts.code(percept)
                owner.similarity_codebook.W[row].zero_()
                owner.similarity_codebook.W[row, :min(len(code), owner.similarity_codebook.W.shape[1])].copy_(code[:owner.similarity_codebook.W.shape[1]])
            self.percepts.append(percept)
            self.rows.append(row)

    def read(self, inputs):
        from FieldProjection import inventory, close
        from FamiliarTier import observe_intersection, promote_candidates
        owner = self.model._concept_owner()
        raw = inputs.reshape(len(inputs), -1)
        values = feeling(raw if raw.is_floating_point() else raw.float())
        if values.shape[1] != self.dimensions:
            raise ValueError('numeric input disagrees with its declared dimensions')
        rows = inventory(owner)
        evidence = values.new_zeros(len(rows), len(inputs), 1, 2)
        for axis, row in enumerate(self.rows):
            evidence[rows.index(row), :, 0, 0] = values[:, axis]
        if self.model.training and owner.conceptual_pi:
            for batch in range(len(inputs)):
                lanes = [(row, 0) for axis, row in enumerate(self.rows) if float(values[batch,axis]) > 0]
                observe_intersection(owner, lanes, batch=batch)
                owner.mtm.close(None, stream=batch, references=getattr(owner, '_field_activated_references', {}).get(batch, ()))
            promote_candidates(owner)
            expanded = inventory(owner)
            if expanded != rows:
                old = {row: evidence[i] for i, row in enumerate(rows)}
                evidence = torch.stack([old.get(row, evidence.new_zeros(len(inputs),1,2)) for row in expanded])
                rows = expanded
        evidence = close(owner, rows, evidence)
        ids = torch.tensor([owner.concept_id_at_row(row) for row in rows], device=values.device)
        where = torch.zeros(len(inputs), 1, 2, device=values.device, dtype=torch.long)
        when = torch.full_like(where, self.model.present())
        return ConceptualField(ids, evidence, where, when, None)

    def reverse(self, field):
        owner = self.model._concept_owner()
        ids = field.concept_ids
        values = []
        for row in self.rows:
            selected = ids == owner.concept_id_at_row(row)
            values.append(field.evidence[selected,...,0].amax((0,2)) if bool(selected.any())
                          else field.evidence.new_zeros(field.evidence.shape[1]))
        return number(torch.stack(values,-1)).to(field.evidence).unsqueeze(-1)

    def forward(self, inputs):
        from Spaces import SubSpace
        from FieldProjection import key
        # The numeric stem bypasses InputSpace's word staging, which normally
        # establishes the active batch for the common answer/thought owner.
        self.model.symbolSpace.ensure_microbatch(len(inputs), 1)
        field = self.read(inputs)
        owner = self.model._concept_owner()
        point = key(owner, field)
        carrier = SubSpace.from_tensor(point, inputShape=point.shape[1:], outputShape=point.shape[1:],
                                      nInputDim=point.shape[-1], nOutputDim=point.shape[-1])
        carrier._concept_ids, carrier._concept_activations = field.concept_ids, field.evidence
        self.model._combine_last_cs_sub = carrier
        self.model._numeric_field = field
        output = self.model.outputSpace.read_field(field, owner)
        return inputs, point, output, 0
