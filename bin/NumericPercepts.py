"""Nonnegative numbers as level percepts at their input places."""
import torch
from Understanding import ConceptualField


def feeling(numbers):
    if bool((torch.isnan(numbers) | (numbers < 0)).any()):
        raise ValueError('numeric percepts require numbers from zero to infinity')
    # MPS has no float64; its native float32 chart still preserves the
    # certificate's unit values within the same reconstruction tolerance.
    precision = torch.float32 if numbers.device.type == 'mps' else torch.float64
    g = torch.log1p(numbers.to(precision))
    return torch.where(torch.isinf(g), 1., g/(1.+g)).to(numbers.dtype)


def number(values):
    values = values.to(torch.float32 if values.device.type == 'mps' else torch.float64)
    return torch.expm1(values/(1.-values))


class NumericPercepts:
    """An adapter into the existing basic percept and concept inventories."""
    def __init__(self, model, dimensions):
        self.model, self.dimensions = model, int(dimensions)
        self.rows = []
        owner = model._concept_owner()
        alloc = owner._concept_allocator
        for axis in range(self.dimensions):
            cid = owner.new_concept(order=0)
            alloc.settle(cid)
            alloc.symbol_types.setdefault(cid, set()).update(('concept', 'object', 'percept_collection'))
            row = owner._csw_concept_row(0, cid)
            with torch.no_grad():
                owner.similarity_codebook.W[row].zero_()
            self.rows.append(row)

    def read(self, inputs):
        """Each axis place and the input's bracket are readings. A reading's
        active basic percepts are the level percepts it holds and the
        nominated conjunctions over them; the order-zero Sigma reads their
        combination, and the definitions above order zero are closed."""
        from FieldProjection import inventory, close
        from PerceptCollections import collect
        from FamiliarTier import observe_intersection, promote_candidates
        from OrderZero import combine
        owner = self.model._concept_owner()
        raw = inputs.reshape(len(inputs), -1)
        values = feeling(raw if raw.is_floating_point() else raw.float())
        if values.shape[1] != self.dimensions:
            raise ValueError('numeric input disagrees with its declared dimensions')
        basic = owner.basic_percepts
        B, A = len(inputs), self.dimensions
        axes = torch.arange(A, device=values.device)
        places = self.model.where_registry.intervals('numeric', axes.expand(B, -1))
        percepts = torch.full_like(places[..., 0], -1)
        for axis, row in enumerate(self.rows):
            observed = []
            for value in values[:, axis].unique():
                if value <= 0:
                    continue
                percept = basic.perceive_value(value, place=int(places[0, axis, 0]))
                percepts[:, axis] = torch.where(values[:, axis] == value, percept, percepts[:, axis])
                observed.append(percept)
            if observed:
                collect(owner, row, observed)
        if self.model.training:
            for batch in range(B):
                lanes = [(row, 0) for axis, row in enumerate(self.rows) if float(values[batch,axis]) > 0]
                observe_intersection(owner, lanes, batch=batch)
                owner.mtm.close(None, stream=batch, references=getattr(owner, '_field_activated_references', {}).get(batch, ()))
            promote_candidates(owner)
        # Readings: each axis place, then the bracket over the whole input.
        nominated = [row for row in range(len(basic.index)) if bool(basic.promoted[row])]
        presence = values.new_zeros(len(nominated), B, A + 1, 2)
        for index, row in enumerate(nominated):
            held = percepts == row
            presence[index, :, :A, 0] = held.to(values)
            presence[index, :, A, 0] = held.any(1).to(values)
        members = [('bp', row) for row in nominated]
        lanes = list(presence)
        for index, row in enumerate(nominated):
            if row in basic.conjunctions:
                lanes[index] = basic.pi(torch.stack(lanes), (row,), members)[0]
        presence = torch.stack(lanes) if lanes else presence
        codes = (basic.codes[nominated] if nominated else basic.codes[:0]).to(values)
        where = self.model.where_registry.encode('numeric', torch.cat((axes, axes[:1])).expand(B, -1)).to(values)
        moment = self.model.present()
        when = self.model.when_encoding.encode(torch.full((B, A + 1), moment, device=values.device)).to(values)
        combined = combine(presence, codes, where=where, when=when)
        rows = inventory(owner)
        evidence = owner.order_zero_field(combined, rows)
        evidence = close(owner, rows, evidence)
        ids = torch.tensor([owner.concept_id_at_row(row) for row in rows], device=values.device)
        bracket = torch.stack((places[:, 0, 0], places[:, -1, 1]), -1)
        where = torch.cat((places, bracket[:, None]), 1)
        when = torch.full_like(where, moment)
        return ConceptualField(ids, evidence, where, when, percepts)

    def reverse(self, field):
        owner = self.model._concept_owner()
        if field.percepts is None:
            raise ValueError('numeric reconstruction requires its field\'s level percepts')
        basic = owner.basic_percepts
        valid = field.percepts >= 0
        codes = basic.lookup_rows(field.percepts, valid=valid).to(field.evidence)
        values = codes[..., basic.identity_width - 1]
        present = torch.stack([field.evidence[field.concept_ids == owner.concept_id_at_row(row), :, axis, 0].amax(0) > 0
                               for axis, row in enumerate(self.rows)], -1)
        values = torch.where(present, values, 0.)
        return number(values).to(field.evidence).reshape(len(values), *self.model.inputSpace.inputShape)

    def forward(self, inputs):
        from Spaces import SubSpace
        from FieldProjection import key
        # The numeric stem bypasses InputSpace's word staging, which normally
        # establishes the active batch for the common answer/thought owner.
        self.model.symbolSpace.ensure_microbatch(len(inputs), 1)
        field = self.read(inputs)
        from IntuitionField import open_field
        bracket = torch.stack((field.where[..., 0].amin(1), field.where[..., 1].amax(1)), -1)
        field = open_field(self.model, field, bracket=bracket)
        owner = self.model._concept_owner()
        point = key(owner, field, order=0)
        carrier = SubSpace.from_tensor(point, inputShape=point.shape[1:], outputShape=point.shape[1:],
                                      nInputDim=point.shape[-1], nOutputDim=point.shape[-1])
        carrier._concept_ids, carrier._concept_activations = field.concept_ids, field.evidence
        self.model._combine_last_cs_sub = carrier
        self.model._numeric_field = field
        output = self.model.outputSpace.read_field(field, owner)
        return inputs, point, output, 0
