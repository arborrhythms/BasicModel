"""Intensities in 0..1 at their input places, read as an image."""
from dataclasses import replace

import torch
from Understanding import ConceptualField


def intensities(numbers):
    """A reader's input as it is: each number an intensity from 0 to 1. A
    number below 0, above 1 or not a number is refused; numbers that are
    not in 0..1 are charted onto it before they are read (:func:`feeling`)."""
    if bool((torch.isnan(numbers) | (numbers < 0) | (numbers > 1)).any()):
        raise ValueError('a reader of intensities takes numbers from 0 to 1; '
                         'chart other numbers onto 0..1 (NumericPercepts.feeling) before they are read')
    return numbers


def feeling(numbers):
    """The logarithmic chart of numbers from zero to infinity onto 0..1, a
    step before a reader for numbers that are not intensities."""
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
    """The numeric stem: its input is an image, one intensity at each place
    of its box, read as an image's bank by the Pi over values."""
    def __init__(self, model, dimensions):
        self.model, self.dimensions = model, int(dimensions)

    def read(self, inputs):
        """Each place and the bracket over the input are readings. The input
        is an image's bank: each place whose intensity is above zero is an
        occurrence of its value's row of the table of values. Its places are
        met, and a place's row is nominated on the count that nominates a
        form; the promoted rows are read by the Pi over values, and the
        order-zero Sigma reads them through the one Not. The definitions
        above order zero are closed."""
        from FieldProjection import inventory, close
        from FamiliarTier import observe_intersection, promote_candidates
        from Places import meet
        owner = self.model._concept_owner()
        raw = inputs.reshape(len(inputs), -1)
        values = intensities(raw if raw.is_floating_point() else raw.float())
        if values.shape[1] != self.dimensions:
            raise ValueError('numeric input disagrees with its declared dimensions')
        basic = owner.basic_percepts
        B, A = len(inputs), self.dimensions
        axes = torch.arange(A, device=values.device)
        places = self.model.where_registry.intervals('numeric', axes.expand(B, -1))
        percepts = torch.full_like(places[..., 0], -1)
        for value in values.unique():
            if value > 0:
                percept = basic.perceive_value(value)
                percepts = torch.where(values == value, percept, percepts)
        bank = basic.value_bank(values, places, registry=self.model.where_registry,
                                when_encoding=self.model.when_encoding,
                                time=self.model.present())
        meet(owner, bank)
        nominated = [row for row in range(len(basic.index)) if bool(basic.promoted[row])]
        bank = replace(bank, rows=tuple(nominated), codes=basic.bank_codes(nominated).to(values.device))
        presence = basic.read_values(bank)
        if self.model.training:
            # The places' concepts read together at the bracket recur, and
            # their conjunction is nominated.
            concepts = {index: owner.concept_of_basic_percept(row) for index, row in enumerate(nominated)
                        if row in basic.place_rows}
            for batch in range(B):
                lanes = [(concept, 0) for index, concept in concepts.items()
                         if concept is not None and float(presence[index, batch, A, 0].detach()) > 0]
                observe_intersection(owner, lanes, batch=batch)
                owner.mtm.close(None, stream=batch, references=getattr(owner, '_field_activated_references', {}).get(batch, ()))
            promote_candidates(owner)
        rows = inventory(owner)
        evidence = owner.order_zero_field(presence, nominated, rows)
        evidence = close(owner, rows, evidence)
        ids = torch.tensor([owner.concept_id_at_row(row) for row in rows], device=values.device)
        when = torch.full_like(bank.where, self.model.present())
        return ConceptualField(ids, evidence, bank.where, when, percepts)

    def reverse(self, field):
        """The way back: each place's exact value, read from the table of
        values by the value's row the field holds there, where the field at
        that place's reading reads for (its for lane above zero) a concept
        of order zero that reads the value's row, or a place's row whose
        cone holds that place, by an entry that is not a Not; nothing
        elsewhere."""
        from Spaces import _concept_alloc_of
        owner = self.model._concept_owner()
        if field.percepts is None:
            raise ValueError('numeric reconstruction requires its field\'s value percepts')
        basic = owner.basic_percepts
        B, A = field.percepts.shape
        # The places each place's row holds: the places whose wires lie
        # within its cone, as the Pi over values reads them.
        wires = self.model.where_registry.band(field.where[0, :A, 0]).detach().cpu()
        holds = {row: ((wires >= plus - 1e-6) & (1 - wires >= minus - 1e-6)).all(-1)
                 for row, (plus, minus, _what) in basic.place_rows.items()}
        position = {int(cid): index for index, cid in enumerate(field.concept_ids.tolist())}
        readers = {}
        for concept, percept, negated in _concept_alloc_of(owner).order_zero_members:
            cid = owner.concept_id_at_row(concept)
            if not negated and cid is not None and int(cid) in position:
                readers.setdefault(percept, []).append(position[int(cid)])
        at = [[row for row, held in holds.items() if bool(held[a])] for a in range(A)]
        read = field.evidence[..., :A, 0].detach() > 0                       # [C, B, A]
        values = torch.zeros(field.percepts.shape, device=field.evidence.device, dtype=basic.value_codes.dtype)
        for b, a in (field.percepts >= 0).nonzero().tolist():
            row = int(field.percepts[b, a])
            concepts = list(readers.get(row, ()))
            concepts += [c for place in at[a] for c in readers.get(place, ())]
            if concepts and bool(read[concepts, b, a].any()):
                values[b, a] = basic.felt([row])[0].to(values)
        return values.to(field.evidence).reshape(B, *self.model.inputSpace.inputShape)

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
