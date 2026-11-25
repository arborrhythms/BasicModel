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
        occurrence of the basic percept of intensity whose region holds it.
        A basic percept and its places are met, and each is nominated on the
        count that nominates a form and brings its concept then; so is each
        thing, adjacent atoms within the aperture, each an occurrence with a
        reading of its own (:meth:`BasicPercepts.BasicPercepts.with_things`);
        the promoted rows are read by the Pi over values, no value's row over
        more places than the aperture holds, and the order-zero Sigma reads
        them through the one Not. The definitions above order zero are
        closed."""
        from FieldProjection import inventory, close
        from FamiliarTier import observe_intersection, promote_candidates
        from Places import meet, nominate
        owner = self.model._concept_owner()
        raw = inputs.reshape(len(inputs), -1)
        values = intensities(raw if raw.is_floating_point() else raw.float())
        if values.shape[1] != self.dimensions:
            raise ValueError('numeric input disagrees with its declared dimensions')
        basic = owner.basic_percepts
        B, A = len(inputs), self.dimensions
        axes = torch.arange(A, device=values.device)
        places = self.model.where_registry.intervals('numeric', axes.expand(B, -1))
        percepts = basic.perceive_intensities(values)
        nominate(owner, percepts, 'value')
        bank = basic.value_bank(values, places, registry=self.model.where_registry,
                                when_encoding=self.model.when_encoding,
                                time=self.model.present())
        meet(owner, bank)
        # The things, runs of touching places, after the places are met.
        bank, things = basic.with_things(bank, percepts)
        nominate(owner, things, 'thing')
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
        evidence = close(owner, rows, evidence, (presence, nominated))
        ids = torch.tensor([owner.concept_id_at_row(row) for row in rows], device=values.device)
        when = torch.full_like(bank.where, self.model.present())
        return ConceptualField(ids, evidence, bank.where, when, percepts)

    def reverse(self, field):
        """The way back from what is read for: at each place, the centroid
        of the basic percept of intensity that a concept of order zero of the
        field reads, by an entry that is not a Not, where the field at that
        place's reading reads that concept for (its for lane above zero);
        nothing where none is. A row reads for only inside its region, so
        one is read for at a place. A thing's concept read for at a reading
        gives its parts back in their order at their places, each its
        centroid (:meth:`BasicPercepts.BasicPercepts.thing_back`); where a
        thing's and a pixel's both give a place back, the lesser."""
        from Spaces import _concept_alloc_of
        owner = self.model._concept_owner()
        basic = owner.basic_percepts
        A = self.dimensions
        found = field.evidence[..., 0].detach() > 0                          # [C, B, K]
        read = found[..., :A]                                                # [C, B, A]
        position = {int(cid): index for index, cid in enumerate(field.concept_ids.tolist())}
        readers, things = {}, {}
        for concept, percept, negated in _concept_alloc_of(owner).order_zero_members:
            cid = owner.concept_id_at_row(concept)
            if not negated and cid is not None and int(cid) in position:
                if basic.is_value(percept):
                    readers.setdefault(percept, []).append(position[int(cid)])
                elif percept in basic.things:
                    things.setdefault(percept, []).append(position[int(cid)])
        values = torch.zeros(read.shape[1:], device=read.device, dtype=basic.value_codes.dtype)
        for row, concepts in readers.items():
            here = read[concepts].any(0)
            values = torch.where(here, torch.maximum(values, basic.felt([row])[0].to(values)), values)
        if things:
            registry = self.model.where_registry
            lo, hi = registry.slices['numeric']
            degree = torch.stack([found[concepts].any(0) for concepts in things.values()]).to(values)
            back, given = basic.thing_back(list(things), degree, field.where, registry, lo, hi)
            back = back.to(values)
            values = torch.where(given & (values > 0), torch.minimum(values, back), torch.where(given, back, values))
        return values.to(field.evidence).reshape(len(values), *self.model.inputSpace.inputShape)

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
