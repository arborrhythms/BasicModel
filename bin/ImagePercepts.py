"""An image of intensities read with its sentence.

At a sentence's opening a row that carries an image (``InputSpace._scenes``)
has an image's bank: each pixel an occurrence of the basic percept of
intensity whose region holds it, at its place of the box ``scene``. Its
values are perceived and its places met, as the numeric stem does. The
image's rows, read by the Pi over values
(:meth:`BasicPercepts.BasicPercepts.read_values`), and
the sentence's, read by the first Pi over forms
(:meth:`BasicPercepts.BasicPercepts.read_bank`), stand in one table of
lanes, the image's readings after the sentence's: a row of one reads
nothing at a reading of the other. The table goes to the one Not and the
Sigma of order zero in one call (``cs_read_percept_field``).

The image's readings are the places its pixels hold, each a place of the
box, and the bracket over the box. A row with no image has none: its
columns are empty brackets. An empty image has the bracket alone.

At the opening's reconstruction the image is put back from the field alone
(:func:`put_back`) and charged against what was felt (:func:`charge`).
"""
import torch
from torch.nn import functional as F
from PerceptField import PerceptField


def imaged(model, sid):
    """Whether this opening reads an image: the first sentence of a batch
    in which some row carries one."""
    scenes = model.inputSpace._scenes
    return int(sid) == 0 and scenes is not None and any(scene is not None for scene in scenes)


def image_bank(model):
    """The images of the batch as one bank (:meth:`value_bank`), with the
    intensity of each occurrence ``[B, A]``: each row's pixels in the order
    of their addresses, padded with places that hold nothing; the bracket of
    a row with an image is the box, of a row with none an empty one."""
    owner = model._concept_owner()
    registry = model.where_registry
    basic = owner.basic_percepts
    lo, hi = registry.slices['scene']
    origin = registry.boxes['scene'][0]
    scenes = model.inputSpace._scenes
    device = owner.similarity_codebook.W.device
    B, A = len(scenes), max(1, max(len(scene) for scene in scenes if scene is not None))
    values = torch.zeros(B, A, device=device, dtype=basic.value_codes.dtype)
    addresses = torch.full((B, A), lo, device=device, dtype=torch.long)
    ends = addresses.clone()
    for b, scene in enumerate(scenes):
        if not scene:
            continue
        at = registry.address(torch.tensor([[o + p for o, p in zip(origin, (x, y, z))]
                                            for x, y, z, _ in scene], device=device))
        order = at.argsort()
        n = len(scene)
        addresses[b, :n] = at[order]
        ends[b, :n] = at[order] + 1
        values[b, :n] = torch.tensor([value for *_, value in scene], device=device,
                                     dtype=values.dtype)[order]
    places = torch.stack((addresses, ends), -1)
    bank = basic.value_bank(values, places, registry=registry, when_encoding=model.when_encoding,
                            time=model.present())
    has = torch.tensor([scene is not None for scene in scenes], device=device)
    bracket = torch.where(has[:, None], torch.tensor([lo, hi], device=device), lo)
    from dataclasses import replace
    where = torch.cat((bank.where[:, :-1], bracket[:, None]), 1)
    # Every reading of an image is at the moment it is felt.
    return replace(bank, where=where, when=torch.full_like(where, int(model.present()))), values


def perceive(model, bank, values):
    """The image's values perceived, each basic percept of intensity
    nominated on the count that nominates a form and its concept brought then, and its places met,
    as the numeric stem does."""
    from Places import meet, bring
    owner = model._concept_owner()
    basic = owner.basic_percepts
    percepts = basic.perceive_intensities(values)
    for percept in percepts.unique().tolist():
        if percept >= 0:
            for b in (percepts == percept).any(1).nonzero().flatten().tolist():
                if owner.nominates(('value', percept), owner.reading(b)):
                    bring(owner, percept, 'value')
    meet(owner, bank)


def image_rows(model):
    """The promoted rows of the image's space: the values' rows, the rows
    of the places of the box, and the conjunctions over them alone."""
    owner = model._concept_owner()
    basic = owner.basic_percepts
    registry = model.where_registry
    lo, hi = registry.slices['scene']
    promoted = [row for row in range(len(basic.index)) if bool(basic.promoted[row])]
    places = [row for row in promoted if row in basic.place_rows]
    inside = set()
    if places:
        cones = torch.stack([basic.place_rows[row][0] for row in places])
        at = registry.decode(cones.to(registry.grid))
        inside = {row for row, address in zip(places, at.tolist()) if lo <= address < hi}
    own = {row for row in promoted if basic.is_value(row) or row in inside}
    own |= {row for row in promoted if row in basic.conjunctions
            and all(member in own for member in basic.conjunctions[row])}
    return [row for row in promoted if row in own]


def joined(model, bank, sentence, image, values):
    """The sentence's percept field (``sentence``, of the sentence's
    ``bank``) and the image's (of its bank ``image`` of the intensities
    ``values``) in one table of lanes, and each reading's ``.where`` wires:
    the image's readings after the sentence's, a row of one space reading
    nothing at a reading of the other."""
    from dataclasses import replace
    from Places import readings
    from util import TheXMLConfig
    basic = model._concept_owner().basic_percepts
    rows = image_rows(model)
    image = replace(image, rows=tuple(rows), codes=basic.bank_codes(rows).to(values.device))
    lanes = basic.read_values(image,                                                  # [R, B, Ki, 2]
                              where_width=int(TheXMLConfig.get('architecture.intensityWhereWidth')))
    events = basic.value_events(image) if rows else image.table.new_zeros(0, *image.token.shape, 2)
    own = set(rows)
    keys = tuple(sentence.keys) + tuple(('bp', row) for row in rows if ('bp', row) not in sentence.keys)
    N, B, K = len(keys), sentence.values.shape[1], sentence.values.shape[2]
    Ki, Oi = lanes.shape[2], events.shape[2]
    kept = torch.tensor([key[1] not in own for key in sentence.keys], device=lanes.device)
    position = torch.tensor([keys.index(('bp', row)) for row in rows], device=lanes.device, dtype=torch.long)
    pad = lambda x, n: F.pad(x, (0, 0, 0, 0, 0, 0, 0, n))
    value = torch.cat((pad(sentence.values * kept[:, None, None, None].to(sentence.values), N - len(sentence.keys)),
                       lanes.new_zeros(N, B, Ki, 2).index_copy(0, position, lanes.to(sentence.values))), 2)
    event = torch.cat((pad(sentence.events * kept[:, None, None, None].to(sentence.events), N - len(sentence.keys)),
                       events.new_zeros(N, B, Oi, 2).index_copy(0, position, events.to(sentence.events))), 2)
    boundaries = None if sentence.boundaries is None else F.pad(
        sentence.boundaries * kept[None, :, None, None, None].to(sentence.boundaries),
        (0, 0, 0, Ki, 0, 0, 0, N - len(sentence.keys)))
    when_band = None if sentence.when_band is None else torch.cat(
        (sentence.when_band, model.when_encoding.encode(image.when[..., 0]).to(sentence.when_band)), 1)
    occurrence = torch.arange(Oi, device=image.token.device)[None] + bank.token.shape[1]
    real = image.token >= 0
    perception = PerceptField(keys, value, event, torch.cat((sentence.spans, image.span.to(sentence.spans)), 1),
        torch.cat((sentence.where, image.where.to(sentence.where)), 1),
        torch.cat((sentence.when, image.when.to(sentence.when)), 1),
        sentence.percept_where, when_band, boundaries, sentence.parts,
        torch.cat((sentence.reading, torch.where(real, image.token + K, -1)), 1),
        torch.cat((sentence.index, torch.where(real, occurrence, -1)), 1))
    wires = torch.cat((readings(bank)[0], readings(image)[0].to(bank.table)), 1)
    return perception, wires


def read(model, sid, bank):
    """The order-zero concepts of a sentence's opening: of its bank alone
    (``cs_read_bank``) where no row of the batch has an image; otherwise of
    its bank and the image's, joined before the one Not."""
    owner = model._concept_owner()
    if not imaged(model, sid):
        return owner.cs_read_bank(bank)
    # The image is perceived before either is read: a row it nominates is
    # written before the reading that holds it.
    image, values = image_bank(model)
    perceive(model, image, values)
    perception, wires = joined(model, bank, owner.basic_percepts.read_bank(bank), image, values)
    field = owner.cs_read_percept_field(perception)
    object.__setattr__(owner, '_cs_field_wires', wires)
    return field


def put_back(model, field):
    """The image put back from the field alone, ``[B, places of the box]``:
    at each place, the greatest over the values' rows of the row's centroid
    times whether a concept of the field reads for it at that place's
    reading (its for lane above zero), the derivative of that for lane
    passed through; nothing where none reads."""
    from Spaces import _concept_alloc_of
    owner = model._concept_owner()
    basic = owner.basic_percepts
    lo, hi = model.where_registry.slices['scene']
    where = field.where                                                                  # [B, E, 2]
    B, E = where.shape[:2]
    place = (where[..., 0] >= lo) & (where[..., 1] <= hi) & (where[..., 1] - where[..., 0] == 1)
    found = field.evidence[..., 0]                                                       # [C, B, E]
    degree = (found > 0).to(found) + (found - found.detach())
    readers = {}
    for concept, percept, negated in _concept_alloc_of(owner).order_zero_members:
        if not negated and basic.is_value(percept):
            cid = owner.concept_id_at_row(concept)
            if cid is not None:
                readers.setdefault(percept, set()).add(int(cid))
    ids = field.concept_ids if field.concept_ids.ndim == 2 else field.concept_ids[:, None].expand(-1, B)
    rows = list(readers)
    # [V, C, B]: whether the field's concept reads the value's row.
    member = torch.tensor([[[int(cid) in readers[row] for cid in column] for column in ids.T.tolist()]
                           for row in rows], dtype=torch.bool, device=found.device)
    member = member.reshape(len(rows), B, ids.shape[0]).transpose(1, 2)
    value = basic.felt(rows).to(found) if rows else found.new_zeros(0)
    candidates = (degree[None] * member[..., None].to(degree)).amax(1) * value[:, None, None]   # [V, B, E]
    back = torch.cat((found.new_zeros(1, B, E), candidates)).amax(0) * place
    index = (where[..., 0] - lo).clamp(0, hi - lo - 1)
    spread = F.one_hot(index, hi - lo).to(back) * place[..., None].to(back)            # [B, E, P]
    return (back[..., None] * spread).amax(1)


def felt(model, device, dtype):
    """Each row's image over the box, ``[B, places of the box]``: a pixel's
    value at its place, nothing where none is; and whether the row has an
    image ``[B]``."""
    registry = model.where_registry
    lo, hi = registry.slices['scene']
    origin = registry.boxes['scene'][0]
    scenes = model.inputSpace._scenes
    image = torch.zeros(len(scenes), hi - lo, device=device, dtype=dtype)
    for b, scene in enumerate(scenes):
        for x, y, z, value in scene or ():
            address = int(registry.address(torch.tensor([o + p for o, p in zip(origin, (x, y, z))])))
            image[b, address - lo] = value
    return image, torch.tensor([scene is not None for scene in scenes], device=device)


def charge(model, errors, sid, field, active):
    """The image's term of the opening's errors, ``reconstruction_order_zero.image``:
    the squared difference place by place over the box between what is put
    back and what was felt, against what was felt (``Error.squared``),
    weighted as the sentence's reconstruction is. A row with no image is
    charged nothing for one. ``None`` where this opening reads no image."""
    if not imaged(model, sid):
        return None
    back = put_back(model, field)
    image, has = felt(model, back.device, back.dtype)
    errors.squared('reconstruction_order_zero.image', back, image, mask=active.to(has.device) & has,
                   weight=model.loss.reconstruction_scale, category='reconstruction')
    return back
