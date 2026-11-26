"""Intensities out.

The certificates F1 to F4 of §42.1 of
doc/plans/2026-11-18-item1-6-felt-math-world.md, each printing what it
measured, on the text model of the echo test with the scene's box
(``test_item16_scene_field``'s). ``reconstructModalities`` gives, beside the
sentence and the value beside the words, the image: for each row the
intensities over the scene's box, or nothing where the row had no image. It
is read from the field the image was read into, by the one way back
(``ImagePercepts.put_back``), and from nothing kept from the input.
"""
import os
from dataclasses import replace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch


@pytest.fixture
def model(tmp_path, monkeypatch):
    from reading_fixtures import use_eager_reading
    from test_item16_scene_field import scene_model
    m = scene_model(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    yield m
    m.End()


def out(model, row):
    """The understanding of ``row`` read once, not trained, and what
    ``reconstructModalities`` gives out from it."""
    model.eval()
    try:
        with torch.no_grad():
            understanding = model.understand(model.inputSpace.prepInput([row]))
            return understanding, model.reconstructModalities(understanding)
    finally:
        model.End()
        model.train()


def centroids(model, image):
    """``image`` ``[places]`` with each pixel's value replaced by its basic
    percept's centroid."""
    basic = model._concept_owner().basic_percepts
    back = torch.zeros_like(image)
    for i in image.nonzero().flatten().tolist():
        row = basic.perceive_intensity(image[i].to(basic.value_codes), 0)
        back[i] = basic.felt([row])[0].to(back)
    return back


def lit(image):
    return [(int(i), round(float(image[i]), 4)) for i in image.nonzero().flatten()]


THING = [[1, 1, 0, 51], [2, 1, 0, 128], [3, 1, 0, 51]]
LONE = [[5, 5, 0, 200]]


# -- F1: an image out ----------------------------------------------------------------

def test_f1_an_image_out(model):
    """An image of a thing of three places and a lone pixel, presented until
    it is known, comes out place by place to its percepts' centroids: from
    every concept of the field, and from the thing's concepts alone at the
    thing's places."""
    import ImagePercepts
    from test_item16_scene_field import known, the_felt
    from test_item16_things import _thing_concepts, _only
    row = dict(text='', scene=THING + LONE)
    known(model, [row])
    understanding, result = out(model, row)
    image, = result['image']
    felt = the_felt(model, row['scene'])
    expected = centroids(model, felt)
    field = understanding.reconstruction_carriers['image_field']
    concepts = _thing_concepts(model._concept_owner())
    alone = ImagePercepts.given(model, _only(field, concepts))[0]
    thing = the_felt(model, THING)
    print(f"F1 {row['scene']}: out {lit(image)}; centroids {lit(expected)}; felt {lit(felt)}; "
          f"thing's concepts {sorted(concepts)}, from them alone {lit(alone)}")
    assert concepts
    assert torch.equal(image, expected)
    assert torch.equal(alone, centroids(model, thing))


# -- F2: both out --------------------------------------------------------------------

def test_f2_both_out(model):
    """A sentence with its image gives its words out as it does without the
    image, and the image beside them."""
    from test_item17_echo import text
    from test_item17_places import nominate
    from test_item16_scene_field import category, known, the_felt
    rows, _, _ = category(2)[0]
    row = rows[0]
    nominate(model, (row['text'],))
    known(model, [row])
    _, alone = out(model, row['text'])
    _, both = out(model, row)
    words, words_alone = text(both['input'][0]), text(alone['input'][0])
    image, = both['image']
    felt = the_felt(model, row['scene'])
    print(f"F2 {row!r}: words out {words!r}, without the image {words_alone!r}; image out {lit(image)}, "
          f"felt {lit(felt)}; without the image, image out {alone['image']}")
    assert words == words_alone
    assert torch.equal(image, centroids(model, felt))
    assert alone['image'] == ()


# -- F3: none ------------------------------------------------------------------------

def test_f3_none(model):
    """A row with no image gives no image out; an empty image gives one with
    nothing in it."""
    from test_item16_scene_field import known
    from test_item17_places import nominate
    import ImagePercepts
    known(model, [dict(text='', scene=LONE)])
    sentence = 'this is eight.'
    nominate(model, (sentence,))
    _, nulled = out(model, dict(text=sentence, scene=None))
    _, empty = out(model, dict(text='', scene=[]))
    lo, hi = model.where_registry.slices['scene']
    image, = empty['image']
    # The model reads one stream, so a batch of a row with an image and a
    # row with none is the image's field given to two rows, the second's
    # readings in the box made the empty brackets a row with no image has.
    understanding, lone = out(model, dict(text='', scene=LONE))
    field = understanding.reconstruction_carriers['image_field']
    where = field.where.expand(2, -1, -1).clone()
    inside = (where[1, :, 0] >= lo) & (where[1, :, 1] <= hi)
    where[1, inside] = lo
    evidence = field.evidence.expand(-1, 2, -1, -1).clone()
    evidence[:, 1, inside] = 0
    ids = field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids.expand(-1, 2)
    pair = ImagePercepts.given(model, replace(field, concept_ids=ids, evidence=evidence, where=where,
                                              when=field.when.expand(2, -1, -1), wires=None))
    print(f"F3 a row with scene null: image out {nulled['image']}; an empty image: "
          f"{tuple(image.shape)} places, {int(image.count_nonzero())} lit; a batch of {LONE} and a row "
          f"with no image: {[None if p is None else lit(p) for p in pair]}")
    assert nulled['image'] == ()
    assert image.shape == (hi - lo,) and not bool(image.any())
    assert pair[1] is None and torch.equal(pair[0], lone['image'][0]) and bool(pair[0].any())


# -- F4: from the field --------------------------------------------------------------

def test_f4_from_the_field(model):
    """With the field's evidence at nothing, nothing comes out."""
    from Understanding import Understanding
    from test_item16_scene_field import known
    row = dict(text='', scene=THING + LONE)
    known(model, [row])
    understanding, result = out(model, row)
    field = understanding.reconstruction_carriers['image_field']
    carriers = dict(understanding.reconstruction_carriers,
                    image_field=replace(field, evidence=torch.zeros_like(field.evidence)))
    silent = replace(understanding, reconstruction_carriers=carriers)
    assert isinstance(silent, Understanding)
    model.eval()
    try:
        with torch.no_grad():
            image, = model.reconstructModalities(silent)['image']
    finally:
        model.train()
    print(f"F4 {row['scene']}: out {lit(result['image'][0])}; with the field's evidence at nothing "
          f"{int(image.count_nonzero())} of {image.numel()} places lit")
    assert bool(result['image'][0].any())
    assert not bool(image.any())
