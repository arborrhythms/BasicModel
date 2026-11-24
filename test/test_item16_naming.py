"""Item 1.6, step D2: nouns, the name, ``createMetaConcepts``.

The certificates N1 to N7 of §34.1 of
doc/plans/2026-11-18-item1-6-felt-math-world.md, each printing what it
measured, on the text model of the echo test with the scene's box and the
key ``createMetaConcepts`` as the scene's configuration has them
(``data/scene/BasicModel_scene.xml``: false).

A word's concept and a value's concept, each known from its own
occurrences, each have a symbol of order one: the word's is read by the kind
of words, the value's is nominated with its concept
(``ImagePercepts.noun``). A pair of concepts of two modalities present
together in an opened field is counted for itself, and on the count that
nominates becomes a unit across modality, a Sigma of order one over the two
symbols (``FamiliarTier.observe_units``, ``promote_candidates``). Met alone,
either raises the unit, and by the way down the unit raises the other
(``FamiliarTier.raised``). With the key false, ``interpret`` of a word gives
the other member of its unit, or the word.

The test's configuration, beside the scene's two keys: MTM's promotion from
the opened field on (``ConceptualSpace.attentionPromotion``, false in the
text model), which a unit across modality is made by; and for N7, room for
category 1 and 2's words, values and places at order zero and their units at
order one (``ConceptualSpace.nVectors``, :data:`CORPUS_ROWS`), and for
what one opening reads (``architecture.serialWordBuckets``,
:data:`CORPUS_BUCKETS`).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

#: The concept rows of the model of N7 (the echo test's model has 64).
CORPUS_ROWS = 512
#: The serial word buckets of the model of N7 (the echo test's model has
#: '8'): an opening registers every concept it reads, and an image reads
#: the concepts of every lesser value known.
CORPUS_BUCKETS = '8,256'

WORD, OTHER_WORD = 'three', 'five'
#: The other pixel's value is the greater: by the Pi over values a lesser
#: value's concept reads for at a greater value's pixel, so a lesser one
#: would be present at every presentation of the word's own.
PIXEL, OTHER_PIXEL = [3, 2, 0, 200], [5, 5, 0, 240]


def scene_key():
    """``createMetaConcepts`` as the scene's configuration has it."""
    from test_item16_scene import _flat, _DATA, _SCENE
    return _flat(os.path.join(_DATA, _SCENE))["/architecture/createMetaConcepts"]


def naming_model(tmp_path, monkeypatch, *, key=None, concept_rows=None, word_buckets=None):
    """The echo test's text model with the scene's box, the key (by default
    the scene configuration's), MTM's promotion from the opened field, and
    ``concept_rows`` concept rows and ``word_buckets`` (the serial word
    buckets, whose greatest is the room for the concepts a sentence's
    opening reads) where given."""
    import test_compiled_word_chunk as built
    from test_item17_echo import echo_model
    from test_item16_scene import _declared_shape
    tiny = built._tiny_canonical_model

    def configured(*args, **kwargs):
        kwargs['conceptual_overrides'] = dict(kwargs.get('conceptual_overrides') or {}, attentionPromotion=True)
        if concept_rows is not None:
            kwargs['concept_rows'] = concept_rows
        if word_buckets is not None:
            kwargs['word_buckets'] = word_buckets
        return tiny(*args, **kwargs)
    monkeypatch.setattr(built, '_tiny_canonical_model', configured)
    try:
        model = echo_model(tmp_path, monkeypatch, architecture_overrides={
            "sceneShape": _declared_shape(), "createMetaConcepts": scene_key() if key is None else key})
    finally:
        monkeypatch.setattr(built, '_tiny_canonical_model', tiny)
    return model


class Fields:
    """Reads, never writes: the field each opening opened."""
    def __init__(self, monkeypatch):
        import SentenceField
        self.seen = []
        opening = SentenceField.open_field

        def opened(model, sid, a0):
            field = opening(model, sid, a0)
            if int(sid) == 0:
                self.seen.append(field)
            return field
        monkeypatch.setattr(SentenceField, 'open_field', opened)

    @property
    def last(self):
        return self.seen[-1]


def word_of(model, form):
    return model._concept_owner().interpret.word_for_form(form)


def value_of(model, steps):
    """The concept of order zero of the value ``steps`` / 255."""
    owner = model._concept_owner()
    basic = owner.basic_percepts
    from test_item16_scene_field import value_and_place_concepts
    values, _ = value_and_place_concepts(model)
    for cid in values:
        percepts = owner.concept_percepts(owner._concept_row_of(cid))
        if any(round(float(basic.felt([p])[0]) * 255) == steps for p in percepts if basic.is_value(p)):
            return cid
    return None


def symbols_of(model, concept):
    from Spaces import _concept_alloc_of
    return _concept_alloc_of(model._concept_owner()).symbol_table().reading(int(concept))


def reading(field, concept, owner):
    """What ``concept`` reads in ``field`` (for, against): nothing where it
    has no row there."""
    from SentenceField import inventory
    rows = inventory(owner, field)
    rows = (rows[:, 0] if rows.ndim == 2 else rows).tolist()
    row = owner._concept_row_of(int(concept))
    if row not in rows:
        return (0., 0.)
    return tuple(float(x) for x in field.evidence[rows.index(row), 0].detach().amax(0))


def made_directly(model):
    """What interpret makes directly: definitions of words and blank objects."""
    from Spaces import _concept_alloc_of
    owner = model._concept_owner()
    alloc = _concept_alloc_of(owner)
    blanks = [cid for cid in alloc.concepts if 'blank' in alloc.symbol_types.get(cid, ())]
    return dict(definitions=len(tuple(owner.definitions.word_ids)), blanks=len(blanks))


def unit_table(model):
    from FamiliarTier import units
    from Spaces import _concept_alloc_of
    owner = model._concept_owner()
    store = _concept_alloc_of(owner).layer(0)
    out = []
    for cid, order, members in units(owner):
        row = owner._concept_row_of(cid)
        entries = {member: [round(float(store.values[at]), 4) for at in
                            owner.concept_entries(row, owner._concept_row_of(member))] for member in members}
        out.append(dict(unit=cid, order=order, members=members, entries=entries))
    return out


# -- the stages of naming on one model: N1 (key false), N2, N3, N5, N6 ------------

@pytest.fixture(scope="module")
def stages(tmp_path_factory):
    """One model taken through the stages in order: a word alone and a pixel
    alone each known; each met alone; the two together as often as
    nominates and once more; each met alone again."""
    from test_item16_scene_field import present, known
    from reading_fixtures import use_eager_reading
    from Assessment import units_across_modality
    from FamiliarTier import raised
    record = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        model = naming_model(tmp_path_factory.mktemp("stages"), monkeypatch)
        use_eager_reading(monkeypatch)
        fields = Fields(monkeypatch)
        owner = model._concept_owner()
        word, pixel = dict(text=WORD, scene=None), dict(text='', scene=[PIXEL])
        try:
            known(model, [word])
            known(model, [pixel])
            known(model, [dict(text=OTHER_WORD, scene=None)])
            w, v = word_of(model, WORD), value_of(model, PIXEL[3])
            record['alone'] = dict(word=w, value=v, word_symbols=symbols_of(model, w),
                                   value_symbols=symbols_of(model, v), units=units_across_modality(model),
                                   made=made_directly(model))

            def met_alone():
                present(model, [word], train=False)
                by_word = fields.last
                present(model, [pixel], train=False)
                by_pixel = fields.last
                return dict(by_word=by_word, by_pixel=by_pixel,
                            value_down=raised(owner, by_word, [v])[v], word_down=raised(owner, by_pixel, [w])[w])
            record['before'] = met_alone()
            known(model, [dict(text=WORD, scene=[PIXEL])])
            record['unit'] = dict(units=units_across_modality(model), table=unit_table(model))
            after = met_alone()
            unit = record['unit']['table'][0]['unit'] if record['unit']['table'] else None
            after.update(unit_by_word=None if unit is None else reading(after['by_word'], unit, owner),
                         unit_by_pixel=None if unit is None else reading(after['by_pixel'], unit, owner))
            record['after'] = after
            other = word_of(model, OTHER_WORD)
            record['interpret'] = dict(
                word=owner.interpret.forward(w), word_reading=owner.interpret.reading(w)[0],
                other=other, other_word=owner.interpret.forward(other), value=v)
            record['made'] = made_directly(model)
        finally:
            model.End()
    return record


def test_n1_the_key(tmp_path, monkeypatch, stages):
    """With the key true a word read is given its blank object and its meta
    concept, as before; with it false (the scene's configuration) neither
    is made by interpret."""
    from test_item16_scene_field import known
    from reading_fixtures import use_eager_reading
    model = naming_model(tmp_path, monkeypatch, key='true')
    use_eager_reading(monkeypatch)
    try:
        known(model, [dict(text=WORD, scene=None)])
        word = word_of(model, WORD)
        made_true = made_directly(model)
        blank = model._concept_owner().interpret.blank_of(word)
    finally:
        model.End()
    made_false = stages['made']
    print(f"N1 key true: {made_true}, the word's blank object {blank}; key false ({scene_key()!r} in the "
          f"scene's configuration), after every stage: {made_false}")
    assert scene_key() == 'false'
    assert made_true['definitions'] >= 1 and made_true['blanks'] >= 1 and blank is not None
    assert made_false == dict(definitions=0, blanks=0)


def test_n2_each_known_alone(stages):
    alone = stages['alone']
    print(f"N2 the word {WORD!r} concept {alone['word']} symbols {alone['word_symbols']}; the pixel {PIXEL} "
          f"concept {alone['value']} symbols {alone['value_symbols']}; units across modality {alone['units']}")
    assert alone['word'] is not None and alone['value'] is not None
    assert alone['word_symbols'] and alone['value_symbols']
    assert alone['units'] == {}


def test_n3_the_unit(stages):
    unit, alone = stages['unit'], stages['alone']
    print(f"N3 after {WORD!r} with {PIXEL} as often as nominates: units across modality {unit['units']}; "
          f"{unit['table']}")
    assert unit['units'] == {1: 1}
    entry, = unit['table']
    assert set(entry['members']) == {alone['word'], alone['value']}
    # A Sigma over the two nouns' symbols: an entry on each, at one.
    assert all(any(weight > 0 for weight in weights) for weights in entry['entries'].values())


def test_n5_one_raises_the_other(stages):
    before, after = stages['before'], stages['after']
    print(f"N5 the word alone: the unit reads {after['unit_by_word']}, the object's concept by the way down "
          f"{after['value_down']} (before the unit {before['value_down']}); the pixel alone: the unit reads "
          f"{after['unit_by_pixel']}, the word's concept {after['word_down']} (before {before['word_down']})")
    assert after['unit_by_word'][0] > 0 and after['unit_by_pixel'][0] > 0
    assert after['value_down'][0] > before['value_down'][0]
    assert after['word_down'][0] > before['word_down'][0]


def test_n6_interpret(stages):
    got = stages['interpret']
    print(f"N6 interpret of {WORD!r} ({stages['alone']['word']}): {got['word']} (its reading {got['word_reading']}), "
          f"the object's concept {got['value']}; of {OTHER_WORD!r} ({got['other']}), with no unit: {got['other_word']}")
    assert got['word'] == got['value'] and got['word_reading'] == got['value']
    assert got['other'] is not None and got['other_word'] == got['other']


# -- N4: the pair that recurs --------------------------------------------------------

def test_n4_the_pair_that_recurs(tmp_path, monkeypatch):
    """The word with its own pixel at every presentation, and with another
    pixel too at fewer presentations than nominate."""
    from test_item16_scene_field import present, known
    from reading_fixtures import use_eager_reading
    from FamiliarTier import across
    model = naming_model(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    owner = model._concept_owner()
    try:
        known(model, [dict(text=WORD, scene=None)])
        known(model, [dict(text='', scene=[PIXEL])])
        known(model, [dict(text='', scene=[OTHER_PIXEL])])
        with_other = owner.concept_mint_count - 1
        sequence = ([dict(text=WORD, scene=[PIXEL, OTHER_PIXEL])] * with_other
                    + [dict(text=WORD, scene=[PIXEL])] * (owner.concept_mint_count + 1))
        for row in sequence:
            present(model, [row])
        word, own, other = word_of(model, WORD), value_of(model, PIXEL[3]), value_of(model, OTHER_PIXEL[3])
        united = across(owner, word)
        table = unit_table(model)
    finally:
        model.End()
    print(f"N4 {WORD!r} ({word}) with {PIXEL} ({own}) {len(sequence)} times, with {OTHER_PIXEL} ({other}) "
          f"{with_other} of them: united with {united}; units {table}")
    assert own in united and other not in united


# -- N7: the corpus (a measure) ------------------------------------------------------

def test_n7_the_corpus(tmp_path, monkeypatch):
    """Categories 1 and 2 (train) taught in order, each row once. Reported;
    nothing is asserted."""
    import felt_world_corpus as felt
    from test_item16_scene_field import present
    from reading_fixtures import use_eager_reading
    from Assessment import units_across_modality
    from FamiliarTier import units
    model = naming_model(tmp_path, monkeypatch, concept_rows=CORPUS_ROWS, word_buckets=CORPUS_BUCKETS)
    use_eager_reading(monkeypatch)
    owner = model._concept_owner()
    basic = owner.basic_percepts
    taught, refused = 0, []
    try:
        answers = set()
        for category in (1, 2):
            for document in felt.load(category, 'train'):
                rows, _, answer = felt.rows(document)
                if answer is not None:
                    answers.add(answer)
                for row in rows:
                    try:
                        present(model, [row])
                        taught += 1
                    except RuntimeError as error:
                        refused.append((category, row, str(error)[:80]))
        counts = units_across_modality(model)
        steps_of = {}
        for cid in {member for _, _, members in units(owner) for member in members}:
            row = owner._concept_row_of(cid)
            percepts = owner.concept_percepts(row) if row is not None else []
            values = [round(float(basic.felt([p])[0]) * 255) for p in percepts if basic.is_value(p)]
            if values:
                steps_of[cid] = values[0]
        by_word = {}
        from FamiliarTier import across
        for form in sorted(answers):
            word = word_of(model, form)
            united = [] if word is None else sorted(steps_of[m] for m in across(owner, word) if m in steps_of)
            by_word[form] = dict(word=word, steps=len(united), united=united)
        every = {}
        for cid in {member for _, _, members in units(owner) for member in members} - set(steps_of):
            every[owner.interpret.lexical_form(cid) or cid] = len([m for m in across(owner, cid) if m in steps_of])
        caps = tuple(int(c) for c in owner._order_caps())
    finally:
        model.End()
    print(f"N7 categories 1 and 2 (train), {taught} rows taught, {len(refused)} refused {refused[:3]}; "
          f"order caps {caps}; units across modality by order {counts}")
    print(f"N7 every word in a unit, with the count of steps it is united with: {every}")
    for form, entry in by_word.items():
        print(f"N7 {form!r}: united with {entry['steps']} steps {entry['united']}")
