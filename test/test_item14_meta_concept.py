"""Item 1.4, the meta-concept by word/object learning (plan §15.1 and
§15.2; Alec, 2026-11-27).

A word and its object are concepts of two classes: here the word ``two``,
read as text, and the intensity 2, a pixel of 50 steps (the band of 2).
The word is linked to the object and the object to the word, at the
conceptual level, by conditioning: each is presented serially before the
other, and where the second is the model's answer to the first, its output
error is the penalty. Each link is a relation (M1: ``STM1``, ``MTM1``,
``LTM1``), ``word < object`` and ``object < word``, and causes its own MTM
promotion. With both, the two concepts are improper parts of one another
and cannot be sorted by order: the only resolution is an equivalence class
over them, one order up. That class is the meta-concept.

The model is the naming model of item 1.6 (``test_item16_naming``) with
``createMetaConcepts`` false, so that no meta-concept is made by rule on a
word's first reading. The stages are lettered, since M0 and M1 name the
tiers of memory (M0 the ideas: ``STM0``, ``MTM0``, ``LTM0``; M1 the
relations: ``STM1``, ``MTM1``, ``LTM1``), and each prints what it measured:

- A: the two concepts, of the two classes, each known alone;
- B: conditioning in both directions, each with its output error (the word
  the answer to the intensity, the intensity the answer to the word), links
  them both ways at M1;
- C: given both links at M1, however they came, the next readings resolve
  them into an equivalence class one order up, with neither above the other.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

#: The word, and its object: a pixel whose intensity is 2 (50 steps of 255).
WORD, PIXEL = 'two', [3, 2, 0, 50]
#: The object as an answer: the intensity in the output's range, 0 to 1.
LEVEL = PIXEL[3] / 255.
#: Rounds of serial presentation in each direction.
ROUNDS = 6


def concepts(model):
    """The word's concept and the intensity's."""
    from test_item16_naming import word_of, value_of
    return word_of(model, WORD), value_of(model, PIXEL[3])


def tiers(model, word, intensity):
    """What each tier of memory holds of the two concepts.

    M0, the ideas: the rows of ``LTM0``, how many name the word's concept in
    a slot, how many the intensity's, how many both, and the ideas MTM has
    staged. M1, the relations: ``word < intensity`` and ``intensity < word``
    in ``LTM1`` (each relation's row, or ``None``), and the rows on trial in
    MTM (``MTM1`` beside the Sigma, the pair rows beside the Pi) that hold
    both. ``STM0`` and ``STM1`` are a sentence's, and are not kept between
    readings.
    """
    from Spaces import _concept_alloc_of
    owner = model._concept_owner()
    alloc = _concept_alloc_of(owner)
    ideas = model.symbolSpace.ltm_store
    slots = ideas.episodic_concepts[:len(ideas)].tolist()
    staged = owner.mtm.__dict__.get('_staged_ideas')

    def on_trial(pool):
        if pool is None or not len(pool):
            return 0
        return sum(1 for pair in pool.refs[:len(pool)].tolist() if word in pair and intensity in pair)
    return dict(
        M0=dict(LTM0=len(ideas), naming_word=sum(word in row for row in slots),
                naming_intensity=sum(intensity in row for row in slots),
                naming_both=sum(word in row and intensity in row for row in slots),
                MTM0=0 if staged is None else len(staged)),
        M1=dict(LTM1=dict(word_in_intensity=alloc.parts.find(word, intensity),
                          intensity_in_word=alloc.parts.find(intensity, word)),
                MTM1=on_trial(getattr(owner.mtm, '_candidate_relations', None)),
                MTM_pairs=on_trial(getattr(owner.mtm, '_candidate_pairs', None))))


def classes_over(model, word, intensity):
    """The concepts that hold both as parts in ``LTM1``, with their orders."""
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(model._concept_owner())
    over = set(alloc.parts.wholes(word)) & set(alloc.parts.wholes(intensity))
    return {int(c): alloc.order_of(int(c)) for c in over - {word, intensity} if c in alloc.concepts}


def state(model, word, intensity):
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(model._concept_owner())
    return dict(tiers=tiers(model, word, intensity), classes=classes_over(model, word, intensity),
                orders=(alloc.order_of(word), alloc.order_of(intensity)))


def answered(model, monkeypatch, row, answer, *, train=True, optimizer=None):
    """``row`` read with ``answer`` as its supervised answer (the output
    error is the penalty); what the model answered."""
    from test_item17_echo import echo
    data = model.inputSpace.data
    monkeypatch.setattr(data, 'has_supervised_outputs', True)
    monkeypatch.setattr(data, 'text_answers', {'train': [answer], 'validation': []})
    try:
        batch = echo(model, 0, row, train=train, optimizer=optimizer or model.getOptimizer())
        return None if batch is None else batch.outputPred
    finally:
        model.End()
        monkeypatch.setattr(data, 'has_supervised_outputs', False)


@pytest.fixture(scope="module")
def taught(tmp_path_factory):
    """One model: the word and the intensity each known alone (A), then
    conditioned in both directions (B)."""
    from test_item16_naming import naming_model, said_back
    from test_item16_scene_field import known, present
    from test_item17_echo import answer_text
    from reading_fixtures import use_eager_reading
    record = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        model = naming_model(tmp_path_factory.mktemp("meta"), monkeypatch)
        use_eager_reading(monkeypatch)
        word_row, intensity_row = dict(text=WORD, scene=None), dict(text='', scene=[PIXEL])
        level = torch.tensor([[LEVEL]])
        try:
            known(model, [word_row])
            record['said'] = said_back(model, monkeypatch, [WORD])
            known(model, [intensity_row])
            word, intensity = concepts(model)
            record['concepts'] = (word, intensity)
            record['alone'] = None if None in (word, intensity) else state(model, word, intensity)
            if None not in (word, intensity):
                optimizer = model.getOptimizer()
                # The word, then the intensity, the intensity its answer. An
                # answer the model cannot score is recorded, and that
                # direction is then presented with no penalty.
                refused = None
                for _ in range(ROUNDS):
                    try:
                        if refused is None:
                            answered(model, monkeypatch, word_row, level, optimizer=optimizer)
                    except (RuntimeError, TypeError, NotImplementedError) as error:
                        refused = f'{type(error).__name__}: {error}'
                    if refused is not None:
                        present(model, [word_row])
                    present(model, [intensity_row])
                    # The intensity, then the word, the word its answer.
                    answered(model, monkeypatch, intensity_row, WORD, optimizer=optimizer)
                    present(model, [word_row])
                record['intensity_answer_refused'] = refused
                record['conditioned'] = state(model, word, intensity)
                with torch.no_grad():
                    record['answer_to_intensity'] = answer_text(
                        answered(model, monkeypatch, intensity_row, WORD, train=False))
        finally:
            model.End()
    return record


def test_a_a_word_and_an_intensity_of_two_classes(taught):
    """The word ``two`` and the intensity 2 are each a concept, of order
    zero, one a word and the other an intensity, and nothing yet links
    them at M1."""
    word, intensity = taught['concepts']
    alone = taught['alone']
    print(f"A said back {taught['said']}; the word {WORD!r} {word}, the intensity {PIXEL[3]} steps "
          f"{intensity}; alone: {alone}")
    assert word is not None and intensity is not None and word != intensity
    assert alone['orders'] == (0, 0)
    assert alone['tiers']['M1']['LTM1'] == dict(word_in_intensity=None, intensity_in_word=None)
    assert not alone['classes']


def test_b_conditioning_links_word_and_intensity_both_ways(taught):
    """After serial presentation in both directions, each with its output
    error, M1 holds the word as a part of the intensity and the intensity as
    a part of the word: kept in ``LTM1``, or on trial in MTM."""
    got = taught['conditioned']
    print(f"B after {ROUNDS} rounds each way: {got}; the intensity is answered with "
          f"{taught['answer_to_intensity']!r}; the intensity as the word's answer ({LEVEL:.4f}): "
          f"{'scored' if taught['intensity_answer_refused'] is None else taught['intensity_answer_refused']}")
    # Each direction has its output error.
    assert taught['intensity_answer_refused'] is None
    assert taught['answer_to_intensity'] == WORD
    relations = got['tiers']['M1']
    kept = relations['LTM1']['word_in_intensity'] is not None and relations['LTM1']['intensity_in_word'] is not None
    assert kept or relations['MTM1'] or relations['MTM_pairs']


def test_c_mutual_parts_resolve_into_an_equivalence_class(tmp_path, monkeypatch):
    """With both relations in ``LTM1`` (written here, so that this stage
    does not wait on B), the next readings make one concept one order up
    over the two, and neither is ordered above the other."""
    from Spaces import _concept_alloc_of
    from test_item16_naming import naming_model
    from test_item16_scene_field import known, present
    from reading_fixtures import use_eager_reading
    model = naming_model(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    word_row, intensity_row = dict(text=WORD, scene=None), dict(text='', scene=[PIXEL])
    try:
        known(model, [word_row])
        known(model, [intensity_row])
        word, intensity = concepts(model)
        assert None not in (word, intensity)
        alloc = _concept_alloc_of(model._concept_owner())
        alloc.add(intensity, 'part', ('sym', word))      # word < object
        alloc.add(word, 'part', ('sym', intensity))      # object < word
        linked = state(model, word, intensity)
        for _ in range(model._concept_owner().concept_mint_count + 1):
            present(model, [word_row])
            present(model, [intensity_row])
        after = state(model, word, intensity)
    finally:
        model.End()
    print(f"C both relations written: {linked}; after the readings: {after}")
    assert None not in linked['tiers']['M1']['LTM1'].values()
    # Neither is above the other: the two keep one order.
    assert after['orders'][0] == after['orders'][1]
    # One class over the two, one order up.
    assert len(after['classes']) == 1
    (_, order), = after['classes'].items()
    assert order == after['orders'][0] + 1
