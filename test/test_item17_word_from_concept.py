"""The shared input inverse spells a nominated word from its opening field."""
import torch

from test_item17_own_reading import COUNTING
from test_item17_places import LESSON, minimal, model_for, nominate, words_of

WORDS = words_of(LESSON) + COUNTING[:6]


def recreated_what(owner, value, *, places=True):
    """``value`` ``[order-zero concepts]`` -> the Sigma's way back -> each
    first-Pi row's kept `.what` (its for lane), weighted by the row's value
    above zero and summed. Returns the `.what` and the four greatest rows
    ``(value, row, is a place row)``. With ``places`` false the place rows'
    `.what` is left out of the sum (fourteenth pass, step 3)."""
    sigma = owner.install_order_zero()
    basic = owner.basic_percepts
    rows = list(range(len(basic.index)))
    values = sigma.reverse(value, rows)
    what = basic.kept_percept(rows)[0][:, 0]
    if not places:
        what = what * torch.tensor([r not in basic.place_rows for r in rows], dtype=what.dtype)[:, None]
    top = sorted(((round(float(v), 3), r, r in basic.place_rows)
                  for r, v in enumerate(values.tolist()) if v), reverse=True)[:4]
    return (values.clamp_min(0)[:, None] * what).sum(0), top


def concept_values(owner, field, reading):
    """A field's signed value (for less against) at ``reading`` on each
    order-zero concept row."""
    sigma = owner.install_order_zero()
    value = torch.zeros(sigma.nOutput)
    for i, cid in enumerate(field.concept_ids.tolist()):
        row = owner._concept_row_of(int(cid)) if cid >= 0 else None
        if row is not None and row < sigma.nOutput:
            value[row] = float(field.evidence[i, 0, reading, 0] - field.evidence[i, 0, reading, 1])
    return value


def test_the_leaf_filled_by_its_concepts_inverse(tmp_path, monkeypatch):
    import SentenceField
    from test_item18_expect import read
    model = model_for(tmp_path, monkeypatch)
    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson + (' '.join(COUNTING[:4]) + '.', ' '.join(COUNTING[4:]) + '.'))
        opened = SentenceField.reconstruction
        responses = []

        def measured(owner, field, **options):
            returned = opened(owner, field, **options)
            surface = bytes(returned.materialize()[0, 0].tolist()).rstrip(b'\0')
            responses.append((field.concept_ids.detach().flatten().tolist(), surface))
            return returned

        monkeypatch.setattr(SentenceField, 'reconstruction', measured)
        with torch.no_grad():
            for word in WORDS:
                responses.clear()
                read(model, word)
                model.End()
                (concept_ids, surface), = responses
                expected = model._concept_owner().interpret.word_for_form(word)
                print(word, expected, surface)
                assert expected in concept_ids and surface == word.encode()
    finally:
        model.End()
