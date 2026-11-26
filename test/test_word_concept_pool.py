"""MTM chooses words in reading order without spelling or host decisions."""
import copy
import random

import pytest
import torch

from WordConceptPool import WordConceptPool


@pytest.fixture(params=['cpu', 'mps'])
def device(request):
    if request.param == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    return torch.device(request.param)


def test_nomination_is_visible_in_its_reading_and_not_earlier_streams(device):
    pool = WordConceptPool(32, torch.tensor([71, 75], device=device), threshold=3)
    forms = torch.tensor([[4, 4, -1], [4, 8, 8], [4, 8, -1]], device=device)
    words, arrivals = pool.read(forms)
    assert words.tolist() == [[-1, -1, -1], [-1, -1, -1], [71, -1, -1]]
    assert arrivals[arrivals[..., 0] >= 0].tolist() == [[4, 71]]
    assert pool.counts[[4, 8]].tolist() == [3, 2]
    words, arrivals = pool.read(forms)
    assert words.tolist() == [[71, 71, -1], [71, 75, 75], [71, 75, -1]]
    assert arrivals[arrivals[..., 0] >= 0].tolist() == [[8, 75]]


def test_pool_matches_sequential_readings_including_capacity(device):
    rng = random.Random(1505)
    pool = WordConceptPool(20, torch.tensor([51, 52, 53, 54], device=device), threshold=3)
    counts, words, used = {}, {}, 0
    for _ in range(15):
        rows = [[rng.randrange(-1, 20) for _ in range(9)] for _ in range(4)]
        expected, new = [], []
        for row in rows:
            seen, result = set(), []
            for form in row:
                if form >= 0 and form not in words:
                    if form not in seen:
                        counts[form] = counts.get(form, 0) + 1
                    if counts[form] >= 3 and used < 4:
                        words[form] = 51 + used
                        used += 1
                        new.append([form, words[form]])
                seen.add(form)
                result.append(words.get(form, -1))
            expected.append(result)
        result, arrivals = pool.read(torch.tensor(rows, device=device))
        assert result.tolist() == expected
        assert arrivals[arrivals[..., 0] >= 0].tolist() == new
        assert int(pool.used) == used


def test_frozen_read_and_checkpoint_keep_the_pending_count(device):
    pool = WordConceptPool(20, torch.tensor([51], device=device), threshold=2)
    forms = torch.tensor([[3, 3, -1]], device=device)
    pool.read(forms)
    saved = copy.deepcopy(pool.state_dict())
    result, arrivals = pool.read(forms, learn=False)
    assert result.tolist() == [[-1, -1, -1]]
    assert (arrivals == -1).all()
    for name, value in saved.items():
        assert torch.equal(value, pool.state_dict()[name])
    restored = WordConceptPool(20, torch.tensor([51], device=device), threshold=2)
    restored.load_state_dict(saved)
    a, b = restored.read(forms), pool.read(forms)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert a[0].tolist() == [[51, 51, -1]]


def test_fullgraph_pool_updates_match_eager(device):
    pool = WordConceptPool(20, torch.tensor([51, 52, 53], device=device), threshold=2)
    other = copy.deepcopy(pool)
    captured = torch.compile(other.read, backend='eager', fullgraph=True)
    for rows in ([[3, 3, 2], [2, 9, -1]], [[9, 2, 3], [4, 4, 4]]):
        forms = torch.tensor(rows, device=device)
        eager, compiled = pool.read(forms), captured(forms)
        assert all(torch.equal(a, b) for a, b in zip(eager, compiled))
        assert all(torch.equal(value, other.state_dict()[name])
                   for name, value in pool.state_dict().items())


def test_empty_or_exhausted_pool_keeps_forms_pending(device):
    pool = WordConceptPool(20, torch.empty(0, dtype=torch.long, device=device), threshold=1)
    result, arrivals = pool.read(torch.tensor([[3, -1]], device=device))
    assert result.tolist() == [[-1, -1]]
    assert (arrivals == -1).all()
    result, arrivals = pool.read(torch.empty((2, 0), dtype=torch.long, device=device))
    assert result.shape == (2, 0) and arrivals.shape == (2, 0, 2)


@pytest.mark.usefixtures('eager_reading')
@pytest.mark.parametrize('text', ['love', 'ab' * 20])
def test_sentence_uses_mtm_name_across_a_change_of_visible_cover(monkeypatch, text):
    from pathlib import Path
    from test_mm_xor import _fresh_model, _PROJECT
    model = _fresh_model(str(Path(_PROJECT) / 'data/XOR_grammar.xml'))[0]
    owner = model._concept_owner()
    original = owner.interpret.lookup_word
    nominations = []

    def admitted(*args, **kwargs):
        assert kwargs.get('pool_word') is not None, 'sentence used spelling nomination'
        result = original(*args, **kwargs)
        nominations.append(result)
        return result

    with torch.no_grad():
        for reading in range(2):
            with monkeypatch.context() as patch:
                patch.setattr(owner.interpret, 'lookup_word', admitted)
                model.understand(model.inputSpace.prepInput([text] * 4))
            fi = model.perceptualSpace._forward_input
            if not reading:
                words = fi['pool_word_ids'][:, 0].tolist()
                assert words[:2] == [-1, -1]
                word = words[2]
                assert word > 0 and words[3] == word
                assert int(fi['word_part_mask'][0, 0].sum()) == min(len(text), 32)
            else:
                assert fi['pool_word_ids'][:, 0].tolist() == [word] * 4
                assert int(fi['word_part_mask'][0, 0].sum()) == 1
            model.End()
            model.dispatch_per_row_reset([True] * 4)
            model.perceptualSpace.admit_counted_forms()
    assert nominations == [word]
    assert owner.interpret.word_for_form(text) == word


def test_mtm_checkpoint_restores_word_names_and_pending_readings(device):
    from Memory import MediumTermMemory
    memory = MediumTermMemory()
    memory.word_pool = WordConceptPool(20, torch.tensor([51, 52], device=device), threshold=2)
    forms = torch.tensor([[3, 3, 4]], device=device)
    memory.word_pool.read(forms)
    state = memory.state()
    restored = MediumTermMemory()
    restored.restore(state, device=device)
    expected = memory.word_pool.read(forms)
    actual = restored.word_pool.read(forms)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


@pytest.mark.usefixtures('eager_reading')
def test_legacy_pending_readings_keep_the_same_nomination_sentence():
    from pathlib import Path
    from test_mm_xor import _fresh_model, _PROJECT
    from nomination_fixtures import lexical_forms, short_of_nomination
    model = _fresh_model(str(Path(_PROJECT) / 'data/XOR_grammar.xml'))[0]
    owner = model._concept_owner()
    raw, parts = lexical_forms(model, ['love'])[0]
    short_of_nomination(owner.interpret, parts, model.wholeSpace.property_rows_for_bytes(raw), form=raw)
    with torch.no_grad():
        model.understand(model.inputSpace.prepInput(['love', '', '', '']))
    word = model.perceptualSpace._forward_input['pool_word_ids'][0, 0]
    assert int(word) > 0
    assert owner.interpret.word_for_form('love') == int(word)
    assert ('form', 'love') not in owner.mtm.sightings


def test_no_free_concept_row_leaves_a_nomination_pending(device):
    pool = WordConceptPool(20, torch.tensor([51, 52], device=device), threshold=1)
    forms = torch.tensor([[4, 3]], device=device)
    words, _ = pool.read(forms, available=forms.new_zeros(()))
    assert words.tolist() == [[-1, -1]] and int(pool.used) == 0
    words, arrivals = pool.read(forms, available=forms.new_ones(()))
    assert words.tolist() == [[51, -1]]
    assert arrivals.tolist() == [[[4, 51], [-1, -1]]]
