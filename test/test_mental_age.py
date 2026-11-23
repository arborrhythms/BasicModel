"""Mental age is measured on the instance that will use it."""
import torch

from AgeChart import chart
from Assessment import assess_age, at_age
from Models import BaseModel
from test_compiled_word_chunk import _tiny_canonical_model


def test_a_higher_pass_cannot_bridge_one_missing_measure_at_the_same_age():
    from Assessment import _attained_age
    first = dict(age=.70, status='passed')
    one = dict(age=.75, status='passed')
    later = dict(age=1., status='passed')
    for incomplete in ('failed', 'unavailable'):
        other = dict(age=.75, status=incomplete)
        assert _attained_age([first, one, other, later]) == .70
        assert _attained_age([first, other, one, later]) == .70
    assert _attained_age([first, one, one, later]) == 1.
    assert _attained_age([dict(age=3.25, status='passed'), dict(age=3.75, status='failed')]) == 3.25


def test_chart_uses_the_committed_echo_and_all_measured_rows():
    rows = chart()
    assert len(rows) == 27
    assert [row[0] for row in rows] == sorted(row[0] for row in rows)
    _, name, cases = next(row for row in rows if row[0] == 1.)
    assert name == 'test_echo_a_word_with_its_own_look'
    assert len(cases) == len({case.text for case in cases}) == 19
    assert all(case.answer == case.text and case.prompt is None for case in cases)
    from test_item17_places import minimal
    _, two, sentences = minimal()
    by_name = {name: tuple(case.text for case in cases) for _, name, cases in rows}
    assert by_name['test_echo_a_sentence_of_two_words'] == two
    assert by_name['three-word lesson sentences'] == sentences
    assert len(by_name['four-word lesson sentences']) == 4


def test_assessment_uses_the_returned_answer_and_preserves_the_model(tmp_path, monkeypatch, eager_reading):
    m = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1)
    m._tensor_peer_while_eager = True
    try:
        assert float(m.age) == 0 and not dict(m.named_parameters())['age'].requires_grad
        original = {key: value.clone() for key, value in m.named_parameters() if key != 'age'}
        trie = m.perceptualSpace.percept_store.radix_trie
        trie_before = trie.serialize()
        table = m.inputSpace.data.text_answers
        supervised = m.inputSpace.data.has_supervised_outputs
        run = m.runBatch
        returned = []
        def observe(*args, **kwargs):
            question, = kwargs['questions']
            assert not kwargs['train'] and kwargs['optimizer'] is None
            result = run(*args, **kwargs)
            value = None if result[0] is None else result[0].outputPred
            spelling = None if value is None or value.is_floating_point() else bytes(
                value.detach().reshape(-1).tolist()).rstrip(b'\0').decode('latin1')
            returned.append((question.where, m.inputSpace.data.text_answers['train'][question.where], spelling))
            return result
        monkeypatch.setattr(m, 'runBatch', observe)
        report = assess_age(m, all_rows=False)
        assert returned == [(index, case['expected'], case['returned'])
                            for row in report['rows'] for index, case in enumerate(row.get('cases', ()))
                            if 'returned' in case]
        assert report['rows'] and report['rows'][0]['cases']
        levels = {row['age'] for row in report['rows']}
        earned = max((level for level in levels if all(row['status'] == 'passed'
                      for row in report['rows'] if row['age'] <= level)), default=0.)
        assert m.age.item() == m.age.new_tensor(earned).item()
        for row in report['rows']:
            if 'cases' in row:
                assert row['passed'] == sum(c['expected'] == c.get('returned') for c in row['cases'])
        assert trie.serialize() == trie_before
        assert m.inputSpace.data.text_answers is table
        assert m.inputSpace.data.has_supervised_outputs == supervised
        for key, value in m.named_parameters():
            if key != 'age':
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        assert at_age(m, earned, lambda: 'measured')['result'] == 'measured'
    finally:
        m.End()


def test_checkpoint_saves_age_and_an_older_checkpoint_inherits_xml(tmp_path, monkeypatch):
    load = BaseModel.load_weights
    path = str(tmp_path / 'age.ckpt')
    first = _tiny_canonical_model(tmp_path, monkeypatch, architecture_overrides={'age': 2.25})
    try:
        first.save_weights(path)
    finally:
        first.End()
    other = _tiny_canonical_model(tmp_path, monkeypatch, architecture_overrides={'age': .75})
    try:
        assert float(other.age) == .75
        assert load(other, path, strict=True)
        assert float(other.age) == 2.25
        legacy = torch.load(path, map_location='cpu', weights_only=False)
        del legacy['state_dict']['age']
        torch.save(legacy, path)
        assert load(other, path, strict=True)
        assert float(other.age) == .75
    finally:
        other.End()


def test_xor_assessment_uses_the_same_models_four_returned_answers():
    from test_mm_xor import _fresh_model
    model, _, _ = _fresh_model()
    try:
        report = assess_age(model)
        xor = report['rows'][-1]
        assert xor['measure'] == 'exclusive or'
        actual = torch.tensor(xor['returned']).reshape(-1)
        assert actual.numel() == 4
        mse = float((actual - torch.tensor([0., 1., 1., 0.])).square().mean())
        assert xor['status'] == ('passed' if mse < .20 else 'failed')
        assert report['age'] == 0.
    finally:
        model.End()


def test_identity_at_a_requested_age(tmp_path, monkeypatch, at_mental_age):
    model = _tiny_canonical_model(tmp_path, monkeypatch)
    try:
        def measure():
            raise AssertionError('an unqualified model entered the identity measurement')
        assert at_mental_age(model, 2.25, measure)['status'] == 'not qualified at that age'
    finally:
        model.End()


def test_counting_above_the_input_bound_reports_the_unavailable_presentation(tmp_path, monkeypatch, eager_reading):
    from Assessment import _text_row
    from test_item17_places import model_for
    model = model_for(tmp_path, monkeypatch)
    try:
        cases = next(cases for _, name, cases in chart() if name == 'count to nine')
        result = _text_row(model, cases)
        assert result['status'] == 'unavailable'
        assert result['passed'] == 0 and result['total'] == result['unavailable'] == 1
        assert '9 words' in result['cases'][0]['unavailable']
    finally:
        model.End()
