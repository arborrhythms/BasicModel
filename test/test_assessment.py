"""The assessment observes the native read and leaves no presentation behind."""
import random
from copy import deepcopy
from dataclasses import fields, is_dataclass

import numpy as np
import pytest
import torch

from Assessment import named_tree, tree_text, read_sentence, format_record


def test_repeated_operations_are_counted_and_open_references_stay_marked():
    words = [dict(form='token', position=0)]
    tree = named_tree(['bind.open', ['bind.open', ['lower', 0]]], words)
    assert tree['count'] == 2 and tree['open']
    assert tree['children'][0]['count'] == 1
    assert tree_text(tree) == 'bind.open x2(lower(token))'


@pytest.mark.parametrize('count', (20, 21))
def test_ladder_checks_earlier_rungs_without_supplying_a_tree(monkeypatch, count):
    import AssessmentProbe as probe
    calls = []
    def read(model, text, **kwargs):
        calls.append((text, kwargs))
        word = dict(form=text, position=0, known=True, attention='interpreted', trusted=True, objects=[text])
        return dict(text=text, words=[word], forest=[0], trees=[dict(position=0, word=text)],
                    missing=[], meaning=None, context_meanings=[], answer=text, evidence_kind=None)
    monkeypatch.setattr(probe, 'read_sentence', read)
    vocabulary = dict(word='a', words=[str(i) for i in range(count)], name='a', verb='b')
    report = probe.known_probe(None, vocabulary, source='fixture')
    assert [row['status'] for row in report['rungs'][:4]] == ['passed', 'passed', 'passed', 'failed']
    assert report['rungs'][4]['status'] == 'not applicable'
    assert len(calls) == 3+3*count+1+1
    assert all(not {'tree', 'meaning', 'reference'}.intersection(kwargs) for _, kwargs in calls)
    assert probe.readiness(report, 4) == 'not ready: failed'


def test_failed_lower_rung_blocks_a_declared_later_rung(monkeypatch):
    import AssessmentProbe as probe
    calls = []
    def read(model, text, **kwargs):
        calls.append(text)
        return dict(text=text, words=[], forest=[], trees=[], missing=[0], meaning=None,
                    context_meanings=[], answer=None, evidence_kind=None)
    monkeypatch.setattr(probe, 'read_sentence', read)
    report = probe.known_probe(None, dict(word='a', words=[str(i) for i in range(20)], name='a'), source='fixture')
    assert report['rungs'][0]['status'] == 'failed'
    assert report['rungs'][1]['status'] == report['rungs'][2]['status'] == 'blocked'
    assert len(calls) == 3


def test_a_shared_verb_is_not_the_earlier_subject():
    from AssessmentProbe import _checks
    rung = dict(tree=['verb', ['bind', 0], 1], checks=['earlier_referent'])
    meaning = dict(references=[('sym', 2), None, None], children=[])
    record = dict(missing=[], forest=[rung['tree']], trees=[], words=[],
                  meaning=meaning, context_meanings=[meaning], context_subjects=[('sym', 1)])
    assert 'no earlier referent' in _checks(record, rung)
    meaning['references'][0] = ('sym', 1)
    assert not _checks(record, rung)


def test_native_lane_attribution_keeps_extent_identity_and_all_tied_sources():
    from types import SimpleNamespace
    from Assessment import _native_word_fields
    calls = []
    # Unsorted native identities; an unrelated concept must contribute neither
    # lane. The repeated word has different evidence at its two extents.
    field = torch.tensor([
        [[[.7, 0.], [.1, .2]]],
        [[[0., .8], [0., 0.]]],
        [[[.3, .8], [.9, .6]]],
        [[[1., 1.], [1., 1.]]]])
    def read(*args):
        calls.append(args)
        return field
    owner = SimpleNamespace(similarity_codebook=SimpleNamespace(getW=lambda: torch.zeros(1)),
        cs_read_memberships=read, _cs_field_concept_ids=torch.tensor([101, 103, 102, 999]),
        interpret=SimpleNamespace(word_for_form=lambda form: 101),
        definitions=SimpleNamespace(objects=lambda word: (102, 103)),
        _csw_row_of=lambda concept: concept-100)
    spans = torch.tensor([[[0, 1], [2, 3]]])
    model = SimpleNamespace(_concept_owner=lambda: owner, _staged_concepts_in=torch.zeros(1, 3),
        perceptualSpace=SimpleNamespace(_forward_input=dict(native_indices=torch.tensor([[0, 1]]),
            native_part_spans=spans)), wholeSpace=SimpleNamespace(inputShape=(3,),
            concept_evidence_layout=lambda *args: (spans, None),
            subspace=SimpleNamespace(what=SimpleNamespace(primitive_properties=None))))
    import ModelAttention
    original = ModelAttention.native_word_poles
    with _native_word_fields(model) as observations:
        actual = ModelAttention.native_word_poles(model, spans, [['token', 'token']],
            torch.ones(1, 2, dtype=torch.bool))
    assert ModelAttention.native_word_poles is original
    assert owner.cs_read_memberships is read and len(calls) == 1
    torch.testing.assert_close(actual, torch.tensor([[[.7, .8], [.9, .6]]]))
    first, second = observations[-1][0]
    assert first['span'] == [0, 1] and second['span'] == [2, 3]
    assert [source['concept'] for source in first['lanes']['for']['sources']] == [101]
    assert [source['concept'] for source in first['lanes']['against']['sources']] == [102, 103]
    assert first['lanes']['for']['sources'][0]['role'] == 'word'
    assert all(source['role'] == 'object' for source in first['lanes']['against']['sources'])
    assert [source['concept'] for source in second['lanes']['for']['sources']] == [102]
    assert [source['concept'] for source in second['lanes']['against']['sources']] == [102]
    assert {source['concept'] for source in first['concepts']} == {101, 102, 103}
    record = dict(text='token', words=[dict(form='token', position=0, known=True,
        attention='left', native_field=first)], trees=[], missing=[0], answer=None)
    printed = format_record(record)
    assert 'for' in printed and 'word 101' in printed and 'object 102' in printed and 'object 103' in printed


def snapshot(model):
    from CorpusCheckpoint import collect
    return dict(parameters={name: value.detach().clone() for name, value in model.named_parameters()},
        state=deepcopy(model.state_dict()), structures=deepcopy(model._collect_structural_extras()),
        continuation=collect(model), clock=model.when_time.clone(),
        rng=torch.random.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(),
        modes={name: module.training for name, module in model.named_modules()})


def equal(actual, expected):
    if torch.is_tensor(expected):
        if expected.layout != torch.strided:
            actual, expected = actual.to_dense(), expected.to_dense()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            equal(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal(a, b)
    else:
        assert actual == expected


def test_probe_restores_memory_clocks_parameters_and_random_streams(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1)
    model._tensor_peer_while_eager = True
    try:
        # "alpha" becomes a known word on its concept_mint_count-th reading
        # (rule 20); "novel" is read for the first time in the probe.
        from nomination_fixtures import readings_before_nomination
        readings_before_nomination(model, lambda: model.understand(model.inputSpace.prepInput(['alpha beta'])))
        model.understand(model.inputSpace.prepInput(['alpha beta']))
        model.End()
        model.train()
        before = snapshot(model)
        record = read_sentence(model, 'alpha novel', trace=True)
        assert [word['form'] for word in record['words']] == ['alpha', 'novel']
        assert record['words'][0]['known'] and not record['words'][1]['known']
        assert record['words'][0]['native_field']['concepts']
        assert 'tree' in format_record(record)
        equal(snapshot(model), before)
        assert read_sentence(model, 'alpha novel', trace=True) == record
        equal(snapshot(model), before)
        # Removing only the observer leaves the same native reading and walk.
        from contextlib import nullcontext
        import Assessment
        with monkeypatch.context() as patch:
            patch.setattr(Assessment, '_native_word_fields', lambda model: nullcontext([]))
            unobserved = read_sentence(model, 'alpha novel', trace=True)
        compared = deepcopy(record)
        for word in compared['words']:
            word['native_field'] = None
        assert unobserved == compared
        equal(snapshot(model), before)
        from reading_fixtures import read_in_source_order
        read_in_source_order(model, monkeypatch)
        contextual = read_sentence(model, 'alpha novel', context=('alpha beta',), trace=True)
        assert contextual['context_meanings']
        equal(snapshot(model), before)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_failed_probe_restores_the_same_state(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1)
    model._tensor_peer_while_eager = True
    try:
        model.understand(model.inputSpace.prepInput(['alpha']))
        model.End()
        original = model.understand
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            random.random(); np.random.random(); torch.rand(1)
            raise RuntimeError('after the ordinary read')
        monkeypatch.setattr(model, 'understand', fail)
        before = snapshot(model)
        with pytest.raises(RuntimeError, match='after the ordinary read'):
            read_sentence(model, 'novel')
        equal(snapshot(model), before)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_probe_restores_an_enclosing_trial_activity_journal(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    from MemoryActivity import trial_activity, accessed, snapshot as activity_snapshot
    from test_definition_rows import admit
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1)
    model._tensor_peer_while_eager = True
    try:
        word, obj = admit(model, 'alpha')
        owner = model._concept_owner()
        parts = owner._closed_clause_store().parts
        row = owner.definitions.row(word, obj)
        original = model.understand
        def read(*args, **kwargs):
            # A real relation read belongs to the probe's nested transaction.
            parts.evidence(row)
            return original(*args, **kwargs)
        monkeypatch.setattr(model, 'understand', read)
        @trial_activity(commit=True)
        def outer():
            accessed(parts, row)
            before = activity_snapshot()
            read_sentence(model, 'alpha')
            assert activity_snapshot() == before
        activity = float(parts.activity[row])
        outer()
        assert float(parts.activity[row]) == pytest.approx(min(1., activity + parts.activity_config.step))
    finally:
        model.End()
        model.symbolSpace.soft_reset()
