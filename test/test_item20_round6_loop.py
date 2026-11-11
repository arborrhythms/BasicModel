"""Direct loop certificates on the constructed native sentence path."""
import torch
import pytest


@pytest.mark.parametrize("separate", (False, True))
def test_sentence_trials_share_the_open_field_and_common_reverse(tmp_path, monkeypatch, separate):
    from test_native_attention_field import _model
    from SentenceField import sentence_generated
    from OutputOccurrences import realize
    import DerivationReconstruction
    from Language import SymbolSpace
    from test_compiled_word_chunk import _tiny_canonical_model
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=32, training_overrides={'reconstructionPlacement': 'eager'},
        architecture_overrides={'separateChooser': separate})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    fields, records = [], []
    original = model._sentence_path_cost
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        records.append(result[2]['record'])
        assert result[1][2].shape == (len(records[-1].root),)
        fields.append(records[-1].open_field)
        return result
    monkeypatch.setattr(model, '_sentence_path_cost', capture)
    try:
        source = model.inputSpace.prepInput(['alpha beta'])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=1,
                       batch_override=(source, torch.empty(1, 0)))
        assert fields and all(field is fields[0] for field in fields)
        field = fields[0]
        assert field.percepts is None
        def forbidden(*args, **kwargs):
            raise AssertionError('the input decoder entered System 2')
        monkeypatch.setattr(model, '_chart_generate_from_stm', forbidden)
        monkeypatch.setattr(SymbolSpace, 'reverse', forbidden)
        monkeypatch.setattr(DerivationReconstruction, 'reconstruct', forbidden)
        first = model.reverse(model.conceptualSpace.subspace.carrier_like(), field=field)
        second = model.reverse(model.conceptualSpace.subspace.carrier_like(), field=field)
        assert torch.equal(first.materialize(), second.materialize())
        assert torch.equal(first._reconstruction_percepts, second._reconstruction_percepts)
        inverse_calls = []
        inverse = model.languageSpace.reverse_binary_step
        def reversed_step(*args, **kwargs):
            inverse_calls.append(True)
            return inverse(*args, **kwargs)
        monkeypatch.setattr(model.languageSpace, 'reverse_binary_step', reversed_step)
        model.reverse(model.conceptualSpace.subspace.carrier_like(), field=field)
        assert not inverse_calls
        generation = sentence_generated(model, records[0])
        realize(model, generation)
        assert inverse_calls
        assert not hasattr(model.conceptualSpace.subspace, '_thought_occurrence')
        assert not hasattr(model, '_answer_surface_from_percepts')
        assert not hasattr(model.conceptualSpace, 'synthesis_layer')
        assert not hasattr(model.perceptualSpace, 'synthesis_layer')
        assert [step['operation'] for step in model._last_answer_construction.trace[-3:]] == [
            'generate:grammar', 'generate:occurrences', 'reverse:order_zero']
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_emptying_memory_before_the_open_does_not_change_input_reconstruction(tmp_path, monkeypatch):
    from test_native_attention_field import _model
    from Understanding import ConceptualField
    from torch.nn import functional as F
    model = _model(tmp_path, monkeypatch)
    owner = model._concept_owner()
    calls = []
    original = owner.cs_read_memberships
    def capture(percepts, brackets):
        calls.append((percepts, brackets))
        return original(percepts, brackets)
    monkeypatch.setattr(owner, 'cs_read_memberships', capture)
    try:
        with torch.no_grad():
            source = model.inputSpace.prepInput(['alpha beta'])
            model.runBatch(train=False, batchSize=1, batch_override=(source, torch.empty(1, 0)))
            percepts, brackets = calls[-1]
            def opened():
                a0 = original(percepts, brackets)
                return ConceptualField(owner._cs_field_concept_ids,
                    F.pad(a0, (0, 0, 0, 0, 0, 0, 0, sum(owner._field_caps()) - len(a0))),
                    owner._cs_field_where, owner._cs_field_when, None)
            before_field = opened()
            before = model.reverse(owner.subspace.carrier_like(), field=before_field)
            owner.stm.clear()
            owner.mtm.restore(None)
            after_field = opened()
            after = model.reverse(owner.subspace.carrier_like(), field=after_field)
        assert torch.equal(before_field.evidence, after_field.evidence)
        assert torch.equal(before.materialize(), after.materialize())
        assert torch.equal(before._reconstruction_percepts, after._reconstruction_percepts)
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_registry_uses_the_two_reconstructions_and_post_thought_target(tmp_path, monkeypatch):
    from test_native_attention_field import _model
    model = _model(tmp_path, monkeypatch)
    terms, targets = [], []
    original = model._sentence_path_cost
    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        terms.append(model._sentence_cost_registry._terms)
        return value
    monkeypatch.setattr(model, '_sentence_path_cost', capture)
    import ConceptSpread
    complete = ConceptSpread.complete
    def completed(*args, **kwargs):
        value = complete(*args, **kwargs)
        prediction = model._concept_prediction
        targets.append(prediction.target)
        owner = model._concept_owner()
        assert not prediction.target.requires_grad
        assert all(owner._row_order(int(row)) == 0 for row in prediction.rows)
        torch.testing.assert_close(prediction.target,
            owner._observed_knowing[:len(prediction.active), prediction.rows])
        assert 'expectation.concepts' in model._sentence_close_errors._terms
        return value
    monkeypatch.setattr(ConceptSpread, 'complete', completed)
    try:
        source = model.inputSpace.prepInput(['alpha beta'])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=1,
                       batch_override=(source, torch.empty(1, 0)))
        assert targets
        for registry in terms:
            assert all(prefix + '.' + lane in registry for prefix in
                ('reconstruction_reverse', 'reconstruction_order_zero') for lane in ('what', 'where', 'when'))
            for lane in ('form', 'for', 'against'):
                term = registry['reconstruction.walk_residual.' + lane]
                assert term['objective'] == 'reconstruction' and term['side'] == 'R'
            assert all(not registry[name]['trained'] for name in
                       ('reconstruction.free_bytes', 'reconstruction.coverage'))
            assert not any(name.startswith(('reconstruction.field', 'reconstruction.iterations',
                'reconstruction.forward_discard', 'reconstruction.content', 'reconstruction.walk_policy',
                'reconstruction.decomposition', 'reconstruction.compose_score_function')) for name in registry)
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_parallel_input_without_composed_sentences_books_only_field_reconstruction(tmp_path):
    from test_grounded_xor import grounded_model
    from ConceptLessons import admit_native_forms
    from Layers import Error
    model, inputs = grounded_model(tmp_path)
    try:
        admit_native_forms(model, ('0', '1'))
        understanding = model.understand(inputs)
        errors = Error()
        decoded, cost = model.reverseReconstruct(understanding, target=inputs, registry=errors)
        assert torch.isfinite(cost)
        assert set(errors._terms) == {'reconstruction_reverse.' + band
            for band in ('what', 'where', 'when')}
        field = understanding.reconstruction_carriers['field']
        again = model.reverse(model.conceptualSpace.subspace.carrier_like(), field=field)
        assert torch.equal(decoded, again.materialize())
        assert field.percepts is None
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_kept_input_reconstruction_selects_each_stream_and_its_graph():
    from Understanding import InputReconstruction
    from SentenceField import selected
    first = torch.arange(2., requires_grad=True)
    second = (torch.arange(2.) + 4).requires_grad_()
    def product(value):
        return InputReconstruction(value[:, None, None], value[:, None, None], value, value,
            value > 2, value[:, None], value[:, None, None])
    kept = selected(product(first), product(second), torch.tensor([False, True]))
    torch.testing.assert_close(kept.byte_cost, torch.tensor([0., 5.]))
    torch.testing.assert_close(kept.percepts[:, 0, 0], kept.byte_cost)
    a, b = torch.autograd.grad(kept.byte_cost.sum(), (first, second))
    torch.testing.assert_close(a, torch.tensor([1., 0.]))
    torch.testing.assert_close(b, torch.tensor([0., 1.]))
