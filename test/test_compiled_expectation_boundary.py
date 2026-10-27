"""The native unpacked training boundary must survive dropped graph side effects."""
import pytest
import torch


@pytest.mark.slow
@pytest.mark.parametrize("gain", [0., .5])
@pytest.mark.parametrize("compiled", [True, False])
def test_unpacked_forward_observes_each_published_sentence_once(
        tmp_path, monkeypatch, compiled, gain):
    from test_compiled_word_chunk import (
        _tiny_canonical_model, _stage_fullgraph_tensor_peer)
    from Models import _ensure_grad_anchors

    _ensure_grad_anchors(torch.device("cpu"))
    model = _tiny_canonical_model(tmp_path, monkeypatch,
        input_width=16, word_buckets='8', stm_capacity=3, chooser_depth=1,
        training_overrides={f'{level}ExpectationGain': gain
                            for level in ('percept', 'symbol', 'row', 'document')})
    model.set_sentence_expectation(True)
    import util
    monkeypatch.setattr(util, "TheCompileBackend", "eager" if compiled else "none")
    model._tensor_peer_while_eager = not compiled
    model._chart_compose_per_word = lambda: None
    # The numerical word bricks compile; the common boundary stays eager.
    observed, costs = [], []
    original_score = model._sentence_path_cost
    def score(*args):
        result = original_score(*args)
        observed.append(result[2])
        # Expectation trains its predictor but no longer selects a trial.
        costs.append(model._sentence_cost_registry.total(objective='expectation'))
        return result
    monkeypatch.setattr(model, '_sentence_path_cost', score)
    discourse = model.symbolSpace.expectation
    try:
        for index, samples in enumerate((["alpha beta", "gamma delta"],
                                         ["alpha gamma", "beta delta"])):
            raw = _stage_fullgraph_tensor_peer(model, samples)
            with model._sentence_run():
                result = model._forward_with_compiled_sentence_state(raw)
            assert len(result) == 21
            if compiled:
                # A compiler may discard an attribute-only escape. The
                # explicit outputs must be sufficient, on every invocation.
                model._tensor_final_end_slots = None
                model._tensor_final_end_depth = None
            model._publish_compiled_sentence_state(result)
            model._publish_compiled_sentence_state(result)
            model._end_step()
            model._end_step()
            assert discourse.expectation_metrics()["observations"] == 2 * (index + 1)
            # The cover is the first predecessor in each stream.
            assert discourse.expectation_metrics()["predicted_targets"] == 2 * (index + 1)
            for row in range(2):
                expected = observed[-1]['observed'][row]
                if index:
                    comparison = discourse.last_expectation_comparison(row)
                    torch.testing.assert_close(comparison.observed, expected)
                    assert not comparison.observed.requires_grad
            if index:
                loss = costs[-1].mean()
                assert loss is not None and loss.requires_grad
                loss.backward()
                assert any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in discourse._inter_predictor.parameters())
            model.End()
            model.symbolSpace.soft_reset()
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_explicit_boundary_retains_one_entry_and_skips_masked_rows(monkeypatch):
    from Layers import SymbolExpectation
    from reading_fixtures import commit_reading, finish_reading
    from test_clause_acceptance import SentenceFixture
    fixture = SentenceFixture(monkeypatch)
    discourse = SymbolExpectation(n_symbols=8, max_depth=8, n_dim=8,
        concept_dim=8)
    discourse.set_symbol_bank(torch.eye(discourse.concept_dim))
    entry = fixture.program(('lift', 'cat', ('verb', 'chases', 'mouse')))
    for active in (True, False):
        commit_reading(fixture.language, fixture.registry, entry, None,
                       discourse=discourse, active=active, document='doc')
    assert discourse.expectation_metrics()["observations"] == 1
    assert len(discourse._inter_context[0]) == 1
    depth, payload, mask = discourse._inter_context[0][0]
    assert depth == 1
    clause = finish_reading(fixture.language, entry, registry=fixture.registry)
    from PartClosing import PartClosing
    closing = PartClosing(fixture.cs, fixture.store)
    expected = closing.prediction_point(closing.prepare(clause))
    torch.testing.assert_close(payload[0], expected)
    assert not payload[1:].any()
    assert mask.tolist() == [True, False, False]
