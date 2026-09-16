"""The native unpacked training boundary must survive dropped graph side effects."""
import pytest
import torch


@pytest.mark.parametrize("compiled", [True, False])
def test_unpacked_forward_observes_each_published_sentence_once(
        tmp_path, monkeypatch, compiled):
    from test_compiled_word_chunk import (
        _tiny_canonical_model, _stage_fullgraph_tensor_peer)
    from Models import _ensure_grad_anchors

    _ensure_grad_anchors(torch.device("cpu"))
    model = _tiny_canonical_model(tmp_path, monkeypatch)
    model.set_sentence_expectation(True)
    model._prewarm_checkpoint_shapes()
    model._compiled_word_loop_fullgraph = compiled
    model._tensor_peer_while_eager = not compiled
    model._chart_compose_per_word = lambda: None
    source = lambda _unused: model._forward_with_compiled_sentence_state(None)
    forward = torch.compile(source, backend="eager", fullgraph=True) if compiled else source
    discourse = model.symbolSpace.discourse
    try:
        for index, samples in enumerate((["alpha beta", "gamma delta"],
                                         ["alpha gamma", "beta delta"])):
            raw = _stage_fullgraph_tensor_peer(model, samples)
            model._active_compiled_step = forward if compiled else None
            result = forward(raw)
            assert len(result) == 21
            if compiled:
                # A compiler may discard an attribute-only escape. The
                # explicit outputs must be sufficient, on every invocation.
                model._pending_stm_end_state = None
            model._publish_compiled_sentence_state(result)
            model._publish_compiled_sentence_state(result)
            model._end_step()
            model._end_step()
            assert discourse.expectation_metrics()["observations"] == 2 * (index + 1)
            assert discourse.expectation_metrics()["predicted_targets"] == 2 * index
            for row in range(2):
                expected, occupied = discourse._canonical_meaning(
                    result[19][row], int(result[20][row]), "stm")
                if index:
                    comparison = discourse.last_expectation_comparison(row)
                    torch.testing.assert_close(comparison.observed, expected)
                    assert not comparison.observed.requires_grad
            if index:
                loss = discourse.consume_inter_loss()
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


def test_explicit_boundary_retains_depth_two_and_skips_masked_rows():
    from types import SimpleNamespace
    from Models import BasicModel
    from test_sentence_expectation import layer

    discourse = layer()
    slots = torch.arange(24.).reshape(2, 3, 4).requires_grad_()
    host = SimpleNamespace(
        symbolSpace=SimpleNamespace(discourse=discourse, ltm_store=None),
        conceptualSpace=SimpleNamespace(
            _ltm_consolidation=False,
            stm_end_state_trust=lambda *_: None),
        _pending_stm_end_state=(slots, torch.tensor([2, 3]), torch.tensor([True, False])),
        _expectation_documents_for_slot=lambda *_: ["doc", "padding"])
    BasicModel._drain_pending_stm_end_state(host)
    BasicModel._drain_pending_stm_end_state(host)
    assert discourse.expectation_metrics()["observations"] == 1
    assert len(discourse._inter_context[0]) == 1
    assert len(discourse._inter_context) == 1 or not discourse._inter_context[1]
    depth, payload, _ = discourse._inter_context[0][0]
    assert depth == 2
    expected, _ = discourse._canonical_meaning(slots[0], 2, "stm")
    torch.testing.assert_close(payload, expected)
