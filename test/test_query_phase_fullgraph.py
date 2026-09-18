"""The sentence-phase guard must preserve real fullgraph forward/backward."""

import torch

from Models import _ensure_grad_anchors
from test_compiled_word_chunk import _stage_fullgraph_tensor_peer, _tiny_canonical_model


def test_query_mask_preserves_real_fullgraph_forward_backward_across_lengths(tmp_path, monkeypatch):
    torch.manual_seed(313)
    model = _tiny_canonical_model(tmp_path, monkeypatch)
    model._prewarm_checkpoint_shapes()
    _ensure_grad_anchors(torch.device("cpu"))
    model._compiled_word_loop_fullgraph = True
    raw = _stage_fullgraph_tensor_peer(
        model, ["alpha beta gamma delta", "epsilon zeta"]
    )
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(
        lambda unused: model._forward_with_compiled_sentence_state(None),
        backend="eager", fullgraph=True,
    )
    try:
        first = model._publish_compiled_sentence_state(compiled(raw))
        assert model._query_sentence_depth == 0
        assert int(model._tensor_peer_trip_count) == 4
        (first[0].square().mean() + first[2].square().mean()).backward()
        assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) == 1
        model.zero_grad(set_to_none=True)
        model.End()
        model.symbolSpace.soft_reset()
        raw = _stage_fullgraph_tensor_peer(
            model, ["one two three four five six seven", "one two"]
        )
        second = model._publish_compiled_sentence_state(compiled(raw))
        assert model._query_sentence_depth == 0
        assert int(model._tensor_peer_trip_count) == 7
        (second[0].square().mean() + second[2].square().mean()).backward()
        assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) == 1
        assert all(
            parameter.grad is None
            or bool(torch.isfinite(
                parameter.grad.coalesce().values()
                if parameter.grad.is_sparse else parameter.grad
            ).all())
            for parameter in model.parameters()
        )
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
