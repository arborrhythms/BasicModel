"""The whole-sentence numerical path leaves native publication on the host."""
import copy

import torch


def test_parallel_compiled_body_keeps_native_publication_eager(monkeypatch):
    import util
    from test_mlx_export import _staged_input

    with torch.no_grad():
        model, staged = _staged_input()
    model.mask_rate = 0.0
    model._sentence_run_active = True
    reference = copy.deepcopy(model)
    with torch.no_grad():
        expected = reference.forward(staged)[2].clone()
    numerical_graphs = []
    def backend(graph, inputs):
        numerical_graphs.append(graph)
        return graph.forward
    compiled_names = []
    def capture(fn, **kwargs):
        compiled_names.append(fn.__name__)
        return torch.compile(fn, backend=backend, fullgraph=True)
    monkeypatch.setattr(util, 'compile', capture)
    published = []
    native_publish = model._publish_reading_symbols
    def publish(body):
        assert not torch.compiler.is_compiling(), 'native membership publication entered the tensor graph'
        published.append(True)
        return native_publish(body)
    monkeypatch.setattr(model, '_publish_reading_symbols', publish)
    model.enable_compiled_step()
    with torch.no_grad():
        actual = model._compiled_step(staged)[2]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert compiled_names == ['_forward_body', '_forward_head']
    assert len(numerical_graphs) == 2
    assert published == [True]
