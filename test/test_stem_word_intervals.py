"""§68.73: brackets use the intervals the stem already computed."""
import ast
import inspect

import pytest
import torch


def test_stage_input_does_not_search_or_reparse_the_surface():
    import ModelAttention
    tree = ast.parse(inspect.getsource(inspect.unwrap(ModelAttention.stage_input)))
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(call, ast.Attribute) and call.attr in
                   ('find', 'percept_spans', 'word_spans') for call in calls)


@pytest.mark.usefixtures('eager_reading')
@pytest.mark.parametrize('surface', ['one one', 'hello world'])
def test_whole_slab_keeps_repeated_and_cut_word_intervals(surface, monkeypatch):
    from reading_fixtures import field_text_config, keep_existing_references
    from test_mm_xor import _fresh_model
    import Meronomy
    import ModelAttention
    keep_existing_references(monkeypatch)
    with field_text_config() as config:
        model, _, _ = _fresh_model(config)
    model.eval()
    computed, staged = [], []
    split = Meronomy.percept_spans
    stage = ModelAttention.stage_input

    def observed_split(raw):
        result = split(raw)
        computed.append(result)
        return result

    def observed_stage(current):
        fi = current.perceptualSpace._forward_input
        intervals = fi['word_intervals']
        assert any(intervals[0] is value for value in computed)
        assert current.inputSpace._ar_word_part_offsets is None
        stage(current)
        width = current._attention_spans.shape[1]
        expected = torch.tensor([intervals[0] + [(0, 0)] *
                                 (width - len(intervals[0]))])
        torch.testing.assert_close(current._attention_spans.cpu(), expected)
        staged.append(current._attention_spans.clone())

    monkeypatch.setattr(Meronomy, 'percept_spans', observed_split)
    monkeypatch.setattr(ModelAttention, 'stage_input', observed_stage)
    try:
        with torch.no_grad():
            model(model.inputSpace.prepInput([surface]))
        assert staged
        if surface == 'hello world':
            # Eight emitted parts end inside the second word; its interval
            # still comes from the complete surface at the stem.
            assert model.inputSpace._ar_embedded_N.shape[1] == 8
            assert staged[0][0, 1].tolist() == [6, 11]
    finally:
        model.End()
