"""Reading derives word concepts and symbols under both supported bindings."""
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize('configuration', ['MM_xor.xml', 'XOR_grammar.xml', 'aligned'])
def test_every_read_word_has_a_concept_and_symbol(tmp_path, monkeypatch, configuration):
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    if configuration == 'aligned':
        from test_compiled_word_chunk import _tiny_canonical_model
        model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8')
        model.reconstruct_in_loop = False
        model.loss.reconstruction_scale = 0.
        model._tensor_peer_while_eager = True
        model._chart_compose_per_word = lambda: None
    else:
        from test_mm_xor import _fresh_model
        model, _, _ = _fresh_model(str(Path(__file__).resolve().parents[1] / 'data' / configuration))
    model.eval()
    sentences = ['hi no', 'hi go', 'we no', 'we go']
    try:
        inputs = model.inputSpace.prepInput(sentences)
        with torch.no_grad():
            _, symbols, _, _ = model(inputs)
        owner = model._concept_owner()
        words = sorted(set(' '.join(sentences).split()))
        assert all(owner.word_concepts(word) for word in words), {
            word: owner.word_concepts(word) for word in words}
        assert torch.isfinite(symbols).all()
        assert symbols.abs().sum() > 0, 'derived concepts must have symbol activations'
        if configuration != 'aligned':
            from ConceptEvidence import decode
            carrier = model.symbol_cache
            evidence = carrier._concept_activations
            assert evidence.any(), 'symbol bands cannot stand in for concept evidence'
            expected = decode(evidence, carrier._concept_codes)
            torch.testing.assert_close(symbols[..., :expected.shape[-1]], expected)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_word_publication_preserves_the_composed_concept_field():
    from types import SimpleNamespace
    from Models import BasicModel
    from ConceptEvidence import decode
    # A completed symbolic phase may know both poles and higher-order rows.
    # Publishing symbols must not replace that result with a new native read.
    evidence = torch.tensor([[[[1., 0.]]], [[[0., 1.]]], [[[1., 1.]]]])
    codes = torch.eye(3)
    carrier = SimpleNamespace(_concept_activations=evidence, _concept_codes=codes)
    def read_again(*args):
        pytest.fail('completed symbolic evidence was replaced by word admission')
    owner = SimpleNamespace(_sparse_active=lambda: True, cs_read_memberships=read_again)
    host = SimpleNamespace(_reading_word_percepts=object(), _reading_word_extents=None,
        _concept_owner=lambda: owner,
        symbolSpace=SimpleNamespace(forward_concept_to_symbol=lambda value:
            decode(value._concept_activations, value._concept_codes)))
    result = BasicModel._publish_reading_symbols(host, carrier)
    torch.testing.assert_close(result, decode(evidence, codes), atol=0, rtol=0)
