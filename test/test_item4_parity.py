"""The sole §9.1 parity configuration does not disable prediction."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_defaults_and_declared_parity_configuration():
    parity = json.loads((ROOT/'test/fixtures/item4-parity.json').read_text())['training']
    for filename in ('BasicModel.xml', 'model.xml'):
        training = ET.parse(ROOT/'data'/filename).getroot().find('./architecture/training')
        assert training.findtext('expectationPrime') == 'true'
        assert float(training.findtext('explorationUniformFraction')) == .5
        for level in ('percept', 'symbol'):
            assert float(training.findtext(level+'ExpectationGain')) == parity[level+'ExpectationGain'] == 0.
    assert parity['sentenceExpectation'] and parity['wordExpectation']
    assert not parity['expectationPrime'] and parity['explorationUniformFraction'] == 1.


def test_parity_configuration_removes_selection_prime_without_muting_the_predictor(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model, _stage_fullgraph_tensor_peer
    from ModelAttention import stage_symbol_bank
    parity = json.loads((ROOT/'test/fixtures/item4-parity.json').read_text())['training']
    model = _tiny_canonical_model(tmp_path, monkeypatch, input_width=16, word_buckets='8',
        stm_capacity=3, chooser_depth=1, training_overrides=parity)
    try:
        model.eval()
        _stage_fullgraph_tensor_peer(model, ['cat', 'dog'])
        head = model.symbolSpace.expectation
        stage_symbol_bank(model, head)
        assert head.expectation_enabled and 'word' in head.enabled_levels
        assert not model.expectation_prime_enabled
        assert model._stm_reducer().exploration_fraction == 1.
        # Ported (eleventh pass): the context's projection that read the
        # order prime is gone; attend reads the conceptual field (rule 80),
        # so no prime reaches the request.
        import RequestRead
        assert 'prime' not in Path(RequestRead.__file__).read_text()
    finally:
        model.End(); model.symbolSpace.soft_reset()
