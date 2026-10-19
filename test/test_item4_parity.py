"""The sole §9.1 parity configuration does not disable prediction."""
import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_defaults_and_declared_parity_configuration():
    parity = json.loads((ROOT/'test/fixtures/item4-parity.json').read_text())['training']
    for filename in ('BasicModel.xml', 'model.xml'):
        training = ET.parse(ROOT/'data'/filename).getroot().find('./architecture/training')
        assert training.findtext('expectationPrime') == 'true'
        assert float(training.findtext('explorationUniformFraction')) == .5
        for level in ('percept', 'symbol', 'row', 'document'):
            assert float(training.findtext(level+'ExpectationGain')) == parity[level+'ExpectationGain'] == 0.
    assert parity['sentenceExpectation'] and parity['wordExpectation']
    assert not parity['expectationPrime'] and parity['explorationUniformFraction'] == 1.


def test_parity_configuration_removes_selection_prime_without_muting_the_predictor(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model, _stage_fullgraph_tensor_peer
    from ModelAttention import stage_symbol_bank
    from ModelCandidateAttention import context
    from meaning_fixtures import symbol_prediction
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
        predicted = replace(symbol_prediction(head._bank_keys, torch.ones(len(head._bank_keys), 2)),
                            symbol_ids=head._bank_ids)
        head._inter_last_meaning = [SimpleNamespace(prediction=predicted) for _ in range(2)]
        addresses = torch.zeros(2,1,dtype=torch.long)
        values = torch.ones(2,1,2)
        valid = torch.ones(2,1,dtype=torch.bool)
        before = context(model, values, addresses, valid)
        count = sum(model._candidate_geometry[0])
        assert not before[:,5*count:7*count].any()
        model.expectation_prime_enabled = True
        after = context(model, values, addresses, valid)
        assert after[:,5*count:7*count].any()
        torch.testing.assert_close(before[:,:5*count], after[:,:5*count], rtol=0, atol=0)
        torch.testing.assert_close(before[:,7*count:], after[:,7*count:], rtol=0, atol=0)
    finally:
        model.End(); model.symbolSpace.soft_reset()
