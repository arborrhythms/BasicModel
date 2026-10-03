"""The retained reading modes reconstruct; absent candidates are counted."""
from types import SimpleNamespace

import pytest
import torch

from Models import BasicModel


@pytest.mark.parametrize('synthesis,analysis,grammar,expected', [
    ('meronomy', 'meronomy', True, 'understanding'),
    ('meronomy', 'meronomy', False, 'perception: no grammar'),
])
def test_reconstruction_scope_follows_live_reading_and_grammar(synthesis, analysis, grammar, expected):
    model = SimpleNamespace(serial=True,
        perceptualSpace=SimpleNamespace(_meronomy=synthesis == 'meronomy'),
        wholeSpace=SimpleNamespace(analysis_mode=analysis),
        symbolSpace=SimpleNamespace(languageLayer=SimpleNamespace(
            operation_layer=object() if grammar else None)))
    assert BasicModel._understanding_reconstruction_scope(model) == expected


def test_candidate_coverage_counts_missing_sentences_without_hiding_present_ones():
    from test_reconstruction_bank_contract import _bank_fixture
    validate, isp = _bank_fixture('cpu')
    isp._ar_bank_valid[:, 1] = False
    validate()
    assert isp._reconstruction_sentence_available.tolist() == [[True, False, False]]
    assert int(isp._reconstruction_missing_sentence_count) == 1


def test_missing_candidate_cost_is_zero_with_zero_gradient():
    from test_word_store import _surface_snapshot_model
    model = _surface_snapshot_model()
    try:
        isp = model.inputSpace
        reference = isp._ar_word_object_atoms
        ready, bank, values, valid = model._snapshot_tables(reference)
        target_ready, target, mask = model._byte_tables(*reference.shape[:2])
        idea = reference[:, 0].detach().clone().requires_grad_(True)
        cost = model._byte_word_cost(idea, torch.tensor(0), bank, values,
                                    torch.zeros_like(valid), target, mask, ready and target_ready)
        assert torch.equal(cost, torch.zeros_like(cost))
        gradient, = torch.autograd.grad(cost.sum(), idea)
        assert torch.equal(gradient, torch.zeros_like(gradient))
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_missing_packed_sentence_does_not_dilute_the_owned_reconstruction(tmp_path, monkeypatch):
    import util
    from test_packed_reconstruction_parity import build_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = build_model(tmp_path, word_capacity=8)
    stage = model._stage_snapshot_bytes
    def without_second_sentence():
        stage()
        isp = model.inputSpace
        isp._ar_bank_valid[isp._ar_concept_lookup_sentence_ids == 1] = False
    monkeypatch.setattr(model, '_stage_snapshot_bytes', without_second_sentence)
    try:
        model._install_unit_span_fn()
        raw = model.inputSpace.prepPackedInput([['quorp flarn', 'wug blim']])
        with torch.no_grad():
            model.runBatch(train=False, batchSize=1, split='validation',
                batch_override=(raw, torch.empty(1, 0)))
        assert int(model.inputSpace._reconstruction_missing_sentence_count) == 1
        assert float(model._recon_sentence_costs[0, 1]) == 0.
        torch.testing.assert_close(model._recon_cost, model._recon_sentence_costs[:, 0])
    finally:
        model.End()
        model.symbolSpace.soft_reset()
