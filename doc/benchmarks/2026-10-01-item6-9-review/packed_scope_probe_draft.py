"""Probe prepared while the policy workers hold the source snapshot."""
import pytest
import torch


@pytest.mark.parametrize('whitespace_mask', [False, True])
def test_sentence_inverse_does_not_pop_leaves_from_an_earlier_sentence(monkeypatch, whitespace_mask):
    from pathlib import Path
    import Models
    import util
    from test_mm_xor import _fresh_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    def eager_while(condition, body, values):
        while bool(condition(*values)):
            values = body(*values)
        return values
    monkeypatch.setattr(torch, 'while_loop', eager_while)
    model, _, _ = _fresh_model(str(Path(Models.__file__).resolve().parents[1] / 'data/XOR_grammar.xml'))
    try:
        isp = model.inputSpace
        # Two one-leaf sentences, optionally each followed by whitespace.
        width = 4 if whitespace_mask else 2
        isp._word_active_mask = torch.ones(1, width, dtype=torch.bool)
        isp._sentence_pack_enabled = True
        isp._packed_sentence_ids = torch.tensor([[0, 0, 1, 1]] if whitespace_mask else [[0, 1]])
        isp._ar_grammar_leaf_mask = (torch.tensor([[True, False, True, False]])
                                    if whitespace_mask else None)
        model._prepare_reconstruction_choices(1, width, torch.device('cpu'))
        dimension = int(model.conceptualSpace.stm.concept_dim)
        atoms = torch.arange(2 * dimension, dtype=torch.float32).reshape(1, 2, dimension) + 1
        roots = atoms.new_zeros(1, 2, 3 * dimension)
        roots[:, :, :dimension] = atoms
        depths = torch.ones(1, 2, dtype=torch.long)
        reference = atoms.new_zeros(1, width, dimension)
        reference[:, 0] = atoms[:, 0]
        second = 2 if whitespace_mask else 1
        reference[:, second] = atoms[:, 1]
        recovered, idea_cost, byte_cost, unavailable, costs = model._reconstruct_sentences(
            atoms[:, 1], reference, roots, depths,
            roots[:, 1].reshape(1, 3, dimension), depths[:, 1], torch.tensor(1),
            candidate_basis=(atoms, torch.ones(1, 2, dtype=torch.bool)), keep_ideas=True)
        assert not bool(unavailable.any())
        torch.testing.assert_close(recovered[:, second], atoms[:, 1], rtol=0, atol=0)
        assert not bool(recovered[:, :second].any())
        assert not bool(idea_cost.any())
        assert not bool(byte_cost.any())
        assert not bool(costs[:, 0].any())
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
