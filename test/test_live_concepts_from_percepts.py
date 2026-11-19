"""Live order-zero handoff: addressed evidence, the word's Sigma, and exact carriers."""
import copy

import pytest
import torch
from torch import nn

from Layers import SigmaConceptsFromPercepts
from OrderZero import OrderZeroSigma, read_word
from Spaces import ConceptualSpace


def _space():
    cs = ConceptualSpace.__new__(ConceptualSpace)
    nn.Module.__init__(cs)
    cs.nWhat, cs.nWhere, cs.nWhen = 2, 1, 1
    cs.concept_dim = 4
    cs.expect_stack = cs.idea_stack = None
    cs.inputShape = cs.outputShape = [2, 4]
    # A basic percept's vector is two wide; a location's place is two wide.
    cs.order_zero = OrderZeroSigma(4, 16, content_width=2)
    return cs


def test_identified_evidence_masks_padding_without_erasing_observed_negative_match():
    cs = _space()
    part = torch.tensor([[[.8, 0., 3., 4.]]], requires_grad=True)
    whole = torch.tensor([[[0., .7, 5., 6.], [.6, 0., 7., 8.]]], requires_grad=True)
    references = torch.tensor([[[1., 0.], [0., 1.], [1., 0.], [0., 1.],
                                [0., 0.], [0., 0.], [0., 0.], [0., 0.]]])
    roles = torch.tensor([[0, 1, 1, 0, -1, -1, -1, -1]])
    evidence, mask = cs.percept_code_evidence(
        part, whole, references, references, roles, torch.tensor([True]),
        part_n_what=2, whole_n_what=2)
    assert mask[0, 0, :4].all() and not mask[0, 0, 4:].any()
    assert not mask[0, 1, 0] and not mask[0, 1, 3]  # no padded part observation
    assert evidence[0, 0, 0] > 0 and evidence[0, 0, 3] < 0
    evidence.sum().backward()
    assert part.grad is not None and whole.grad is not None
    assert torch.isfinite(part.grad).all() and torch.isfinite(whole.grad).all()
    assert torch.count_nonzero(part.grad[..., :2]) > 0
    assert torch.count_nonzero(whole.grad[..., :2]) > 0
    assert torch.count_nonzero(part.grad[..., 2:]) == 0  # bands are not evidence


def test_live_reduction_uses_sigma_once_and_preserves_bands_not_prior_mean():
    cs = _space()
    parts = (torch.tensor([[[.8, 0., 3., 4.]]]),)
    wholes = (torch.tensor([[[0., .7, 5., 6.], [.6, 0., 7., 8.]]]),)
    evidence = torch.zeros(1, 2, 8, requires_grad=True)
    with torch.no_grad():
        evidence[0, 0, :2] = torch.tensor([.8, .7])
        evidence[0, 1, 1] = .6
    mask = torch.zeros_like(evidence, dtype=torch.bool)
    mask[0, 0, :2] = True
    mask[0, 1, 1] = True
    rows = torch.tensor([[3]])
    atoms = torch.tensor([[[1., -1., 999., 999.]]])
    prior = torch.full((1, 2, 4), 99.)
    column = torch.tensor([.5, .25, .1, -.1], requires_grad=True)
    code = torch.tensor([.6, 0.])
    reading = torch.cat((column, code))[None]
    event, orders, row, activation, returned_prior, validity, locations = (
        cs.reduce_aligned_word_peers(
            parts, wholes, prior, torch.tensor([[True, False]]),
            rows[:, 0], torch.tensor([2]), torch.tensor([True]),
            staged_rows=rows, staged_atoms=atoms, part_n_what=2, whole_n_what=2,
            percept_evidence=evidence[..., :2], evidence_mask=mask[..., :2],
            order_zero_reading=reading, return_location_activations=True))
    # The basic percept's Pi over its two members on the for lane, then its
    # vector on the gaps and the location's own place, read by the column.
    presence = torch.tensor([.8 * .7, .6])
    place = torch.tensor([[3., 4.], [7., 8.]])
    combined = torch.cat((1 - (1 - code) ** presence[:, None], presence[:, None] * place), -1)
    expected = (combined @ column.detach()).tanh()[None]
    torch.testing.assert_close(locations, expected)
    torch.testing.assert_close(activation, expected[:, 0])
    torch.testing.assert_close(event[..., :2], expected.unsqueeze(-1) * atoms[:, :, :2])
    torch.testing.assert_close(event[..., 2:], torch.tensor([[[3., 4.], [7., 8.]]]))
    assert torch.equal(returned_prior[:, 0], prior[:, 0])
    assert torch.count_nonzero(returned_prior[:, 1]) == 0
    event[..., :1].sum().backward()
    assert torch.count_nonzero(evidence.grad[~mask]) == 0
    assert column.grad is not None and torch.count_nonzero(column.grad) > 0


def test_native_field_survives_symbolic_selection_and_unbind_returns_terminal_folds():
    class Carrier:
        pass
    cs = _space()
    cs.subspace = Carrier()
    cs.CSsub = Carrier()
    cs._subspaceForPS = Carrier()
    parts = (torch.rand(1, 1, 5), torch.rand(1, 1, 5))
    wholes = (torch.rand(1, 2, 7), torch.rand(1, 2, 7))
    field = cs.commit_percept_field(
        parts, wholes, reference_codes=torch.arange(8).reshape(1, 8),
        reference_roles=torch.tensor([[0, 1] * 4]),
        evidence=torch.ones(1, 2, 8), evidence_mask=torch.ones(1, 2, 8, dtype=torch.bool))
    cs.subspace.symbol = torch.tensor([1])  # selecting a description erases no field
    part_back, whole_back = cs.unbind()
    torch.testing.assert_close(part_back, parts[-1])
    torch.testing.assert_close(whole_back, wholes[-1])
    for depth in range(2):
        assert torch.equal(field[0][:, depth], parts[depth])
        assert torch.equal(field[1][:, depth], wholes[depth])


def test_dense_staged_readout_compiles_fullgraph_with_live_gradients():
    evidence = torch.randn(2, 3, 2, requires_grad=True) * .1
    reading = torch.rand(2, 1, 6 + 2, requires_grad=True)  # a column of 6, a vector of 2
    place = torch.rand(2, 3, 4)
    mask = torch.rand(2, 3, 2) > .3
    compiled = torch.compile(read_word, backend="eager", fullgraph=True)
    result = compiled(evidence, mask, reading, place, content_width=2)
    torch.testing.assert_close(result, read_word(evidence, mask, reading, place, content_width=2))
    result.sum().backward()
    assert reading.grad is not None and torch.isfinite(reading.grad).all()


def test_observed_property_absence_is_negative_and_padding_is_unobserved():
    cs = _space()
    part = torch.zeros(1, 1, 4)
    whole = torch.ones(1, 2, 4)
    refs = torch.ones(1, 8, 2)
    roles = torch.tensor([[1, 1, -1, -1, -1, -1, -1, -1]])
    presence = torch.full((1, 2, 8), -1.)
    presence[0, 0, :2] = torch.tensor([1., 0.])
    evidence, mask = cs.percept_code_evidence(
        part, whole, refs, refs, roles, torch.tensor([True]),
        part_n_what=2, whole_n_what=2, whole_presence=presence)
    torch.testing.assert_close(evidence[0, 0, :2], torch.tensor([1., -1.]))
    assert mask[0, 0, :2].all() and not mask[0, 1].any()
    assert torch.count_nonzero(evidence[0, 1]) == 0


def test_aligned_compatibility_binder_does_not_take_bands_from_padded_parts():
    from test_prior_stm_symbol_peer import _bare_cs, _Carrier
    cs = _bare_cs(n_locations=2)
    part = torch.tensor([[[.2] * 4 + [3.] * 4]])
    whole = torch.tensor([[[.4] * 4 + [5.] * 4, [.6] * 4 + [7.] * 4]])
    for use_folds in (False, True):
        out = _Carrier(torch.zeros(1, 2, 8))
        if use_folds:
            cs.bind_fold_streams([_Carrier(part)], [_Carrier(whole)], out)
        else:
            cs.bind_aligned_streams(_Carrier(part), _Carrier(whole), out)
        torch.testing.assert_close(out.materialize()[0, :, 4:],
                                   torch.tensor([[3.] * 4, [7.] * 4]))
        assert out._aligned_source_validity[0].tolist() == [[True, False], [True, True]]


def test_native_field_lifecycle_clears_only_reset_rows_then_releases_graph():
    class Carrier:
        pass
    cs = _space()
    cs.layers = nn.ModuleList()
    cs.subspace, cs.CSsub = Carrier(), Carrier()
    field = cs.commit_percept_field(
        [torch.ones(2, 1, 4, requires_grad=True)],
        [torch.ones(2, 2, 4, requires_grad=True)],
        reference_codes=torch.ones(2, 8, dtype=torch.long),
        reference_roles=torch.zeros(2, 8, dtype=torch.long),
        evidence=torch.ones(2, 2, 8), evidence_mask=torch.ones(2, 2, 8, dtype=torch.bool))
    updated = cs.commit_percept_field(
        [torch.full((2, 1, 4), .25)], [torch.full((2, 2, 4), .5)],
        reference_codes=field[2], reference_roles=field[3],
        evidence=field[4], evidence_mask=field[5], active_rows=torch.tensor([True, False]))
    torch.testing.assert_close(updated[0][1], field[0][1])
    torch.testing.assert_close(updated[1][1], field[1][1])
    assert (updated[0][0] == .25).all() and (updated[1][0] == .5).all()
    cs.Reset(batch=0, hard=False)
    for sub in (cs.CSsub, cs.subspace):
        for index, value in enumerate(sub._percept_field):
            assert (value[0] == (-1 if index in (2, 3) else 0)).all()
            torch.testing.assert_close(value[1], field[index][1])
    cs.End()
    assert cs.CSsub._percept_field is None and cs.subspace._percept_field is None
    object.__setattr__(cs.CSsub, "_percept_field", field)
    cs.Start()
    assert cs.CSsub._percept_field is None


def _checkpoint_space():
    from Memory import MediumTermMemory
    from types import SimpleNamespace
    cs = _space()
    cs.mtm = MediumTermMemory()
    from Spaces import Codebook
    from types import MethodType
    book = SimpleNamespace(W=cs.order_zero.weight)
    book.placement_extras = MethodType(Codebook.placement_extras, book)
    book.load_placement_extras = MethodType(Codebook.load_placement_extras, book)
    cs.similarity_codebook = book
    cs.legacy = nn.Linear(2, 2)
    cs.concept_source_readout = SigmaConceptsFromPercepts(9, 1)
    cs.layers = nn.ModuleList([cs.legacy, cs.concept_source_readout])
    return cs


def test_pre_readout_checkpoint_initializes_only_new_modules_with_strict_loading():
    source, target = _checkpoint_space(), _checkpoint_space()
    checkpoint = {key: value for key, value in source.state_dict().items()
                  if key.startswith(("legacy.", "layers.0.", "order_zero."))}
    initial_readout = target.concept_source_readout.sigma.layer.W.detach().clone()
    target.load_state_dict(checkpoint, strict=True)
    torch.testing.assert_close(target.legacy.weight, source.legacy.weight)
    torch.testing.assert_close(target.concept_source_readout.sigma.layer.W, initial_readout)
    partial = copy.deepcopy(source.state_dict())
    del partial["concept_source_readout.sigma.layer.W"]
    with pytest.raises(RuntimeError, match="Missing key"):
        target.load_state_dict(partial, strict=True)


def test_conceptual_checkpoint_metadata_and_tensor_state_roundtrip_together():
    source, target = _checkpoint_space(), _checkpoint_space()
    with torch.no_grad():
        source.order_zero.weight[0, 3] = -.8
    assert all(torch.is_tensor(value) for value in source.state_dict().values())
    target.load_state_dict(source.state_dict(), strict=True)
    target.load_vocab_extras(source.vocab_extras())
    torch.testing.assert_close(target.order_zero.weight, source.order_zero.weight)
    assert not getattr(target, "_legacy_whole_structure", None)
