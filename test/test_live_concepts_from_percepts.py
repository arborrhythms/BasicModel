"""Live Sigma handoff: addressed evidence, sparse learning, and exact carriers."""
import copy

import pytest
import torch
from torch import nn

from Layers import IndexedSigmaConceptsFromPercepts, SigmaConceptsFromPercepts
from Optimizer import RowLocalAdam
from Spaces import ConceptualSpace


def _space():
    cs = ConceptualSpace.__new__(ConceptualSpace)
    nn.Module.__init__(cs)
    cs.nWhat, cs.nWhere, cs.nWhen = 2, 1, 1
    cs.concept_dim = 4
    cs.inputShape = cs.outputShape = [2, 4]
    cs.concepts_from_percepts = IndexedSigmaConceptsFromPercepts(8, 16)
    return cs


def test_indexed_readout_matches_plain_sigma_with_independent_concept_rows():
    layer = IndexedSigmaConceptsFromPercepts(3, 16).double()
    plain = SigmaConceptsFromPercepts(3, 2).double()
    with torch.no_grad():
        plain.input_weights.copy_(torch.tensor([[.4, -.2], [.1, .3], [-.7, .9]]))
        plain.concept_bias.copy_(torch.tensor([[.1, -.4]]))
        layer.coefficients[3, :-1].copy_(plain.input_weights[:, 0])
        layer.coefficients[7, :-1].copy_(plain.input_weights[:, 1])
        layer.coefficients[3, -1] = .1
        layer.coefficients[7, -1] = -.4
    evidence = torch.tensor([[.2, .6, -.1]], dtype=torch.float64)
    actual = layer(evidence.expand(2, -1), torch.tensor([3, 7]))
    torch.testing.assert_close(actual, plain(evidence)[0])


def test_eight_total_references_not_four_per_role_and_no_slot_reassignment():
    layer = IndexedSigmaConceptsFromPercepts(8, 16)
    slots = layer.admit_references(3, range(7), [12])
    assert slots[:2] == ((0, 0), (1, 12))
    assert len(slots) == 8 and sum(role == 0 for role, _ in slots) == 7
    assert layer.admit_references(3, [99], [13]) == slots
    pair = layer.admit_references(5, [91], [12])
    assert pair == ((0, 91), (1, 12))  # two references need no six fillers


def test_sparse_update_changes_only_selected_concepts_and_unknown_is_zero():
    layer = IndexedSigmaConceptsFromPercepts(8, 1_000_000)
    before = layer.coefficients[[0, 2, 17, 100]].detach().clone()
    optimizer = RowLocalAdam(layer.parameters(), lr=.01)
    rows = torch.tensor([2, 17, -1])
    output = layer(torch.ones(3, 8) * .1, rows)
    assert output[-1] == 0
    output.sum().backward()
    grad = layer.coefficients.grad
    assert grad.is_sparse and grad._nnz() <= 3
    assert torch.isfinite(grad.coalesce().values()).all()
    optimizer.step()
    after = layer.coefficients[[0, 2, 17, 100]].detach()
    assert torch.equal(before[[0, 3]], after[[0, 3]])
    assert not torch.equal(before[1:3], after[1:3])
    assert optimizer.state[layer.coefficients]["exp_avg"].numel() < 1000


def test_indexed_checkpoint_roundtrip_growth_and_invalid_reference_rejection():
    source = IndexedSigmaConceptsFromPercepts(8, 16)
    source.admit_references(3, [11, 12], [4])
    with torch.no_grad():
        source.coefficients[3, 1] = -.25
    saved = copy.deepcopy(source.state_dict())
    restored = IndexedSigmaConceptsFromPercepts(8, 32)
    restored.load_state_dict(saved, strict=True)
    restored.load_reference_state(source.reference_state())
    assert restored.references == source.references
    torch.testing.assert_close(restored.coefficients[:16], source.coefficients)
    assert (restored.coefficients[16:, :-1] == .5).all()
    with pytest.raises(ValueError, match="invalid"):
        restored.load_reference_state({"version": 1, "references": {3: [(2, 11)]}})


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
    event, orders, row, activation, returned_prior, validity, locations = (
        cs.reduce_aligned_word_peers(
            parts, wholes, prior, torch.tensor([[True, False]]),
            rows[:, 0], torch.tensor([2]), torch.tensor([True]),
            staged_rows=rows, staged_atoms=atoms, part_n_what=2, whole_n_what=2,
            percept_evidence=evidence, evidence_mask=mask,
            return_location_activations=True))
    expected = torch.tensor([[.75, .3]]).tanh()
    torch.testing.assert_close(locations, expected)
    torch.testing.assert_close(activation, expected[:, 0])
    torch.testing.assert_close(event[..., :2], expected.unsqueeze(-1) * atoms[:, :, :2])
    torch.testing.assert_close(event[..., 2:], torch.tensor([[[3., 4.], [7., 8.]]]))
    assert torch.equal(returned_prior[:, 0], prior[:, 0])
    assert torch.count_nonzero(returned_prior[:, 1]) == 0
    event[..., :1].sum().backward()
    assert torch.count_nonzero(evidence.grad[~mask]) == 0
    assert cs.concepts_from_percepts.coefficients.grad.is_sparse


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
    evidence = torch.randn(2, 3, 8, requires_grad=True) * .1
    coefficients = torch.rand(2, 1, 9, requires_grad=True)
    mask = torch.rand(2, 3, 8) > .3
    compiled = torch.compile(SigmaConceptsFromPercepts.from_coefficients,
                             backend="eager", fullgraph=True)
    result = compiled(evidence, coefficients, mask=mask)
    torch.testing.assert_close(result, SigmaConceptsFromPercepts.from_coefficients(
        evidence, coefficients, mask=mask))
    result.sum().backward()
    assert coefficients.grad is not None and torch.isfinite(coefficients.grad).all()


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
    cs.concepts_from_percepts.admit_references(3, [1], [2])
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
    assert cs.concepts_from_percepts.references[3] == ((0, 1), (1, 2))


def _checkpoint_space():
    cs = _space()
    cs.legacy = nn.Linear(2, 2)
    cs.concept_source_readout = SigmaConceptsFromPercepts(9, 1)
    cs.layers = nn.ModuleList([cs.legacy, cs.concepts_from_percepts,
                              cs.concept_source_readout])
    return cs


def test_pre_readout_checkpoint_initializes_only_new_modules_with_strict_loading():
    source, target = _checkpoint_space(), _checkpoint_space()
    checkpoint = {key: value for key, value in source.state_dict().items()
                  if key.startswith(("legacy.", "layers.0."))}
    initial_readout = target.concepts_from_percepts.coefficients.detach().clone()
    target.load_state_dict(checkpoint, strict=True)
    torch.testing.assert_close(target.legacy.weight, source.legacy.weight)
    torch.testing.assert_close(target.concepts_from_percepts.coefficients, initial_readout)
    partial = copy.deepcopy(source.state_dict())
    del partial["concepts_from_percepts.coefficients"]
    with pytest.raises(RuntimeError, match="Missing key"):
        target.load_state_dict(partial, strict=True)


def test_conceptual_checkpoint_metadata_and_tensor_state_roundtrip_together():
    source, target = _checkpoint_space(), _checkpoint_space()
    refs = source.concepts_from_percepts.admit_references(3, [11], [2])
    with torch.no_grad():
        source.concepts_from_percepts.coefficients[3, 0] = -.8
    assert all(torch.is_tensor(value) for value in source.state_dict().values())
    target.load_state_dict(source.state_dict(), strict=True)
    target.load_vocab_extras(source.vocab_extras())
    assert target.concepts_from_percepts.references[3] == refs
    torch.testing.assert_close(target.concepts_from_percepts.coefficients,
                               source.concepts_from_percepts.coefficients)
    assert not getattr(target, "_legacy_whole_structure", None)


def test_autoload_capacity_preflight_expands_all_readout_aliases_in_place():
    from Models import BaseModel
    model = BaseModel()
    model.name, model.concept_binding, model.serial = "ReadoutCheckpoint", "aligned", True
    model.nConceptCodes = model.nSymbols = 16
    cs = _checkpoint_space()
    model.conceptualSpaces = nn.ModuleList([cs])
    live = dict(model.state_dict())
    saved = {key: value.detach().clone() for key, value in live.items()}
    keys = [key for key in saved if key.endswith(".coefficients")]
    assert len(keys) == 2  # named owner plus layers compatibility alias
    for key in keys:
        saved[key] = torch.full((8, 9), .123)
    assert model._expand_aligned_codebook_checkpoint_state(saved, live) == len(keys)
    assert len({saved[key].data_ptr() for key in keys}) == 1
    for key in keys:
        assert saved[key].shape == live[key].shape
        assert (saved[key][:8] == .123).all()
        assert (saved[key][8:, :-1] == .5).all()
    model.load_state_dict(saved, strict=True)
