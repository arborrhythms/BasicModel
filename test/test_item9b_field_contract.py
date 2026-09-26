"""Initial contracts from the September 25 item 9b plan, sections 1 and 4b–4d."""
from types import SimpleNamespace

import pytest
import torch

import Spaces
from PerceptProperties import PrimitiveProperties
from test_cs_sparse_weights import _cs
from test_structural_checkpoint import _model_with


def _one_part(part=7):
    bracket = torch.tensor([[[0, 1]]])
    return (torch.tensor([[part]]), bracket, None, torch.tensor([[65]]), bracket), bracket


@pytest.mark.parametrize("source_serial", [False, True])
@pytest.mark.parametrize("binding", ["mixing", "aligned"])
def test_inventory_checkpoint_preserves_definition_between_modes(tmp_path, source_serial, binding):
    source = _cs(nS=64, order=3)
    source._serial = source_serial
    source._concept_binding = binding
    cid = source.new_concept()
    row = source._csw_concept_row(0, cid)
    source.add_concept_feature(row, "ps", 7, .75)
    native, bracket = _one_part()
    before = source.cs_read_memberships(native, bracket)
    slot = (source._cs_field_concept_ids == cid).nonzero().flatten().item()
    before_pair = before[slot, 0, 0].detach().clone()
    model = _model_with(source, SimpleNamespace())
    model.conceptualSpaces = torch.nn.ModuleList([source])
    model.serial = source_serial
    model.concept_binding = binding
    model.nConceptCodes = source.nVectors
    checkpoint = tmp_path / "shared-inventory.pt"
    model.save_weights(checkpoint)

    restored = _cs(nS=64, order=3)
    restored._serial = not source_serial
    restored._concept_binding = binding
    target = _model_with(restored, SimpleNamespace())
    target.conceptualSpaces = torch.nn.ModuleList([restored])
    target.serial = not source_serial
    target.concept_binding = binding
    target.nConceptCodes = restored.nVectors
    assert target.load_weights(checkpoint, strict=True, require_match=True)
    assert target.serial is (not source_serial)
    assert restored._serial is (not source_serial)
    assert restored._csw_row_of(cid) == row
    old_store = Spaces._concept_alloc_of(source).layer(0)
    new_store = Spaces._concept_alloc_of(restored).layer(0)
    assert new_store.features._index == old_store.features._index
    torch.testing.assert_close(new_store.features.values, old_store.features.values)
    torch.testing.assert_close(restored.similarity_codebook.getW(), source.similarity_codebook.getW())
    after = restored.cs_read_memberships(native, bracket)
    slot = (restored._cs_field_concept_ids == cid).nonzero().flatten().item()
    torch.testing.assert_close(after[slot, 0, 0], before_pair)


def test_one_active_percept_reaches_every_referencing_concept():
    cs = _cs(nS=64, order=3)
    cs.outputShape[0] = 8
    identities = []
    for _ in range(12):
        cid = cs.new_concept()
        row = cs._csw_concept_row(0, cid)
        cs.add_concept_feature(row, "ps", 7, 1.)
        identities.append(cid)
    native, bracket = _one_part()
    field = cs.cs_read_memberships(native, bracket)
    # Attention selects percepts, not a top-eight subset of their definitions.
    # One percept is within the focused 8/8 budget as well as open attention.
    for cid in identities:
        slots = (cs._cs_field_concept_ids == cid).nonzero().flatten()
        assert slots.numel() == 1, f"referenced concept {cid} was truncated from the field"
        torch.testing.assert_close(field[slots[0], 0, 0], torch.tensor([1., 0.]))


@pytest.mark.parametrize("bracket, expected", [
    ((0, 2), (1., 1.)),
    ((0, 1), (1., 0.)),
    ((1, 2), (0., 1.)),
    ((0, 0), (0., 0.)),
])
def test_observed_complement_is_pooled_inside_one_bracket(bracket, expected):
    cs = _cs()
    primitive = PrimitiveProperties(1)
    primitive.teach(0, [48, 49], [0., 1.])
    cs.add_concept_feature(0, "ws", 0, 1.)
    raw = torch.tensor([[49, 48]])
    positions = torch.tensor([[[0, 1], [1, 2]]])
    scope = torch.tensor([[bracket]])
    field = cs.cs_read_memberships((raw, positions, primitive, raw, positions), scope)
    # A property witnessed true and false within one bracket reports both.
    # Narrowing attention isolates either observation; an empty scope is unknown.
    torch.testing.assert_close(field[0, 0, 0], torch.tensor(expected))


def test_padding_does_not_create_observed_complements():
    cs = _cs()
    primitive = PrimitiveProperties(1)
    primitive.teach(0, [48, 49], [0., 1.])
    cs.add_concept_feature(0, "ws", 0, 1.)
    raw = torch.tensor([[0, 0]])
    positions = torch.tensor([[[0, 1], [1, 2]]])
    scope = torch.tensor([[[0, 2]]])
    field = cs.cs_read_memberships((raw, positions, primitive, raw, positions), scope)
    assert field.count_nonzero() == 0


@pytest.mark.parametrize('source_serial', [False, True])
def test_native_learning_updates_one_definition_for_both_modes(source_serial):
    cs = _cs(nS=64, order=3)
    cs._serial = source_serial
    cs._concept_binding = 'aligned'
    cid = cs.new_concept()
    row = cs._csw_concept_row(0, cid)
    cs.add_concept_feature(row, 'ws', 0, 1.)
    primitive = PrimitiveProperties(1)
    primitive.teach(0, [65], [.5])
    spans = torch.tensor([[[0, 1]]])
    native = (torch.tensor([[7]]), spans, primitive, torch.tensor([[65]]), spans)
    store = Spaces._concept_alloc_of(cs).layer()
    parameter = store.features.values
    initial = parameter.detach().clone()
    optimizer = torch.optim.SGD([parameter], lr=.1)
    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        evidence = cs.cs_read_memberships(native, spans)
        loss = (evidence[0, 0, 0, 0] - .8).square()
        losses.append(float(loss.detach()))
        loss.backward(); optimizer.step(); store.project_parts()
    assert losses[-1] < losses[0]
    assert not torch.equal(parameter, initial)
    before = cs.cs_read_memberships(native, spans).detach()
    cs._serial = not source_serial
    after = cs.cs_read_memberships(native, spans)
    assert cs._csw_row_of(cid) == row and store.features.values is parameter
    torch.testing.assert_close(after, before)


def test_vector_head_selects_symbols_without_truncating_knowing(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path, inventory=128)
    cs = model._concept_owner()
    pid, = model.perceptualSpace.percept_store.spell_out(b'a')
    for _ in range(20):
        cid = cs.new_concept()
        row = cs._csw_concept_row(0, cid)
        cs.add_concept_feature(row, 'ps', pid, 1.)
    with torch.no_grad():
        state = model.understand(model.inputSpace.prepInput(['a']))
    field = state.reconstruction_carriers['field']
    assert field.evidence.shape[0] >= 20
    assert field.evidence[..., 0].count_nonzero() >= 20
    assert model.outputSpace.subspace.materialize().shape[1:] == tuple(model.outputSpace.outputShape)
    model.End()
