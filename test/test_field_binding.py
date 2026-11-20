from basic_fixtures import feature as basic_feature, order_zero_expected, published_presence
from order_fixtures import force_order
from types import SimpleNamespace

import torch

import Spaces

from Models import BasicModel

from test_cs_sparse_weights import _cs, _mint_row

from test_concept_memberships import binary_features


def test_bound_field_can_include_unwritten_inventory_addresses():
    cs = _cs(nS=128, order=2)
    store = Spaces._concept_alloc_of(cs).layer()
    # A definition-free store may precede the expanded inventory.
    store.grow_inventory(128)
    view = cs._concept_field_store(torch.tensor([0, 112, 127, -1]))
    assert view.nnz == 0 and view.nOutput == 4



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
    force_order(source, cid, 0)
    row = source._csw_concept_row(0, cid)
    basic_feature(source, row, "ps", 7, .75)
    native, bracket = _one_part()
    before = source.cs_read_memberships(native, bracket)
    slot = (source._cs_field_concept_ids == cid).nonzero().flatten().item()
    before_pair = before[slot, 0, 0].detach().clone()
    model = _model_with(source, SimpleNamespace())
    model.conceptualSpaces = torch.nn.ModuleList([source])
    model.word_brackets = source_serial
    model.concept_binding = binding
    model.nConceptCodes = source.nVectors
    checkpoint = tmp_path / "shared-inventory.pt"
    model.save_weights(checkpoint)

    restored = _cs(nS=64, order=3)
    restored._serial = not source_serial
    restored._concept_binding = binding
    # The model installs the order-zero Sigma where it binds the basic
    # percepts; the fixture binds them by hand.
    restored.install_order_zero()
    target = _model_with(restored, SimpleNamespace())
    target.conceptualSpaces = torch.nn.ModuleList([restored])
    target.word_brackets = not source_serial
    target.concept_binding = binding
    target.nConceptCodes = restored.nVectors
    assert target.load_weights(checkpoint, strict=True, require_match=True)
    assert target.word_brackets is (not source_serial)
    assert restored._serial is (not source_serial)
    assert restored._csw_row_of(cid) == row
    assert (dict(Spaces._concept_alloc_of(restored).order_zero_members)
            == dict(Spaces._concept_alloc_of(source).order_zero_members))
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
        force_order(cs, cid, 0)
        row = cs._csw_concept_row(0, cid)
        basic_feature(cs, row, "ps", 7, 1.)
        identities.append(cid)
    native, bracket = _one_part()
    field = cs.cs_read_memberships(native, bracket)
    presence = published_presence(cs)
    # Attention selects percepts, not a top-eight subset of their definitions.
    # One percept is within the focused 8/8 budget as well as open attention.
    for cid in identities:
        slots = (cs._cs_field_concept_ids == cid).nonzero().flatten()
        assert slots.numel() == 1, f"referenced concept {cid} was truncated from the field"
        y, lanes = order_zero_expected(cs, cs._csw_row_of(cid), presence)
        # Its own row present at one reads tanh(1).
        assert y > .76
        torch.testing.assert_close(field[slots[0], 0, 0], lanes.to(field), atol=1e-5, rtol=1e-5)


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
    basic_feature(cs, 0, "ws", 0, 1., parts=(48, 49))
    raw = torch.tensor([[49, 48]])
    positions = torch.tensor([[[0, 1], [1, 2]]])
    scope = torch.tensor([[bracket]])
    field = cs.cs_read_memberships((raw, positions, primitive, raw, positions), scope)
    # A property witnessed true and false within one bracket reports both,
    # across the rows of the basic percepts that carry it: each row is its
    # part's presence in the bracket times the property's lanes where the
    # part lies (rule 6; it holds at 49 and not at 48). Narrowing attention
    # isolates either observation; an empty scope is unknown.
    percepts, basic = cs._percept_field, cs.basic_percepts
    assert len(percepts.keys) == 2
    inside = raw[0, bracket[0]:bracket[1]]
    at = {49: torch.tensor([1., 0.]), 48: torch.tensor([0., 1.])}
    for part in (48, 49):
        index = percepts.keys.index(('bp', basic.index[((part,), (0,))]))
        presence = float((inside == part).any())
        torch.testing.assert_close(percepts.values[index, 0, 0].detach(), presence * at[part])
    torch.testing.assert_close(cs.order_zero_both(percepts, [0])[0, 0, 0], torch.tensor(min(expected)))
    # The concept is one signed number, for less against.
    _, lanes = order_zero_expected(cs, 0, published_presence(cs))
    torch.testing.assert_close(field[0, 0, 0], lanes.to(field), atol=1e-5, rtol=1e-5)


def test_padding_does_not_create_observed_complements():
    cs = _cs()
    primitive = PrimitiveProperties(1)
    primitive.teach(0, [48, 49], [0., 1.])
    basic_feature(cs, 0, "ws", 0, 1., parts=(48, 49))
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
    force_order(cs, cid, 0)
    row = cs._csw_concept_row(0, cid)
    basic_feature(cs, row, 'ws', 0, 1., parts=(7,))
    primitive = PrimitiveProperties(1)
    # A membership of .5 makes the row read [.5, .5] (rule 6): its combined
    # for and against vectors are equal and the column reads nothing to
    # learn from. The property is held at .8, so the row reads [.8, .2].
    primitive.teach(0, [65], [.8])
    spans = torch.tensor([[[0, 1]]])
    native = (torch.tensor([[7]]), spans, primitive, torch.tensor([[65]]), spans)
    store = Spaces._concept_alloc_of(cs).layer()
    # The concept reads its column of the order-zero Sigma.
    parameter = cs.order_zero.value
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
    assert cs._csw_row_of(cid) == row and cs.order_zero.value is parameter
    torch.testing.assert_close(after, before)


def test_vector_head_selects_symbols_without_truncating_knowing(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path, inventory=128)
    cs = model._concept_owner()
    pid, = model.perceptualSpace.percept_store.spell_out(b'a')
    for _ in range(20):
        cid = cs.new_concept()
        force_order(cs, cid, 0)
        row = cs._csw_concept_row(0, cid)
        basic_feature(cs, row, 'ps', pid, 1.)
    with torch.no_grad():
        state = model.understand(model.inputSpace.prepInput(['a']))
    field = state.reconstruction_carriers['field']
    assert field.evidence.shape[0] >= 20
    assert field.evidence[..., 0].count_nonzero() >= 20
    assert model.outputSpace.subspace.materialize().shape[1:] == tuple(model.outputSpace.outputShape)
    model.End()
