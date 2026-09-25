"""Learn XOR through native, independently coded perceptual towers.

Teaching supplies byte memberships and one conceptual property. Located cases
are witnessed without labels; a learned sigma row's positive pole must be XOR.
No random seed is set, and every declared run is an assertion.
"""
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import pytest
import torch

import Language
import Models
import Spaces
from util import init_config
from test_wholespace_property_migration import _set_text

ROOT = Path(__file__).resolve().parents[1]


def grounded_model(tmp_path, pool=4, inventory=None, load_data=False, field_slots=None):
    tree = ET.parse(ROOT / 'data/MM_xor_fixture.xml')
    root = tree.getroot()
    slots = 2 * pool if field_slots is None else field_slots
    values = {
        'architecture/serial': False, 'architecture/symbolicOrder': 1,
        'architecture/subsymbolicOrder': 2, 'architecture/conceptualPi': True,
        'architecture/symbolTower': True, 'architecture/conceptBinding': 'aligned',
        'architecture/attentionPromotion': True, 'architecture/conceptPoolSize': pool,
        'architecture/training/autoload': False, 'architecture/training/maskRate': 0.,
        'InputSpace/nVectors': slots, 'InputSpace/nOutput': slots,
        'PartSpace/nInput': slots, 'PartSpace/nOutput': slots,
        'PartSpace/nVectors': slots, 'PartSpace/synthesis': 'meronomy',
        'PartSpace/chunkPromotionThreshold': 100000,
        'ConceptualSpace/nInput': slots, 'ConceptualSpace/nOutput': slots,
        'ConceptualSpace/nVectors': 8 * pool if inventory is None else inventory,
        'WholeSpace/nInput': slots, 'WholeSpace/nOutput': slots,
        'WholeSpace/nVectors': 16, 'WholeSpace/propertyBasis': True,
        'WholeSpace/analysis': 'meronomy', 'WholeSpace/digitWholes': False,
        'WholeSpace/divideWithinWhole': False, 'OutputSpace/nInput': slots,
    }
    if inventory is not None:
        values['architecture/mereologyRaise'] = True
    if load_data:
        values['architecture/data/dataset'] = 'xor'
    for key, value in values.items():
        _set_text(root, key, value)
    grammar = root.find('SymbolSpace/language/grammar')
    grammar.clear()
    for name in ('pi', 'sigma'):
        ET.SubElement(grammar, 'S').text = f'S = {name}(S)'
    path = tmp_path / 'grounded.xml'
    tree.write(path)
    init_config(path=str(path), defaults_path=str(ROOT / 'data/model.xml'))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if load_data:
            from recon_bench import _build_model
            model, *_ = _build_model(str(path))
        else:
            model, _ = Models.BasicModel.from_config(str(path))
    model.eval()
    model.set_sigma(0.)
    x = torch.zeros(4, 1, slots, dtype=torch.long)
    x[:, 0, :2] = torch.tensor([[48, 48], [48, 49], [49, 48], [49, 49]])
    return model, x


def learn_grounded_xor(tmp_path, pool):
    model, x = grounded_model(tmp_path, pool, field_slots=4 * pool)
    cs, ws = model.conceptualSpaces[0], model.wholeSpaces[0]
    prior = ws.subspace.what.primitive_properties
    # Teach primitive memberships, then the name's feature definition.
    # No conceptual truth table is installed as an activation.
    for row, targets in ((8, [0., 1.]), (9, [1., 0.])):
        before, after = prior.teach(row, [48, 49], targets)
        assert before > .1 and after < 1e-12
    assert cs._csw_concept_row(0, 10001) == 0
    cs.add_concept_feature(0, 'ws', 8, 1.)
    for negative in (False, True):
        cs.add_concept_feature(0, 'ws', 9, 0., negated=negative)
    native = torch.zeros(3, 1, x.shape[-1], dtype=x.dtype, device=x.device)
    native[:, 0, 0] = torch.tensor([48, 49, 65])
    store = Spaces._concept_alloc_of(cs).layer()
    optimizer = torch.optim.Adam([store.features.values], lr=.03)
    # Name single primitive occurrences. The two-position truth table must
    # subsequently arise from extent readout, never from these targets.
    named = torch.tensor([[0., 1.], [1., 0.], [0., 0.]])
    for _ in range(32):
        optimizer.zero_grad()
        model.forward(native)
        loss = (model._combine_last_cs_sub._concept_activations[0, :, 0] - named).square().mean()
        loss.backward()
        optimizer.step()
        store.project_parts()
        model.End()
    model.forward(x)
    expected = torch.tensor([[0., 1.], [1., 1.], [1., 1.], [1., 0.]])
    torch.testing.assert_close(cs._cs_last_a0[0, :, 0], expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(cs._cs_extents, torch.tensor([[[0, 2]]]).expand(4, -1, -1))
    # Composition check only, on the two located primitive readings. It is
    # independent of the output row's learned case selection below.
    positions = cs._cs_position_evidence[0, :, 0]
    spans = cs._cs_position_spans
    located = []
    for start in (0, 1):
        mask = (spans[..., 0] == start) & (spans[..., 1] == start + 1)
        located.append((positions * mask[..., None]).amax(1))
    a, b = located
    or_positive = torch.maximum(a[:, 0], b[:, 0])
    not_and = torch.maximum(a[:, 1], b[:, 1])
    torch.testing.assert_close(torch.minimum(or_positive, not_and), torch.tensor([0., 1., 1., 0.]))
    pool_rows = [r for r in store.provisional.nonzero().flatten().tolist()
                 if cs._order0_inventory_row(r)]
    assert len(pool_rows) == pool
    assert torch.count_nonzero(store.participation[pool_rows]) == 0
    assert not store.participation.requires_grad
    # The ordinary observer receives all four unlabeled primitive fields.
    # It records complete located witnesses, including both repeated-pole
    # patterns. No XOR-specific part or output edge is supplied.
    cs.promotion_observe()
    rows = [r for r in pool_rows if bool(store.assigned[r])]
    assert len(rows) == 4
    assert all(sum(len(v) for (r, _), v in store.conjunctive.locations.items() if r == row) >= 2
               for row in rows)
    for _ in range(8):
        model.End()
        model.forward(x)
        cs.observe_concept_use(model._combine_last_cs_sub._concept_activations)
    cs.promotion_pass()
    assert all(not bool(store.provisional[r]) and store.participation[r] == 1 for r in rows)
    xor = cs._csw_concept_row(1, 20001)
    assert xor is not None
    for row in rows:
        cs.add_concept_edge(xor, row)  # all exponents start unwritten
    initial = store.values.detach().clone()
    optimizer = torch.optim.Adam([store.values], lr=.03)
    target = torch.tensor([0., 1., 1., 0.])
    initial_mse = None
    for step in range(32):
        optimizer.zero_grad()
        model.End()
        model.forward(x)
        leg = model.symbolSpace.forward_concept_to_symbol(model._combine_last_cs_sub)
        bound = leg._concept_inventory_rows
        slot = int((bound == xor).nonzero()[0, 0])
        output = leg._symbol_evidence[slot, :, 0]
        loss = (output - target).square().mean()
        if step == 0:
            initial_mse = float(loss.detach())
            assert initial_mse >= .25, 'the unwritten XOR row must fail before learning'
        loss.backward()
        optimizer.step()
        store.project_parts()
    with torch.no_grad():
        model.End()
        model.forward(x)
        leg = model.symbolSpace.forward_concept_to_symbol(model._combine_last_cs_sub)
        bound = leg._concept_inventory_rows
        slot = int((bound == xor).nonzero()[0, 0])
        output = leg._symbol_evidence[slot, :, 0]
        mse = float((output - target).square().mean())
    print({'pool': pool, 'primitive_updates': 32, 'definition_updates': 32,
           'concept_updates': 32, 'initial_mse': initial_mse, 'mse': mse, 'output': output.tolist(),
           'conjunctions': len(rows), 'gates': store.participation[rows].tolist(),
           'xor_parts': cs.concept_weights(xor),
           'located_parts': repr(store.conjunctive.locations)})
    assert mse < 1e-6
    torch.testing.assert_close(output, target, atol=1e-6, rtol=0)
    assert not torch.equal(initial, store.values)
    learned = [(r, weight) for r, weight in cs.concept_weights(xor) if weight > 0]
    assert len(learned) == 2
    assert store.features.values[store.features._index[0, 4 * 9 + 3]] > 0
    # Pi was performed on retained order-0 positions, before symbolization.
    # The higher-order XOR row contains only sigma edges to those cases.
    assert all(cs._order0_inventory_row(r) and cs._order0_inventory_row(c % (store.nOutput + 1))
               for r, c in store.conjunctive._index)
    assert all(r != xor for r, _ in store.conjunctive._index)
    assert set(r for r, _ in learned).issubset(rows)
    torch.testing.assert_close(leg._concept_extents, cs._cs_extents)
    torch.testing.assert_close(leg._concept_position_spans, cs._cs_position_spans)
    assert leg.materialize().shape[1] == 2 * sum(cs._order_caps())
    model.End()
    # Learned exclusion is exact at every read scope, without a noise floor.
    controls = x[:1].expand(4, -1, -1).clone()
    controls[:, 0, :2] = torch.tensor([[65, 65], [90, 90], [50, 50], [33, 33]])
    with torch.no_grad():
        model.forward(controls)
        leg = model.symbolSpace.forward_concept_to_symbol(model._combine_last_cs_sub)
        assert cs._cs_position_evidence.count_nonzero() == 0
        assert leg._concept_activations.count_nonzero() == 0
        assert leg._symbol_evidence.count_nonzero() == 0
    model.End()
    return mse


@pytest.mark.parametrize('pool', [4, 8])
@pytest.mark.parametrize('trial', range(3))
def test_native_unseeded_xor(tmp_path, pool, trial):
    learn_grounded_xor(tmp_path, pool)
