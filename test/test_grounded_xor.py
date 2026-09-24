"""Learn XOR through native, independently coded perceptual towers.

Teaching supplies byte memberships and one conceptual property. The extent
read retains both symbols; XOR learns their conjunction from primitive input.
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


def grounded_model(tmp_path, pool=4, inventory=None, load_data=False):
    tree = ET.parse(ROOT / 'data/MM_xor_fixture.xml')
    root = tree.getroot()
    slots = 2 * pool
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
    model, x = grounded_model(tmp_path, pool)
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
    assert int(store.provisional.sum()) == pool
    rows = store.provisional.nonzero().flatten().tolist()[:4]
    # Four provisional hypotheses begin with the same witnessed positive
    # property. Only its opposite symbol is offered at zero; supervision
    # must learn that both are necessary. Participation begins at zero.
    for row in rows:
        cs.add_concept_edge(row, 0, 1., conjunctive=True)
        store.assigned[row] = True
    cs._prepare_part_learning()
    negative = [store.conjunctive._index[row, store.nOutput + 1] for row in rows]
    assert torch.count_nonzero(store.conjunctive.values[negative]) == 0
    assert torch.count_nonzero(store.participation[rows]) == 0
    assert not store.participation.requires_grad
    optimizer = torch.optim.Adam([store.conjunctive.values], lr=.03)
    target = torch.tensor([0., 1., 1., 0.])
    for _ in range(120):
        optimizer.zero_grad()
        model.forward(x)
        leg = model.symbolSpace.forward_concept_to_symbol(model._combine_last_cs_sub)
        output = leg._symbol_evidence[rows]
        loss = (output[..., 0] - target[None]).square().mean()
        loss.backward()
        optimizer.step()
        store.project_parts()
        cs.observe_concept_use(leg._concept_activations)
        model.End()
    with torch.no_grad():
        model.forward(x)
        leg = model.symbolSpace.forward_concept_to_symbol(model._combine_last_cs_sub)
        output = leg._symbol_evidence[rows]
        mse = float((output[..., 0] - target[None]).square().mean())
    print({'pool': pool, 'primitive_updates': 32, 'definition_updates': 32,
           'concept_updates': 120, 'mse': mse, 'output': output.tolist(),
           'conjunctions': len(rows), 'gates': store.participation[rows].tolist(),
           'positive_parts': [cs.concept_weights(row, conjunctive=True) for row in rows],
           'negative_parts': [cs.concept_weights(row, conjunctive=True, negated=True) for row in rows]})
    assert mse < 1e-6
    torch.testing.assert_close(output[..., 0], target[None].expand(len(rows), -1), atol=1e-6, rtol=0)
    assert bool((store.conjunctive.values[negative] > 0).all())
    assert all({source for source, weight in cs.concept_weights(row, conjunctive=True)
                if weight > 0} == {0} for row in rows)
    assert all({source for source, weight in cs.concept_weights(row, conjunctive=True, negated=True)
                if weight > 0} == {0} for row in rows)
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
