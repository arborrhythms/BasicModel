"""Learn XOR through native, independently coded perceptual towers.

Teaching supplies byte memberships and the names of their two conceptual
reads. The towers compute OR/AND. XOR supervision supplies no literal sign.
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


def grounded_model(tmp_path, pool=4):
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
        'ConceptualSpace/nVectors': 8 * pool,
        'WholeSpace/nInput': slots, 'WholeSpace/nOutput': slots,
        'WholeSpace/nVectors': 16, 'WholeSpace/propertyBasis': True,
        'WholeSpace/analysis': 'meronomy', 'WholeSpace/digitWholes': False,
        'WholeSpace/divideWithinWhole': False, 'OutputSpace/nInput': slots,
    }
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
        model, _ = Models.BasicModel.from_config(str(path))
    model.eval()
    model.set_sigma(0.)
    x = torch.zeros(4, 1, slots, dtype=torch.long)
    x[:, 0, :2] = torch.tensor([[48, 48], [48, 49], [49, 48], [49, 49]])
    return model, x


def learn_grounded_xor(tmp_path, pool):
    model, x = grounded_model(tmp_path, pool)
    cs, ws = model.conceptualSpaces[0], model.wholeSpaces[0]
    # An unnamed property receives the two primitive observations. No OR,
    # AND or XOR table is supplied to a runtime activation.
    prior = ws.subspace.what.primitive_properties
    before, after = prior.teach(8, [48, 49], [0., 1.])
    assert before > .1 and after < 1e-12
    model.forward(x)
    parameters = [*cs.percept_read.parameters(), cs.similarity_codebook.W,
                  ws.subspace.what.W, model.perceptualSpace.subspace.what.W]
    optimizer = torch.optim.Adam(parameters, lr=.03)
    for _ in range(900):
        optimizer.zero_grad()
        model.forward(x)
        loss = 0.
        first = 0
        for tower, projection in enumerate(cs._cs_position_projections):
            spans = cs._cs_position_spans[:, first:first + projection.shape[1]]
            first += projection.shape[1]
            valid = spans[..., 1] > spans[..., 0]
            byte = x[:, 0].gather(1, spans[..., 0].clamp(0, x.shape[-1] - 1))
            target = torch.zeros_like(projection)
            # Both names mean the same observed primitive at their own
            # positions. The native extent folds create the difference.
            target[..., 1 if tower == 0 else 0] = 2 * (byte == 49).float() - 1
            loss = loss + ((projection - target).square() * valid[..., None]).sum() / valid.sum()
        loss.backward()
        optimizer.step()
        model.End()
    model.forward(x)
    towers = cs._cs_tower_evidence.detach()
    expected_or = torch.tensor([[0., 1.], [1., 0.], [1., 0.], [1., 0.]])
    expected_and = torch.tensor([[0., 1.], [0., 1.], [0., 1.], [1., 0.]])
    torch.testing.assert_close(towers[0, :, 0, 1], expected_or, atol=.08, rtol=0)
    torch.testing.assert_close(towers[1, :, 0, 0], expected_and, atol=.08, rtol=0)
    assert float(towers[0, :, 0, 0].max()) < .08
    assert float(towers[1, :, 0, 1].max()) < .08
    torch.testing.assert_close(cs._cs_extents, torch.tensor([[[0, 2]]]).expand(4, -1, -1))
    torch.testing.assert_close(cs._cs_position_spans[:, :2],
                               torch.tensor([[[0, 1], [1, 2]]]).expand(4, -1, -1))

    ly = Spaces._concept_alloc_of(cs).layer()
    assert int(ly.provisional.sum()) == pool
    # Four positively described conjunctions receive XOR supervision: the
    # observed OR, the observed AND, their co-presence, and the standing
    # presence. They begin with different positive definitions, never with
    # a supplied negative. All four must learn the XOR readout.
    parts = (cs._cs_last_a0[:, 1, 0, 0] > cs.concept_use_floor).nonzero().flatten().tolist()
    assert parts == [0]
    rows = [cs._assign_concept_parts(1, definition, torch.zeros(ly.nOutput), conjunctive=True)
            for definition in (parts, [1], [0, 1], [ly.nOutput])]
    assert len(set(rows)) == 4 and None not in rows
    assert not any(col > ly.nOutput for r, col in ly.conjunctive._index if r in rows)
    cs.getParameters()
    negative = [ly.conjunctive._index[row, 1 + ly.nOutput + 1] for row in rows]
    assert torch.count_nonzero(ly.conjunctive.values[negative]) == 0
    assert torch.count_nonzero(ly.participation[rows]) == 0
    assert not ly.participation.requires_grad
    optimizer = torch.optim.Adam([ly.conjunctive.values], lr=.03)
    target = torch.tensor([[0., 1.], [1., 0.], [1., 0.], [0., 1.]])
    for _ in range(600):
        optimizer.zero_grad()
        model.forward(x)
        leg = model.symbolSpace.forward_concept_to_symbol(model._combine_last_cs_sub)
        output = leg._symbol_evidence[rows]
        loss = (output - target[None]).square().mean()
        loss.backward()
        optimizer.step()
        ly.project_parts()
        cs.observe_concept_use(leg._concept_activations)
        model.End()
    with torch.no_grad():
        model.forward(x)
        leg = model.symbolSpace.forward_concept_to_symbol(model._combine_last_cs_sub)
        output = leg._symbol_evidence[rows]
        mse = float((output - target[None]).square().mean())
    print({'pool': pool, 'primitive_updates': 900, 'concept_updates': 600,
           'mse': mse, 'output': output.tolist(),
           'conjunctions': len(rows), 'gates': ly.participation[rows].tolist(),
           'positive_parts': [cs.concept_weights(row, conjunctive=True) for row in rows],
           'negative_parts': [cs.concept_weights(row, conjunctive=True, negated=True) for row in rows]})
    assert mse < .1
    assert torch.equal(output[..., 0] > .5, target[None, :, 0].bool().expand(len(rows), -1))
    assert bool((ly.conjunctive.values[negative] > 0).all())
    # Native readout keeps the independent pair and the positions. No raw
    # symbol-leg row is used as a concept dictionary index.
    torch.testing.assert_close(leg._concept_extents, cs._cs_extents)
    torch.testing.assert_close(leg._concept_position_spans, cs._cs_position_spans)
    assert leg.materialize().shape[1] == 2 * sum(cs._order_caps())
    model.End()
    return mse


@pytest.mark.parametrize('pool', [4, 8])
@pytest.mark.parametrize('trial', range(3))
def test_native_unseeded_xor(tmp_path, pool, trial):
    learn_grounded_xor(tmp_path, pool)
