"""Surviving form centroid contracts from the retired cover tests."""
from types import SimpleNamespace
import pytest
import torch
from Meaning import ConceptualMeaning
from test_reference_close import fixture as reference_fixture

def fixture(*, percept=False):
    from MemoryIndex import configure_model_index
    resolver, store, atoms = reference_fixture(percept=percept)
    configure_model_index(SimpleNamespace(conceptualSpace=resolver.components.owner), store)
    return resolver, store, atoms


def test_higher_order_fold_is_evidence_weighted_and_preserves_rows():
    from MemoryConsolidation import _fold
    resolver, store, atoms = fixture()
    starts = [store.append_meaning(ConceptualMeaning.from_description(atoms[n]),
        evidence=pair)
        for n, pair in enumerate(((1., 0.), (.25, .25), (0., 0.)))]
    rows = list(range(len(store)))
    before = store.slots.clone()
    folded = _fold(resolver.components.owner, store, rows)
    torch.testing.assert_close(folded, (atoms[0] + .5*atoms[1]) / 1.5)
    torch.testing.assert_close(store.slots, before, rtol=0, atol=0)


def test_centroid_centers_parts_and_native_wholes_by_their_separate_counts():
    from MeaningKernels import centroid, split_centroid
    from FormCentroids import SpanWholes
    parts = torch.tensor([[.2, .6], [.8, .4]])
    codes = torch.tensor([[.9, .1], [.7, .3], [.0, 1.]])
    coefficients = torch.zeros(3, 256)
    coefficients[:2, [10, 20]] = 1.
    coefficients[2, 10] = 1.
    counts = torch.zeros(1, 2, 256)
    counts[0, 0, 10] = counts[0, 1, 20] = 1.
    bank = SpanWholes(torch.cat((counts[:, :1]*0, counts.cumsum(1)), 1), coefficients, codes)
    whole = bank.read(torch.tensor([[[0, 2]]]))
    torch.testing.assert_close(whole[0][0, 0], codes[:2].mean(0))
    assert whole[1].item() == 2
    left, right = parts[0].reshape(1, 1, -1), parts[1].reshape(1, 1, -1)
    value = centroid((left, right), (3, 1), whole=whole)
    expected = ((parts[0]*3+parts[1])/4 + codes[:2].mean(0))/2
    torch.testing.assert_close(value[0, 0], expected)
    torch.testing.assert_close(split_centroid(value, right, (3, 1), whole=whole), left)
    absent = bank.read(torch.tensor([[[-1, -1]]]))
    torch.testing.assert_close(centroid((left, right), (3, 1), whole=absent),
                               centroid((left, right), (3, 1)))


def test_whole_contribution_follows_the_composed_stamp():
    from types import SimpleNamespace
    from FormCentroids import selected
    mean, count = torch.ones(1, 2, 4), torch.ones(1, 2, 1)
    orders = (torch.tensor([[0, 1]]), torch.tensor([[0, 1]]))
    rule = SimpleNamespace(head_role=2)
    actual = selected({2: (mean, count, orders)}, 2, mean, rule)
    assert actual[1].flatten().tolist() == [1., 0.]
    rule.determiner_mode = 'kind'
    assert selected({2: (mean, count, orders)}, 2, mean, rule)[1].flatten().tolist() == [1., 0.]


@pytest.mark.parametrize('membership', (.3, 1.))
@pytest.mark.parametrize('centered', (False, True))
def test_free_inverse_reads_native_wholes_from_candidates_without_a_span(monkeypatch, centered, membership):
    from test_meaning_kernels import paired
    # Use the common paired-layout setup directly; no parse or target bytes
    # are supplied to the inverse search.
    paired.__wrapped__(monkeypatch)
    from Language import LanguageSpace, ProductLayer
    from FormCentroids import CandidateWholes
    op = ProductLayer(12, 12).double()
    a, b = torch.linspace(.1, .7, 12, dtype=torch.float64), torch.linspace(.8, .2, 12, dtype=torch.float64)
    support = torch.zeros(1, 2, 256, dtype=torch.bool)
    support[0, 0, 10] = support[0, 1, 20] = True
    properties = torch.zeros(1, 256, dtype=torch.float64)
    properties[:, [10, 20]] = membership
    wholes = CandidateWholes(torch.stack((a[:4], b[:4]))[None], support,
        torch.ones(1, 2, dtype=torch.bool), properties, torch.tensor([[.9, .7, .6, .3]], dtype=torch.float64))
    native = wholes.read(a[None], b[None])
    value = op.compose_blocks(op.compose, a[None], b[None], counts=(3, 1),
                              whole=native if centered else None)
    left, right, valid = LanguageSpace._bounded_binary_reconstruction(op, value,
        torch.zeros_like(value), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool),
        torch.stack((a, b))[None], torch.ones(1, 2, dtype=torch.bool), 2, whole_bank=wholes)
    assert valid.all()
    torch.testing.assert_close(left, a[None], rtol=0, atol=1e-12)
    torch.testing.assert_close(right, b[None], rtol=0, atol=1e-12)


@pytest.mark.parametrize('membership', (.3, 1.))
@pytest.mark.parametrize('centered', (False, True))
def test_free_unary_inverse_uses_the_same_native_form_equations(monkeypatch, centered, membership):
    from test_meaning_kernels import paired
    paired.__wrapped__(monkeypatch)
    from Language import LanguageSpace, SurfaceLayer, SurfaceUnaryLayer
    from FormCentroids import CandidateWholes
    op = SurfaceUnaryLayer(SurfaceLayer(12, 12)).double()
    value = torch.linspace(.15, .65, 12, dtype=torch.float64)[None]
    support = torch.zeros(1, 1, 256, dtype=torch.bool)
    support[..., 10] = True
    properties = torch.zeros(1, 256, dtype=torch.float64)
    properties[:, 10] = membership
    bank = CandidateWholes(value[:, None, :4], support, torch.ones(1, 1, dtype=torch.bool),
        properties, torch.tensor([[.9, .7, .6, .3]], dtype=torch.float64))
    parent = op.compose_blocks(op.compose, value, whole=bank.read(value) if centered else None)
    language = SimpleNamespace(_generate_unary_ops=(op,))
    actual = LanguageSpace.generate_unary_step(language, parent, torch.tensor([0]),
        torch.tensor([True]), whole_bank=bank)
    torch.testing.assert_close(actual, value, rtol=0, atol=1e-12)


def test_free_unary_center_requires_a_valid_recovered_form():
    from FormCentroids import CandidateWholes, unary
    # A closer centered dictionary word is insufficient when uncentering
    # this different parent would leave the membership chart.
    forms = torch.tensor([[[.3, .9]]])
    support = torch.zeros(1, 1, 256, dtype=torch.bool)
    support[..., 10] = True
    properties = torch.zeros(1, 256)
    properties[:, 10] = 1.
    bank = CandidateWholes(forms, support, torch.ones(1, 1, dtype=torch.bool),
                           properties, torch.tensor([[.8, .1]]))
    parent = torch.tensor([[.35, .5, .2, .7]])
    assert (((forms + bank.codes)*.5-parent[:, None, :2]).square().sum()
            < (forms-parent[:, None, :2]).square().sum())
    torch.testing.assert_close(unary(parent, bank), parent, rtol=0, atol=0)
