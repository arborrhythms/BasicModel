"""The vertical spread: expectation on the structure's own matrices
(sigma-pi §21, rules 23 to 28, 66 and 68; certificates 14, 15, 17, 19, 20,
21 and 57)."""
import itertools
import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ConceptSpread
from ConceptSpread import Structure, spread


def graph(concepts, symbols, edges):
    """A structure of ``concepts`` concept nodes and ``symbols`` symbol nodes
    from edges ``(parent, child, weight, negated)``."""
    parent, child, weight, negated = zip(*edges)
    return Structure(torch.tensor(parent), torch.tensor(child), torch.tensor(weight, dtype=torch.float),
                     torch.tensor(negated), concepts, symbols, 0, 0, ())


OWNER = SimpleNamespace(_order_caps=lambda: (4, 4, 4))


def present(n, at, value=1.):
    observed = torch.zeros(n, 1, 2)
    observed[at, 0, 0] = value
    return observed


def word_object_meta(*, other=0.):
    """Concepts: the word 0, its object 1, the meta concept 2 (and another
    kind 3 reading the word's symbol at ``other``); symbols: the word's 4,
    the object's 5. The meta concept reads both symbols at .8."""
    edges = [(4, 0, 1., False), (5, 1, 1., False), (2, 4, .8, False), (2, 5, .8, False)]
    if other:
        edges.append((3, 4, other, False))
    return graph(4, 2, edges)


def test_14_up_a_word_found_raises_its_meta_concept_by_its_share():
    value, _ = spread(OWNER, present(4, 0), structure_=word_object_meta(other=.8))
    # The word's symbol is read by two kinds at .8 each: each takes half.
    assert float(value[2, 0, 0]) == pytest.approx(.5)
    assert float(value[3, 0, 0]) == pytest.approx(.5)


def test_15_down_a_primed_meta_concept_raises_its_absent_object():
    value, _ = spread(OWNER, present(4, 0), structure_=word_object_meta())
    # Up: the meta concept takes the word's whole value; down: it gives half
    # to each symbol it reads, and the object's symbol all of it to the object.
    assert float(value[2, 0, 0]) == pytest.approx(1.)
    assert float(value[1, 0, 0]) == pytest.approx(.5)
    # The exact way back returns the inputs as they were: the object, absent,
    # receives nothing from it.
    from Layers import SigmaLayer3
    sigma = SigmaLayer3(2, 2, scale=0.)
    with torch.no_grad():
        sigma.weight.copy_(torch.tensor([[.8, .1], [.8, -.3]]))
    lanes = sigma.reverse(sigma(torch.tensor([1., 0.])))
    assert lanes[..., 1].abs().sum() < 1e-5
    print(dict(meta=float(value[2, 0, 0]), object=float(value[1, 0, 0])))


def test_15b_a_not_entry_passes_to_the_other_lane():
    g = graph(3, 1, [(3, 0, 1., False), (2, 3, 1., True), (1, 3, 1., False)])
    value, _ = spread(OWNER, present(3, 0), structure_=g)
    assert value[2, 0].tolist() == pytest.approx([0., .5])
    assert value[1, 0].tolist() == pytest.approx([.5, 0.])


def test_17_evidence_unchanged_and_no_parameter():
    g = word_object_meta(other=.3)
    observed = present(4, 0, .7)
    before = observed.clone()
    weights = g.weight.clone()
    value, _ = spread(OWNER, observed, structure_=g)
    assert torch.equal(observed, before) and torch.equal(g.weight, weights)
    assert not value.requires_grad
    # The spread holds no module and no parameter of its own.
    source = Path(ConceptSpread.__file__).read_text()
    assert 'nn.Module' not in source and 'Parameter(' not in source


def test_19b_the_join_stays_below_one_and_in_order():
    # k present words, each read by its own kind, each kind reading one other
    # word too: the other word takes a half from each kind. Summed, three
    # halves would pass one; joined, it stays below one and rises with k.
    values = []
    for k in (1, 2, 3, 6):
        n = 2 * k + 1
        edges = [(k + 1 + j, j, 1., False) for j in range(k)] + [(k + 1 + j, k, 1., False) for j in range(k)]
        observed = torch.zeros(n, 1, 2)
        observed[:k, 0, 0] = 1.
        value, _ = spread(SimpleNamespace(_order_caps=lambda: (n,)), observed, structure_=graph(n, 0, edges))
        values.append(float(value[k, 0, 0]))
    print(dict(other_word_by_kinds=values))
    assert values == pytest.approx([1 - .5 ** k for k in (1, 2, 3, 6)])
    assert all(a < b for a, b in zip(values, values[1:])) and values[-1] < 1


def test_19_siblings_below_the_cause():
    parents = [(2 + k, child, 1., False) for k in range(5) for child in (0, 1)]
    value, _ = spread(SimpleNamespace(_order_caps=lambda: (7,)), present(7, 0),
                      structure_=graph(7, 0, parents))
    # Each parent takes a fifth of the cause and gives half of it to the
    # sibling; the five shares join as the product on complements.
    assert float(value[1, 0, 0]) == pytest.approx(1 - .9 ** 5)
    rng = random.Random(0)
    greatest = 0.
    for _ in range(300):
        parents, children = rng.randint(1, 7), rng.randint(2, 9)
        edges = [(children + p, c, rng.uniform(.1, 1.), False)
                 for p in range(parents) for c in range(children) if rng.random() < .6]
        edges += [(children, 0, 1., False), (children, 1, 1., False)]
        n = children + parents
        value, _ = spread(SimpleNamespace(_order_caps=lambda: (n,)), present(n, 0),
                          structure_=graph(n, 0, edges))
        siblings = value[1:children, 0, 0]
        greatest = max(greatest, float(siblings.max()))
        assert bool((siblings < 1).all())
    print(dict(two_children_five_parents=float(1 - .9 ** 5), greatest_sibling=greatest))


def test_20_a_class_reads_members_of_every_lower_order():
    from test_orders_learned import equality_fixture
    owner, _, alloc = equality_fixture()
    word, obj, meta, peer = (owner.new_concept(order=order) for order in (0, 1, 2, 2))
    assert alloc.class_member(word, meta) and alloc.class_member(obj, meta)
    assert not alloc.class_member(peer, meta) and not alloc.class_member(meta, obj)
    # A meta concept over a word of order zero and an object of order one
    # primes the object in one pass, with no symbol between them.
    value, _ = spread(OWNER, present(4, 0), structure_=word_object_meta())
    assert float(value[1, 0, 0]) > 0


def test_21_the_structure_is_readable_and_unchanged_by_the_read():
    from test_orders_learned import equality_fixture
    from StructureRead import parts, wholes
    owner, _, alloc = equality_fixture()
    member, kind = owner.new_concept(order=0), owner.new_concept(order=1)
    assert owner.kind_entry(kind, member) is not None
    store = alloc.layer(0)
    values = store.values.detach().clone()
    knowing = owner._expected_knowing.clone()
    found = parts(owner, kind)
    above = wholes(owner, member)
    print(dict(parts=found, wholes=above))
    assert [entry.reference for entry in found] == [('sym', member)]
    assert [entry.reference for entry in above] == [('sym', kind)]
    assert found[0].weight > 0 and not found[0].negated
    assert torch.equal(store.values.detach(), values)
    assert torch.equal(owner._expected_knowing, knowing)


def test_57a_no_module_of_expectation_beside_the_structure():
    root = Path(__file__).resolve().parents[1] / 'bin'
    for path in root.glob('*.py'):
        text = path.read_text()
        assert not re.search(r'\bexpect_stack\b|\bconcept_predictor\b|expectationMap', text), path.name
    assert 'intuit_map' not in (root / 'ConceptSpread.py').read_text()
    assert 'intuit_map' not in (root / 'LookForecast.py').read_text()


def test_57c_what_is_expected_changes_no_evidence():
    g = word_object_meta(other=.5)
    observed = present(4, 0)
    first, _ = spread(OWNER, observed, structure_=g)
    assert observed[1].abs().sum() == 0 and first[1].abs().sum() > 0


def test_57d_a_stage_cost_reaches_its_own_entries_alone_and_is_written_anew():
    from test_orders_learned import equality_fixture
    owner, _, alloc = equality_fixture()
    member, kind, other = owner.new_concept(order=0), owner.new_concept(order=1), owner.new_concept(order=0)
    owner.kind_entry(kind, member), owner.kind_entry(kind, other)
    rows = [owner._concept_row_of(c) for c in (member, kind, other)]
    V = int(owner.nVectors)
    before = torch.zeros(1, V, 2)
    before[0, rows[0], 0] = 1.
    arrived = torch.zeros(1, V, 2)
    arrived[0, rows[2], 0] = 1.
    store = alloc.layer(0)
    cost = ConceptSpread.look_cost(owner, before, arrived)
    assert cost.requires_grad
    held = [p for p in owner.parameters() if p.requires_grad]
    grads = torch.autograd.grad(cost.sum(), held, allow_unused=True)
    reached = [p for p, g in zip(held, grads) if g is not None and bool(g.abs().sum() > 0)]
    # Only the concepts' own entries: the store's Sigma (and the Sigma of
    # order zero, where it has entries). Nothing of intuit's map, of the
    # codebook, or of any other parameter.
    allowed = {id(store.values)} | ({id(owner.order_zero.value)} if getattr(owner, 'order_zero', None) is not None else set())
    assert reached and all(id(p) in allowed for p in reached)
    # Written anew at each look: nothing earlier is kept.
    owner._expected_knowing.fill_(.3)
    ConceptSpread.write_look(owner, torch.tensor([rows[2]]), torch.full((1, 1, 2), .1))
    assert float(owner._expected_knowing[0].sum()) == pytest.approx(.2)
    print(dict(cost=float(cost.sum()), reached=len(reached)))


def test_68_intuits_map_starts_each_row_on_a_few_concepts_and_learns_exact_concepts():
    """intuit's own map (rule 68): rows started on two concepts learn an
    exact-concept field that rows started on every concept do not."""
    from SigmaPiStack import SigmaPiStack
    from Layers import signed_lanes
    torch.manual_seed(0)
    width = 16
    inputs = torch.zeros(8, width)
    for b in range(8):
        inputs[b, b] = 1.
    target = torch.zeros(8, width)
    for b in range(8):
        target[b, b + 8] = 1.
    lanes = torch.stack((inputs, torch.zeros_like(inputs)), -2)
    learned = {}
    for label, few in (('every', None), ('few', 2)):
        stack = SigmaPiStack(width, width, hidden=64, pairs=1, seed=3, few=few)
        optimizer = torch.optim.Adam(stack.parameters(), lr=.05)
        for _ in range(300):
            optimizer.zero_grad()
            error = (signed_lanes(stack(lanes))[..., 0] - target).square().mean()
            error.backward()
            optimizer.step()
        out = signed_lanes(stack(lanes))[..., 0].detach()
        learned[label] = int(sum(int(out[b].argmax()) == b + 8 and float(out[b, b + 8]) > .5
                                 for b in range(8)))
    print(dict(of_8=learned))
    assert learned['few'] == 8


def test_16_a_meta_concept_primed_by_another_word_prefers_its_object():
    """Rule 3 of 1.8 §55.1: finding a word raises its meta concepts and their
    objects; of the word's two objects, the one whose meta concept another
    word of the sentence primed is preferred at comprehension."""
    from Interpret import InterpretLayer
    # Concepts: the word 0, another word 1, two objects 2 and 3, their meta
    # concepts 4 and 5, a third word 6; symbols 7 to 11 of the word, the
    # other word, the two objects and the third word. Meta concept 4 reads
    # the word, its object and the other word; meta concept 5 the word, its
    # object and the third word, so the two are alike when the word is alone.
    edges = [(7, 0, 1., False), (8, 1, 1., False), (9, 2, 1., False), (10, 3, 1., False), (11, 6, 1., False),
             (4, 7, 1., False), (4, 9, 1., False), (4, 8, 1., False),
             (5, 7, 1., False), (5, 10, 1., False), (5, 11, 1., False)]
    g = graph(7, 5, edges)
    alone, _ = spread(OWNER, present(7, 0), structure_=g)
    observed = present(7, 0)
    observed[1, 0, 0] = 1.
    both, _ = spread(OWNER, observed, structure_=g)
    print(dict(alone=dict(meta=[float(alone[4, 0, 0]), float(alone[5, 0, 0])],
                          objects=[float(alone[2, 0, 0]), float(alone[3, 0, 0])]),
               primed=dict(meta=[float(both[4, 0, 0]), float(both[5, 0, 0])],
                           objects=[float(both[2, 0, 0]), float(both[3, 0, 0])])))
    # The word alone raises both meta concepts and both objects, equally.
    assert float(alone[4, 0, 0]) > 0 and float(alone[5, 0, 0]) > 0
    assert float(alone[2, 0, 0]) == pytest.approx(float(alone[3, 0, 0])) and float(alone[2, 0, 0]) > 0
    # With the other word, the primed meta concept's object is the higher.
    assert float(both[2, 0, 0]) > float(both[3, 0, 0])
    owner = SimpleNamespace(_expected_knowing=both[:7].transpose(0, 1).clone(),
                            _concept_row_of=lambda cid: int(cid), _concept_source_order=lambda cid: 1)
    layer = SimpleNamespace(owner=owner)
    assert InterpretLayer.preferred_objects(layer, (3, 2), batch=0) == (2,)
    assert InterpretLayer.preferred_objects(layer, (2, 3), batch=0) == (2,)


def test_57d_the_percepts_stage_is_registered_beside_the_concepts():
    """Rule 66: the basic percepts are a stage of their own, registered as
    ``expectation.percepts`` beside ``expectation.concepts``; the output is
    no stage."""
    terms = {}
    registry = SimpleNamespace(error=lambda name, value, baseline, **kw: terms.__setitem__(name, (value, kw)))
    mask = torch.tensor([True])
    model = SimpleNamespace(_expectation_looks=[(torch.tensor([.3]), mask)],
                            _expectation_percept_looks=[(torch.tensor([.2]), mask), (torch.tensor([.1]), mask)])
    ConceptSpread.complete(model, mask, registry)
    assert set(terms) == {'expectation.concepts', 'expectation.percepts'}
    assert float(terms['expectation.percepts'][0]) == pytest.approx(.3)
    assert all(kw['category'] == 'expectation' for _, kw in terms.values())
    assert model._expectation_looks == [] and model._expectation_percept_looks == []
