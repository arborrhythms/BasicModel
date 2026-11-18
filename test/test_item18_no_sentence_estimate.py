"""Item 1.8 §68.54.2 and §68.60: native history and the surviving field loss."""
from types import SimpleNamespace as NS
from pathlib import Path
import ast
import pytest
import torch


def field_cost(weight):
    from Layers import SymbolExpectation
    from SymbolField import SymbolField
    from Models import BasicModel
    head = SymbolExpectation(4, 8, 4)
    head.set_symbol_bank(torch.eye(4))
    field = SymbolField(head, strength=0., epsilon=1e-6, saturation='clip')
    values = torch.eye(4)[:2][None]
    state = (values, torch.tensor([2]), torch.zeros(1, 2, dtype=torch.long),
             torch.zeros(1, 2, dtype=torch.long), torch.tensor([[0, 1]]), torch.ones(1, 2))
    field.anticipate((torch.zeros_like(values), torch.tensor([0]), *state[2:]))
    field.arrive(state, torch.tensor([True]))
    root = values[:, 0]
    observation = dict(record=NS(root=root), reference_plans=[], entries=[])
    host = NS(reconstruct_in_loop=False, symbol_field_loss_weight=weight,
        symbolSpace=NS(expectation=None),
        _tensor_pushed_ideas=values, _publish_sentence_scratch=lambda *_: None,
        _trial_understanding=lambda *_: observation['record'],
        _sentence_observation=lambda *_, **__: observation,
        _sentence_symbol_field=field)
    lang = [None]*12
    lang[6] = root[:, None]
    lang[10] = root.new_zeros(1, 1, 12)
    lang[11] = torch.ones(1, 1, dtype=torch.long)
    BasicModel._sentence_path_cost(host, (None, lang, None), 0, torch.tensor([True]))
    return host._sentence_cost_registry, field, tuple(head.parameters())


def test_field_loss_keeps_the_former_weight_and_parameter_gradients():
    errors, field, parameters = field_cost(.1)
    actual = errors.total(objective='expectation')
    # This is the former booking: the same field loss, .1, and the present row.
    expected = field.loss()*.1
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    old = torch.autograd.grad(expected.sum(), parameters, retain_graph=True)
    new = torch.autograd.grad(actual.sum(), parameters)
    for a, b in zip(old, new, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
    assert any(bool(value.abs().sum()) for value in new)


def test_zero_field_weight_books_no_term_and_ladder_values_stay_zero():
    import xml.etree.ElementTree as ET
    errors, field, _ = field_cost(0.)
    assert field.loss() is not None
    assert not any(name == 'expectation.symbols' for name, *_ in errors.terms())
    for name in ('MM_ladder.xml', 'MM_ladder_idiom.xml'):
        root = ET.parse(Path(__file__).parents[1]/'data'/name).getroot()
        assert float(root.findtext('architecture/training/symbolFieldLossWeight')) == 0.
        assert root.find('architecture/training/sentenceExpectationLossWeight') is None


def test_consumed_surface_relation_is_admitted_without_a_prediction_observation():
    from test_reference_close import fixture, known_concept
    from ClauseRow import Composition, Clause, ClausePredicate, predicate_point
    from Meaning import ConceptualMeaning
    from SentenceStrata import IdeaStack
    from PartClosing import PartClosing
    from HeldConcepts import Evaluation
    from Comprehension import from_stack
    from Memory import OCCURRENCE
    from Layers import SymbolExpectation
    resolver, store, points = fixture()
    owner = resolver.components.owner
    ids = [known_concept(owner, points[i], order=0) for i in range(4)]
    leaves = [Composition(points[i], reference=ids[i], order=0) for i in range(4)]
    left = Composition(points[0]+points[1], tuple(leaves[:2]), operation='surface',
                       head_role=2, reference=ids[1], order=0)
    right = Composition(points[2]+points[3], tuple(leaves[2:]), operation='surface',
                        head_role=2, reference=ids[3], order=0)
    root = Composition(right.point, (left, right), operation='part', order=0)
    predicate = ClausePredicate(predicate_point('part', points[0]), 'part')
    clause = Clause(ConceptualMeaning(torch.stack((left.point, predicate.point, right.point)),
        torch.ones(3, dtype=torch.bool)), relation='part', refs=(ids[1], predicate, ids[3]), composition=root)
    stack = IdeaStack(store.nDim, store.capacity, relation_capacity=store.parts.capacity)
    tree = NS(roots=((OCCURRENCE, 100),), semantic_start=101)
    stack.prepare(store, tree, {id(node): i+50 for i, node in enumerate([*leaves, left, right, root])})
    closing = PartClosing(owner, store, working=stack)
    closing.stage(clause)
    held = Evaluation(None, None).retain_arrivals(clause, (), closing.staged_points)
    kept = from_stack(clause, {**held.values, **closing.staged_points})
    assert len(stack) == 0 and len(stack.parts) == 1
    assert kept.composition.children[0].point is None
    torch.testing.assert_close(kept.meaning.roles[0], left.point)
    disc = SymbolExpectation(512, 3, 12)
    disc._ltm_store = store
    row = store.write_clause(kept)
    occurrence = store.row(row)['occurrence']
    # The close's real admission needs no preceding forecast target.
    disc.include_occurrence(0, occurrence, symbols=(ids[1], ids[3]))
    frame, = disc.situation_references()
    assert frame.occurrence == occurrence and frame.symbol_ids == (ids[1], ids[3])
    assert not hasattr(disc, 'observe_stm_end_state')


def test_production_has_no_sentence_estimator_calls():
    removed = {'expect_next_meaning', 'predict_next_end_state', 'sentence_prediction_cost',
               'observe_stm_end_state', 'predict_and_observe_stm_end_state',
               'bind_observation_occurrence', 'consume_inter_loss', 'generate_sentence'}
    for path in (Path(__file__).parents[1]/'bin').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                name = getattr(node.func, 'attr', getattr(node.func, 'id', None))
                assert name not in removed, (path.name, node.lineno, name)


def test_first_documents_kept_closes_keep_the_same_recent_frames():
    """Replay the same native admissions, independently of subsequent learning."""
    import json
    from Layers import SymbolExpectation
    from Meaning import ConceptualMeaning
    fixture = json.loads((Path(__file__).parent/'fixtures/item18-first-document-situation.json').read_text())
    def tuples(value):
        return tuple(map(tuples, value)) if isinstance(value, list) else value
    def plain(value):
        return [plain(v) for v in value] if isinstance(value, (tuple, list)) else value
    rows = {}
    store = NS(location=lambda occurrence: occurrence if occurrence in rows else None,
               row=lambda occurrence: rows[occurrence])
    head = SymbolExpectation(1, 1, len(fixture['closes'][0]['retained'][0]['roles'][0]))
    head._ltm_store = store
    for close, expected in zip(fixture['closes'], fixture['expected'], strict=True):
        # These are the rows the historical native close actually wrote;
        # the test adapter only exposes their existing row-reader interface.
        for saved in close['retained']:
            occurrence = tuples(saved['occurrence'])
            rows[occurrence] = dict(saved, occurrence=occurrence,
                point=None if saved['point'] is None else torch.tensor(saved['point']),
                time=None if saved['time'] is None else torch.tensor(saved['time']),
                meaning=ConceptualMeaning(torch.tensor(saved['roles']), torch.tensor(saved['occupied'])))
        head.include_occurrence(0, tuples(close['occurrence']),
            symbols=tuples(close['symbols']), activations=tuples(close['activations']))
        actual = [dict(occurrence=plain(f.occurrence), symbols=plain(f.symbol_ids),
                       activations=plain(f.activations)) for f in head.situation_references()]
        assert actual == expected
