"""Round A's operator contracts; certificate 70 is deferred by §31.3."""
from pathlib import Path
from types import SimpleNamespace
import re
import json
import xml.etree.ElementTree as ET

import pytest
import torch


def test_69_complete_inventory():
    root = ET.parse(Path(__file__).parents[1] / 'data/complete.grammar').getroot()
    def names(section):
        return {re.search(r'=\s*(\w+)\.', row.text)[1]
                for row in root.findall('Symbolic/' + section + '/rule')}
    common = {'interpret', 'part', 'whole', 'implies', 'ask', 'not', 'non',
              'intersection', 'union', 'add', 'subtract', 'multiply', 'divide',
              'change', 'lift', 'lower', 'verb', 'compound', 'marker',
              'modifier', 'preposition', 'surface', 'swap', 'bind'}
    assert names('compose') == common | {'focus', 'unfocus', 'sign'}
    assert names('generate') == common - {'bind'}
    assert names('thought') == {'ask', 'query', 'part', 'whole', 'not', 'non',
        'bind', 'isTrue', 'exist', 'isPart', 'isWhole', 'isEqual', 'infer',
        'inferBackward', 'intuit', 'implies', 'define'}
    assert all('referent' not in row.attrib for section in ('compose', 'generate', 'thought')
               for row in root.findall('Symbolic/' + section + '/rule'))


def test_69_focus_splits_the_existing_bracket_at_one_act():
    from Language import FocusLayer, UnfocusLayer
    from Attention import BracketTable
    assert FocusLayer.bracket_action == (0, 1)
    assert UnfocusLayer.bracket_action == 2
    table = BracketTable.open(torch.tensor([11, 9]), budget=4)
    result = table.split(torch.tensor([0, 0]), torch.tensor([5, 4]), descend=True)
    assert result.intervals[:, :2].tolist() == [[[0, 5], [5, 11]], [[0, 4], [4, 9]]]
    assert result.level[:, :2].tolist() == [[1, 1], [1, 1]]
    assert result.spent.tolist() == [1, 1]


@pytest.mark.parametrize('name', ['add', 'subtract', 'multiply', 'divide'])
def test_71_arithmetic_is_in_the_chart_on_each_lane(name, monkeypatch):
    from Language import GRAMMAR_LAYER_CLASSES
    from util import TheXMLConfig
    monkeypatch.setitem(TheXMLConfig.data, 'architecture', {'data': {'dataType': 'numeric'}})
    left = torch.tensor([[.1, .9, .3, 0., .8, .6]], dtype=torch.float64)
    right = torch.tensor([[.8, .2, 0., .4, .1, .7]], dtype=torch.float64)
    a, b = left.atanh(), right.atanh()
    expected = {'add': a+b, 'subtract': (a-b).clamp_min(0), 'multiply': a*b,
                'divide': torch.where(b != 0, a / torch.where(b != 0, b, 1.), 0.)}[name].tanh()
    op = GRAMMAR_LAYER_CLASSES[name](6, 6)
    result = op.compose(left, right)
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-15)
    assert bool(((result >= 0) & (result <= 1)).all())
    if name == 'subtract':
        assert result[0, 0] == result[0, 5] == 0
    if name == 'divide':
        assert result[0, 2] == 0


def test_72_lift_and_lower_have_opposite_orders_and_kernels(monkeypatch):
    from Language import LiftLayer, LowerLayer, LowerUnaryLayer
    from ReferenceContext import denoted_order
    from util import TheXMLConfig
    monkeypatch.setitem(TheXMLConfig.data, 'architecture', {'data': {'dataType': 'embedding'}})
    monkeypatch.setitem(TheXMLConfig.data, 'PartSpace', {'nDim': 12})
    monkeypatch.setitem(TheXMLConfig.data, 'ConceptualSpace', {'nDim': 20})
    base = LowerLayer(20, 20)
    up, down = LiftLayer(base), LowerUnaryLayer(base)
    assert up.arity == down.arity == 1
    orders = torch.tensor([-1, 0, 1, 2])
    assert denoted_order(down, (orders,), (), None).tolist() == [-1, -1, 0, 1]
    lifted = denoted_order(up, (orders,), (), None)
    assert lifted.tolist() == [-1, 1, 2, 3]
    torch.testing.assert_close(denoted_order(down, (lifted,), (), None), orders)
    value = torch.linspace(.1, .8, 20).reshape(1, 20)
    with torch.no_grad():
        base.meaning_kernel.log_scale.fill_(.2)
        base.meaning_kernel.bias.fill_(.05)
    torch.testing.assert_close(down.compose(up.compose(value)), value, rtol=1e-5, atol=1e-6)


def test_72_lower_is_not_an_admitted_order_zero_choice():
    from ReferenceContext import prepare_operands, ReferenceBank
    from Language import Grammar
    grammar = Grammar()
    grammar.configure({'compose': {'rule': {'implementation': 'lower_unary',
        '_': 'lower_argout1 = lower.forward(lower_argin1)'}}})
    value = torch.tensor([[[.2, .3, .4, .5]], [[.2, .3, .4, .5]]])
    ids = torch.tensor([[1], [2]])
    bank = ReferenceBank(ids, value, torch.ones(2, 1, dtype=torch.bool),
        torch.zeros(2, 1, dtype=torch.bool), value[:, 0], torch.zeros(2, dtype=torch.bool))
    result = prepare_operands(value, ids, torch.zeros_like(ids), torch.zeros_like(ids),
        orders=torch.tensor([[0], [1]]), rules=(), unary_rules=tuple(grammar.rules_upward),
        bank=bank, live=None, active=torch.ones_like(ids, dtype=torch.bool), chunk_state=())
    assert not result['unary_valid'][0].any()
    assert result['unary_valid'][1].any()


def test_72_order_changes_are_distinct_at_the_identity_kernel():
    from Language import OperationSelectionLayer
    point = torch.tensor([[[.2, .3]]])
    found = OperationSelectionLayer._distinct_departures(point, torch.tensor([1]),
        point.expand(1, 3, 2), torch.zeros(1, 3), torch.tensor([2]),
        0, 2, 0, 2, order_deltas=torch.tensor([1, -1, 0]))
    assert found.tolist() == [[True, True, False]]


def test_74_is_whole_is_the_converse_part_read():
    from Language import Grammar
    from Queries import GrammaticalThoughtRegistry
    from test_query_vp_boundaries import _context
    from test_cs_symbol_table import _cs
    cs = _cs()
    left, right = (cs.new_concept(order=0) for _ in range(2))
    cs.add_whole(left, ('sym', right))
    grammar = Grammar()
    grammar.configure({'thought': {'rule': [
        'isPart_argout1 = isPart.thought(isPart_argin1, isPart_argin2)',
        {'_': 'isWhole_argout1 = isWhole.thought(isWhole_argin1, isWhole_argin2)',
         'family': 'isPart', 'permutation': 'argin2,argin1'}]}})
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    first = registry.execute(registry.form('isPart', ('sym', left), ('sym', right)), _context(cs))
    second = registry.execute(registry.form('isWhole', ('sym', right), ('sym', left)), _context(cs))
    assert first.support_true == second.support_true == 1
    assert first.support_false == second.support_false == 0
    assert first.evidence['path'] == second.evidence['path']


def test_73_lower_reads_the_words_existing_object_one_order_down():
    from ReferenceContext import ReferenceTypes, ReferenceBank, prepare_operands
    from Language import Grammar
    grammar = Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    rule = next(rule for rule in grammar.rules_upward if rule.method_name == 'lower_unary')
    types = ReferenceTypes(torch.tensor([[10]]), torch.tensor([0, 1, 2]),
        torch.tensor([[[11, 12, 13]]]), torch.arange(12.).reshape(1, 1, 3, 4))
    for order in (1, 2):
        value = types.values[:, :, order]
        ids = types.identities[:, :, order]
        bank = ReferenceBank(ids, value, torch.ones_like(ids, dtype=torch.bool),
            torch.zeros_like(ids, dtype=torch.bool), value[:, 0], torch.tensor([False]),
            types=types, close_references=True)
        read = prepare_operands(value, ids, torch.zeros_like(ids), torch.zeros_like(ids),
            orders=torch.full_like(ids, order), rules=(), unary_rules=(rule,), bank=bank,
            live=None, active=torch.ones_like(ids, dtype=torch.bool), chunk_state=())
        assert read['unary_refs'][0, 0, 0, 0] == types.identities[0, 0, order-1]
        torch.testing.assert_close(read['unary'][0, 0, 0], types.values[0, 0, order-1])
        assert read['unary_valid'].all()


def test_74_kept_definition_has_a_checked_chain_and_writes_both_directions(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from test_query_vp_boundaries import _context
    from ThoughtStream import materialize, write
    from ThoughtReferences import bindings, validate_local_references
    f = SentenceFixture(monkeypatch)
    left, right = (f.noun(word) for word in ('first', 'second'))
    request = f.registry.form('define', ('sym', left), ('sym', right))
    raw = f.registry.execute(request, _context(f.cs, store=f.store))
    meaning = materialize(raw).evidence['meaning']
    validate_local_references(meaning)
    assert bindings(meaning)['_producing_operation'] == 'define'
    assert meaning.role_refs == request.role_refs
    write(SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=f.store),
                          grammatical_thoughts=f.registry), meaning, row=0)
    for a, b in ((left, right), (right, left)):
        edge = f.store.parts.find(a, b)
        assert edge is not None
        assert tuple(f.store.parts.evidence(edge)) == pytest.approx(
            (raw.support_true, raw.support_false))


def test_73_lowered_word_keeps_its_selected_concept_through_an_event_close(monkeypatch):
    from test_modal_timelines import fixture, close
    from chunk_fixtures import record
    f, action, modifier = fixture(monkeypatch)
    high = f.noun('kind', order=2)
    low = f.cs.new_concept(order=1)
    f.cs.interpret.define(f.words['kind'][0], low)
    with torch.no_grad():
        f.cs.similarity_codebook.W[f.cs._csw_row_of(high)].fill_(.4)
        f.cs.similarity_codebook.W[f.cs._csw_row_of(low)].fill_(.2)
    f.ops['lower_unary'] = SimpleNamespace(compose=lambda value: value)
    reading, _, _ = record(f, ('change', ('lower_unary', 'kind'), ('modifier', 'move', 'shift')))
    lowered = next(i for i, (kind, local, _) in enumerate(reading.actions.tolist())
                   if kind == 2 and f.unary[local].method_name == 'lower_unary')
    assert reading.operation_refs[lowered, 0] == low
    assert reading.identity_actions[lowered, 0] > 0
    clause, _, row, _ = close(f, reading, actions=(action, modifier))
    entry, = f.cs.mtm.entries.values()
    assert clause.order == f.store.order_of_row(row) == 1
    assert not entry.events and len(entry.timelines[1].events) == 1
    assert f.store.raw_meaning_of(f.store.location(entry.occurrences[0])).role_refs[0] == ('sym', low)


def test_74_define_writes_two_parts_and_counts_their_witnesses():
    from ClauseRow import Clause, Composition, predicate_identity
    from Meaning import ConceptualMeaning
    from Memory import SYMBOL
    from ChainRead import point
    from test_orders_learned import equality_fixture
    from test_corpus_orders import named_individual
    owner, store, alloc = equality_fixture()
    members = [named_individual(owner, store, alloc, str(i), 3*i+1) for i in range(4)]
    shared = members[0]
    name, = owner.definitions.words(shared)
    for member in members[1:]:
        values = tuple(Composition(point(store, (SYMBOL, cid)), reference=cid,
            word_id=cid, order=alloc.order_of(cid)) for cid in (shared, member))
        term = Composition(values[0].point, values, operation='define', order=1)
        store.write_clause(Clause(ConceptualMeaning.from_description(term.point),
            relation='define', refs=(shared, predicate_identity('define'), member),
            composition=term, order=1), evidence=(1., 0.))
        assert alloc.parts.find(shared, member) is not None
        assert alloc.parts.find(member, shared) is not None
    parent = owner.interpret.forward(name)
    assert alloc.order_of(parent) == 2 and alloc.order_of(shared) == 1
    assert set(alloc.refs(parent, 'part')) == {('sym', member) for member in members[1:]}
    assert name in owner.definitions.words(shared)


def test_non_thought_withholds_both_lanes():
    from ThoughtFaces import withhold
    from Meaning import ConceptualMeaning
    from ThoughtReferences import with_slots, evidence_pair
    for lanes in ((.7, 0.), (0., .8), (.7, .8)):
        value = with_slots(ConceptualMeaning.from_description(torch.ones(8)), (), pair=lanes)
        result = withhold(None, {'argin1': value})
        assert result['support_true'] == result['support_false'] == 0
        assert evidence_pair(result['meaning']) == (0., 0.)
        assert not result['meaning'].roles.any()


def test_69_76_78_split_keeps_the_fixed_action_and_its_reverse(monkeypatch):
    from Language import ChangeLayer, VerbLayer, PrepositionLayer, SwapLayer
    from MeaningKernels import LaneKernel
    from util import TheXMLConfig
    monkeypatch.setitem(TheXMLConfig.data, 'architecture', {'data': {'dataType': 'embedding'}})
    monkeypatch.setitem(TheXMLConfig.data, 'PartSpace', {'nDim': 12})
    monkeypatch.setitem(TheXMLConfig.data, 'ConceptualSpace', {'nDim': 20})
    fixture = json.loads((Path(__file__).parent / 'fixtures/item18-change-action.json').read_text())
    change, verb, preposition = (cls(20, 20).double() for cls in (ChangeLayer, VerbLayer, PrepositionLayer))
    assert not hasattr(verb, 'meaning_kernel')
    assert isinstance(preposition.meaning_kernel, LaneKernel)
    change.meaning_action.bind_basis(verb.meaning_basis)
    parameters = fixture['parameters']
    with torch.no_grad():
        change.meaning_action.readout.copy_(torch.tensor(parameters['readout'], dtype=torch.float64))
        change.meaning_action.groups.vectors.copy_(torch.tensor(parameters['groups.vectors'], dtype=torch.float64))
        verb.meaning_basis.vectors.copy_(torch.tensor(parameters['object_vectors'], dtype=torch.float64))
    left, v, obj = (torch.tensor(fixture[key], dtype=torch.float64) for key in ('left', 'verb', 'object'))
    phrase = verb.compose(v, obj)
    torch.testing.assert_close(phrase, torch.tensor(fixture['phrase'], dtype=torch.float64), rtol=0, atol=0)
    recovered_v, recovered_o = verb.generate(phrase, object_what=obj)
    torch.testing.assert_close(recovered_v, v, rtol=0, atol=1e-15)
    torch.testing.assert_close(recovered_o, obj, rtol=0, atol=0)
    for name, current, counts, context in (
            ('object', phrase, (1, 2), dict(object_value=obj, object_present=torch.tensor([True]), group=torch.tensor([0]))),
            ('group', v, (1, 1), dict(group=torch.tensor([0])))):
        parent = change.compose_blocks(change.compose, left, current, counts=counts, action_context=context)
        torch.testing.assert_close(parent, torch.tensor(fixture[name+'_parent'], dtype=torch.float64), rtol=0, atol=1e-15)
        reverse = change.reverse_blocks(parent, current, counts=counts, action_context=context)
        for value, expected in zip(reverse, fixture[name+'_reverse']):
            torch.testing.assert_close(value, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=1e-15)
    for found, expected in zip(SwapLayer().compose(left, v), fixture['swap']):
        torch.testing.assert_close(found, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0)
    assert all(parameter is not verb.meaning_basis.vectors for parameter in change.parameters())


def test_77_np_and_pp_meanings_use_one_basis_reader(monkeypatch):
    from Language import ChangeLayer, VerbLayer, PrepositionLayer
    from util import TheXMLConfig
    monkeypatch.setitem(TheXMLConfig.data, 'architecture', {'data': {'dataType': 'embedding'}})
    monkeypatch.setitem(TheXMLConfig.data, 'PartSpace', {'nDim': 12})
    monkeypatch.setitem(TheXMLConfig.data, 'ConceptualSpace', {'nDim': 20})
    change, verb, prep = (cls(20, 20).double() for cls in (ChangeLayer, VerbLayer, PrepositionLayer))
    change.meaning_action.bind_basis(verb.meaning_basis)
    with torch.no_grad():
        verb.meaning_basis.vectors[0, 1, 0] = .3
        prep.meaning_kernel.context.copy_(torch.eye(4, dtype=torch.float64) * .1)
    v = torch.linspace(.1, .5, 20, dtype=torch.float64).reshape(1, 20)
    obj = v.flip(-1)
    first = prep.compose(v, obj)
    second = prep.compose(v * .7, obj)
    seen = []
    read = verb.meaning_basis.read
    monkeypatch.setattr(verb.meaning_basis, 'read', lambda point: (seen.append(point), read(point))[1])
    matrices = []
    for phrase in (obj, first, second):
        attached = verb.compose(v, phrase)
        torch.testing.assert_close(attached[:, 12:], v[:, 12:], rtol=0, atol=0)
        matrices.append(change.meaning_action.basis(v[:, 12:], object_value=phrase[:, 12:])[0])
    assert len(seen) == 3
    assert not torch.equal(matrices[1], matrices[2])
