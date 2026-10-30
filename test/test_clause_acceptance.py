"""Two-truths §7: forced grammatical derivations, with the real operator faces.

The strings label reconstruction leaves. Only selected rules classify them;
these fixtures do not assert that an untrained chooser parses these sentences.
"""
from types import SimpleNamespace
import torch
import pytest

import Language
from ClauseRow import attach_clause_index
from reading_fixtures import finish_reading
from Layers import LTM0
from Queries import GrammaticalThoughtRegistry
from Understanding import Reading
from test_cs_sparse_weights import _cs


class SentenceFixture:
    def __init__(self, monkeypatch, *, native_cube=True, percept_width=0):
        self.native_cube = native_cube
        self.cs = _cs(nS=512, order=5)
        if native_cube:
            # These are native concept fixtures; signed event charts are not
            # valid identity inputs, including newly admitted objectless forms.
            with torch.no_grad():
                self.cs.similarity_codebook.W.copy_(self.cs.similarity_codebook.W.sigmoid())
        self.grammar = Language.Grammar()
        self.grammar.load_from_grammar_file('complete.grammar')
        monkeypatch.setattr(Language, 'TheGrammar', self.grammar)
        self.registry = GrammaticalThoughtRegistry.install(self.cs, self.grammar)
        self.binary = tuple(r for r in self.grammar.rules_upward if r.arity == 2 and r.space_role == 'CS')
        self.unary = tuple(r for r in self.grammar.rules_upward if r.arity == 1)
        self.ops = {}
        def operation(rule):
            name = rule.method_name
            if name not in self.ops:
                cls = Language.GRAMMAR_LAYER_CLASSES[name]
                if issubclass(cls, Language.MarkerUnaryLayer):
                    base = Language.GRAMMAR_LAYER_CLASSES[cls.marker_base](8, 8)
                    self.ops[name] = cls(base)
                else:
                    self.ops[name] = cls(8, 8) if name in ('lift', 'verb', 'lower', 'surface', 'sum', 'implies') else cls()
            return self.ops[name]
        self.operation = operation
        self.language = SimpleNamespace(_compose_binary_rules=self.binary, _compose_unary_rules=self.unary,
            # Retrieval of unrecorded estimates uses an explicit stop-only
            # generate fixture; it cannot manufacture a reconstruction trace.
            _generate_binary_ops=(), _generate_unary_ops=(), generate_policy_logits=lambda top,features=None: top.new_zeros(len(top), 1),
            forward_binary_step=lambda a, b, op, _: operation(self.binary[int(op[0])]).compose(
                a[:, None], b[:, None]).reshape_as(a),
            forward_unary_step=lambda a, op, _: operation(self.unary[int(op[0])]).compose(a).reshape_as(a))
        self.words = {}
        from PartsOfSpeech import PartsOfSpeech, formal_arguments
        self.language.parts_of_speech = PartsOfSpeech(formal_arguments((*self.binary, *self.unary)))
        self.store = LTM0(8, capacity=64, percept_width=percept_width)
        self.model = SimpleNamespace(languageSpace=self.language, grammatical_thoughts=self.registry)
        attach_clause_index(self.model, self.store, self.cs)

    def noun(self, word):
        if word not in self.words:
            wid = self.cs.interpret.lookup_word([len(self.words) + 1], [], form=word)
            # These certificates supply object meanings. A bare new form
            # alone now deliberately remains an unbound reference site.
            obj = self.cs.new_concept()
            self.cs.add_part(obj, len(self.words) + 1)  # The fixture supplies the witnessed form.
            seat = self.cs._csw_concept_row(0, obj)
            if self.native_cube:
                with torch.no_grad():
                    self.cs.similarity_codebook.W[seat].copy_(torch.eye(8)[len(self.words) % 8])
            self.cs.interpret.define(wid, obj)
            self.words[word] = wid, self.cs.interpret.forward(wid)
        return self.words[word][1]

    def program(self, tree, *, located=True):
        actions, leaves, words, ids, word_rows, word_ids = [], [], [], [], [], []
        def visit(node):
            if isinstance(node, str):
                obj = self.noun(node)
                wid = self.words[node][0]
                point = self.registry._payload(('sym', obj))
                actions.append((0, -1, len(leaves)))
                leaves.append(point); ids.append(obj); words.append(node)
                word_rows.append(self.cs._csw_row_of(obj))
                word_ids.append(wid)
                return point
            name, *children = node
            values = tuple(visit(child) for child in children)
            field_bind = name == 'bind'
            catalog = self.binary if len(children) == 2 or field_bind else self.unary
            local = next(i for i, rule in enumerate(catalog) if rule.method_name == name)
            actions.append((1 if len(children) == 2 or field_bind else 2, local, -1))
            if field_bind:
                values = (*values,torch.zeros_like(values[0]))
            return self.operation(catalog[local]).compose(*(v.reshape(1, 1, -1) for v in values)).reshape(-1)
        root = visit(tree)
        count = len(leaves)
        stop = len(self.binary) + len(self.unary)
        targets = [stop if kind == 0 else op if kind == 1 else len(self.binary) + op
                   for kind, op, _ in reversed(actions)]
        return Reading(rows=torch.tensor([self.cs._csw_row_of(cid) for cid in ids]),
            word_rows=torch.tensor(word_rows), word_ids=torch.tensor(word_ids), activations=torch.ones(count), leaves=torch.stack(leaves),
            actions=torch.tensor(actions), targets=torch.tensor(targets), concept_ids=torch.tensor(ids),
            end_state=torch.stack((root, root * 0, root * 0)), lexical_forms=(None,) * count,
            symbol_where=torch.ones(count, 4) if located else None,
            symbol_when=torch.tensor([1., 0., 0., 1.]).expand(count, 4).clone() if located else None)

    def clause(self, tree, **kwargs):
        return finish_reading(self.language, self.program(tree, **kwargs), registry=self.registry)


def test_01_compound_vp_has_one_idea_and_readable_cat_chase_mouse(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('lift', ('lower', 'the', 'cat'), ('verb', 'chased', ('lower', 'the', 'mouse'))))
    row = f.store.write_clause(clause, trust=.7)
    assert len(f.store) == 1 and f.store.ideas().tolist() == [row]
    assert not hasattr(f.store, 'refs')
    assert len(f.store.parts) == 4  # existing definitions; no composing rows
    # The factored target belongs to the open reading; only the fused point
    # survives. Its numeric operands are the actual compound NP/VP fields.
    assert clause.subject_word_id == f.words['cat'][0]
    torch.testing.assert_close(f.store.meaning_of(row).roles[0], clause.point)
    assert not f.store.meaning_of(row).roles[1:].any()


def test_04_absolute_clauses_close_bottom_up_with_their_part_edges(monkeypatch):
    f = SentenceFixture(monkeypatch)
    inner = ('lift', 'it', ('verb', 'is', ('surface', 'so', 'beautiful')))
    middle = ('lift', 'she', ('verb', 'said', inner))
    clause = f.clause(('lift', 'he', ('verb', 'said', middle)))
    row = f.store.write_clause(clause, trust=.8)
    assert row == 0 and f.store.ideas().tolist() == [row]
    assert len(f.store.parts) == len(f.words)  # existing definitions only
    torch.testing.assert_close(f.store.meaning_of(0).roles[0], clause.point)
    assert not hasattr(f.store, 'clause_derivation')


def test_05_embedded_part_is_unasserted_and_propagates_operator_kind(monkeypatch):
    f = SentenceFixture(monkeypatch)
    inner = ('part', 'cats', 'animals')
    clause = f.clause(('lift', 'he', ('verb', 'said', inner)))
    row = f.store.write_clause(clause, trust=.6)
    child = f.store.parts.find(f.noun('cats'), f.noun('animals'))
    outer = f.store.row(row)
    verb = int(outer['refs'][1])
    assert f.store.parts.refs[verb, 1] == child
    assert len(f.store) == 0
    assert f.store.row(('sym', child))['evidence'] == (0., 0.)
    assert outer['evidence'] == pytest.approx((.6, 0.))


def test_12_generic_subject_is_a_relation_with_either_predicate(monkeypatch):
    f = SentenceFixture(monkeypatch)
    for predicate in ('breathe', ('verb', 'are', 'animals')):
        relative = f.clause(('lift', ('generic', 'cats'), predicate))
        absolute = f.clause(('lift', ('lower', 'this', 'cat'), predicate))
        assert relative.relation == 'part' and absolute.relation is None


def test_02_part_uses_object_concepts_and_reassertion_joins_evidence(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('part', 'cats', 'animals'))
    row = f.store.write_clause(clause, trust=.6)
    assert f.store.row(row)['refs'].tolist() == [f.noun('cats'), f.noun('animals')]
    count = len(f.store.parts)
    assert f.store.write_clause(clause, trust=.8) == row and len(f.store.parts) == count
    assert f.store.row(row)['evidence'] == pytest.approx((.8, 0.))
    assert f.noun('animals') in [ref for _, ref, _ in f.store.consequents_by_row(f.noun('cats'))]


@pytest.mark.parametrize('predicate', ['black', ('verb', 'is', 'animal')])
def test_03_particular_subject_ends_an_idea(monkeypatch, predicate):
    f = SentenceFixture(monkeypatch)
    row = f.store.write_clause(f.clause(('lift', ('lower', 'the', 'cat'), predicate)))
    assert f.store.ideas().tolist() == [row]
    assert len(f.store.parts) == len(f.words)


def test_06_implication_keeps_two_unasserted_relation_operands(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('implies', ('part', 'cats', 'animals'), ('part', 'cats', 'breathers')))
    row = f.store.write_clause(clause, trust=.7)
    left, right = f.store.relation_operands(row)
    assert left['refs'].tolist() == [f.noun('cats'), f.noun('animals')]
    assert right['refs'].tolist() == [f.noun('cats'), f.noun('breathers')]
    assert left['evidence'] == right['evidence'] == (0., 0.)
    assert f.store.row(row)['readings'] == ('conditional',)
    assert f.store.row(row)['evidence'] == pytest.approx((.7, 0.))
    assert len(f.store) == 0


def test_implication_closes_bare_np_arguments_as_idea_clauses(monkeypatch):
    f = SentenceFixture(monkeypatch)
    row = f.store.write_clause(f.clause(('implies', 'rain', 'wet')))
    assert f.store.ideas().tolist() == []
    assert f.store.row(row)['refs'].tolist() == [f.noun('rain'), f.noun('wet')]


def test_07_equality_is_one_membership_in_a_singleton(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('equal', 'morning-star', 'evening-star'))
    count = len(f.store.parts)
    first = f.store.write_clause(clause, trust=.7, evidence=(.7, 0.))
    a, b = f.noun('morning-star'), f.noun('evening-star')
    assert len(f.store.parts) == count + 1 and len(f.store) == 0
    assert f.store.row(first)['refs'].tolist() == [a, b]
    assert 'equality' in f.store.row(first)['readings']
    assert f.store.parts.find(b, a) is None
    f.store.set_evidence(first, .2, .4)
    assert f.store.row(first)['evidence'] == pytest.approx((.2, .4))


def test_08_direct_bare_np_rewitnesses_its_address(monkeypatch):
    f = SentenceFixture(monkeypatch, percept_width=8)
    direct = f.store.write_clause(f.clause('cat', located=False))
    assert f.store.row_ids[direct] == f.store.address_keys[direct] != f.noun('cat')
    located = f.store.write_clause(f.clause('cat'))
    assert located == direct and int(f.store.witness_count[direct]) == 2
    assert f.store.entry_when(located).any()


def test_09_attribution_cannot_assert_its_child_and_negation_is_content(monkeypatch):
    f = SentenceFixture(monkeypatch)
    inner = ('part', 'cats', 'animals')
    for attitude in ('certain', 'doubtful'):
        f.store.write_clause(f.clause(('lift', 'it', ('verb', attitude, inner))), trust=.8)
    edge = ('sym', f.store.parts.find(f.noun('cats'), f.noun('animals')))
    assert f.store.row(edge)['evidence'] == (0., 0.)
    f.store.write_clause(f.clause(('not', inner)), trust=.6, evidence=(0., .6))
    assert f.store.row(edge)['evidence'] == pytest.approx((0., .6))


def test_end_clause_survives_a_later_head_projection(monkeypatch):
    f = SentenceFixture(monkeypatch)
    earlier = ('lift', 'rain', 'falls')
    clause = f.clause(('lift', ('lower', earlier, 'cat'), 'runs'))
    row = f.store.write_clause(clause)
    assert row == 0 and len(f.store) == 1
