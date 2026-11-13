"""Data-driven authority gates; fixtures supply views and kept occupations."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch
from ClauseRow import Clause, Composition
from ConceptTrust import positive_length
from IdentityRows import mint
from Meaning import ConceptualMeaning
from definition_fixtures import agreeing_views, learned_operation
from test_review13_subspaces import fixture as paired_fixture


def gate_data(monkeypatch):
    from util import TheXMLConfig
    data=dict(closedTrust=.1,threshold=.75)
    original=TheXMLConfig.space
    monkeypatch.setattr(TheXMLConfig,'space',lambda name,key,*args,**kw:
        data if name=='ConceptualSpace' and key=='gates' else original(name,key,*args,**kw))
    return data


def gate_fixture(monkeypatch):
    import Language
    from ClauseRow import attach_clause_index
    from Memory import LTM0
    from PartsOfSpeech import PartsOfSpeech,formal_arguments
    grammar=Language.Grammar();grammar.load_from_grammar_file('complete.grammar')
    monkeypatch.setattr(Language.TheGrammar,'rules_upward',grammar.rules_upward)
    monkeypatch.setattr(Language.TheGrammar,'ws_start_patterns',grammar.ws_start_patterns)
    monkeypatch.setattr(Language.TheGrammar,'_ensure_configured',lambda:None)
    owner,bank,_,alloc = paired_fixture(meaning_pairs=4, capacity=160)
    binary = tuple(rule for rule in grammar.rules_upward if rule.arity == 2)
    unary = tuple(rule for rule in grammar.rules_upward if rule.arity == 1)
    def operation(rule):
        if rule.method_name is None: return None
        cls = Language.GRAMMAR_LAYER_CLASSES[rule.method_name]
        if issubclass(cls, Language.MarkerUnaryLayer):
            return cls(Language.GRAMMAR_LAYER_CLASSES[cls.marker_base](owner.nDim, owner.nDim))
        return cls(owner.nDim, owner.nDim) if rule.method_name in ('lift','verb','lower','surface','sum','implies') else cls()
    language=SimpleNamespace(parts_of_speech=PartsOfSpeech(formal_arguments(grammar.rules_upward)),
        _compose_binary_rules=binary, _compose_unary_rules=unary,
        _tree_layer=lambda arity: SimpleNamespace(ops=tuple(map(operation,binary)), unary_ops=tuple(map(operation,unary))))
    store=LTM0(owner.nDim,capacity=64)
    attach_clause_index(SimpleNamespace(languageSpace=language),store,owner)
    return SimpleNamespace(cs=owner,store=store,language=language)


def words(fixture, count):
    # Different admitted units keep their own word identities.
    ids = [fixture.cs.interpret.lookup_word([4+i], (), form=f'token{i}') for i in range(count)]
    bank=fixture.cs.similarity_codebook
    for i, word in enumerate(ids):
        row=fixture.cs._csw_row_of(word)
        bank.mereology.placement[row]=torch.eye(bank.mereology.meaning_pairs)[i % bank.mereology.meaning_pairs]
        bank.mereology.placement_witnesses[row]=1
    return ids


def composition(fixture, left, right, operation):
    owner = fixture.cs
    leaves = tuple(Composition(owner.similarity_codebook.lookup_rows(owner._csw_row_of(word)),
        reference=word,word_id=word,order=0) for word in (left,right))
    node = Composition(leaves[0].point,leaves,operation=operation,formal_name=operation)
    return Clause(ConceptualMeaning.from_description(node.point),composition=node,point=node.point)


def test_fixed_and_declared_root_operations_start_open(monkeypatch):
    from PartsOfSpeech import FIXED_OPERATIONS
    gate_data(monkeypatch)
    f=gate_fixture(monkeypatch)
    gates=f.store.parts.gates
    assert all(gates.open[name] == (name in gates.exempt) for name in gates.names)
    assert 'interpret' in gates.exempt and FIXED_OPERATIONS <= gates.exempt
    assert any(not value for value in gates.open.values())


def test_nonroot_evidence_stays_capped_until_kept_ancestors_are_definite(monkeypatch):
    data=gate_data(monkeypatch)
    f=gate_fixture(monkeypatch)
    gates=f.store.parts.gates
    parent=next(name for name in gates.names if name in gates.exempt)
    child=next(name for name in gates.names if name not in gates.exempt)
    assert gates.cap(named=False,operation=child)==data['closedTrust']
    gates.observe(parent,{child})
    assert gates.measurements(f.cs,None)[child] is None
    assert not gates.advance(child,None)
    assert gates.advance(child,.8,when=7)
    assert gates.cap(named=False,operation=child)==1.
    assert not gates.advance(child,0.,when=8)


def test_open_gates_survive_checkpoint_and_epoch_and_missing_state_is_rejected(monkeypatch):
    from IndependentComponents import _pack_state,_unpack_state
    from Memory import LTM1
    gate_data(monkeypatch)
    f=gate_fixture(monkeypatch);bank=f.store.parts
    child=next(name for name in bank.gates.names if name not in bank.gates.exempt)
    parent=next(name for name in bank.gates.names if name in bank.gates.exempt)
    bank.gates.observe(parent,{child});bank.gates.advance(child,.8,when=7)
    state=copy.deepcopy(bank.state_dict());expected=bank.gates.state()
    bank.begin_epoch();assert bank.gates.state()==expected
    restored=LTM1(bank.capacity)
    restored.configure_operations(f.language.parts_of_speech)
    pointers=tuple(getattr(restored,name).data_ptr() for name in restored.columns)
    restored.load_state_dict(state)
    assert restored.gates.state()==expected
    assert pointers==tuple(getattr(restored,name).data_ptr() for name in restored.columns)
    missing=copy.deepcopy(state);extra=_unpack_state(missing['_extra_state']);extra.pop('gate_open');missing['_extra_state']=_pack_state(extra)
    with pytest.raises(ValueError,match='gate state'):restored.load_state_dict(missing)


def test_readiness_records_each_operation_without_a_lesson_run(monkeypatch):
    from corpus_lessons import readiness_report
    gate_data(monkeypatch);f=gate_fixture(monkeypatch)
    report=readiness_report(f.cs)
    assert set(report['gates'])==set(f.store.parts.gates.names)
    assert report['trust_measurements']==f.cs.trust_measurements()


@pytest.mark.parametrize('field,value',[('closedTrust',float('nan')),('threshold',1.1),('threshold',-.1)])
def test_invalid_gate_data_fails_at_construction(monkeypatch,field,value):
    from Memory import LTM1
    data=gate_data(monkeypatch);data[field]=value
    with pytest.raises(ValueError):LTM1(8)


def test_retired_group_configuration_fails_loudly(monkeypatch):
    from Memory import LTM1
    data=gate_data(monkeypatch);data['interpret']=dict(start='open',fraction=.5)
    with pytest.raises(ValueError,match='retired'):LTM1(8)
