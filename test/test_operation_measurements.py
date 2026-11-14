"""Operation measurements remain available to the lesson harness."""
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


def operation_fixture(monkeypatch):
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








def test_readiness_records_each_operation_without_a_lesson_run(monkeypatch):
    from corpus_lessons import readiness_report
    f=operation_fixture(monkeypatch)
    report=readiness_report(f.cs)
    assert report['trust_measurements']==f.cs.trust_measurements()




