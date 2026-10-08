"""Preserved September 17 poison boundaries, on the current native owners.

These are unconditional correctness checks, not arithmetic-learning claims.
The original source is retained in the item-8 receipt.
"""
from dataclasses import replace
import pytest
import torch

from What import What
from test_output_path_supervised import _native_answer_model
from test_output_walk import _capture_program_probe, _stop


def _poison_oracles(monkeypatch):
    import exact
    def forbidden(*args, **kwargs):
        pytest.fail('oracle arithmetic reached from the learner')
    for name in ('eval_expr', 'linearize', 'numeral_code', 'referent_code'):
        # Removed helpers stay trapped if an old call is ever resurrected.
        monkeypatch.setattr(exact, name, forbidden, raising=False)
    for name in ('from_surface', 'lookup', 'evaluate', 'bind', 'substitute',
                 'constrain', 'execute'):
        monkeypatch.setattr(exact.ExactState, name, forbidden, raising=False)
    monkeypatch.setattr(exact.ExactLexer, 'lex', forbidden)


def test_arbitrary_symbol_query_and_native_renaming_do_not_call_arithmetic(monkeypatch):
    from test_normal_thought_controller import _catalog_world
    model, registry, _, part, whole = _catalog_world()
    question = registry.form('isPart', part, whole, bindings={'x': part})
    renamed = replace(question, role_refs=(('sym', 901), ('sym', 337), ('sym', 29)),
                      bindings={'x': ('sym', 901)})
    _poison_oracles(monkeypatch)
    feature = lambda q: model._selected_thought_context(q, level=0, pressure=0)
    torch.testing.assert_close(feature(question), feature(renamed), rtol=0, atol=0)
    with model._query_boundary_scope((0,)):
        forward = model.run_selected_thought(question, work_budget=128)
    assert forward.result.support_true == 1
    model._end_finished_selected_thought_episodes()
    with model._query_boundary_scope((0,)):
        reverse = model.run_selected_thought(registry.form('isPart', whole, part), work_budget=128)
    assert reverse.result.support_true == 0
    model._end_finished_selected_thought_episodes()



def test_renamed_native_vocabulary_preserves_checked_relation_answers(monkeypatch):
    from types import SimpleNamespace
    from Language import Grammar
    from Queries import (GrammaticalThoughtRegistry, ThoughtGrammarContext,
                         ThoughtConceptualCapability, ThoughtTaxonomyCapability)
    from QueryWork import QueryWorkBudget
    from test_cs_symbol_table import _cs
    def world(prefix, names):
        cs = _cs()
        for _ in range(prefix):
            cs.new_concept()
        grammar = Grammar()
        grammar.load_from_grammar_file('complete.grammar')
        registry = GrammaticalThoughtRegistry.install(cs, grammar)
        base = cs.new_concept()
        part = ('sym', cs.synthesize_higher_order([('sym', base)]))
        whole = ('sym', cs.new_concept())
        for ref, name in zip((part, whole), names):
            cs._csw_concept_row(0, ref[1])
            cs.bind_word_concept(name.encode(), ref[1])
        cs.add_whole(part[1], whole)
        context = ThoughtGrammarContext(word_stream=(),
            conceptual_space=ThoughtConceptualCapability(
                cs, lambda left, right: float(torch.allclose(left, right))),
            primed_symbols=(), ltm=__import__('Queries').ThoughtLTMCapability(store=lambda:None,equal=lambda a,b:0.,tau_id=.6), taxonomy=ThoughtTaxonomyCapability(cs),
            work=QueryWorkBudget(128), continuation=None, boundary=lambda _row: None)
        return cs, registry, part, whole, context
    first, second = world(0, ('one', 'two')), world(11, ('cedar', 'birch'))
    x = first[1].form('isPart', first[2], first[3])
    y = second[1].form('isPart', second[2], second[3])
    assert all(x.role_refs[slot] != y.role_refs[slot] for slot in (0, 2))
    assert first[1].signature_for(x).operation.semantic_id == second[1].signature_for(y).operation.semantic_id == 'isPart'
    with torch.no_grad():
        for slot in (0, 1, 2):
            value, ref = x.roles[slot], y.role_refs[slot]
            row = second[0]._csw_concept_row(0, ref[1])
            second[0].similarity_codebook.getW()[row].copy_(value)
    y = second[1].form('isPart', second[2], second[3])
    torch.testing.assert_close(x.roles, y.roles)
    _poison_oracles(monkeypatch)
    a, b = first[1].execute(x, first[4]), second[1].execute(y, second[4])
    assert a.support_true == b.support_true == 1
    assert a.support_false == b.support_false == 0
    reverse = second[1].execute(second[1].form('isPart', second[3], second[2]), second[4])
    assert reverse.support_true == 0



@pytest.mark.usefixtures('eager_reading')
@pytest.mark.parametrize('texts', [('1 plus 2', '3 plus 4'),
                                  ('cedar joins birch', 'maple joins oak')])
def test_numeric_and_renamed_sentence_paths_keep_oracles_out(tmp_path, monkeypatch, texts):
    model = _native_answer_model(tmp_path, True)
    _poison_oracles(monkeypatch)
    model.eval()
    _stop(model)
    try:
        with torch.no_grad():
            understood = _capture_program_probe(model, list(texts))
            reconstructed = model.reverseReconstruct(understood)
            resolved = model.resolveAnswer(understood, (What.supervised(0), What.supervised(1)))
            answer = model.reverseOutput(understood, resolved)
        assert all(p is not None for p in understood.sentence_states)
        assert bool(torch.isfinite(answer.actual).all())
        assert not torch.equal(answer.concepts[0], answer.concepts[1])
        assert reconstructed is not None
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_supervised_update_cannot_use_exact_arithmetic_or_fallback_codes(tmp_path, monkeypatch, eager_reading):
    model = _native_answer_model(tmp_path, True)
    _poison_oracles(monkeypatch)
    optimizer = model.getOptimizer(lr=1e-3)
    steps = []
    real_step = optimizer.step
    def step(*args, **kwargs):
        steps.append(getattr(model, '_sentence_trial', None) or 'batch')
        return real_step(*args, **kwargs)
    monkeypatch.setattr(optimizer, 'step', step)
    try:
        batch = (model.inputSpace.prepInput(['1 plus 2', '3 plus 4']),
                 torch.zeros(2, 1, 1))
        result, _ = model.runBatch(train=True, batchSize=2, split='train',
            optimizer=optimizer, batch_override=batch,
            questions=(What.supervised(0), What.supervised(1)))
        assert steps == ['exploit', 'explore']
        assert model._sentence_reader_updates == 1
        assert bool(torch.isfinite(result.lossOut))
        assert model._last_answer_mask.tolist() == [True, True]
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
