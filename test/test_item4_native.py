"""Native Item 4 carrier certificates with controlled, causal priors."""
import pytest
import torch
from dataclasses import replace
from types import SimpleNamespace
from test_clause_acceptance import SentenceFixture
from reading_fixtures import commit_reading
from Layers import SymbolExpectation
from Models import BasicModel
from SymbolField import SymbolField
from Understanding import Understanding
from What import What
import ThoughtClosing


def case(omitted=False, relation=False):
    with pytest.MonkeyPatch.context() as patch:
        fixture = SentenceFixture(patch)
        entry = fixture.program(('part', 'cat', 'dog') if relation else 'cat')
        entry = replace(entry, leaf_evidence=torch.tensor([[1., 0.]]).expand(len(entry.leaves), -1))
        missing = fixture.program('dog') if omitted else None
        bank = entry.leaves if missing is None else torch.cat((entry.leaves, missing.leaves))
        ids = entry.concept_ids if missing is None else torch.cat((entry.concept_ids, missing.concept_ids))
        if relation:
            clause = fixture.clause(('part', 'cat', 'dog'))
            bank = clause.meaning.roles
            ids = torch.tensor([value if type(value) is int else value.identity for value in clause.refs])
        width = fixture.cs.nDim
        head = SymbolExpectation(8, 8, width, concept_dim=width)
        head.set_symbol_bank(bank, identities=ids, orders=torch.ones(len(ids), dtype=torch.long))
        with torch.no_grad():
            for parameter in head._inter_predictor.parameters():
                parameter.zero_()
            head._inter_predictor.local.bias.copy_(torch.tensor([60., -60.]))
        host = BasicModel()
        host.reconstruct_in_loop = False
        captured = []
        commit = BasicModel._commit_sentence
        def capture(owner, state, sid, active, observations, *args, **kwargs):
            captured.append(owner._sentence_closing_images(observations[0], active))
            return commit(owner, state, sid, active, observations, *args, **kwargs)
        patch.setattr(BasicModel, '_commit_sentence', capture)
        for sid in (0, 1):
            commit_reading(fixture.language, fixture.registry, entry, fixture.store,
                           discourse=head, sid=sid, document='cat-stream', owner=host)
        host.conceptualSpace.stm = SimpleNamespace(concept_dim=width)
        patch.setattr(host, '_what_grammar_context', lambda *a, **kw: (None, ()))
        host.eval()
        field, image = host._sentence_fields[1][0], captured[-1][0]
        understanding = Understanding(sentence_states=(field,))
        if not relation:
            answer = host.resolveAnswer(understanding, What.present(0))
            idea, ready, _, _ = host._materialize_answer_idea(understanding, answer, What.present(0))
            # In the complete grammar the answer is the operands of the
            # executed output(meaning) calls, and a closing with no emission
            # is silent (778e41ce78). No thought was executed here.
            assert 'output' in host.grammatical_thoughts.executable_operation_ids
            assert not answer.selected_thoughts and answer.answer_meanings == ((),)
            assert not idea.any()
            assert ready.all() and answer.resolved
            # Given to the materializer as an emission gives it, the observed
            # cat is the answer idea with no attenuation.
            emitted = replace(answer, conceptual_answer=field.meaning.roles[None],
                              answer_meanings=((field.meaning,),))
            idea, ready, _, _ = host._materialize_answer_idea(understanding, emitted, What.present(0))
            torch.testing.assert_close(idea[0], entry.end_state, atol=0, rtol=0)
            assert ready.all() and answer.resolved
        index = fixture.store.location(field.row_id)
        retained = fixture.store.meaning_of(index)
        floor = 1.
        torch.testing.assert_close(image.conceived, image.observed*floor, atol=1e-7, rtol=1e-6)
        if relation:
            # Two order-zero symbol operands form an untyped chain; the
            # taxonomy reader does not invent a third predicate row.
            expected = torch.stack((fixture.registry._payload(('sym', clause.refs[0])),
                fixture.registry._payload(('sym', clause.refs[2])), bank[0]*0))
            assert retained.role_refs == (
                ('sym', clause.refs[0]), ('sym', clause.refs[2]), None)
        else:
            expected = entry.end_state
        torch.testing.assert_close(retained.roles, expected, atol=0, rtol=0)
        assert image.omission is None and image.expected is None
        assert image.row_expected is None
        return dict(answer_correct=None if relation else True, store_correct=True)



@pytest.mark.parametrize('omitted', [False, True])
def test_observed_cat_survives_native_answer_and_store(omitted):
    case(omitted=omitted)


def test_relation_chain_survives_native_close_and_store():
    case(relation=True)


def test_sentence_prime_is_cleared_before_thought_and_next_peer_input(tmp_path,monkeypatch,eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model, _stage_fullgraph_tensor_peer
    model=_tiny_canonical_model(tmp_path,monkeypatch,input_width=16,word_buckets='8',stm_capacity=3,chooser_depth=1)
    model.eval();model.set_sentence_expectation(True)
    reducer=model._stm_reducer();seen=[];close=ThoughtClosing.close
    def boundary(*args,**kwargs):
        assert getattr(reducer,'_expectation_prime',None) is None
        assert getattr(reducer,'_symbol_expectation_read',None) is None
        seen.append(True)
        return close(*args,**kwargs)
    monkeypatch.setattr(ThoughtClosing,'close',boundary)
    try:
        for words in (['cat','dog'],['dog','cat']):
            assert getattr(reducer,'_expectation_prime',None) is None
            assert getattr(reducer,'_symbol_expectation_read',None) is None
            with torch.no_grad():
                raw=_stage_fullgraph_tensor_peer(model,words)
                with model._sentence_run():
                    result=model._forward_with_compiled_sentence_state(raw)
                model._publish_compiled_sentence_state(result);model._end_step()
            model.End();model.symbolSpace.soft_reset()
        assert len(seen)>=2
        with pytest.raises(RuntimeError,match='interrupted sentence'):
            with model._sentence_run():
                reducer._expectation_prime=torch.ones(1)
                reducer._symbol_expectation_read=object()
                raise RuntimeError('interrupted sentence')
        assert reducer._expectation_prime is None and reducer._symbol_expectation_read is None
    finally:
        model.End();model.symbolSpace.soft_reset()
