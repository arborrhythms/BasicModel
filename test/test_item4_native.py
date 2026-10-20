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


def case(row_gain, document_gain, *, omitted=False, relation=False):
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
        head = SymbolExpectation(8, 8, 8, concept_dim=8)
        head.set_symbol_bank(bank, identities=ids, orders=torch.ones(len(ids), dtype=torch.long))
        with torch.no_grad():
            for parameter in head._inter_predictor.parameters():
                parameter.zero_()
            head._inter_predictor.local.bias.copy_(torch.tensor([60., -60.]))
        host = BasicModel()
        host.document_expectation_gain = document_gain
        host.expectation_gain = row_gain
        host.reconstruct_in_loop = False
        captured = []
        close = ThoughtClosing.close
        def capture(*args, **kwargs):
            captured.append(kwargs['images'])
            return close(*args, **kwargs)
        patch.setattr(ThoughtClosing, 'close', capture)
        observation = BasicModel._sentence_observation
        row_prior = ()
        def observe(*args, **kwargs):
            value = observation(*args, **kwargs)
            value['row_expectations'] = row_prior
            return value
        patch.setattr(BasicModel, '_sentence_observation', observe)
        for sid in (0, 1):
            raw = entry.end_state[None]
            state = (raw, torch.ones(1, dtype=torch.long),
                     torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long),
                     torch.cat((ids[:1], ids.new_full((2,), -1)))[None], torch.ones(1, 3))
            field = SymbolField(head, strength=0., epsilon=1e-6, saturation='clip')
            field.anticipate((torch.zeros_like(raw), torch.zeros_like(state[1]), *state[2:]))
            field.arrive(state, torch.tensor([True]), lanes=entry.leaf_evidence[:1][None].expand(1, 3, 2))
            row_prior = field.closing_predictions()
            commit_reading(fixture.language, fixture.registry, entry, fixture.store,
                           discourse=head, sid=sid, document='cat-stream', owner=host)
        host.conceptualSpace.stm = SimpleNamespace(concept_dim=8)
        patch.setattr(host, '_what_grammar_context', lambda *a, **kw: (None, ()))
        host.eval()
        field, image = host._sentence_fields[1][0], captured[-1][0]
        understanding = Understanding(sentence_states=(field,))
        if not relation:
            answer = host.resolveAnswer(understanding, What.present(0))
            idea, ready, _, _ = host._materialize_answer_idea(understanding, answer, What.present(0))
            torch.testing.assert_close(idea[0], entry.end_state, atol=0, rtol=0)
            assert ready.all() and answer.resolved
        index = fixture.store.location(field.row_id)
        retained = fixture.store.expectation_pair(index, gain=document_gain)
        floor = (1-row_gain)*(1-document_gain)
        torch.testing.assert_close(image.conceived, image.observed*floor, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(retained['observation'].roles, clause.meaning.roles if relation else entry.end_state, atol=0, rtol=0)
        if omitted:
            assert image.omission[1, 0] == 1 and image.excess[1].sum() == 0
            assert retained['omission'][1, 0] == 1 and retained['excess'][1].sum() == 0
            torch.testing.assert_close(image.conceived, entry.end_state*floor, atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(retained['content'], entry.end_state*(1-document_gain), atol=1e-7, rtol=1e-6)
        return dict(row_gain=row_gain, document_gain=document_gain, floor=floor,
                    content_norm=float(image.conceived.norm()), raw_norm=float(image.observed.norm()),
                    answer_correct=None if relation else True, store_correct=True, omitted=omitted, relation=relation,
                    excess=image.excess.tolist(), omission=image.omission.tolist(),
                    prime=image.expected.tolist(), row_prime=image.row_expected.tolist())


@pytest.mark.parametrize('row_gain,document_gain,omitted', [
    (0.,0.,False),(.5,0.,False),(1.,0.,False),(0.,.5,False),
    (0.,1.,False),(.5,.5,False),(.5,.5,True)])
def test_expected_cat_and_omission_survive_native_answer_and_store(row_gain,document_gain,omitted):
    case(row_gain,document_gain,omitted=omitted)


@pytest.mark.parametrize('row_gain,document_gain',[(1.,0.),(0.,1.),(.5,.5)])
def test_three_role_relation_uses_symbol_confidence_through_native_close_and_store(row_gain,document_gain):
    case(row_gain,document_gain,relation=True)


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
