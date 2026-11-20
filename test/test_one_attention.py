"""One reading by looks, level expectation and fixed word whole."""
import pytest
import torch
from chooser_steering import stand_in


def _field_model():
    from reading_fixtures import field_text_config
    from test_mm_xor import _fresh_model
    with field_text_config() as config:
        return _fresh_model(config)


def test_word_extent_is_letters_and_numeric_percepts_remain_separate():
    import Meronomy
    raw=b'abc123 cat-dog 01 A1B'
    assert [raw[a:b] for a,b in Meronomy.word_spans(raw)] == [b'abc',b'cat',b'dog',b'A',b'B']
    assert [raw[a:b] for a,b in Meronomy.number_spans(raw)] == [b'123',b'01',b'1']
    assert [raw[a:b] for a,b in Meronomy.percept_spans(raw)] == [b'abc',b'123',b'cat',b'-',b'dog',b'01',b'A',b'1',b'B']


def test_word_whole_membership_is_fixed_while_other_properties_can_learn():
    from PerceptProperties import PrimitiveProperties
    props=PrimitiveProperties(12)
    props.define_word(8)
    before=props.coefficients().detach().clone()
    assert before[8,ord('A')] == 1 and before[8,ord('0')] == 0
    optimizer=torch.optim.SGD(props.parameters(),lr=.5)
    (-props.coefficients().sum()).backward();optimizer.step()
    torch.testing.assert_close(props.coefficients()[8],before[8])
    assert not torch.equal(props.coefficients()[0],before[0])
    assert not props.coefficients()[8].requires_grad or props.members.grad[8].count_nonzero() == 0


@pytest.mark.parametrize('name',['subsymbolicOrder','symbolicOrder','subsymbolicLoop','serial','modeSchedule','readingAttention','globalAttention','globalAttentionConsume','selectedThoughtBudget'])
def test_old_attention_controls_are_rejected_at_ingestion(name):
    from util import XMLConfig
    with pytest.raises(ValueError,match='retired'):
        XMLConfig._apply_legacy_renames({'architecture':{name:1}},'probe.xml')


def test_attention_modules_retire_after_the_answer_reader_takes_their_contract():
    import Spaces
    from Attention import PrimedSymbolReader
    assert not hasattr(Spaces,'ReadingAttention') and not hasattr(Spaces,'GlobalAttention')
    reader=PrimedSymbolReader()
    keys=torch.tensor([[[1.,0.],[0.,1.]]],requires_grad=True)
    result=reader(concept_q=torch.ones(1,2),symbol_q=None,
                  spaces=[{'id':0,'keys':keys}])
    base=torch.ones(1,2)
    torch.testing.assert_close(reader.consume(base,result['content']),base)
    reader.consume_gate.data.fill_(.5)
    reader.consume(base,result['content']).square().sum().backward()
    assert keys.grad is None
    assert any(p.grad is not None for p in reader.scorer.parameters())


def test_word_expectation_is_a_trained_distribution_with_one_gain():
    from Layers import SymbolExpectation
    layer=SymbolExpectation(n_symbols=4,max_depth=4,n_dim=4,concept_dim=4)
    layer.set_symbol_bank(torch.eye(layer.concept_dim))
    bank=torch.eye(4,requires_grad=True)[None]
    words=bank[:,[0,1,2]].detach().requires_grad_()
    targets=torch.tensor([[0,1,2]])
    out=layer.expect('word',words,bank,torch.ones(1,4,dtype=torch.bool),targets)
    assert out.probabilities.shape==(1,3,4)
    torch.testing.assert_close(out.probabilities.sum(-1),torch.ones(1,3))
    torch.testing.assert_close(out.content, words.detach()*out.gain[...,None])
    torch.testing.assert_close(out.surprise, out.excess+out.omission)
    out.loss.sum().backward()
    assert words.grad is None
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in layer.word_predictor.parameters())
    assert layer.levels == ('word','field')
    assert layer.enabled_levels == ('word','field')


def test_word_expectation_only_reads_earlier_observations():
    from Layers import SymbolExpectation
    layer=SymbolExpectation(n_symbols=4,max_depth=4,n_dim=4,concept_dim=4)
    layer.set_symbol_bank(torch.eye(layer.concept_dim))
    bank=torch.eye(4)[None];valid=torch.ones(1,4,dtype=torch.bool)
    a=layer.expect('word',bank[:,[0,1,2]],bank,valid,torch.tensor([[0,1,2]]))
    b=layer.expect('word',bank[:,[0,3,2]],bank,valid,torch.tensor([[0,3,2]]))
    torch.testing.assert_close(a.probabilities[:,:2],b.probabilities[:,:2])
    assert not torch.equal(a.probabilities[:,2],b.probabilities[:,2])
    c=layer.expect('word',bank[:,[0,1,2]],bank,valid,torch.tensor([[0,1,2]]))
    d=layer.expect('word',bank[:,[0,3,2]],bank,valid,torch.tensor([[0,3,2]]))
    torch.testing.assert_close(c.probabilities,a.probabilities)
    torch.testing.assert_close(d.probabilities,b.probabilities)


def test_word_forecast_conditions_on_observed_prefix_without_reading_future():
    from Layers import SymbolExpectation
    layer=SymbolExpectation(n_symbols=4,max_depth=4,n_dim=4,concept_dim=4)
    layer.set_symbol_bank(torch.eye(layer.concept_dim))
    bank=torch.eye(4)[None];valid=torch.ones(1,4,dtype=torch.bool)
    prefix=torch.tensor([[True,True,False]])
    def predict(sequence):
        return layer.expect('word',bank[:,sequence],bank,valid,torch.tensor([sequence]),
                            observed_prefix=prefix)
    a=predict([0,1,2]);b=predict([0,1,3]);c=predict([1,0,2])
    torch.testing.assert_close(a.probabilities,b.probabilities)
    assert not torch.equal(a.probabilities[:,2],c.probabilities[:,2])


def test_normal_model_opens_reads_and_scores_word_expectation(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    from reading_fixtures import read_in_source_order
    model = _tiny_canonical_model(tmp_path,monkeypatch,batch_size=4)
    model._tensor_peer_while_eager=True
    read_in_source_order(model,monkeypatch)
    model.train()
    value=model.inputSpace.prepInput(['hello world','hello there','loving world','loving there'])
    model.forward(value)
    reading=model._attention_words
    assert reading.actions.shape[1] == model.attention_budget
    assert (reading.looks <= model.attention_budget).all()
    assert model._open_read is not None and not model._open_read.requires_grad
    expected=model._word_expectation
    assert expected.probabilities.shape[:2] == model._attention_words.accepted.shape
    active = model._word_expectation_mask
    torch.testing.assert_close(model._word_expectation.surprise, expected.surprise)
    assert active.any()
    torch.testing.assert_close(expected.probabilities.sum(-1),active.to(expected.loss))
    expected.loss.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.symbolSpace.expectation.word_predictor.parameters())


@pytest.mark.parametrize('operation,arity',[('and',1),('or',1),('not',1)])
def test_boolean_field_cannot_be_declared_between_symbols(operation,arity):
    from Language import Grammar
    with pytest.raises(ValueError,match='field'):
        Grammar()._fill_rule_list([],{'rule':{'_':f'S = {operation}.forward(S)','operands':'symbol'}})


def test_unknown_repeated_word_is_read_whole_once_at_each_place():
    """Two occurrences of a word with no word concept: each is returned whole
    by the look, one is taken, the done mark lowers it, and the next look
    takes the other."""
    from types import SimpleNamespace
    from SentenceLooks import word_activity, lower
    from Attention import draw_among
    bank=SimpleNamespace(token=torch.tensor([[0,0,0,1,1,1]]))
    own=torch.ones(1,6,2)
    field=SimpleNamespace(concept_ids=torch.tensor([5]),evidence=torch.zeros(1,1,2,2))
    concepts=torch.full((1,2),-1); known=torch.zeros(1,2,dtype=torch.bool)
    read=[]
    for _ in range(2):
        extent=torch.stack([(own[0,:3,0]>0).any(),(own[0,3:,0]>0).any()])[None]
        words=word_activity(field,concepts,known,extent,own,bank)
        index,drawn,_,_=draw_among(words)
        read.append(int(index)); own=lower(own,bank,index,torch.tensor([True]))
    assert sorted(read) == [0,1]
    assert not own.any()


def test_native_attention_keeps_every_complete_letter_run(tmp_path,monkeypatch,eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    model=_tiny_canonical_model(tmp_path,monkeypatch,word_buckets='8')
    model._tensor_peer_while_eager=True
    model._chart_compose_per_word=lambda:None
    from reading_fixtures import read_in_source_order
    read_in_source_order(model,monkeypatch)
    raw=model.inputSpace.prepInput(['wheels belong to bicycles','bicycles are equal to wheels'])
    with torch.no_grad():
        model.understand(raw)
    reading=model._attention_words
    assert reading.accepted.sum(-1).tolist() == [4,5], {
        'forms':model._attention_forms[0],
        'accepted':reading.accepted,'looks':reading.looks,
        'actions':reading.actions,
        'offsets':getattr(model.inputSpace,'_ar_word_part_offsets',None)}


def test_answer_reader_empty_registry_rows_have_no_content_or_address():
    from Attention import PrimedSymbolReader
    reader=PrimedSymbolReader()
    out=reader(concept_q=torch.ones(2,4),symbol_q=None,
               spaces=[{'id':0,'keys':torch.ones(2,3,4),'valid':torch.zeros(2,3,dtype=torch.bool)}])
    assert not out['alpha'].any() and not out['content'].any()
    assert out['space_id'].tolist()==[-1,-1]


def test_answer_space_choice_uses_shared_chooser_and_bracket_allowance():
    from Attention import PrimedSymbolReader
    from Language import OperationSelectionLayer
    from QueryWork import QueryWorkBudget
    reader=PrimedSymbolReader(); chooser=OperationSelectionLayer(d_model=4)
    calls=[]; original=chooser.select_logits
    def choose(*args,**kwargs):
        calls.append(True);return original(*args,**kwargs)
    chooser.select_logits=choose
    work=(QueryWorkBudget(1),QueryWorkBudget(0))
    out=reader(concept_q=torch.ones(2,4),symbol_q=None,
        spaces=[{'id':5,'keys':torch.eye(4)}],chooser=chooser,work=work)
    assert calls and work[0].spent==1 and work[1].spent==0
    assert out['space_id'].tolist()==[5,-1]
    assert not out['content'][1].any()


def test_small_grammar_model_has_the_same_fixed_word_whole():
    from test_mm_xor import _fresh_model, _PROJECT
    from pathlib import Path
    model,_,_=_fresh_model(str(Path(_PROJECT)/'data/XOR_grammar.xml'))
    whole=model.wholeSpace.subspace.what.primitive_properties
    assert len(whole.fixed_rows)>8 and bool(whole.fixed_rows[8])
    assert whole.coefficients()[8,ord('a')]==1 and whole.coefficients()[8,ord('1')]==0


def test_old_sentence_expectation_checkpoint_keeps_sentence_weights():
    from Layers import SymbolExpectation
    owner=SymbolExpectation(n_symbols=4,max_depth=4,n_dim=4,concept_dim=4)
    owner.set_symbol_bank(torch.eye(owner.concept_dim))
    state={name:value.clone() for name,value in owner.state_dict().items()
           if not name.startswith('word_predictor.')}
    restored=SymbolExpectation(n_symbols=4,max_depth=4,n_dim=4,concept_dim=4)
    restored.set_symbol_bank(torch.eye(restored.concept_dim))
    restored.load_state_dict(state,strict=True)
    for name,value in state.items():torch.testing.assert_close(restored.state_dict()[name],value,rtol=0,atol=0)


def test_fixed_word_storage_rejects_conflicting_teaching():
    from PerceptProperties import PrimitiveProperties
    properties=PrimitiveProperties(9)
    properties.define_word(8)
    before={k:v.clone() for k,v in properties.state_dict().items()}
    with pytest.raises(ValueError,match='fixed'):
        properties.teach(8,[65],[0.])
    for key,value in before.items():
        torch.testing.assert_close(properties.state_dict()[key],value,rtol=0,atol=0)


def test_fixed_word_storage_survives_projection():
    from PerceptProperties import PrimitiveProperties
    properties=PrimitiveProperties(9)
    properties.define_word(8)
    before=properties.members[8].detach().clone()
    with torch.no_grad():properties.members[8].add_(.25)
    properties.project()
    torch.testing.assert_close(properties.members[8],before,rtol=0,atol=0)


def test_prior_thought_reserves_the_input_allowance():
    from test_negative_expectation import _anticipating_model
    model, owner, meaning = _anticipating_model()
    model.train(); model.attention_budget = 4
    model._stage_attention_meters(training=True)
    # Global expectation opens no speculative thought before a closing.
    assert not hasattr(owner, '_inter_last_meaning')
    assert model._pending_attention_meters[0].spent == 0
    assert model._pending_attention_meters[0].remaining == 4


def test_normal_input_stages_and_defers_its_reading_to_the_sentence():
 from test_mm_xor import _fresh_model
 model,_,_=_fresh_model('data/MM_text.xml'); model.train()
 value=model.inputSpace.prepInput(['hello world','hello there','loving world','loving there'])
 model._lex_embed_stem(value)
 assert model._attention_words is model._attention_greedy
 # With no sentence no look is taken: no word is admitted.
 assert not model._attention_words.accepted.any()
 assert not model._attention_words.looks.any()
 assert not hasattr(model, '_attention_score_term')
 assert not hasattr(model, '_last_attention_comparison')


def test_narrowing_selection_has_no_percept_stage_scorer():
    import ModelAttention, WalkTrials
    assert not hasattr(ModelAttention, 'percept_reconstruction_score')
    assert not hasattr(WalkTrials, 'narrowing_pair')
    assert not hasattr(WalkTrials, 'attention_score_function')


def test_native_pole_read_omits_padding_without_changing_evidence(monkeypatch):
    from test_mm_xor import _fresh_model
    from ModelAttention import native_word_poles
    from nomination_fixtures import nominated_forms
    model, _, _ = _field_model()
    model.eval()
    # The forms are words, nominated by MTM before this reading.
    nominated_forms(model, ['hello world'])
    with torch.no_grad():
        model._lex_embed_stem(model.inputSpace.prepInput(['hello world']))
        forms, _, _, live = model._attention_forms
        spans = model._attention_spans
        known = torch.tensor([[bool(form) and model._concept_owner().interpret.word_for_form(form)
                               is not None for form in row] for row in forms]) & live
        assert known.any()
        width = int(torch.where(known, torch.arange(known.shape[1])[None]+1, 0).max())
        reference = native_word_poles(model, spans[:, :width],
                                     [row[:width] for row in forms], known[:, :width])
        assert reference[known[:, :width]].any(), 'admitted forms retain native identification evidence'
        # A word read has only its blank object (rule 20), no learned one.
        owner = model._concept_owner()
        assert all('blank' in owner._concept_allocator.symbol_types[obj]
                   for obj in owner.definitions.object_ids)
        fi = model.perceptualSpace._forward_input
        native_width = fi['native_indices'].shape[1]
        fi['native_indices'] = torch.nn.functional.pad(fi['native_indices'], (0, 19), value=-1)
        fi['native_part_spans'] = torch.nn.functional.pad(fi['native_part_spans'], (0, 0, 0, 19))
        read = model._concept_owner().cs_read_memberships
        def bounded(percepts, extents, **kwargs):
            assert extents.shape[1] == width, 'padded words expanded the native field read'
            assert percepts[0].shape[1] == native_width, 'padded events expanded the native field read'
            return read(percepts, extents, **kwargs)
        monkeypatch.setattr(model._concept_owner(), 'cs_read_memberships', bounded)
        result = native_word_poles(model, spans, forms, known)
    torch.testing.assert_close(result[:, :width], reference, rtol=0, atol=0)
    assert not result[:, width:].count_nonzero()


def test_inference_reuses_constant_padding_forecast_without_changing_results():
    from Layers import SymbolExpectation
    owner = SymbolExpectation(n_symbols=4, max_depth=4, n_dim=4, concept_dim=4,  )
    owner.set_symbol_bank(torch.eye(owner.concept_dim))
    bank = torch.eye(4)[None].expand(2, -1, -1)
    words = torch.nn.functional.pad(bank[:, :2], (0, 0, 0, 30))
    active = torch.zeros(2, 32, dtype=torch.bool)
    active[:, :2] = True
    active[1, 0] = False
    targets = torch.full((2, 32), -1, dtype=torch.long)
    targets[:, :2] = torch.tensor([0, 1])
    args = ('word', words, bank, torch.ones(2, 4, dtype=torch.bool), targets)
    reference = owner.expect(*args,  active=active)
    calls = []
    hook = owner.word_predictor.register_forward_hook(lambda *args: calls.append(True))
    try:
        with torch.no_grad():
            actual = owner.expect(*args,  active=active)
    finally:
        hook.remove()
    for before, after in zip(vars(reference).values(), vars(actual).values()):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert len(calls) == 3, 'inactive trailing columns recomputed an unchanged forecast'


def test_glance_forecast_reads_only_live_word_extents(tmp_path, monkeypatch):
    from test_item18_expect import lesson, read
    from LookForecast import LookForecast
    model = lesson(tmp_path, monkeypatch)
    try:
        read(model, 'hello world')
        opened = model._sentence_look_fields[0][0][0]['field']
        owner = model._concept_owner()
        from unittest.mock import patch
        import ConceptSpread
        calls = []
        expected = ConceptSpread.expected
        def measure(current, knowing):
            calls.append(len(knowing))
            return expected(current, knowing)
        with patch.object(ConceptSpread, 'expected', measure):
            short = LookForecast(owner, opened, opened.where)
            padded = torch.nn.functional.pad(opened.where, (0, 0, 0, 7))
            long = LookForecast(owner, opened, padded)
        # One forecast for each stream's look, whatever padding follows.
        assert calls == [1, 1]
        torch.testing.assert_close(short.latest, long.latest, rtol=0, atol=0)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_a_read_with_no_sentence_holds_every_word_place_and_reads_none():
    from test_mm_xor import _fresh_model
    model, _, _ = _fresh_model('data/MM_text.xml')
    model.eval()
    with torch.no_grad():
        model._lex_embed_stem(model.inputSpace.prepInput(['hello world', 'hello']))
    assert model._attention_words.values.shape[:2] == model.inputSpace._word_active_mask.shape
    assert not model._attention_words.accepted.any()

