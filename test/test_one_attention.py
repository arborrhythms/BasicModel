"""6.8's typed brackets, one chooser, level expectation and fixed word whole."""
import pytest
import torch


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


def test_bracket_table_is_bounded_typed_and_shares_one_meter():
    from Attention import BracketTable, SPACE_INPUT, SPACE_SYMBOL
    table=BracketTable.open(torch.tensor([11,9]),budget=4)
    assert table.intervals.shape == (2,4,2)
    assert table.valid.sum(1).tolist() == [1,1]
    assert table.space[:,0].tolist() == [SPACE_INPUT,SPACE_INPUT]
    table=table.split(torch.tensor([0,0]),torch.tensor([5,4]),space=torch.tensor([SPACE_INPUT,SPACE_SYMBOL]))
    assert table.valid.sum(1).tolist() == [2,2]
    assert table.spent.tolist() == [1,1]
    assert table.remaining.tolist() == [3,3]
    assert table.space[1,:2].tolist() == [SPACE_SYMBOL,SPACE_SYMBOL]
    assert table.intervals[0,:2].tolist() == [[0,5],[5,11]]


def test_pooling_keeps_both_poles_and_unknown_neither_exactly_zero():
    from Attention import pooled_read
    poles=torch.tensor([[[[1.,0.],[0.,1.],[0.,0.]]],[[[0.,0.],[0.,0.],[0.,0.]]]])
    spans=torch.tensor([[[0,2],[2,4],[4,5]]])
    reading=pooled_read(poles,spans,torch.tensor([[[0,4],[4,5]]]))
    assert reading.poles.tolist() == [[[[1.,1.],[0.,0.]],[[0.,0.],[0.,0.]]]]
    assert reading.both.tolist() == [[True,False]]
    assert reading.neither.tolist() == [[False,True]]
    assert not reading.pure.any()


def test_pinned_masks_known_word_never_descends_and_wide_never_glosses():
    from Attention import narrowing_mask
    pure=torch.tensor([[True,False,False,True]])
    both=torch.tensor([[False,True,False,False]])
    neither=torch.tensor([[False,False,True,False]])
    word=torch.tensor([[True,False,True,False]])
    mask=narrowing_mask(pure,both,neither,singular=pure,word=word,can_split=~word,has_parts=torch.ones_like(word))
    assert mask.tolist() == [[[False,False,True],[True,True,False],[False,True,False],[False,True,False]]]


def test_normal_field_answer_reaches_every_binding_and_keeps_first_word():
    from test_mm_xor import _fresh_model
    model,_,_= _fresh_model()
    model.train()
    value=model.inputSpace.prepInput(['hello world','hello there','loving world','loving there'])
    _,_,output,_=model.forward(value)
    output.square().sum().backward()
    for space in model.conceptualSpaces:
        parameters=list(space.combine.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in parameters), [(name,p.requires_grad,None if p.grad is None else p.grad.abs().sum().item()) for name,p in space.combine.named_parameters()]
    carrier=model._combine_last_cs_sub.materialize()
    assert (carrier[:,0].norm(dim=-1)>0).all()


def test_narrowing_uses_the_shared_operation_chooser_and_compiles():
    from Language import OperationSelectionLayer
    from Attention import narrow_words
    chooser=OperationSelectionLayer(d_model=4)
    keys=torch.tensor([[[1.,0.,0.,0.],[0.,1.,0.,0.]]])
    spans=torch.tensor([[[0,5],[6,11]]])
    known=torch.tensor([[True,False]])
    def run(keys,known):
        return narrow_words(chooser,keys,spans,known,budget=8)
    value=run(keys,known)
    compiled=torch.compile(run,backend='aot_eager',fullgraph=True)(keys,known)
    torch.testing.assert_close(compiled.table.intervals,value.table.intervals)
    assert value.accepted.tolist()==[[True,True]]
    assert value.descended.tolist()==[[False,True]]
    assert value.table.spent.tolist()==[4]
    assert value.table.remaining.tolist()==[4]
    assert not value.values.requires_grad
    assert value.probabilities.requires_grad
    assert chooser.bracket_anchor.grad is None


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


def test_word_expectation_is_a_trained_distribution_with_one_negative_image():
    from Layers import BracketExpectation
    layer=BracketExpectation(n_symbols=4,max_depth=4,n_dim=4,concept_dim=4,p=2,q=1)
    bank=torch.eye(4,requires_grad=True)[None]
    words=bank[:,[0,1,2]].detach().requires_grad_()
    targets=torch.tensor([[0,1,2]])
    out=layer.expect('word',words,bank,torch.ones(1,4,dtype=torch.bool),targets,teacher_forcing=True)
    assert out.probabilities.shape==(1,3,4)
    torch.testing.assert_close(out.probabilities.sum(-1),torch.ones(1,3))
    torch.testing.assert_close(out.surprise,words.detach()+out.negative_image)
    out.loss.sum().backward()
    assert words.grad is None
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in layer.word_predictor.parameters())
    assert layer.levels == ('byte','word','sentence','row')
    assert layer.enabled_levels == ('word','sentence')


def test_word_expectation_teacher_forcing_only_reads_earlier_targets():
    from Layers import BracketExpectation
    layer=BracketExpectation(n_symbols=4,max_depth=4,n_dim=4,concept_dim=4,p=2,q=1)
    bank=torch.eye(4)[None];valid=torch.ones(1,4,dtype=torch.bool)
    a=layer.expect('word',bank[:,[0,1,2]],bank,valid,torch.tensor([[0,1,2]]),teacher_forcing=True)
    b=layer.expect('word',bank[:,[0,3,2]],bank,valid,torch.tensor([[0,3,2]]),teacher_forcing=True)
    torch.testing.assert_close(a.probabilities[:,:2],b.probabilities[:,:2])
    assert not torch.equal(a.probabilities[:,2],b.probabilities[:,2])
    c=layer.expect('word',bank[:,[0,1,2]],bank,valid,torch.tensor([[0,1,2]]),teacher_forcing=False)
    d=layer.expect('word',bank[:,[0,3,2]],bank,valid,torch.tensor([[0,3,2]]),teacher_forcing=False)
    torch.testing.assert_close(c.probabilities,d.probabilities)


def test_word_forecast_conditions_on_observed_prefix_without_reading_future():
    from Layers import BracketExpectation
    layer=BracketExpectation(n_symbols=4,max_depth=4,n_dim=4,concept_dim=4,p=2,q=1)
    bank=torch.eye(4)[None];valid=torch.ones(1,4,dtype=torch.bool)
    prefix=torch.tensor([[True,True,False]])
    def predict(sequence):
        return layer.expect('word',bank[:,sequence],bank,valid,torch.tensor([sequence]),
                            teacher_forcing=False,observed_prefix=prefix)
    a=predict([0,1,2]);b=predict([0,1,3]);c=predict([1,0,2])
    torch.testing.assert_close(a.probabilities,b.probabilities)
    assert not torch.equal(a.probabilities[:,2],c.probabilities[:,2])


def test_empty_word_bracket_is_a_masked_noop():
    from Attention import narrow_words
    from Language import OperationSelectionLayer
    result=narrow_words(OperationSelectionLayer(d_model=4),torch.empty(2,0,4),
        torch.empty(2,0,2,dtype=torch.long),torch.empty(2,0,dtype=torch.bool),budget=8)
    assert result.values.shape==(2,0,4)
    assert not result.table.valid.any() and not result.table.spent.any()


def test_optional_field_choices_cannot_exhaust_the_word_completion_deadline():
    from types import SimpleNamespace
    from Attention import narrow_words
    def field_first(keys,legal,space,**kwargs):
        priority=torch.arange(legal.shape[-1],device=keys.device).expand_as(legal)
        logits=priority.to(keys).masked_fill(~legal,-torch.inf).flatten(1)
        return logits.argmax(-1),keys.new_ones(len(keys)),legal.flatten(1).sum(-1)>1, dict(probability=keys.new_ones(len(keys)), alternative_count=(legal.flatten(1).sum(-1)-1).clamp_min(0))
    chooser=SimpleNamespace(attention_operations=tuple(range(6)),attend=field_first)
    starts=torch.arange(8)*2
    spans=torch.stack((starts,starts+1),-1)[None]
    result=narrow_words(chooser,torch.eye(8)[None],spans,torch.zeros(1,8,dtype=torch.bool),budget=32)
    assert result.accepted.all()
    assert result.table.spent.item()<=32


def test_normal_model_opens_narrows_and_scores_word_expectation():
    from test_mm_xor import _fresh_model
    model,_,_ = _fresh_model()
    model.train()
    value=model.inputSpace.prepInput(['hello world','hello there','loving world','loving there'])
    model.forward(value)
    table=model._attention_words.table
    assert table.intervals.shape[1] == model.attention_budget
    assert (table.spent <= model.attention_budget).all()
    assert model._open_read is not None and not model._open_read.requires_grad
    expected=model._word_expectation
    assert expected.probabilities.shape[:2] == model._attention_words.accepted.shape
    torch.testing.assert_close(expected.probabilities.sum(-1),torch.ones_like(expected.loss))
    assert model._word_surprise is expected.surprise
    expected.loss.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.symbolSpace.expectation.word_predictor.parameters())


@pytest.mark.parametrize('operation,arity',[('and',1),('or',1),('not',1)])
def test_boolean_field_cannot_be_declared_between_symbols(operation,arity):
    from Language import Grammar
    with pytest.raises(ValueError,match='field'):
        Grammar()._fill_rule_list([],{'rule':{'_':f'S = {operation}.forward(S)','operands':'symbol'}})


def test_bracket_candidates_are_grammar_declared_field_operations_without_generate():
    from Language import Grammar
    g=Grammar();rules=[]
    for name in ('divide','descend','gloss'):
        g._fill_rule_list(rules,{'rule':{'_':f'X = {name}.forward(X)','operands':'field'}})
        with pytest.raises(ValueError,match='generate'):
            g._fill_rule_list([],{'rule':f'X = {name}.reverse(X)'},face='generate')
    assert [r.operand_kinds for r in rules] == [('field',)]*3


def test_shared_meter_cannot_replenish_between_input_and_thought():
    from QueryWork import QueryWorkBudget
    from Attention import BracketTable
    table=BracketTable.open(torch.tensor([7]),budget=4)
    table=table.accept(torch.tensor([0]))
    work=QueryWorkBudget.from_brackets(table,row=0)
    assert work.spent == 1 and work.remaining == 3
    assert work.consume('thought',3)
    assert not work.consume('more')


def test_unknown_repeated_word_has_one_byte_descent():
    from Attention import narrow_words
    from Language import OperationSelectionLayer
    keys=torch.eye(4)[None,:2]
    out=narrow_words(OperationSelectionLayer(d_model=4),keys,
        torch.tensor([[[0,3],[4,7]]]),torch.tensor([[False,False]]),
        identities=torch.tensor([[9,9]]),budget=8)
    assert out.table.spent.tolist()==[4]
    assert out.accepted.all()


def test_native_attention_keeps_every_complete_letter_run(tmp_path,monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    model=_tiny_canonical_model(tmp_path,monkeypatch,word_buckets='8')
    model._tensor_peer_while_eager=True
    model._chart_compose_per_word=lambda:None
    raw=model.inputSpace.prepInput(['wheels belong to bicycles','bicycles are equal to wheels'])
    model._lex_embed_stem(raw)
    reading=model._attention_words
    assert reading.accepted.sum(-1).tolist() == [4,5], {
        'forms':model._attention_forms[0], 'table':reading.table.intervals,
        'accepted':reading.accepted,'spent':reading.table.spent,
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
    from Layers import BracketExpectation
    owner=BracketExpectation(n_symbols=4,max_depth=4,n_dim=4,concept_dim=4)
    state={name:value.clone() for name,value in owner.state_dict().items()
           if not name.startswith('word_predictor.')}
    restored=BracketExpectation(n_symbols=4,max_depth=4,n_dim=4,concept_dim=4)
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


def test_normal_narrowing_reads_native_poles(monkeypatch):
    from test_mm_xor import _fresh_model
    import ModelAttention
    model, _, _ = _fresh_model()
    model.eval()
    value = model.inputSpace.prepInput(['hello world', 'hello there', 'loving world', 'loving there'])
    with torch.no_grad(): model.forward(value)
    calls=[]
    original=ModelAttention.narrow_words
    def traced(*args, **kwargs):
        calls.append(kwargs.get('poles'))
        return original(*args, **kwargs)
    monkeypatch.setattr(ModelAttention, 'narrow_words', traced)
    with torch.no_grad(): model.forward(value)
    assert calls and all(torch.is_tensor(poles) for poles in calls), 'normal narrowing discards native paired evidence'


def test_prior_thought_reserves_the_input_allowance():
    from test_negative_expectation import _anticipating_model
    model, owner, meaning = _anticipating_model()
    model.train(); model.attention_budget = 4
    model._stage_expectation_queries(training=True)
    # Global expectation opens no speculative thought before a closing.
    assert owner._inter_last_meaning[0] is None
    assert model._pending_attention_meters[0].spent == 0
    assert model._pending_attention_meters[0].remaining == 4


def test_normal_input_stages_greedy_and_defers_comparison_to_sentence(monkeypatch):
 from test_mm_xor import _fresh_model
 import ModelAttention
 model,_,_=_fresh_model(); model.train()
 calls=[]; versions=[]
 original=ModelAttention.narrow_words
 def traced(*args,**kwargs):
  calls.append('explore' if kwargs.get('exploit') is not None else 'greedy')
  versions.append(tuple(p._version for p in model._stm_reducer().parameters()))
  return original(*args,**kwargs)
 monkeypatch.setattr(ModelAttention,'narrow_words',traced)
 value=model.inputSpace.prepInput(['hello world','hello there','loving world','loving there'])
 model._lex_embed_stem(value)
 assert calls==['greedy'], calls
 assert model._attention_read is not None
 assert model._attention_words is model._attention_greedy
 assert not hasattr(model, '_attention_score_term')
 assert not hasattr(model, '_last_attention_comparison')


def test_field_operation_changes_the_next_mask_and_children_read_native_poles():
    from Attention import narrow_words
    from types import SimpleNamespace
    masks=[]
    def choose(keys,legal,space,**kwargs):
        masks.append(legal.clone())
        index = torch.tensor([4]) if len(masks)==1 else legal.flatten(1).long().argmax(-1)
        return index, keys.new_ones(1), legal.flatten(1).sum(-1)>1, dict(probability=keys.new_ones(1), alternative_count=(legal.flatten(1).sum(-1)-1).clamp_min(0))
    chooser=SimpleNamespace(attention_operations=tuple(range(6)),attend=choose)
    result=narrow_words(chooser,torch.eye(2)[None],torch.tensor([[[0,1],[2,3]]]),
        torch.ones(1,2,dtype=torch.bool),poles=torch.tensor([[[1.,0.],[0.,1.]]]),budget=6)
    assert masks[0][0,0,0]  # native both permits divide
    assert not masks[1][0,0,0]  # or leaves a pure parent
    assert result.accepted.all()
    torch.testing.assert_close(result.values,torch.tensor([[[1.,0.],[0.,1.]]]))


def test_narrowing_selection_has_no_percept_stage_scorer():
    import ModelAttention, WalkTrials
    assert not hasattr(ModelAttention, 'percept_reconstruction_score')
    assert not hasattr(WalkTrials, 'narrowing_pair')
    assert not hasattr(WalkTrials, 'attention_score_function')


def test_divide_isolates_impure_first_word_and_reaches_pure_neighbors():
    from Attention import narrow_words
    from Language import OperationSelectionLayer
    result = narrow_words(OperationSelectionLayer(d_model=3), torch.eye(3)[None],
        torch.tensor([[[0, 2], [3, 5], [6, 8]]]), torch.ones(1, 3, dtype=torch.bool),
        poles=torch.tensor([[[1., 1.], [1., 0.], [1., 0.]]]), budget=16)
    assert result.accepted.tolist() == [[False, True, True]]
    assert not result.descended.any()
    assert result.table.spent.item() <= 16


def test_native_pole_read_omits_padding_without_changing_evidence(monkeypatch):
    from test_mm_xor import _fresh_model
    from ModelAttention import native_word_poles
    model, _, _ = _fresh_model()
    model.eval()
    with torch.no_grad():
        model._lex_embed_stem(model.inputSpace.prepInput(['hello world']))
        forms, _, _, live = model._attention_forms
        spans = model._attention_spans
        known = torch.tensor([[bool(form) and model._concept_owner().definitions.word(form=form)
                               is not None for form in row] for row in forms]) & live
        assert known.any()
        width = int(torch.where(known, torch.arange(known.shape[1])[None]+1, 0).max())
        reference = native_word_poles(model, spans[:, :width],
                                     [row[:width] for row in forms], known[:, :width])
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
    from Layers import BracketExpectation
    owner = BracketExpectation(n_symbols=4, max_depth=4, n_dim=4, concept_dim=4, p=2, q=1)
    bank = torch.eye(4)[None].expand(2, -1, -1)
    words = torch.nn.functional.pad(bank[:, :2], (0, 0, 0, 30))
    active = torch.zeros(2, 32, dtype=torch.bool)
    active[:, :2] = True
    active[1, 0] = False
    targets = torch.full((2, 32), -1, dtype=torch.long)
    targets[:, :2] = torch.tensor([0, 1])
    args = ('word', words, bank, torch.ones(2, 4, dtype=torch.bool), targets)
    reference = owner.expect(*args, teacher_forcing=False, active=active)
    calls = []
    hook = owner.word_predictor.register_forward_hook(lambda *args: calls.append(True))
    try:
        with torch.no_grad():
            actual = owner.expect(*args, teacher_forcing=False, active=active)
    finally:
        hook.remove()
    for before, after in zip(reference, actual):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert len(calls) == 3, 'inactive trailing columns recomputed an unchanged forecast'


def test_input_priming_reads_live_word_extent_without_changing_prior(monkeypatch):
    from test_mm_xor import _fresh_model
    from Attention import BracketKeys
    model, _, _ = _fresh_model()
    model.eval()
    original = BracketKeys._codebook_retrieval_prior
    calls = []
    def bounded(keys, *args):
        calls.append(keys.shape[1])
        assert keys.shape[1] == 2, 'padding expanded the priming lookup'
        actual = original(keys, *args)
        reference = original(torch.nn.functional.pad(keys, (0, 0, 0, 7)), *args)
        torch.testing.assert_close(actual, reference[:, :2], rtol=2e-6, atol=2e-7)  # GEMM reduction width rounding
        assert not reference[:, 2:].count_nonzero()
        return actual
    monkeypatch.setattr(BracketKeys, '_codebook_retrieval_prior', staticmethod(bounded))
    with torch.no_grad():
        model._lex_embed_stem(model.inputSpace.prepInput(['hello world', 'hello']))
    assert calls == [2]


def test_normal_narrowing_omits_padding_and_preserves_the_complete_walk(monkeypatch):
    from test_mm_xor import _fresh_model
    import ModelAttention
    model, _, _ = _fresh_model()
    model.eval()
    original = ModelAttention.narrow_words
    calls = []
    def bounded(chooser, keys, spans, known, **kwargs):
        calls.append(keys.shape[1])
        assert keys.shape[1] == 2, 'padding multiplied the bracket work'
        actual = original(chooser, keys, spans, known, **kwargs)
        padded = dict(kwargs)
        for name in ('identities', 'prior', 'poles'):
            if padded.get(name) is not None:
                padded[name] = torch.nn.functional.pad(padded[name],
                    (0, 0, 0, 7) if name == 'poles' else (0, 7),
                    value=-1 if name == 'identities' else 0)
        reference = original(chooser,
            torch.nn.functional.pad(keys, (0, 0, 0, 7)),
            torch.nn.functional.pad(spans, (0, 0, 0, 7)),
            torch.nn.functional.pad(known, (0, 7)), **padded)
        for before, after in zip(reference.table, actual.table):
            torch.testing.assert_close(after, before)
        torch.testing.assert_close(actual.actions, reference.actions)
        torch.testing.assert_close(actual.values, reference.values[:, :2])
        torch.testing.assert_close(actual.accepted, reference.accepted[:, :2])
        return actual
    monkeypatch.setattr(ModelAttention, 'narrow_words', bounded)
    with torch.no_grad():
        model._lex_embed_stem(model.inputSpace.prepInput(['hello world', 'hello']))
    assert calls == [2]
    assert model._attention_words.values.shape[:2] == model.inputSpace._word_active_mask.shape
