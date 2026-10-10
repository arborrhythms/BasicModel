"""Curriculum supervision stays at its declared reader and inverse boundaries."""
from types import SimpleNamespace
import torch
from AttentionLesson import reading_lesson
from AttentionTraversal import FieldTraversal
from DerivationReconstruction import reconstruct


def test_lesson_cross_entropy_teaches_named_support_without_a_choice_derivative():
    spans = torch.tensor([[[0., 1.], [1., 2.]]])
    lanes = torch.ones(1, 2, 1, 2, requires_grad=True)
    field = FieldTraversal(lanes, spans, spans, torch.ones(1, 2, dtype=torch.bool), torch.tensor([2]))
    logits = torch.tensor([[0., 4.]], requires_grad=True)
    taught = torch.tensor([[True, False]])
    field.select(logits, taught=taught)
    assert field.reads[0]['action'].tolist() == [0]
    assert not field.reads[0]['admitted'].requires_grad
    field.reads[0]['lesson_cross_entropy'].sum().backward()
    assert logits.grad[0, 0] < 0 < logits.grad[0, 1] and lanes.grad is None
    free = FieldTraversal(lanes, spans, spans, torch.ones(1, 2, dtype=torch.bool), torch.tensor([2]))
    free.select(logits)
    assert free.reads[0]['action'].tolist() == [1]


def test_reading_lesson_resolves_named_occurrences_and_restores_after_exception():
    model = SimpleNamespace(_attention_forms=([['red', 'blue', 'red']], None))
    field = SimpleNamespace(model=model,remaining=torch.ones(1, 3,dtype=torch.bool),
        valid=torch.ones(1, 3,dtype=torch.bool),positions=torch.arange(3),
        active=torch.tensor([True]),iterations=torch.tensor([0]))
    prior = object(); model._attention_reading_lesson = prior
    try:
        with reading_lesson(model, [[('red', 1), 'blue', ('red', 0)]]) as lesson:
            assert lesson.target(field).tolist() == [[False, False, True]]
            raise RuntimeError('end lesson')
    except RuntimeError:
        pass
    assert model._attention_reading_lesson is prior


def test_derivation_inverse_recovers_composed_children_and_depends_on_the_root():
    # A four-leaf binary tree with a real invertible arithmetic kernel. Its
    # co-operands include compounds, without adding them to a word dictionary.
    class Language:
        _compose_binary_rules = [SimpleNamespace(relation_kind=None)]
        def _tree_layer(self, arity):
            return SimpleNamespace(ops=[SimpleNamespace(inverse_kind='residual')])
        def reverse_binary_step(self, parent, op, valid, reference, **kwargs):
            return 2*parent-reference, reference, ~valid
    leaves = torch.tensor([[[1., 0.], [0., 2.], [3., 0.], [0., 4.]]])
    first, second = leaves[:, :2].mean(1), leaves[:, 2:].mean(1)
    root = ((first+second)/2).detach().requires_grad_()
    actions = torch.tensor([[0,-1,0],[0,-1,1],[1,0,-1],[0,-1,2],[0,-1,3],[1,0,-1],[1,0,-1]])
    frames = torch.zeros(7,3,2)
    frames[2] = torch.stack((leaves[0,0],leaves[0,1],first[0]))
    frames[5] = torch.stack((leaves[0,2],leaves[0,3],second[0]))
    frames[6] = torch.stack((first[0],second[0],root.detach()[0]))
    record = SimpleNamespace(word_values=leaves,word_valid=torch.ones(1,4,dtype=torch.bool),
        end_depth=torch.tensor([1]),end_slots=root[:,None],primed=SimpleNamespace(case_bank=None))
    program = SimpleNamespace(actions=actions,operation_values=frames)
    recovered,bad,defined = reconstruct(Language(),record,[program],lambda b:torch.arange(4))
    torch.testing.assert_close(recovered,leaves)
    assert not bad.any()
    assert defined.all()
    recovered[:,0].sum().backward()
    torch.testing.assert_close(root.grad,torch.full_like(root,4.))
    record.end_slots = (root.detach()+1)[:,None]
    changed,_,_ = reconstruct(Language(),record,[program],lambda b:torch.arange(4))
    assert not torch.equal(changed,leaves)


def test_native_lesson_teaches_scorer_and_keeps_one_word_per_read(tmp_path,monkeypatch):
    from test_native_attention_field import _model
    model = _model(tmp_path,monkeypatch)
    try:
        before = model.candidate_attention.readout.weight.detach().clone()
        inputs = model.inputSpace.prepInput(['red blue','blue red'])
        optimizer = model.getOptimizer(lr=.01)
        attention = next(o for o in optimizer.optimizers if getattr(o,'objective_owner',None)=='attention')
        assert isinstance(attention.inner,torch.optim.Adam)
        assert all(group['lr']==.01 for group in attention.param_groups)
        with reading_lesson(model,[['red','blue'],['blue','red']]):
            model.runBatch(train=True,optimizer=optimizer,batchSize=2,batch_override=(inputs,torch.empty(2,0)))
        field = model._last_sentence_field
        assert field['iterations'].tolist() == [2,2]
        assert field['trials'][0]['lesson_cross_entropy'].gt(0).all()
        assert all(read['admitted'].sum(-1).eq(1).all() for read in field['trials'][0]['reads'])
        for trial in field['trials']:
            for read in trial['reads']:
                live=read['admitted'].any(-1)
                undefined=live[:,None] & ~read['defined_coordinates']
                torch.testing.assert_close(read['unrecovered_coordinates'],undefined.sum(-1))
                torch.testing.assert_close(read['unrecovered_words'],undefined.any(-1).long())
        assert not torch.equal(model.candidate_attention.readout.weight,before)
        owners = model.ownership_gradient_diagnostics(optimizer)['parameters']
        assert all(row['writers']==['attention'] for row in owners if row['owner']=='attention')
        assert model._attention_reading_lesson is None
    finally:model.End()


def test_lossy_derivation_has_no_inverse_or_search_and_reports_every_coordinate():
    class Language:
        _compose_binary_rules = [SimpleNamespace(relation_kind=None)]
        def _tree_layer(self, arity):
            return SimpleNamespace(ops=[SimpleNamespace(inverse_kind='search')])
        def reverse_binary_step(self, *args, **kwargs):
            raise AssertionError('the derivation must not search for missing operands')
    words = torch.tensor([[[1., 2.], [3., 4.]]])
    record = SimpleNamespace(word_values=words, word_valid=torch.ones(1,2,dtype=torch.bool),
        end_depth=torch.tensor([1]), end_slots=torch.ones(1,1,2))
    program = SimpleNamespace(actions=torch.tensor([[0,-1,0],[0,-1,1],[1,0,-1]]),
        operation_values=torch.stack((torch.zeros(3,2),torch.zeros(3,2),
            torch.stack((words[0,0],words[0,1],torch.ones(2))))))
    recovered, missing, defined = reconstruct(Language(),record,[program],lambda b:torch.arange(2))
    assert missing.all() and not defined.any() and not recovered.any()


def test_partial_product_inverse_keeps_defined_coordinates_and_root_gradient():
    class Language:
        _compose_binary_rules = [SimpleNamespace(relation_kind=None)]
        def _tree_layer(self, arity):
            return SimpleNamespace(ops=[SimpleNamespace(inverse_kind='product')])
        def reverse_binary_step(self, parent, op, valid, reference, **kwargs):
            # The existing native dispatcher reports this row unavailable,
            # although division still recovers the nonzero coordinates.
            good = reference.abs() > 1e-8
            return torch.where(good,parent/torch.where(good,reference,1.),0.), reference, ~good.all(-1)
    words = torch.tensor([[[2., 7.], [3., 0.]]])
    root = torch.tensor([[[6., 0.]]],requires_grad=True)
    record = SimpleNamespace(word_values=words,word_valid=torch.ones(1,2,dtype=torch.bool),
        end_depth=torch.tensor([1]),end_slots=root,primed=SimpleNamespace(case_bank=None))
    frames=torch.zeros(3,3,2);frames[2,:2]=words[0]
    program=SimpleNamespace(actions=torch.tensor([[0,-1,0],[0,-1,1],[1,0,-1]]),operation_values=frames)
    recovered,missing,defined=reconstruct(Language(),record,[program],lambda b:torch.arange(2))
    assert defined.tolist()==[[[True,False],[True,True]]]
    assert missing.tolist()==[[True,False]]
    recovered[defined].sum().backward()
    torch.testing.assert_close(root.grad,torch.tensor([[[1/3,0.]]]))


def test_native_four_word_inverse_follows_composed_children_without_free_decoder(tmp_path,monkeypatch):
    from test_native_attention_field import _model
    from test_reverse_traversal import _select_completed_binary_path
    from attention_field_gates import words_from_leaves
    model = _model(tmp_path,monkeypatch)
    model.candidate_attention = None
    _select_completed_binary_path(model)
    try:
        texts=['red blue green gold','gold green blue red']
        with torch.no_grad():
            model.runBatch(train=False,batchSize=2,batch_override=(model.inputSpace.prepInput(texts),torch.empty(2,0)))
        trial=model._last_sentence_field['trials'][0]
        record=model._last_sentence_understanding
        recovered=trial['reconstructed'][:,:4]
        result=[' '.join(row) for row in words_from_leaves(recovered,torch.tensor([4,4]),record.primed)]
        assert result==texts
        assert not trial['unavailable'].any()
        torch.testing.assert_close(recovered,record.word_values[:,:4],atol=1e-5,rtol=1e-5)
    finally:model.End()


def test_supplied_text_reader_preserves_shared_work_spent_during_sentence_scoring(tmp_path,monkeypatch):
    from test_native_attention_field import _model
    from attention_curriculum_gates import present
    model = _model(tmp_path,monkeypatch)
    model.candidate_attention = None
    stage = dict(fields=['red','blue'],prompts=['which word ?']*2,targets=['red','blue'])
    try:
        # Scoring the two sentence trials invokes the ordinary output reader
        # before the chosen bracket work is charged. That shared work must
        # remain spent, rather than being subtracted from the bracket charge.
        trained = present(model,stage,train=True,optimizer=model.getOptimizer(lr=.003))
        assert all(row.get('space-read',0) > 0 for row in trained['bracket_meter'])
        evaluated = present(model,stage)
        assert evaluated['answer_readbacks'] is not None
        assert len(evaluated['answer_readbacks']) == 2
        assert all(isinstance(value,str) for value in evaluated['answer_readbacks'])
    finally:model.End()


def test_curriculum_gate_counts_one_word_only_for_active_rows(tmp_path,monkeypatch):
    from test_native_attention_field import _model
    from test_reverse_traversal import _select_completed_binary_path
    from attention_curriculum_gates import present
    model = _model(tmp_path,monkeypatch)
    model.candidate_attention = None
    _select_completed_binary_path(model)
    try:
        result = present(model,dict(fields=['red','blue green']))
        assert result['readbacks'] == ['red','blue green']
        assert result['iterations'] == [1,2]
        assert result['trials'][0]['reads'][-1]['supported_words'] == [0,1]
        assert result['passed']
    finally:model.End()


def test_order_lesson_gate_reports_list_inverse_separately(tmp_path,monkeypatch):
    from test_native_attention_field import _model
    import attention_curriculum_gates as curriculum
    model=_model(tmp_path,monkeypatch)
    model.candidate_attention=None
    # Corrupt only the reported list decoder, not the native reads or cost.
    monkeypatch.setattr(curriculum,'words_from_leaves',
        lambda leaves,counts,bank:[['<unrecovered>']*int(n) for n in counts])
    try:
        result=curriculum.present(model,dict(fields=['red blue','blue red'],lesson=True))
        assert result['orders']==[[0,1],[0,1]]
        assert result['passed'] and result['read_gate_passed']
        assert not result['reconstruction_passed'] and result['exact_readbacks']==0
    finally:model.End()


def test_prompt_need_reaches_existing_input_without_an_answer_or_action_target(tmp_path,monkeypatch):
    from test_native_attention_field import _model
    from AttentionLesson import prompt_need
    from ModelCandidateAttention import _needs
    model = _model(tmp_path,monkeypatch)
    width = model.conceptualSpace.stm.concept_dim
    like = model.conceptualSpace.stm._buffer
    try:
        with prompt_need(model,['the second word','the middle phrase']) as need:
            first = _needs(model,2,width,like)
            torch.testing.assert_close(first[:,:width],need)
            assert not torch.equal(need[0],need[1])
            model.inputSpace.data.text_answers = {'train':['a different answer','another']}
            torch.testing.assert_close(_needs(model,2,width,like),first,rtol=0,atol=0)
            assert getattr(model,'_attention_part_lesson',None) is None
            assert getattr(model,'_attention_reading_lesson',None) is None
        assert model._attention_prompt_need is None
    finally:model.End()
