"""Curriculum supervision stays at its declared reader and inverse boundaries."""
from types import SimpleNamespace
import pytest
import torch
from AttentionLesson import output_lesson, supplied_reading
from DerivationReconstruction import reconstruct


def test_the_output_cost_teaches_the_word_said():
    """Ported (eleventh pass) from the reading lesson's cross entropy: the
    output's cost (``SentenceLooks.Output``) books the word said against the
    output's word, and is the mean of what it booked."""
    from SentenceLooks import Output
    logits = torch.tensor([[0., 4.]], requires_grad=True)
    output = Output([[0]], [0], torch.zeros(1))
    target, length, given = output.target(0, logits.device)
    assert target.tolist() == [0] and length.tolist() == [1] and given.tolist() == [False]
    output.book(torch.nn.functional.cross_entropy(logits, target, reduction='none'), torch.tensor([True]))
    output.book(torch.ones(1), torch.tensor([False]))
    output.cost().sum().backward()
    assert logits.grad[0, 0] < 0 < logits.grad[0, 1]
    assert Output([[0, 1]], [1], torch.zeros(1)).target(0, 'cpu')[2].tolist() == [True]


def test_supplied_reading_resolves_named_occurrences_and_restores_after_exception():
    """Ported (eleventh pass) from the reading lesson, whose resolution of
    named occurrences the supplied reading shares."""
    model = SimpleNamespace()
    prior = object()
    model._attention_supplied_reading = prior
    with pytest.raises(RuntimeError, match='end lesson'):
        with supplied_reading(model, [[('red', 1), 'blue', ('red', 0)]]) as lesson:
            assert lesson.positions([['red','blue','red']], torch.tensor([[[0,3],[4,8],[9,12]]])) == [[2,1,0]]
            raise RuntimeError('end lesson')
    assert model._attention_supplied_reading is prior
    model._attention_output_lesson = prior
    with pytest.raises(RuntimeError, match='end lesson'):
        with output_lesson(model, [['1', '2']], supplied=[1]) as lesson:
            assert lesson.words == (('1', '2'),) and lesson.supplied == (1,)
            raise RuntimeError('end lesson')
    assert model._attention_output_lesson is prior


def test_derivation_inverse_recovers_composed_children_and_depends_on_the_root():
    # A four-leaf binary tree with a real invertible arithmetic kernel. Its
    # co-operands include compounds, without adding them to a word dictionary.
    class Language:
        _compose_binary_rules = [SimpleNamespace(relation_kind=None, method_name='compose')]
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
    program = SimpleNamespace(tree=None,actions=actions,operation_values=frames)
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
        before = model._stm_reducer().chooser.tool_embedding.detach().clone()
        # MTM nominates a form on recurrence (sigma-pi §21, rule 20).
        from nomination_fixtures import nominated_forms
        nominated_forms(model,['red blue','blue red'])
        inputs = model.inputSpace.prepInput(['red blue','blue red'])
        optimizer = model.getOptimizer(lr=.01)
        attention = next(o for o in optimizer.optimizers if getattr(o,'objective_owner',None)=='attention')
        assert isinstance(attention.inner,torch.optim.Adam)
        assert all(group['lr']==.01 for group in attention.param_groups)
        # Ported (eleventh pass): the reading lesson named the order read;
        # the output (the sentence said, rule 80) names no order.
        with output_lesson(model,[['red','blue'],['blue','red']]):
            model.runBatch(train=True,optimizer=optimizer,batchSize=2,batch_override=(inputs,torch.empty(2,0)))
        field = model._last_sentence_field
        assert field['iterations'].tolist() == [2,2]
        assert field['trials'][0]['output_cost'].gt(0).all()
        for b in range(2):
            assert sorted(int(read['index']) for read in field['trials'][0]['reads'] if bool(read['rows'][b])) == [0, 1]
        # Each read is of the rows that took that word at that look; with no
        # order named, the two rows may take different words at one look.
        assert all(read['rows'].any() for read in field['trials'][0]['reads'])
        owned = model._last_understanding.input_reconstruction
        assert torch.isfinite(owned.percepts).all()
        # Ported (sixteenth pass): the one reconstruction puts back what the
        # recorded composition's inverses give, so a byte may be missing
        # (a rule with no inverse); that is measured, not asserted away.
        assert owned.truncated.shape == (2,)
        print('readings put back whole:', int((~owned.truncated).sum()), 'of', len(owned.truncated))
        # The request's read is attention's own (rule 80).
        owners = model.ownership_gradient_diagnostics(optimizer)['parameters']
        writers = [row['writers'] for row in owners if row['owner']=='attention']
        print(dict(attention_writers=writers,
                   chooser_moved=not torch.equal(model._stm_reducer().chooser.tool_embedding,before)))
        assert model._attention_output_lesson is None
    finally:model.End()


def test_lossy_derivation_has_no_inverse_or_search_and_reports_every_coordinate():
    class Language:
        _compose_binary_rules = [SimpleNamespace(relation_kind=None, method_name='compose')]
        def _tree_layer(self, arity):
            return SimpleNamespace(ops=[SimpleNamespace(inverse_kind='search')])
        def reverse_binary_step(self, *args, **kwargs):
            raise AssertionError('the derivation must not search for missing operands')
    words = torch.tensor([[[1., 2.], [3., 4.]]])
    record = SimpleNamespace(word_values=words, word_valid=torch.ones(1,2,dtype=torch.bool),
        end_depth=torch.tensor([1]), end_slots=torch.ones(1,1,2))
    program = SimpleNamespace(tree=None,actions=torch.tensor([[0,-1,0],[0,-1,1],[1,0,-1]]),
        operation_values=torch.stack((torch.zeros(3,2),torch.zeros(3,2),
            torch.stack((words[0,0],words[0,1],torch.ones(2))))))
    recovered, missing, defined = reconstruct(Language(),record,[program],lambda b:torch.arange(2))
    assert missing.all() and not defined.any() and not recovered.any()


def test_partial_product_inverse_keeps_defined_coordinates_and_root_gradient():
    class Language:
        _compose_binary_rules = [SimpleNamespace(relation_kind=None, method_name='compose')]
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
    program=SimpleNamespace(tree=None,actions=torch.tensor([[0,-1,0],[0,-1,1],[1,0,-1]]),operation_values=frames)
    recovered,missing,defined=reconstruct(Language(),record,[program],lambda b:torch.arange(2))
    assert defined.tolist()==[[[True,False],[True,True]]]
    assert missing.tolist()==[[True,False]]
    recovered[defined].sum().backward()
    torch.testing.assert_close(root.grad,torch.tensor([[[1/3,0.]]]))


@pytest.mark.parametrize('known', [False, True], ids=['blank-objects', 'admitted-objects'])
def test_native_four_word_inverse_follows_composed_children_without_free_decoder(tmp_path,monkeypatch,known):
    from test_native_attention_field import _model
    from test_reverse_traversal import _select_completed_binary_path
    from attention_field_gates import words_from_leaves
    model = _model(tmp_path,monkeypatch)
    from reading_fixtures import read_in_source_order
    read_in_source_order(model, monkeypatch)
    _select_completed_binary_path(model)
    try:
        texts=['red blue green gold','gold green blue red']
        if known:
            from test_definition_rows import admit
            for form in texts[0].split():
                assert admit(model, form) is not None
        else:
            # MTM nominates a form on recurrence (sigma-pi §21, rule 20); a
            # word with no object is given its blank object when read.
            from nomination_fixtures import nominated_forms
            nominated_forms(model, texts)
        from reading_fixtures import capture_readings
        with capture_readings(model) as programs, torch.no_grad():
            model.runBatch(train=False,batchSize=2,batch_override=(model.inputSpace.prepInput(texts),torch.empty(2,0)))
        record=model._last_sentence_understanding
        # The derivation inverse is now an explicit audit, outside reconstruction.
        recovered, unavailable, defined = reconstruct(model.languageSpace, record, programs[0],
            lambda b: model._sentence_leaf_positions(record, b))
        recovered = recovered[:, :4]
        result=[' '.join(row) for row in words_from_leaves(recovered,torch.tensor([4,4]),record.primed)]
        assert result==texts
        assert not unavailable.any()
        ops = model.languageSpace._tree_layer(2).ops
        kernel = next(getattr(op, 'gl', op) for op in ops
                      if getattr(getattr(op, 'gl', op), 'percept_band', None) is not None)
        # Both admitted and blank objects are evaluated after the word
        # tree. Its inverse retains the word's form presence in either case.
        target = record.word_values[:, :4]
        if not known:
            ids = [[model._concept_owner().interpret.word_for_form(w) for w in text.split()] for text in texts]
            owner = model._concept_owner()
            bank = owner.similarity_codebook
            blanks = torch.stack([bank.lookup_rows([owner._concept_row_of(owner.interpret.blank_of(word))
                                                    for word in row]) for row in ids])
            assert not torch.equal(kernel.content_operand(target), kernel.content_operand(blanks))
        torch.testing.assert_close(kernel.content_operand(recovered),
            kernel.content_operand(target), atol=1e-5, rtol=1e-5)
    finally:model.End()


def test_supplied_text_reader_preserves_shared_work_spent_during_sentence_scoring(tmp_path,monkeypatch):
    from test_native_attention_field import _model
    from attention_curriculum_gates import present
    model = _model(tmp_path,monkeypatch)
    from reading_fixtures import read_in_source_order
    read_in_source_order(model, monkeypatch)
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
    from reading_fixtures import read_in_source_order
    read_in_source_order(model, monkeypatch)
    _select_completed_binary_path(model)
    try:
        # MTM nominates a form on recurrence (sigma-pi §21, rule 20).
        from nomination_fixtures import nominated_forms
        nominated_forms(model,['red','blue green'])
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
    from reading_fixtures import read_in_source_order
    read_in_source_order(model, monkeypatch)
    # Corrupt only the reported list decoder, not the native reads or cost.
    monkeypatch.setattr(curriculum,'words_from_field',
        lambda surface,where,orders:[['<unrecovered>']*len(order) for order in orders])
    try:
        # MTM nominates a form on recurrence (sigma-pi §21, rule 20).
        from nomination_fixtures import nominated_sentences; nominated_sentences(model, ['red blue','blue red'])
        result=curriculum.present(model,dict(fields=['red blue','blue red'],lesson=True))
        assert result['orders']==[[0,1],[0,1]]
        assert result['passed'] and result['read_gate_passed']
        assert not result['reconstruction_passed'] and result['exact_readbacks']==0
    finally:model.End()


def test_prompt_need_reaches_existing_input_without_an_answer_or_action_target(tmp_path,monkeypatch):
    from test_native_attention_field import _model
    from AttentionLesson import prompt_need
    model = _model(tmp_path,monkeypatch)
    width = model.conceptualSpace.stm.concept_dim
    like = model.conceptualSpace.stm._buffer
    try:
        # Ported (eleventh pass): the context that read the need is gone
        # (rule 80); the need is encoded from the question's words alone.
        with prompt_need(model,['the second word','the middle phrase']) as need:
            assert need.shape[-1] == width and need.dtype == like.dtype
            assert not torch.equal(need[0],need[1])
            assert getattr(model,'_attention_part_lesson',None) is None
        assert model._attention_prompt_need is None
    finally:model.End()
