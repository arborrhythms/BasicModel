from pathlib import Path
p=Path('test/test_output_walk.py');s=p.read_text()
a=s.index('def test_walk_unreduces_a_rule_stamped_top');z=s.index('\ndef test_walk_emits_terminals',a)
s=s[:a]+'''def _independent_split(m, rule, event):
    """An explicit two-child bank and a tensor policy; no compose journal."""
    language=m.languageSpace
    index=list(language._generate_binary_names).index(rule)
    op=language._generate_binary_ops[index]
    left,right=event[:,0].clone(),event[:,1].clone()
    cw=m._stamp_channel(event.shape[-1])
    left[:,cw:]=0;right[:,cw:]=0
    parent=op.compose(left,right)
    event=event.clone()
    event[:,0]=-left*.17
    event[:,1]=parent
    bank=torch.stack((left,right),1)
    def policy(top):
        match=(top-parent).abs().amax(-1)<1e-6
        scores=top.new_full((top.shape[0],language.generate_policy.out_features),-1000.)
        scores[:,-1]=0.
        scores[:,index]=torch.where(match,1.,-1.)
        return scores
    language.generate_policy_logits=policy
    return event,bank


def test_walk_unreduces_an_independently_selected_parent_and_emits_its_constituents():
    m = _model()
    for rule in ('lift','lower'):
        event,gl,cw,live=_stamped_event(m,rule)
        event,bank=_independent_split(m,rule,event)
        out,n_emitted,truncated,_cost=m._output_generate_walk(event,8,basis=bank)
        assert n_emitted.tolist()==[live+1]
        assert not bool(truncated.any())
        below,left,right=out[0,0],out[0,1],out[0,2]
        torch.testing.assert_close(below,event[0,0],rtol=0,atol=0)
        assert float(left[cw])==0. and float(right[cw])==0.
        recomposed=gl.compose(left[None],right[None])[0]
        assert torch.allclose(recomposed[:cw],event[0,1,:cw],atol=1e-3),rule
        assert float(out[0,3:].abs().sum())==0.

''' +s[z:]
a=s.index('def test_walk_reports_truncation_when_work_is_pending');z=s.index('\ndef test_walk_compiles',a)
s=s[:a]+'''def test_walk_reports_truncation_when_work_is_pending():
    """A legal unary rewrite can spend all three trips without emitting."""
    m = _model()
    event,gl,cw,live=_stamped_event(m,'lift',N=2)
    language=m.languageSpace
    _prefer(m,len(language._generate_binary_ops)+list(language._generate_unary_names).index('null'))
    out,n_emitted,truncated,_cost=m._output_generate_walk(event,budget=3)
    assert bool(truncated.all()) and n_emitted.tolist()==[0]

''' +s[z:]
a=s.index('def test_walk_handles_mixed_output_lengths_per_row');z=s.index('\ndef test_walk_is_invariant',a)
s=s[:a]+'''def test_walk_handles_mixed_output_lengths_per_row():
    m=_model()
    e0,gl,cw,live=_stamped_event(m,'lift')
    e0,bank=_independent_split(m,'lift',e0)
    e1=e0.clone();e1[:,1]=bank[:,0]
    event=torch.cat((e0,e1),0)
    out,n_emitted,truncated,_cost=m._output_generate_walk(event,8,basis=bank.expand(2,-1,-1))
    assert n_emitted.tolist()==[live+1,live]
    assert not bool(truncated.any())
    assert torch.equal(out[1,:live],event[1,:live])
    assert float(out[0,2].abs().sum())>0.

''' +s[z:]
a=s.index('def test_generate_policy_returns_sampled_action_credit_without_imitation');z=s.index('\ndef test_generate_policy_decides',a)
s=s[:a]+'''def test_generate_policy_has_numerical_credit_without_imitation():
    """The decoder's owner supplies the loss; no sampled imitation term."""
    m=_model()
    try:
        language=m.languageSpace
        event,gl,cw,live=_stamped_event(m,'lift')
        _stop(m)
        m.zero_grad(set_to_none=True)
        deterministic=m._output_generate_walk(event,8,basis=event[:,:live])
        assert not bool(deterministic[3].any())
        torch.manual_seed(7)
        out,n_emitted,truncated,cost=m._output_generate_walk(event,8,basis=event[:,:live],sample_actions=True)
        assert cost.shape==(1,) and float(cost.detach())==0.
        assert bool(torch.isfinite(cost))
        out.square().sum().backward()
        assert language.generate_policy.weight.grad is not None
        assert float(language.generate_policy.weight.grad.abs().sum())>0
    finally:
        m.End();m.symbolSpace.soft_reset()

''' +s[z:]
# An independent binary inverse must have its candidate bank.
s=s.replace('m._output_generate_walk(event, budget=1)','m._output_generate_walk(event, budget=1, basis=event[:, :live])')
# Keep the original seed in the training probe; read the real ownership update.
a=s.index('def _policy_training_probe');z=s.index('\ndef test_question_conditioners_persist',a)
s=s[:a]+'''def _policy_training_probe(m, opt, questions):
    """Inspect actual decoder gradients after the restricted owned backward."""
    params=list(m.languageSpace.generate_policy.parameters())
    backward=m._backward_training_loss
    observed={'reconstruction':[], 'output':[]}
    before=[p.detach().clone() for p in params]
    def probe(*args,**kwargs):
        result=backward(*args,**kwargs)
        owner='reconstruction' if getattr(m,'_sentence_backward',False) else 'output'
        observed[owner].append(tuple(None if p.grad is None else p.grad.detach().clone() for p in params))
        return result
    m._backward_training_loss=probe
    try:
        batch=(m.inputSpace.prepInput(['12 plus 1','3 plus 4']),torch.zeros(2,1,1))
        torch.manual_seed(11)
        m.runBatch(train=True,batchSize=2,split='train',optimizer=opt,batch_override=batch,questions=questions)
    finally:
        m._backward_training_loss=backward
    observed['changed']=[not torch.equal(p,old) for p,old in zip(params,before)]
    observed['mask']=m._last_answer_mask.tolist()
    assert not hasattr(m,'_output_policy_cost')
    assert all(g is None or not bool(g.abs().any()) for row in observed['output'] for g in row)
    return observed


@pytest.mark.parametrize('supplied',[True,False])
def test_runbatch_trains_generate_policy_from_reconstruction(supplied,eager_reading):
    from What import What
    m=_model();m._tensor_peer_while_eager=True;m._chart_compose_per_word=lambda:None
    m.inputSpace.data.has_supervised_outputs=supplied
    opt=m.getOptimizer(lr=1e-3)
    try:
        got=_policy_training_probe(m,opt,(What.supervised(0),What.supervised(1)))
        assert any(g is not None and bool(g.abs().sum()>0) for row in got['reconstruction'] for g in row)
        assert all(got['changed'])
        owned=[id(p) for group in opt.param_groups for p in group['params']]
        assert all(owned.count(id(p))==1 for p in m.languageSpace.generate_policy.parameters())
        groups=m.objective_parameter_groups(opt)
        assert all(any(p is candidate for candidate in groups['reconstruction']) for p in m.languageSpace.generate_policy.parameters())
    finally:
        m.End();m.symbolSpace.soft_reset()


def test_runbatch_answer_availability_never_reassigns_the_decoder(eager_reading):
    from What import What
    m=_model();m._tensor_peer_while_eager=True;m._chart_compose_per_word=lambda:None
    opt=m.getOptimizer(lr=1e-3)
    try:
        for supplied in (True,False):
            m.inputSpace.data.has_supervised_outputs=supplied
            got=_policy_training_probe(m,opt,(What.supervised(0),What.supervised(1)))
            assert all(got['changed'])
            groups=m.objective_parameter_groups(opt)
            assert not any(any(p is q for q in groups['output']) for p in m.languageSpace.generate_policy.parameters())
        m.answer_synthesis=False
        got=_policy_training_probe(m,opt,(What.present(0),What.present(1)))
        assert all(got['changed'])
    finally:
        m.End();m.symbolSpace.soft_reset()


def test_runbatch_output_masks_rows_without_reassigning_decoder_credit(eager_reading):
    from What import What
    m=_model();m._tensor_peer_while_eager=True;m._chart_compose_per_word=lambda:None
    opt=m.getOptimizer(lr=1e-3)
    try:
        got=_policy_training_probe(m,opt,(What.supervised(0),What.inference(1,split='train')))
        assert got['mask']==[True,False]
        assert all(got['changed'])
    finally:
        m.End();m.symbolSpace.soft_reset()

''' +s[z:]
# Words separated by spaces now have intervening primitive slots. Keep this
# snapshot identity test at two explicit word positions.
a=s.index('def test_materialised_idea_follows_the_symbol_rows');z=s.index('\ndef test_reverseoutput_evaluation',a)
part=s[a:z].replace('["12 plus 1", "3 plus 4"]','["one plus two", "three plus four"]')
part=part.replace('rows[:, :2]','rows[:, (0, 2)]').replace('rows[:, 1]','rows[:, 2]')
part=part.replace('table[:, :2] = table[:, :2].flip(1)','table[:, (0, 2)] = table[:, (0, 2)].flip(1)')
part=part.replace('m._word_symbol_rows()[:, :2]','m._word_symbol_rows()[:, (0, 2)]').replace('["plus 12 1", "plus 3 4"]','["plus one two", "plus three four"]')
s=s[:a]+part+s[z:]
a=s.index('def test_output_rule_inventory_comes_from_generate_even_without_compose_rule');z=s.index('\n@pytest.mark.parametrize',a)
s=s[:a]+'''def test_output_rule_inventory_comes_from_generate_even_without_compose_rule(tmp_path):
    m=_generate_variant(tmp_path,('sum',),remove_compose=('sum',))
    try:
        language=m.languageSpace
        assert 'sum' not in language._tree_layer(2).op_names
        assert language.generate_policy.out_features==2
        event,_,cw,live=_stamped_event(m,'sum')
        event,bank=_independent_split(m,'sum',event)
        out,count,truncated,cost=m._output_generate_walk(event,8,basis=bank)
        assert count.tolist()==[live+1]
        assert not bool(truncated.any()) and not bool(cost.any())
        torch.testing.assert_close((out[:,1]+out[:,2])/2,event[:,1])
        compiled=torch.compile(m._output_generate_walk,backend='eager',fullgraph=True)
        result=compiled(event,8,basis=bank)
        for actual,expected in zip(result,(out,count,truncated,cost)):
            torch.testing.assert_close(actual,expected)
    finally:
        m.End();m.symbolSpace.soft_reset()

''' +s[z:]
p.write_text(s)
# Sum is now mean. The additive chunk inverse owns this exact halving fixture.
p=Path('test/test_prepared_answer_boundary.py');s=p.read_text().replace('index("sum")','index("chunk")').replace("# Sum's witness-free inverse halves its parent.","# Chunk is additive; its equal-child inverse halves its parent.")
# Provide the full bounded candidate family for free inverse search, not a recorded child.
s=s.replace('''            construction = model.reverseOutput(understanding, held)
            lengths =''','''            bank=held.primed_symbols
            codes=torch.stack((direction*.25,direction*.5,direction),0)[None].expand(2,-1,-1)
            bank=replace(bank,codes=codes,rows=torch.arange(3)[None].expand(2,-1),
                         weights=torch.ones(2,3),own=torch.ones(2,3,dtype=torch.bool),
                         bytes=bank.bytes[:,:1].expand(-1,3,-1),byte_valid=bank.byte_valid[:,:1].expand(-1,3,-1))
            held=replace(held,primed_symbols=bank)
            construction = model.reverseOutput(understanding, held)
            lengths =''')
p.write_text(s)
