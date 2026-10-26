"""Fixed shared-working-set mechanisms; vectors and choices are explicit."""
import pytest
import torch

import STMChunks as chunks


def put(state, scalar, *, stack=False, identity=-1):
    return chunks.deposit(state, state[0].new_full((state[0].shape[0], state[0].shape[-1]), scalar),
        torch.ones(state[0].shape[0], dtype=torch.bool), stack=stack,
        identity=identity, order=1, grammar_order=0)


def test_shared_set_evicts_least_read_off_stack_and_refuses_all_stack():
    state = chunks.empty(1, 8, 4)
    for value in range(1, 9):
        state, index = put(state, value, stack=value in (1, 4))
        assert int(index) == value-1
    state = chunks.read(state, torch.tensor([1]))
    before = tuple(value.clone() for value in state)
    state, index = put(state, 9)
    assert int(index) == 2
    assert state[chunks.STACK][0, 0] == 1
    assert state[chunks.STACK][0, 3] == 0
    torch.testing.assert_close(state[0][0, 0], before[0][0, 0])
    torch.testing.assert_close(state[0][0, 3], before[0][0, 3])
    for _ in range(6):
        state, _ = put(state, 10, stack=True)
    before = tuple(value.clone() for value in state)
    refused, index = put(state, 11, stack=True)
    assert int(index) == -1
    for actual, expected in zip(refused, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_content_is_an_occupied_chunk_with_ordinary_recency():
    state = chunks.empty(1,8,4)
    state,_ = put(state,0.,identity=80)
    state,second = put(state,.2,identity=81)
    assert int(second) == 1
    for number in range(2,8):
        state,_ = put(state,number/10)
    state = chunks.read(state,torch.tensor([0]))
    state,index = put(state,.9)
    assert int(index) == 1 and int(state[chunks.IDENTITY][0,0]) == 80


def test_menu_offer_is_not_read_and_kept_bind_reads_its_field_position():
    from Spaces import LanguageOperationChoice
    state = chunks.empty(1, 8, 4)
    state, first = put(state, .2, identity=100)
    state, second = put(state, .6, identity=101)
    state, mention = put(state, .4, stack=True)
    ages = state[chunks.READ].clone()
    values, ids, indices, valid = chunks.bind_menu(state, torch.full((1,1,4), .4))
    assert valid.sum() == 3
    assert ids[valid].tolist() == [0,100,101]
    torch.testing.assert_close(ages, state[chunks.READ])
    choice = LanguageOperationChoice(values[:,0,2], torch.ones(1,dtype=torch.long),
        torch.zeros(1,dtype=torch.long), torch.zeros(1,dtype=torch.long),
        torch.ones(1,dtype=torch.bool), torch.ones(1), torch.zeros(1,dtype=torch.long),
        torch.ones(1,dtype=torch.bool), order=torch.ones(1,dtype=torch.long),
        identity=ids[:,0,2], head_role=torch.ones(1,dtype=torch.long),
        stack_arity=torch.ones(1,dtype=torch.long), chunk=indices[:,0,2])
    kept = chunks.apply_choice(state, choice)
    assert kept[1].item() == 1
    assert kept[chunks.READ][0,second] > ages[0,second]
    assert kept[chunks.READ][0,first] == ages[0,first]
    assert kept[chunks.IDENTITY][0,mention] == 101
    torch.testing.assert_close(chunks.stack_view(kept)[0][:,0], torch.full((1,4),.6))


def test_close_keeps_one_recent_chunk_and_fork_does_not_share_mutation():
    from SentenceFork import detached
    state = chunks.empty(1, 8, 4)
    state, recent = put(state, .2, identity=80)
    state, older = put(state, .4, stack=True)
    state, top = put(state, .6, stack=True)
    initial = detached((state, (), ()))
    refs = torch.tensor([[[chunks.OCCURRENCE,123],[0,0]]])
    stamp = torch.tensor([[1.,2.,3.,4.]])
    kept = chunks.keep_close(state, torch.tensor([True]), refs=refs, when=stamp)
    assert kept[1].item() == 0
    assert kept[chunks.STACK].eq(-1).all()
    assert kept[0][0,older].eq(0).all()
    torch.testing.assert_close(kept[0][0,recent], state[0][0,recent])
    torch.testing.assert_close(kept[chunks.REFS][0,top], refs)
    torch.testing.assert_close(kept[chunks.WHEN][0,top], stamp)
    assert initial[0][1].item() == 2
    assert initial[0][chunks.REFS].eq(0).all()


def test_identity_pool_is_fixed_releases_at_promotion_and_rejects_old_layout():
    from MediumTermMemory import MediumTermMemory
    pool = MediumTermMemory(2, n_vectors=90)
    value = torch.tensor([.1,.3,.5,.7])
    first, second, third = [('ltm','fixture',i) for i in (1,2,3)]
    pointer = pool.transient_ids.data_ptr()
    pool.open(first, kind=7, when=1, value=value, where=value)
    pool.bind(first, second, kind=7, when=2, value=value, where=value.flip(0))
    assert pool.entries[first].transient == 90
    assert pool.token_of(90) == first
    torch.testing.assert_close(pool.entries[first].where, value.flip(0))
    pool.promote(first, 22)
    assert pool.entries[first].transient is None
    assert not pool.transient_used.any()
    pool.open(third, kind=7, when=3, value=value)
    assert pool.entries[third].transient == 90
    state = pool.state()
    restored = MediumTermMemory(2, n_vectors=90)
    restored.restore(state)
    assert restored.token_of(90) == third
    with pytest.raises(ValueError, match='version 8'):
        restored.restore(dict(state, version=7))
    restored.forget(third)
    assert not restored.transient_used.any()
    assert pool.transient_ids.data_ptr() == pointer


def test_published_identity_inherits_stamp_and_refresh_preserves_it():
    from ChunkJournal import ChunkJournal
    journal = ChunkJournal((),(),pool_capacity=2,percept_band=(4,12,12))
    gate = torch.tensor([True])
    where,when = torch.tensor([[.2,.4,.6,.8]]),torch.tensor([[.1,.3,.5,.7]])
    state,_ = chunks.deposit(chunks.empty(1,8,12),torch.zeros(1,12),gate,
        stack=True,where=where,when=when)
    identity,timeline = torch.tensor([80]),torch.tensor([0])
    state = journal.publish(state,identity,timeline,torch.full((1,12),.25),gate)
    index = int((state[chunks.IDENTITY][0] == 80).nonzero()[0])
    torch.testing.assert_close(state[0][0,index,4:],torch.cat((where,when),-1)[0])
    torch.testing.assert_close(state[chunks.WHEN][0,index],when[0])
    age = state[chunks.READ][0,index].clone()
    state = journal.publish(state,identity,timeline,torch.full((1,12),.75),gate)
    assert state[chunks.READ][0,index] > age
    torch.testing.assert_close(state[0][0,index,:4],torch.full((4,),.75))
    torch.testing.assert_close(state[0][0,index,4:],torch.cat((where,when),-1)[0])


def test_publication_and_refresh_choose_the_field_beside_a_recent_sentence():
    from ChunkJournal import ChunkJournal,read_identity
    journal = ChunkJournal((),())
    gate = torch.tensor([True])
    state = chunks.empty(1,8,4)
    state,_ = chunks.deposit(state,torch.full((1,4),.1),gate,identity=80,
        refs=torch.tensor([[[chunks.OCCURRENCE,42],[0,0]]]))
    state,_ = chunks.deposit(state,torch.full((1,4),.3),gate,identity=80,
        refs=torch.tensor([[[chunks.IDENTITY_REF,80],[0,0]]]))
    state,head = chunks.deposit(state,torch.full((1,4),.3),gate,stack=True,identity=80)
    state = journal.publish(state,torch.tensor([80]),torch.tensor([0]),torch.full((1,4),.8),gate)
    state,actual = chunks.refresh(state,head[:,None])
    torch.testing.assert_close(actual,torch.full((1,1,4),.8))
    current,found,index = read_identity(state,torch.tensor([[80]]),torch.tensor([[0]]))
    assert found.all() and int(index) == 1
    torch.testing.assert_close(current,actual)
    torch.testing.assert_close(state[0][0,0],torch.full((4,),.1))


def test_stm_construction_bounds_are_loud_and_soft_reset_retains_recent():
    from Layers import ShortTermMemory
    memory = ShortTermMemory(batch=2, capacity=8, concept_dim=4)
    state, _ = put(memory.state, .3, identity=9)
    state, _ = put(state, .6, stack=True)
    memory.set_state(state)
    memory.begin_forward(2)
    assert memory._depth.eq(0).all()
    assert memory._buffer.ne(0).any(-1).sum() == 2
    with pytest.raises(ValueError, match='nStreams'):
        memory.ensure_batch(3)
    with pytest.raises(ValueError, match='stmCapacity'):
        memory.ensure_capacity(9)


def test_reduce_retains_the_head_percept_in_buffer_and_sidecars():
    from types import SimpleNamespace
    from chunk_fixtures import Walk,choice
    walk = Walk((SimpleNamespace(method_name='product',head_role=1),),())
    walk.journal.percept_band = (4,12,12)
    state = walk.empty(torch.zeros(2,12),1)
    state = walk.shift(state,torch.full((2,12),.2),word=1,index=0)
    state = walk.shift(state,torch.full((2,12),.4),word=2,index=1)
    stm = list(state[0])
    head = chunks.stack_indices(stm)[:,1]
    where = torch.tensor([[.1,.2,.3,.4]]).expand(2,-1)
    when = torch.tensor([[.5,.6,.7,.8]]).expand(2,-1)
    stm[chunks.WHERE] = chunks.write(stm[chunks.WHERE],head,where,torch.ones(2,dtype=torch.bool))
    stm[chunks.WHEN] = chunks.write(stm[chunks.WHEN],head,when,torch.ones(2,dtype=torch.bool))
    stm = chunks.clear(tuple(stm),torch.tensor([[True]*8,[False]*8]))
    state = (stm,*state[1:])
    selected = choice(torch.full((2,12),.3),head=1)._replace(applied=torch.tensor([False,True]))
    state = walk.apply(state,selected,torch.zeros(2,2,12),0)
    actual = state[0]
    index = int(chunks.at(actual)[1])
    torch.testing.assert_close(actual[0][1,index,4:],torch.cat((where,when),-1)[1])
    torch.testing.assert_close(actual[chunks.WHEN][1,index],when[1])
    for before,after in zip(stm,actual):
        torch.testing.assert_close(before[0],after[0],rtol=0,atol=0)


def test_metadata_prefix_writes_preserve_the_constructed_stream_extent():
    from Layers import ShortTermMemory
    from ChunkJournal import empty_scope
    memory = ShortTermMemory(batch=4,capacity=8,concept_dim=4)
    shapes = {name:value.shape for name,value in memory.__dict__.items()
              if name.startswith('_live_') and torch.is_tensor(value)}
    memory._clause_state = empty_scope(memory._buffer)+7
    memory._wholes = torch.full((4,8,5),9,dtype=torch.long)
    memory._last_unary = torch.full((4,8),11,dtype=torch.long)
    memory._unary_history_depth = torch.full((4,),3,dtype=torch.long)
    memory.begin_forward(3)
    memory._clause_state = torch.zeros(3,8,2,dtype=torch.long)
    memory._wholes = torch.zeros(3,8,3,dtype=torch.long)
    memory._last_unary = torch.zeros(3,8,dtype=torch.long)
    memory._unary_history_depth = torch.zeros(3,dtype=torch.long)
    assert memory._live_clause_state[3].eq(7).all() == False  # reference column began at -1
    assert memory._live_clause_state[3,:,1].eq(6).all()
    assert memory._live_wholes[3].eq(9).all()
    assert memory._live_last_unary[3].eq(11).all()
    assert memory._live_unary_history_depth[3] == 3
    memory.begin_forward(4)
    assert memory.ensure_clause_state().shape == (4,8,2)
    assert memory.ensure_whole_state().shape == (4,8,3)
    assert shapes == {name:value.shape for name,value in memory.__dict__.items()
                      if name.startswith('_live_') and torch.is_tensor(value)}
    with pytest.raises(ValueError,match='provenance width'):
        memory._clause_state = torch.zeros(4,8,15,dtype=torch.long)
