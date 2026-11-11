"""Stream capture and introduction orders at the native closing boundary."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


def test_two_stream_capture_keeps_each_streams_own_word_codes():
    from FormCentroids import BasicSpanWholes, SpanWholes, capture
    from Understanding import Reading
    codes=torch.tensor([[[1.,0.],[0.,1.]],[[.3,0.],[0.,.5]]])
    weights=torch.eye(2).expand(2,-1,-1)
    prefix=torch.cat((weights.new_zeros(2,1,2),weights.cumsum(1)),1)
    bank=BasicSpanWholes(prefix,codes)
    program=Reading(rows=torch.arange(2),word_rows=torch.arange(2),activations=torch.ones(2),
        leaves=torch.eye(2),actions=torch.tensor([[0,-1,0],[0,-1,1],[1,0,-1]]),
        targets=None,end_state=None,leaf_orders=torch.zeros(2,dtype=torch.long))
    language=SimpleNamespace(_compose_binary_rules=(SimpleNamespace(method_name='part',head_role=0),))
    for stream in range(2):
        result=capture(program,bank,stream,(0,1),language)
        torch.testing.assert_close(result.form_wholes[-1],codes[stream].mean(0))
        torch.testing.assert_close(result.form_whole_presence[-1],codes[stream].mean(0))
    # Even a divisible number of spans cannot legalize a mixed stream view.
    with pytest.raises(ValueError,match='same streams'):
        bank._replace(prefix=prefix[:1]).read_presence(torch.tensor([[[0,1],[1,2]]]))
    shared=SpanWholes(prefix,torch.eye(2),codes[0])
    assert shared.for_stream(1).codes is shared.codes
    torch.testing.assert_close(shared.for_stream(1).prefix,prefix[1:])


@pytest.mark.parametrize('named',[False,True])
def test_local_introduction_claims_the_order_written_by_commit(named):
    from test_reference_close import fixture,known_concept
    from test_medium_term_memory import mention,keep
    resolver,store,atoms=fixture()
    source=known_concept(resolver.components.owner,atoms[0],order=2)
    node=mention(atoms[0],reference=source)
    if named:
        node=replace(node,subject_word_id=source)
    row,plan=keep(resolver,node)
    address=store.occurrence_of(row)
    token=resolver.components.owner.mtm.token_of(address)
    assert plan['items'][0]['member_order']==2
    assert resolver.components.owner.mtm.entries[token].refinement['member_order']==2
    assert store.order_of_row(row)==2


def test_local_alias_refuses_the_order_the_commit_would_reject():
    from ClauseRow import Clause
    from Meaning import ConceptualMeaning
    from MemoryLink import Occurrence
    from chunk_fixtures import context
    from test_reference_close import fixture,known_concept
    from test_medium_term_memory import mention
    resolver,store,atoms=fixture()
    owner=resolver.components.owner
    low=known_concept(owner,atoms[0],order=0)
    high=known_concept(owner,atoms[0],order=1)
    first=replace(mention(atoms[0],reference=low),subject_word_id=low)
    alias=replace(mention(atoms[0],mode='bind'),subject_word_id=high)
    alias=replace(alias,composition=replace(alias.composition,reference=high))
    root=Clause(ConceptualMeaning.from_description(atoms[0]),point=atoms[0],
        children=(first,alias),refs=(('clause',0),-1,('clause',1)))
    plan=resolver.preview(root,context(resolver,()))
    assert plan['individuals'][('new',0)]['member_order']==0
    assert ('new',0) not in plan['items'][1]['candidates']
    assert plan['items'][1]['own_object']
    kept=resolver.prepare(root,plan)
    written={}
    row=store.write_clause(kept,written_rows=written)
    resolver.commit(plan,row,written,document='orders',position=0,training=False)
    first_address=plan['mention_occurrences'][id(plan['items'][0]['written_node'])]
    alias_address=plan['mention_occurrences'][id(plan['items'][1]['written_node'])]
    first_token=owner.mtm.token_of(first_address)
    assert owner.mtm.entries[first_token].refinement['member_order']==0
    assert owner.mtm.entries[owner.mtm.token_of(alias_address)].refinement['member_order']==1
    with pytest.raises(ValueError,match='cannot mix collection kinds or orders'):
        owner.link(Occurrence(alias_address,concept=high,selected=first_token),{},
            value=atoms[0],where=None,when=1,stream=None)
