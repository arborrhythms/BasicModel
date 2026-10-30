"""Independent signed witnesses teach sparse sigma, never the codebook."""
from types import SimpleNamespace
import torch
from test_familiar_tier import placed_fixture
from ConceptIndex import index_part_row
from DefinitionFold import examples, prepare, loss
from LTM1 import SYMBOL


def fixture():
    cs, cb, alloc, store, (a,b,base) = placed_fixture()
    ids = [cs.concept_id_at_row(row) for row in (a,b,base)]
    cid = index_part_row(cs, ids[0], ids[2])
    row = cs._csw_row_of(cid)
    witness = alloc.parts.find((SYMBOL,ids[0]),(SYMBOL,cid))
    alloc.parts.add(ids[0],cid,truth_minus=.8)
    field = torch.zeros(sum(cs._order_caps()),1,2,2,requires_grad=True)
    with torch.no_grad():
        field[a,0,0,0]=.6
    return cs,cb,alloc,cid,row,a,witness,field


def test_witnessed_lanes_merge_by_max_without_absence_as_a_label():
    cs,cb,alloc,cid,row,a,witness,field=fixture()
    staged=prepare(cs,field)
    matrix=alloc.layer(0)
    with torch.no_grad():
        matrix.values[matrix._index[row,a]]=.2
    records=examples(cs,field)
    assert [(e.concept,e.witness,e.source) for e in records]==[(cid,witness,a)]
    assert records[0].names==() and not records[0].activation.requires_grad
    cost=loss(cs,staged)
    cost.sum().backward()
    assert matrix.values.grad[matrix._index[row,a]]<0
    assert matrix.values.grad[matrix._index[row,a+matrix.nOutput+1]]<0
    assert field.grad is None and cb.W.grad is None
    assert alloc.parts.truth_plus.grad is None and alloc.parts.truth_minus.grad is None
    report=cs._last_definition_fold['readings'][0]
    assert report['teacher'][0,0].tolist()==[.6,.6] or torch.allclose(report['teacher'][0,0],torch.tensor([.6,.6]))
    assert not report['witnessed'][0,1].any()
    # Repeated testimony remains idempotent in each independent lane.
    alloc.parts.add(cs.concept_id_at_row(a),cid,truth_plus=.2,truth_minus=.3)
    torch.testing.assert_close(loss(cs,prepare(cs,field)),cost)


def test_named_and_unnamed_classes_share_the_teacher_and_opposite_edges_keep_adam():
    cs,cb,alloc,cid,row,a,witness,field=fixture()
    matrix=alloc.layer(0)
    optimizer=torch.optim.Adam([matrix.values],lr=.01)
    matrix.values.sum().backward();optimizer.step();optimizer.zero_grad()
    old=optimizer.state[matrix.values]['exp_avg'].clone()
    word=cs.new_concept()
    alloc.symbol_types[word]={'word'}
    alloc.reference_orders[word]=0
    cs._csw_concept_row(0,word)
    owner=SimpleNamespace(_optimizer=optimizer)
    object.__setattr__(cs,'_model',owner)
    staged=prepare(cs,field)
    assert torch.equal(optimizer.state[matrix.values]['exp_avg'][:len(old)],old)
    unnamed=loss(cs,staged).detach()
    cs.interpret.define(word,cid,description=dict(forms=('label',),parts=((4,),),wholes=()))
    named=loss(cs,prepare(cs,field))
    torch.testing.assert_close(named,unnamed)
    assert any(e.names==(word,) for e in examples(cs,field))
    assert all(e.concept!=word for e in examples(cs,field))


def test_existing_stronger_pole_is_not_erased_and_missing_lane_has_no_gradient():
    cs,cb,alloc,cid,row,a,witness,field=fixture()
    alloc.parts.truth_minus[witness]=0
    alloc.parts.truth_plus[witness]=.2
    staged=prepare(cs,field)
    value=loss(cs,staged)
    assert value.eq(0).all()
    value.sum().backward()
    assert not alloc.layer(0).values.grad.any()


def test_definition_fold_uses_the_kept_row_mask_under_its_existing_writer():
    from Layers import Error
    from ObjectiveOwnership import registry_costs
    parameter=torch.nn.Parameter(torch.tensor([.1,.2]))
    registry=Error(row_mask=torch.ones(2,dtype=torch.bool))
    registry.error('reconstruction.definition_fold',parameter.square(),1.,category='reconstruction')
    cost=registry_costs(registry,definition_rows=torch.tensor([False,True]))['reconstruction']
    cost.backward()
    assert parameter.grad[0]==0 and parameter.grad[1]>0
