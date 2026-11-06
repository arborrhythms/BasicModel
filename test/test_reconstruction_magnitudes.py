"""§40. A small observed parent cannot magnify the joint pair residual."""
import pytest
import torch
from test_meaning_kernels import paired


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_pair_residual_at_vanishing_parent_has_native_squared_error(dtype):
    from DecompositionChooser import DecompositionChooser
    from Language import LanguageSpace, SumLayer

    op=SumLayer()
    basis=torch.tensor([[[.1,.2],[.8,.6]]],dtype=dtype)
    chooser=DecompositionChooser().to(dtype=dtype)
    valid=torch.ones(1,2,dtype=torch.bool)
    flag=torch.zeros(1,dtype=torch.bool)
    residuals=[]
    for scale in (1.e-6,0.):
        parent=torch.tensor([[scale,2*scale]],dtype=dtype)
        _,_,ready,details=LanguageSpace._bounded_binary_reconstruction(
            op,parent,torch.zeros_like(parent),flag,flag,basis,valid,2,chooser=chooser,return_details=True)
        folded=op.compose(basis[:,:,None],basis[:,None,:])
        expected=(folded-parent[:,None,None]).square().mean(-1)
        torch.testing.assert_close(details['pair_residual'],expected)
        assert ready.all()
        residuals.append(details['pair_residual'])
    torch.testing.assert_close(residuals[0],residuals[1],atol=5e-6,rtol=0.)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_lower_pair_residual_reads_tiny_evidence_in_its_own_chart(monkeypatch,dtype):
    import copy
    from util import TheXMLConfig
    from DecompositionChooser import DecompositionChooser
    from Language import LanguageSpace, LowerLayer

    # Construct the fixture's actual two-form/four-meaning chart. Assigning
    # a layout after construction leaves a default-sized learned kernel.
    config=copy.deepcopy(TheXMLConfig.data)
    config.setdefault('architecture',{}).setdefault('data',{})['dataType']='embedding'
    config.setdefault('PartSpace',{})['nDim']=2
    config.setdefault('ConceptualSpace',{})['nDim']=6
    monkeypatch.setattr(TheXMLConfig,'_data',config)
    op=LowerLayer(6,6).to(dtype=dtype)
    assert op.meaning_layout==(2,6)
    basis=torch.tensor([[[1.,0.,0.,0.,0.,0.],[0.,1.,1.,0.,0.,1.]]],dtype=dtype)
    valid=torch.ones(1,2,dtype=torch.bool)
    flag=torch.zeros(1,dtype=torch.bool)
    chooser=DecompositionChooser().to(dtype=dtype)
    for stance in (5e-8,0.):
        parent=torch.tensor([[.5,.5,stance,0.,0.,stance]],dtype=dtype)
        _,_,ready,details=LanguageSpace._bounded_binary_reconstruction(op,parent,torch.zeros_like(parent),
            flag,flag,basis,valid,2,chooser=chooser,return_details=True)
        assert ready.all() and details['allowed'][0,0,1]
        # The form of this orientation folds exactly to the parent. Its
        # two evidence lanes come from the determiner's right operand.
        expected=(basis[0,1,2:]-parent[0,2:]).square().mean()
        torch.testing.assert_close(details['pair_residual'][0,0,1],expected)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_committed_centroids_invert_the_operand_each_step_actually_read(paired,dtype):
    """A refreshed STM field is a different point from the preceding result."""
    from types import MethodType, SimpleNamespace
    from DerivationReconstruction import reconstruct
    from Language import LanguageSpace, PrepositionLayer, PrepositionUnaryLayer

    op=PrepositionUnaryLayer(PrepositionLayer(12,12)).to(dtype)
    op.percept_band=(2,4,12)
    layer=SimpleNamespace(unary_ops=[op])
    language=SimpleNamespace(_tree_layer=lambda _:layer,_compose_unary_rules=[op])
    language.reverse_unary_step=MethodType(LanguageSpace.reverse_unary_step,language)
    leaf=torch.tensor([[.125,.75,.25,.5]+[0.]*8],dtype=dtype)
    field=torch.tensor([[.75,.25,.625,.875]+[0.]*8],dtype=dtype)
    mean=torch.tensor([[.5,.125]],dtype=dtype)
    count=torch.ones(1,1,dtype=dtype)
    depth=17
    frames=[leaf.new_zeros(3,12)]
    for i in range(depth):
        operand=field if i else leaf
        value=op.compose_blocks(None,op.content_operand(operand),whole=(mean,count))
        frames.append(torch.cat((operand,torch.zeros_like(operand),value)))
    program=SimpleNamespace(actions=torch.tensor([[0,-1,0]]+[[2,0,0]]*depth),
        operation_values=torch.stack(frames),identity_actions=torch.zeros(depth+1,6,dtype=torch.long),
        operand_counts=leaf.new_tensor([[1,0]]*(depth+1)),form_wholes=mean.expand(depth+1,-1),
        form_whole_counts=count.expand(depth+1,-1))
    # The last chunk has its own physical location. A nonzero error in its
    # actual form must survive the inverse. The scored target is never a witness.
    for delta in (0.,2.**-20):
        end=value.clone();end[0,0]+=delta;end[0,2:4]=field[0,2:4]
        record=SimpleNamespace(word_values=(leaf+1/32)[None],word_valid=torch.ones(1,1,dtype=torch.bool),
            end_slots=end[None],end_depth=torch.ones(1,dtype=torch.long),primed=None)
        recovered,missing,_=reconstruct(language,record,[program],lambda _: [0])
        expected=leaf.clone();expected[0,0]+=2**depth*delta
        assert not missing.any()
        torch.testing.assert_close(recovered[0],expected)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_retained_lower_transports_reread_evidence_in_its_native_chart(paired, dtype):
    from types import MethodType, SimpleNamespace
    from DerivationReconstruction import reconstruct
    from Language import LanguageSpace, LowerLayer, LowerUnaryLayer
    from Layers import cube_chart, cube_unchart

    op = LowerUnaryLayer(LowerLayer(12, 12)).to(dtype)
    layer = SimpleNamespace(unary_ops=[op])
    language = SimpleNamespace(_tree_layer=lambda _: layer, _compose_unary_rules=[op])
    language.reverse_unary_step = MethodType(LanguageSpace.reverse_unary_step, language)
    # The second node rereads a different identity. Two coordinates remain
    # direct numerical edges, including one at each of the cube's rails.
    leaf = torch.tensor([[.25] * 4 + [.96, 0., 1., .4, .04, 1., 0., .6]], dtype=dtype)
    read = leaf.clone()
    read[0, [4, 8]] = .5
    frames = torch.stack([leaf.new_zeros(3, 12)] + [
        torch.cat((x, torch.zeros_like(x), op.compose(x))) for x in (leaf, read)])
    program = SimpleNamespace(actions=torch.tensor([[0, -1, 0], [2, 0, 0], [2, 0, 0]]),
                              operation_values=frames, identity_actions=torch.zeros(3, 6))
    for delta in (.0459874057769775, 0.):
        end = read.clone()
        end[0, 4] += delta
        end[0, 8] -= delta
        end.requires_grad_()
        record = SimpleNamespace(word_values=leaf[None], word_valid=torch.ones(1, 1, dtype=torch.bool),
                                 end_slots=end[None], end_depth=torch.ones(1, dtype=torch.long))
        recovered, missing, _ = reconstruct(language, record, [program], lambda _: [0])
        expected = cube_unchart(cube_chart(leaf[:, [4, 8]]) +
                                (cube_chart(end[:, [4, 8]]) - cube_chart(read[:, [4, 8]])))
        assert not missing.any()
        torch.testing.assert_close(recovered[0, :, [4, 8]], expected)
        direct = [5, 6, 7, 9, 10, 11]
        assert torch.equal(recovered[0, :, direct], leaf[:, direct])
        gradient, = torch.autograd.grad(recovered[..., [4, 8]].sum(), end)
        assert gradient.isfinite().all() and (gradient[:, [4, 8]] > 0).all()


@pytest.mark.parametrize('count', [2, 5])
def test_unfinished_forest_trains_from_every_live_stack_root(count):
    from types import SimpleNamespace
    from DerivationReconstruction import reconstruct
    from chunk_fixtures import from_stack

    # A negative publication depth identifies an unfinished sentence. The
    # physical grammar stack can retain more than the three sentence slots.
    values = torch.arange(count * 2, dtype=torch.float32).reshape(1, count, 2).requires_grad_()
    tags = torch.zeros(1, count, dtype=torch.long)
    stm = from_stack(values, torch.tensor([count]), tags, tags, torch.full_like(tags, -1), torch.ones_like(tags).float())
    lang = SimpleNamespace()
    program = SimpleNamespace(actions=torch.tensor([[0, -1, i] for i in range(count)]))
    record = SimpleNamespace(word_values=torch.full_like(values, .25), word_valid=torch.ones_like(tags, dtype=torch.bool),
                             end_slots=values[:, :3], end_depth=torch.tensor([-1]))
    recovered, missing, defined = reconstruct(lang, record, [program], lambda _: range(count), stm)
    assert defined.all() and not missing.any()
    torch.testing.assert_close(recovered, values.flip(1))
    gradient, = torch.autograd.grad((recovered - record.word_values).square().sum(), values)
    torch.testing.assert_close(gradient, 2 * (values - .25))
    record.end_depth = torch.tensor([1])
    with pytest.raises(ValueError, match='forest depth'):
        reconstruct(lang, record, [program], lambda _: range(count), stm)
