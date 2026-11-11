"""The existing unseeded XOR and packing assertions under all three settings."""
import torch
import pytest
from test_one_backward import receipt,checkpoint_without_random_state


@pytest.mark.parametrize('separate,jitter',[(True,0.),(False,0.),(False,.5)])
@pytest.mark.parametrize('pool',[4,8])
@pytest.mark.parametrize('trial',range(3))
def test_grounded_xor_with_sentence_backward(tmp_path,separate,jitter,pool,trial):
    from test_grounded_xor import learn_grounded_xor
    from Layers import Error
    calls=[]
    def backward(model,optimizer,loss):
        if separate:
            # The standing gate supplies named primitive/definition teaching,
            # not a competing reconstruction or sentence chooser objective.
            loss.backward()
        else:
            registry=getattr(model,'_sentence_close_errors',None)
            try:
                model._sentence_close_errors=Error()
                model.errors.add('output.named_pole',loss,category='output')
                model._backward_training_loss(loss,optimizer=optimizer)
            finally:model._sentence_close_errors=registry
        calls.append(1)
    error=learn_grounded_xor(tmp_path,pool,separate_chooser=separate,
        reconstruction_jitter=jitter,backward=backward)
    assert len(calls)==64
    receipt(tmp_path,f'xor-{separate}-{jitter}-{pool}-{trial}',dict(
        separate_chooser=separate,reconstruction_jitter=jitter,pool=pool,trial=trial,
        mse=error,backwards=len(calls),seed=None))


@pytest.mark.parametrize('separate,jitter',[(True,0.),(False,0.),(False,.5)])
def test_native_packing_after_training_with_selected_backward(tmp_path,monkeypatch,eager_reading,separate,jitter):
    import gc
    from Layers import GrammarLayer
    from test_packed_reconstruction_parity import build_model,warm_vocabulary,measure_layout,reset,ATOL,RTOL
    from Language import OperationSelectionLayer
    from operation_fixtures import selected_action
    forward=OperationSelectionLayer.forward
    def compose(layer,value,**kwargs):
        refs=kwargs.get('reference_data')
        binary=tuple(range(layer.r_reduce)) if refs is None else refs['binary_ops']
        unary=tuple(range(layer.r_apply)) if refs is None else refs['unary_ops']
        stop=(value.shape[1]-1)*len(binary)+value.shape[1]*len(unary)
        depth=kwargs.get('depth',torch.full((len(value),),value.shape[1]))
        action=torch.where(depth>1,binary.index(layer.op_names.index('sum')),stop)
        with selected_action(layer,action):return forward(layer,value,**kwargs)
    monkeypatch.setattr(OperationSelectionLayer,'forward',compose)
    rows=[['9 plus 1','14 plus 1'],['2 plus 1']]
    first=build_model(tmp_path,separate_chooser=separate,reconstruction_jitter=jitter)
    models=[first]
    try:
        first.candidate_attention=None  # retained source-order control
        first._install_unit_span_fn()
        warm_vocabulary(first,rows)
        first.answer_synthesis = False  # This fixture's supplied labels use the numeric forward head.
        first.begin_presentation()
        optimizer=first.getOptimizer(lr=.001)
        # Training owns its actual sampled departure. The forced operation is
        # only the retained packing control, used before and after training.
        monkeypatch.setattr(OperationSelectionLayer,'forward',forward)
        try:
            first.runBatch(train=True,batchSize=1,optimizer=optimizer,
                batch_override=(first.inputSpace.prepInput(['9 plus 1']),torch.empty(1,0)))
        except RuntimeError:
            receipt(tmp_path,f'packing-training-failure-{separate}-{jitter}',dict(
                terms={name:dict(category=term['category'],objective=term.get('objective'),
                    trained=term.get('trained',True),weight=term['weight'],
                    requires_grad=first.errors._value(term).requires_grad,
                    value=first.errors._value(term).detach().tolist())
                    for name,term in first.errors._terms.items()},
                sentence_steps=first._single_sentence_steps))
            raise
        if not separate:assert first._single_backward_calls==1 and first._single_sentence_steps==1
        assert first._sentence_reader_updates > 0
        reset(first,packed=False,final=True,batch=1)
        checkpoint=tmp_path/'trained-parity.ckpt'
        first._optimizer=optimizer
        first.save_weights(checkpoint)
        checkpoint_without_random_state(checkpoint)
        first.End();first.symbolSpace.soft_reset();models.clear()
        GrammarLayer.set_chart_authority(None)
        first=None;optimizer=None
        gc.collect()
        monkeypatch.setattr(OperationSelectionLayer,'forward',compose)
        results=[]
        for packed in (True,False):
            # Both layouts begin at the same restored training boundary,
            # including caches rebuilt by checkpoint admission.
            model=build_model(tmp_path,separate_chooser=separate,reconstruction_jitter=jitter)
            models.append(model)
            model.candidate_attention=None
            assert model.load_weights(checkpoint,require_match=True)
            model._install_unit_span_fn()
            model.begin_presentation()
            results.append(measure_layout(model,rows,packed=packed))
            model.End();model.symbolSpace.soft_reset()
            GrammarLayer.set_chart_authority(None)
            models.clear()
            del model
            gc.collect()
        packed,single=results
        for key,actual in packed.items():
            expected=single[key]
            torch.testing.assert_close(actual['truncated'],expected['truncated'],rtol=0,atol=0)
            for name in ('reference','actions','content_root','recovered','byte_cost'):
                torch.testing.assert_close(actual[name],expected[name],atol=ATOL,rtol=RTOL,
                    msg=lambda message:f'{key} {name}: {message}')
            torch.testing.assert_close(actual['program_root'],actual['root'],rtol=0,atol=0)
        receipt(tmp_path,f'packing-{separate}-{jitter}',dict(separate_chooser=separate,
            reconstruction_jitter=jitter,sentences=len(packed),packing_parity=True,
            initialization='one ambient checkpoint after one training step',seed=None))
    finally:
        for model in models:model.End();model.symbolSpace.soft_reset()
        torch._dynamo.reset()
