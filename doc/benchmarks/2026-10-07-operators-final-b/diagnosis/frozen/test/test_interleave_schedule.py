"""Schedule retirement: one input, one opening, one retained owner."""
import pytest
import torch


@pytest.mark.parametrize('value',['','interleave:0','interleave:-1','interleave:1.5','mixed','parallel','serial','interleave:2'])
def test_schedule_rejected_for_all_old_values(value):
    from util import XMLConfig
    with pytest.raises(ValueError,match='modeSchedule'):
        XMLConfig._apply_legacy_renames({'architecture':{'modeSchedule':value}},'retired.xml')


def test_schedule_module_is_retired():
    import importlib.util
    assert importlib.util.find_spec('ModeSchedule') is None


@pytest.mark.parametrize('training',[False,True])
def test_one_native_read_retains_shared_owner(tmp_path,monkeypatch,training,eager_reading):
    import ModelAttention
    from test_compiled_word_chunk import _tiny_canonical_model
    from What import What
    model=_tiny_canonical_model(tmp_path,monkeypatch,word_buckets='8',
        concept_rows=128,input_width=16,batch_size=1,chooser_depth=1,
        training_overrides={'reconstructionPlacement':'eager'},
        architecture_overrides={'answerSynthesis':False})
    model._tensor_peer_while_eager=True
    model._chart_compose_per_word=lambda:None
    model.checkpoint_every_batches=0
    owner=model._concept_owner();cache=owner.similarity_codebook.W
    opt=model.getOptimizer(lr=1e-3) if training else None
    opens=[];original=ModelAttention.read_code_field
    def read(*args,**kwargs):
        assert not torch.is_grad_enabled();opens.append(True);return original(*args,**kwargs)
    monkeypatch.setattr(ModelAttention,'read_code_field',read)
    texts=('a b','a c')
    for step,text in enumerate(texts*(2 if training else 1)):
        raw=model.inputSpace.prepInput([text])
        model.runBatch(train=training,optimizer=opt,batchNum=step,batchSize=1,
            split='validation',batch_override=(raw,torch.empty(1,0)),
            questions=(What.present(0,split='validation'),))
    assert len(opens)==(4 if training else 2)
    assert model._training_step_count==(4 if training else 0)
    assert model._concept_owner() is owner
    assert owner.similarity_codebook.W is cache
    current=owner.similarity_codebook.getW()
    torch.testing.assert_close(current, cache, rtol=0, atol=0)
    assert not isinstance(cache, torch.nn.Parameter) and not cache.requires_grad
    assert not hasattr(model,'mode_schedule') and not hasattr(owner,'_label_feedback')
    model.End()
