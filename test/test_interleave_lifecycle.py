"""Open-bracket lifecycle replaces the removed schedule cursor.

Native packing, source addresses, resume and error handling remain model/
loader contracts. No private parallel-first replay queue survives.
"""
from functools import wraps
import pytest
import torch


@pytest.mark.parametrize('packed',[False,True])
def test_open_read_covers_exact_native_input(tmp_path,monkeypatch,packed,eager_reading):
    import ModelAttention
    from test_packed_reconstruction_parity import build_model
    model=build_model(tmp_path,word_capacity=16 if packed else 8)
    texts=['the wug sat','the wug flew'];seen=[]
    original=ModelAttention.read_code_field
    def read(codes,valid):
        assert not torch.is_grad_enabled()
        seen.append((codes.shape,valid.sum().item()))
        return original(codes,valid)
    monkeypatch.setattr(ModelAttention,'read_code_field',read)
    raw=model.inputSpace.prepPackedInput([texts]) if packed else model.inputSpace.prepInput(texts)
    with torch.no_grad():model(raw)
    assert len(seen)==1 and seen[0][0][0]==(1 if packed else 2)
    assert int(model._attention_words.accepted.sum())==6
    assert not hasattr(model,'mode_schedule')


def test_epoch_keeps_the_short_final_batch_once(tmp_path,monkeypatch,eager_reading):
    import ModelAttention
    from test_packed_reconstruction_parity import build_model
    model=build_model(tmp_path,word_capacity=8);seen=[]
    original=ModelAttention.stage_input
    def read(owner):
        seen.extend(owner.inputSpace._last_sentences);return original(owner)
    monkeypatch.setattr(ModelAttention,'stage_input',read)
    texts=['the wug sat','the wug flew','the wug ran']
    with model.inputSpace.data.runtime_batch(texts):model.runEpoch(None,batchSize=1,split='runtime')
    assert seen==texts


def test_checkpoint_has_no_pending_attention_queue(tmp_path,monkeypatch,eager_reading):
    from test_packed_reconstruction_parity import build_model
    model=build_model(tmp_path,word_capacity=8)
    with torch.no_grad():model(model.inputSpace.prepInput(['the wug sat']))
    checkpoint=tmp_path/'attention.ckpt';model.save_weights(checkpoint)
    state=torch.load(checkpoint,weights_only=False,map_location='cpu')
    assert 'mode_schedule' not in state.get('training_state',{})
    assert not any('mode_schedule' in key for key in state['state_dict'])
    target=tmp_path/'target';target.mkdir();restored=build_model(target,word_capacity=8)
    assert restored.load_weights(checkpoint,strict=True,require_match=True)
    assert not hasattr(restored,'mode_schedule')
    with torch.no_grad():restored(restored.inputSpace.prepInput(['the wug flew']))
    assert int(restored._attention_words.accepted.sum())==3


def test_failed_open_read_leaves_the_input_and_teacher_address(tmp_path,monkeypatch,eager_reading):
    import ModelAttention
    from test_packed_reconstruction_parity import build_model
    model=build_model(tmp_path,word_capacity=8)
    raw=model.inputSpace.prepInput(['the wug sat']);saved=raw.clone()
    model.teacher.stage_batch_sources('validation',[0])
    def fail(*args,**kwargs):raise RuntimeError('open failed')
    original=ModelAttention.read_code_field
    monkeypatch.setattr(ModelAttention,'read_code_field',fail)
    with pytest.raises(RuntimeError,match='open failed'):model(raw)
    torch.testing.assert_close(raw,saved,rtol=0,atol=0)
    assert model.teacher._staged_source_rows==[0]
    monkeypatch.setattr(ModelAttention,'read_code_field',original)
    with torch.no_grad():model(raw)
    assert model._attention_words.accepted.sum()==3


@pytest.mark.parametrize('packed',[False,True])
def test_input_encoder_bytes_and_packed_sentence_boundaries(tmp_path,packed,eager_reading):
    from test_packed_reconstruction_parity import build_model
    model=build_model(tmp_path,word_capacity=16)
    texts=['abc1 cat','dog 01']
    raw=model.inputSpace.prepPackedInput([texts]) if packed else model.inputSpace.prepInput(texts)
    with torch.no_grad():model(raw)
    spans=model._attention_spans;accepted=model._attention_words.accepted
    assert int(accepted.sum())==6
    assert ((spans[...,1]>spans[...,0])|~accepted).all()
    if packed:
        ids=model.inputSpace._packed_sentence_ids
        assert ids[accepted].tolist()==[0,0,0,1,1,1]
    encoded=model.inputSpace.prepInput(['éééé'])
    assert encoded.flatten()[:4].tolist()==[63,63,63,63]
