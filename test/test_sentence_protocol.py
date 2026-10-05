"""The former pump protocol is the unconditional no-gradient open bracket."""
import pytest
import torch


def _make_model():
    from test_mm_xor import _fresh_model, _PROJECT
    from pathlib import Path
    return _fresh_model(str(Path(_PROJECT)/'data/MM_xor_loopback.xml'))[0]


@pytest.mark.usefixtures('eager_reading')
@pytest.mark.parametrize('config',['MM_xor.xml','MM_xor_loopback.xml'])
def test_open_read_precedes_every_native_word(config,monkeypatch):
    from test_mm_xor import _fresh_model, _PROJECT
    from pathlib import Path
    import ModelAttention
    model=_fresh_model(str(Path(_PROJECT)/'data'/config))[0]
    events=[];original=ModelAttention.read_code_field
    def read(*args,**kwargs):
        assert not torch.is_grad_enabled();events.append('open')
        return original(*args,**kwargs)
    monkeypatch.setattr(ModelAttention,'read_code_field',read)
    admit=model._stage_reading_word_concepts
    def words():
        assert events==['open'];events.append('words');return admit()
    monkeypatch.setattr(model,'_stage_reading_word_concepts',words)
    model._lex_embed_stem(model.inputSpace.prepInput(['hello world']))
    assert events==['open','words']
    assert not model._open_read.requires_grad and not model._last_gist.requires_grad


@pytest.mark.usefixtures('eager_reading')
def test_open_read_has_no_optimizer_or_parameter_write(monkeypatch):
    import ModelAttention
    model=_make_model();original=ModelAttention.read_code_field
    def read(*args,**kwargs):
        before=[p.detach().clone() for p in model.parameters()]
        result=original(*args,**kwargs)
        assert all(torch.equal(a,b) for a,b in zip(before,model.parameters()))
        assert not result.requires_grad
        return result
    monkeypatch.setattr(ModelAttention,'read_code_field',read)
    model._lex_embed_stem(model.inputSpace.prepInput(['hello world']))


@pytest.mark.usefixtures('eager_reading')
def test_no_mid_sentence_reopening_even_under_conflict(monkeypatch):
    import ModelAttention
    model=_make_model();truth=model._get_truth_layer()
    hot=torch.zeros(int(truth.nDim));hot[0]=.9
    truth.record(hot,degree=1.);truth.record(-hot,degree=1.)
    assert truth.conflict_mass()==pytest.approx(.9,abs=1e-5)
    calls=[];original=ModelAttention.read_code_field
    def read(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    monkeypatch.setattr(ModelAttention,'read_code_field',read)
    model(model.inputSpace.prepInput(['hello world']))
    assert len(calls)==1


@pytest.mark.usefixtures('eager_reading')
def test_repeated_word_is_not_reminted():
    model=_make_model();model.eval()
    with torch.no_grad():
        model(model.inputSpace.prepInput(['hello hello']))
        owner=model._concept_owner();first=owner.definitions.word(form='hello')
        assert first is not None
        model.End();model(model.inputSpace.prepInput(['hello hello']))
    assert owner.definitions.word(form='hello')==first
    assert not model._attention_words.descended.any()
