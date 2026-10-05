import torch

def test_normal_narrowing_reads_native_poles(monkeypatch):
    from test_mm_xor import _fresh_model
    import ModelAttention
    model, _, _ = _fresh_model()
    model.eval()
    value = model.inputSpace.prepInput(['hello world', 'hello there', 'loving world', 'loving there'])
    with torch.no_grad(): model.forward(value)
    calls=[]
    original=ModelAttention.narrow_words
    def traced(*args, **kwargs):
        calls.append(kwargs.get('poles'))
        return original(*args, **kwargs)
    monkeypatch.setattr(ModelAttention, 'narrow_words', traced)
    with torch.no_grad(): model.forward(value)
    assert calls and all(torch.is_tensor(poles) for poles in calls), 'normal narrowing discards native paired evidence'
