"""A mixing grammar reads objects; whitespace remains a perceived byte unit."""
import inspect
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize('unadmitted', [False, True])
def test_mixing_separator_is_perceived_but_not_pushed_or_read_back(monkeypatch, unadmitted):
    import Models
    import util
    from Layers import ShortTermMemory
    from test_mm_xor import _fresh_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    def eager_while(condition, body, values):
        while bool(condition(*values)):
            values = body(*values)
        return values
    monkeypatch.setattr(torch, 'while_loop', eager_while)
    model, _, data = _fresh_model(str(Path(Models.__file__).resolve().parents[1] / 'data/XOR_grammar.xml'))
    if unadmitted:
        stage = model._stage_reading_grammar_references
        def without_first_identity(*args):
            stage(*args)
            model.inputSpace._ar_grammar_object_rows[:, 0] = -1
        monkeypatch.setattr(model, '_stage_reading_grammar_references', without_first_identity)
    pushes, readings, targets = [], [], []
    push = ShortTermMemory.functional_push_step_masked
    def observed_push(*args):
        frame = inspect.currentframe().f_back
        if frame.f_code.co_name == 'stage_cs_lang':
            pushes.append((int(frame.f_locals['index']), args[7].detach().clone()))
        return push(*args)
    monkeypatch.setattr(ShortTermMemory, 'functional_push_step_masked', staticmethod(observed_push))
    commit = model._commit_sentence
    def observed_commit(state, sid, active, *rest):
        readings.append(model.reconstruct_grammar_sentence(state, sid, active))
        return commit(state, sid, active, *rest)
    monkeypatch.setattr(model, '_commit_sentence', observed_commit)
    event_loss = model._reverse_event_loss
    def observed_loss(pred, target, **kwargs):
        targets.append(target.detach().clone())
        return event_loss(pred, target, **kwargs)
    monkeypatch.setattr(model, '_reverse_event_loss', observed_loss)
    try:
        texts = ['hello world', 'hello there', 'loving world', 'loving there']
        with torch.no_grad():
            model(model.inputSpace.prepInput(texts))
            model._d3_reconstruction_loss()
        isp = model.inputSpace
        assert isp._word_active_mask[:, :3].tolist() == [[True, True, True]] * 4
        assert torch.equal(isp._ar_embedded[:, 1], isp._ar_embedded_N[:, 1])
        assert bool(isp._ar_embedded[:, 1].abs().any())
        assert targets and torch.equal(targets[-1], isp._ar_embedded)
        assert [p for p, _ in pushes] == [0, 1, 2]
        assert [gate.reshape(-1).tolist() for _, gate in pushes] == [[True]*4, [False]*4, [True]*4]
        assert len(readings) == 1
        decoded, unavailable = readings[0]
        assert all(len(text.split()) == 2 for text in decoded), decoded
        assert not bool(unavailable.any())
        assert model._word_symbol_concept_ids() is None
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


@pytest.mark.parametrize('carrier', ['tokens', 'word_texts', 'part_spans'])
def test_whitespace_classification_keeps_unknown_words_and_punctuation(carrier):
    from types import SimpleNamespace
    from Models import BasicModel
    forms = ['newword', ' ', '\t', '\n', ',', '\u00a0', '']
    active = torch.ones(1, len(forms), dtype=torch.bool)
    isp = SimpleNamespace(_word_active_mask=active,
                          _ar_grammar_object_rows=torch.full_like(active, -1, dtype=torch.long))
    raw = ''.join(forms).encode('utf-8')
    percepts = {carrier: [forms]}
    if carrier == 'part_spans':
        spans, lo = [], 0
        for form in forms:
            hi = lo + len(form.encode('utf-8'))
            spans.append((lo, hi))
            lo = hi
        percepts = dict(part_spans=torch.tensor([spans]))
    model = SimpleNamespace(inputSpace=isp, perceptualSpace=SimpleNamespace(_forward_input=percepts),
                            _staged_concepts_in=torch.tensor([list(raw)]))
    BasicModel._stage_mixing_grammar_leaf_mask(model)
    assert isp._ar_grammar_leaf_mask.tolist() == [[True, False, False, False, True, False, True]]
    assert bool(isp._word_active_mask.all())
