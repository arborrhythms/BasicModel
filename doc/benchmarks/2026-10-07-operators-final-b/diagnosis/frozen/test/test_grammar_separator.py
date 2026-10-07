"""A mixing grammar reads objects; whitespace remains a perceived byte unit."""
import inspect
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize('unadmitted', [False, True])
@pytest.mark.parametrize('trailing', [False, True])
def test_mixing_separator_is_perceived_but_not_pushed_or_read_back(monkeypatch, unadmitted, trailing):
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
    # This separator fixture supplies the same binary form operation to
    # composition and its fixed inverse below. An arbitrary fresh unary
    # can erase a word before the separator boundary is exercised.
    layer = model.languageSpace._tree_layer(2)
    binary_scores = layer.chooser.score_binary
    unary_scores = layer.chooser.score_unary
    compose_names = [rule.method_name for rule in model.languageSpace._compose_binary_rules]
    def compose_binary(*args, **kwargs):
        stop, scores = binary_scores(*args, **kwargs)
        bias = scores.new_full((layer.r_reduce,), -1e6)
        bias[compose_names.index('conjunction')] = 1e6
        return stop, scores + bias
    def compose_unary(*args, **kwargs):
        stop, scores = unary_scores(*args, **kwargs)
        if kwargs.get('op_offset') == layer.r_reduce + layer.r_apply:
            return stop, scores  # attention keeps its own policy
        return stop, scores - 1e6
    monkeypatch.setattr(layer.chooser, 'score_binary', compose_binary)
    monkeypatch.setattr(layer.chooser, 'score_unary', compose_unary)
    if unadmitted:
        stage = model._stage_reading_grammar_references
        def without_first_identity(*args):
            stage(*args)
            model.inputSpace._ar_grammar_object_rows[:, 0] = -1
        monkeypatch.setattr(model, '_stage_reading_grammar_references', without_first_identity)
    # Independent generation has no compose stamp to replay. Fix one legal
    # inverse then STOP so this separator proof still checks two emitted words.
    walk = model._output_generate_walk
    def two_words(*args, **kwargs):
        language = model.languageSpace
        policy = language.generate_policy_logits
        count = [0]
        choice = list(language._generate_binary_names).index('conjunction')
        def logits(top):
            value = top.new_full((top.shape[0], language.generate_policy.out_features), -1000.)
            value[:, choice if count[0] == 0 else -1] = 0.
            count[0] += 1
            return value
        language.generate_policy_logits = logits
        try:
            return walk(*args, **kwargs)
        finally:
            language.generate_policy_logits = policy
    monkeypatch.setattr(model, '_output_generate_walk', two_words)
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
    word_cost = model._byte_word_cost
    def observed_cost(idea, word, bank_n, bank_bytes, bank_valid,
                      target_bytes, target_valid, ready, *, priming=None):
        if ready:
            targets.append(target_bytes.detach().clone())
        return word_cost(idea, word, bank_n, bank_bytes, bank_valid,
                         target_bytes, target_valid, ready, priming=priming)
    monkeypatch.setattr(model, '_byte_word_cost', observed_cost)
    try:
        texts = ['hello world', 'hello there', 'loving world', 'loving there']
        if trailing:
            texts = [text + ' ' for text in texts]
        with torch.no_grad():
            model(model.inputSpace.prepInput(texts))
        isp = model.inputSpace
        assert isp._word_active_mask[:, :3].tolist() == [[True, True, True]] * 4
        assert isp._word_active_mask.sum(-1).tolist() == [4 if trailing else 3] * 4
        assert model.where_registry.slices['input'][1] == (
            max(model.inputSpace.data.inputLength,
                int(model.perceptualSpace.outputShape[0]) * model.serial_residual_part_capacity))
        assert torch.equal(isp._ar_embedded[:, 1], isp._ar_embedded_N[:, 1])
        assert bool(isp._ar_embedded[:, 1].abs().any())
        assert targets
        for b in range(4):
            assert bytes(isp._ar_target_word_bytes[b, 1][isp._ar_target_word_mask[b, 1]].tolist()) == b' '
            expected=isp._ar_target_word_bytes[b, isp._ar_grammar_leaf_mask[b]]
            torch.testing.assert_close(targets[-1][b,:len(expected)],expected,rtol=0,atol=0)
            assert not bool(targets[-1][b,len(expected):].any())
        assert [p for p, _ in pushes] == list(range(4 if trailing else 3))
        expected = [[True]*4, [False]*4, [True]*4] + ([[False]*4] if trailing else [])
        assert [gate.reshape(-1).tolist() for _, gate in pushes] == expected
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


def test_missing_word_text_row_uses_observed_bytes():
    from types import SimpleNamespace
    from Models import BasicModel
    active=torch.ones(1,3,dtype=torch.bool)
    raw=b'x y'
    model=SimpleNamespace(inputSpace=SimpleNamespace(_word_active_mask=active),
        perceptualSpace=SimpleNamespace(_forward_input=dict(word_texts=[None],
            part_spans=torch.tensor([[[0,1],[1,2],[2,3]]]))),
        _staged_concepts_in=torch.tensor([list(raw)]))
    BasicModel._stage_mixing_grammar_leaf_mask(model)
    assert model.inputSpace._ar_grammar_leaf_mask.tolist()==[[True,False,True]]
    assert model.inputSpace._word_active_mask.all()
