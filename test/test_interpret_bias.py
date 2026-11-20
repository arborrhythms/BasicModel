"""Rule 5b. An offered word can be interpreted independently of policy."""
import pytest
import torch


def explore_configuration(tmp_path):
    """The whole-paths benchmark's configuration (a receipt under doc/), less
    the key conceptualPi that sigma-pi §21, rule 6 removed: a basic percept
    is a row of the first Pi in every model."""
    from pathlib import Path
    source = Path('doc/benchmarks/2026-11-11-whole-paths/configurations/explore-1.xml').read_text()
    path = tmp_path / 'explore-1.xml'
    path.write_text(source.replace('<conceptualPi>true</conceptualPi>', ''))
    return str(path)


def test_fresh_reader_interprets_each_word_once(monkeypatch, eager_reading, tmp_path):
    from Corpus import admit_training_forms
    import ModelAttention
    from Models import BaseModel
    from SentenceAttention import SentenceField
    from SentenceLooks import Looks
    from data import TheData
    text = 'red cat sleeps'
    TheData.processLM({s: dict(text=[text] if s == 'train' else [], label=[])
                       for s in ('train', 'validation', 'test')})
    model = BaseModel.from_config(
        explore_configuration(tmp_path), data=TheData)[0]
    admit_training_forms(model, [text])
    model.eval()
    bricks, words, offers, calls = [], [], [], []
    observe, remember, run = ModelAttention.observe_percept, SentenceField.remember, Looks.run

    def brick(m, forecast, payload, sub, index, active):
        bricks.append(int(index))
        return observe(m, forecast, payload, sub, index, active)

    def word_step(field, index, rows, language, *, closing=False):
        if not closing and bool(rows.any()):
            words.append(int(index))
        return remember(field, index, rows, language, closing=closing)

    def observe_walk(path, *args, **kwargs):
        result = run(path, *args, **kwargs)
        calls.append(len(path.walk.state.actions))
        offers.extend(int(d['index'][0]) for d in path.decisions if torch.isfinite(d['logits'][0, 1]))
        return result

    monkeypatch.setattr(ModelAttention, 'observe_percept', brick)
    monkeypatch.setattr(SentenceField, 'remember', word_step)
    monkeypatch.setattr(Looks, 'run', observe_walk)
    try:
        with torch.no_grad():
            model.understand(model.inputSpace.prepInput([text]))
        assert sorted(words) == [0, 2, 4]
        assert bricks == words == offers
        assert len(calls) == 1 and calls[0] < model.attention_budget
    finally:
        model.End()


def chooser():
    from Language import OperationSelectionLayer
    op = torch.nn.Identity()
    op.rule_name = 'interpret'
    model = OperationSelectionLayer(d_model=2, unary_ops=[op])
    return model


def test_zero_preserves_policy_and_rng():
    model = chooser()
    model.interpret_bias = 0.
    word = torch.ones(3, 2)
    offer = torch.tensor([True, False, True])
    before = torch.get_rng_state()
    old = model.choose_interpret(word, offer)
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    new = model.interpret_at_look(word, offer)
    assert torch.equal(torch.get_rng_state(), after)
    for key in old:
        torch.testing.assert_close(new[key], old[key], atol=0, rtol=0)


def test_partial_bias_draws_only_on_offers_and_replays(monkeypatch):
    model = chooser()
    model.interpret_bias = .5
    word = torch.zeros(4, 2)
    allowed = torch.tensor([True, False, True, True])
    draws, scored = [], []
    def uniforms(count, *, device):
        draws.append(count)
        return torch.tensor([.2, .5, .1], device=device)
    monkeypatch.setattr(torch, 'rand', uniforms)
    policy = model.choose_interpret
    def score(keys, offered, **opts):
        scored.append(len(keys))
        return policy(keys, offered, **opts)
    monkeypatch.setattr(model, 'choose_interpret', score)
    d = model.interpret_at_look(word, allowed)
    assert draws == [3] and scored == [1]
    assert d['action'].tolist() == [1, 0, 0, 1]
    assert d['eligible'].tolist() == [False, False, True, False]
    assert d['alternatives'].tolist() == [0, 0, 1, 0]
    replay = model.interpret_at_look(word, allowed, replay_bias=d['biased'].long(), replay_action=d['action'])
    assert draws == [3] and torch.equal(d['action'], replay['action'])
    for bias in (0., 1.):
        model.interpret_bias = bias
        decision = model.interpret_at_look(word, allowed)
        assert draws == [3]
        if bias:
            assert decision['interpreted'].equal(allowed) and not decision['eligible'].any()
            assert not decision['probability'].requires_grad


@pytest.mark.parametrize('value', [-.001, 1.001, float('nan')])
def test_construction_rejects_invalid_bias(value, monkeypatch):
    from util import TheXMLConfig
    get = TheXMLConfig.get
    monkeypatch.setattr(TheXMLConfig, 'get', lambda key, *a, **kw:
                        value if key == 'architecture.interpretBias' else get(key, *a, **kw))
    with pytest.raises(ValueError, match='interpretBias'):
        chooser()
