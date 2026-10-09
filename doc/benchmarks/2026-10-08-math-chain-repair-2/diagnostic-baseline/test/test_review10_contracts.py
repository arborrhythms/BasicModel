"""Review §10: departures sample, and frozen interpretation cannot admit."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch


def high_draw(*shape, **kwargs):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return torch.full(shape, .999, **kwargs)


@pytest.mark.parametrize('kind', ['compose', 'input'])
def test_departure_can_choose_below_second_rank_and_preserves_credit(monkeypatch, kind):
    from Language import OperationSelectionLayer
    chooser = OperationSelectionLayer(d_model=2)
    logits = torch.tensor([[3., 2., 1., -torch.inf], [3., 2., 1., -torch.inf]], requires_grad=True)
    monkeypatch.setattr(torch, 'rand', high_draw)
    action, probability, legal = chooser.select_logits(logits, structural=None,
        masked_action=torch.tensor([0, -1]), replay_action=torch.tensor([-1, -1]))
    assert action.tolist() == [2, 0]
    assert legal.all()
    torch.testing.assert_close(probability, logits.softmax(-1))
    probability[0, 2].backward()
    assert logits.grad[0, 0] != 0  # exclusion changes the draw, not policy credit


@pytest.mark.parametrize('kind', ['thought', 'anticipation'])
def test_thought_departure_samples_remaining_actions_and_replay_wins(monkeypatch, kind):
    from Language import SelectedThoughtChooser
    chooser = SelectedThoughtChooser(context_dim=2)
    logits = torch.tensor([3., 2., 1., -torch.inf], requires_grad=True)
    monkeypatch.setattr(chooser, 'logits', lambda *a: logits)
    monkeypatch.setattr(torch, 'rand', high_draw)
    selected, credit = chooser.choose(torch.ones(4, 2), torch.zeros(4), excluded=0)
    assert selected == 2
    torch.testing.assert_close(credit, logits.log_softmax(-1)[2])
    replayed, _ = chooser.choose(torch.ones(4, 2), torch.zeros(4), excluded=0, forced=1)
    assert replayed == 1


def test_decoder_departure_samples_lower_rank_and_keeps_greedy_suffix(monkeypatch):
    from test_decoder_exploration import decoder
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _ = decoder(monkeypatch)
    monkeypatch.setattr(torch, 'rand', high_draw)
    event = torch.tensor([[[1., 2.], [0., 0.], [0., 0.]]])
    greedy = model._output_generate_walk(event, 4, basis=event, return_candidates=True)
    explored = model._output_generate_walk(event, 4, basis=event,
        exploit_actions=greedy[4], departure=torch.tensor([0]), return_trace=True)
    assert greedy[4].tolist() == [[2, -1, -1, -1]]
    assert explored[4].tolist() == [[1, 2, -1, -1]]


def test_frozen_interpretation_reads_known_without_learning_or_reserving():
    from test_word_interpretation import _operator
    from Spaces import _concept_alloc_of
    from eval_nanochat_grammar import frozen_online_learning
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [1], form='known')
    obj = interpret.forward(word)
    pending = interpret.lookup_word([8], [], form='pending')
    before = deepcopy(cs.definitions.description(word))
    identities = dict(_concept_alloc_of(cs).placement)
    definitions = len(cs.definitions._store())
    reservations = deepcopy(interpret._pending)
    model = SimpleNamespace(conceptualSpaces=[cs])
    with frozen_online_learning(model):
        assert interpret.lookup_word([9], [], form='novel') is None
        assert interpret.lookup_word([10], [2], form='known') == word
        assert interpret.forward(word, occurrence=(99, 0)) == obj
        assert interpret.forward(pending) is None
        assert interpret.define(pending, obj) is None
        assert len(cs.definitions._store()) == definitions
        assert dict(_concept_alloc_of(cs).placement) == identities
        assert interpret._pending == reservations
        assert cs.definitions.description(word) == before
    assert not cs._online_learning_frozen


def test_margin_probe_records_only_training_gradient_and_actual_policy_motion():
    from decoder_margin_probe import DecoderMarginProbe
    from test_decoder_exploration import decoder
    model, policy = decoder()
    parent = torch.tensor([[1., 2.]])
    logits = model.languageSpace.generate_policy_logits(parent)
    loss = -logits.log_softmax(-1)[0, 0]
    events = []
    probe = DecoderMarginProbe(lambda kind, **data: events.append(dict(kind=kind, **data)))
    rng = torch.random.get_rng_state().clone()
    probe.capture(model, logits, parent, torch.ones_like(logits, dtype=torch.bool),
        round=0, explore=True, live=torch.tensor([True]), metadata=dict(epoch=0, batch=0))
    gradient, = torch.autograd.grad(loss, logits, retain_graph=True)
    assert probe.active is None and not probe.ready
    optimizer = torch.optim.SGD(policy.parameters(), lr=.01, momentum=.9)
    probe.begin_backward()
    loss.backward()
    probe.end_backward()
    probe.before_step()
    optimizer.step()
    probe.after_step()
    step = events[-1]
    assert step['kind'] == 'decoder_margin_step' and step['step'] == 0
    walk, = step['walks']
    assert walk['gradient_calls'] == 1
    torch.testing.assert_close(torch.tensor(walk['gradient']), gradient)
    torch.testing.assert_close(torch.tensor(walk['stop_minus_undo_gradient']),
                               gradient[:, 2:3]-gradient[:, 0:1])
    assert walk['fixed_parent_margin_change'][0][0] < 0
    assert torch.equal(rng, torch.random.get_rng_state())


def test_small_frozen_evaluator_keeps_definition_rows_and_reservations():
    from Spaces import _concept_alloc_of
    import eval_nanochat_grammar as gate
    model, data = gate.build_eval_model('data/MM_grammar_wording.xml', autoload=False)
    manifest = gate.load_manifest(gate.DEFAULT_MANIFEST)
    def inventory():
        return [dict(words=cs.definitions.word_ids, objects=cs.definitions.object_ids,
            rows=deepcopy(cs.definitions._store()._definition_rows),
            pending=deepcopy(cs.interpret._pending),
            placement=dict(_concept_alloc_of(cs).placement), capacity=cs.definitions._store().capacity)
            for cs in model.conceptualSpaces]
    before = inventory()
    result = gate.score_manifest(model, data, manifest, limit=3)
    assert inventory() == before
    assert result['metrics']['items'] == 3 and result['metrics']['choices'] == 16
    for item in result['items']:
        assert torch.isfinite(torch.tensor(item['intact_scores']+item['shuffled_scores'])).all()
        assert item['intact_prediction_steps'] > 0 and item['shuffled_prediction_steps'] > 0
