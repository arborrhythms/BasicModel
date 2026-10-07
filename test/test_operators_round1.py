"""Decided round-one contracts; fixed examples never set a training seed."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch


@pytest.mark.parametrize('advantage', [-2., 0., 3.])
def test_attention_score_function_has_greedy_baseline_and_uniform_counts(advantage):
    from SentenceCredit import score_function
    logits = torch.tensor([[.2, .6, -.1]], dtype=torch.float64, requires_grad=True)
    probability = logits.softmax(-1)
    loss, record = score_function(probability[:,1:2], torch.tensor([[6]]),
        torch.tensor([[True]]), torch.tensor([[4.,4.+advantage]]))
    assert record['advantage'].item() == advantage
    if advantage == 0:
        assert not loss.requires_grad and loss.item() == 0
        return
    gradient, = torch.autograd.grad(loss.sum(), logits)
    p = probability.detach()[0]
    expected = -p * p[1]; expected[1] += p[1]
    torch.testing.assert_close(gradient[0], expected * 6 * advantage)
    epsilon = 1e-5
    plus, minus = logits.detach().clone(), logits.detach().clone()
    plus[0, 1] += epsilon; minus[0, 1] -= epsilon
    numerical = 6 * advantage * (plus.softmax(-1)[0, 1] - minus.softmax(-1)[0, 1]) / (2 * epsilon)
    torch.testing.assert_close(numerical, gradient[0, 1])


def test_attention_uniform_departure_and_no_pathwise_credit(monkeypatch):
    from Language import OperationSelectionLayer
    chooser = OperationSelectionLayer(d_model=2, chooser='mlp')
    keys = torch.tensor([[[1., 0.]]], requires_grad=True)
    legal = torch.tensor([[[True, True, True, False, False, False]]])
    seen = []
    import Language
    original = Language.sample_eligible_logits
    def proposal(logits, *args):
        seen.append(logits.detach())
        return original(logits, torch.tensor([[.75]]))
    monkeypatch.setattr(Language, 'sample_eligible_logits', proposal)
    action, credit, _, details = chooser.attend(keys, legal, torch.zeros(1, 1, dtype=torch.long),
        masked_action=torch.tensor([0]), return_details=True)
    assert action.item() in (1, 2) and details['alternative_count'].item() == 2
    assert torch.equal(seen[0][torch.isfinite(seen[0])], torch.zeros(2))
    assert not credit.requires_grad
    details['probability'].sum().backward()
    assert keys.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in chooser.parameters())


def test_saved_review22_miss_has_exact_argmin_precedence():
    from Language import ConjunctionLayer, LanguageSpace
    from DecompositionChooser import DecompositionChooser
    path = Path(__file__).resolve().parents[1] / 'doc/benchmarks/2026-10-05-operators-update/review22-miss.json'
    saved = json.loads(path.read_text())
    bank = torch.tensor(saved['codes'])[None].expand(4, -1, -1)
    roots = torch.tensor(saved['roots'])
    chooser = DecompositionChooser()
    with torch.no_grad():
        chooser.weight.copy_(torch.tensor(saved['weights']))
        chooser.weight[1:] *= 1000  # Even extreme context cannot outvote an exact fit.
    left, right, available, details = LanguageSpace._bounded_binary_reconstruction(
        ConjunctionLayer(), roots, torch.zeros_like(roots), torch.zeros(4, dtype=torch.bool),
        torch.zeros(4, dtype=torch.bool), bank, torch.ones(4, 4, dtype=torch.bool), 4,
        chooser=chooser, return_details=True)
    assert available.all() and details['exact'].all()
    rows = torch.tensor(saved['rows'])
    indices = details['selected']
    recovered = torch.stack((rows[indices // 4], rows[indices % 4]), -1).sort(-1).values
    assert recovered.tolist() == [[0, 1], [0, 2], [1, 4], [2, 4]]
    torch.testing.assert_close(ConjunctionLayer().compose(left, right), roots)


def test_activation_features_standardized_only_over_valid_candidates():
    from test_decomposition_chooser import search
    from DecompositionChooser import DecompositionChooser
    bank = torch.tensor([[[.2, .8], [.7, .3], [.8, .4]]])
    details = search(bank, torch.tensor([[.12, .35]]), DecompositionChooser(),
        left_valid=torch.tensor([[True, True, False]]))[3]
    feature = details['features'][0, :2, 0, 1]
    torch.testing.assert_close(feature.mean(), torch.tensor(0.), atol=1e-6, rtol=0)
    torch.testing.assert_close(feature.square().mean(), torch.tensor(1.))
    assert details['features'][0, 2, 0, 1] == 0


def test_walk_teacher_covers_binary_unary_and_leaf_stop_without_code_gradient():
    from Models import BasicModel
    from Language import LanguageSpace
    leaves = torch.tensor([[.2, .7], [.4, .3]], requires_grad=True)
    values = torch.zeros(4, 3, 2, requires_grad=True)
    policy = torch.nn.Linear(2, 3)
    language = SimpleNamespace(_generate_binary_ops=(object(),), _generate_unary_ops=(object(),),
        generate_policy=policy, _generate_rule_keys=torch.tensor([20, 10, 0]),
        _compose_binary_rules=(20,), _compose_unary_rules=(10,),
        _generate_rule_key=lambda rule, arity: rule, generate_policy_logits=policy)
    program = SimpleNamespace(leaves=leaves, operation_values=values,
        actions=torch.tensor([[0, -1, 0], [2, 0, -1], [0, -1, 1], [1, 0, -1]]))
    model = SimpleNamespace(languageSpace=language)
    loss = BasicModel._decomposition_walk_teacher_loss(model, {'entries': [program]},
        SimpleNamespace(root=leaves[:1]))
    assert [r['target'] for r in model._last_decomposition_walk_teacher] == [2, 1, 2, 0]
    loss.sum().backward()
    assert policy.weight.grad is not None and policy.bias.grad.abs().sum() > 0
    assert leaves.grad is None and values.grad is None


def test_pole_operators_preserve_identity_and_keep_both_and_neither():
    from Language import NotLayer, NonLayer, ConjunctionLayer
    p = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.], [.7, .2]])
    q = p.roll(1, 0)
    negate, withdraw = NotLayer(representation='poles'), NonLayer(representation='poles')
    conjunction = ConjunctionLayer(representation='poles')
    torch.testing.assert_close(negate(p), p.flip(-1))
    torch.testing.assert_close(withdraw(p), torch.stack((torch.zeros(5), p[:, 1]), -1))
    torch.testing.assert_close(conjunction(p, q), torch.stack((torch.minimum(p[:, 0], q[:, 0]),
                                                            torch.maximum(p[:, 1], q[:, 1])), -1))
    assert not torch.equal(conjunction(p, q), conjunction(negate(p), q))
    code = torch.tensor([[.6, -.2, .3]])
    assert torch.equal(NotLayer()(code), code) and torch.equal(NonLayer()(code), code)


@pytest.mark.parametrize('attribute', ['footprintReads', 'footprintWrites'])
def test_footprint_mismatch_fails_at_load(attribute):
    from Language import Grammar
    with pytest.raises(ValueError, match='footprint'):
        Grammar()._fill_rule_list([], {'rule': dict(_='S = not.forward(S)', **{attribute: 'form'})})


def test_footprint_requires_a_declared_subsystem_writer(monkeypatch):
    from Language import Grammar, GRAMMAR_LAYER_CLASSES, NotLayer
    class Broken(NotLayer):
        effect_writes = ('budget',)
    monkeypatch.setitem(GRAMMAR_LAYER_CLASSES, 'broken', Broken)
    with pytest.raises(ValueError, match='footprint writes poles'):
        Grammar()._fill_rule_list([], {'rule': 'S = broken.forward(S)'})


@pytest.mark.parametrize('kind', ['verb', 'adverb'])
def test_modifier_can_write_silent_coordinate_and_reverse_with_witness(kind):
    from Language import VerbLayer, AdverbLayer
    op = (VerbLayer if kind == 'verb' else AdverbLayer)(nInput=3, nOutput=3)
    shift = op._verb_shift if kind == 'verb' else op._adv_shift
    operand = torch.tensor([[.2, 0., -.3]], requires_grad=True)
    modifier = torch.tensor([[0., 1., 0.]])
    with torch.no_grad():
        shift[1, 1] = .25
    actual = op.compose(operand, modifier)
    assert actual[0, 1] > 0
    recovered, _ = op.reverse(actual, **{('verb_what' if kind == 'verb' else 'adverb_what'): modifier})
    torch.testing.assert_close(recovered, operand, atol=1e-6, rtol=1e-6)
    actual.sum().backward()
    assert shift.grad[1, 1] != 0


def test_identity_unary_cannot_exhaust_free_numeric_decoder(monkeypatch):
    from test_decoder_exploration import decoder
    model, policy = decoder(monkeypatch)
    model.languageSpace.generate_unary_step = lambda parent, *a, **k: (
        parent, torch.zeros(parent.shape[0], dtype=torch.bool))
    with torch.no_grad():
        policy.bias.copy_(torch.tensor([-1., 10., 2.]))
    event = torch.tensor([[[1., 2.], [0., 0.]]])
    output, count, truncated, _, actions = model._output_generate_walk(event, 8,
        require_symbols=False, return_trace=True)
    assert count.tolist() == [1] and not truncated.any()
    assert actions[0, 0] == 2
    torch.testing.assert_close(output[:, 0], event[:, 0])


def test_shipped_rule_footprints_and_alias_are_exhaustive():
    from Language import Grammar
    from AccessibleMind import OperatorFootprint, OperatorEffects
    grammar = Grammar(); grammar.load_from_grammar_file('complete.grammar')
    for rule in grammar.rules + grammar.thought_rules + grammar.ps_rules:
        OperatorFootprint(rule.footprint_reads, rule.footprint_writes).check_effects(
            OperatorEffects(rule.effect_reads, rule.effect_writes))
    rules = []
    grammar._fill_rule_list(rules, {'rule': dict(_='S = alias.forward(S)', implementation='not',
        footprintReads='poles', footprintWrites='poles')})
    assert rules[0].footprint_reads == rules[0].footprint_writes == ('poles',)


@pytest.mark.parametrize('kind', ['verb', 'adverb'])
def test_gain_only_checkpoint_loads_with_zero_translation(kind):
    from Language import VerbLayer, AdverbLayer
    cls = VerbLayer if kind == 'verb' else AdverbLayer
    old, restored = cls(nInput=3, nOutput=3), cls(nInput=3, nOutput=3)
    state = {k: v for k, v in old.state_dict().items() if not k.endswith('_shift')}
    restored.load_state_dict(state, strict=True)
    x, y = torch.tensor([[.3, 0., -.2]]), torch.tensor([[.1, .8, .2]])
    torch.testing.assert_close(restored.compose(x, y), old.compose(x, y), atol=0, rtol=0)


def test_answer_reader_covers_later_roles_after_pole_only_exclusion():
    """A zero first role must not silence the remaining answer at initialization."""
    from Layers import LDUReadout
    state = torch.random.get_rng_state().clone()
    reader = LDUReadout(48, 4)
    assert torch.equal(torch.random.get_rng_state(), state)
    value = torch.zeros(2, 48, requires_grad=True)
    with torch.no_grad():
        value[0, 16] = .25
        value[1, 32] = .75
    output = reader(value)
    assert not torch.equal(output[0], output[1])
    gradient, = torch.autograd.grad(output.sum(), value)
    assert bool(gradient.ne(0).all())
