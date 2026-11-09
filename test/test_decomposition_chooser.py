"""The fixed inverse selects the joint residual without learning or RNG."""
import torch
import pytest
from itertools import permutations
from DecompositionChooser import DecompositionChooser
from Language import LanguageSpace


class Product:
    @staticmethod
    def compose(left, right):
        return left * right


def search(bank, parent, chooser=None, **kwargs):
    b = len(bank)
    return LanguageSpace._bounded_binary_reconstruction(Product(), parent,
        torch.zeros_like(parent), torch.zeros(b, dtype=torch.bool),
        torch.zeros(b, dtype=torch.bool), bank,
        torch.ones(bank.shape[:2], dtype=torch.bool), bank.shape[1],
        chooser=chooser, return_details=True, **kwargs)


def test_initial_choice_is_exactly_the_residual_argmin_without_rng():
    bank = torch.tensor([[[.2, .8], [.7, .3], [.8, .4]]])
    parent = torch.tensor([[.14, .24]])
    state = torch.random.get_rng_state()
    chooser = DecompositionChooser()
    assert torch.equal(state, torch.random.get_rng_state())
    old, new = search(bank, parent), search(bank, parent, chooser)
    for a, b in zip(old[:3], new[:3]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(old[3]['selected'], new[3]['selected'])


@pytest.mark.parametrize('order', list(permutations(range(3))))
def test_fixed_joint_argmin_selects_exact_multiset_in_every_bank_order(order):
    bank = torch.tensor([[[.2, .8], [.7, .3], [.2 + 1e-7, .8]]])
    parent = Product.compose(bank[:, 0], bank[:, 1])
    priming = torch.tensor([[1., 2., 4.]])[:, order]
    left, right, available, details = search(bank[:, order], parent,
        DecompositionChooser(), left_priming=priming, right_priming=priming)
    assert available.all() and details['exact'].all()
    selected = int(details['selected'].item())
    left_row = int(details['left_indices'][0, selected // 3])
    right_row = int(details['right_indices'][0, selected % 3])
    assert sorted((order[left_row], order[right_row])) == [0, 1]
    residual = details['pair_residual'].reshape(1, -1)
    torch.testing.assert_close(residual[:, selected], residual.amin(-1), rtol=0, atol=0)
    torch.testing.assert_close(Product.compose(left, right), parent, rtol=0, atol=0)


def test_uniform_round_and_action_proposal_carries_both_count_factors(monkeypatch):
    from types import SimpleNamespace
    from Models import BasicModel
    trace = SimpleNamespace(_choice_actions=torch.tensor([[0, 1, 2]]),
        _choice_attempted=torch.ones(1, 3, dtype=torch.bool),
        _choice_explorable=torch.tensor([[True, True, False]]),
        _choice_alternative_counts=torch.tensor([[2, 4, 0]]))
    model = SimpleNamespace(_reconstruction_stack=lambda: trace,
        _compose_round_owners=lambda actions: (torch.zeros_like(actions),
            torch.zeros(1, 1, dtype=torch.long), None))
    monkeypatch.setattr(torch, 'rand', lambda *args, **kwargs: torch.tensor([[.8, .2, .9]]))
    _, forced, _ = BasicModel._exploration_constraints(model)
    assert forced.tolist() == [[True, False, False]]
    assert model._compose_sampling_scale.tolist() == [[4, 8, 0]]


def test_retired_checkpoint_weight_is_reported_and_never_installed():
    owner = torch.nn.Module()
    owner.decomposition_chooser = DecompositionChooser()
    saved = {'decomposition_chooser.weight': torch.tensor([2.37, 4.85, 7.81, 0., 0.])}
    with pytest.warns(UserWarning, match='Dropped retired decomposition chooser checkpoint key decomposition_chooser.weight'):
        incompatible = owner.load_state_dict(saved, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    torch.testing.assert_close(owner.decomposition_chooser.weight,
                               torch.tensor([1., 0., 0., 0., 0.]), rtol=0, atol=0)
    assert list(owner.parameters()) == [] and owner.state_dict() == {}


def test_native_training_keeps_joint_argmin_out_of_optimizer_and_registry(tmp_path, monkeypatch):
    from pathlib import Path
    import Models
    import util
    from test_mm_xor import _fresh_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    config = Path(Models.__file__).resolve().parents[1] / 'data/XOR_grammar.xml'
    model, _, data = _fresh_model(str(config))
    try:
        chooser = model.languageSpace.decomposition_chooser
        optimizer = model.getOptimizer(lr=.01)
        raw, target = next(iter(data.data_loader(split='train', num_streams=4)))
        batch = model.inputSpace.prepInput(raw), model.outputSpace.prepOutput(target)
        expected = chooser.weight.new_tensor([1., 0., 0., 0., 0.])
        for _ in range(2):
            model.runBatch(train=True, optimizer=optimizer, batchSize=4, split='train', batch_override=batch)
            torch.testing.assert_close(chooser.weight, expected, rtol=0, atol=0)
            assert not chooser.weight.requires_grad and chooser.weight.grad is None
            assert list(chooser.parameters()) == []
            assert all(p is not chooser.weight for group in optimizer.param_groups for p in group['params'])
            assert 'reconstruction.decomposition' not in model._sentence_cost_registry._terms
            assert 'reconstruction.walk_policy' not in model._sentence_cost_registry._terms

        checkpoint = tmp_path / 'retired_pair.ckpt'
        model.save_weights(str(checkpoint))
        saved = torch.load(checkpoint, weights_only=False)
        aliases = [name + '.weight' for name, module in model.named_modules(remove_duplicate=False)
                   if module is chooser]
        assert aliases
        assert not any(key in saved['state_dict'] for key in aliases)
        for key in aliases:
            saved['state_dict'][key] = expected.new_tensor([2.37, 4.85, 7.81, 0., 0.])
        torch.save(saved, checkpoint)
        messages = []
        monkeypatch.setattr(Models, 'TheMessage', lambda message, *args, **kwargs: messages.append(str(message)))
        assert model.load_weights(str(checkpoint), strict=True, require_match=True)
        torch.testing.assert_close(chooser.weight, expected, rtol=0, atol=0)
        for key in aliases:
            assert any('Dropped retired decomposition chooser checkpoint key ' + key in message
                       for message in messages)
    finally:
        model.End()


