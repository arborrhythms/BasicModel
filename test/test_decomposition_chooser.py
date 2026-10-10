"""The supervised inverse changes pair inference, never perception's codes."""
import torch
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


def test_teacher_recovers_true_shortlisted_pair_despite_misleading_priming():
    bank = torch.tensor([[[.2, .8], [.7, .3], [.8, .4]]], requires_grad=True)
    # A lossy/noisy root is near the heavily primed distractor, but not exact.
    parent = torch.tensor([[.161, .321]], requires_grad=True)
    chooser = DecompositionChooser()
    optimizer = torch.optim.SGD(chooser.parameters(), lr=.1)
    rows = torch.tensor([[10, 11, 12]])
    target = torch.tensor([[10, 11]])
    options = dict(left_valid=torch.tensor([[True, False, False]]),
                   right_valid=torch.tensor([[False, True, True]]),
                   left_priming=torch.tensor([[1., 1., 10.]]),
                   right_priming=torch.tensor([[1., 1., 10.]]))
    before = search(bank, parent, chooser, **options)[3]
    loss, present, correct = chooser.teacher_loss(before, rows, rows, target)
    assert present.all() and not correct.any()
    for _ in range(20):
        details = search(bank, parent, chooser, **options)[3]
        loss, _, _ = chooser.teacher_loss(details, rows, rows, target)
        optimizer.zero_grad(); loss.sum().backward(); optimizer.step()
    after = search(bank, parent, chooser, **options)
    _, present, correct = chooser.teacher_loss(after[3], rows, rows, target)
    assert present.all() and correct.all()
    torch.testing.assert_close(after[1], bank[:, 1].detach())
    assert bank.grad is None and parent.grad is None
    assert not torch.equal(chooser.weight.detach(), torch.tensor([1., 0., 0., 0., 0.]))


def test_absent_target_is_counted_and_has_no_training_gradient():
    chooser = DecompositionChooser()
    bank = torch.tensor([[[.2, .8], [.7, .3]]])
    details = search(bank, torch.tensor([[.14, .24]]), chooser)[3]
    loss, present, correct = chooser.teacher_loss(details, torch.tensor([[10, 11]]),
        torch.tensor([[10, 11]]), torch.tensor([[10, 99]]))
    assert not present.any() and not correct.any() and loss.item() == 0
    loss.sum().backward()
    assert not chooser.weight.grad.any()


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


def test_teacher_uses_resolved_input_rows_when_the_legacy_word_lane_is_absent():
    from types import SimpleNamespace
    from Models import BasicModel
    chooser = DecompositionChooser()
    bank = SimpleNamespace(rows=torch.tensor([[10, 11]]),
        codes=torch.tensor([[[.2, .8], [.7, .3]]]),
        valid=torch.ones(1, 2, dtype=torch.bool), weights=torch.ones(1, 2))
    parent = Product.compose(bank.codes[:, 0], bank.codes[:, 1])
    values = torch.zeros(3, 3, 2)
    values[2] = torch.stack((bank.codes[0, 0], bank.codes[0, 1], parent[0]))
    program = SimpleNamespace(word_rows=torch.tensor([-1, -1]),
        actions=torch.tensor([[0, -1, 0], [0, -1, 1], [1, 0, -1]]), operation_values=values)
    record = SimpleNamespace(primed=bank, root=parent,
        word_rows=bank.rows, word_valid=bank.valid)
    language = SimpleNamespace(decomposition_chooser=chooser,
        language_layer=SimpleNamespace(operation_layer=SimpleNamespace(ops=[Product()])),
        _bounded_binary_reconstruction=LanguageSpace._bounded_binary_reconstruction)
    model = SimpleNamespace(languageSpace=language, reconstruction_basis_limit=2)
    model._sentence_leaf_positions = lambda record, row: BasicModel._sentence_leaf_positions(model, record, row)
    loss = BasicModel._decomposition_teacher_loss(model, {'entries': [program]}, record)
    assert model._last_decomposition_teacher[0]['target'] == [10, 11]
    assert model._last_decomposition_teacher[0]['present']
    loss.sum().backward()
    assert chooser.weight.grad is not None and chooser.weight.grad.abs().sum() > 0
