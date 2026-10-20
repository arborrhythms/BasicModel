"""Regressions for the two outstanding Item 4 invariants, without seeded passes."""
import pytest
import torch


@pytest.mark.parametrize('name', ('conjunction', 'disjunction', 'sum'))
@pytest.mark.parametrize('stance', ('for', 'against'))
def test_inverse_recovers_each_lane_from_an_against_only_candidate(name, stance):
    from MeaningCodes import compose, exchange, recover_lanes
    # Each code wins at least one coordinate, so both min and max retain
    # enough evidence for the native nonnegative scalar read to be exact.
    a = torch.tensor([[1., 0., .2, .5, .9, 0., 0., 0.]])
    b = torch.tensor([[0., 1., .6, .4, .8, 0., 0., 0.]])
    operands = (a, b)
    if stance == 'against':
        operands = tuple(torch.cat((x[:, :2], exchange(x[:, 2:])), -1) for x in operands)
    parent = torch.cat((torch.ones(1, 2), compose(name, *(x[:, 2:] for x in operands))), -1)
    negative_candidate = torch.cat((a[:, :2], exchange(a[:, 2:])), -1)
    left, right = recover_lanes(name, parent, negative_candidate, b, 2)
    for recovered, target in zip((left, right), operands):
        torch.testing.assert_close(recovered, target, rtol=1e-6, atol=1e-7)
        assert bool((recovered >= 0).all())


@pytest.mark.parametrize('count', (2, 7, 33))
def test_initial_reference_candidates_are_uniform_and_the_difference_can_learn(count):
    from Language import MLPTransformChooser
    chooser = MLPTransformChooser(d_model=8, n_copy=1, n_op=1, binding_choices=True)
    source = torch.randn(2, 2, 8)
    # Wide candidate magnitudes expose why shrinking random weights alone
    # cannot establish the conditional initialization contract.
    candidates = torch.randn(2, 2, count, 8) * torch.logspace(-2, 2, count)[None, None, :, None]
    context = torch.randn(2, 2, count, 18)
    _, logits = chooser.score_unary(source, candidates, None, None,
        op_indices=torch.zeros(count, dtype=torch.long), reference_ctx=context)
    torch.testing.assert_close(logits.softmax(-1), torch.full_like(logits, 1 / count),
                               rtol=0, atol=0)
    (-logits.log_softmax(-1)[..., 0].mean()).backward()
    assert bool(chooser.reference_choice.weight.grad.ne(0).any())
    assert bool(chooser.mlp[0].weight.grad[:, 8:16].ne(0).any())


@pytest.mark.parametrize('count', (2, 7, 257))
def test_enabled_prime_preserves_the_initial_binding_band_for_any_menu(count):
    from types import SimpleNamespace
    from Language import OperationSelectionLayer
    owner = SimpleNamespace(d_model=8)
    candidates = torch.randn(2, count, 8)
    prime = torch.randn(2, 8) * 100
    # Opposite extremes exercise the bound independent of the random draw.
    candidates[:, 0] = prime
    candidates[:, 1] = -prime
    bias = OperationSelectionLayer.prime_bias(owner, candidates, prime, binding_choices=True)
    ratio = bias.softmax(-1) * count
    assert bool(((ratio >= .9) & (ratio <= 1.1)).all())
    assert bool((bias[:, 0] > bias[:, 1]).all())
    # A conditional subset must satisfy the same contract; duplicate keys
    # and arbitrary legal-menu masks cannot bypass the log-odds bound.
    ratio = bias[:, ::2].softmax(-1) * bias[:, ::2].shape[-1]
    assert bool(((ratio >= .9) & (ratio <= 1.1)).all())


def test_episode_state_audit_does_not_materialize_sparse_capacity(monkeypatch):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'doc/benchmarks/2026-10-08-math-chain-repair-2/closing_episode_state.py'
    spec = importlib.util.spec_from_file_location('sparse_episode_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def forbidden(*args, **kwargs):
        raise AssertionError('state audit densified sparse capacity')
    monkeypatch.setattr(torch.Tensor, 'to_dense', forbidden)
    indices = torch.tensor([[19, 2], [41, 7]])
    values = torch.tensor([.25, .5])
    shape = (1_000_000, 1_000_000)
    original = torch.sparse_coo_tensor(indices, values, shape)
    reordered = torch.sparse_coo_tensor(indices.flip(1), values.flip(0), shape)
    changed = torch.sparse_coo_tensor(indices, values + .25, shape)
    assert module.value_digest(original) == module.value_digest(reordered)
    assert module.value_digest(original) != module.value_digest(changed)


@pytest.mark.parametrize('kind', ('anchordot', 'mlp'))
def test_bounded_bracket_scoring_preserves_menu_draw_and_gradients(kind):
    from copy import deepcopy
    from Language import OperationSelectionLayer
    full = OperationSelectionLayer(d_model=8, chooser=kind)
    bounded = deepcopy(full)
    keys = torch.randn(2, 64, 8)
    legal = torch.zeros(2, 64, 6, dtype=torch.bool)
    legal[0, :3, :3] = True
    legal[1, :2, 1:4] = True
    space = torch.zeros(2, 64, dtype=torch.long)
    prior = torch.randn(2, 64)
    options = dict(prior=prior, masked_action=torch.tensor([0, 2]), return_details=True)
    state = torch.get_rng_state()
    expected = full.attend(keys, legal, space, **options)
    after = torch.get_rng_state()
    torch.set_rng_state(state)
    actual = bounded.attend(keys, legal, space, slot_limit=3, **options)
    assert torch.equal(torch.get_rng_state(), after)
    for left, right in zip(expected[:3], actual[:3]):
        torch.testing.assert_close(left, right)
    for name in expected[3]:
        torch.testing.assert_close(expected[3][name], actual[3][name])
    expected[3]['probability'].log().sum().backward()
    actual[3]['probability'].log().sum().backward()
    for left, right in zip(full.parameters(), bounded.parameters()):
        if left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad)
