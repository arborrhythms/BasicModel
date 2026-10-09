"""Keep the toy comparison honest, reproducible, and independently runnable."""
import pytest
import torch

from bench_concepts_from_percepts import decode, make_problem, run_case


@pytest.mark.parametrize("support", [2, 6])
def test_problem_support_and_exact_target_decode(support):
    problem = make_problem(support, seed=5, train_size=16, test_size=32)
    assert ((problem["weights"] != 0).sum(0) == support).all()
    assert (problem["weights"][:4].abs().sum(0) > 0).all()  # part evidence
    assert (problem["weights"][4:].abs().sum(0) > 0).all()  # whole evidence
    inputs = problem["test"]
    targets = torch.tanh(inputs @ problem["weights"] + problem["bias"])
    torch.testing.assert_close(decode(targets, problem), inputs, atol=1e-6, rtol=1e-5)
    for name, value in make_problem(support, seed=5, train_size=16, test_size=32).items():
        assert torch.equal(value, problem[name])


def test_plain_sigma_learns_and_weak_lasso_retains_pair_but_strong_lasso_loses_it():
    problem = make_problem(2, seed=7, train_size=64, test_size=128)
    settings = dict(variant="sigma", seed=7, steps=500, learning_rate=.5,
                    support_threshold=.02)
    plain = run_case(problem, **settings)
    sparse = run_case(problem, **settings, l1_lambda=.01)
    strong = run_case(problem, **settings, l1_lambda=.1)
    assert plain["reconstruction_mse"] < 1e-5
    assert sparse["reconstruction_mse"] < .005
    assert sparse["exact_nonzero_per_concept"] < plain["exact_nonzero_per_concept"]
    assert sparse["effective_inputs_per_concept"] == 2
    assert sparse["true_support_recall"] == 1.
    assert strong["true_support_recall"] < sparse["true_support_recall"]
    assert strong["reconstruction_mse"] > sparse["reconstruction_mse"] * 10


@pytest.mark.parametrize("variant", ["gated", "sigma"])
def test_reproducible_smoke_run_and_parameter_count(variant):
    problem = make_problem(6, seed=3, train_size=16, test_size=32)
    settings = dict(variant=variant, seed=3, steps=10, learning_rate=.2,
                    support_threshold=.02)
    first = run_case(problem, **settings)
    second = run_case(problem, **settings)
    assert first.pop("fit_seconds") > 0
    assert second.pop("fit_seconds") > 0
    assert first == second
    assert first["trainable_parameters"] == (136 if variant == "gated" else 72)


def test_invalid_benchmark_settings_rejected():
    with pytest.raises(ValueError, match="support"):
        make_problem(3, seed=1)
    problem = make_problem(2, seed=1)
    settings = dict(variant="sigma", seed=1, steps=1, learning_rate=.2,
                    support_threshold=.02)
    for override in ({"variant": "typo"}, {"steps": 0}, {"learning_rate": float("inf")},
                     {"support_threshold": -.1}, {"variant": "gated", "l1_lambda": .1}):
        with pytest.raises(ValueError):
            run_case(problem, **(settings | override))
