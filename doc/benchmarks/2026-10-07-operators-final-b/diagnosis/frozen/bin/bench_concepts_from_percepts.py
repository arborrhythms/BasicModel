"""Controlled concept-readout comparison, NOT a BasicModel throughput test.

Run: BASICMODEL_DEVICE=cpu .venv/bin/python bin/bench_concepts_from_percepts.py

Eight stable input coordinates stand for four part and four whole codes.
Each synthetic concept depends on either two or six of them. Targets use a
known nonsingular mixing matrix followed by tanh. A fixed, known inverse
decodes predicted activations, so training has ONE smooth objective: input
reconstruction. There is no learned decoder that could rescale away lasso,
no quantization, no answer loss, and no vocabulary selection in this probe.

All variants see identical samples and start with identical effective
weights. Plain SGD is followed by an L1 proximal step for Sigma variants;
L1 is NOT also backpropagated. We report exact and effective support, held-
out reconstruction and activation errors, and fit time. The static-context
gated run tests the extra parameterization, not the benefit of local context.
"""
import argparse
import json
import math
import os
import time

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from Layers import GatedConceptsFromPercepts, SigmaConceptsFromPercepts


def make_problem(support, *, seed, train_size=256, test_size=2048):
    """Paired P/W support with a fixed, well-conditioned inverse chart."""
    if support not in (2, 6):
        raise ValueError("support must be 2 or 6")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Coordinate pairs 0/4, 1/5, 2/6, 3/7 cross the part/whole blocks.
    weights = torch.eye(8, device="cpu") * .85
    weights += torch.roll(torch.eye(8, device="cpu"), 4, dims=0) * .35
    if support == 6:
        for shift, strength in [(1, .12), (2, .18), (5, .12), (6, .18)]:
            weights += torch.roll(torch.eye(8, device="cpu"), shift, dims=0) * strength
    bias = torch.linspace(-.1, .1, 8, device="cpu")
    train = torch.rand(train_size, 8, generator=generator, device="cpu") * 1.5 - .75
    test = torch.rand(test_size, 8, generator=generator, device="cpu") * 1.5 - .75
    return {"weights": weights, "bias": bias, "inverse": torch.linalg.inv(weights),
            "train": train, "test": test}


def decode(activation, problem):
    """Known target inverse; never claim this is a learned collective decoder."""
    logits = torch.atanh(activation.clamp(-1 + 1e-6, 1 - 1e-6))
    return (logits - problem["bias"]) @ problem["inverse"]


def run_case(problem, *, variant, seed, steps, learning_rate, support_threshold,
             l1_lambda=0.0):
    """Train one matched control. Only timing is nondeterministic on CPU."""
    if variant not in ("gated", "sigma"):
        raise ValueError("variant must be gated or sigma")
    if steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive and finite")
    if not math.isfinite(support_threshold) or support_threshold < 0:
        raise ValueError("support_threshold must be finite and nonnegative")
    # Match the initial EFFECTIVE W, not merely the global RNG seed: gates
    # start at .5 and would otherwise halve the gated initial readout.
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
    initial_weights = (torch.rand(8, 8, generator=generator, device="cpu") * 2 - 1) / 8
    if variant == "gated":
        if l1_lambda:
            raise ValueError("gated control has no L1 penalty")
        layer = GatedConceptsFromPercepts(8, 8).to("cpu")
    else:
        layer = SigmaConceptsFromPercepts(8, 8, l1_lambda=l1_lambda).to("cpu")
    with torch.no_grad():
        scale = .5 if variant == "gated" else 1.
        layer.input_weights.copy_(initial_weights / scale)
        layer.concept_bias.zero_()
    optimizer = torch.optim.SGD(layer.parameters(), lr=learning_rate, momentum=0)
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        reconstructed = decode(layer(problem["train"]), problem)
        loss = (reconstructed - problem["train"]).square().mean()
        loss.backward()
        optimizer.step()
        if variant == "sigma":
            layer.proximal_step_(learning_rate)
    fit_seconds = time.perf_counter() - start
    with torch.no_grad():
        inputs = problem["test"]
        target = torch.tanh(inputs @ problem["weights"] + problem["bias"])
        predicted = layer(inputs)
        recon_mse = (decode(predicted, problem) - inputs).square().mean().item()
        activation_mse = (predicted - target).square().mean().item()
        coefficients = layer.input_weights
        if variant == "gated":
            coefficients = coefficients * layer.gates()
        effective = coefficients.abs() > support_threshold
        truth_support = problem["weights"] != 0
        result = {
            "variant": variant, "l1_lambda": l1_lambda, "seed": seed,
            "steps": steps, "learning_rate": learning_rate,
            "support_threshold": support_threshold,
            "target_inputs_per_concept": truth_support.sum(0).float().mean().item(),
            "reconstruction_mse": recon_mse, "activation_mse": activation_mse,
            "exact_nonzero_per_concept": (coefficients != 0).sum(0).float().mean().item(),
            "effective_inputs_per_concept": effective.sum(0).float().mean().item(),
            "true_support_recall": (effective & truth_support).sum().item()
                                   / truth_support.sum().item(),
            "extra_edges_per_concept": (effective & ~truth_support).sum(0).float().mean().item(),
            "trainable_parameters": sum(p.numel() for p in layer.parameters()),
            "fit_seconds": fit_seconds,
        }
    for name, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"non-finite {name} for {variant} seed={seed}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=.3)
    parser.add_argument("--l1", type=float, nargs="+", default=[.01, .1])
    parser.add_argument("--support-threshold", type=float, default=.02)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for support in (2, 6):
        for seed in args.seeds:
            problem = make_problem(support, seed=seed)
            for variant, strength in [("gated", 0.), ("sigma", 0.)] + [
                    ("sigma", strength) for strength in args.l1]:
                result = run_case(
                    problem, variant=variant, seed=seed, steps=args.steps,
                    learning_rate=args.learning_rate, l1_lambda=strength,
                    support_threshold=args.support_threshold)
                print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
