"""Item 4 rev-5 pre-handoff measurements; see PROTOCOL.md for scope/budget.

Only the quantizer and credit audit call production code. Everything else is
a mathematical toy, never a substitute for BasicModel's learning gates.
Run from basicmodel with PYTHONPATH=bin .venv/bin/python <this file>.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


EPS = 1e-6
DTYPE = torch.float64


def encode(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def gain_image(observed, expected, *, gain, mode="relative_clip",
               detach="full", attention=1., confidence=1., spared=0.):
    """Last axis is evidence lanes; one gain per symbol, not per feature.

    Both signs of error are retained for EACH evidence lane. The observation
    is live on the content path; the full saturation is cut from that path.
    Predictor training uses a separate detached-target loss.
    """
    residual = observed - expected
    plus, minus = residual.relu(), (-residual).relu()
    numerator = (plus + minus).sum(-1)
    if detach == "numerator_only":
        numerator = numerator.detach()
    denominator = expected.abs().sum(-1) + EPS
    if mode == "relative_clip":
        saturation = (numerator / denominator).clamp(max=1.)
    elif mode == "relative_smooth":
        saturation = numerator / (denominator + numerator)
    elif mode == "absolute_clip":
        saturation = numerator.clamp(max=1.)
    else:
        raise ValueError(mode)
    if detach == "full":
        saturation = saturation.detach()
    attenuation = 1. - gain * (1. - spared) * confidence * (1. - saturation)
    return attention * attenuation[..., None] * observed, plus, minus, attenuation


def toy2():
    from SentenceCredit import comparison

    # Stored order is R,E,A, as in the landed comparison implementation.
    costs = np.array([[.1, 1., 0.], [.4, 0., 0.]])
    results = []
    for seed in range(16):
        initial = np.random.default_rng(seed).normal(0., .01, 2)
        for name, columns in (("R+A", [0, 2]), ("R+E+A", [0, 1, 2])):
            logits = initial.copy()
            terminal = costs[:, columns].sum(-1)
            for _ in range(256):
                probability = np.exp(logits - logits.max())
                probability /= probability.sum()
                greedy = int(logits.argmax())
                alternate = 1 - greedy
                gradient = -probability
                gradient[alternate] += 1.
                logits += .1 * (terminal[greedy] - terminal[alternate]) * gradient
            probability = np.exp(logits - logits.max())
            probability /= probability.sum()
            results.append(dict(seed=seed, arm=name, explanation_probability=probability[0],
                                greedy=int(logits.argmax())))
    parts = torch.tensor([[[.3, 99., .6], [.4, 0., .1]]], dtype=DTYPE)
    audit = comparison(parts, torch.tensor([True]))
    changed = parts.clone()
    changed[..., 1] = torch.tensor([[0., 999.]], dtype=DTYPE)
    other = comparison(changed, torch.tensor([True]))
    assert torch.equal(audit["costs"], other["costs"])
    assert torch.equal(audit["wins"], other["wins"])
    return dict(terminal_R_E_A=costs, runs=results, landed=dict(
        costs=audit["costs"], keep_costs=audit["keep_costs"], wins=audit["wins"],
        advantage=audit["advantage"], E_independent=True,
        answer_against_keep=audit["answer_against_keep"]))


def toy5():
    stream = np.r_[np.full(128, .2), np.full(256, .8), np.full(64, .2)]
    mean, variance, rate = .5, .25, .05
    history = []
    for t, observed in enumerate(stream):
        # Forecast and score precede update; no current-target leakage.
        z = min(1., abs(observed - mean) / (np.sqrt(variance) + EPS))
        confidence = float(t >= 32)
        history.append(dict(step=t, observed=observed, mean_before=mean,
                            variance_before=variance, z=z,
                            salience=confidence * z, no_statistic_control=0.))
        delta = observed - mean
        mean += rate * delta
        variance = (1. - rate) * (variance + rate * delta * delta)
    salience = np.array([row["salience"] for row in history])
    # Exactly calibrated marginal moments cannot tell alternation from a
    # repeated symbol: both 0 and 1 are one standard deviation from .5.
    ordered = np.tile([0., 1.], 16)
    broken = ordered.copy()
    broken[16:18] = broken[16:18][::-1]  # breaks order, preserves both counts
    normalized = lambda x: np.minimum(1., np.abs(x - .5) / (.5 + EPS))
    assert np.array_equal(normalized(ordered), normalized(broken))
    return dict(history=history, summary=dict(
        cold_max=salience[:32].max(), before_violation=salience[127],
        first_violation=salience[128], habituated_last32=salience[352:384].mean(),
        return_violation=salience[384]), order_control=dict(
            ordered=ordered, broken=broken, same_salience=True,
            normalized_residual=normalized(ordered)))


def beta_kl(a, b, c, d):
    """KL(Beta(a,b) || Beta(c,d)), vectorized in float64."""
    log_beta = lambda x, y: torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)
    return (log_beta(c, d) - log_beta(a, b)
            + (a - c) * torch.digamma(a) + (b - d) * torch.digamma(b)
            + (c + d - a - b) * torch.digamma(a + b))


def toy6():
    runs = []
    for seed in range(8):
        observations = torch.tensor(np.random.default_rng(seed).integers(0, 2, 4096), dtype=DTYPE)
        successes = observations.cumsum(0) - observations
        a = 1. + successes
        b = 1. + torch.arange(4096, dtype=DTYPE) - successes
        information = beta_kl(a + observations, b + 1. - observations, a, b)
        raw = (observations - .5).abs()
        normalized = (raw / (.5 + EPS)).clamp(max=1.)
        runs.append(dict(seed=seed, raw_residual=raw.mean(), normalized_residual=normalized.mean(),
                         information_first64=information[:64].mean(),
                         information_last1024=information[-1024:].mean(),
                         information_min=information.min()))
    return dict(runs=runs, calibrated_mean=.5, calibrated_variance=.25)


def adversarial_gain(detach, *, downstream=True):
    estimate = torch.nn.Parameter(torch.tensor([[.6]], dtype=DTYPE))
    observed = torch.tensor([[.75]], dtype=DTYPE, requires_grad=True)
    content, _, _, _ = gain_image(observed, estimate, gain=.9, detach=detach)
    downstream_gradient = torch.autograd.grad((content - .75).square().sum(), estimate,
                                               allow_unused=True)[0]
    local = (estimate - observed.detach()).square().sum()
    source_gradient = torch.autograd.grad(local, observed, allow_unused=True)[0]
    first_gradient = 0. if downstream_gradient is None else float(downstream_gradient.item())
    assert source_gradient is None
    for _ in range(256):
        content, _, _, _ = gain_image(observed, estimate, gain=.9, detach=detach)
        loss = (estimate - observed.detach()).square().mean()
        if downstream:
            loss = loss + 4. * (content - .75).square().mean()
        gradient, = torch.autograd.grad(loss, estimate)
        with torch.no_grad():
            estimate.sub_(.05 * gradient).clamp_(0., 1.)
    content, _, _, _ = gain_image(observed, estimate, gain=.9, detach=detach)
    return dict(detach=detach, downstream=downstream, final_prediction=estimate.item(),
                final_content=content.item(), local_mse=(estimate - .75).square().item(),
                initial_downstream_gradient=first_gradient, local_source_gradient=None)


def toy7():
    runs = []
    for seed in range(8):
        stream = torch.tensor(np.stack((np.full(1024, .75),
            np.random.default_rng(seed).integers(0, 2, 1024)), -1), dtype=DTYPE)[..., None]
        predictor = torch.nn.Parameter(torch.full((2, 1), .5, dtype=DTYPE))
        for _ in range(128):
            loss = (predictor - stream.detach()).square().mean()
            gradient, = torch.autograd.grad(loss, predictor)
            with torch.no_grad():
                predictor.sub_(.5 * gradient)
        for mode in ("relative_clip", "relative_smooth", "absolute_clip"):
            for gain in (0., .5, .9, 1.):
                content, plus, minus, attenuation = gain_image(stream, predictor, gain=gain, mode=mode)
                assert bool(((content >= 0.) & (content <= stream)).all())
                assert bool(((attenuation >= 1. - gain - 1e-12) & (attenuation <= 1.)).all())
                if gain == 0.:
                    assert torch.equal(content, stream)
                subtraction = stream - gain * predictor.detach()
                runs.append(dict(seed=seed, mode=mode, gain=gain, prediction=predictor,
                    content_mean=content.mean(0), content_min=content.min(), content_max=content.max(),
                    residual_mean=(plus + minus).mean(0),
                    subtraction_min=subtraction.min(), subtraction_negative=int((subtraction < 0.).sum())))
    expected = torch.tensor([[.8, 0.]], dtype=DTYPE)
    omission = gain_image(torch.zeros_like(expected), expected, gain=1.)
    mask_cases = {}
    for name, kwargs in (("ordinary", {}), ("outside_region", dict(attention=0.)),
                         ("no_confidence", dict(confidence=0.)), ("object_spared", dict(spared=1.))):
        value = gain_image(expected, expected, gain=.9, **kwargs)
        mask_cases[name] = dict(content=value[0], attenuation=value[3])
    reversal = gain_image(torch.tensor([[1., 0.]], dtype=DTYPE),
                          torch.tensor([[0., 1.]], dtype=DTYPE), gain=1.)
    # The separate residuals do not net opposing evidence to an apparent match.
    assert torch.equal(reversal[0], torch.tensor([[1., 0.]], dtype=DTYPE))
    assert torch.equal(omission[0], torch.zeros_like(expected))
    assert torch.equal(omission[2], expected)
    assert torch.equal(mask_cases["no_confidence"]["content"], expected)
    assert torch.equal(mask_cases["object_spared"]["content"], expected)
    choice_costs = []
    for detach in ("none", "numerator_only", "full"):
        arrived = torch.tensor([[.75]], dtype=DTYPE)
        content, _, _, _ = gain_image(arrived, arrived, gain=.9, detach=detach)
        reconstructions = torch.tensor([.75, 0.], dtype=DTYPE)
        raw_cost = (arrived.item() - reconstructions).square()
        attenuated_cost = (content.item() - reconstructions).square()
        choice_costs.append(dict(detach=detach, whole_target_costs=raw_cost,
            attenuated_target_costs=attenuated_cost,
            whole_target_choice=int(raw_cost.argmin()),
            attenuated_target_choice=int(attenuated_cost.argmin())))
    return dict(runs=runs, omission=dict(content=omission[0], plus=omission[1], minus=omission[2]),
        mask_cases=mask_cases, lane_reversal=dict(content=reversal[0], plus=reversal[1], minus=reversal[2]),
        chooser_costs=choice_costs,
        gaming=[adversarial_gain(mode) for mode in ("none", "numerator_only", "full")]
               + [adversarial_gain("full", downstream=False)])


def toy8():
    from Layers import VectorQuantize

    prototypes = torch.tensor([[0., 0., 1.], [.8, .8, .2]])
    observed = prototypes[1:2]
    expected = torch.tensor([[.8, .8, 0.]])
    per_coordinate = ((observed - expected).abs() / (expected.abs() + EPS)).clamp(max=1.)
    rows = []
    for cosine in (True, False):
        # All initialization is overwritten; no seed selects an outcome.
        with torch.random.fork_rng(devices=[]):
            quantizer = VectorQuantize(dim=3, codebook_size=2, use_cosine_sim=cosine,
                ema_update=False, allocate_ema_state=False).eval()
        quantizer.codebook = prototypes
        def identify(value):
            with torch.no_grad():
                return int(quantizer(value)[1].item())
        identity = identify(observed)
        for gain in (0., .5, .9, 1.):
            coordinate_gain = (1. - gain) + gain * per_coordinate
            scalar_gain = coordinate_gain.mean(-1, keepdim=True)
            coordinate = observed * coordinate_gain
            scalar = observed * scalar_gain
            perfect = (1. - gain) * observed
            rows.append(dict(metric="cosine" if cosine else "euclidean", gain=gain,
                original_identity=identity, coordinate_identity=identify(coordinate),
                scalar_identity=identify(scalar), scalar_gain=scalar_gain.item(),
                perfect_identity=identify(perfect), perfect_gain=1. - gain,
                identify_before_gain=identity, coordinate_content=coordinate, scalar_content=scalar,
                scalar_cosine=float(F.cosine_similarity(observed, scalar).item())))
        assert torch.equal(quantizer.codebook, prototypes)
    return dict(prototypes=prototypes, expected=expected, runs=rows,
                identity_names=["whiskers", "cat"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    # Avoid nested BLAS thread overhead for tiny tensors; no model constructed.
    torch.set_num_threads(1)
    results = dict(protocol="PROTOCOL.md", basicmodel_baseline="9207f5e03",
                   wikioracle_baseline="929e4a8", spec_revision=5,
                   torch_version=torch.__version__, numpy_version=np.__version__)
    for number, run in ((2, toy2), (5, toy5), (6, toy6), (7, toy7), (8, toy8)):
        results[f"toy{number}"] = run()
        print(f"Toy {number}: measurement complete", flush=True)
    args.out.write_text(json.dumps(results, default=encode, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
