"""Measurement helpers for native, detached symbol-activation predictions.

Metric version 2 is activation MSE. It is not a regrade of the archived
role-vector learning certificates. Native training protocols live with their
receipts; this module cannot run the retired root/structured benchmark.
"""
import statistics
import types
import torch
import torch.nn.functional as F

def meaning_windows(documents, context_window=4, occupancy=None):
    """Only predecessors from this document, with one cold start per document."""
    values, masks, targets, addresses, target_masks = [], [], [], [], []
    for document, rows in enumerate(documents):
        for index in range(1, len(rows)):
            prefix = rows[max(0, index-context_window):index].detach()
            padding = context_window - len(prefix)
            values.append(F.pad(prefix, (0, 0, 0, 0, padding, 0)))
            present = torch.zeros(context_window, 3, dtype=torch.bool, device=rows.device)
            if occupancy is None:
                present[padding:] = True
                target_masks.append(torch.ones(3, dtype=torch.bool, device=rows.device))
            else:
                present[padding:] = occupancy[document][max(0, index-context_window):index]
                target_masks.append(occupancy[document][index])
            masks.append(present)
            targets.append(rows[index].detach())
            addresses.append((document, index))
    if not values:
        return None
    return (torch.stack(values), torch.stack(masks), torch.stack(targets), addresses,
            torch.stack(target_masks))



def summarize_steps(steps, warmup=2):
    """Use observed counts, never configured B times the number of calls."""
    measured = steps[warmup:]
    if not measured:
        raise ValueError("no measured steps remain after warmup")
    seconds = sum(step["seconds"] for step in measured)
    observations = sum(step["observations"] for step in measured)
    inputs = sum(step["input_sentences"] for step in measured)
    pairs = sum(step["predicted_targets"] for step in measured)
    supplied = sum(step.get("supplied_answer_rows", 0) for step in measured)
    return {
        "steps": len(measured), "seconds": seconds,
        "warmup_seconds": sum(step["seconds"] for step in steps[:warmup]),
        "median_step_seconds": statistics.median(s["seconds"] for s in measured),
        "input_sentences": inputs, "observations": observations, "predicted_targets": pairs,
        "input_sentences_per_second": inputs / seconds,
        "predicted_targets_per_second": pairs / seconds,
        "supplied_answer_rows": supplied,
        "supplied_answer_rows_per_second": supplied / seconds,
        "document_boundary_fraction": sum(s["cold_starts"] for s in measured) / max(1, observations),
        "reconstruction_mean": statistics.mean(s["reconstruction"] for s in measured),
        "supplied_answer_mean": statistics.mean(s["answer"] for s in measured),
    }



def _controlled_inputs(values, masks, control, seed):
    if control == "context_free":
        return torch.zeros_like(values), torch.zeros_like(masks)
    if control == "shuffled":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(len(values), generator=generator, device="cpu")
        return values[order.to(values.device)], masks[order.to(masks.device)]
    return values, masks



def observe_optimizer_steps(optimizer, callback):
    """Observe successful outer steps, including the native MultiOptimizer.

    Counting child Adam calls would inflate the model's update count. Restore
    the exact instance surface when the observer is removed.
    """
    previous_step = optimizer.step
    had_override = "step" in vars(optimizer)
    previous_override = vars(optimizer).get("step")

    def observed_step(*args, **kwargs):
        result = previous_step(*args, **kwargs)
        callback(optimizer, args, kwargs)
        return result

    optimizer.step = observed_step

    def remove():
        if had_override:
            optimizer.step = previous_override
        else:
            del optimizer.step

    return types.SimpleNamespace(remove=remove)



def score_prediction_thought(prediction, target, occupied):
    """Score a held prior after arrival, outside the ordinary thought controller."""
    from Layers import MeaningExpectation
    from Expectation import symbol_activations
    if not isinstance(prediction, MeaningExpectation):
        return dict(answered=False, activation_mse=None, metric_version=2)
    actual = symbol_activations(prediction.symbol_keys[None], target.detach()[None], occupied[None])[0]
    return dict(answered=True, activation_mse=float((actual-prediction.expected_activation).square().mean()),
                metric_version=2)


@torch.no_grad()
def _score_head(head, values, masks, targets, *, bank, target_masks=None):
    """The same native bank is supplied to ordered, shuffled and cold controls."""
    from Expectation import ActivationPredictor, symbol_activations
    parameter = next(head.parameters())
    values, masks, targets = values.to(parameter), masks.to(parameter.device), targets.to(parameter)
    keys = bank.detach().to(parameter)[None].expand(len(values), -1, -1)
    target_masks = (torch.ones(targets.shape[:2], dtype=torch.bool, device=parameter.device)
                    if target_masks is None else target_masks.to(parameter.device))
    source = symbol_activations(keys, values.flatten(1, 2), masks.flatten(1))
    context = ActivationPredictor.context_from(values, masks)
    expected = head(keys, context, source).sigmoid()
    actual = symbol_activations(keys, targets, target_masks)
    return dict(metric_version=2, activation_mse=float((actual-expected).square().mean()),
                predicted_targets=len(targets), bank_size=keys.shape[1],
                target_activation_variance=float(actual.var(0, unbiased=False).mean()),
                estimate_activation_variance=float(expected.var(0, unbiased=False).mean()))
