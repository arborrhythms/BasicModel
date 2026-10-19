"""Explicit durable inputs for predictor tests that do not run a grammar."""
import torch
from Meaning import ConceptualMeaning


def record_observation(store, payload, depth, *, meaning, expectation=None):
    prediction = None if expectation is None else expectation.estimate
    if prediction is not None and expectation.source_occurrences:
        estimate = ConceptualMeaning(prediction.roles, torch.ones(3, dtype=torch.bool),
            mode='unspecified', bindings=prediction.retained_bindings(), scope=prediction.scope)
        from dataclasses import replace
        bindings = dict(meaning.bindings)
        bindings['_observed_symbol_activations'] = tuple((int(i), tuple(a)) for i,a in
            zip(prediction.symbol_ids.tolist(), expectation.activation.tolist()))
        meaning = replace(meaning, bindings=bindings)
        return store.append_expectation_pair(estimate, meaning,
            presence_logits=prediction.presence_logits,
            source_occurrences=expectation.source_occurrences,
            stream=expectation.stream, document=expectation.document)[1]
    surprise = -1. if prediction is None else expectation.surprise
    return store.append_meaning(meaning, kind='observation', surprise=surprise)


def symbol_prediction(keys, activation, *, bindings=(), scope=()):
    from Layers import MeaningExpectation
    keys, activation = torch.as_tensor(keys), torch.as_tensor(activation)
    logits = torch.where(activation >= 1, 60., torch.where(activation <= 0, -60.,
        torch.logit(activation.clamp(1e-7, 1.-1e-7))))
    roles = torch.cat((keys[:1], keys.new_zeros(2, keys.shape[-1])))
    return MeaningExpectation(roles, keys.new_tensor([30., -30., -30.]), bindings, scope,
        keys, logits, torch.arange(len(keys)), torch.zeros(len(keys), dtype=torch.long))


def estimate_meaning(prediction):
    return ConceptualMeaning(prediction.roles, torch.ones(3, dtype=torch.bool),
        mode='unspecified', bindings=prediction.retained_bindings(), scope=prediction.scope)
