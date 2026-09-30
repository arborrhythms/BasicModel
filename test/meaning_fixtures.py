"""Explicit durable inputs for predictor tests that do not run a grammar."""
import torch
from Meaning import ConceptualMeaning, expectation_surprise


def record_observation(store, payload, depth, *, meaning, expectation=None):
    prediction = None if expectation is None else expectation.estimate
    if prediction is not None and expectation.source_occurrences:
        estimate = ConceptualMeaning(prediction.roles, torch.ones(3, dtype=torch.bool),
            mode='unspecified', bindings=prediction.bindings, scope=prediction.scope)
        return store.append_expectation_pair(estimate, meaning,
            presence_logits=prediction.presence_logits,
            source_occurrences=expectation.source_occurrences,
            stream=expectation.stream, document=expectation.document)[1]
    surprise = -1. if prediction is None else expectation_surprise(meaning.roles,
        prediction.roles, meaning.role_mask, prediction.presence_logits.sigmoid())
    return store.append_meaning(meaning, kind='observation', surprise=surprise)
