"""Explicit durable inputs for predictor tests that do not run a grammar."""
import torch
from Meaning import ConceptualMeaning


def record_observation(store, payload, depth, *, meaning, expectation=None):
    prediction = None if expectation is None else expectation.estimate
    if prediction is not None and expectation.source_occurrences:
        estimate = ConceptualMeaning(prediction.roles, torch.ones(3, dtype=torch.bool),
             bindings=prediction.retained_bindings(), scope=prediction.scope)
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
         bindings=prediction.retained_bindings(), scope=prediction.scope)


def unsupported(meaning, slots=None):
    """A fixture with explicit zero source evidence, independent of form."""
    from ThoughtReferences import open_slots, with_slots
    return with_slots(meaning, open_slots(meaning) if slots is None else slots,
                      pair=(0., 0.), source_evidence=(0., 0.))


def replay_first_choice(model, monkeypatch, index):
    """Record a first decision through the existing chooser replay input."""
    chooser = model.shared_grammar
    select, count = chooser.select_logits, [0]
    def recorded(logits, **kwargs):
        if count[0] == 0:
            kwargs['replay_action'] = torch.tensor([index % logits.shape[-1]], device=logits.device)
        count[0] += 1
        return select(logits, **kwargs)
    monkeypatch.setattr(chooser, 'select_logits', recorded)


def replay_thought_plan(model, monkeypatch, plan):
    """Choose a recorded operation while retaining the real scorer and walk."""
    choose = model._choose_selected_thought_action
    decisions = iter(plan) if not callable(plan) else None
    def recorded(root, active, actions, **kwargs):
        selected = plan(root, active, actions, **kwargs) if callable(plan) else next(decisions)
        index = next(i for i, action in enumerate(actions)
                     if action is selected or (isinstance(selected, str) and action is not None
                                               and action.semantic_id == selected))
        chooser = model._selected_thought_chooser(active)
        select = chooser.select_logits
        def replay(logits, **options):
            # A recorded alternative is still subject to the walk's exclusion.
            if options.get('masked_action') is None:
                options['replay_action'] = torch.tensor([index], device=logits.device)
            return select(logits, **options)
        with monkeypatch.context() as patch:
            patch.setattr(chooser, 'select_logits', replay)
            return choose(root, active, actions, **kwargs)
    monkeypatch.setattr(model, '_choose_selected_thought_action', recorded)
