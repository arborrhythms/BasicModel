"""Loss-side fixtures for the single whole-path answer reader."""
from types import SimpleNamespace, MethodType
import torch


def answer(model, meanings, surface, *, registry=None):
    from Layers import Error
    from Models import BasicModel
    from WholePaths import answer_cost
    from ThoughtReferences import bindings
    from dataclasses import replace
    active = torch.ones(len(meanings), dtype=torch.bool)
    registry = Error(row_mask=active) if registry is None else registry
    model._set_answer_silence_mask = MethodType(BasicModel._set_answer_silence_mask, model)
    model._output_target_cost = lambda construction, targets: surface
    thoughts = tuple((row, SimpleNamespace(meaning=replace(meaning,
        bindings=dict(bindings(meaning), _evidence_kind='retrieval'))))
        for row, meaning in enumerate(meanings))
    return answer_cost(model, None, thoughts, model._sentence_answer_questions, active, registry)


def thoughts(model, meaning, score, **options):
    """A complete-path fixture with constant reading/reconstruction/output stubs.

    The actual thought fork, departure draw, final judge and policy credit
    are the production whole-path implementation. No local judge is installed.
    """
    from WholePaths import Thinking, credit
    from WalkTrials import ThoughtTrialState
    from SentenceCredit import departure, comparison
    from ThoughtStream import commit
    from Layers import Error
    active = torch.ones(1, dtype=torch.bool)
    empty = active[:, None][:, :0]
    start = ThoughtTrialState(model)
    paths, results, effects = [], [], []
    old = getattr(model, '_path_thinking', None)
    try:
        for index in range(2):
            if index:
                start.restore()
                eligible = paths[0].thinking.eligible(active)
                # The legacy certificate supplies its one specific branch;
                # keep that setup explicit now that every fork is retained.
                forks = paths[0].thinking.traces[0].get('forks', {})
                eligible.zero_()
                if forks:
                    eligible[0, min(forks)] = True
                draw = departure(None, empty, active=active, thought=eligible)
            else:
                draw = None
            thinking = Thinking(None if not index else paths[0].thinking,
                                None if draw is None else draw['thought_round'])
            model._path_thinking = thinking
            result = model.run_selected_thought(meaning, **options)
            costs = score(result)
            costs = dict(reconstruction=0., answer=costs) if not isinstance(costs, dict) else costs
            parts = torch.tensor([[float(costs.get('reconstruction', 0.)), 0., float(costs.get('answer', 0.))]])
            paths.append(SimpleNamespace(thinking=thinking, parts=parts, draw=draw,
                registry=Error(row_mask=active), observation={}, output_trace=None))
            results.append(result)
            effects.append(ThoughtTrialState(model))
        credit(model, paths[0], paths[1], active)
        audit = comparison(torch.stack([path.parts for path in paths], 1), active)
        winner = int(audit['winner'][0])
        effects[winner].restore()
        if results[winner].records:
            commit(model, results[winner], row=0)
        model.errors.merge(paths[1].registry)
        at = int(draw['thought_round'][0])
        term = next(iter(paths[1].registry._terms.values()), None)
        value = None if term is None else paths[1].registry._value(term).sum()
        value = value if value is not None and value.requires_grad else None
        choices = [path.thinking.traces[0]['choices'] for path in paths]
        model._fixture_path_credit = dict(surrogate=value,
            category='reconstruction' if at < 0 else choices[1][at].get('category', 'reconstruction'))
        model._fixture_path_audit = dict(costs=tuple(float(v) for v in audit['costs'][0]),
            keep_costs=tuple(float(v) for v in audit['costs'][0]), explore_kept=bool(winner),
            departure=at, work_spent=tuple(result.work.spent for result in results),
            components=tuple((float(path.parts[0, 0]), float(path.parts[0, 2])) for path in paths),
            bind_choices=tuple(tuple(dict(step=i, **choice['bind']) for i, choice in enumerate(trace)
                                    if 'bind' in choice) for trace in choices))
        return results[winner]
    finally:
        model._path_thinking = old


def thought_surrogate(trace, other, departure, costs):
    """Price a supplied thought fork through the production complete-path credit."""
    from WholePaths import credit
    from Layers import Error
    probability = other['choices'][max(0, departure)]['probability']
    parts = probability.new_tensor([[costs[0], 0., 0.], [costs[1], 0., 0.]])
    first = SimpleNamespace(parts=parts[:1], observation={}, output_trace=None)
    path = SimpleNamespace(parts=parts[1:], observation={}, output_trace=None,
        thinking=SimpleNamespace(traces={0: other}), registry=Error(),
        draw=dict(thought_round=torch.tensor([departure]),
                  walk_count=torch.ones(1, dtype=torch.long),
                  walk_rounds=torch.tensor([sum(trace['eligible'])])))
    credit(None, first, path, torch.ones(1, dtype=torch.bool))
    term = next(iter(path.registry._terms.values()), None)
    return probability.new_zeros(()) if term is None else path.registry._value(term).sum()
