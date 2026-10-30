"""The ordinary presented/comparison readers score a concluded binding."""
from dataclasses import replace
import torch
from Layers import Error


def accepted(meaning):
    from ThoughtReferences import bindings, open_slots
    data = bindings(meaning)
    return (not open_slots(meaning) and data.get('_evidence_kind') != 'intuition'
            and (data.get('_producing_operation') in ('infer', 'query', 'isTrue')
                 or data.get('_evidence_kind') in ('pervasion', 'retrieval', 'truth')))


def bound_view(state, observation, sentence, row, meaning):
    """Substitute only this row's completed content into the existing reader."""
    language = list(state[1])
    for index in (6, 10, 11):
        language[index] = language[index].clone()
    count = int(meaning.role_mask.sum())
    from ThoughtReferences import open_slots
    if count not in (1, 3) or not accepted(meaning):
        raise ValueError('the answer reader requires a completed one- or three-role meaning')
    slots = meaning.roles[meaning.role_mask]
    # Sentence physical order is NP2, NP1, VP for a ternary row.
    if count == 3:
        slots = slots[[2,0,1]]
    packed = torch.zeros_like(meaning.roles)
    packed[:count] = slots
    language[6][row,sentence] = slots[-1]
    language[10][row,sentence] = packed.flatten()
    language[11][row,sentence] = count
    view = dict(observation)
    view['meanings'] = list(observation['meanings'])
    view['meanings'][row] = meaning
    view['clauses'] = list(observation['clauses'])
    clause = view['clauses'][row]
    if clause is not None:
        view['clauses'][row] = replace(clause, meaning=meaning,
            point=slots[0] if count == 1 else None,
            relation=None if count == 1 else (clause.relation or 'operator'))
    record = observation['record']
    root, end, depth = record.root.clone(),record.end_slots.clone(),record.end_depth.clone()
    root[row],end[row],depth[row] = slots[-1],packed,count
    view['record'] = replace(record,root=root,end_slots=end,end_depth=depth)
    return (state[0],tuple(language),state[2]),view


def scorer(model, state, observation, sentence):
    """Targets remain loss-side data and never become an operation candidate."""
    trials = []
    def score(row, result):
        from ThoughtReferences import open_slots
        bound, view = ((state, observation) if (not accepted(result.meaning)
                         or int(result.meaning.role_mask.sum()) not in (1, 3)) else
                      bound_view(state, observation, sentence, row, result.meaning))
        active = torch.zeros(len(observation['meanings']),device=result.meaning.roles.device,dtype=torch.bool)
        active[row] = True
        presented, comparison = Error(row_mask=active), Error(row_mask=active)
        # Both readers are output-owned. Held forward values survive other
        # owners' sentence updates before the ordinary batch-end backward.
        # Freeze mutable parameter/buffer views once per version. Immutable
        # reader intermediates retain their existing storage, as at the
        # enclosing sentence boundary, rather than cloning every saved use.
        from SentenceCompose import saved_sentence_values
        with saved_sentence_values(model.parameters(), model.buffers()):
            value = model._sentence_reader_error(bound,sentence,active,view,registry=presented)
            reader = getattr(model,'comparison_reader',None)
            if reader is not None and value is not None:
                value = reader(model,bound,sentence,active,view,registry=comparison)
        trials.append((presented,comparison,getattr(model,'_thought_walk',None)))
        data = model.inputSpace.data
        questions = model._sentence_answer_questions
        answer = data.what(questions[row]) if row < len(questions) else None
        if answer is not None and answer.available and answer.kind == 'referent':
            from BindingAnswers import cost
            text_cost = 0. if value is None else value[row] if value.ndim else value
            return (cost(model, result.meaning, answer.what,stream=row) if accepted(result.meaning)
                    else result.meaning.roles.new_ones(())) + text_cost
        return 0. if value is None else value[row] if value.ndim else value
    return score,trials


def train_readers(model,trials):
    if not trials:
        return
    for presented,comparison,trace in trials:
        # Compose's exposure rule: presented reader sees the kept binding;
        # comparison reader sees both trials when a departure was available.
        # Per-trial exposure is distinct from the named loss priority: a
        # batch can contain both a single read and a pair of half exposures.
        model.errors.merge(presented, prefix='thought_answer.',
                           exposure=1. if trace is None else trace.get('reader_weight',0.))
        model.errors.merge(comparison, prefix='thought_comparison.',
                           exposure=1. if trace is None else trace.get('comparison_weight',0.))
