"""Complete sentence derivations, with one keep and one credit per departure."""
from dataclasses import dataclass, replace
import torch
from Layers import Error
from Understanding import Understanding, SentenceEndState
from Meaning import ConceptualMeaning


def walk_options(trace, departure, *, reuse=None):
    return ({} if trace is None or not bool((departure >= 0).any() | (False if reuse is None else reuse.any())) else
            dict(exploit_actions=trace[0], departure=departure, exploit_prefix=trace[5], reuse_rows=reuse))


class Thinking:
    def __init__(self, base=None, departure=None):
        self.base, self.departure = base, departure
        self.traces, self.absences = {}, {}

    def thought_trace(self, row):
        old = None if self.base is None else self.base.traces.get(row)
        at = -1 if self.departure is None else int(self.departure[row])
        if old is None or at < 0:
            trace, fork = dict(actions=[], eligible=[], choices=[], collect_fork=True), None
        else:
            fork = old['forks'][at]
            trace = dict(actions=old['actions'][:at], eligible=old['eligible'][:at],
                choices=old['choices'][:at], departure=at, excluded=fork['excluded'])
        self.traces[row] = trace
        return trace, fork

    def eligible(self, active):
        width = max((len(trace['eligible']) for trace in self.traces.values()), default=0)
        result = active.new_zeros(len(active), width)
        for row, trace in self.traces.items():
            result[row, :len(trace['eligible'])] = result.new_tensor(trace['eligible'])
        return result


@dataclass
class Path:
    state: tuple
    reconstruction: tuple
    observation: dict
    pending: object
    registry: Error
    parts: torch.Tensor
    thinking: Thinking
    effects: object
    before_output: object
    construction: object = None
    output_trace: object = None
    draw: object = None


def fields(model, observation, sid):
    """Resolve the scored reference plan without creating an occurrence."""
    from ThoughtClosing import closing_question
    from TimelineRead import request, phase_request
    from ThoughtReferences import bindings
    registry = model.grammatical_thoughts
    store = model.symbolSpace.ltm_store
    resolver = model._concept_owner().components.reference_close
    def semantic_clause(clause, original=None):
        """Give temporary operands the same semantic names as the row writer."""
        from ClauseRow import ClausePredicate, predicate_identity, predicate_point
        original = clause if original is None else original
        children = tuple(semantic_clause(child,
            original.children[i] if i < len(original.children) else child)
            for i, child in enumerate(clause.children))
        value = replace(clause.meaning, roles=original.meaning.roles,
                        role_mask=original.meaning.role_mask)
        refs = []
        for reference in clause.refs:
            if isinstance(reference, ClausePredicate):
                refs.append(('sym', reference.identity))
            elif isinstance(reference, tuple):
                refs.append(('constituent', reference[1]))
            else:
                refs.append(None if reference == -1 else store.semantic_reference(reference))
        if clause.relation in ('part', 'equal', 'implies'):
            roles = value.roles.clone()
            roles[1] = predicate_point(clause.relation, roles[1])
            refs[1] = ('sym', predicate_identity(clause.relation))
            value = replace(value, roles=roles)
        # Completed journal children can include unreferenced side clauses.
        # Only semantic operands belong to this thought's working structure.
        used = sorted({ref[1] for ref in refs if ref is not None and ref[0] == 'constituent'})
        positions = {old: new for new, old in enumerate(used)}
        refs = tuple(('constituent', positions[ref[1]]) if ref is not None and ref[0] == 'constituent'
                     else ref for ref in refs)
        value = replace(value, role_refs=refs, constituents=tuple(children[i].meaning for i in used))
        return replace(clause, meaning=value, point=original.point, children=children)
    result = []
    for row, clause in enumerate(observation['clauses']):
        if clause is None:
            result.append(None)
            continue
        plans = observation.get('reference_plans')
        original = clause
        if plans:
            clause = resolver.prepare(clause, plans[row], preview=True)
        clause = semantic_clause(clause, original)
        meaning = (clause.meaning if clause.relation is not None else
            replace(ConceptualMeaning.from_description(clause.point),
                    polarity=clause.meaning.polarity, sentence_kind='idea'))
        program = observation['entries'][row]
        query = closing_question(meaning, clause, meaning, program, store=store)
        # No current occurrence exists to exclude. Retained event origins
        # remain available through the same timeline request builder.
        event = request(clause, {}, store)
        if query is not None and event is not None:
            query = replace(query, bindings=dict(bindings(query), _query_event=event))
            if clause.identity.get('case_value') is not None:
                query = phase_request(query, clause.identity['case_value'])
        when = model.when_encoding.encode(model._sentence_ticks[int(sid)]).to(meaning.roles)
        result.append(SentenceEndState(meaning, where=clause.where, when=when, query=query,
            order=clause.order, evidence=clause.evidence,
            trust=model.conceptualSpace._incoming_trust_multiplier()))
    return result


def residual(registry, trace, active, owner):
    if trace is None:
        return
    for lane, name in enumerate(('form', 'for', 'against')):
        registry.error(owner + '.walk_residual.' + name, trace[3][:, lane], 1.,
            mask=active, category=owner, objective='reconstruction',
            side='R' if owner == 'reconstruction' else 'A')


def output(model, state, observation, sid, active, registry, *, trace=None, departure=None, reuse=None):
    """The single reader takes a thought answer, or the numeric sentence state."""
    if not model.inputSpace.data.has_supervised_outputs:
        return None, None, None
    if not model.answer_synthesis:
        cost = model._sentence_numeric_error(state, sid, active, observation, registry=registry)
        return cost, None, None
    fields = tuple(observation['thought_fields'])
    understanding = Understanding(conceptual_state=observation['record'].root,
        sentence_states=fields, sentence_records=(observation['record'],))
    from What import What, WhatAnswer
    supplied = model._sentence_answer_questions
    questions = supplied or tuple(What.inference(row) for row in range(len(active)))
    derivation = model._resolve_answer(understanding, questions,
                                      selected_thoughts=observation['thoughts'])
    options = walk_options(trace, departure, reuse=reuse)
    construction = model.reverseOutput(understanding, derivation,
                                      _walk_options=dict(options, return_candidates=True))
    cost = answer_cost(model, construction, observation['thoughts'], questions, active, registry, supplied=bool(supplied))
    residual(registry, model._last_output_walk_trace, active, 'output')
    model._sentence_answer_cost = cost
    model._sentence_answer_predictions = construction.actual.detach()
    model._sentence_answer_raw_cost = cost.detach()
    return cost, construction, model._last_output_walk_trace


def answer_cost(model, construction, thoughts, questions, active, registry, *, supplied=True):
    """Score one complete output, including silence and a requested binding."""
    from What import WhatAnswer
    answers = (tuple(model.inputSpace.data.what(question) for question in questions) if supplied
               else tuple(WhatAnswer.silence(question) for question in questions))
    model._set_answer_silence_mask(answers)
    values = [answer.what if answer.available else '' for answer in answers]
    cost = model._output_target_cost(construction, values)
    from BindingAnswers import cost as binding_cost
    from ThoughtAnswer import accepted
    thoughts = dict(thoughts)
    binding = []
    for row, answer in enumerate(answers):
        result = thoughts.get(row)
        binding.append(cost.new_zeros(()) if not answer.available or answer.kind != 'referent'
            else cost.new_ones(()) if result is None or not accepted(result.meaning)
            else binding_cost(model, result.meaning, answer.what, stream=row))
    cost = cost + torch.stack(binding)
    registry.error('answer', cost, 1., mask=active, category='prediction', objective='output')
    return cost


def finish(model, state, sid, active, *, base=None, draw=None):
    """Run only the suffix after this path's departure, ending before a write."""
    if base is not None and not bool((draw['walk'] >= 0).any()):
        return replace(base, draw=draw)
    from SentenceField import publish, reconstruct
    from WalkTrials import ThoughtTrialState
    from SentenceCredit import components
    from ThoughtClosing import close
    reading = base is None or bool(((draw['walk'] >= 0) & (draw['walk'] < 4)).any())
    regeneration = base is None or reading or bool((draw['reconstruction_round'] >= 0).any())
    rethink_rows = active if base is None else active & (draw['walk'] >= 0) & (draw['walk'] < 6)
    rethink = bool(rethink_rows.any())
    before = ThoughtTrialState(model)
    original_errors = getattr(model, '_sentence_close_errors', None)
    original_fields = dict(model._sentence_fields)
    old_thinking = getattr(model, '_path_thinking', None)
    try:
        if reading:
            model._path_reconstruction_options = (None if base is None or not model._sentence_reconstruction else
                walk_options(base.observation['generation'].trace, draw['reconstruction_round'],
                             reuse=(draw['walk'] >= 5) | (draw['walk'] < 0)))
            _, rebuilt, observation, pending = model._sentence_path_cost(state, sid, active)
            registry = model._sentence_cost_registry
            expectation = model._sentence_expectation_comparison
        else:
            rebuilt, observation, pending = base.reconstruction, dict(base.observation), base.pending
            registry = Error(row_mask=active)
            registry._terms = {name: dict(term) for name, term in base.registry._terms.items()
                if term.get('objective') != 'output' and not name.startswith('output.')}
            expectation = base.parts[:, 1]
            if regeneration:
                registry._terms = {name: term for name, term in registry._terms.items()
                    if not name.startswith(('reconstruction_order_zero.', 'reconstruction.walk_residual.',
                                            'reconstruction_reverse.', 'reconstruction.free_', 'reconstruction.coverage'))}
                model._path_reconstruction_options = walk_options(base.observation['generation'].trace,
                                                                  draw['reconstruction_round'], reuse=(draw['walk'] >= 5) | (draw['walk'] < 0))
                model._sentence_walk_residuals = []
                rebuilt = reconstruct(model, observation['record'], registry, active,
                                      previous=observation['reverse'])
                observation.update(generation=model._sentence_generation,
                    order_zero_field=model._sentence_generation.field,
                    order_zero_reverse=model._sentence_order_zero_reverse,
                    input_reconstruction=model._input_reconstructions[int(sid)])
                residual(registry, model._sentence_generation.trace, active, 'reconstruction')
        if 'order_zero_field' in observation:
            publish(model, observation['order_zero_field'])
        model._sentence_close_errors = registry
        supervised = model.inputSpace.data.has_supervised_outputs
        thinking = Thinking(None if base is None else base.thinking,
                            None if draw is None else draw['thought_round'])
        if base is not None:
            thinking.traces.update(base.thinking.traces)
            thinking.absences.update({row: list(values) for row, values in base.thinking.absences.items()})
        if supervised and rethink:
            if base is not None:
                ThoughtTrialState.keep([base.before_output, before], rethink_rows.long())
                publish(model, observation['order_zero_field'])
            preview = fields(model, observation, sid)
            if base is not None:
                preview = [value if bool(rethink_rows[row]) else base.observation['thought_fields'][row]
                           for row, value in enumerate(preview)]
            model._sentence_fields[sid] = tuple(preview)
            inputs = {row: field.query if field.query is not None else field.meaning
                      for row, field in enumerate(preview) if field is not None}
            ready = tuple(row for row, field in enumerate(preview) if field is not None and bool(rethink_rows[row]))
            model._path_thinking = thinking
            images = model._sentence_closing_images(observation, active, pending)
            # The reading prime belongs to composition. Thought now starts
            # before the old outer finally that clears these transient views.
            reducer = model._stm_reducer()
            if reducer is not None:
                reducer._expectation_prime = None
                reducer._symbol_expectation_read = None
            with model._committed_thought_scope(sid, ready):
                thoughts = close(model, preview, sentence=sid, ready=ready, images=images,
                                 pending=pending)
            if base is not None:
                thoughts = tuple(sorted((*thoughts, *(item for item in base.observation['thoughts']
                    if not bool(rethink_rows[item[0]]))), key=lambda item: item[0]))
            observation.update(thoughts=thoughts, thought_fields=tuple(preview),
                               thought_absences=thinking.absences, thought_inputs=inputs)
        elif supervised:
            thinking = base.thinking
            base.before_output.restore()
        before_output = ThoughtTrialState(model)
        if supervised:
            cost, construction, output_trace = output(model, state, observation, sid, active, registry,
                trace=None if base is None else base.output_trace,
                departure=None if draw is None else draw['output_round'],
                reuse=None if draw is None else draw['walk'] < 0)
            observation['answer_predictions'] = (None if cost is None else
                model._sentence_answer_predictions)
        else:
            cost, construction, output_trace = None, None, None
        effects = ThoughtTrialState(model)
        parts = components(registry, expectation, observation['record'].root.sum(-1)*0, answer=cost)
        return Path(state, rebuilt, observation, pending, registry, parts, thinking, effects, before_output,
                    construction, output_trace, draw)
    finally:
        before.restore()
        model._sentence_fields = original_fields
        model._sentence_close_errors = original_errors
        model._path_thinking = old_thinking
        model._path_reconstruction_options = None


def credit(model, exploit, path, active):
    """A departure's sole reward is its complete path's change in R + A."""
    from WalkTrials import generate_credit
    from SentenceCredit import score_function
    draw = path.draw
    costs = torch.stack((exploit.parts[:, 0] + exploit.parts[:, 2],
                         path.parts[:, 0] + path.parts[:, 2]), -1)
    for name, first, other, owner in (
            ('reconstruction', exploit.observation.get('generation'), path.observation.get('generation'), 'reconstruction'),
            ('output', exploit.output_trace, path.output_trace, 'output')):
        first = first.trace if name == 'reconstruction' and first is not None else first
        other = other.trace if name == 'reconstruction' and other is not None else other
        if first is None or other is None or not bool((draw[name + '_round'] >= 0).any()):
            continue
        audit = dict(departure=draw[name + '_round'], greedy=first[0], explore=other[0],
            eligible=first[1], costs=costs, proposal_rounds=draw['walk_count'] * draw['walk_rounds'])
        value = generate_credit(model.languageSpace, audit, other[-1], other[-2])
        path.registry.add(owner + '.walk_credit', value, category=owner,
            objective='chooser' if owner == 'output' else 'reconstruction', side='A' if owner == 'output' else 'R')
    for row, at in enumerate(draw['thought_round'].tolist()):
        if at < 0:
            continue
        choice = path.thinking.traces[row]['choices'][at]
        value, _ = score_function(choice['probability'].reshape(1, 1),
            costs.new_tensor([[float(draw['walk_count'][row] * draw['walk_rounds'][row]) * choice['alternatives']]]),
            active[row:row+1, None], costs[row:row+1])
        category = choice.get('category', 'reconstruction')
        rows = costs.new_zeros(len(active))
        rows[row] = value[0]
        path.registry.add('output.thought_score_function', rows,
            category='output', objective='chooser' if category == 'reconstruction' else category, side='A')


def objectives(paths, parts, active, draws):
    """Two reconstruction/expectation exposures and one answer exposure total."""
    from SentenceCredit import reader_weights
    from ObjectiveOwnership import registry_costs
    from SentenceTraining import combine
    weights = reader_weights(parts, active, draws)
    result = {}
    scale = 2. / len(paths)
    for index, path in enumerate(paths):
        registry = Error(row_mask=path.registry.row_mask)
        registry._terms = dict(path.registry._terms)
        for name, term in tuple(registry._terms.items()):
            if name.startswith('output.walk_residual.'):
                if not bool(weights[:, index].any()):
                    del registry._terms[name]
                else:
                    registry._terms[name] = dict(term,
                        multiplier=term.get('multiplier', 1.) * weights[:, index])
        costs = registry_costs(registry, reader_rows=weights[:, index])
        combine(result, {name: value * (scale if name.split('.')[0] in ('reconstruction', 'expectation')
                                       and not name.endswith('.A') else 1.)
                         for name, value in costs.items()})
    return result, weights


def think_kept(model, paths, selected, sid, active):
    """Without an answer target, only the reconstruction-kept path thinks."""
    from ThoughtClosing import close
    for index, path in enumerate(paths):
        ready_mask = active & (selected == index)
        if not bool(ready_mask.any()):
            continue
        preview = fields(model, path.observation, sid)
        inputs = {row: field.query if field.query is not None else field.meaning
                  for row, field in enumerate(preview) if field is not None}
        ready = tuple(row for row, field in enumerate(preview) if field is not None and bool(ready_mask[row]))
        model._sentence_fields[sid] = tuple(preview)
        model._path_thinking = path.thinking
        try:
            with model._committed_thought_scope(sid, ready):
                results = close(model, preview, sentence=sid, ready=ready,
                    images=model._sentence_closing_images(path.observation, active, path.pending),
                    pending=path.pending)
            path.observation.update(thoughts=results, thought_fields=tuple(preview),
                                    thought_absences=path.thinking.absences, thought_inputs=inputs)
        finally:
            model._path_thinking = None


def keep_output(model, paths, selected, sid):
    """Retain the already realized per-stream output for the public handoff."""
    from SentenceBoundary import detached
    from Output import AnswerConstruction
    rows = [paths[index].construction for index in selected.tolist()]
    if not rows or all(row is None for row in rows):
        model._kept_path_output = None
        return
    if any(row is None for row in rows):
        raise RuntimeError('one output modality must serve every stream of a sentence')
    def stack(name):
        values = [getattr(value, name)[row:row+1] for row, value in enumerate(rows)]
        shape = tuple(max(value.shape[axis] for value in values) for axis in range(1, values[0].ndim))
        padded = []
        for value in values:
            pads = tuple(n for axis in reversed(range(1, value.ndim)) for n in (0, shape[axis-1]-value.shape[axis]))
            padded.append(torch.nn.functional.pad(value, pads))
        return torch.cat(padded, 0)
    derivation = rows[0].derivation
    if derivation is not None:
        derivations = [value.derivation for value in rows]
        thoughts = tuple((row, result) for row, item in enumerate(derivations)
                         if (result := dict(item.selected_thoughts).get(row)) is not None)
        fields = {name: tuple(getattr(item, name)[row] for row, item in enumerate(derivations))
                  for name in ('sentence_states', 'answer_meanings', 'row_sources')}
        # This report carries the kept rows' semantic provenance. Its surface
        # has already been realized; no other path's prepared root is retained.
        for name in ('conceptual_answer', 'conditioning_context', 'answer_symbol'):
            values = [getattr(item, name) for item in derivations]
            if all(torch.is_tensor(value) for value in values):
                fields[name] = torch.cat([value[row:row+1] for row, value in enumerate(values)], 0)
        derivation = replace(derivation, **fields, selected_thoughts=thoughts)
    result = AnswerConstruction(actual=stack('actual'), derivation=derivation,
        concepts=stack('concepts'), percepts=stack('percepts'), surface=stack('surface'),
        trace=rows[0].trace, texts=tuple(value.texts[row] for row, value in enumerate(rows)))
    model._kept_path_output = detached(result)
    model._kept_path_output_actions = torch.stack([paths[index].output_trace[0][row]
                                                  for row, index in enumerate(selected.tolist())]).detach()


def output_only(model, understanding, questions, *, score=None):
    """A batch without a sentence read uses the same path judge, over output."""
    from SentenceCredit import comparison, departure, reader_weights
    from WalkTrials import ThoughtTrialState, generate_credit
    closed = any(field is not None and field.thought_completed
                 for field in understanding.sentence_states)
    if closed:
        # An unsupervised sentence already ran its kept thought. Its report
        # reads that conclusion, including silence, without reopening the
        # controller or falling back to the input meaning as an answer.
        derivation = model._resolve_answer(understanding, questions,
            selected_thoughts=model._last_closing_thoughts)
    else:
        derivation = model.resolveAnswer(understanding, questions)
    before = ThoughtTrialState(model)
    constructions, traces, costs, draws = [], [], [], []
    try:
        first = model.reverseOutput(understanding, derivation,
                                    _walk_options=dict(return_candidates=True))
        constructions.append(first)
        traces.append(model._last_output_walk_trace)
        effects = [ThoughtTrialState(model)]
        active = torch.ones(len(first.actual), dtype=torch.bool, device=first.actual.device)
        empty = active[:, None][:, :0]
        costs.append(first.actual.new_zeros(len(active)) if score is None else score(first))
        for _ in range(model.explore_paths if model.training and torch.is_grad_enabled() and score is not None else 0):
            draw = departure(None, empty, active=active, output=traces[0][1], used=draws)
            draws.append(draw)
            if not bool((draw['walk'] >= 0).any()):
                constructions.append(first)
                traces.append(traces[0])
                costs.append(costs[0])
                effects.append(effects[0])
                continue
            before.restore()
            other = model.reverseOutput(understanding, derivation,
                _walk_options=dict(walk_options(traces[0], draw['output_round'],
                    reuse=draw['walk'] < 0), return_candidates=True))
            constructions.append(other)
            traces.append(model._last_output_walk_trace)
            costs.append(score(other))
            effects.append(ThoughtTrialState(model))
        values = torch.stack(costs, 1)
        parts = torch.stack((torch.zeros_like(values), torch.zeros_like(values), values), -1)
        winner = comparison(parts, active)['winner']
        weights = reader_weights(parts, active, draws)
        for index in range(len(costs)):
            if bool(weights[:, index].any()):
                model.errors.add('answer', costs[index] * weights[:, index], category='prediction', objective='output')
            if index:
                draw = draws[index-1]
                audit = dict(departure=draw['output_round'], greedy=traces[0][0], explore=traces[index][0],
                    eligible=traces[0][1], costs=values[:, [0, index]],
                    proposal_rounds=draw['walk_count'] * draw['walk_rounds'])
                model.errors.add('output.walk_credit', generate_credit(model.languageSpace, audit,
                    traces[index][-1], traces[index][-2]), category='output', objective='chooser', side='A')
        from types import SimpleNamespace
        paths = [SimpleNamespace(construction=c, output_trace=t) for c, t in zip(constructions, traces)]
        ThoughtTrialState.keep(effects, winner)
        keep_output(model, paths, winner, None)
        return model._kept_path_output
    finally:
        before.restore(effects_only=True)
