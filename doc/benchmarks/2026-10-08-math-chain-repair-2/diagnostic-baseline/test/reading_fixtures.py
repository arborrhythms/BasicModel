"""Execute explicitly forced readings for mechanism tests, before a row exists.

These helpers supply the live numerical journal that the real word brick
publishes. They never read a completed row or recover a stored derivation.
"""
from dataclasses import replace
import torch


def resolve_reading_references(language, entry, *, frames=(), prediction=None,
                               forced=None):
    """Force a mechanism reading through the live tensor reference chooser.

    This fixture runs before ``record_reading`` executes its operators. The
    production word brick resolves each candidate's operands directly.
    """
    from ReferenceContext import ReferenceBank, reference_requests, resolve_operand
    ids = (entry.reference_ids if entry.reference_ids is not None else entry.concept_ids).clone()
    orders = (entry.reference_orders.clone() if entry.reference_orders is not None
              else torch.full_like(ids, -1))
    values = list(entry.leaves.unbind())
    relations = torch.zeros_like(ids, dtype=torch.bool)
    anchors = tuple(frame for frame in frames if frame.order == 1)
    points = [torch.zeros_like(values[0]) if frame.point is None else frame.point
              for frame in anchors]
    bank = ReferenceBank(
        ids.new_tensor([[frame.row_id for frame in anchors]]),
        torch.stack(points)[None] if points else entry.leaves.new_zeros(1, 0, entry.leaves.shape[-1]),
        torch.ones(1, len(anchors), dtype=torch.bool),
        torch.tensor([[frame.point is None for frame in anchors]], dtype=torch.bool),
        (entry.leaves[0] if prediction is None else prediction.roles[0])[None],
        torch.tensor([prediction is not None]))
    requests = reference_requests(language, entry.actions)
    for leaf, (order, mode) in sorted(requests.items()):
        if order != 1:
            continue
        live_orders = orders.clone()
        for index, (requested, _kind) in requests.items():
            live_orders[index] = requested
        value, identity, relative, available = resolve_operand(
            values[leaf][None, None], ids[leaf].reshape(1, 1), ids.new_tensor([[leaf]]),
            mode=mode, bank=bank,
            live=(torch.stack(values)[None], ids[None], live_orders[None],
                  relations[None], torch.arange(len(values))[None]),
            active=torch.ones(1, 1, dtype=torch.bool),
            forced_reference=None if forced is None else forced.get(leaf))
        if not bool(available.all()):
            raise ValueError('pronoun has no particular in the bounded situation')
        values[leaf], ids[leaf], relations[leaf] = value[0, 0], identity[0, 0], relative[0, 0]
        orders[leaf] = 1
    return replace(entry, reference_ids=ids, reference_orders=orders,
                   reference_values=torch.stack(values), reference_relations=relations)


def record_reading(language, entry):
    import Language
    stack, frames, references = [], [], []
    values = entry.reference_values if entry.reference_values is not None else entry.leaves
    ids = entry.reference_ids if entry.reference_ids is not None else entry.concept_ids
    width = values.shape[-1]
    operations = {}
    for kind, local, word in entry.actions.tolist():
        if kind == 0:
            stack.append((values[word], int(ids[word])))
            frames.append(values.new_zeros(3, width))
            references.append((-1, -1))
            continue
        arity = 2 if kind == 1 else 1
        rules = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
        rule = rules[local]
        operands = stack[-arity:]
        del stack[-arity:]
        name = rule.method_name
        if (kind, local) not in operations:
            cls = Language.GRAMMAR_LAYER_CLASSES[name]
            operations[kind, local] = (cls(width, width) if name in
                ('lift', 'verb', 'lower', 'surface', 'sum', 'implies') else cls())
        result = operations[kind, local].compose(
            *(value.reshape(1, 1, -1) for value, _ in operands)).reshape(-1)
        head = getattr(rule, 'head_role', 0)
        if not head and name in ('lower', 'bind', 'surface', 'preposition'):
            head = 2
        identity = operands[head - 1][1] if head else -1
        stack.append((result, identity))
        frames.append(torch.stack((operands[0][0], operands[1][0] if arity == 2 else result * 0, result)))
        references.append((operands[0][1], operands[1][1] if arity == 2 else -1))
    end = values.new_zeros(3, width)
    for slot, (value, _) in enumerate(stack):
        end[slot] = value
    return replace(entry, operation_values=torch.stack(frames),
                   operation_refs=torch.tensor(references), end_state=end)


def finish_reading(language, entry, **kwargs):
    from ClauseJournal import finish_clause
    entry = record_reading(language, entry)
    return finish_clause(language, entry, **kwargs)


def sentence_state(language, entry, registry):
    from Understanding import SentenceEndState
    from Meaning import ConceptualMeaning
    query = language.program_meaning(entry, registry)
    return SentenceEndState(ConceptualMeaning.from_description(entry.end_state[0]),
        query=query if query is not None and query.mode == 'interrogative' else None)


from contextlib import contextmanager


@contextmanager
def capture_operation_traces(model):
    """Let a test observe the winner's trace before the boundary discards it."""
    from types import SimpleNamespace
    captured = []
    original = model._sentence_observation
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get('admit'):
            trace = model._reconstruction_stack()
            captured.append(SimpleNamespace(**{
                name: getattr(trace, name).detach().clone() for name in (
                    '_choice_rule_ids', '_choice_arities', '_choice_mask',
                    '_choice_positions', '_choice_attempted')}))
        return result
    model._sentence_observation = observe
    try:
        yield captured
    finally:
        model._sentence_observation = original


def force_absolute_reading(model):
    """Force an absolute grammar for untrained provisioning mechanism fixtures.

    Both binary absolute operators remain eligible for the explore deviation;
    a unary negation permits a deviation from a one-slot stop. This is an
    explicit supplied reading, never a claim about learned English parsing.
    """
    layer = model.languageSpace._tree_layer(2)
    binary = layer.chooser.score_binary
    unary = layer.chooser.score_unary
    binary_names = [rule.method_name for rule in model.languageSpace._compose_binary_rules]
    unary_names = [rule.method_name for rule in model.languageSpace._compose_unary_rules]
    allowed = ('lift', 'union', 'intersection', 'sum')
    def binary_scores(*args, **kwargs):
        stop, scores = binary(*args, **kwargs)
        mask = torch.tensor([name in allowed for name in binary_names], device=scores.device, dtype=torch.bool)
        if kwargs.get('op_indices') is not None:
            mask = mask.index_select(0, kwargs['op_indices'])
        return stop, scores.masked_fill(~mask, -torch.inf) + 1e6
    def unary_scores(*args, **kwargs):
        stop, scores = unary(*args, **kwargs)
        indices = kwargs.get('op_indices')
        if (kwargs.get('op_offset') == layer.r_reduce + layer.r_apply
                or (indices is not None and bool((indices >= layer.r_reduce+layer.r_apply).all()))):
            return stop, scores  # supplied grammar does not force attention
        mask = torch.tensor([name in ('not', 'non') for name in unary_names], device=scores.device, dtype=torch.bool)
        if kwargs.get('op_indices') is not None:
            mask = mask.index_select(0, kwargs['op_indices'])
        return stop, scores.masked_fill(~mask, -torch.inf) - 1e6
    layer.chooser.score_binary = binary_scores
    layer.chooser.score_unary = unary_scores


@contextmanager
def capture_readings(model):
    """A caller-owned probe of the open sentence; nothing is attached to rows."""
    captured = {}
    original = model._sentence_observation
    def observe(state, sid, active, **kwargs):
        result = original(state, sid, active, **kwargs)
        captured[sid] = result['entries']
        return result
    model._sentence_observation = observe
    try:
        yield captured
    finally:
        model._sentence_observation = original


def commit_reading(language, registry, entry, store, *, discourse=None, sid=0,
                   trust=.9, document=None, owner=None, active=True):
    """Drive the single production closing with a forced numerical reading."""
    from types import SimpleNamespace
    from Models import BasicModel
    entry = record_reading(language, entry)
    width = entry.leaves.shape[-1]
    if owner is None:
        owner = SimpleNamespace()
    owner.languageSpace, owner.grammatical_thoughts = language, registry
    if not hasattr(language, 'program_meaning'):
        from Language import LanguageSpace
        from types import MethodType
        language.program_meaning = MethodType(LanguageSpace.program_meaning, language)
    owner.conceptualSpace = SimpleNamespace(_ltm_consolidation=store is not None,
        _incoming_trust_multiplier=lambda: trust)
    owner._concept_owner = lambda: owner.conceptualSpace
    owner.symbolSpace = SimpleNamespace(ltm_store=store, expectation=discourse)
    # These are closing/storage fixtures, with no learned thought policy.
    # Unknown references still take the real zero-budget question path.
    from Layers import WhatInteractionMemory
    owner.symbolSpace.what_memory = WhatInteractionMemory(batch=1,capacity=32,detach_mode='episode')
    owner.attention_budget = 0
    owner.training = False
    owner.spaces = []
    readings = (entry if active else None,)
    owner._capture_reading_programs = lambda **kwargs: (readings, {sid: readings})
    owner._expectation_documents_for_slot = lambda *args: [document]
    owner._prime_sentence_symbols = lambda *args: None
    owner._publish_sentence_scratch = lambda *args: None
    owner._sentence_fields = getattr(owner, '_sentence_fields', {})
    owner._query_sentence_depth = 0
    owner._query_ready_rows = None
    from types import MethodType
    for name in ('_assert_queries_outside_sentence','_assert_query_boundary',
                 '_query_boundary_scope','_committed_thought_scope',
                 'run_selected_thought','_run_selected_thought_once',
                 '_what_memory','_end_finished_selected_thought_episodes'):
        setattr(owner,name,MethodType(getattr(BasicModel,name),owner))
    owner._sentence_understandings = {}
    owner._last_sentence_understanding = None

    owner._clause_end_state = BasicModel._clause_end_state
    owner._discard_sentence_record = BasicModel._discard_sentence_record
    owner._sentence_observation = lambda *args, **kwargs: BasicModel._sentence_observation(owner, *args, **kwargs)
    stm = (entry.end_state[None], torch.ones(1, dtype=torch.long),
           torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long),
           torch.full((1, 3), -1, dtype=torch.long), torch.ones(1, 3))
    lang = [torch.zeros(1, 1) for _ in range(26)]
    lang[9] = torch.zeros(1, sid + 1, width)
    lang[13] = entry.end_state.reshape(1, 1, 3 * width).expand(1, sid + 1, -1).clone()
    lang[14] = torch.ones(1, sid + 1, dtype=torch.long)
    lang[20] = torch.zeros(1, 3, 2, dtype=torch.long)
    lang.append(torch.full((1, 3), -1, dtype=torch.long))
    lang.append(torch.zeros_like(lang[4], dtype=torch.bool))
    state = stm, tuple(lang), None
    active = torch.tensor([active], dtype=torch.bool)
    view = owner._sentence_observation(state, sid, active)
    cost, pending = None, None
    if discourse is not None:
        cost, _, pending = discourse.sentence_prediction_cost(
            view['observed_depths'], view['observed'], active, documents=[document],
            layout=view['layout'], role_masks=view['roles'],
            sentence_kinds=[None if meaning is None else meaning.sentence_kind
                            for meaning in view['meanings']])
    BasicModel._commit_sentence(owner, state, sid, active, [view], [pending],
                               torch.zeros(1, dtype=torch.bool))
    return owner, view, cost


def use_eager_reading(monkeypatch):
    """Execute the same loop bodies without graph capture for behavior tests."""
    import util
    def eager_while(condition, body, values):
        while bool(condition(*values)):
            values = body(*values)
        return values
    monkeypatch.setattr(util, "TheCompileBackend", "none")
    monkeypatch.setattr(torch, "while_loop", eager_while)
