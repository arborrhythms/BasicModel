"""D1: observe both native arrival trials without changing their reads or costs.

Use beside arrival_capture, with ARRIVAL_KEEP_FILE and ARRIVAL_KEEP_CASE.
Only already-produced values are inspected; no working store read, new model
query, random draw, allocation, target, or scoring term is introduced.
"""
import json
import os
from pathlib import Path

import pytest

from arrival_capture import plain


def reading(value):
    from ThoughtReferences import bindings, open_slots
    return None if value is None else dict(refs=plain(value.role_refs),
        open=plain(open_slots(value)), bindings=plain(bindings(value)),
        mask=plain(value.role_mask))


def clause_graph(root):
    nodes, terms = [], []
    seen, composed = {}, {}

    def composition(term):
        if term is None:
            return None
        if id(term) not in composed:
            index = composed[id(term)] = len(terms)
            terms.append(None)
            terms[index] = dict(reference=term.reference, word_id=term.word_id,
                operation=term.operation, formal_name=term.formal_name,
                head_role=term.head_role, order=term.order, leaves=term.leaves,
                children=[composition(child) for child in term.children],
                binding=composition(term.binding))
        return composed[id(term)]

    def visit(node):
        if node is None:
            return None
        if id(node) not in seen:
            index = seen[id(node)] = len(nodes)
            nodes.append(None)
            request = node.reference_request
            nodes[index] = dict(relation=node.relation, refs=node.refs,
                order=node.order, minted=node.minted, meaning=reading(node.meaning),
                subject_word_id=node.subject_word_id,
                request=None if request is None else {key: request.get(key)
                    for key in ('mode', 'role', 'leaves', 'forward_identity')},
                children=[visit(child) for child in node.children],
                companions=[visit(child) for child in node.companions],
                composition=composition(node.composition))
        return seen[id(node)]

    return plain(dict(root=visit(root), nodes=nodes, compositions=terms))


@pytest.fixture(autouse=True)
def observe_arrival_keep(request, monkeypatch):
    if 'test_forced_ordinary_pending_premise_in_both_orders' not in request.node.name:
        return
    from Models import BasicModel
    target = (Path(os.environ['ARRIVAL_KEEP_FILE']) if 'ARRIVAL_KEEP_FILE' in os.environ else
              Path(os.environ['ARRIVAL_KEEP_DIR']) / (request.node.name + '.jsonl'))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    observed = []
    count = 0
    original_cost = BasicModel._sentence_path_cost

    def cost(model, *args, **kwargs):
        result = original_cost(model, *args, **kwargs)
        observation = result[2]
        observed.append(plain(dict(trial=model._sentence_trial,
            meanings=[reading(value) for value in observation['meanings']],
            clauses=[clause_graph(clause) for clause in observation['clauses']],
            programs=[None if program is None else dict(actions=program.actions,
                concept_ids=program.concept_ids, word_ids=program.word_ids)
                for program in observation['entries']],
            reference_plans=[dict(cost=plan['cost'], decisions=plan.get('decisions', ()), arrivals=plan.get('arrivals', ()), items=[{key: item.get(key)
                for key in ('mode', 'role', 'selected', 'candidates', 'features',
                            'probabilities', 'leaves')}
                for item in plan['items']])
                for plan in observation.get('reference_plans', ())],
            reference_credit=observation.get('reference_credit'),
            answer_cost=model._sentence_answer_cost,
            readers=model._sentence_reader_observation,
            registry=model._sentence_cost_registry.breakdown())))
        return result

    original_batch = BasicModel.runBatch

    def batch(model, *args, **kwargs):
        nonlocal count
        result = original_batch(model, *args, **kwargs)
        count += 1
        field = model._last_sentence_field
        trials = [] if field is None else [dict(positions=trial['positions'],
            cost=trial['cost'], reads=[dict(action=read['action'], admitted=read['admitted'])
                for read in trial['reads']]) for trial in field['trials']]
        data = model.inputSpace.data
        sources = kwargs.get('source_rows')
        item = dict(case=os.environ.get('ARRIVAL_KEEP_CASE', request.node.name), batch=count,
            source=os.environ['ARRIVAL_KEEP_SOURCE'], replay=os.environ.get('ARRIVAL_REPLAY'),
            source_rows=sources,
            sentences=None if sources is None else [data.train_input[i] for i in sources],
            supplied_targets=data.text_answers.get('train'),
            readings=list(observed), field=trials,
            credit={key: value for key, value in model._last_sentence_credit.items()
                    if key not in ('actions', 'narrowing_actions', 'readers')},
            words=model._concept_owner()._concept_allocator.word_forms,
            reference_close_audit=getattr(model, '_reference_close_audit', {}),
            mtm={str(key): vars(entry) for key, entry in model._concept_owner().mtm.entries.items()})
        with target.open('a') as stream:
            stream.write(json.dumps(plain(item))+'\n')
        observed.clear()
        return result

    monkeypatch.setattr(BasicModel, '_sentence_path_cost', cost)
    monkeypatch.setattr(BasicModel, 'runBatch', batch)
