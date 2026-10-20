"""Observe the committed identity graph in the unchanged decomposition certificate."""
import json
import os
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def observe_binding_graph(monkeypatch, request):
    import ClauseJournal
    from Models import BasicModel
    from ThoughtReferences import bindings, open_slots
    original = BasicModel.run_selected_thought
    original_finish = ClauseJournal.finish_clause

    def finish(language, program, **kwargs):
        result = original_finish(language, program, **kwargs)
        actions = []
        for i, (kind, local, word) in enumerate(program.actions.tolist()):
            if kind < 0:
                break
            catalog = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
            rule = catalog[local] if kind in (1, 2) else None
            actions.append(dict(action=(kind, local, word), rule=None if rule is None else
                {key: getattr(rule, key, None) for key in ('method_name', 'clause_form', 'head_role',
                    'reference_roles', 'arg_orders', 'determiner_mode')},
                refs=None if program.operation_refs is None else program.operation_refs[i].tolist()))
        def describe(clause):
            return dict(refs=clause.refs, relation=clause.relation,
                bindings=bindings(clause.meaning), request=clause.reference_request,
                children=[describe(child) for child in clause.children])
        report = dict(concepts=program.concept_ids.tolist(),
            references=None if program.reference_ids is None else program.reference_ids.tolist(),
            orders=None if program.leaf_orders is None else program.leaf_orders.tolist(),
            actions=actions, clause=describe(result))
        with Path(os.environ['BIND_GRAPH_TRACE']+'.closings').open('a') as stream:
            stream.write(json.dumps(report, default=str)+'\n')
        return result

    def run(model, meaning, **kwargs):
        result = original(model, meaning, **kwargs)
        questions = getattr(model, '_sentence_answer_questions', ())
        row = kwargs['row']
        answer = model.inputSpace.data.what(questions[row])
        if not answer.available:
            return result
        store = model.symbolSpace.ltm_store
        owner = model._concept_owner()
        rows = []
        for index in range(len(store)):
            value = store.meaning_of(index)
            data = bindings(value)
            rows.append(dict(identity=int(store.row_ids[index]), refs=value.role_refs,
                mask=value.role_mask.tolist(), kind=value.sentence_kind,
                relation=int(store.rel_type[index]), raw_refs=store.refs[index].tolist(),
                bound=data.get('_bound_roles'), open=open_slots(value),
                forward=data.get('_forward_references'), context=data.get('_individual_context'),
                occurrences=data.get('_individual_occurrences')))
        report = dict(test=request.node.nodeid, row=row, target=answer.what,
            known=owner.word_concepts(answer.what), before=meaning.role_refs,
            after=result.meaning.role_refs, rows=rows)
        with Path(os.environ['BIND_GRAPH_TRACE']).open('a') as stream:
            stream.write(json.dumps(report, default=str)+'\n')
        return result

    monkeypatch.setattr(BasicModel, 'run_selected_thought', run)
    monkeypatch.setattr(ClauseJournal, 'finish_clause', finish)
