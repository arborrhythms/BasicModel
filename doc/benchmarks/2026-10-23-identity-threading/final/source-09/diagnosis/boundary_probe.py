"""Observe the three retained failures without changing values or choices."""
import gzip
import json
import os
from pathlib import Path

import pytest

from arrival_capture import plain


def graph(root):
    seen, nodes = {}, []
    def visit(node):
        if node is None:
            return None
        if id(node) not in seen:
            index = seen[id(node)] = len(nodes)
            nodes.append(None)
            identity = node.identity or {}
            nodes[index] = dict(order=node.order, relation=node.relation, refs=node.refs,
                point=node.point, request=node.reference_request,
                identity={key: identity.get(key) for key in ('handle', 'action', 'event')},
                composition=None if node.composition is None else dict(
                    operation=node.composition.operation, leaves=node.composition.leaves,
                    word_id=node.composition.word_id, reference=node.composition.reference),
                children=[visit(child) for child in node.children],
                companions=[visit(child) for child in node.companions])
        return seen[id(node)]
    return plain(dict(root=visit(root), nodes=nodes))


@pytest.fixture(autouse=True)
def boundary_probe(request, monkeypatch):
    records = []
    def failure(error):
        result = dict(error=type(error).__name__, message=str(error), frames=[])
        trace = error.__traceback__
        while trace:
            frame, local = trace.tb_frame, trace.tb_frame.f_locals
            if frame.f_code.co_name == 'preview':
                result['frames'].append(dict(name='preview', handles=plain(local.get('handles')),
                    handle=plain(local.get('handle')), tag=plain(local.get('tag')),
                    address=plain(local.get('address')), request=plain(local.get('request')),
                    clause=graph(local.get('clause'))))
            if frame.f_code.co_name == 'frozen_values':
                result['frames'].append(dict(name='frozen_values',
                    admitted=graph(local.get('admitted')), scored=graph(local.get('scored'))))
            trace = trace.tb_next
        return result
    from Models import BasicModel
    cost = BasicModel._sentence_path_cost
    def scored(model, *args, **kwargs):
        item = dict(call='score', trial=plain(getattr(model, '_sentence_trial', None)))
        records.append(item)
        try:
            result = cost(model, *args, **kwargs)
        except Exception as error:
            item.update(failure(error))
            raise
        item.update(cost=plain(result[0]), registry=plain(model._sentence_cost_registry.breakdown()),
            answer=plain(model._sentence_answer_cost),
            clauses=[graph(node) for node in result[2]['clauses']])
        return result
    monkeypatch.setattr(BasicModel, '_sentence_path_cost', scored)
    commit = BasicModel._commit_sentence
    def committed(model, state, sid, active, observations, predictions, wins):
        item = dict(call='commit', wins=plain(wins),
            credit=plain(getattr(model, '_last_sentence_credit', None)))
        records.append(item)
        try:
            return commit(model, state, sid, active, observations, predictions, wins)
        except Exception as error:
            item.update(failure(error))
            raise
    monkeypatch.setattr(BasicModel, '_commit_sentence', committed)
    import Generative
    inverse = Generative.inverse_menu
    def menu(*args, **kwargs):
        result = inverse(*args, **kwargs)
        if 'free_policy_has_no_pathwise' in request.node.name:
            records.append(dict(call='inverse', parent=plain(args[1]),
                options=plain(result[0]), others=plain(result[1]), legal=plain(result[2])))
        return result
    monkeypatch.setattr(Generative, 'inverse_menu', menu)
    yield
    path = Path(os.environ['CONTRACT_CAPTURE']).parent/'boundary-probe.json.gz'
    path.write_bytes(gzip.compress((json.dumps(plain(records), indent=2)+'\n').encode(), mtime=0))
