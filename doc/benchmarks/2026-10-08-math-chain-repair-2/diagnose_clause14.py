"""Replay saved development entropy to inspect a failed clause admission.

This is diagnosis of a retained failed development learner, never a fresh
measurement attempt. The observer and the ordinary training path stay live.
"""
import json
from pathlib import Path
import pickle
import random
import sys
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test'), str(HERE)]
import development_epoch
from ClauseRow import ClauseRows
from Models import BasicModel
import ClauseJournal
from math_chain_corpus import ChainDocument, MathChainCorpus

source, output = map(Path, sys.argv[1:])
presentation = {split: tuple(ChainDocument(**dict(item,
    sentences=tuple(item['sentences']),
    pair=None if item['pair'] is None else tuple(item['pair']),
    steps=tuple(tuple(step) for step in item['steps']),
    statement_references=tuple(tuple(ref) for ref in item['statement_references'])))
    for item in documents)
    for split, documents in json.loads((source/'presentation.json').read_text()).items()}
current, programs = {}, []


def encode(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def field(value):
    return dict(relation=value.relation, refs=value.refs,
        slots=value.slots.shape[0], meaning=value.meaning,
        children=[field(child) for child in value.children],
        companions=[field(child) for child in value.companions])


ordinary_finish = ClauseJournal.finish_clause
ordinary_write = ClauseRows.write_clause
ordinary_batch = BasicModel.runBatch
ordinary_presentation = MathChainCorpus.presentation
presentation_calls = 0


def saved_presentation(self, **kwargs):
    global presentation_calls
    presentation_calls += 1
    return presentation if presentation_calls == 1 else ordinary_presentation(self, **kwargs)


def finish(language, program, **kwargs):
    result = ordinary_finish(language, program, **kwargs)
    rules = (language._compose_binary_rules, language._compose_unary_rules)
    actions = []
    for kind, local, word in program.actions.detach().cpu().tolist():
        if kind < 0:
            break
        rule = None if kind == 0 else repr(rules[kind-1][local])
        actions.append(dict(kind=kind, local=local, word=word, rule=rule))
    programs.append(dict(actions=actions, refs=program.reference_ids,
        concept_ids=program.concept_ids, reference_relations=program.reference_relations,
        operation_refs=program.operation_refs, operation_relations=program.operation_relations,
        result=field(result)))
    return result


def write(store, clause, **kwargs):
    try:
        return ordinary_write(store, clause, **kwargs)
    except ValueError as error:
        target = output/'clause-failure.json'
        if not target.exists():
            frames, tb = [], error.__traceback__
            while tb is not None:
                local = tb.tb_frame.f_locals
                if tb.tb_frame.f_code.co_name == 'preflight':
                    value = local.get('value')
                    ref = local.get('ref')
                    index = store.index_of_row(ref) if type(ref) is int else None
                    frames.append(dict(field=None if value is None else field(value),
                        ref=ref, slot=local.get('slot'), relation=local.get('relation'),
                        row_index=index, retained=None if index is None else store.row(index)))
                tb = tb.tb_next
            target.write_text(json.dumps(encode(dict(error=repr(error), batch=current,
                frames=frames, programs=programs, clause=field(clause))), indent=2)+'\n')
        raise


def batch(model, *args, **kwargs):
    programs.clear()
    rows = list(kwargs.get('source_rows', ()))
    split = kwargs.get('split', 'train')
    current.update(rows=rows, texts=[getattr(model.inputSpace.data, split+'_input')[row]
                                    for row in rows])
    return ordinary_batch(model, *args, **kwargs)


torch.set_num_threads(1)
with (source/'initial-rng.pkl').open('rb') as stream:
    states = pickle.load(stream)
torch.set_rng_state(states['torch'])
np.random.set_state(states['numpy'])
random.setstate(states['python'])
with patch.object(ClauseJournal, 'finish_clause', finish), \
     patch.object(ClauseRows, 'write_clause', write), \
     patch.object(BasicModel, 'runBatch', batch), \
     patch.object(MathChainCorpus, 'presentation', saved_presentation):
    try:
        development_epoch.main(output, 'answer_and_expectation')
    finally:
        (output/'diagnostic.json').write_text(json.dumps(dict(kind='development replay',
            source=str(source), entropy_restored=True, saved_presentation=True,
            measurement=False), indent=2)+'\n')
