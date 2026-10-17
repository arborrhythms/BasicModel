"""One diagnostic reproduction of the aborted run's first batch, not a rerun."""
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import warnings

import numpy as np
import pytest
import torch

from constituent_corpus import DEST, load
from constituent_measurement import observe
from identity_experiment import make_model, restore_warmup, save

from Models import BaseModel, _boundary_registry
from What import What
from ClauseJournal import finish_clause


out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1)
random.seed(20261010); np.random.seed(20261010); torch.manual_seed(20261010)
restore = BaseModel.load_weights
texts, labels = load('train')
indices = np.random.default_rng(20261010).permutation(len(texts)).tolist()[:4]
with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    model = make_model(out, patch)
    try:
        restore_warmup(model, Path('output/item6-orders-paired-accepted-code-20261015/warmup.ckpt'), restore)
        data = model.inputSpace.data
        data.grammar_lessons = {'train': json.loads((DEST/'train.lessons.json').read_text())}
        data.source_addresses = {'train': [dict(document=row['id'], sentence=0,
            document_key=row['id'], sentence_index=0, split='train') for row in texts]}
        optimizer = model.getOptimizer(.003)
        steps = [0]
        step = optimizer.step
        def counted(*args, **kwargs):
            result = step(*args, **kwargs); steps[0] += 1
            return result
        optimizer.step = counted
        audit = []
        native = model._commit_sentence
        def structure(clause):
            return dict(order=clause.order, relation=clause.relation,
                request=None if clause.reference_request is None else clause.reference_request['mode'],
                children=[structure(child) for child in clause.children],
                companions=[structure(child) for child in clause.companions])
        def counts(clause):
            return (tuple(counts(child) for child in clause.children),
                    tuple(counts(child) for child in clause.companions))
        def commit(state, sid, active, observations, predictions, wins):
            for b, win in enumerate(wins.tolist()):
                source = observations[-1 if win else 0]
                program = source['entries'][b]
                frozen = program.detached()
                new = finish_clause(model.languageSpace, frozen,
                    meaning=source['meanings'][b], registry=_boundary_registry(model))
                old = finish_clause(model.languageSpace, replace(frozen, close_references=False),
                    meaning=source['meanings'][b], registry=_boundary_registry(model))
                scored = source['clauses'][b]
                audit.append(dict(text=texts[indices[b]]['text'],
                    old_structure_matches=counts(old) == counts(scored),
                    new_structure_matches=counts(new) == counts(scored),
                    scored=structure(scored), old=structure(old), repaired=structure(new)))
            return native(state, sid, active, observations, predictions, wins)
        model._commit_sentence = commit
        with observe(model) as captured:
            inputs = model.inputSpace.prepPackedInput([[texts[i]['text']] for i in indices])
            model.runBatch(train=True, optimizer=optimizer, batchNum=0, batchSize=4, split='train',
                batch_override=(inputs, torch.empty(4, 0)), source_rows=[[i] for i in indices],
                questions=tuple(What.present(b, split='train') for b in range(4)))
        result = dict(optimizer_steps=steps[0], audit=audit,
            observed=[dict(text=texts[i]['text'], **captured[0, b]) for b, i in enumerate(indices)])
        save(out/'result.json', result)
        assert steps[0] == 2
        assert all(row['new_structure_matches'] for row in audit)
        assert any(not row['old_structure_matches'] for row in audit)
        print(json.dumps(dict(optimizer_steps=steps[0],
            before=sum(row['old_structure_matches'] for row in audit),
            after=sum(row['new_structure_matches'] for row in audit))), flush=True)
    finally:
        model.End()
