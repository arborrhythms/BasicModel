"""Exact whole-sentence production probes from frozen pairs, without targets in input."""
import json
from pathlib import Path


def production_probe(model, rows):
    """Read on an isolated model using the existing future question interface.

    This reports production only. It does not select a learning signal or
    certify acquisition after the two training-signal nulls.
    """
    import torch
    from eval_nanochat_grammar import frozen_online_learning, _reset_model
    from What import WhatQuestion
    data = model.inputSpace.data
    observations = []
    with frozen_online_learning(model), torch.no_grad():
        for row in rows:
            _reset_model(model)
            produced = []
            for index, source in enumerate(row['sources']):
                last = index+1 == len(row['sources'])
                with data.runtime_batch([source], document_keys=[('game', row['key'])]):
                    x = model.inputSpace.prepInput([source])
                    y = model.outputSpace.prepOutput([torch.zeros(1)])
                    model.runBatch(train=False, batchNum=index, batchSize=1,
                        split='runtime', batch_override=(x,y), source_rows=[0],
                        questions=(WhatQuestion.future(0,1,split='runtime'),) if last else ())
                    if last:
                        produced = list(getattr(getattr(model,'_last_answer_construction',None),'texts',()) or ())
                model.flush_word_buffers()
                model.dispatch_per_row_reset([last])
                model.dispatch_soft_reset()
                model.post_tick_compact()
            observations.append(dict(key=row['key'],target=row['target'],produced=produced,
                correct=produced == [row['target']]))
    return dict(status='complete' if observations else 'unavailable',
        signal='future question; diagnostic only, learning signal unqualified',
        count=len(observations), correct=sum(row['correct'] for row in observations),
        value=None if not observations else sum(row['correct'] for row in observations)/len(observations),
        rows=observations)


def game_rows(root, *, split='test'):
    from eval_operators import digest
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    games = {}
    for entry in manifest['files']:
        path = root/entry['path']
        if digest(path) != entry['sha256']:
            raise ValueError('frozen game has changed: ' + entry['path'])
        games.setdefault(entry['game'], []).extend(row for row in map(json.loads,path.read_text().splitlines())
            if row['split'] == split)
    return games
