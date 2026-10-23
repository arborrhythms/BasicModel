"""Regenerate only the original stages-1–2 input artifact for stage 3."""
from datetime import datetime, timezone
import json
import random
import time
import warnings

import numpy as np
import pytest
import torch

from attention_curriculum_gates import PERCEPTS, INTERRUPTION, NOVEL, present
from constituent_budget import paired_updates_only
from identity_corpus import KINDS, PROPERTIES, VERBS
from identity_experiment import make_model, reset, save, digest
from paired_identity_experiment import kept_programs


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(20261010); np.random.seed(20261010); torch.manual_seed(20261010)
    schedule = [('first', PERCEPTS), ('repeat', PERCEPTS), ('interruption', INTERRUPTION),
                ('after_gap', PERCEPTS), ('known', PERCEPTS), ('novel_first', NOVEL), ('novel_second', NOVEL)]
    atoms = list((*KINDS, *PROPERTIES, *VERBS))
    schedule += [(f'lexical:{repeat}:{start}', atoms[start:start+4])
                 for repeat in range(4) for start in range(0, len(atoms), 4)]
    protocol = dict(seed=20261010, learning_rate=.003, batch_size=4,
        training_presentations=76, paired_optimizer_steps=38,
        producer='test/paired_identity_experiment.py: stages 1–2 schedule',
        schedule=schedule, following_identity_curriculum=False,
        historical_singleton_gate='belongs to the following identity curriculum; stage3 did not read it',
        batch_end_optimizer='disabled', no_retries=True,
        started=datetime.now(timezone.utc).isoformat())
    save(args.out/'protocol.json',protocol)
    records, reports, current = [], [], dict(phase='warmup')
    started = time.monotonic()
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = make_model(args.out,patch)
        optimizer = model.getOptimizer(lr=.003)
        try:
            with paired_updates_only(model,optimizer) as updates, kept_programs(model,records,current):
                for batch,(name,fields) in enumerate(schedule):
                    current.update(batch=batch,fields=fields)
                    report = present(model,dict(fields=fields),train=True,optimizer=optimizer)
                    assert len(report['trials']) == 2, 'paired training did not expose both trials'
                    reports.append(dict(name=name,fields=fields,training=True,**report))
                    save(args.out/'presentations.json',reports)
                    save(args.out/'singleton-presentations.json',records)
                    progress = dict(batch=batch+1, presentations=sum(len(row['fields']) for row in reports),
                        updates=dict(updates), elapsed_seconds=time.monotonic()-started)
                    save(args.out/'progress.json',progress)
                    print(json.dumps(progress),flush=True)
                assert sum(len(fields) for _,fields in schedule) == 76
                assert updates['paired'] == 38
                reset(model)
                artifact = args.out/'warmup.ckpt'
                model.save_weights(str(artifact))
                save(args.out/'results.json',dict(completed=True,protocol=protocol,
                    updates=dict(updates),artifact=dict(path=str(artifact),sha256=digest(artifact)),
                    learning_certificate=False,elapsed_seconds=time.monotonic()-started))
        finally:
            model.End()
