"""Run one original lesson producer with retained state and keep evidence."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT/'bin'), str(ROOT/'test'),
    str(ROOT/'doc/benchmarks/2026-10-21-item3-5-reference-reconstruction'),
    str(ROOT/'doc/benchmarks/2026-10-23-item3-5-step2c/development')]

import numpy as np
import torch
from arrival_capture import plain
from identity_keep_probe import clause_graph
from bounded_tests import source_snapshot

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def rng():
    return dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state())

def save(value, path):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--stage', required=True,
                        choices=('warmup','stage3','lower','definite','description','adjective','definite_adjective','final'))
    parser.add_argument('--warmup', type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'], text=True).strip():
        raise RuntimeError('lesson source must be clean and frozen')
    args.out = args.out.resolve()
    if args.stage!='warmup' and args.warmup is None:
        parser.error('--warmup is required after the warmup stage')
    args.warmup = None if args.warmup is None else args.warmup.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    capture = args.out.with_name(args.out.name+'-capture')
    capture.mkdir()
    source = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    audit = source_snapshot(ROOT)
    manifest = dict(source=source, stage=args.stage, warmup=None if args.warmup is None else str(args.warmup),
        warmup_sha256=None if args.warmup is None else digest(args.warmup), files=audit,
        started=datetime.now(timezone.utc).isoformat())
    save(manifest,capture/'start.json')
    import identity_experiment
    from Models import BasicModel
    native_model = identity_experiment.make_model
    batch, commit = BasicModel.runBatch, BasicModel._commit_sentence
    models, calls = [], []
    def state(model, optimizer=None):
        return dict(state_dict=model.state_dict(), vocab_extras=model._collect_vocab_extras(),
            structural_extras=model._collect_structural_extras(), rng=rng(),
            optimizer=None if optimizer is None else optimizer.state_dict())
    def create(*values, **options):
        model = native_model(*values, **options)
        models.append(model)
        torch.save(state(model),capture/'initial.pt')
        return model
    def run_batch(model, *values, **options):
        number = len(calls)
        torch.save(rng(),capture/f'batch-{number:03d}-rng.pt')
        torch.save(dict(state(model,options.get('optimizer')), args=values,
            kwargs={k:v for k,v in options.items() if k!='optimizer'}),capture/'last-batch-start.pt')
        calls.append(dict(batch=number, training=options.get('train'),
            source_rows=options.get('source_rows')))
        save(plain(calls),capture/'batches.json')
        try:
            return batch(model,*values,**options)
        except BaseException:
            torch.save(state(model,options.get('optimizer')),capture/'failure.pt')
            raise
    def keep(model, state_value, sid, active, observations, predictions, wins):
        row = dict(batch=len(calls)-1,sentence=sid,credit=model._last_sentence_credit,
            trials=[dict(clauses=[clause_graph(c) for c in o['clauses']],
                plans=o.get('reference_plans')) for o in observations])
        with gzip.open(capture/'keep.jsonl.gz','at') as stream:
            stream.write(json.dumps(plain(row))+'\n')
        return commit(model,state_value,sid,active,observations,predictions,wins)
    identity_experiment.make_model = create
    BasicModel.runBatch, BasicModel._commit_sentence = run_batch, keep
    try:
        module = importlib.import_module('warmup' if args.stage=='warmup' else
            'constituent_order' if args.stage=='stage3' else 'np_experiment')
        module.run(args)
        manifest['completed'] = True
    except BaseException:
        manifest.update(completed=False,error=traceback.format_exc())
        if models and not (capture/'failure.pt').exists():
            torch.save(state(models[-1]),capture/'failure.pt')
        raise
    finally:
        manifest.update(finished=datetime.now(timezone.utc).isoformat(),
            source_unchanged=source_snapshot(ROOT)==audit)
        save(manifest,capture/'run.json')
        BasicModel.runBatch, BasicModel._commit_sentence = batch, commit
        identity_experiment.make_model = native_model

if __name__=='__main__':
    main()
