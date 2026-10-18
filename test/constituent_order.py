"""Stage 3 prerequisite for stage 4e; one fixed run, stop at its order gate."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import time
import traceback
import warnings

import numpy as np
import pytest
import torch

from attention_curriculum_gates import STAGES, ORDER_CE_MAX, present
from constituent_budget import paired_updates_only
from identity_experiment import make_model, restore_warmup, save

SEED, BATCHES, RATE = 20261010, 64, .003


def run(args):
    from Models import BaseModel
    restore = BaseModel.load_weights
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    protocol = dict(seed=SEED, batches_per_lesson=BATCHES, batch_size=4,
        training_presentations=2*BATCHES*4, paired_optimizer_steps=2*BATCHES*2,
        evaluation_presentations=16, learning_rate=RATE, warmup=str(args.warmup),
        candidate=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        gate='both lessons: exact free source order on every row and final per-row CE <= .01 nats/read',
        batch_end_optimizer='disabled', no_retries=True)
    save(args.out/'protocol.json', protocol)
    progress, stages = [], []
    started = time.monotonic()
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = make_model(args.out, patch)
        try:
            drift = restore_warmup(model, args.warmup, restore)
            save(args.out/'start.json', dict(loader_drift=drift))
            optimizer = model.getOptimizer(lr=RATE)
            with paired_updates_only(model, optimizer) as updates:
                for stage in [stage for stage in STAGES if stage['stage'] == 3]:
                    initial = present(model, stage)
                    rows = []
                    for batch in range(BATCHES):
                        row = present(model, stage, train=True, optimizer=optimizer)
                        rows.append(row)
                        progress.append(dict(lesson=stage['name'], batch=batch+1,
                            ce=row['lesson_cross_entropy'], updates=dict(updates),
                            elapsed_seconds=time.monotonic()-started))
                        save(args.out/'progress.json', progress)
                        print(json.dumps(progress[-1]), flush=True)
                    stages.append(dict(name=stage['name'], initial=initial, training=rows))
                    save(args.out/'stages.json', stages)
                # Both free gates are measured after both lessons, so a later
                # lesson cannot silently erase the earlier order skill.
                for stage, row in zip([s for s in STAGES if s['stage'] == 3], stages):
                    final = present(model, stage)
                    ce = row['training'][-1]['lesson_cross_entropy']
                    row.update(final=final, passed=final['read_gate_passed'] and max(ce) <= ORDER_CE_MAX)
                save(args.out/'stages.json', stages)
                assert updates['paired'] == protocol['paired_optimizer_steps']
                result = dict(protocol=protocol, stages=stages, updates=dict(updates),
                    completed=True, passed=all(row['passed'] for row in stages),
                    elapsed_seconds=time.monotonic()-started)
                model.save_weights(str(args.out/'order.ckpt'))
                save(args.out/'results.json', result)
                print(json.dumps(dict(completed=True, passed=result['passed'], updates=updates)), flush=True)
        except BaseException:
            save(args.out/'failure.json', dict(error=traceback.format_exc(),
                completed_batches=len(progress), elapsed_seconds=time.monotonic()-started))
            raise
        finally:
            model.End()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--warmup', type=Path,
        default=Path('output/item6-orders-paired-accepted-code-20261015/warmup.ckpt'))
    run(parser.parse_args())
