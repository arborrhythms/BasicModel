"""Measure paired evidence on a trained, live parallel model.

Seed 42 fixes a measurement, never a passing learning assertion. Feature
permutations preserve the trained field's marginals while destroying its
alignment with the dictionary; they are a control, not semantic labels.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('MODEL_COMPILE', 'eager')
os.environ.setdefault('BASICMODEL_DEVICE', 'cpu')
os.environ.setdefault('BASIC_AUTOLOAD', 'false')

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]
import numpy as np
import torch
import torch.nn.functional as F
import Language
from ConceptEvidence import admit, symbols, union
from Models import BaseModel
from data import TheData
from util import init_config, init_device
from bounded_tests import source_snapshot


def quantiles(x):
    levels = torch.tensor([0., .5, .9, .99, .999, 1.])
    return {str(float(q)): float(v) for q, v in zip(levels, x.flatten().float().quantile(levels))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--config', default='data/MM_sparse_concept.xml')
    parser.add_argument('--updates', default=16, type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    init_device('cpu')
    init_config(args.config, defaults_path='data/model.xml')
    cfg = BaseModel.load_config(args.config)
    TheData.load(cfg['architecture']['data']['dataset'], dat=dict(cfg['architecture']['data']))
    Language.TheGrammar._configured = False
    model, _ = BaseModel.from_config(args.config, data=TheData)
    model.set_sigma(0)
    model.checkpoint_every_batches = 0
    optimizer = model.getOptimizer(lr=float(cfg['architecture']['training']['learningRate']))
    cs = model.conceptualSpaces[0]
    snap = cs.cs_snap_order0
    samples, losses = [], []
    original_batch = model.runBatch

    def capture(event, **kwargs):
        samples.append(event.detach().cpu().clone())
        return snap(event, **kwargs)

    def batch(*a, **kw):
        result = original_batch(*a, **kw)
        losses.append({'reconstruction': float(result[0].lossIn.detach()),
                       'answer': float(result[0].lossOut.detach())})
        return result

    cs.cs_snap_order0, model.runBatch = capture, batch
    source = source_snapshot(ROOT)
    report = dict(source=source, seed=42, config=args.config,
                  config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                  device='cpu', threads=1, torch=torch.__version__, batch_size=4)
    began = time.perf_counter()
    try:
        for _ in range(args.updates):
            model.runEpoch(optimizer=optimizer, batchSize=4, split='train', max_batches=1)
        report['training'] = dict(updates=len(losses), seconds=time.perf_counter() - began, losses=list(losses))
        samples.clear()
        model.runEpoch(optimizer=None, batchSize=4, split='validation', max_batches=1)
        report['validation'] = losses[-1]
        if not samples:
            raise RuntimeError('measurement did not execute the parallel cutover')
        event = torch.cat(samples)
        atoms = F.normalize(cs.similarity_codebook.getW()[:cs._order_caps()[0]].detach().cpu(), dim=-1)
        d = min(event.shape[-1], atoms.shape[-1])
        event, atoms = event[..., :d], atoms[..., :d]
        projection = event @ atoms.T / math.sqrt(d)
        generator = torch.Generator().manual_seed(1042)
        controls = torch.stack([event[..., torch.randperm(d, generator=generator)] @ atoms.T / math.sqrt(d)
                                for _ in range(128)])
        calibration, heldout = controls[:64], controls[64:]
        # Round upward from the maximum of a declared calibration sample;
        # report the independent control sample and accumulated evidence too.
        floor = math.ceil(float(calibration.abs().max()) * 1000) / 1000
        report['projection'] = dict(shape=list(projection.shape), magnitude=quantiles(projection.abs()),
                                    winning_magnitude=quantiles(projection.abs().amax(-1)),
                                    calibration=quantiles(calibration.abs()), heldout=quantiles(heldout.abs()),
                                    proposed_floor=floor, configured_floor=cs.concept_evidence_floor)
        report['control_scopes'] = []
        floors = sorted(set((0., floor, cs.concept_evidence_floor)))
        for tau in floors:
            for slots in (8, 64, 256):
                flat = heldout.flatten(0, 2)
                n = len(flat) // slots
                scoped = flat[:n * slots].reshape(n, slots, atoms.shape[0])
                pair = union(admit(scoped, tau), dim=1)
                report['control_scopes'].append(dict(floor=tau, slots=slots, scopes=n,
                    positive_max=float(pair[..., 0].max()), negative_max=float(pair[..., 1].max()),
                    both_max=float(pair.prod(-1).max())))
        report['live'] = {}
        cs.eval()
        for enabled in (False, True):
            cs.conceptual_pi = enabled
            for tau in floors:
                cs.concept_evidence_floor = tau
                a0 = cs.cs_snap_order0(event)
                content, field = cs.cs_forward_content(a0, cs.similarity_codebook.getW())
                pair = symbols(field)
                report['live'][f'pi={enabled},floor={tau}'] = dict(
                    field_shape=list(field.shape), symbol_shape=list(content.shape),
                    positive=quantiles(pair[..., 0]), negative=quantiles(pair[..., 1]),
                    both=quantiles(pair.prod(-1)), active_symbols=int((pair > 0).sum()),
                    above_use_floor=int((pair > cs.concept_use_floor).sum()),
                    order_max=list(cs._cs_level_acts))
        report['source_unchanged'] = source_snapshot(ROOT) == source
        args.out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'source'}, indent=2))
    finally:
        cs.cs_snap_order0, model.runBatch = snap, original_batch
        model.End()


if __name__ == '__main__':
    main()
