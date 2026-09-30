"""Item-8 development measurements. The protocol fixes every seed and budget."""
import argparse
import json
from pathlib import Path
import random
import sys
import warnings

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test'),
                str(Path(__file__).resolve().parent)]


def xor(path, seed):
    import numpy as np
    import torch
    from test_mm_xor import _fresh_model
    from util import init_device
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    init_device('cpu')
    model, _, data = _fresh_model(str(ROOT / 'data/MM_grammar.xml'))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    losses, readings = [], []
    try:
        loader = data.data_loader(split='train', num_streams=4)
        for epoch in range(900):
            texts, outputs = next(iter(loader))
            inputs = model.inputSpace.prepInput(texts)
            target = model.outputSpace.prepOutput(outputs)
            optimizer.zero_grad()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _, _, actual, _ = model.forward(inputs)
            target = target.to(actual)
            while target.ndim < actual.ndim:
                target = target.unsqueeze(-1)
            loss = (actual - target.expand_as(actual)).square().mean()
            assert torch.isfinite(loss), 'nonfinite XOR loss'
            loss.backward()
            losses.append(float(loss.detach()))
            if epoch in (0, 99, 299, 599, 899):
                readings.append(dict(update=epoch + 1, mse=losses[-1],
                    output=actual.detach().reshape(-1).tolist(),
                    gradients={name: float(p.grad.norm()) for name, p in model.named_parameters()
                               if p.grad is not None and ('chooser' in name or 'output' in name.lower())}))
            optimizer.step()
        report = dict(seed=seed, updates=900, initial=losses[0], final=losses[-1],
            minimum=min(losses), inherited_bar=.20, below_bar=min(losses) < .20,
            losses=losses, readings=readings, learned_utility='unproven')
        Path(path).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ('losses', 'readings')}))
        # A failing diagnostic remains nonzero, while the supervisor retains
        # it and continues the other predeclared seeds without selection.
        return 0 if report['below_bar'] else 1
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def baseline(path):
    """Unchanged reviewed driver plus observational, post-batch routing reads."""
    import probe
    from GrammarEvidence import RoutingCoverage, digest
    saved = probe.BaseModel.from_config
    reports = {}
    def build(*args, **kwargs):
        model, cfg = saved(*args, **kwargs)
        run_batch, run_epoch = model.runBatch, model.runEpoch
        phase, counter = None, 0
        collectors = {}
        def epoch(*a, **kw):
            nonlocal phase
            phase = ('training' if kw.get('optimizer') is not None else
                     'after_training' if 'training' in collectors else 'before_training')
            collectors[phase] = RoutingCoverage(model.languageSpace,
                corpus=digest(model.inputSpace.data.source_manifest), stage=phase)
            result = run_epoch(*a, **kw)
            reports[phase] = collectors[phase].report()
            return result
        def batch(*a, **kw):
            nonlocal counter
            from reading_fixtures import capture_readings
            with capture_readings(model) as programs:
                result = run_batch(*a, **kw)
            for slot, rows in programs.items():
                for row, program in enumerate(rows):
                    if program is not None:
                        collectors[phase].record(program, sentence_id=f'{counter}:{row}:{slot}')
            counter += 1
            return result
        model.runEpoch, model.runBatch = epoch, batch
        return model, cfg
    probe.BaseModel.from_config = staticmethod(build)
    sys.argv = [str(Path(__file__).with_name('probe.py')), '--out', str(path)]
    try:
        probe.main()
        Path(path).with_name('routing.json').write_text(json.dumps(reports, indent=2) + '\n')
    finally:
        probe.BaseModel.from_config = staticmethod(saved)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('baseline', 'xor'))
    parser.add_argument('--seed', type=int, choices=(0, 1, 2))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    raise SystemExit(xor(args.out, args.seed) if args.mode == 'xor' else baseline(args.out))
