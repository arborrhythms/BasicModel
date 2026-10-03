"""One unseeded MM_xor proof with passive input/answer geometry observations."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]
os.environ.setdefault('BASICMODEL_DEVICE', 'cpu')
os.environ.setdefault('MODEL_COMPILE', 'eager')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
from test_mm_xor import _fresh_model
from util import init_device


def tensor(value):
    return value.detach().cpu().tolist() if torch.is_tensor(value) else value


def geometry(value, targets):
    x = value.detach().double().reshape(4, -1).cpu()
    y = targets.detach().double().reshape(4, -1).cpu()
    centered = x - x.mean(0)
    design = torch.cat((x, torch.ones(4, 1, dtype=x.dtype)), 1)
    fit = torch.linalg.lstsq(design, y, driver='gelsd')
    predictions = design @ fit.solution
    return dict(shape=list(value.shape), centered_singular_values=tensor(torch.linalg.svdvals(centered)),
                affine_rank=int(fit.rank), affine_predictions=tensor(predictions),
                affine_mse=float((predictions-y).square().mean()),
                affine_weight_norm=float(fit.solution[:-1].norm()),
                values=tensor(value))


def run(output, epochs):
    init_device('cpu')
    torch.set_num_threads(1)
    start = time.monotonic()
    m, cfg, data = _fresh_model(str(ROOT/'data/MM_xor.xml'))
    optimizer = torch.optim.Adam(m.parameters(), lr=.01)
    criterion = torch.nn.MSELoss()
    report = dict(seed=None, config='data/MM_xor.xml', budget=epochs, threshold=.20,
                  serial=m.serial, useGrammar=m.useGrammar,
                  snapshots=[], losses=[], best_loss=float('inf'))
    print(json.dumps({k:v for k,v in report.items() if k not in ('snapshots','losses')}), flush=True)
    captured = {}
    original_head = m._forward_head
    def head(sub, **kwargs):
        result = original_head(sub, **kwargs)
        if captured.get('enabled'):
            captured['head_input'] = sub.materialize().detach().clone() if sub is not None else None
            captured['root'] = getattr(m, '_stm_single_S', None)
        return result
    m._forward_head = head
    loader = m.inputSpace.data.data_loader(split='train', num_streams=4)
    try:
        for epoch in range(epochs):
            texts, answers = next(iter(loader))
            inp = m.inputSpace.prepInput(texts)
            target = m.outputSpace.prepOutput(answers)
            optimizer.zero_grad()
            # Observe every state, retain only first/final plus promotion transitions.
            captured['enabled'] = True
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _, symbols, prediction, _ = m.forward(inp)
            target = target.to(prediction.device)
            while target.dim() < prediction.dim(): target = target.unsqueeze(-1)
            target = target.expand_as(prediction)
            loss = criterion(prediction, target)
            assert torch.isfinite(loss)
            value = float(loss.detach())
            report['losses'].append(value)
            report['best_loss'] = min(report['best_loss'], value)
            state = m.perceptualSpace._forward_input
            ids = tensor(state.get('indices'))
            transition = ids != report.get('last_percept_ids')
            final = epoch+1 == epochs or value < .20
            if epoch == 0 or transition or final:
                sub = m.perceptualSpace.subspace
                snap = dict(epoch=epoch, texts=texts, targets=tensor(target), predictions=tensor(prediction),
                    loss=value, percept_ids=ids,
                    word_texts=state.get('word_texts'), word_groups=tensor(state.get('word_groups')),
                    native_indices=tensor(state.get('native_indices')), native_spans=tensor(state.get('native_part_spans')),
                    percept_shape=list(m.perceptualSpace._embedded_input.shape),
                    nWhat=int(sub.nWhat), nWhere=int(sub.nWhere), nWhen=int(sub.nWhen),
                    percept_values=tensor(m.perceptualSpace._embedded_input),
                    answer_input=geometry(captured['head_input'], target) if captured['head_input'] is not None else None,
                    root=geometry(captured['root'], target) if torch.is_tensor(captured.get('root')) else None,
                    symbols=geometry(symbols,target),
                    derivations=('parallel subsymbolic path; no sentence grammar choices' if not m.serial else
                                 repr(getattr(m,'_last_sentence_compose_trace',None))))
                report['snapshots'].append(snap)
            report['last_percept_ids'] = ids
            report['epochs_completed'] = epoch+1
            report['elapsed_seconds'] = time.monotonic()-start
            output.write_text(json.dumps(report,indent=2)+'\n')
            loss.backward()
            optimizer.step()
            if final: break
        report['passed'] = report['best_loss'] < .20
        print(json.dumps({k:v for k,v in report.items() if k not in ('snapshots','losses','last_percept_ids')}),flush=True)
    finally:
        report['elapsed_seconds'] = time.monotonic()-start
        output.write_text(json.dumps(report,indent=2)+'\n')
        m.End()
        m.symbolSpace.soft_reset()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--epochs',type=int,default=200)
    args=parser.parse_args()
    run(args.output,args.epochs)
