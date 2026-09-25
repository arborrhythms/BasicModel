"""Item 10 measurements in real layers; no run or random seed is selected."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]
import torch
import torch.nn.functional as F
from Layers import SigmaLayer, PiLayer
from bounded_tests import source_snapshot


from fold_evaluation import xor_trial


def weight_measurements():
    rows = []
    for width in (8, 64, 264, 1032):
        for scaled in (False, True):
            layer = SigmaLayer(width, width, monotonic=True, invertible=True, normalize=True).double()
            if not scaled:
                with torch.no_grad():
                    layer.layer.raw_L.fill_(-5.)
                    layer.layer.raw_U.fill_(-5.)
            w, _ = layer.normalized_weights()
            x = torch.rand(32, width, dtype=torch.float64)
            y = layer(x)
            rows.append(dict(width=width, scaled_init=scaled, own_min=float(w.diag().min()),
                own_mean=float(w.diag().mean()), spread_ratio=float(y.std(-1).mean()/x.std(-1).mean()),
                condition=float(torch.linalg.cond(w)),
                reverse_error=float((layer.reverse(y)-x).abs().max())))
    return rows


def floor_measurements():
    pi = PiLayer(2, 1, monotonic=True, normalize=True).double()
    with torch.no_grad():
        pi.layer.W.fill_(0.)  # equal nonnegative weights
        pi.raw_beta.fill_(-30.)
    rows = []
    for u in (0., 1e-7, 1e-6, 2e-6, 1e-4, .5):
        x = torch.tensor([[u, 1.]], dtype=torch.float64, requires_grad=True)
        y = pi(x)
        grad, = torch.autograd.grad(y.sum(), x)
        rows.append(dict(input=u, output=float(y), slope=float(grad[0, 0])))
    return rows


def depth_measurements():
    rows = []
    for width in (8, 64, 264):
        x = (.05+.9*torch.rand(4, width, dtype=torch.float64)).requires_grad_()
        y = x
        for depth in range(1, 31):
            kind = SigmaLayer if depth % 2 else PiLayer
            layer = kind(width, width, monotonic=True, invertible=True, normalize=True).double()
            y = layer(y)
            if depth in (1, 10, 20, 30):
                grad, = torch.autograd.grad(y.sum(), x, retain_graph=True)
                rows.append(dict(width=width, depth=depth, min=float(y.min()), max=float(y.max()),
                    spread_ratio=float(y.std(-1).mean()/x.std(-1).mean()),
                    gradient_norm_ratio=float(grad.norm()/torch.ones_like(grad).norm())))
    return rows


def signed_depth_measurements():
    rows=[]
    for width in (8,64,264):
        products={name:torch.eye(width,dtype=torch.float64) for name in ('l1','l2')}
        x=F.normalize(torch.randn(32,width,dtype=torch.float64),dim=-1)*.9
        for depth in range(1,31):
            layer=SigmaLayer(width,width,invertible=True,normalize=True).double()
            with torch.no_grad():
                layer.layer.raw_L.uniform_(-.0067,.0067)
                layer.layer.raw_U.uniform_(-.0067,.0067)
            w,scale=layer.normalized_weights()
            raw=w*scale
            alpha,_=layer._mean_bias()
            for name in products:
                weights=w if name=='l2' else raw/raw.abs().sum(0)
                products[name]=products[name]@(alpha*weights).detach()
                if depth in (1,10,20,30):
                    y=x@products[name]
                    rows.append(dict(width=width,depth=depth,norm=name,
                        energy_ratio=float(y.norm(dim=-1).mean()/x.norm(dim=-1).mean()),
                        reverse_gain=float(torch.linalg.svdvals(products[name]).min().reciprocal())))
    return rows


def kernels_and_binary():
    output = []
    for kind in (SigmaLayer, PiLayer):
        for normalize in (False, True):
            layer = kind(64, 64, invertible=True, normalize=normalize)
            x = F.normalize(torch.rand(16, 64)*2-1, dim=-1)*.9
            for _ in range(5):
                layer(x)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                layer(x)
            start = time.perf_counter()
            for _ in range(100):
                layer(x)
            micros = (time.perf_counter()-start)*1e4
            parent = layer.compose(x, x.flip(0))
            left, right = layer.generate(parent)
            output.append(dict(kind=kind.__name__, normalize=normalize,
                microseconds=micros, aten_calls=sum(e.count for e in prof.key_averages() if e.key.startswith('aten::')),
                operators={e.key:e.count for e in prof.key_averages() if e.key.startswith('aten::')},
                parent_roundtrip=float((layer.compose(left,right)-parent).abs().max()),
                distinct_children_reconstruction=float((left-x).square().mean())))
    return output


def union_comparison():
    histories = {}
    for mode in ('arithmetic', 'max', 'probabilistic', 'tanh'):
        x = torch.tensor(1., dtype=torch.float64)
        history = []
        for _ in range(5):
            members = torch.cat((x[None], torch.zeros(7, dtype=x.dtype)))
            x = {'arithmetic': lambda: members.mean(), 'max': lambda: members.max(),
                 'probabilistic': lambda: -torch.expm1(torch.log1p(-members).sum()),
                 'tanh':lambda: members.sum().tanh()}[mode]()
            history.append(float(x))
        histories[mode] = history
    histories['unrelated_zero'] = {str(n): float(torch.zeros(n).amax()) for n in (1,8,256,4096)}
    histories['normalized_membership_zero_input']={}
    for kind in (SigmaLayer,PiLayer):
        layer=kind(8,8,invertible=True,monotonic=True,normalize=True)
        histories['normalized_membership_zero_input'][kind.__name__]=float(layer(torch.zeros(1,8)).max())
    histories['historical_11a_projection_limit_positions'] = 256
    return histories


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=source_snapshot(ROOT)
    report=dict(source=source, probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                torch=torch.__version__, device='cpu', threads=1, xor_runs=[])
    def save():
        report['source_unchanged']=source_snapshot(ROOT)==source
        args.out.write_text(json.dumps(report, indent=2)+'\n')
    for mode, unit in (('current',True),('selective',True),('selective',False),('both',True)):
        for width in (4,8):
            for run in range(8):
                result=xor_trial(width,mode,unit)
                result['run']=run
                report['xor_runs'].append(result)
                save()
            print(mode, unit, width, [round(r['mse'],6) for r in report['xor_runs'][-8:]], flush=True)
    for name, fn in (('width',weight_measurements),('floor',floor_measurements),
                     ('depth',depth_measurements),('signed_depth',signed_depth_measurements),('kernels',kernels_and_binary),('unions',union_comparison)):
        report[name]=fn()
        save()
        print(name, 'complete', flush=True)


if __name__=='__main__':
    main()
