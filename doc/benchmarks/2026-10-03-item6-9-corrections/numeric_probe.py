"""Isolate eager/Inductor rounding on the saved deterministic byte input."""
import json
from pathlib import Path
import sys
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'bin'))
from SentenceUnderstanding import readback_scores
from torch._inductor.utils import run_and_get_code

x = torch.arange(1., 129.).reshape(2, 64)
w = torch.nn.functional.normalize(torch.arange(1., 2049.).reshape(2, 16, 64), dim=-1)
def components(x, w):
    square = w.square().sum(-1)
    dot = (x[:, None] * w).sum(-1)
    cosine = torch.nn.functional.cosine_similarity(x[:, None], w, dim=-1)
    logits = readback_scores(x, w, torch.ones_like(square)) / .1
    probability = torch.softmax(torch.cat((logits, logits.new_zeros(2, 1)), -1), -1)
    return square, dot, cosine, logits, probability

eager = components(x, w)
compiled, code = run_and_get_code(torch.compile(components, fullgraph=True), x, w)
double = components(x.double(), w.double())
result = {}
for name, e, c, d in zip(('square','dot','cosine','logits','probability'), eager, compiled, double):
    result[name] = dict(eager=e.tolist(), compiled=c.tolist(), double=d.tolist(),
                       difference=float((e-c).abs().max()),
                       eager_error=float((e-d).abs().max()), compiled_error=float((c-d).abs().max()))
(HERE/'numeric-before.json').write_text(json.dumps(result, indent=2)+'\n')
for i, source in enumerate(code):
    (HERE/f'numeric-before-kernel-{i}.py').write_text(source)
print(json.dumps({k:{n:v[n] for n in ('difference','eager_error','compiled_error')} for k,v in result.items()}, indent=2))
