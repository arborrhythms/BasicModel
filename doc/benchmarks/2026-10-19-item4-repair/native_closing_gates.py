"""§9.3/4: registered identities through the native close, answer and store.

Controlled perfect predictor inputs certify the carrier mechanism, not learned
surface recognition. Row priors use SymbolField's pre-arrival production read.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
import torch
from test_item4_native import case


def run():
    torch.set_num_threads(1)
    return [case(r, d) for r, d in ((0., 0.), (.5, 0.), (1., 0.), (0., .5), (0., 1.), (.5, .5))] + [case(.5, .5, omitted=True)]


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
