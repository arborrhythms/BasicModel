"""Observe byte scoring with the same recurrence bodies executed in Python.

The primary receipt uses native torch.while_loop. This diagnostic removes
only its tracing overhead so every scorer input can be saved and replayed.
Its values must match that receipt before the diagnostic is interpreted.
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test'), str(HERE)]


def main(mode, output):
    import torch
    import Models
    import probe
    def eager_loop(cond, body, carried_inputs):
        while bool(cond(*carried_inputs)):
            carried_inputs = body(*carried_inputs)
        return carried_inputs
    torch.while_loop = eager_loop
    scorer = Models.BasicModel._byte_word_cost
    captured = []
    def score(self, idea, word, bank_n, bank_bytes, bank_valid,
              target_bytes, target_valid, ready):
        result = scorer(self, idea, word, bank_n, bank_bytes, bank_valid,
                        target_bytes, target_valid, ready)
        if idea.shape[0] == 2:
            captured.append({key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for key, value in dict(idea=idea, word=word, bank_n=bank_n,
                    bank_bytes=bank_bytes, bank_valid=bank_valid,
                    target_bytes=target_bytes, target_valid=target_valid,
                    ready=ready, cost=result).items()})
        return result
    Models.BasicModel._byte_word_cost = score
    sys.argv = [str(HERE / 'probe.py'), '--config', str(HERE / 'parity.xml'),
                '--parity', mode, '--out', str(output)]
    probe.main()
    torch.save(captured, Path(output).with_suffix('.scorer.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('packed', 'single'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.mode, args.out)
