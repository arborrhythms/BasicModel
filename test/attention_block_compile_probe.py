"""Probe the native sentence body's fullgraph boundary without changing it.

The eager compiler backend tests Dynamo capture only. It is neither an
accelerator benchmark nor a claim of accelerator code generation.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from attention_thinking_certificate import lesson, restore


class ProbeComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    from pytest import MonkeyPatch
    from reading_fixtures import use_eager_reading
    from Models import BasicModel
    with MonkeyPatch.context() as patch:
        use_eager_reading(patch)
        compiled = torch.compile(BasicModel._run_sentence_word_bricks,
                                 backend='eager', fullgraph=True, dynamic=True)
        report = dict(backend='eager', fullgraph=True, dynamic=True,
                      fixture='native first lesson sentence', seed_selected=False)
        def attempt(*values, **kwargs):
            try:
                compiled(*values, **kwargs)
            except (torch._dynamo.exc.Unsupported, torch._dynamo.exc.UserError) as error:
                report.update(captured=False, error_type=type(error).__name__, error=str(error))
            else:
                report.update(captured=True)
            raise ProbeComplete
        patch.setattr(BasicModel, '_run_sentence_word_bricks', attempt)
        metadata = json.loads((args.replay/'initial.json').read_text())
        assert os.environ.get('PYTHONHASHSEED') == metadata['pythonhashseed']
        restore(torch.load(args.replay/'initial-rng.pt', weights_only=False))
        observer = SimpleNamespace(reference_seconds=0., records=[], calls=[], actions=[], costs=[])
        try:
            lesson(observer)
        except ProbeComplete:
            pass
        assert 'captured' in report, 'the native sentence body was not reached'
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
