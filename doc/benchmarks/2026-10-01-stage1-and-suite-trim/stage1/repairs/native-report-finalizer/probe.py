"""Fail before the observer skips the unused console-only decoding scan."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch


def check(path):
    messages, calls = [], []
    namespace = dict(torch=torch, os=os, TheMessage=messages.append)
    exec(compile(Path(path).read_text(), str(path), 'exec'), namespace)
    predictions = torch.arange(16.)[:, None]
    data = SimpleNamespace(test_input=[b'one'] * 16, test_output=[0.] * 16)

    class Model:
        inputSpace = SimpleNamespace(data=data)
        _data_len = staticmethod(len)
        _slice_data = staticmethod(lambda value, n: value[:n])

        def set_sigma(self, value):
            calls.append(('sigma', value))

        def runEpoch(self, **kwargs):
            calls.append(('epoch', kwargs))
            return 0., 0., predictions, torch.ones(16, 256, 4)

        def _decode_reconstructed_inputs(self, *args):
            raise AssertionError('native objective receipt entered console-only concept-bank scan')

    logs = []
    namespace['_objective_probe'] = SimpleNamespace(log=lambda *a, **k: logs.append((a, k)))
    namespace['_reconstructionReport'](Model())
    assert calls == [('sigma', 0), ('epoch', {'batchSize': 16, 'split': 'test'}), ('sigma', .5)]
    assert len(data.reconstructed_output) == 16
    assert torch.equal(torch.stack(data.reconstructed_output), predictions)
    return dict(pass_=True, endpoint_calls=1, endpoint_batch=16,
                endpoint_predictions_unchanged=True, console_decoding_calls=0, logs=logs)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('body')
    p.add_argument('output')
    a = p.parse_args()
    try:
        result = check(a.body)
    except BaseException as e:
        result = dict(pass_=False, error=type(e).__name__, message=str(e))
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    raise SystemExit(0 if result['pass_'] else 1)
