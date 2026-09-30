"""Observation-only pytest plugin; no source patch, RNG calls or gate changes."""
import json
import os
from pathlib import Path
import pytest


def record(value):
    with Path(os.environ['ITEM7_XOR_MEASUREMENTS']).open('a') as handle:
        handle.write(json.dumps(value) + '\n')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    module = item.module
    cli = getattr(module, '_run_cli', None)
    if cli is not None:
        def observed_cli(*args, **kwargs):
            result = cli(*args, **kwargs)
            rc, stdout, stderr = result
            record(dict(nodeid=item.nodeid, kind='cli', config=args[0],
                        returncode=rc, stdout=stdout, stderr=stderr,
                        mse=module._parse_output_mse(stdout),
                        reconstructed=module._parse_input_match_counts(stdout)))
            return result
        module._run_cli = observed_cli
    grammar = getattr(module, '_run_xor_grammar_in_process', None)
    if grammar is not None:
        def observed_grammar(*args, **kwargs):
            model = grammar(*args, **kwargs)
            record(dict(nodeid=item.nodeid, kind='grammar',
                        accuracy=[float(x) for x in model.rCorrect]))
            return model
        module._run_xor_grammar_in_process = observed_grammar
    mse_forward = None
    measurement = dict(nodeid=item.nodeid, kind='mm', calls=0, best=float('inf'))
    if item.name in ('test_convergence', 'test_learns_xor_signal',
                      'test_mm_grammar_learns_xor_signal'):
        import torch
        mse_forward = torch.nn.MSELoss.forward
        def observed_mse(self, output, target):
            result = mse_forward(self, output, target)
            if output.numel() == 4:
                value = float(result.detach())
                measurement.update(calls=measurement['calls'] + 1,
                                   best=min(measurement['best'], value), last=value,
                                   predictions=output.detach().cpu().flatten().tolist(),
                                   targets=target.detach().cpu().flatten().tolist())
            return result
        torch.nn.MSELoss.forward = observed_mse
    try:
        yield
    finally:
        if cli is not None:
            module._run_cli = cli
        if grammar is not None:
            module._run_xor_grammar_in_process = grammar
        if mse_forward is not None:
            torch.nn.MSELoss.forward = mse_forward
            record(measurement)
