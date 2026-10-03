"""Exercise the real two gate bodies with a counted, deterministic stand-in.

No model is trained, no RNG is called, and no gate assertion is replaced.
The deliberately failing before run demonstrates the duplicated training.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODELS = []


def pytest_collection_modifyitems(items):
    module = items[0].module
    mode = os.environ.get("SHARING_PROBE_MODE", "pass")

    def counted_training():
        texts = ["hello world", "hello there", "loving world", "loving there"]
        targets = [torch.tensor(x) for x in (0., 1., 1., 0.)]
        answers = [torch.tensor(.5)] * 4 if mode in ("class", "both") else targets
        read_backs = ["hello"] * 4 if mode in ("reconstruction", "both") else texts
        model = SimpleNamespace(
            rCorrect=[1., 1.],
            inputSpace=SimpleNamespace(
                data=SimpleNamespace(reconstructed_output=answers, test_output=targets,
                                     test_input=texts, reconstructed_input=read_backs),
                getTestData=lambda: (texts, targets)),
            _grammar_gate_reconstructions=read_backs,
            _grammar_gate_unavailable=[False] * 4,
            _bytes_to_text=lambda text: text)
        MODELS.append(model)
        return model

    module._run_xor_grammar_in_process = counted_training


def pytest_sessionfinish(session, exitstatus):
    expected = {"pass": 0, "class": 1, "reconstruction": 1, "both": 2}[
        os.environ.get("SHARING_PROBE_MODE", "pass")]
    result = dict(training_calls=len(MODELS), expected_training_calls=1,
                  failures=session.testsfailed, expected_failures=expected,
                  selected=session.testscollected, original_exitstatus=int(exitstatus))
    Path(os.environ["SHARING_PROBE_OUTPUT"]).write_text(json.dumps(result, indent=2))
    if len(MODELS) != 1 or session.testsfailed != expected or session.testscollected != 2:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
