"""Review regressions: explicit dictionary ownership and non-invasive diagnostics."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch

from GradientDiagnostics import objective_agreement, record_opposition

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIGS = ("BasicModel", "BasicModel_answers_tied_benchmark",
                    "MM_grammar_wording", "MM_ladder", "MM_ladder_idiom")


@pytest.mark.parametrize("name", CANONICAL_CONFIGS)
def test_canonical_dictionary_has_one_distributional_owner(name):
    root = ET.parse(ROOT / "data" / (name + ".xml")).getroot()
    assert float(root.findtext("./architecture/training/conceptualContextLearningRate")) == 0
    assert float(root.findtext("./architecture/training/conceptualSimilarityScale", "0")) == 0


def test_unused_operator_sample_preserves_opposition():
    p = torch.nn.Parameter(torch.ones(1))
    history = {}
    for output in (-p.sum(), None, -p.sum(), 0 * p.sum(), -p.sum()):
        report = objective_agreement({"reconstruction": p.sum(), "output": output}, {"op": [p]})
        record_opposition(report, history)
    assert report["op"]["output_negative_streak"] == 3
    assert report["op"]["persistent_opposition"] == ["output"]
    report = objective_agreement({"reconstruction": p.sum(), "output": p.sum()}, {"op": [p]})
    record_opposition(report, history)
    assert history[("op", "output")] == 0




def _xor_model(monkeypatch):
    import Models
    from data import TheData
    from util import init_config, init_device
    monkeypatch.setenv("MODEL_COMPILE", "none")
    init_device("cpu")
    config = str(ROOT / "data/MM_xor.xml")
    init_config(path=config, defaults_path=str(ROOT / "data/model.xml"))
    TheData.load("xor")
    return Models.BaseModel.from_config(config, data=TheData)[0]




def test_diagnostic_failure_does_not_abort_real_training(monkeypatch):
    model = _xor_model(monkeypatch)
    model.branch_diagnostics_every = 1
    model.loss.reconstruction_scale = .25
    optimizer = model.getOptimizer(lr=1e-4)
    calls = []
    original = optimizer.step
    def step(*a, **kw):
        calls.append(getattr(model, '_sentence_trial', None))
        return original(*a, **kw)
    optimizer.step = step
    def unavailable(*a, **kw):
        raise FloatingPointError("diagnostic probe failure")
    monkeypatch.setattr(model, "ownership_gradient_diagnostics", unavailable)
    inputs, outputs = next(iter(model.inputSpace.data.data_loader(split="train", num_streams=2)))
    batch = (model.inputSpace.prepInput(inputs), model.outputSpace.prepOutput(outputs))
    with pytest.warns(RuntimeWarning, match="diagnostic probe failure"):
        result, _ = model.runBatch(train=True, batchSize=2, split="train", optimizer=optimizer,
                                   batch_override=batch)
    assert result is not None and calls == [None]  # This model has no serial compose.
    model.End()


def test_query_signatures_use_only_checked_subsystems():
    from AccessibleMind import Subsystem, check_access
    from Queries import BUILTIN_QUERIES
    for signature in BUILTIN_QUERIES.values():
        assert all(isinstance(scope, Subsystem) for scope in (*signature.read_scope, *signature.write_scope))
        check_access("thought", signature.read_scope, signature.write_scope)
        with pytest.raises(ValueError, match="Subsystem"):
            replace(signature, read_scope=("ltm.facts",))
