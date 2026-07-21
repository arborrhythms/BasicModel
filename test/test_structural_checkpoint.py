"""Focused persistence tests for host-side learned structure.

Concept ids, relation records and word-keyed mereology registries are plain
Python state by design, so ``state_dict`` cannot cover them.  These tests pin
the integrated checkpoint sidecar that keeps a resume from re-minting them.
"""

import os
import sys
import types

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_ROOT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

import Models
from Layers import ConceptAllocator


def _allocator():
    return ConceptAllocator(
        layer_sizer=lambda _order: (
            9, 8, torch.device("cpu"), None),
        order_cap=lambda: 2,
    )


def _model_with(cs, ws):
    model = Models.BaseModel()
    model.name = "StructuralCheckpointTest"
    model.spaces = []
    model.conceptualSpaces = [cs]
    model.wholeSpaces = [ws]
    return model


def test_integrated_checkpoint_restores_allocator_and_identity_caches(tmp_path):
    alloc = _allocator()
    word, obj, meta = (alloc.new_concept() for _ in range(3))
    alloc.add(word, "part", 7)
    alloc.add(word, "whole", 41)
    alloc.add(obj, "part", -1)
    alloc.add(obj, "whole", 0)
    alloc.layer().embed_pair(
        meta, whole_ref=("sym", word), part_ref=("sym", obj))
    alloc.settle(meta)
    alloc.word_obj_meta["cat"] = (word, obj, meta)
    alloc.joint[("cat", "sat")] = meta
    alloc.relate_idx[(7, 41)] = word
    alloc.chain_idx[("chain", (word, obj))] = meta
    alloc.raised.add(meta)
    alloc.singletons.add(obj)
    alloc.identity[word] = (7, 41)

    layer = alloc.layer()
    layer.assign_row(("snap", word), capacity=4, base=0)
    layer.assign_row(("pool", meta), capacity=4, base=4)
    layer.add_edge(5, 2, weight=0.4)
    with torch.no_grad():
        layer.values[0] = 0.625

    cs = types.SimpleNamespace(
        _concept_allocator=alloc,
        _autobound_percept_ids={7},
        _recognized_words={"cat": 7},
        _words_concept_id=word,
        _percept_word_concept={7: (word, obj)},
        _object_word_concept={obj: word},
        _priming_bridge={word: (frozenset({7}), 41)},
        _frozen_concepts={word},
        _frozen_named={"words": word},
        _concept_admission_drops={"word/object/META": 2},
    )
    ws = types.SimpleNamespace(
        _word_whole_ss={"cat": 41},
        _mereology_raised={41},
        _property_class_whole={(1, 3): 44},
        _anchored_pids={7: "operator"},
        _pending_words_summary=[(41, 1)],
        _standalone_run_bytes={ord("c")},
        _lbg_disp_sum={41: torch.tensor([1.0, 2.0])},
        _lbg_disp_sum_sq={41: torch.tensor([1.0, 4.0])},
        _lbg_count={41: 3},
    )
    source = _model_with(cs, ws)
    path = tmp_path / "structure.ckpt"
    source.save_weights(path)

    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["structural_extras"]["version"] == 1

    restored_alloc = _allocator()
    restored_cs = types.SimpleNamespace(_concept_allocator=restored_alloc)
    restored_ws = types.SimpleNamespace()
    target = _model_with(restored_cs, restored_ws)
    assert target.load_weights(path)

    assert restored_alloc.word_obj_meta["cat"] == (word, obj, meta)
    assert restored_alloc.joint[("cat", "sat")] == meta
    assert restored_alloc.records(meta) == [
        ("whole", ("sym", word)), ("part", ("sym", obj))]
    assert restored_alloc.layer()._tensor_rows == layer._tensor_rows
    assert restored_alloc.layer()._row_next == layer._row_next
    assert restored_alloc.layer()._rows == [5]
    assert restored_alloc.layer()._cols == [2]
    torch.testing.assert_close(
        restored_alloc.layer().values.detach(), torch.tensor([0.625]))
    assert restored_cs._word_obj_meta is restored_alloc.word_obj_meta
    assert restored_cs._joint_concepts is restored_alloc.joint
    assert restored_cs._percept_word_concept[7] == (word, obj)
    assert restored_cs._recognized_words == {"cat": 7}
    assert restored_cs._concept_admission_drops == {"word/object/META": 2}

    assert restored_ws._word_whole_ss == {"cat": 41}
    assert restored_ws._mereology_raised == {41}
    assert restored_ws._property_class_whole == {(1, 3): 44}
    assert restored_ws._anchored_pids == {7: "operator"}
    torch.testing.assert_close(
        restored_ws._lbg_disp_sum[41], torch.tensor([1.0, 2.0]))

    # The monotonic allocator resumes above every restored id. A repeat lookup
    # therefore reuses the checkpoint's identity instead of colliding/reminting.
    assert restored_alloc.word_obj_meta["cat"] == (word, obj, meta)
    assert restored_alloc.new_concept() > meta


def test_weights_path_environment_override_is_project_relative(monkeypatch):
    model = Models.BaseModel()
    monkeypatch.setenv("BASIC_WEIGHTS_PATH", "output/isolated-15m.ckpt")
    expected = os.path.join(
        Models.ProjectPaths.PROJECT_DIR, "output", "isolated-15m.ckpt")
    assert model._checkpoint_path() == os.path.abspath(expected)
    assert model._checkpoint_path(suffix="emergency").endswith(
        "isolated-15m.emergency.ckpt")


def test_wall_clock_cap_returns_through_normal_trial_tail(monkeypatch):
    """A partial epoch still returns the fixed trial shape used by run()."""
    model = Models.BasicModel()
    model.name = "DeadlineTest"
    model.symbolSpace = None
    model.reversible = False

    model.getOptimizer = types.MethodType(
        lambda self, lr=0.01: object(), model)
    model.set_sigma = types.MethodType(lambda self, _sigma: None, model)

    def _one_partial_epoch(self, **_kwargs):
        assert self._training_deadline_monotonic == 115.0
        self._training_deadline_reached = True
        return torch.tensor(1.0), torch.tensor(2.0), [], []

    model.runEpoch = types.MethodType(_one_partial_epoch, model)
    monkeypatch.setattr(Models.time, "monotonic", lambda: 100.0)
    monkeypatch.setenv("BASIC_MAX_SECONDS", "15")
    monkeypatch.delenv("BASIC_RUN_TEST", raising=False)
    monkeypatch.setattr(Models.TheReport, "plotLoss", lambda *_a, **_k: None)
    monkeypatch.setattr(Models.TheReport, "mnistReport", lambda *_a, **_k: 0)

    accuracy = model.runTrial(numEpochs=3, batchSize=1, lr=0.01)
    assert accuracy == [0.0, 0.0, 0.0]
    assert model._training_deadline_reached is True
