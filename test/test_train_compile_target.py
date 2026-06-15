"""Focused tests for train.py compile-target selection."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

_BIN = Path(__file__).resolve().parent.parent / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import train


def _args(target="gpu", mode=None):
    return SimpleNamespace(compile_target=target, compile_mode=mode)


def test_parse_compile_target_mlx():
    args = train.parse_args([
        "--model", "data/MM_20M.xml",
        "--compile-target", "mlx",
        "--mlx-output", "output/mlx/test.pte",
    ])
    assert args.model == "data/MM_20M.xml"
    assert args.compile_target == "mlx"
    assert args.mlx_output == "output/mlx/test.pte"


def test_gpu_compile_target_sets_defaults_without_clobbering():
    env = {}
    train.apply_compile_target_env(_args("gpu"), env)
    assert env["BASICMODEL_DEVICE"] == "gpu"
    assert env["MODEL_COMPILE"] == "auto"

    env = {"BASICMODEL_DEVICE": "cuda:1", "MODEL_COMPILE": "eager"}
    train.apply_compile_target_env(_args("gpu"), env)
    assert env["BASICMODEL_DEVICE"] == "cuda:1"
    assert env["MODEL_COMPILE"] == "eager"


def test_compile_mode_is_forwarded_for_gpu_target():
    env = {}
    train.apply_compile_target_env(_args("gpu", "reduce-overhead"), env)
    assert env["MODEL_COMPILE_MODE"] == "reduce-overhead"


def test_mlx_compile_target_does_not_set_training_compile_env():
    env = {}
    train.apply_compile_target_env(_args("mlx"), env)
    assert "BASICMODEL_DEVICE" not in env
    assert "MODEL_COMPILE" not in env


def test_default_mlx_output_path_uses_model_stem(tmp_path):
    xml_path = tmp_path / "data" / "MM_20M.xml"
    out = train.default_mlx_output_path(str(tmp_path), str(xml_path))
    assert out == str(tmp_path / "output" / "mlx" / "MM_20M.pte")


def test_venv_python_honors_basicmodel_python(monkeypatch):
    monkeypatch.setenv("BASICMODEL_PYTHON", "~/custom-python")
    assert train.venv_python("/tmp/project") == os.path.expanduser("~/custom-python")


def test_mm20m_inherits_raw_lexer_for_embedding_skip():
    # MM_20M's lexer is now EXPLICIT (Phase 4b home: WholeSpace;
    # the config had carried no live lexer at all after the migration,
    # staging every sentence as the same slab). ``byte`` keeps the
    # purpose this test pins: train.py's Phase-1 word-embedding skip
    # applies to raw AND byte lexers (train.py ``lexer in ("byte",
    # "bytes", "raw")``).
    cfg = train.read_xml_config(str(Path(__file__).resolve().parent.parent / "data" / "MM_20M.xml"))
    assert cfg["lexer"] in ("byte", "raw")
    assert cfg["lexer"] == "byte"
