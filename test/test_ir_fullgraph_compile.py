"""Integration gate for the fullgraph (zero-graph-break) refactor.

The compiled per-batch forward is built with ``fullgraph=True`` (the strict
no-graph-break gate in ``BaseModel.enable_compiled_step``). IR / autobind
configs such as ``data/idempotent.xml`` historically broke that gate with
host-side symbol creation (``_maybe_autobind_meta`` -> ``insert_meta`` /
``record_lbg_pull``), data-dependent rule-prob normalization, and fail-loud
``isfinite`` checks. This test pins the end state: the IR forward must trace
with zero graph breaks.

``MODEL_COMPILE=eager`` keeps the run on the Dynamo TRACE path (where graph
breaks live) without invoking the Inductor C++ toolchain (which is unrelated
and additionally broken on repo checkouts whose path contains a space). With
the default ``BASIC_FULLGRAPH=1`` strict gate, any remaining graph break
raises and fails the run.
"""
import os
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_MODELS_PY = os.path.join(_PROJECT, "bin", "Models.py")
_VENV_PYTHON = os.path.join(_PROJECT, ".venv", "bin", "python")


def _run_cli(config_relpath, env_extra=None, timeout=240):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["BASICMODEL_DEVICE"] = "cpu"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONPATH"] = os.path.join(_PROJECT, "bin")
    env["BASIC_FULLGRAPH"] = "1"  # strict no-graph-break gate
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [_VENV_PYTHON, _MODELS_PY, config_relpath],
        cwd=_PROJECT, env=env, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def test_idempotent_forward_compiles_fullgraph_eager():
    rc, out, err = _run_cli("data/idempotent.xml",
                            env_extra={"MODEL_COMPILE": "eager"})
    combined = out + err
    tail = combined[-3000:]
    assert "Unsupported" not in combined, tail
    assert "torch._dynamo.exc" not in combined, tail
    assert "Graph break" not in combined, tail
    assert rc == 0, f"rc={rc}\n{tail}"


@pytest.mark.xfail(strict=False, reason=(
    "Serial per-word fullgraph compile is PARTIALLY landed. The STM "
    "data-dependent guards are resolved: ``_stm_shift_and_push`` "
    "(``if d >= cap`` -> ``u0 >= 8``) and ``_stm_reduce_to_single_S`` "
    "(``if bool(rel.any())`` -> ``Eq(u0, 1)``) are now tensorized. The "
    "REMAINING break is host-side learning called from inside the captured "
    "forward: ``ConceptualSpace.learn_relations_from_stm`` (Spaces.py:11433) "
    "does ``mask.tolist()`` + per-row taxonomy/codebook mutation, which is "
    "not tensorizable -- it must be HOISTED out of the captured ``forward`` "
    "into host-side post-step orchestration (the discourse ``observe_stm_"
    "end_state`` block at Models.py:6291 is the same pattern, gated off for "
    "MM_grammar). XPASS here once that hoist lands."))
def test_serial_per_word_forward_compiles_fullgraph_eager():
    """The serial ``conceptualMode`` path (per-word forward dispatch) must
    trace fullgraph too. ``MM_grammar.xml`` drives ``_forward_body_per_word``
    -> ``ConceptualSpace.forward`` -> ``_stm_shift_and_push`` -> the STM
    reduce sweep -> host-side relation learning. The two STM data-dependent
    guards are fixed (see the tensorized shift/reduce); the relation-learning
    hoist is tracked by the xfail above. Capped at one epoch -- we only need
    the first batch to TRACE."""
    rc, out, err = _run_cli("data/MM_grammar.xml",
                            env_extra={"MODEL_COMPILE": "eager",
                                       "BASIC_NUM_EPOCHS": "1"})
    combined = out + err
    tail = combined[-3000:]
    # The STM tensorizations are landed: these specific guards must stay gone.
    assert "u0 >= 8" not in combined, tail
    assert "Eq(u0, 1)" not in combined, tail
    # Full-path gate (currently xfail until learn_relations_from_stm hoist):
    assert "Unsupported" not in combined, tail
    assert "torch._dynamo.exc" not in combined, tail
    assert "Graph break" not in combined, tail
    assert rc == 0, f"rc={rc}\n{tail}"
