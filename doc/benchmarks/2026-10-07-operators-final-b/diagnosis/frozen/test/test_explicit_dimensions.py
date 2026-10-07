"""Failing tests that gate the removal of ``<nInputDim>-1</nInputDim>``.

These tests run the CLI end-to-end (``python bin/Models.py data/<config>.xml``)
because the existing unit tests bypass ``ModelFactory.run`` (no compile, no
global state) and therefore mask the regressions a user sees when they invoke
the CLI directly.

Each XML config below currently relies on ``nInputDim=-1`` to flatten
``[N, D] -> [1, N*D]`` and side-step the dim mismatch introduced by
``ConceptualSpace._build_combined_input`` (see ``bin/Spaces.py:7478``).
The flatten obscures the per-vector identity that ``decode_reverse_meta``
relies on, which is why reconstruction breaks for ``XOR_exact.xml`` and
``XOR_spaces.xml`` even though XOR prediction itself converges.

Other XML configs that currently use ``-1`` (kept here for inventory --
when we replace ``-1`` with explicit widths these must all keep passing
elsewhere)::

    data/BasicModel.xml      data/MM_400M.xml
    data/MM_xor_step3.xml    data/MM_xor_step4.xml
    data/XOR_exact.xml       data/XOR_spaces.xml
    data/XOR_recon.xml       data/XOR_pos.xml
    data/stream_smoke.xml

Existing tests touching those configs (must remain green when ``-1`` is
removed):

    test/test_basicmodel.py            (XOR_exact, XOR_pos)
    test/test_xor_spaces.py            (XOR_spaces)
    test/test_lexicon_ownership.py     (XOR_exact)
    test/test_streaming_ar_training.py (BasicModel, stream_smoke)
    test/test_stream_smoke.py          (stream_smoke)
    test/test_use_flags.py             (MM_xor_step4, MM_400M)
    test/test_testpoint.py             (BasicModel, XOR_exact, XOR_spaces,
                                        XOR_recon, XOR_pos)
"""

import os
import re
import subprocess
import sys
import unittest

import pytest
import torch

_RUN_SLOW = os.getenv("RUN_SLOW") == "1"

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENV_PYTHON = os.path.join(_PROJECT, ".venv", "bin", "python")
_MODELS_PY = os.path.join(_PROJECT, "bin", "Models.py")
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

def _run_cli(config_relpath, env_extra=None, timeout=180):
    """Invoke ``python bin/Models.py data/<config>.xml`` as a subprocess.

    Returns (returncode, stdout, stderr).
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("BASIC_SEED", None)  # capability checks use a fresh initialization
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [_VENV_PYTHON, _MODELS_PY, config_relpath],
        cwd=_PROJECT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _parse_piecewise_overall(stdout):
    """Pull the final 'Piecewise overall: M/T (P%)' percentage from stdout.

    Returns the integer percentage, or ``None`` if the line is absent.
    """
    m = re.search(r"Piecewise overall:\s+\d+/\d+\s+\((\d+)%\)", stdout)
    if not m:
        return None
    return int(m.group(1))


def _parse_correctly_predicted(stdout):
    """Pull 'Correctly predicted 0' / 'Correctly predicted 1' floats.

    Returns (acc0, acc1) -- either may be ``None`` if absent.
    """
    m0 = re.search(r"Correctly predicted 0:\s+([\d.]+)", stdout)
    m1 = re.search(r"Correctly predicted 1:\s+([\d.]+)", stdout)
    return (float(m0.group(1)) if m0 else None,
            float(m1.group(1)) if m1 else None)


def _parse_input_match_counts(stdout):
    """Count OK vs MISMATCH on the per-input reconstruction report lines.

    Each line (from ``BasicModel._reconstructionReport``) looks like::

        row[0] input='hello world' -> reconstructed='hello world' label=0.0000 predicted=-0.0056 OK

    Returns ``(ok_count, total_count)``. This is the right reconstruction
    metric for configs with ``nWhere=0`` -- the Piecewise metric
    structurally reports 0% in that regime because no per-token offsets
    are tracked.
    """
    lines = re.findall(
        r"^\s*row\[\d+\]\s+input=.*?\s+predicted=\S+\s+(OK|MISMATCH)\s*$",
        stdout, flags=re.MULTILINE)
    return sum(1 for s in lines if s == "OK"), len(lines)


def _parse_output_mse(stdout):
    """Mean-squared error of the XOR OUTPUT predictions vs labels, from the
    per-input report lines ``... label=<L> predicted=<P> (OK|MISMATCH)``.

    Returns ``(mse, n)`` -- ``(None, 0)`` when no rows are found. This is the
    metric the accuracy + reconstruction gates MISS: predictions can round to
    the right class (4/4 accuracy, reconstruction OK) yet sit near 0.5 (high
    MSE), as in the pre-lrScale readout-convergence regression (MSE ~0.22)."""
    pairs = re.findall(
        r"label=(-?\d+\.\d+)\s+predicted=(-?\d+\.\d+)", stdout)
    if not pairs:
        return None, 0
    se = [(float(p) - float(l)) ** 2 for l, p in pairs]
    return sum(se) / len(se), len(se)


class TestIdempotentCliRuns(unittest.TestCase):
    """``data/idempotent.xml`` is the minimal C-S round-trip config.

    It deliberately has ``numEpochs=0`` and an empty ``<dataset>inline</dataset>``
    so the test pass produces no ``outputDataPred``. The non-AR branch in
    ``runBatch`` (Models.py:3067) calls ``outputDataPred.squeeze()`` without a
    ``None`` guard, which crashes with ``AttributeError``.
    """

    @pytest.mark.slow
    def test_cli_does_not_crash(self):
        rc, stdout, stderr = _run_cli("data/idempotent.xml", timeout=60)
        self.assertEqual(
            rc, 0,
            f"CLI crashed (rc={rc})\nstdout tail:\n{stdout[-2000:]}\n"
            f"stderr tail:\n{stderr[-2000:]}",
        )




class TestXorExactCliReconstruction(unittest.TestCase):
    """``data/XOR_exact.xml`` should reconstruct its inputs end-to-end.

    The native field witnesses located order-0 conjunctions from primitive
    byte memberships. The supervised output learns an order-1 sigma and
    reads its positive pole. Reconstruction attributes owned forward field
    evidence to native rows and decodes by activity, without input events.
    """

    @pytest.mark.slow
    def test_at_least_50_pct_inputs_reconstruct(self):
        # Initialization may expose a learning defect; never select a passing basin.
        rc, stdout, stderr = _run_cli(
            "data/XOR_exact.xml", timeout=480,
            env_extra={"MODEL_COMPILE": "eager"})
        self.assertEqual(rc, 0, f"CLI failed: stderr={stderr[-1000:]}")
        ok, total = _parse_input_match_counts(stdout)
        print(f'XOR_exact reconstruction count: {ok}/{total}')
        print('\n'.join(line for line in stdout.splitlines() if 'Reconstructed:' in line))
        self.assertGreater(total, 0,
                           "Did not find any 'Input: ... -> Reconstructed: ...' lines")
        self.assertGreaterEqual(
            2 * ok, total,
            f"XOR_exact reconstruction: {ok}/{total} inputs match "
            f"(expected >=50%).",
        )

    @pytest.mark.slow
    def test_output_mse_is_crisp(self):
        # Accuracy alone misses predictions clustered near .5; keep MSE < .05.
        rc, stdout, stderr = _run_cli(
            "data/XOR_exact.xml", timeout=480,
            env_extra={"MODEL_COMPILE": "eager"})
        self.assertEqual(rc, 0, f"CLI failed: stderr={stderr[-1000:]}")
        mse, n = _parse_output_mse(stdout)
        self.assertIsNotNone(
            mse, "no 'label=.. predicted=..' report lines found")
        self.assertEqual(n, 4, f"expected 4 XOR rows, got {n}")
        self.assertLess(
            mse, 0.05,
            f"XOR_exact output MSE={mse:.4f} over {n} rows (expected <0.05; "
            f"crisp basin ~0.004). Predictions near 0.5 => the readout "
            f"under-converged (regression of the OutputSpace lrScale=0.5 "
            f"two-timescale fix).")


def _run_xor_grammar_in_process():
    """Run the grammar capability gate without selecting an initialization."""
    os.environ["MODEL_COMPILE"] = "none"
    from unittest.mock import patch
    from Models import BasicModel, ModelFactory
    commit = BasicModel._commit_sentence
    def capture(model, state, sid, active, *args):
        if not model._sentence_training:
            # Observe the existing evaluation before its temporary trace is
            # discarded. This neither advances the clock nor keeps the trace.
            with torch.no_grad():
                texts, unavailable = model.reconstruct_grammar_sentence(state, sid, active)
            model._grammar_gate_reconstructions = texts
            model._grammar_gate_unavailable = unavailable.detach().cpu().tolist()
        return commit(model, state, sid, active, *args)
    with patch.object(BasicModel, '_commit_sentence', capture):
        results = ModelFactory.run("data/XOR_grammar.xml")
    return results[0][2]  # (name, rCorrect, model)


@pytest.fixture(scope="module")
def trained_xor_grammar():
    """One unseeded training supplies both independent capability bars."""
    if not _RUN_SLOW:
        pytest.skip("slow (XOR_grammar train) -- set RUN_SLOW=1")
    return _run_xor_grammar_in_process()


@pytest.fixture
def _bind_trained_xor_grammar(request, trained_xor_grammar):
    # unittest assertions stay intact; pytest supplies the shared module fixture.
    request.instance.trained_xor_grammar = trained_xor_grammar


@pytest.mark.usefixtures("_bind_trained_xor_grammar")
@pytest.mark.shared_training("xor_grammar")
class TestXorGrammarLearnsXor(unittest.TestCase):
    """Grammar XOR convergence must survive an arbitrary initialization."""

    @unittest.skipIf(not _RUN_SLOW, "slow (~60s end-to-end XOR_grammar train) -- set RUN_SLOW=1")
    def test_xor_class_accuracy(self):
        model = self.trained_xor_grammar
        # Read the answers saved by the final evaluation. Another forward
        # would advance the clock; rCorrect is an embedding-report placeholder.
        data = model.inputSpace.data
        answers = torch.stack(data.reconstructed_output).reshape(-1)
        targets = torch.stack(data.test_output).reshape(-1).to(answers)
        self.assertEqual(answers.numel(), 4)
        self.assertEqual(targets.numel(), 4)
        self.assertTrue(bool(torch.isfinite(answers).all()))
        correct = (answers > 0.5) == (targets > 0.5)
        mse = torch.mean((answers - targets).square()).item()
        self.assertTrue(
            bool(correct.all()),
            f"XOR_grammar requires all four answers correct: {answers.tolist()}",
        )
        self.assertLess(
            mse, 0.05,
            f"XOR_grammar requires MSE <0.05, got {mse}: {answers.tolist()}",
        )


@pytest.mark.usefixtures("_bind_trained_xor_grammar")
@pytest.mark.shared_training("xor_grammar")
class TestXorGrammarReconstruction(unittest.TestCase):
    """All four understandings return their own words, allowing transpositions."""

    @unittest.skipIf(not _RUN_SLOW, "slow (~65s end-to-end XOR_grammar train) -- set RUN_SLOW=1")
    def test_piecewise_overall_at_least_50_pct(self):
        # Keep the historical selector; the decided contract is now all four.
        from collections import Counter
        model = self.trained_xor_grammar
        recon_texts = model._grammar_gate_reconstructions
        test_input, _ = model.inputSpace.getTestData()
        self.assertEqual(len(test_input), 4)
        self.assertEqual(len(recon_texts), 4)
        self.assertFalse(any(model._grammar_gate_unavailable),
                         'every sentence must have an available grammar inverse')
        pairs = [(model._bytes_to_text(original).replace("\x00", " "), recovered)
                 for original, recovered in zip(test_input, recon_texts)]
        perfect = sum(recovered is not None and Counter(original.split()) ==
                      Counter(recovered.replace("\x00", " ").split())
                      for original, recovered in pairs)
        self.assertEqual(perfect, 4,
            f"XOR_grammar reconstruction: {perfect}/4 word multisets recovered; {pairs}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
