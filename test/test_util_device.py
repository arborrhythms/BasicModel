"""Unit tests for device auto-detection sanity checks."""

import os
import subprocess
import sys
import unittest
from unittest import mock
import warnings

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import util


class TestCudaSanityChecks(unittest.TestCase):
    def setUp(self):
        util._visible_nvidia_gpu_present.cache_clear()
        util._warn_if_cuda_unavailable_but_nvidia_visible.cache_clear()

    def tearDown(self):
        util._visible_nvidia_gpu_present.cache_clear()
        util._warn_if_cuda_unavailable_but_nvidia_visible.cache_clear()

    @mock.patch("util.platform.system", return_value="Linux")
    @mock.patch("util.os.path.exists", return_value=False)
    @mock.patch("util.shutil.which", return_value="/usr/bin/nvidia-smi")
    @mock.patch("util.subprocess.run")
    def test_visible_nvidia_gpu_present_via_nvidia_smi(
        self, run_mock, which_mock, exists_mock, system_mock
    ):
        run_mock.return_value = subprocess.CompletedProcess(
            args=["nvidia-smi", "-L"],
            returncode=0,
            stdout="GPU 0: Test GPU (UUID: GPU-123)\n",
            stderr="",
        )
        self.assertTrue(util._visible_nvidia_gpu_present())

    @mock.patch("util.platform.system", return_value="Linux")
    @mock.patch("util.torch.cuda.is_available", return_value=False)
    @mock.patch("util._visible_nvidia_gpu_present", return_value=True)
    def test_warns_when_nvidia_visible_but_cuda_unavailable(
        self, gpu_mock, cuda_mock, system_mock
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            util._warn_if_cuda_unavailable_but_nvidia_visible()
        self.assertEqual(len(caught), 1)
        self.assertIn("torch.cuda.is_available() is False", str(caught[0].message))

    @mock.patch.dict(os.environ, {"BASICMODEL_DEVICE": "cpu"}, clear=False)
    @mock.patch("util._warn_if_cuda_unavailable_but_nvidia_visible")
    def test_auto_device_cpu_override_skips_cuda_sanity(self, warn_mock):
        self.assertEqual(str(util.auto_device()), "cpu")
        warn_mock.assert_not_called()

    @mock.patch.dict(os.environ, {}, clear=False)
    @mock.patch("util._warn_if_cuda_unavailable_but_nvidia_visible")
    @mock.patch("util.torch.backends.mps.is_available", return_value=False)
    @mock.patch("util.torch.cuda.is_available", return_value=False)
    def test_auto_device_warns_before_cpu_fallback(
        self, cuda_mock, mps_mock, warn_mock
    ):
        previous = os.environ.pop("BASICMODEL_DEVICE", None)
        try:
            self.assertEqual(str(util.auto_device()), "cpu")
        finally:
            if previous is not None:
                os.environ["BASICMODEL_DEVICE"] = previous
        warn_mock.assert_called_once()

    @mock.patch.dict(os.environ, {}, clear=False)
    @mock.patch("util.resolve_device", return_value="mps")
    def test_auto_device_default_delegates_to_resolve_device(self, resolve_mock):
        previous = os.environ.pop("BASICMODEL_DEVICE", None)
        try:
            self.assertEqual(str(util.auto_device()), "mps")
        finally:
            if previous is not None:
                os.environ["BASICMODEL_DEVICE"] = previous
        resolve_mock.assert_called_once_with("gpu")

    @mock.patch("util.torch.backends.mps.is_available", return_value=True)
    @mock.patch("util.torch.cuda.is_available", return_value=False)
    @mock.patch("util._warn_if_cuda_unavailable_but_nvidia_visible")
    def test_resolve_device_gpu_falls_back_to_mps(
        self, warn_mock, cuda_mock, mps_mock
    ):
        self.assertEqual(str(util.resolve_device("gpu")), "mps")
        warn_mock.assert_called_once()

    @mock.patch("util.torch.cuda.is_available", return_value=False)
    @mock.patch("util._warn_if_cuda_unavailable_but_nvidia_visible")
    def test_resolve_device_cuda_unavailable_raises(
        self, warn_mock, cuda_mock
    ):
        with self.assertRaises(ValueError) as ctx:
            util.resolve_device("cuda")
        self.assertIn("CUDA is not available", str(ctx.exception))
        warn_mock.assert_called_once()

    @mock.patch("util.torch.backends.mps.is_available", return_value=False)
    def test_resolve_device_mps_unavailable_raises(self, mps_mock):
        with self.assertRaises(ValueError) as ctx:
            util.resolve_device("mps")
        self.assertIn("MPS is not available", str(ctx.exception))

    @mock.patch("util.torch.cuda.is_available", return_value=True)
    def test_resolve_device_cuda_index(self, cuda_mock):
        self.assertEqual(str(util.resolve_device("cuda:1")), "cuda:1")

    def test_resolve_device_invalid_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            util.resolve_device("definitely-not-a-device")
        self.assertIn("Unknown device override", str(ctx.exception))

    def test_rewrite_inductor_cmd_line_quotes_spacey_prefix(self):
        cmd = "-I/Users/test/Library/Mobile Documents/proj/include -L/Users/test/Library/Mobile Documents/proj/lib"
        prefix = "/Users/test/Library/Mobile Documents/proj"
        rewritten = util._rewrite_inductor_cmd_line(cmd, (prefix,))
        self.assertIn(f'-I"{prefix}"/include', rewritten)
        self.assertIn(f'-L"{prefix}"/lib', rewritten)

    @mock.patch("util._patch_inductor_paths")
    @mock.patch("util.torch.compile")
    def test_compile_tries_successive_backends(self, compile_mock, patch_mock):
        model = object()

        def _compile_side_effect(model_arg, backend=None, mode=None):
            self.assertEqual(mode, "max-autotune")
            if backend == "inductor":
                raise RuntimeError("inductor failed")
            if backend == "eager":
                return ("compiled", backend)
            raise AssertionError(f"Unexpected backend {backend}")

        compile_mock.side_effect = _compile_side_effect
        result = util.compile(model, verbose=False)
        self.assertEqual(result, ("compiled", "eager"))
        self.assertEqual(
            [call.kwargs["backend"] for call in compile_mock.call_args_list],
            ["inductor", "eager"],
        )
        patch_mock.assert_called_once()

    @mock.patch("util._patch_inductor_paths")
    @mock.patch("util.torch.compile")
    def test_compile_returns_original_model_if_all_backends_fail(self, compile_mock, patch_mock):
        model = object()
        compile_mock.side_effect = RuntimeError("all backends failed")
        result = util.compile(model, verbose=False)
        self.assertIs(result, model)
        self.assertEqual(
            [call.kwargs["backend"] for call in compile_mock.call_args_list],
            ["inductor", "eager", "aot_eager"],
        )
        patch_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
