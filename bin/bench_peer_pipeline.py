"""Small serial-vs-peer-pipeline timing probe.

Usage: ``.venv/bin/python bin/bench_peer_pipeline.py --width 64 --device cpu``.
The callbacks deliberately use independent tensor work, matching the stage
contract rather than measuring Python list overhead alone.
"""
import argparse
import json
import time

import torch

from Models import StaticPeerPipeline


def _work(x):
    return torch.tanh(x @ x.transpose(-1, -2))


def measure(width, device, repeats):
    words = [torch.randn(1, 32, device=device) for _ in range(width)]

    def serial_once():
        feedback = None
        for index, word in enumerate(words):
            a = _work(word)
            b = _work(a) + (0 if feedback is None else feedback)
            feedback = _work(b)

    pipeline = StaticPeerPipeline(width)

    def pipelined_once():
        def stage_a(word, _index):
            return _work(word)

        def stage_b(a, _index, feedback):
            result = _work(a)
            if feedback is not None:
                result = result + feedback
            return result, _work(result)

        def stage_c(result, _index):
            _work(result)

        pipeline.run(words, stage_a, stage_b, stage_c)

    def elapsed(fn):
        if device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            fn()
        if device.type == "mps":
            torch.mps.synchronize()
        return (time.perf_counter() - started) / repeats

    # Warm the same static shape before timing.
    serial_once()
    pipelined_once()
    return {"width": width, "device": str(device), "serial_seconds": elapsed(serial_once),
            "pipelined_seconds": elapsed(pipelined_once)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=64,
                        choices=StaticPeerPipeline.WIDTHS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(measure(args.width, torch.device(args.device), args.repeats), indent=2))
