"""Native graph gate, recording first-forward allocation owners without changing it."""
from functools import wraps
import json
import os
from pathlib import Path
import torch
import Models
import test_word_store as original


def test_native_forward_allocation_owners(monkeypatch):
    destination = Path(os.environ['ITEM7_ALLOCATION_LOG'])
    profiler = None
    started = False
    run = Models.BasicModel._run_batch_once
    @wraps(run)
    def batch(self, *args, **kwargs):
        nonlocal profiler, started
        if not started:
            started = True
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True, profile_memory=True, with_stack=True)
            profiler.start()
        return run(self, *args, **kwargs)
    step = Models.BasicModel._sentence_train_step
    @wraps(step)
    def train(self, loss):
        nonlocal profiler
        if profiler is not None:
            profiler.stop()
            profiler.export_chrome_trace(str(destination))
            totals = []
            for event in profiler.key_averages(group_by_input_shape=True, group_by_stack_n=6):
                totals.append(dict(key=event.key, count=event.count, shapes=event.input_shapes,
                    bytes=event.self_cpu_memory_usage, total_bytes=event.cpu_memory_usage,
                    stack=event.stack))
            destination.with_suffix('.summary.json').write_text(json.dumps(sorted(totals,
                key=lambda e: -e['bytes']), indent=2)+'\n')
            profiler = None
        return step(self, loss)
    monkeypatch.setattr(Models.BasicModel, '_run_batch_once', batch)
    monkeypatch.setattr(Models.BasicModel, '_sentence_train_step', train)
    original.test_two_epoch_training_severs_cross_batch_graph(monkeypatch)
