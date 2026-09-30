"""Dispatch unchanged bounded workers using measured time and memory.

This is a receipt-harness adapter, not a model/test change. It changes only
batch construction and admission order. The existing runner still collects,
executes, enforces every hard limit, handles failures/recycling and verifies
exact coverage. Estimates reserve headroom; they never replace memory guards.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

GIB = 2**30


class AdmissionQueue:
    """Keep blocked batches queued while exposing affordable batches to dispatch."""
    def __init__(self, batches, demand, occupied, budget):
        self.pending = list(batches)
        self.demand, self.occupied, self.budget = demand, occupied, budget
        self.visible = []

    def __len__(self):
        return len(self.pending)

    def __iter__(self):
        available = self.budget - self.occupied()
        self.visible = [i for i, batch in enumerate(self.pending)
                        if self.demand(batch) <= available]
        return iter([self.pending[i] for i in self.visible])

    def __getitem__(self, index):
        return self.pending[self.visible[index]]

    def __delitem__(self, index):
        del self.pending[self.visible[index]]

    def appendleft(self, batch):
        self.pending.insert(0, batch)


class History:
    def __init__(self, paths):
        self.seconds, self.peaks, self.inputs = defaultdict(float), {}, {}
        for path in paths:
            path = Path(path)
            raw = path.read_bytes()
            self.inputs[str(path)] = hashlib.sha256(raw).hexdigest()
            for worker in json.loads(raw).get('workers', ()):
                for report in worker.get('reports', ()):
                    node = report['nodeid']
                    self.seconds[node] = max(self.seconds[node], report.get('duration', 0))
                    self.peaks[node] = max(self.peaks.get(node, 0), worker['peak_memory_bytes'])

    def demand(self, nodes):
        # Unmeasured cases reserve their complete unchanged 8 GiB ceiling.
        if any(node not in self.peaks for node in nodes):
            return 8 * GIB
        floor = 1.5 * GIB if len(nodes) > 1 else GIB
        return min(8 * GIB, max(floor, max(self.peaks[n] for n in nodes) * 1.3 + .25 * GIB))

    def batches(self, nodes, batch_size, max_files, devices):
        batches, small, files, duration = [], [], set(), 0.
        for node in nodes:
            filename = node.split('::', 1)[0]
            expensive = self.seconds[node] > 30 or self.peaks.get(node, 8 * GIB) > 1.5 * GIB
            boundary = (len(small) >= batch_size or
                        (filename not in files and len(files) >= max_files) or
                        duration + self.seconds[node] > 60 or
                        (small and devices[node] != devices[small[0]]))
            if small and (expensive or boundary):
                batches.append(small)
                small, files, duration = [], set(), 0.
            if expensive:
                batches.append([node])
            else:
                small.append(node)
                files.add(filename)
                duration += self.seconds[node]
        if small:
            batches.append(small)
        batches.sort(key=lambda batch: -sum(self.seconds[node] for node in batch))
        assert Counter(n for batch in batches for n in batch) == Counter(nodes)
        return batches


@contextmanager
def scheduled(bounded, *, history, budget, schedule_path):
    """Install queue/launch bookkeeping, leaving guard and result logic intact."""
    original_queue, original_guard, original_batches = (
        bounded.deque, bounded.GuardedProcess, bounded.make_batches)
    live = []

    def occupied():
        return sum(max(worker.reserved, worker.current_memory_bytes)
                   for worker in live if not worker.finished)

    class AccountedGuard(original_guard):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            request = json.loads(Path(command[-2]).read_text())
            self.reserved = 0 if request.get('collect') else history.demand(request['selectors'])

        def start(self):
            result = super().start()
            if not self.finished:
                live.append(self)
            return result

    def batches(nodes, batch_size, max_files, devices):
        result = history.batches(nodes, batch_size, max_files, devices)
        Path(schedule_path).write_text(json.dumps(dict(
            historical_inputs=history.inputs, aggregate_bytes=budget,
            rationale=__doc__, expected_cases=len(nodes),
            batches=[dict(nodes=batch, estimated_seconds=sum(history.seconds[n] for n in batch),
                          reserved_bytes=history.demand(batch)) for batch in result]), indent=2) + '\n')
        return result

    bounded.deque = lambda iterable=(): AdmissionQueue(iterable, history.demand, occupied, budget)
    bounded.GuardedProcess = AccountedGuard
    bounded.make_batches = batches
    try:
        yield
    finally:
        bounded.deque, bounded.GuardedProcess, bounded.make_batches = (
            original_queue, original_guard, original_batches)
