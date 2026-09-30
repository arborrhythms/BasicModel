"""Scheduling probes: no pytest/model behavior is substituted."""
from collections import Counter
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('receipt_resource_schedule', Path(__file__).with_name('resource_schedule.py'))
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)
AdmissionQueue, History, GIB = schedule.AdmissionQueue, schedule.History, schedule.GIB


def dispatch(queue, active, limit, demand):
    while len(active) < limit and queue:
        candidate = next(enumerate(queue), None)
        if candidate is None:
            break
        index, batch = candidate
        assert queue[index] == batch
        del queue[index]
        active.append((batch, demand(batch)))


def test_uses_ten_affordable_slots_without_exceeding_budget():
    active = []
    demand = lambda batch: GIB
    queue = AdmissionQueue([[str(i)] for i in range(20)], demand,
                           lambda: sum(v for _, v in active), 24 * GIB)
    dispatch(queue, active, 10, demand)
    assert len(active) == 10
    assert len(queue) == 10


def test_large_batches_wait_and_small_batches_use_remaining_capacity():
    active = []
    sizes = {'large': 8 * GIB, 'small': GIB}
    demand = lambda batch: sizes[batch[0]]
    queue = AdmissionQueue([['large']] * 5 + [['small']] * 10, demand,
                           lambda: sum(v for _, v in active), 24 * GIB)
    dispatch(queue, active, 10, demand)
    assert len(active) == 3 and sum(v for _, v in active) == 24 * GIB
    active.pop()
    sizes['large'] = 9 * GIB  # Growth blocks the next large job; smaller jobs still fit.
    dispatch(queue, active, 10, demand)
    assert len(active) == 10 and sum(v for _, v in active) == 24 * GIB


@pytest.mark.parametrize('budget', [8, 24])
def test_admission_retains_every_batch_once_including_recycled_work(budget):
    active, finished = [], []
    demand = lambda batch: (1 + int(batch[0]) % 8) * GIB
    wanted = [[str(i)] for i in range(50)]
    queue = AdmissionQueue(wanted[1:], demand, lambda: sum(v for _, v in active), budget * GIB)
    queue.appendleft(wanted[0])
    while queue or active:
        dispatch(queue, active, 10, demand)
        assert active, 'admission deadlocked'
        assert sum(v for _, v in active) <= budget * GIB
        finished.append(active.pop(0)[0][0])
    assert Counter(finished) == Counter(batch[0] for batch in wanted)


def test_grouping_preserves_coverage_devices_and_isolates_heavy_cases():
    h = History([])
    nodes = [f'file{i // 3}.py::case{i}' for i in range(30)]
    h.peaks = {node: .3 * GIB for node in nodes}
    h.seconds.update({node: .01 for node in nodes})
    h.seconds[nodes[8]] = 1200
    devices = {node: ('cpu' if i < 20 else 'mps') for i, node in enumerate(nodes)}
    batches = h.batches(nodes, 8, 3, devices)
    assert Counter(n for batch in batches for n in batch) == Counter(nodes)
    assert batches[0] == [nodes[8]]
    assert all(len(set(devices[n] for n in batch)) == 1 for batch in batches)
    assert all(len(batch) <= 8 and len(set(n.split('::')[0] for n in batch)) <= 3 for batch in batches)
    assert h.demand(['unknown']) == 8 * GIB
