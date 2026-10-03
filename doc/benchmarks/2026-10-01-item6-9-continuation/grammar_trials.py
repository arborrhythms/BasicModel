"""The predeclared ten fresh runs of each gate; keep every outcome."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
import bounded_tests as bounded
from measure import environment

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--companion', type=Path)
parser.add_argument('--gates', type=int, nargs='+', default=[5, 6])
args = parser.parse_args()
out = Path(args.output).resolve()
out.mkdir(exist_ok=False)
source = bounded.source_snapshot(ROOT)
bounded.write_json(out / 'source-manifest.json', dict(validated_source=source))
bounded.write_json(out / 'plan.json', dict(trials_per_gate=10, seed=None,
    class_bar=dict(correct=4, mse_less_than=.05), epochs=400,
    worker_bytes=8 * bounded.GIB, aggregate_bytes=24 * bounded.GIB,
    max_workers=3, companion=str(args.companion),
    harness={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}))
pending = [(gate, trial) for trial in range(10) for gate in args.gates]
active, done, peers = [], [], {}
started, peak = time.monotonic(), 0

try:
    while pending or active:
        external = 0
        if args.companion and args.companion.exists():
            for job in json.loads(args.companion.read_text())['active']:
                pid = job['pid']
                peers.setdefault(pid, bounded.ProcessTree(pid))
                external += peers[pid].sample()
        used = external
        for job in list(active):
            result = job['worker'].poll()
            used += job['worker'].current_memory_bytes
            if result is None:
                continue
            assert source == bounded.source_snapshot(ROOT)
            done.append(dict(gate=job['gate'], trial=job['trial'], process=result))
            active.remove(job)
        peak = max(peak, used)
        if used > 24 * bounded.GIB:
            raise RuntimeError('the shared aggregate memory guard stopped the gate campaign')
        while pending and len(active) < 3 and used + 1.5 * bounded.GIB < 24 * bounded.GIB:
            gate, trial = pending.pop(0)
            path = out / f'gate-{gate:02}-trial-{trial:02}'
            path.mkdir()
            worker = bounded.GuardedProcess(
                [sys.executable, str(HERE / 'measure.py'), 'group', '--root', str(ROOT),
                 '--output', str(path), '--gate', str(gate)],
                cwd=ROOT, env=environment(ROOT), log_path=path / 'supervisor.log',
                memory_bytes=8 * bounded.GIB, timeout=2150).start()
            active.append(dict(gate=gate, trial=trial, worker=worker))
            used += 1.5 * bounded.GIB
        bounded.write_json(out / 'progress.json', dict(completed=done,
            active=[dict(gate=j['gate'], trial=j['trial'], pid=j['worker'].proc.pid) for j in active],
            pending=len(pending), seconds=time.monotonic()-started, peak_memory_bytes=peak))
        time.sleep(.25)
finally:
    for job in active:
        job['worker'].stop(exit_code=130, reason='campaign_stopped')

rows = []
for job in sorted(done, key=lambda j: (j['gate'], j['trial'])):
    gate, trial = job['gate'], job['trial']
    path = out / f'gate-{gate:02}-trial-{trial:02}'
    observations = path / 'observations.jsonl'
    observed = [json.loads(line) for line in observations.read_text().splitlines()
                if json.loads(line)['kind'] == 'grammar'] if observations.exists() else []
    row = dict(gate=gate, trial=trial+1, process=job['process'], observations=observed)
    if len(observed) == 1:
        values = observed[0]
        answers, targets = values['predictions'], values['targets']
        row['mse'] = sum((a-b)**2 for a,b in zip(answers, targets))/4
        row['correct'] = sum((a > .5) == (b > .5) for a,b in zip(answers, targets))
        row['settled_bar'] = len(answers)==4 and row['correct']==4 and row['mse'] < .05
        assert len(values['gate_reconstructions']) == 4
        row['reconstructed'] = sum(Counter(a.split()) == Counter((b or '').replace(chr(0), ' ').split())
                                  for a,b in zip(values['inputs'], values['gate_reconstructions']))
        row['reconstruction_bar'] = row['reconstructed']==4 and not any(values['grammar_reconstruction_unavailable'])
    else:
        row['settled_bar'] = False
    rows.append(row)
classes = [r for r in rows if r['gate']==5]
all_ten = len(classes)==10 and all(r['settled_bar'] and r['process']['exit_code']==0 for r in classes)
summary = dict(rows=rows, all_ten_class_runs_meet_bar=all_ten,
    class_successes=sum(r['settled_bar'] for r in classes),
    reconstruction_successes=sum(r.get('reconstruction_bar', False) for r in rows if r['gate']==6),
    decision='Continue through steps 6–9 regardless of the class count (Alec, October 1).',
    seconds=time.monotonic()-started, peak_memory_bytes=peak)
bounded.write_json(out / 'summary.json', summary)
assert source == bounded.source_snapshot(ROOT)
print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))
