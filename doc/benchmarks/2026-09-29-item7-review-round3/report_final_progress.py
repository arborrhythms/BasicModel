"""Read the repaired source's ongoing receipt without changing it."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
status = HERE / 'finish-after-owner-status.json'
if status.exists():
    print('stage:', json.loads(status.read_text())['stage'])
for kind in ('reconstruction', 'mm-grammar'):
    directory = HERE / ('candidate-' + kind + '-final')
    if not directory.exists():
        continue
    path = directory / 'processes.json'
    done = json.loads(path.read_text()) if path.exists() else {}
    print(kind, 'completed:', len(done), 'failures:',
          [(k, v['reason']) for k, v in done.items() if v['exit_code']])
    for index, process in done.items():
        if process['reason'] not in ('memory', 'aggregate_memory'):
            continue
        stem = f'seed-{index}' if kind == 'reconstruction' else f'run-{int(index):02}'
        diagnostic = directory / (stem + '-diagnostic.json')
        if process.get('diagnostic_only'):
            print('diagnostic only:', index, process['diagnostic_only']['reason'],
                  'exit', process['diagnostic_only']['exit_code'])
        elif diagnostic.exists():
            data = json.loads(diagnostic.read_text())
            print('unguarded diagnostic:', index,
                  [x['name'] for x in data.get('phases', [])]
                  if kind == 'reconstruction' else data.get('completed_epochs'))
        elif (directory / (stem + '-diagnostic.log')).exists():
            print('unguarded diagnostic:', index, 'started')
    for index in range(8 if kind == 'reconstruction' else 10):
        if str(index) in done:
            continue
        stem = f'seed-{index}' if kind == 'reconstruction' else f'run-{index:02}'
        path = directory / (stem + '.json')
        if path.exists():
            data = json.loads(path.read_text())
            print(index, [x['name'] for x in data.get('phases', [])]
                  if kind == 'reconstruction' else data.get('completed_epochs'))
        elif (directory / (stem + '.log')).exists():
            print(index, 'started')
for name in ('final-xor-after-owner', 'explicit-after-owner', 'full-sweep/run'):
    path = HERE / name / 'result.json'
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    print(name, data['reason'], len(data.get('completed', [])), '/',
          len(data.get('selected', [])), 'cases;', len(data.get('groups', [])), 'groups')
    for progress in sorted(path.parent.glob('worker-*.json')):
        if '.' in progress.stem:
            continue
        worker = json.loads(progress.read_text())
        if worker.get('active') and worker.get('reason', 'running') == 'running':
            print('active:', worker['active'])
