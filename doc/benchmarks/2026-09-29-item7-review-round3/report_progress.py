"""Compact read-only progress for the long declared measurements."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
for label in ('head','candidate'):
    for kind in ('reconstruction','mm-grammar'):
        out=HERE/(label+'-'+kind)
        if not out.exists():continue
        p=out/'processes.json';done=json.loads(p.read_text()) if p.exists() else {}
        print(label,kind,'completed:',len(done),'failures:',[(k,v['reason']) for k,v in done.items() if v['exit_code']])
        for index in range(8 if kind=='reconstruction' else 10):
            if str(index) in done:continue
            stem=f'seed-{index}' if kind=='reconstruction' else f'run-{index:02}'
            p=out/(stem+'.json')
            if p.exists():
                d=json.loads(p.read_text())
                print('  ',index,[x['name'] for x in d.get('phases',[])] if kind=='reconstruction' else d.get('completed_epochs'))
            elif (out/(stem+'.log')).exists():print('  ',index,'started')
