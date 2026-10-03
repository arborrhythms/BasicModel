"""Refresh final destinations while preserving intermediate reviewed bodies."""
import json
from port_ledger import HERE, ROOT, LEDGER, definitions
rows=json.loads(LEDGER.read_text())
cache={}
changed=0
for row in rows:
    for new in row['new']:
        file,name=new['id'].split('::',1)
        if file not in cache:
            cache[file]=definitions((ROOT/file).read_text())
        body=cache[file][name]
        if body != new['body']:
            history=new.setdefault('intermediate_bodies',[])
            if new['body'] not in history:
                history.append(new['body'])
            new['body']=body
            changed+=1
LEDGER.write_text(json.dumps(rows,indent=2)+'\n')
lines=['# Test and source dispositions','',
       'The JSON ledger retains complete original, intermediate and final bodies. Retained live-path reasons and checkpoint evidence are also in legacy-case-dispositions.json; the precursor map names its live witnesses or retirement decision.','',
       '| Original definition | Destination or retirement | Reason | Evidence |',
       '|---|---|---|---|']
for row in rows:
    dest=', '.join('`'+n['id']+'`' for n in row['new']) or 'Retired'
    evidence=', '.join('`'+str(e)+'`' for e in row['evidence'])
    lines.append('| `'+row['old_id']+'` | '+dest+' | '+row['reason'].replace('|','\\|').replace('\n',' ')+' | '+evidence+' |')
(HERE/'test-dispositions.md').write_text('\n'.join(lines)+'\n')
print({'records':len(rows),'destinations_refreshed':changed})
