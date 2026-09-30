"""Stream a profiler trace to retain only large allocation owners and the peak."""
from pathlib import Path
import json
import gzip
from bisect import bisect_left, bisect_right
p=Path(__file__).with_name('ag-forward-allocations.json')
def events():
    buf=None
    stream = p.open() if p.exists() else gzip.open(p.with_suffix('.json.gz'), 'rt')
    with stream:
        for line in stream:
            if line.strip() == '{' and line.startswith('  {'):
                buf=[line]
            elif buf is not None:
                buf.append(line)
                if line.startswith('  }'):
                    yield json.loads(''.join(buf).strip().rstrip(','))
                    buf=None
allocations=[];peak=None
for e in events():
    if e.get('name') != '[memory]': continue
    if peak is None or e['args']['Total Allocated'] > peak['args']['Total Allocated']: peak=e
    if e['args']['Bytes'] >= 32 * 2**20: allocations.append(e)
if peak not in allocations: allocations.append(peak)
allocations.sort(key=lambda e:e['ts']);times=[e['ts'] for e in allocations]
for e in allocations:e['owners']=[]
for e in events():
    if e.get('ph') != 'X': continue
    for index in range(bisect_left(times,e['ts']),bisect_right(times,e['ts']+e['dur'])):
        a=allocations[index]
        if a['tid']==e['tid']:
            a['owners'].append(dict(name=e['name'],cat=e.get('cat'),ts=e['ts'],dur=e['dur'],args=e.get('args',{})))
for e in allocations:e['owners'].sort(key=lambda e:-e['dur'])
out=p.with_name('ag-allocation-owners.json');out.write_text(json.dumps(dict(peak=peak,allocations=allocations),indent=2)+'\n')
print('peak GiB',peak['args']['Total Allocated']/2**30,'large allocations',len(allocations))
for e in sorted(allocations,key=lambda e:-e['args']['Bytes'])[:10]:
 print(e['args']['Bytes']/2**30, [(o['name'],o['args'].get('Input Dims')) for o in e['owners'][-12:]])
