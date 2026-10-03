"""Compact read-only progress for the running closing dispatch."""
from pathlib import Path
from collections import Counter
import json
P=Path(__file__).resolve().parent

def read(p):
 try:return json.loads(p.read_text())
 except (OSError,json.JSONDecodeError):return None
if (P/'campaign-complete.json').exists():print('Campaign dispatch complete')
for phase in ('measurements','attribution'):
 r=read(P/phase/'progress.json')
 if r:
  done=r.get('done',[]);print(phase,dict(minutes=round(r['seconds']/60,1),completed=len(done),pending=r.get('pending'),active=[{k:v for k,v in a.items()if k!='memory_bytes'}for a in r['active']],failures=sum(x['process']['exit_code']!=0 for x in done)))
for phase in ('extra-cases','full-sweep'):
 dirs=sorted((P/phase).glob('part-*'))
 if not dirs:continue
 part=dirs[-1];r=read(part/'result.json')
 if r:
  print(phase,part.name,'selected',len(r.get('selected',[])),'completed',len(r.get('completed',[])),'counts',dict(Counter(x['outcome']for w in r.get('workers',[])for x in w.get('reports',[]))))
  active=[]
  for f in sorted(part.glob('worker-*.json')):
   if f.name.endswith(('.request.json','.process.json')):continue
   w=read(f)
   if w and w.get('active'):active.append(w['active'])
  print('active_cases',active)
