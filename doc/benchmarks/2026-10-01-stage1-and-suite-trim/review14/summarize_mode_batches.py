"""Read and report every final first-batch outcome, including errors/guards."""
from pathlib import Path
from collections import Counter
import json
R=Path(__file__).resolve().parent;out=R/'mode-final-batches'
complete=json.loads((out/'complete.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
rows=[]
for process in complete['results']:
 config=process['config'];p=out/(Path(config).stem+'.json');result=json.loads(p.read_text()) if p.exists() else {}
 status='completed' if result.get('completed') and process['exit_code']==0 else 'exception' if result.get('error') else process['reason']
 rows.append(dict(config=config,status=status,process=process,measurement=result))
rows.sort(key=lambda r:r['config'])
summary=dict(source_matched=complete['source_matched'],counts=dict(Counter(r['status'] for r in rows)),configurations=len(rows),seconds=complete['elapsed_seconds'],rows=rows)
(R/'final-configuration-first-batches.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Final migrated configurations: first batch','', 'One unseeded first training batch per configuration, at its own configured batch, on the final candidate. CPU numerical execution with compile backend none excludes graph capture and evaluation. Each fresh worker has the unchanged 8 GiB ceiling and 30-minute deadline; two workers share 16 GiB. Configurations without grammar retain perceptual reconstruction. These are timing attempts, not gate results. Earlier failed attempts and repair probes remain in the adjacent receipt directories.','',f"Outcomes: {summary['counts']}; {len(rows)} configurations; source matched: {summary['source_matched']}; wall time {summary['seconds']/60:.2f} minutes.",'', '| Configuration | Batch | Scope | Training seconds | Worker seconds | Peak GiB | Outcome |','|---|---:|---|---:|---:|---:|---|']
for row in rows:
 m=row['measurement'];p=row['process'];training=m.get('training_seconds');sec='—' if training is None else f'{training:.4f}'
 lines.append(f"| `{row['config']}` | {m.get('batch','—')} | {m.get('scope','not built')} | {sec} | {p['elapsed_seconds']:.3f} | {p['peak_memory_bytes']/1024**3:.3f} | {row['status']} |")
lines+=['','## Incomplete attempts','', 'An exception is a measured inability to finish the batch, not a successful reconstruction. Numeric configurations were also audited because they inherit the template; the four numeric attempts fail while loading MNIST numpy.object_ data, before model construction, so they supply no model timing or evidence about arithmetic. No data, threshold, capacity or optimizer setting was changed to obtain a passing row.','']
for row in rows:
 if row['status']=='completed':continue
 m=row['measurement'];p=row['process']
 lines.append(f"- `{row['config']}`: {m.get('error') or p['reason']}. See `mode-final-batches/{Path(row['config']).stem}.log`.")
(R/'final-configuration-first-batches.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:summary[k] for k in ['source_matched','counts','configurations','seconds']}))
