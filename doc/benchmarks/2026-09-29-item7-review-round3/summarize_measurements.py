"""Report all declared results; no baseline/threshold selection."""
import json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=[]
for seed in range(8):
    record={'seed':seed}
    for label in ('head','candidate'):
        out=HERE/(label+'-reconstruction')
        process=json.loads((out/'processes.json').read_text())[str(seed)]
        result=json.loads((out/f'seed-{seed}.json').read_text()) if (out/f'seed-{seed}.json').exists() else {}
        phases={p['name']:p for p in result.get('phases',[])}
        record[label]=dict(process=process,phases=phases,
            store_rows=result.get('store_rows'),definition_rows=result.get('definition_rows'),
            definition_context_reads=result.get('definition_context_reads'),
            initial_atom_prefix_sha256=result.get('initial_atom_prefix_sha256'))
    rows.append(record)
summary={}
for label in ('head','candidate'):
    values=[r[label]['phases']['after_training']['reconstruction_mean'] for r in rows
        if r[label]['process']['exit_code']==0 and 'after_training' in r[label]['phases']]
    summary[label]=dict(completed=len(values),mean=statistics.mean(values) if values else None,
        median=statistics.median(values) if values else None,minimum=min(values) if values else None,maximum=max(values) if values else None)
report=dict(seeds=list(range(8)),rows=rows,summary=summary,rebaselined=False)
(HERE/'reconstruction-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Reconstruction: all eight declared seeds','',
       'The reviewed reconstruction baseline remains unchanged. Two warmup and five measured training batches; four validation batches before and after, batch size two. No million-sentence campaign.','',
       '| Seed | HEAD before | Candidate before | HEAD after | Candidate after |',
       '|---|---:|---:|---:|---:|']
for row in rows:
    cells=[]
    for phase in ('before_training','after_training'):
        for label in ('head','candidate'):
            value=row[label]['phases'].get(phase,{}).get('reconstruction_mean')
            cells.append('unavailable' if value is None else f'{value:.8f}')
    lines.append('| '+str(row['seed'])+' | '+' | '.join(cells)+' |')
lines+=['','All process results, training means, timings, initial atom fingerprints, definition counts and context-read counts are in [the comparison](reconstruction-comparison.json). These are concurrent-run timings, not a controlled speed comparison.','']
(HERE/'reconstruction-table.md').write_text('\n'.join(lines))

mm=[]
for trial in range(10):
    row={'trial':trial}
    for label in ('head','candidate'):
        out=HERE/(label+'-mm-grammar')
        process=json.loads((out/'processes.json').read_text())[str(trial)]
        path=out/f'run-{trial:02}.json'
        row[label]=dict(process=process,result=json.loads(path.read_text()) if path.exists() else {})
    mm.append(row)
stats={}
for label in ('head','candidate'):
    values=[r[label]['result']['ending_training_mse'] for r in mm
            if r[label]['process']['exit_code']==0 and r[label]['result'].get('completed_epochs')==900]
    stats[label]=dict(completed=len(values),mean=statistics.mean(values) if values else None,
        median=statistics.median(values) if values else None,minimum=min(values) if values else None,
        maximum=max(values) if values else None,ending_below_005=sum(v<.05 for v in values))
(HERE/'mm-grammar-comparison.json').write_text(json.dumps(dict(rows=mm,summary=stats),indent=2)+'\n')
lines=['# MM_grammar: ten fresh unseeded runs per tree','',
       'All 900 epochs run. The main error and predictions are the final training forward, before its update, matching the gate\'s observation point. The raw comparison also reports a separate evaluation after the 900th update. No stopping threshold, seed selection or configuration edit.','',
       '| Run | HEAD ending MSE | HEAD predictions | Candidate ending MSE | Candidate predictions |',
       '|---|---:|---|---:|---|']
for row in mm:
    cells=[]
    for label in ('head','candidate'):
        result=row[label]['result'];value=result.get('ending_training_mse')
        cells += ['unavailable' if value is None else f'{value:.8f}',
                  ', '.join(f'{x:.6g}' for x in result.get('ending_training_predictions',[]))]
    lines.append('| '+str(row['trial'])+' | '+' | '.join(cells)+' |')
lines+=['','All four targets, post-update predictions, memory use and process results are retained in [the complete comparison](mm-grammar-comparison.json).','']
(HERE/'mm-grammar-table.md').write_text('\n'.join(lines))
print(json.dumps(dict(reconstruction=summary,mm_grammar=stats),indent=2))
