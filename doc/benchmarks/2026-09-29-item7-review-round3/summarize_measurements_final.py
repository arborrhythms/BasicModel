"""Report all declared results; no baseline/threshold selection."""
import json,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=[]
for seed in range(8):
    record={'seed':seed}
    for label in ('head','candidate'):
        out=HERE/(label+'-reconstruction'+('-final' if label=='candidate' else ''))
        process=json.loads((out/'processes.json').read_text())[str(seed)]
        result=json.loads((out/f'seed-{seed}.json').read_text()) if (out/f'seed-{seed}.json').exists() else {}
        phases={p['name']:p for p in result.get('phases',[])}
        record[label]=dict(process=process,phases=phases,
            store_rows=result.get('store_rows'),definition_rows=result.get('definition_rows'),
            definition_context_reads=result.get('definition_context_reads'),
            initial_atom_prefix_sha256=result.get('initial_atom_prefix_sha256'))
        diagnostic_path=out/f'seed-{seed}-diagnostic.json'
        if process.get('diagnostic_only'):
            record[label]['diagnostic_only']=dict(process=process['diagnostic_only'],
                result=json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else {})
    rows.append(record)
summary={}
for label in ('head','candidate'):
    values=[r[label]['phases']['after_training']['reconstruction_mean'] for r in rows
        if r[label]['process']['exit_code']==0 and 'after_training' in r[label]['phases']]
    summary[label]=dict(completed=len(values),mean=statistics.mean(values) if values else None,
        median=statistics.median(values) if values else None,minimum=min(values) if values else None,maximum=max(values) if values else None)
diagnostic_inclusive={}
for label in ('head','candidate'):
    values=[]
    for row in rows:
        entry=row[label]
        if entry['process']['exit_code']==0 and 'after_training' in entry['phases']:
            values.append(dict(seed=row['seed'],source='guarded',
                value=entry['phases']['after_training']['reconstruction_mean']))
        else:
            diagnostic=entry.get('diagnostic_only',{})
            phases={p['name']:p for p in diagnostic.get('result',{}).get('phases',[])}
            if diagnostic.get('process',{}).get('exit_code')==0 and 'after_training' in phases:
                values.append(dict(seed=row['seed'],source='unguarded diagnostic only',
                    value=phases['after_training']['reconstruction_mean']))
    numbers=[v['value'] for v in values]
    diagnostic_inclusive[label]=dict(values=values,completed=len(values),
        mean=statistics.mean(numbers) if numbers else None,
        median=statistics.median(numbers) if numbers else None,
        minimum=min(numbers) if numbers else None,maximum=max(numbers) if numbers else None)
report=dict(seeds=list(range(8)),rows=rows,summary=summary,rebaselined=False,
    diagnostic_inclusive_summary=diagnostic_inclusive,
    diagnostic_policy='Distribution diagnosis only; an unguarded value never changes its guarded outcome.')
(HERE/'reconstruction-comparison-final.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Reconstruction: all eight declared seeds','',
       'The reviewed reconstruction baseline remains unchanged. Two warmup and five measured training batches; four validation batches before and after, batch size two. No million-sentence campaign.','',
       '| Seed | HEAD before | Candidate before | HEAD after | Candidate after | HEAD process | Candidate process |',
       '|---|---:|---:|---:|---:|---|---|']
for row in rows:
    cells=[]
    for phase in ('before_training','after_training'):
        for label in ('head','candidate'):
            value=row[label]['phases'].get(phase,{}).get('reconstruction_mean')
            cells.append('unavailable' if value is None else f'{value:.8f}')
    cells += ['complete' if row[label]['process']['exit_code']==0 else row[label]['process']['reason']
              for label in ('head','candidate')]
    lines.append('| '+str(row['seed'])+' | '+' | '.join(cells)+' |')
diagnostics=[(row['seed'],label,row[label]['diagnostic_only']) for row in rows
             for label in ('head','candidate') if row[label].get('diagnostic_only')]
if diagnostics:
    lines += ['', '## One unguarded diagnostic for each memory stop', '',
        'These values are separate from the guarded results above. Each original memory stop remains red.', '',
        '| Seed | Tree | Before | After | Diagnostic process | Peak GiB |',
        '|---|---|---:|---:|---|---:|']
    for seed,label,entry in diagnostics:
        phases={p['name']:p for p in entry['result'].get('phases',[])}
        cells=[]
        for phase in ('before_training','after_training'):
            value=phases.get(phase,{}).get('reconstruction_mean')
            cells.append('unavailable' if value is None else f'{value:.8f}')
        process=entry['process']
        outcome='complete' if process['exit_code']==0 else process['reason']
        lines.append('| '+str(seed)+' | '+label+' | '+' | '.join(cells)+
            f" | {outcome} | {process['peak_memory_bytes']/2**30:.2f} |")
lines+=['','All process results, training means, timings, initial atom fingerprints, definition counts and context-read counts are in [the comparison](reconstruction-comparison-final.json). These are concurrent-run timings, not a controlled speed comparison.','']
(HERE/'reconstruction-table-final.md').write_text('\n'.join(lines))
if '--reconstruction-only' in sys.argv[1:]:
    print(json.dumps(dict(guarded=summary,diagnostic_inclusive=diagnostic_inclusive),indent=2))
    raise SystemExit(0)

mm=[]
for trial in range(10):
    row={'trial':trial}
    for label in ('head','candidate'):
        out=HERE/(label+'-mm-grammar'+('-final' if label=='candidate' else ''))
        process=json.loads((out/'processes.json').read_text())[str(trial)]
        path=out/f'run-{trial:02}.json'
        row[label]=dict(process=process,result=json.loads(path.read_text()) if path.exists() else {})
        if process['exit_code']:
            log=out/f'run-{trial:02}.log'
            lines=log.read_text().splitlines() if log.exists() else []
            errors=[line for line in lines if line.startswith(('RuntimeError:', 'ValueError:',
                    'IndexError:', 'AssertionError:', 'FloatingPointError:'))]
            row[label]['failure']=dict(error=errors[-1] if errors else None,
                last_reported_epoch=row[label]['result'].get('completed_epochs'),
                ending_900_epoch_measurement_available=False)
        diagnostic_path=out/f'run-{trial:02}-diagnostic.json'
        if process.get('diagnostic_only'):
            row[label]['diagnostic_only']=dict(process=process['diagnostic_only'],
                result=json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else {})
    mm.append(row)
stats={}
for label in ('head','candidate'):
    values=[r[label]['result']['ending_training_mse'] for r in mm
            if r[label]['process']['exit_code']==0 and r[label]['result'].get('completed_epochs')==900]
    stats[label]=dict(completed=len(values),mean=statistics.mean(values) if values else None,
        median=statistics.median(values) if values else None,minimum=min(values) if values else None,
        maximum=max(values) if values else None,ending_below_005=sum(v<.05 for v in values))
(HERE/'mm-grammar-comparison-final.json').write_text(json.dumps(dict(rows=mm,summary=stats),indent=2)+'\n')
lines=['# MM_grammar: ten fresh unseeded runs per tree','',
       'Every attempt has a 900-epoch budget. A failed process is shown as failed, with no 900-epoch ending measurement. For completed runs, the main error and predictions are the final training forward, before its update, matching the gate\'s observation point. The raw comparison also reports a separate evaluation after the 900th update. No stopping threshold, seed selection or configuration edit.','',
       '| Run | HEAD ending MSE | HEAD predictions | Candidate ending MSE | Candidate predictions | Candidate process |',
       '|---|---:|---|---:|---|---|']
for row in mm:
    cells=[]
    for label in ('head','candidate'):
        result=row[label]['result']
        complete=row[label]['process']['exit_code']==0 and result.get('completed_epochs')==900
        value=result.get('ending_training_mse') if complete else None
        cells += ['unavailable' if value is None else f'{value:.8f}',
                  ', '.join(f'{x:.6g}' for x in result.get('ending_training_predictions',[])) if complete else 'unavailable']
    process=row['candidate']['process']
    cells.append('complete' if process['exit_code']==0 else f"failed (exit {process['exit_code']})")
    lines.append('| '+str(row['trial'])+' | '+' | '.join(cells)+' |')
failed=[(row['trial'],label,row[label]) for row in mm for label in ('head','candidate')
        if row[label]['process']['exit_code']]
if failed:
    lines += ['', '## Failed attempts', '',
        'Completed-run statistics exclude these unavailable ending measurements; all ten attempts per tree remain in the table and comparison.', '']
    for trial,label,entry in failed:
        failure=entry.get('failure',{})
        epoch=failure.get('last_reported_epoch')
        checkpoint='before the first 50-epoch report' if epoch is None else f'after the last saved report at epoch {epoch}'
        lines.append(f"- {label}, run {trial}: {failure.get('error') or entry['process']['reason']}; {checkpoint}.")
lines+=['','All four targets, post-update predictions, memory use and process results are retained in [the complete comparison](mm-grammar-comparison-final.json).','']
(HERE/'mm-grammar-table-final.md').write_text('\n'.join(lines))
print(json.dumps(dict(reconstruction=summary,mm_grammar=stats),indent=2))
