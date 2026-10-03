"""Summarize stored observations only; never run a forward or training step."""
from collections import Counter
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
OUT = HERE/'measurements'


def read(path):
    return json.loads(path.read_text())


def summarize():
    completion = read(OUT/'complete.json')
    rows, table = [], []
    for job in completion['jobs']:
        folder = OUT/job['name']
        if job['kind'] == 'sum':
            path = folder/'measurement.json'
            rows.append(dict(group='sum', trial=job['trial'],
                process=job['process'], measurement=read(path) if path.exists() else None))
        elif job['kind'] == 'gate':
            path = folder/'run/result.json'
            result = read(path) if path.exists() else None
            records = []
            observed = folder/'observations.jsonl'
            if observed.exists():
                records = [json.loads(line) for line in observed.read_text().splitlines()]
            if job['gate'] in (5, 6):
                observations = [record for record in records if record['kind'] == 'grammar']
                value = None
                if observations:
                    record = observations[-1]
                    answers, targets = record['predictions'], record['targets']
                    y = dict(zip(record['inputs'], answers))
                    read_backs = record['gate_reconstructions']
                    reconstructed = sum(
                        b is not None and Counter(a.split()) == Counter(b.replace('\0', ' ').split())
                        for a, b in zip(record['inputs'], read_backs))
                    correct = sum((a>.5)==(t>.5) for a,t in zip(answers, targets))
                    error = sum((a-t)**2 for a,t in zip(answers, targets))/len(answers)
                    value = dict(answers=answers, targets=targets, inputs=record['inputs'],
                        mse=error, correct=correct, class_bar=correct==4 and error<.05,
                        read_backs=read_backs, reconstructed=reconstructed,
                        unavailable=record['grammar_reconstruction_unavailable'],
                        reconstruction_bar=reconstructed==4 and not any(record['grammar_reconstruction_unavailable']),
                        contrast=y['hello world']+y['loving there']-y['hello there']-y['loving world'])
                rows.append(dict(group='class' if job['gate']==5 else 'reconstruction',
                    trial=job['trial'], process=job['process'], measurement=value))
            if job['trial'] == 1:
                reports = [] if result is None else [r for w in result['workers'] for r in w['reports']]
                table.append(dict(gate=job['gate'], selected=[] if result is None else result['selected'],
                    counts=dict(Counter(r['outcome'] for r in reports)), reports=reports,
                    observations=records, process=job['process']))
    mm = []
    for job in completion['jobs']:
        if job['kind'] == 'mm':
            path = OUT/job['name']/'measurement.json'
            mm.append(dict(trial=job['trial'], process=job['process'],
                measurement=read(path) if path.exists() else None))
    groups = {}
    for group, key in [('class','class_bar'), ('reconstruction','reconstruction_bar'), ('sum','sum_bar')]:
        group_rows = [r for r in rows if r['group']==group]
        groups[group] = dict(attempts=len(group_rows), measured=sum(r['measurement'] is not None for r in group_rows),
            passes=sum(bool(r['measurement'] and r['measurement'][key]) for r in group_rows))
    mm_complete = [r['measurement']['ending_training_mse'] for r in mm
        if r['measurement'] and r['measurement'].get('completed_epochs')==900 and r['process']['exit_code']==0]
    summary = dict(groups=groups, gate_rows=rows, table=sorted(table,key=lambda r:r['gate']),
        table_counts=dict(sum((Counter(r['counts']) for r in table), Counter())),
        mm_rows=mm, mm_completed=len(mm_complete),
        mm_median=None if not mm_complete else statistics.median(mm_complete),
        source_matched=completion['source_matched'], seconds=completion['seconds'],
        baseline=dict(table='44/49', exact_roundtrip='12/15', mm_median=.1066),
        removed_table_entry='SPNN XOR smoke test moved inline by explicit trim item 3',
        table_denominator='49 - 14 repeated exact trials - 1 SPNN smoke case = 34 named cases')
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines = ['# Closing candidate measurements', '',
        'One unseeded attempt per declared run, candidate only. The accepted item-7 baseline remains 44/49, 12/15 exact round trips and MM_grammar median ending MSE .1066. The October 1 receipt rule removes fourteen repeated exact trials, and the explicit trim removes the SPNN smoke case: 49−14−1=34 named cases. No failed run is retried.', '',
        'XOR_grammar now inherits meronomy synthesis and analysis. Its grammar reading reconstructs through the understanding. The only control-fixture change is replacing the grammar operations with sum; the complete patch is saved beside this receipt.', '',
        '| Group | Completed measurements | Passes at decided bar |', '|---|---:|---:|']
    for group, info in groups.items():
        lines.append(f"| {group} | {info['measured']}/{info['attempts']} | {info['passes']}/{info['attempts']} |")
    lines += ['', "Alec clarified during this receipt that demonstration of learning is the primary goal and lack of consistency is acceptable. The observed class passes demonstrate learning; no ten-of-ten consistency requirement is added. The numerical bars and all test assertions remain unchanged. Low reconstruction remains a review concern; the counts alone do not identify its cause."]
    class_rows = [r['measurement'] for r in rows if r['group']=='class' and r['measurement']]
    lines += ['', f"Within the ten class-gate runs, {sum(m['reconstruction_bar'] for m in class_rows)} also meet the reconstruction bar and {sum(m['class_bar'] and m['reconstruction_bar'] for m in class_rows)} meet both bars together. These are cross-scores of the same saved observations, not additional training runs."]
    gate_values = [r['measurement'] for r in rows if r['group'] in ('class', 'reconstruction') and r['measurement']]
    lines += ['', f"Across the class and reconstruction cohorts, {sum(sum(m['unavailable']) for m in gate_values)}/{sum(len(m['unavailable']) for m in gate_values)} observed sentences lack an admitted reconstruction candidate. The unavailable-sentence flags are retained alongside the read-backs."]
    lines += ['', 'Answers are in the input order saved in `summary.json`. The contrast is y(hello world)+y(loving there)−y(hello there)−y(loving world).', '',
        '| Run | Four answers | MSE | Read-backs in the same order | Contrast |', '|---|---|---:|---|---:|']
    for row in rows:
        m = row['measurement']
        if m:
            answers=', '.join(f'{x:.7g}' for x in m['answers'])
            backs='; '.join(str(x).replace('|','\\|') for x in m['read_backs'])
            lines.append(f"| {row['group']} {row['trial']} | {answers} | {m['mse']:.8g} | {backs} | {m['contrast']:.8g} |")
        else:
            lines.append(f"| {row['group']} {row['trial']} | process {row['process']['reason']} | — | — | — |")
    lines += ['', 'The sum control passes only when the absolute checkerboard contrast is at most 1e-4 and the unchanged class bar is not met. Distance from one half remains a diagnostic, not an acceptance condition. Class and reconstruction bars are evaluated independently for every observed run.', '',
        '| Named table group | Cases | Outcomes |', '|---|---:|---|']
    for row in summary['table']:
        lines.append(f"| {row['gate']} | {len(row['selected'])} | {row['counts']} |")
    lines += ['', f"Named-table outcomes: {summary['table_counts']}.", '',
        '| Named case | Outcome |', '|---|---|']
    for row in summary['table']:
        for node in row['selected']:
            outcomes = [r['outcome'] for r in row['reports'] if r['nodeid']==node]
            outcome = ', '.join(outcomes) if outcomes else f"process {row['process']['reason']}/{row['process']['exit_code']}"
            lines.append(f'| `{node}` | {outcome} |')
    lines += ['',
        f"MM_grammar completed {len(mm_complete)}/10 full 900-update runs. Median ending training MSE: {summary['mm_median']} (accepted item 7: .1066).", '',
        'These ten measurements retain the accepted 900-update, direct-forward MSE harness. That harness does not run sentence trials, so its median is not evidence that the joint cost has learned XOR. The named MM_grammar regression in the table is a separate single run that stops when it reaches .20; it is not substituted for a complete 900-update measurement.', '',
        '| MM run | Completed epochs | Ending training MSE | Process |', '|---|---:|---:|---|']
    for row in mm:
        m = row['measurement'] or {}
        lines.append(f"| {row['trial']} | {m.get('completed_epochs',0)} | {m.get('ending_training_mse','—')} | {row['process']['reason']}/{row['process']['exit_code']} |")
    lines += ['', f"Source matched: {completion['source_matched']}. Measurement campaign wall time: {completion['seconds']/60:.2f} minutes. The final full-sweep receipt is recorded separately in `full-sweep/receipt.json`.", '']
    (HERE/'README.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    summary = summarize()
    print(json.dumps({k:summary[k] for k in ('groups','table_counts','mm_completed','mm_median','source_matched')}))
