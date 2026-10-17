"""Summarize the completed native run; never load or present a model."""
import argparse
from collections import Counter
from itertools import groupby
import json
import math
from pathlib import Path
import shutil


def program_name(operations):
    return ' → '.join(name + (f' × {count}' if count > 1 else '')
        for name, values in groupby(operations) for count in [len(list(values))]) or 'leaf → STOP'


def measurement_text(result, audit):
    groups = result['summary']['groups']
    all_bindings = groups['all']
    row_groups = {key: value for key, value in groups.items() if key.startswith('rows=')}
    rows = {key: sum(value[key] for value in row_groups.values())
            for key in ('n', 'resolved', 'correct')}
    lines = [
        '## What does the native balanced identity run establish?', '',
        f"The run completes **{result['training_documents']} training documents** and "
        f"**{result['evaluation_documents']} held-out documents**, with "
        f"**{result['summary']['errors']} native errors**, in "
        f"**{result['elapsed_seconds']:.2f} s**. Evaluation freezes online learning; "
        'episodic first mentions may still allocate individuals. The reversed-recency '
        'evaluation uses this same balanced training branch.', '',
        f"**Bindings:** {all_bindings['resolved']}/{all_bindings['n']} resolved, "
        f"{all_bindings['correct']}/{all_bindings['n']} correct. "
        'The accepted run resolved 0/146 and got 0/146 correct. '
        'The full conditional scores and candidate coverage remain in '
        '[the results](diagnostics/identity-results.json).', '',
        '| Binding group | Cases | Resolved | Correct |',
        '|---|---:|---:|---:|',
    ]
    for prefix in ('all', 'family=', 'candidates=', 'recency=', 'role=', 'first_mention=', 'cell='):
        for key, value in groups.items():
            if key == prefix or (prefix != 'all' and key.startswith(prefix)):
                lines.append(f"| {key.replace('_', ' ')} | {value['n']} | {value['resolved']} | {value['correct']} |")
    if not all_bindings['resolved']:
        lines += ['', 'Conditional binding accuracy is undefined because no case resolves. '
                  'The unresolved neither-candidate control does not establish chance binding.']
    lines += ['', '## Do row pairs resolve after the document closes?', '',
              f"**{rows['resolved']}/{rows['n']} resolved; {rows['correct']}/{rows['n']} correct.** "
              'The accepted run resolved 0/20 and got 0/20 correct. '
              'The observer reads occurrence references after all sentences of the document, '
              'including revisions by later closes.', '',
              '| Row-pair group / expected rows | Cases | Resolved | Correct |',
              '|---|---:|---:|---:|']
    for key, value in row_groups.items():
        lines.append(f"| {key.removeprefix('rows=').replace('_', ' ')} | {value['n']} | {value['resolved']} | {value['correct']} |")
    lines += ['', '## Which columns were minted, with which witnesses?', '',
              f"The warm-up starts with **zero A columns**. Training ends with "
              f"**{audit['training_columns']}**; evaluation ends with **{audit['final_columns']}**. "
              f"There are **{audit['minted']} new columns** and **{audit['witnesses']} final column-witness links**. "
              'The accepted run had zero columns and witnesses. '
              'All column records and their training snapshot are in '
              '[the results](diagnostics/identity-results.json).', '',
              '| Column | Witness row addresses after evaluation |', '|---|---|']
    lines += [f"| `{column['column']}` | `{json.dumps(column['witnesses'])}` |"
              for column in result['columns']] or ['| None | None |']
    lines += ['', '## Which programs are kept?', '',
              '| Phase | Kept program | Clauses | Nonzero recorded closed meaning |',
              '|---|---|---:|---:|']
    for phase in ('identity-training', 'identity-evaluation'):
        data = audit['phases'][phase]
        for program in data['programs']:
            lines.append(f"| {phase.removeprefix('identity-')} | {program['program']} | "
                         f"{program['count']} | {program['nonzero_recorded_meaning']} |")
    lines += ['']
    for phase in ('identity-training', 'identity-evaluation'):
        data = audit['phases'][phase]
        lines.append(f"{phase.removeprefix('identity-').capitalize()}: "
                     f"**{data['nonzero_recorded_meaning']}/{data['clauses']}** clauses have "
                     f"nonzero recorded closed meaning; **{data['requests']}** reference requests in kept readings.")
    lines += ['', 'The preceding balanced identity run had 43/384 nonzero training clauses and 0/556 '
              'nonzero held-out clauses, with zero requests. These are the existing observer’s '
              'recorded meaning-band norms; all 940 recorded norms in this run are finite. '
              'Full programs and per-step norms remain in '
              '[the kept-program file](diagnostics/identity-kept-programs.json); '
              'the kept readings’ requests are in [the closing records](diagnostics/identity-reference-readings.json). '
              'This audit does not count requests in unkept exploratory trials.', '',
              'Annotated reference provenance: ' + '; '.join(
                  f"**{count}** `{reason}`" for reason, count in sorted(audit['annotated_references'].items())) + '.', '',
              '[Full observations](diagnostics/identity-observations.json) preserve every reason and '
              'final reference. First-mention role and position remain confounded in the revision-3 '
              'SVO introductions; the refresh mentions cross role and recency.', '']
    return '\n'.join(lines)


def summarize(folder, destination):
    result = json.loads((folder/'results.json').read_text())
    assert result['completed'] and not result['smoke']
    assert (result['training_documents'], result['evaluation_documents']) == (136, 166)
    programs = json.loads((folder/'kept-programs.json').read_text())
    assert len(programs) == 940
    assert all(math.isfinite(row['closed_meaning_norm']) for row in programs)
    observations = json.loads((folder/'observations.json').read_text())
    readings = json.loads((folder/'reference-readings.json').read_text())
    phases = {}
    for phase in sorted({row['phase'] for row in programs}):
        rows = [row for row in programs if row['phase'] == phase]
        counts = Counter(tuple(row['program']) for row in rows)
        phases[phase] = dict(clauses=len(rows),
            nonzero_recorded_meaning=sum(row['closed_meaning_norm'] > 0 for row in rows),
            requests=sum(len(row['requests']) for row in readings if row['phase'] == phase),
            programs=[dict(program=program_name(list(program)), count=count,
                nonzero_recorded_meaning=sum(row['closed_meaning_norm'] > 0
                    for row in rows if tuple(row['program']) == program))
                      for program, count in counts.most_common()])
    reasons = Counter(token.get('missing_reason') or 'resolved'
        for row in observations for token in row['provenance'].values())
    audit = dict(completed=result['completed'], seed=result['seed'],
        training_documents=result['training_documents'],
        evaluation_documents=result['evaluation_documents'],
        training_columns=len(result['training_columns']), final_columns=len(result['columns']),
        minted=len(result['columns_minted']),
        witnesses=sum(len(column['witnesses']) for column in result['columns']),
        elapsed_seconds=result['elapsed_seconds'], recorded_meaning_norms_finite=len(programs),
        annotated_references=dict(reasons),
        phases=phases, groups=result['summary']['groups'])
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('results.json', 'observations.json', 'kept-programs.json', 'reference-readings.json'):
        shutil.copyfile(folder/name, destination/('identity-'+name))
    (destination/'identity-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    receipt = destination.parent/'README.md'
    placeholder = '<!-- Insert the completed native measurement here before committing. -->'
    body = receipt.read_text()
    assert placeholder in body, 'The receipt already contains measured tables; review before replacing them.'
    receipt.write_text(body.replace(placeholder, measurement_text(result, audit)))
    print(json.dumps({key: value for key, value in audit.items() if key not in ('phases', 'groups')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'diagnostics')
    args = parser.parse_args()
    summarize(args.run, args.out)
