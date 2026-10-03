"""Report every declared measurement without selecting successful runs."""
from collections import Counter
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
out = HERE / 'measurements'
groups, all_rows = {}, []


def read(path):
    return json.loads(path.read_text())


def margin_text(value):
    if value is None or not value['exact']:
        return 'no exact affine solution'
    return f"max weight {value['max_weight']:.9g}; L2 {value['l2_weight']:.9g}"


lines = ['# Ten unseeded runs of each isolated variant', '',
    'Each run uses the unchanged XOR_grammar fixture at 400 epochs. The bar is',
    'all four classes correct and MSE below .05. These patches are measurements;',
    'none is applied to the candidate files. All results are retained.', '',
    'Answers and derivations below follow: hello world, hello there, loving world,',
    'loving there. `L0`, `L1`, `L2` are first word, separator, second word.',
    'Weights are Claude’s pseudoinverse margins, excluding bias. A nonexact',
    'affine solution is explicitly marked, even when its coefficient norm is small.', '',
    'The final margin is for the four selected roots together: one answer map',
    'must satisfy all four. The best-available search applies each enumerated',
    'derivation to all four sentences, using the captured leaves at that epoch.', '',
    '| Variant | Passes | Four correct | Median MSE | MSE range |',
    '|---|---:|---:|---:|---|']
details = []
for variant in ('a', 'b', 'c'):
    rows = []
    details += ['', f'## Variant {variant}', '']
    for trial in range(1, 11):
        name = f'{variant}-{trial:02}'
        result = read(out / name / 'measurement.json')
        process = read(out / name / 'process.json')
        assert process['exit_code'] == 0
        assert result['variant'] == variant and result['training_pairs'] == 400
        assert result['test_inputs'] == ['hello world', 'hello there', 'loving world', 'loving there']
        rows.append(dict(trial=trial, **result, process=process))
        all_rows.append(dict(name=name, **result, process=process))
        answers = ', '.join(f'{v:.9g}' for v in result['answers'])
        details += [f'### Run {trial}', '',
            f"[Full record](measurements/{name}/measurement.json): answers **[{answers}]**; "
            f"MSE **{result['mse']:.10g}**; **{result['correct']}/4** correct; "
            f"bar **{'PASS' if result['settled_bar'] else 'FAIL'}**.", '', '```text']
        details += [f"{sentence}: {' ; '.join(expressions)}"
                    for sentence, expressions in result['final_derivations'].items()]
        details += ['```', '', 'Final derivation margin: **' + margin_text(result['final_margin']) + '**.', '']
        for key, title in (('best_first_epoch', 'First epoch before updates'),
                           ('best_last_epoch', 'Last epoch before updates'),
                           ('best_after_training', 'Final evaluation after training')):
            value = result[key]
            epoch = '' if 'epoch' not in value else f" (epoch {value['epoch']})"
            details.append(f"- {title}{epoch}: `{value['derivation']}`; {margin_text(value['margin'])}; "
                           f"{value['readable_at_5']}/{value['expressions']} enumerated derivations readable at weight ≤5.")
        details += ['', f"[All trial choices](measurements/{name}/trials.jsonl) · "
                         f"[Equal-parameter comparisons](measurements/{name}/pairs.jsonl) · "
                         f"[Process and guard](measurements/{name}/process.json)", '']
    errors = [row['mse'] for row in rows]
    groups[variant] = dict(runs=len(rows), passes=sum(row['settled_bar'] for row in rows),
        correct_four=sum(row['correct'] == 4 for row in rows), median_mse=median(errors),
        minimum_mse=min(errors), maximum_mse=max(errors),
        final_readable_at_5=sum(row['final_margin']['exact'] and row['final_margin']['max_weight'] <= 5 for row in rows),
        median_best_first_weight=median(row['best_first_epoch']['margin']['max_weight'] for row in rows),
        median_best_last_weight=median(row['best_last_epoch']['margin']['max_weight'] for row in rows),
        median_seconds=median(row['process']['elapsed_seconds'] for row in rows), rows=rows)
    summary = groups[variant]
    lines.append(f"| {variant} | {summary['passes']}/10 | {summary['correct_four']}/10 | "
                 f"{summary['median_mse']:.9g} | {min(errors):.9g}–{max(errors):.9g} |")
(HERE / 'variant-results.md').write_text('\n'.join(lines + details) + '\n')

mm = [dict(trial=i, **read(out/f'mm-{i:02}'/'measurement.json'),
           process=read(out/f'mm-{i:02}'/'process.json')) for i in range(1, 11)]
assert all(r['completed_epochs'] == 900 and r['process']['exit_code'] == 0 for r in mm)
mm_median = median(r['ending_training_mse'] for r in mm)
lines = ['# Final candidate MM_grammar receipt', '',
    'Ten fresh unseeded runs, 900 updates each. The repaired candidate is unchanged',
    'and no variant is installed in these processes. The accepted item-7 baseline',
    'median ending MSE is .1066. These are independent initializations.',
    'The comparison is descriptive; no seed or paired initialization is used.',
    '`MODEL_COMPILE=eager` matches the established MM receipt environment.',
    'Three incomplete attempts dispatched with `none` are retained in the',
    '[environment incident](mm-environment-failure/explanation.json); none reached 900 updates.', '',
    f'Candidate median ending training MSE: **{mm_median:.10g}**.', '',
    '| Run | Ending training MSE | After-update evaluation MSE | Seconds |',
    '|---|---:|---:|---:|']
for row in mm:
    lines.append(f"| [{row['trial']}](measurements/mm-{row['trial']:02}/measurement.json) | "
        f"{row['ending_training_mse']:.12g} | {row['after_900_updates_mse']:.12g} | "
        f"{row['elapsed_seconds']:.2f} |")
(HERE / 'final-mm-table.md').write_text('\n'.join(lines)+'\n')

summary = dict(variants=groups, mm=dict(median_ending_mse=mm_median,
    accepted_baseline_median=.1066, rows=mm),
    complete_measurements=40, incomplete_observer_attempts=3,
    incomplete_mm_environment_attempts=3,
    implementation_stays_at_step=5, variants_applied_to_candidate=False,
    decision='Stop for Alec and Claude review; part 4 has not started.')
(HERE / 'measurement-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(variants={key:{k:v for k,v in value.items() if k!='rows'}
                              for key,value in groups.items()}, mm_median=mm_median), indent=2))
