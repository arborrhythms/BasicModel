"""Compare the unchanged native protocol, including its guarded failures."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def number(value):
    return 'unavailable' if value is None else f'{value:.9f}'


rows = []
for label in ('native-before', 'native-prepair', 'native-after'):
    directory = HERE / label
    process = read(directory / 'driver.process.json')
    measurement = directory / 'measurement.json'
    timing = directory / 'batch-timing.json'
    phases = {p['name']: p for p in read(measurement)['phases']} if measurement.exists() else {}
    summaries = read(timing)['summaries'] if timing.exists() else {}
    warmed = summaries.get('warmed_training', {}).get('mean_seconds', {})
    rows.append(dict(label=label, process=process, phases=phases, warmed=warmed))

lines = ['# Native sentence-pair timing and reconstruction', '',
    'These are fresh unseeded runs of the same current `data/MM_ladder.xml` protocol: '
    'four validation batches, two training warm-ups, five measured training batches, '
    'and four final validation batches, with two sentences per batch. '
    'They are not paired initializations. Compilation/setup affects total run time; '
    'the timing breakdown below uses the five warm training batches. '
    'Any later graph-capture work stays in those batches; no timing outlier is dropped.', '',
    '| Source stage | Result | Before MSE | Training MSE | After MSE | Warm sentences/s | Peak worker GiB | Total seconds |',
    '|---|---|---|---|---|---|---|---|']
for row in rows:
    label, process, phases = row['label'], row['process'], row['phases']
    result = 'completed' if process['exit_code'] == 0 else process['reason']
    mses = [phases.get(p, {}).get('reconstruction_mean')
            for p in ('before_training', 'training', 'after_training')]
    throughput = phases.get('training', {}).get('sentences_per_second')
    lines.append(f"| [{label}]({label}/source-manifest.json) | [{result}]({label}/driver.process.json) | "
        + ' | '.join(number(v) for v in (*mses, throughput))
        + f" | {process['peak_memory_bytes']/2**30:.3f} | {process['elapsed_seconds']:.2f} |")
lines += ['', '| Mean seconds per warm batch | Before item | Before step 5 | After step 5 |',
          '|---|---|---|---|']
for name in ('exploit_compose', 'explore_compose', 'sentence_scoring', 'exploit_backward',
             'explore_backward', 'batch_backward', 'snapshot', 'restore', 'other', 'total'):
    lines.append(f"| {name} | " + ' | '.join(number(r['warmed'].get(name)) for r in rows) + ' |')
before, after = rows[1], rows[2]
if 'total' in before['warmed'] and 'total' in after['warmed']:
    time_ratio = after['warmed']['total'] / before['warmed']['total']
    memory_ratio = after['process']['peak_memory_bytes'] / before['process']['peak_memory_bytes']
    lines += ['', f'Measured after/before-step-5 ratios: warm batch time **{time_ratio:.4f}**, '
              f'whole-run peak worker memory **{memory_ratio:.4f}**. '
              'Peak memory includes model construction, graph capture, evaluation and training; '
              'it is not an isolated allocation count for the two trial graphs.']
lines += ['', 'The implementation retains both trial graphs through the existing shared saved-value '
    'hooks; it does not take a full model snapshot. Both costs precede both optimizer steps, '
    'and each backward restores its own perception pullback. Prefix-sharing remains future work.', '',
    'The historical packed/single fixture remains blocked by its retired '
    '`WholeSpace.propertyBasis` configuration element. The failed attempts are recorded separately; '
    'no configuration was changed to manufacture a new parity result.', '']
(HERE / 'native-comparison.md').write_text('\n'.join(lines))
