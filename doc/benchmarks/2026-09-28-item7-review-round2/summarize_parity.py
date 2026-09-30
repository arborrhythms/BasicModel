"""Compare the unchanged numerical packed/single protocol without a tolerance."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'parity'


def flat(value):
    if isinstance(value, list):
        return [v for part in value for v in flat(part)]
    return [value]


processes = json.loads((OUTPUT / 'processes.json').read_text())
assert set(processes) == {'packed', 'single'}
records = {name: json.loads((OUTPUT / (name + '.json')).read_text())
           for name in processes if (OUTPUT / (name + '.json')).exists()}
report = dict(processes=processes, protocol='Prior declared seed 42, vocabulary warmup, 4 validation sentences, no optimizer; no seed selection or tolerance.',
              complete=all(p['exit_code'] == 0 for p in processes.values()))
if report['complete']:
    packed, single = records['packed']['parity'], records['single']['parity']
    report['initial_parameters_equal'] = packed['initial_parameters_sha256'] == single['initial_parameters_sha256']
    report['initial_dictionary_equal'] = packed['initial_dictionary_sha256'] == single['initial_dictionary_sha256']
    report['means'] = {name: r['parity']['mean_sentence_byte_cost'] for name,r in records.items()}
    rows = []
    for a,b in zip(packed['sentences'], single['sentences'], strict=True):
        assert a['source_row'] == b['source_row'] and a['text'] == b['text']
        differences = {}
        for key in ('root', 'reference', 'recovered', 'byte_cost'):
            aa,bb = flat(a[key]),flat(b[key])
            differences[key] = dict(exact=a[key] == b[key],
                max_absolute_difference=max((abs(x-y) for x,y in zip(aa,bb,strict=True)), default=0.))
        rows.append(dict(source_row=a['source_row'], text=a['text'],
            packed_cost=a['byte_cost'], single_cost=b['byte_cost'],
            dictionary_equal=a['dictionary_sha256'] == b['dictionary_sha256'], differences=differences))
    report['sentences'] = rows
    report['all_sentence_values_exact'] = all(v['exact'] for r in rows for v in r['differences'].values())
    report['truncated'] = {name: [s['truncated'] for s in r['parity']['steps']] for name,r in records.items()}
(OUTPUT / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
