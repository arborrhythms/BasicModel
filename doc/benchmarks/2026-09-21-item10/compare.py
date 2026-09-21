"""Report parity as measured, including a null; never choose a passing seed."""
import json
from pathlib import Path
import sys
import torch

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
packed = json.loads((root / 'packed.json').read_text())
single = json.loads((root / 'single.json').read_text())
a, b = packed['parity'], single['parity']
report = {
    'seed': packed['seed'], 'config_sha256': packed['config_sha256'],
    'source_equal': packed['source'] == single['source'],
    'parameters_equal': a['initial_parameters_sha256'] == b['initial_parameters_sha256'],
    'dictionary_equal': a['initial_dictionary_sha256'] == b['initial_dictionary_sha256'],
    'boundary_convention': a['boundary_convention'],
    'atol': 1e-6, 'rtol': 1e-5, 'sentences': [],
    'packed_mean_byte_cost': a['mean_sentence_byte_cost'],
    'single_mean_byte_cost': b['mean_sentence_byte_cost'],
}
for x, y in zip(a['sentences'], b['sentences']):
    record = {'source_row': x['source_row'], 'text': x['text'],
              'packed_units': x['word_count'], 'single_units': y['word_count'],
              'packed_byte_cost': x['byte_cost'], 'single_byte_cost': y['byte_cost']}
    for key in ('root', 'reference', 'recovered'):
        u, v = torch.tensor(x[key]), torch.tensor(y[key])
        same_shape = u.shape == v.shape
        record[key] = {'same_shape': same_shape,
                       'max_abs_difference': float((u-v).abs().max()) if same_shape else None,
                       'close': same_shape and torch.allclose(u, v, atol=report['atol'], rtol=report['rtol'])}
    record['byte_cost_close'] = abs(x['byte_cost'] - y['byte_cost']) <= report['atol'] + report['rtol'] * abs(y['byte_cost'])
    report['sentences'].append(record)
report['parity_demonstrated'] = all(r['byte_cost_close'] and all(r[k]['close'] for k in ('root', 'reference', 'recovered'))
                                  for r in report['sentences'])
(root / 'parity-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
