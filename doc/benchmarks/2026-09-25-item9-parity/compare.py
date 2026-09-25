"""Compare every declared sentence; no retry, seed selection or rounded gate."""
import json
from pathlib import Path
import sys
import torch

root = Path(sys.argv[1])
p, s = [json.loads((root / f'{mode}.json').read_text()) for mode in ('packed', 'single')]
a, b = p['parity'], s['parity']
report = dict(atol=1e-6, rtol=1e-5, seed=p['seed'],
              source_equal=p['source'] == s['source'], config_equal=p['config_sha256'] == s['config_sha256'],
              parameters_equal=a['initial_parameters_sha256'] == b['initial_parameters_sha256'],
              dictionary_equal=a['initial_dictionary_sha256'] == b['initial_dictionary_sha256'],
              packed_mean=a['mean_sentence_byte_cost'], single_mean=b['mean_sentence_byte_cost'], sentences=[])
for x, y in zip(a['sentences'], b['sentences'], strict=True):
    assert x['source_row'] == y['source_row']
    record = dict(source_row=x['source_row'], text=x['text'])
    for name in ('root', 'program_root', 'reference', 'recovered', 'byte_cost'):
        u, v = torch.tensor(x[name]), torch.tensor(y[name])
        record[name] = dict(same_shape=u.shape == v.shape,
                            max_abs_difference=float((u-v).abs().max()) if u.shape == v.shape else None,
                            close=u.shape == v.shape and torch.allclose(u, v, atol=report['atol'], rtol=report['rtol']))
    record['program_owns_seal'] = all(torch.equal(torch.tensor(z['root']), torch.tensor(z['program_root'])) for z in (x, y))
    u,v = x['reverse_inputs'],y['reverse_inputs']
    record['actions_equal'] = u['actions'] == v['actions']
    record['effective_candidates_equal'] = u['effective_candidate_rows'] == v['effective_candidate_rows']
    record['differing_staged_inputs'] = [name for name in u['inputs'] if u['inputs'][name] != v['inputs'][name]]
    report['sentences'].append(record)
report['parity_demonstrated'] = (all(report[k] for k in ('source_equal','config_equal','parameters_equal','dictionary_equal'))
    and all(r['program_owns_seal'] and r['actions_equal'] and r['effective_candidates_equal']
            and all(r[k]['close'] for k in ('root','program_root','reference','recovered','byte_cost')) for r in report['sentences']))
(root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
