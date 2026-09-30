"""Compare the fixed protocol before and after unindexed-operand closing."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEFORE, AFTER = HERE/'measurements', HERE/'closing-measurements'


def numeric(value):
    if isinstance(value, dict):
        return {k: numeric(v) for k, v in value.items()
                if k not in ('seconds', 'sentences_per_second')}
    if isinstance(value, list):
        return [numeric(v) for v in value]
    return value


processes = json.loads((AFTER/'processes.json').read_text())
assert set(processes) == {'serial-baseline', 'packed', 'single'}
assert all(p['exit_code'] == 0 for p in processes.values())
result = {}
for name in processes:
    before = json.loads((BEFORE/(name+'.json')).read_text())
    after = json.loads((AFTER/(name+'.json')).read_text())
    assert before['seed'] == after['seed'] == 42
    assert before['config_sha256'] == after['config_sha256']
    result[name] = dict(
        all_numeric_batch_results_exact=numeric(before['phases']) == numeric(after['phases']),
        complete_parity_report_exact=before.get('parity') == after.get('parity'),
        reconstruction=[dict(phase=p['name'], mean=p['reconstruction_mean'])
                        for p in after['phases']],
        mean_sentence_byte_cost=after.get('parity', {}).get('mean_sentence_byte_cost'))
result['protocol'] = 'The reviewed fixed seed and budgets are unchanged. Only timing and source metadata are excluded from the numerical comparison.'
result['source_match'] = 'review-source-equivalence.json'
(AFTER/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
