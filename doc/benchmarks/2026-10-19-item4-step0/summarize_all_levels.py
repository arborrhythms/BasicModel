"""Tabulate completed arms without discarding failed certificates."""
from collections import Counter
import json
from pathlib import Path

root = Path(__file__).resolve().parent
groups = {
    'Grounded XOR': lambda n: 'test_grounded_xor.py' in n,
    'Native XOR curriculum': lambda n: 'test_concept_output.py' in n,
    'Exact CLI XOR': lambda n: 'TestXorExactCliReconstruction' in n,
    'MM_xor / MM_grammar convergence': lambda n: 'TestMMXorConvergence' in n,
    'XOR_grammar class accuracy': lambda n: 'TestXorGrammarLearnsXor' in n,
    'Grammatical reconstruction': lambda n: 'TestXorGrammarReconstruction' in n,
    'Math-chain certificates': lambda n: 'test_math_chain' in n,
}
summary = dict(arms={}, constructor_comparisons=[], coverage={}, joint_coverage=[])
for arm in ('zero', 'enabled'):
    result = json.loads((root/f'all-levels-{arm}/result.json').read_text())
    reports = {r['nodeid']:r for w in result['workers'] for r in w.get('reports', [])}
    summary['arms'][arm] = dict(selected=len(result['selected']),
        completed=len(result['completed']), outcome=result['reason'],
        groups={group:dict(Counter(r['outcome'] for n,r in reports.items() if match(n)))
                for group,match in groups.items()},
        failures=[dict(nodeid=n,message=r['message']) for n,r in reports.items() if r['outcome']=='failed'])
    coverage = summary['coverage'][arm] = {}
    for path in (root/'all-levels-metrics'/arm).glob('*.json'):
        for case, data in json.loads(path.read_text()).items():
            if case == 'harness-smoke':
                continue
            if arm == 'enabled':
                summary['constructor_comparisons'].extend(dict(case=case, **c) for c in data['constructors'])
            for level, read in data['reads'].items():
                total = coverage.setdefault(level, dict(calls=0,with_prior=0,coordinates=0,
                    below_one=0,minimum=1.,maximum=0.))
                for name in ('calls','with_prior','coordinates','below_one'):
                    total[name] += read[name]
                total['minimum'] = min(total['minimum'],read['minimum'])
                total['maximum'] = max(total['maximum'],read['maximum'])
            if arm == 'enabled' and all(data['reads'].get(level,{}).get('minimum',1.) < 1.
                                      for level in ('P','S','R','D')):
                summary['joint_coverage'].append(dict(case=case,
                    minimum_gains={level:data['reads'][level]['minimum'] for level in ('P','S','R','D')}))
(root/'all-levels-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for group in groups:
    print(group,*(summary['arms'][arm]['groups'][group] for arm in ('zero','enabled')))
print('Exact initial-parameter matches:',sum(c.get('initial_parameters_equal',False)
      for c in summary['constructor_comparisons']),'/',len(summary['constructor_comparisons']))
print('Coverage:',summary['coverage']['enabled'])
