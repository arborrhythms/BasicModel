"""Run the unchanged explicit gates before the final item-7 review sweep."""
import gzip
import json
import os
from pathlib import Path
import sys
import time
from collections import defaultdict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import GIB, run_suite, source_snapshot, documentation_snapshot
from review_source import supporting_inputs


def main():
    source = source_snapshot(ROOT)
    inputs = supporting_inputs(ROOT)
    prior = HERE.parent / '2026-09-27-item7/explicit-verified/result.json'
    if prior.exists():
        previous = json.loads(prior.read_text())
    else:
        previous = json.loads(gzip.decompress(prior.with_suffix('.json.gz').read_bytes()))
    selectors = previous['selected'] + ['test/test_thinking_kernel.py']
    os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager',
                      BASIC_AUTOLOAD='false', RUN_SLOW='1', PYTHONPATH=str(ROOT / 'bin'))
    output = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'explicit')
    output.mkdir(exist_ok=False)
    (output / 'source-manifest.json').write_text(json.dumps(dict(
        validated_source=source, supporting_inputs=inputs,
        recorded_documentation=documentation_snapshot(ROOT)),indent=2)+'\n')
    grouped = defaultdict(list)
    for selector in selectors:
        grouped[selector.split('::',1)[0]].append(selector)
    result = dict(reason='running',exit_code=125,selected=[],completed=[],workers=[],groups=[],
                  rationale='Each gate file has its own bounded process group so a resource failure remains visible without suppressing the other gates.')
    started = time.monotonic()
    for index, (filename, requested) in enumerate(grouped.items()):
        remaining = 5400 - (time.monotonic() - started)
        if remaining <= 0:
            result.update(reason='suite_timeout',exit_code=124)
            (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            return 124
        group = run_suite(root=ROOT, selectors=requested,
            run_dir=output / f'group-{index:02d}', memory_bytes=8 * GIB,
            workers=1, worker_memory_bytes=8 * GIB, timeout=min(1800,remaining),
            suite_timeout=min(2000,remaining), batch_size=32, max_files=1)
        assert source_snapshot(ROOT) == source, 'source changed during explicit gates'
        assert supporting_inputs(ROOT) == inputs, 'fixture inputs changed during explicit gates'
        result['groups'].append(dict(file=filename,reason=group['reason'],exit_code=group['exit_code'],
            receipt=f'group-{index:02d}/result.json'))
        for key in ('selected','completed','workers'):
            result[key].extend(group[key])
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    result['exit_code'] = int(any(group['exit_code'] for group in result['groups']))
    result['reason'] = 'gate_failures' if result['exit_code'] else 'passed'
    result['resource_limited_cases'] = sorted(set(result['selected']) - set(result['completed']))
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
