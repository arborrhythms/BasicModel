"""Freeze only after the full development certificate and declaration."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE),str(ROOT/'test')]
from bounded_tests import source_snapshot


def main():
    from math_train import declaration
    protocol = declaration()
    source = source_snapshot(ROOT)
    protected = ('test/test_explicit_dimensions.py','test/test_mm_xor.py',
                 'data/XOR_grammar.xml','data/MM_xor.xml')
    unchanged = {name:(ROOT/name).read_bytes()==subprocess.check_output(
        ['git','show','e43638a:'+name],cwd=ROOT) for name in protected}
    assert all(unchanged.values())
    original = json.loads((HERE/'frozen-contracts.json').read_text())
    # The verifier's cost correction is expressly required by §14.7.2.
    # Original hashes stay intact; record the authorized exception separately.
    contracts = original.get('files',original)
    # The checked original manifest is a direct path/hash mapping.
    assert isinstance(contracts,dict)
    for name,digest in contracts.items():
        if name=='bin/BindingAnswers.py':
            continue
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    output = HERE/'measured-source'
    output.mkdir(exist_ok=False)
    (output/'source.json').write_text(json.dumps(source,indent=2)+'\n')
    with zipfile.ZipFile(output/'source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in source:
            archive.write(ROOT/name,name)
    dependencies = {HERE/name for name in ('math_train.py','math_campaign.py','campaign.py',
        'thinking_gate.py','verification.py','freeze.py','protocol.json','frozen-contracts.json',
        'measurement_observer.py','development-acceptance-14-8.json',
        'binding-answers-frozen.diff','binding-answers-verifier.json','binding-answers-matches.py.txt')}
    dependencies.update(HERE/protocol['development_certificate_folder']/name
        for name in ('result.json','batches.jsonl'))
    dependencies.add(HERE.parent/'2026-10-07-math-chain/protocol.json')
    for directory in ('2026-10-07-math-chain','2026-10-08-math-chain-repair',
                      '2026-10-03-operators-attention'):
        dependencies.update((HERE.parent/directory).glob('*.py'))
    dependencies.add(HERE.parent/'2026-10-08-math-chain-repair/thinking-gate-plan.json')
    dependencies.update(HERE.parent/'2026-10-01-item6-9-review'/name
                        for name in ('separator_campaign.py','measure.py'))
    helpers = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(dependencies)}
    (output/'measurement-helpers.json').write_text(json.dumps(helpers,indent=2)+'\n')
    with zipfile.ZipFile(output/'measurement-helpers.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in helpers:
            archive.write(ROOT/name,name)
    (output/'tracked-changes.patch').write_bytes(subprocess.check_output(
        ['git','diff','--binary','HEAD'],cwd=ROOT))
    report = dict(source_files=len(source),helpers=len(helpers),protected_matches=unchanged,
        source_sha256=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
        base_commit='e43638a747e373b3b10343c642f76cd1d0757ec1',
        authorized_contract_change='BindingAnswers.cost, thinking §14.7.2; other contracts unchanged.')
    (output/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
