"""Freeze delivered source and measurement dependencies after a green sweep."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as bounded


def main():
    out=HERE/'measured-source';out.mkdir(exist_ok=True)
    source=bounded.source_snapshot(ROOT)
    result=json.loads((HERE/'final-sweep/result.json').read_text())
    validated=json.loads((HERE/'final-sweep/source-manifest.json').read_text())['validated_source']
    assert result['exit_code']==0 and result['reason']=='passed'
    assert sorted(result['completed'])==sorted(result['selected']) and source==validated
    (out/'source.json').write_text(json.dumps(source,indent=2)+'\n')
    with zipfile.ZipFile(out/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in source:z.write(ROOT/name,name)
    dependencies=set(HERE.glob('*.py')) | {HERE/'protocol.json',HERE/'test-retirements.json',ROOT/'test/fixtures/when-readers-round4a0.json'}
    # Historical observer imports are preserved verbatim; capturing the whole
    # helper directory also preserves their indirect imports without guesswork.
    dependencies.update((HERE.parent/'2026-10-03-operators-attention').glob('*.py'))
    dependencies.update(HERE.parent/'2026-10-01-item6-9-review'/name for name in ('separator_campaign.py','measure.py'))
    dependencies.add(HERE.parent/'2026-09-24-item11c/explicit-result.json.gz')
    helpers={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(dependencies)}
    (out/'measurement-helpers.json').write_text(json.dumps(helpers,indent=2)+'\n')
    with zipfile.ZipFile(out/'measurement-helpers.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in helpers:z.write(ROOT/name,name)
    (out/'tracked-changes.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT))
    protected=('test/test_explicit_dimensions.py','test/test_mm_xor.py','data/XOR_grammar.xml','data/MM_xor.xml')
    equal={name:(ROOT/name).read_bytes()==subprocess.check_output(['git','show','631d9e8e44b7c8034263e22b36dc74fa5df4eb75:'+name],cwd=ROOT) for name in protected}
    assert all(equal.values())
    assert json.loads((HERE/'baseline-source.json').read_text()) == json.loads((HERE.parent/'2026-10-07-item6-2/measured-source/source.json').read_text())
    prior=HERE.parent/'2026-10-07-item6-2'
    assert all(hashlib.sha256((prior/name).read_bytes()).hexdigest()==sha for name,sha in json.loads((HERE/'first-receipt-hashes.json').read_text()).items())
    summary=dict(source_files=len(source),source_sha256=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
        helpers=len(helpers),protected_matches=equal,base_commit='631d9e8e44b7c8034263e22b36dc74fa5df4eb75')
    (out/'freeze.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))

if __name__=='__main__':main()
