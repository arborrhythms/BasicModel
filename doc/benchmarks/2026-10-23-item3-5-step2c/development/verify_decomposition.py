"""Verify original snapshot rows and paired RNG capsules, then pack losslessly."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import torch
from pack_capture import pack

ROOT = Path(__file__).resolve().parents[4]
target = (ROOT/sys.argv[1]).resolve()
source = json.loads((target/'frozen-source.json').read_text())['source']
verified, pairs = [], {}
for arm in ('zero','enabled'):
    for metrics in sorted((target/'bind-decomposition-metrics'/arm).glob('*.json')):
        for nodeid, record in json.loads(metrics.read_text()).items():
            comparison, = record['constructors']
            original = ROOT/comparison['original_snapshot']
            sha = hashlib.sha256(original.read_bytes()).hexdigest()
            assert sha == comparison['snapshot_sha256']
            baseline = torch.load(original,weights_only=True)
            capsule = target/f'decomposition-{arm}-{original.parent.name}-{original.stem}-initial.pt'
            if capsule.exists():
                captured_sha = hashlib.sha256(capsule.read_bytes()).hexdigest()
                captured = torch.load(capsule, weights_only=False)
            else:
                raw = gzip.decompress(capsule.with_suffix('.pt.gz').read_bytes())
                captured_sha = hashlib.sha256(raw).hexdigest()
                import io
                captured = torch.load(io.BytesIO(raw),weights_only=False)
            for name,value in captured['parameters'].items():
                expected = baseline[comparison['name_mapping'][name]]
                if name in comparison['catalogue_rows']:
                    value = value[comparison['catalogue_rows'][name]['rows']]
                assert torch.equal(value,expected), name
            pairs.setdefault(nodeid,{})[arm] = captured
            verified.append(dict(arm=arm,nodeid=nodeid,seed=record['seed'],
                captured_state=capsule.name,captured_sha256=captured_sha,**comparison))
            if capsule.exists():
                pack(capsule,target/'packed-captures.json')
assert len(verified)==4
for nodeid,arms in pairs.items():
    left,right=arms['zero'],arms['enabled']
    assert left['parameters'].keys()==right['parameters'].keys()
    assert all(torch.equal(v,right['parameters'][k]) for k,v in left['parameters'].items())
    assert torch.equal(left['torch'],right['torch']) and left['python']==right['python']
    a,b=left['numpy'],right['numpy']
    assert a[0]==b[0] and (a[1]==b[1]).all() and a[2:]==b[2:]
(target/'decomposition-snapshots.json').write_text(json.dumps(dict(source=source,
    verified=True,arms=verified,paired_rng_and_parameters_equal=True),indent=2)+'\n')
print('Verified four original snapshot mappings and equal paired RNG/parameters.')
