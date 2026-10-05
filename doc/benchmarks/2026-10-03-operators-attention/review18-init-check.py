"""Observe the sole initialization change; no seeds or training."""
import os
os.environ['BASICMODEL_DEVICE']='cpu'
import sys,json,hashlib,subprocess,zipfile,ast,difflib
from pathlib import Path
h=Path(__file__).resolve().parent; root=h.parents[2]
sys.path[:0]=[str(root/'test'),str(root/'bin')]
import bounded_tests as bounded
before=json.loads((h/'review18-before/source.json').read_text()); after=bounded.source_snapshot(root)
changed=sorted(name for name in before.keys()|after.keys() if before.get(name)!=after.get(name))
assert changed==['bin/Layers.py'], changed
old=(h/'review18-before/Layers.py').read_text(); new=(root/'bin/Layers.py').read_text()
def init_node(text):
    module=ast.parse(text)
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='BytesFallbackEncoder')
    return module, next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
m0,n0=init_node(old); m1,n1=init_node(new)
body0=ast.get_source_segment(old,n0); body1=ast.get_source_segment(new,n1)
n0.body=n1.body
assert ast.dump(m0,include_attributes=False)==ast.dump(m1,include_attributes=False)
checks=dict(changed_runtime_files=changed,changed_function='BytesFallbackEncoder.__init__',old=body0,new=body1,tests_unchanged=True,seeds_bars_guards_xmls_unchanged=True)
import torch
from Layers import BytesFallbackEncoder
state=torch.random.get_rng_state(); replay=torch.Generator(); replay.set_state(state)
raw=torch.randn(256,6,generator=replay)
enc=BytesFallbackEncoder(6)
expected=(raw/raw.abs().amax(-1,keepdim=True).clamp(min=1e-8)).clamp(0.,1.)
assert torch.equal(enc.byte_codebook.detach(),expected)
assert torch.equal(torch.random.get_rng_state(),replay.get_state())
assert enc.byte_codebook.requires_grad and list(enc.parameters())==[enc.byte_codebook]
checks['initialization_check']=dict(exact_rule=True,same_single_random_draw=True,parameter_registration_unchanged=True,minimum=float(expected.min()),maximum=float(expected.max()),finite=bool(torch.isfinite(expected).all()))
(h/'review18-change-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
(h/'review18-only-change.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='review17/bin/Layers.py',tofile='review18/bin/Layers.py')))
out=h/'review18-source'; out.mkdir(exist_ok=False)
(out/'source.json').write_text(json.dumps(after,indent=2)+'\n')
extra=set(subprocess.check_output(['git','ls-files','-z','test'],cwd=root).decode().split('\0'))-{''}
names=sorted(set(after)|extra)
with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in names:z.write(root/name,name)
(out/'complete-source.json').write_text(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},indent=2)+'\n')
print(json.dumps({'changed':changed,'files':len(after),'initialization':checks['initialization_check']}))
