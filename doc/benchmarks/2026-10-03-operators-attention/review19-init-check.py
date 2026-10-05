"""Check the sole admitted-row initialization change without seeds or training."""
import os
os.environ['BASICMODEL_DEVICE']='cpu'
import ast,difflib,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
sys.path[:0]=[str(ROOT/'test'),str(ROOT/'bin')]
import bounded_tests as bounded
before=json.loads((H/'review19-before/source.json').read_text());after=bounded.source_snapshot(ROOT)
changed=sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k))
assert changed==['bin/Layers.py'],changed
old=(H/'review19-before/Layers.py').read_text();new=(ROOT/'bin/Layers.py').read_text()
def function(text):
    tree=ast.parse(text);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RadixLayer')
    return tree,next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='insert')
a,fn0=function(old);b,fn1=function(new)
oldbody=ast.get_source_segment(old,fn0);newbody=ast.get_source_segment(new,fn1)
fn0.body=fn1.body
assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
import torch
from Layers import RadixLayer
checks=[]
for width in (6,8):
    store=RadixLayer(dim=width,initial_cap=8)
    master=store._basis.W;identity=id(master);pointer=master.data_ptr();oldrows=master.detach().clone()
    replay=torch.Generator();replay.set_state(torch.random.get_rng_state())
    raw=torch.empty(width).normal_(mean=0.,std=1.,generator=replay)
    expected=(raw/raw.abs().amax().clamp(min=1e-8)).clamp(0.,1.)
    row=store.insert(b'x')
    assert row==0 and torch.equal(master[row].detach(),expected)
    assert torch.equal(torch.random.get_rng_state(),replay.get_state())
    assert torch.equal(master[1:].detach(),oldrows[1:])
    assert id(store._basis.W)==identity and store._basis.W.data_ptr()==pointer
    assert master.requires_grad
    state=torch.random.get_rng_state();saved=master.detach().clone()
    assert store.insert(b'x')==row
    assert torch.equal(torch.random.get_rng_state(),state) and torch.equal(master.detach(),saved)
    explicit=torch.linspace(-2.,2.,width)
    row1=store.insert(b'y',init_vector=explicit)
    assert torch.equal(master[row1].detach(),explicit) and torch.equal(torch.random.get_rng_state(),state)
    checks.append(dict(width=width,exact_rule=True,same_single_random_draw=True,other_rows_unchanged=True,
        parameter_identity_and_ownership_unchanged=True,duplicate_is_noop=True,explicit_initializer_unchanged=True,
        initialized_minimum=float(expected.min()),initialized_maximum=float(expected.max())))
result=dict(changed_runtime_files=changed,changed_function='RadixLayer.insert',old=oldbody,new=newbody,
    tests_unchanged=True,seeds_bars_guards_xmls_unchanged=True,checks=checks)
(H/'review19-change-verification.json').write_text(json.dumps(result,indent=2)+'\n')
(H/'review19-only-change.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='review18/bin/Layers.py',tofile='review19/bin/Layers.py')))
out=H/'review19-source';out.mkdir(exist_ok=False)
(out/'source.json').write_text(json.dumps(after,indent=2)+'\n')
extra=set(subprocess.check_output(['git','ls-files','-z','test'],cwd=ROOT).decode().split('\0'))-{''}
names=sorted(set(after)|extra)
with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in names:z.write(ROOT/name,name)
(out/'complete-source.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},indent=2)+'\n')
print(json.dumps({'files':len(after),'changed_function':result['changed_function'],'checks':checks}))
