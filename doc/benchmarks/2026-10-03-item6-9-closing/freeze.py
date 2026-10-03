"""Freeze this closing round and retain complete before/after port bodies."""
import ast
import difflib
import json
from pathlib import Path
import sys
import zipfile
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'test'))
from bounded_tests import source_snapshot, write_json
snapshot = source_snapshot(ROOT)
write_json(HERE/'source-final.json', snapshot)
with zipfile.ZipFile(HERE/'source-final.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for name in snapshot:
        z.write(ROOT/name, name)
ports=[]
patch=[]
with zipfile.ZipFile(HERE/'before.zip') as z:
    before=json.loads((HERE/'before.json').read_text())
    assert all(snapshot[name] == sha for name,sha in before.items() if name.startswith('data/'))
    for name in sorted(set(before)|set(snapshot)):
        old=z.read(name).decode() if name in z.namelist() else ''
        new=(ROOT/name).read_text() if name in snapshot else ''
        if old==new: continue
        patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='before/'+name,tofile='after/'+name))
    mapping=[('test/test_free_reconstruction.py', 'test_free_trial_uses_no_reference_or_offsets_and_only_free_term',
              'test_free_trial_uses_no_reference_or_offsets_and_owns_antipode_term',
              'Observe the decoder result tuple and the added antipode term; retain all no-reference/offset and byte-equality assertions, add actual decoder ownership.'),
             ('test/test_trial_policy_ownership.py', 'test_trial_generation_leaves_policy_credit_at_batch_end',
              'test_trial_and_batch_answer_leave_the_decoder_to_reconstruction',
              'Four parameterized cases: output policy credit is retired. Assert the actual owned backward leaves every decoder gradient absent in trial and batch phases, while answer gradients are nonzero and the understanding remains cut.')]
    def body(source, name):
        node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
        start=min([node.lineno]+[d.lineno for d in node.decorator_list])
        return '\n'.join(source.splitlines()[start-1:node.end_lineno])+'\n'
    for path,oldname,newname,reason in mapping:
        old=z.read(path).decode();new=(ROOT/path).read_text()
        ports.append(dict(path=path,old_name=oldname,new_name=newname,reason=reason,old_body=body(old,oldname),new_body=body(new,newname)))
    path='test/objective_conflicts_probe.py'
    old=z.read(path).decode();new=(ROOT/path).read_text()
    for name in ('before_optimizer_step','derivation','trial','groups','committed'):
        ports.append(dict(path=path,old_name=name,new_name=name,reason='Observation-only audit follows journal state separately from the derivation-free reader record and records the shared decoder.',old_body=body(old,name),new_body=body(new,name)))
write_json(HERE/'ports.json',ports)
(HERE/'closing.patch').write_text(''.join(patch))
print(json.dumps(dict(files=len(snapshot),ports=len(ports),configurations_unchanged=True)))
