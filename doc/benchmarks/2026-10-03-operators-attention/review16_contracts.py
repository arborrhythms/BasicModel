"""Preserve the complete §16.3 test ports and verify seeds, bars, and guards."""
import ast, hashlib, json, subprocess, sys, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def seeds(text):
 return sorted(ast.get_source_segment(text,n) for n in ast.walk(ast.parse(text))
               if isinstance(n,ast.Call) and 'seed' in ast.unparse(n.func).lower())
def assertions(text):return [ast.dump(n) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Assert)]
archive=zipfile.ZipFile(HERE/'review16-before/source.zip')
protected=['test/test_explicit_dimensions.py','test/test_mm_xor.py','test/test_reconstruction_roundtrip.py',
 'test/bounded_tests.py','test/pytest_worker.py','Makefile','pytest.ini','data/eval/nanochat_grammar_gate.json']
checks={p:(ROOT/p).read_bytes()==git('show','HEAD:'+p) for p in protected}
changed=[];ports=[];seed_changes=[];assert_changes=[]
for name in archive.namelist():
 if not name.startswith(('bin/','test/','data/')):continue
 old=archive.read(name);new=(ROOT/name).read_bytes()
 if old==new:continue
 changed.append(name)
 if name.endswith('.py') and seeds(old.decode())!=seeds(new.decode()):seed_changes.append(name)
 if name.startswith('test/'):
  ports.append(dict(path=name,old=old.decode(),new=new.decode()))
  if name.endswith('.py') and assertions(old.decode())!=assertions(new.decode()):assert_changes.append(name)
output_files=['test_prepared_answer_boundary','test_trial_policy_ownership','test_generation_catalog',
 'test_output_path_supervised','test_arithmetic_isolation']
output_checks={name:assertions(archive.read('test/'+name+'.py').decode())==assertions((ROOT/'test'/f'{name}.py').read_text()) for name in output_files}
protected_docs=json.loads((HERE/'review16-protected-docs.json').read_text())
doc_checks={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in protected_docs.items()}
external_docs=json.loads((HERE/'review16-external-docs.json').read_text())
external_checks={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==row['observed_sha256']
                 for name,row in external_docs.items()}
report=dict(head=git('rev-parse','HEAD').decode().strip(),protected=checks,protected_docs=doc_checks,
 externally_changed_docs_preserved=external_checks,
 changed=changed,seed_changes=seed_changes,assertion_ports=assert_changes,
 output_assertions_unchanged=output_checks,complete_changed_test_ports=ports,
 capacity=dict(XOR_grammar=6,MM_grammar=8),
 focused_files=(HERE/'review16-focused-files.txt').read_text().splitlines())
path=HERE/sys.argv[1]
with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
assert all(checks.values()) and not seed_changes and all(output_checks.values())
assert all(unchanged or external_checks.get(name,False) for name,unchanged in doc_checks.items())
print(json.dumps({k:v for k,v in report.items() if k!='complete_changed_test_ports'}))
