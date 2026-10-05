"""Preserve complete §14 ports and verify the protected measurement contract."""
import ast, hashlib, json, subprocess, sys, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def seeds(text):
 return sorted(ast.get_source_segment(text,n) for n in ast.walk(ast.parse(text))
               if isinstance(n,ast.Call) and 'seed' in ast.unparse(n.func).lower())
def asserts(text):
 return [ast.dump(n) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Assert)]
archive=zipfile.ZipFile(HERE/'review14-before/source.zip')
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
  if name.endswith('.py') and asserts(old.decode())!=asserts(new.decode()):assert_changes.append(name)
expected=['test/test_echoic_decoder.py','test/test_free_reconstruction.py','test/test_review13_subspaces.py']
report=dict(head=git('rev-parse','HEAD').decode().strip(),protected=checks,changed=changed,
 seed_changes=seed_changes,assertion_ports=assert_changes,
 declared_assertion_ports=dict(zip(expected,[
  'Delete exactly the two retired antipode tests (§14.6). Other assertions remain.',
  'Detached derived bank and byte-only registry replace the explicitly retired antipode contract.',
  'Net-evidence lattice values and detached native ownership supersede §13 live signed folds.'])),
 complete_changed_test_ports=ports,capacity_change_this_round=False,
 retained_capacity='Shared dictionary/DEF references need an address for every native percept-concept: 6+256 and 8+256.',
 xml_changes='latticeMargin=0; unitRootRead=false default, true in XOR_grammar only; schema declarations.',
 focused_files=(HERE/'review14-focused-files.txt').read_text().splitlines(),
 plan_sha256=hashlib.sha256((ROOT/'doc/plans/2026-09-27-item-6-8-one-attention.md').read_bytes()).hexdigest())
(HERE/sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
assert all(checks.values()) and not seed_changes
assert set(assert_changes)==set(expected),assert_changes
print(json.dumps({k:v for k,v in report.items() if k!='complete_changed_test_ports'}))
