"""Freeze the exact source and existing-test assertion ledger for this pass."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE = 'b38889eb787cc4a0d5b27b675e908a670257ea1a'
fixtures = ['doc/benchmarks/2026-10-20-item3-5-stm1/decomposition_certificate.py',
            'doc/benchmarks/2026-10-20-item3-5-stm1/forced_decomposition_grammar.py']
def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)
paths = [p for directory in ('bin','test','data') for p in (ROOT/directory).rglob('*')
         if p.is_file() and p.suffix in ('.py','.grammar','.xml','.xsd','.json')]
paths += [ROOT/name for name in fixtures]
old = json.loads((HERE.parent/'source-09/source-audit.json').read_text())
paths += [ROOT/name for name in old['files'] if (ROOT/name).exists()]
paths += list(HERE.glob('*.py')) + [HERE/'selections.json', HERE/'original-complete.grammar']
paths += list((HERE.parents[1]/'development').glob('*.py'))
paths += list((HERE.parent/'source-09/diagnosis').glob('*.py'))
files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
changed = git('diff',BASE,'--name-only','--','bin','test','data',*fixtures).splitlines()
(HERE/'source-audit.json').write_text(json.dumps(dict(base=BASE, changed=changed, files=files),indent=2)+'\n')
(HERE/'production.patch').write_text(git('diff',BASE,'--','bin','data'))
(HERE/'test-ports.patch').write_text(git('diff',BASE,'--','test',*fixtures))
records=[]
def functions(text):
    return {f.name:f for f in ast.parse(text).body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))}
def assertions(f):
    return [ast.unparse(n) for n in ast.walk(f) if isinstance(n,ast.Assert)]
for name in changed:
    if not name.startswith('test/') or not name.endswith('.py'): continue
    prior=subprocess.run(['git','show',f'{BASE}:{name}'],cwd=ROOT,capture_output=True,text=True)
    before=functions(prior.stdout) if prior.returncode==0 else {}
    after=functions((ROOT/name).read_text())
    for f in sorted(before.keys() | after.keys()):
        a=assertions(before[f]) if f in before else None
        b=assertions(after[f]) if f in after else None
        if a!=b: records.append(dict(file=name,function=f,before=a,after=b))
(HERE/'assertion-audit.json').write_text(json.dumps(dict(base=BASE,
    note='Exact Python assert statements; test-ports.patch also owns tensor/raises checks and fixture edits.',
    changes=records),indent=2)+'\n')
print('Audited',len(files),'files;',len(changed),'changed paths;',len(records),'assertion-bearing function changes.')
