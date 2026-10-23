"""Exact assertion/fixture ledger and source hashes for the storage-only round."""
import ast, difflib, hashlib, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
BASE='bb0045d7a2b7a7e5be584066ca1ee283fa899d75'
copies={
 'test/observer_probe.py':'doc/benchmarks/2026-10-07-operators-final/observer_probe.py',
 'test/operators_run_audit.py':'doc/benchmarks/2026-10-07-operators-final/operators_run_audit.py'}
renames={
 'test_only_the_inverted_index_is_owned_or_checkpointed':'test_inverted_index_is_derived_after_checkpoint_restore',
 'test_idea_coarsens_parent_and_relation_cascades':'test_idea_coarsens_parent_and_releases_its_child_next_pass',
 'test_closing_excludes_its_own_witness_without_restricting_copular_reads':'test_closing_and_reads_keep_a_name_pending_without_a_resolver_choice',
 'test_forward_name_accepts_a_witnessed_occurrence_complement':'test_forward_name_stays_pending_beside_a_witnessed_occurrence_complement'}

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
paths=set(git('diff','--name-only',BASE,'--','bin','test','data').decode().splitlines())
paths.update(git('ls-files','--others','--exclude-standard','--','bin','test','data').decode().splitlines())
ledger,functions,retired,new_tests,new_helpers=[],[],[],[],[]
for path in sorted(paths):
 new=(ROOT/path).read_text()
 original=copies.get(path,path)
 old=git('show',f'{BASE}:{original}').decode() if git('ls-tree',BASE,'--',original).strip() else ''
 if not path.endswith('.py') or not path.startswith('test/'):continue
 def scan(text):
  result={}
  def visit(nodes,scope=''):
   for node in nodes:
    if isinstance(node,ast.ClassDef):visit(node.body,scope+node.name+'.')
    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
     name=scope+node.name
     checks=[]
     for n in ast.walk(node):
      if isinstance(n,ast.Assert) or (isinstance(n,ast.Call) and (
        ast.unparse(n.func) in ('pytest.raises','pytest.warns') or
        (isinstance(n.func,ast.Attribute) and n.func.attr.startswith('assert')) or
        (isinstance(n.func,ast.Name) and n.func.id.startswith('assert_')))):
       checks.append((n.lineno,ast.get_source_segment(text,n)))
     checks.sort()
     result[name]=dict(start=node.lineno,text=ast.get_source_segment(text,node),checks=checks,
       decorators=[ast.get_source_segment(text,d) for d in node.decorator_list])
  visit(ast.parse(text).body)
  return result
 a,b=scan(old),scan(new)
 for name,value in a.items():
  other=renames.get(name,name)
  if other not in b:
   if name.startswith('test_'):retired.append(dict(file=path,test=name,body=value['text']))
   continue
  fresh=b[other]
  if value['text']==fresh['text'] and value['decorators']==fresh['decorators']:continue
  functions.append(dict(file=path,before=name,after=other,
    before_decorators=value['decorators'],after_decorators=fresh['decorators'],
    diff=''.join(difflib.unified_diff(value['text'].splitlines(True),fresh['text'].splitlines(True),fromfile=name,tofile=other))))
  left,right=[c[1] for c in value['checks']],[c[1] for c in fresh['checks']]
  for kind,i,j,k,l in difflib.SequenceMatcher(a=left,b=right,autojunk=False).get_opcodes():
   if kind!='equal':ledger.append(dict(file=path,function=other,change=kind,
    before=[dict(line=ln,text=t) for ln,t in value['checks'][i:j]],
    after=[dict(line=ln,text=t) for ln,t in fresh['checks'][k:l]]))
  if value['decorators']!=fresh['decorators']:
   ledger.append(dict(file=path,function=other,change='parameters',before=value['decorators'],after=fresh['decorators']))
 for name in b.keys()-{renames.get(n,n) for n in a}:
  if name.startswith('test_'):new_tests.append(dict(file=path,test=name,body=b[name]['text']))
  else:new_helpers.append(dict(file=path,function=name,body=b[name]['text'],
    checks=[dict(line=line,text=text) for line,text in b[name]['checks']]))
(HERE/'assertion-ledger.json').write_text(json.dumps(dict(base=BASE,changed_checks=ledger,
 copied_from=copies,renamed_tests=renames,retired_tests=retired,new_tests=new_tests,
 added_helpers=new_helpers),indent=2)+'\n')
(HERE/'fixture-ledger.json').write_text(json.dumps(functions,indent=2)+'\n')
(HERE/'source-diff.patch').write_bytes(git('diff','--binary',BASE,'--','bin','test','data'))
(HERE/'test-ports.patch').write_bytes(git('diff',BASE,'--','test'))
copied=[]
for path,original in copies.items():
 copied.extend(difflib.unified_diff(git('show',f'{BASE}:{original}').decode().splitlines(True),
     (ROOT/path).read_text().splitlines(True),fromfile=original,tofile=path))
(HERE/'copied-observer-ports.patch').write_text(''.join(copied))
lines=['# Exact assertion ports\n',f'Baseline: `{BASE}`. Line numbers refer to the audited source.\n',
 'Every changed function, including assertion-preserving setup and loop-scope changes, is in [fixture-ledger.json](fixture-ledger.json). '
 'Whole-file edits are in [test-ports.patch](test-ports.patch); copied observer edits are in [copied-observer-ports.patch](copied-observer-ports.patch).\n']
for i,entry in enumerate(ledger,1):
 lines += [f"\n## {i}. {entry['file']} — {entry['function']}\n",f"Change: {entry['change']}.\n"]
 for side in ('before','after'):
  values=entry[side]
  content='\n'.join(v if isinstance(v,str) else v['text'] for v in values) or '(none)'
  lines += [f'\n{side.capitalize()}:\n\n```python\n{content}\n```\n']
lines += ['\n## Retired tests\n']
for entry in retired:lines += [f"\n- `{entry['file']}::{entry['test']}` — adjacency/nudge only (P3).\n"]
lines += ['\n## New regressions\n']
for entry in new_tests:lines += [f"\n- `{entry['file']}::{entry['test']}`\n"]
lines += ['\n## Added fixture helpers, including their new assertions\n']
for entry in new_helpers:
 lines += [f"\n### {entry['file']} — {entry['function']}\n\n```python\n{entry['body']}\n```\n"]
(HERE/'assertion-ports.md').write_text(''.join(lines))
audit=dict(base=BASE,branch=git('branch','--show-current').decode().strip(),
 published_2a='0b030e3ebdeca87f9ebf80739237437c3c75ea8f',
 files={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(paths)},
 retired=[dict(file=r['file'],test=r['test']) for r in retired],
 counts=dict(changed_checks=len(ledger),changed_functions=len(functions),retired_tests=len(retired),new_tests=len(new_tests),added_helpers=len(new_helpers)))
(HERE/'source-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit['counts']))
