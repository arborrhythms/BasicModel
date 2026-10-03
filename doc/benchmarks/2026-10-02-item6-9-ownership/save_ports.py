"""Archive complete before/after test files and enumerate changed test bodies."""
import ast,hashlib,json,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
old=zipfile.ZipFile(HERE/'ownership-start.zip')
reasons={}
for p in HERE.glob('retired-tests-*.json'):
 for row in json.loads(p.read_text()):reasons[(row['file'],row['test'])]=row['reason']
def tests(text):
 result={}
 tree=ast.parse(text)
 def visit(body,parents=()):
  for node in body:
   if isinstance(node,ast.ClassDef):visit(node.body,(*parents,node.name))
   elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('test_'):
    start=min([node.lineno]+[x.lineno for x in node.decorator_list])-1
    result['::'.join((*parents,node.name))]='\n'.join(text.splitlines()[start:node.end_lineno])+'\n'
 visit(tree.body);return result
renames={
 ('test/test_gradient_factorization.py','test_normal_batch_logs_named_shared_operator_gradients'):'test_normal_batch_logs_owned_optimizer_gradients',
 ('test/test_supplied_answer_training.py','test_answer_projection_cost_does_not_include_a_generate_lesson'):'test_answer_cost_does_not_include_a_generate_lesson',
 ('test/test_supplied_answer_training.py','test_real_supplied_trials_cost_before_updates_and_reach_all_three_owners'):'test_real_supplied_trials_cost_before_updates_and_reach_only_reader',
 ('test/test_supplied_answer_training.py','test_native_trial_reads_the_completed_point_with_live_credit'):'test_native_trial_reads_the_completed_point_without_state_credit',
 ('test/test_word_grain_attention.py','test_off_without_word_store_is_plain_reverse'):'test_attention_off_is_plain_reverse',
}
manifest=[];unexplained=[]
paths=set(n for n in old.namelist() if n.startswith('test/') and n.endswith('.py'))
paths|={str(p.relative_to(ROOT)) for p in (ROOT/'test').rglob('*.py')}
with zipfile.ZipFile(HERE/'test-ports.zip','w',zipfile.ZIP_DEFLATED) as archive:
 for name in sorted(paths):
  before=old.read(name).decode() if name in old.namelist() else ''
  path=ROOT/name;after=path.read_text() if path.exists() else ''
  if before==after:continue
  archive.writestr('old/'+name,before);archive.writestr('new/'+name,after)
  bt,at=tests(before),tests(after)
  for test,body in bt.items():
   renamed=renames.get((name,test),test)
   if renamed in at:
    if body!=at[renamed]:manifest.append(dict(file=name,test=test,new_test=renamed,disposition='port',old_body=body,new_body=at[renamed]))
   else:
    reason=reasons.get((name,test)) or reasons.get((name,test.split('::')[-1]))
    if reason is None:unexplained.append((name,test))
    manifest.append(dict(file=name,test=test,disposition='retired',reason=reason,old_body=body,new_body=None))
  for test,body in at.items():
   if test not in bt and test not in {v for (f,k),v in renames.items() if f==name}:manifest.append(dict(file=name,test=test,disposition='added',old_body=None,new_body=body))
(HERE/'test-port-bodies.json').write_text(json.dumps(manifest,indent=2)+'\n')
(HERE/'test-dispositions.json').write_text(json.dumps([{k:v for k,v in r.items() if not k.endswith('_body')} for r in manifest],indent=2)+'\n')
print(json.dumps(dict(ported=sum(r['disposition']=='port' for r in manifest),retired=sum(r['disposition']=='retired' for r in manifest),added=sum(r['disposition']=='added' for r in manifest),missing_reasons=unexplained)))
