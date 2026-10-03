"""Consolidate pure retirement guards; preserve all behavioral negative checks."""
import ast,json,shutil
from collections import defaultdict
from port_ledger import ROOT,HERE,definitions,record,body
candidates=json.loads((HERE/'retired-name-guard-candidates.json').read_text())
keep={
 'test_bounded_workers_ignore_iteration_xdist_request','test_compose_has_no_auxiliary_local_policy_objective',
 'test_serial_relaxes_symbol_dim_passthrough','test_property_read_does_not_save_event_property_primitive_product',
 'test_wholespace_taxonomy_payload_is_dropped_instead_of_quarantined','test_optimizer_excludes_emb_params_when_frozen',
 'test_allows_monotonic_without_cs_codebook','test_reconstruction_bank_validation_has_no_mps_host_sync',
 'test_ws_cache_not_present_after_forward','test_cs_cache_not_present_after_forward',
 'test_runBatch_reverse_path_does_not_depend_on_cs_cache',
 'test_mlx_compile_target_does_not_set_training_compile_env','test_no_endpoint_bracket_api',
 # Already moved inline with their parked classes under item 2.
 'test_layer_classes_parked_in_legacy','test_copy_layer_parked_in_legacy','test_introspection_layers_parked_in_legacy',
}
extra={'test_grammar_has_no_marker_helper_rules','test_no_prototype_table','test_no_legacy_chart_helper_names',
 'test_reconstruct_enum_retired','test_lift_layer_no_raw_gate','test_tie_api_is_retired',
 'test_build_symbol_leg_is_retired','test_cfg_loader_removed'}
candidates.extend(r for r in json.loads((HERE/'additional-guard-review.json').read_text()) if r['id'].split('::')[-1] in extra)
by_file=defaultdict(list);dispositions=[]
for r in candidates:
 name=r['id'].split('::')[-1]
 if name in keep:
  dispositions.append(dict(id=r['id'],action='kept',reason='Checks executed behavior, or moved beside its parked class in item 2.'))
  continue
 filename,qual=r['id'].split('::',1)
 if filename=='test/test_ps_reverse_e2e.py':
  dispositions.append(dict(id=r['id'],action='consolidated',target='test/test_retired_names.py::test_retired_names_remain_absent'))
  continue
 by_file[filename].append(qual)
for filename,names in by_file.items():
 p=ROOT/filename;s=p.read_text();tree=ast.parse(s);old=definitions(s)
 spans=[]
 def walk(nodes,prefix=''):
  for node in nodes:
   if isinstance(node,ast.ClassDef):
    tests=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
    if tests and all(prefix+node.name+'::'+n.name in names for n in tests):
     start=min([node.lineno]+[x.lineno for x in node.decorator_list]);spans.append((start-1,node.end_lineno))
    else:walk(node.body,prefix+node.name+'::')
   elif isinstance(node,ast.FunctionDef) and prefix+node.name in names:
    start=min([node.lineno]+[x.lineno for x in node.decorator_list]);spans.append((start-1,node.end_lineno))
 walk(tree.body)
 lines=s.splitlines(keepends=True)
 for start,end in sorted(spans,reverse=True):del lines[start:end]
 new=''.join(lines);ast.parse(new);p.write_text(new)
 for name in names:
  record(filename,name,old[name],[('test/test_retired_names.py','test_retired_names_remain_absent')],
   'Same retired attribute/source/signature/configuration/grammar checks in one named table. Instance checks remain on constructed instances; same fixture configurations retained; behavioral tests stay separate.',
   ['retired-name-guard-candidates.json','additional-guard-review.json','retired-names-final.py.txt'])
  dispositions.append(dict(id=filename+'::'+name,action='consolidated',target='test/test_retired_names.py::test_retired_names_remain_absent'))
# The earlier precursor guard was transferred unchanged before this consolidation.
p=HERE/'ports-and-retirements.json';rows=json.loads(p.read_text())
newbody=definitions((ROOT/'test/test_retired_names.py').read_text())['test_retired_names_remain_absent']
for r in rows:
 for item in r['new']:
  if item['id']=='test/test_retired_names.py::test_grammar_has_no_and_token':
   r['intermediate_body']=item['body'];item.update(id='test/test_retired_names.py::test_retired_names_remain_absent',body=newbody)
p.write_text(json.dumps(rows,indent=2)+'\n')
shutil.copyfile(ROOT/'test/test_retired_names.py',HERE/'retired-names-final.py.txt')
(HERE/'retired-guard-dispositions.json').write_text(json.dumps(dispositions,indent=2)+'\n')
print('Consolidated',sum(r['action']=='consolidated' for r in dispositions),'definitions; kept',sum(r['action']=='kept' for r in dispositions),'behavioral/parked cases.')
