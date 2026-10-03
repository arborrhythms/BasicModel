"""Move review-round tests by behavior, preserving complete bodies and assertions."""
import ast
import json
import re
from collections import defaultdict
from port_ledger import HERE, ROOT, definitions, body

renames = {
 'item7_acceptance':'clause_acceptance', 'item7_byte_columns':'byte_reconstruction',
 'item7_clause_program':'clause_closing', 'item7_clause_state':'clause_scope',
 'item7_closing_identities':'sentence_identity', 'item7_context':'context_rotation',
 'item7_definition_integrity':'definition_integrity', 'item7_definitions':'definition_rows',
 'item7_end_state_storage':'sentence_end_state', 'item7_expectation':'expectation_kind',
 'item7_grammar':'clause_grammar', 'item7_lesson_gradients':'generation_lesson',
 'item7_nested_references':'relation_references', 'item7_predicate_identity':'predicate_identity',
 'item7_predicate_storage':'predicate_storage', 'item7_property_read_storage':'property_read',
 'item7_property_only':'property_configuration', 'item7_provenance':'assertion_provenance',
 'item7_reduction_deadline':'compose_deadline', 'item7_references':'sentence_references',
 'item7_retention':'sentence_retention', 'item7_row_schema':'identification_evidence',
 'item7_sentence_boundary':'sentence_boundary', 'item7_sentence_gradients':'sentence_gradients',
 'item7_storage':'clause_storage', 'item7_taxonomy':'definition_taxonomy',
 'item7_truth_text':'truth_text', 'item7_unfold_index':'reference_index',
 'item7_unindexed_relations':'relation_closing', 'item7_word_admission':'word_admission',
 'item7_word_interpretation':'word_interpretation', 'item9b_interpret':'word_interpretation',
 'item7_selected_property_rows':'property_read', 'item7_property_cache_release':'property_read',
 'item7_point_owners':'sentence_end_state', 'item9b_erosion':'discrimination_metric',
 'item9b_field_contract':'field_binding', 'item9b_schedule':'interleave_schedule',
 'item9b_occurrence_coordinates':'occurrence_coordinates',
 'item11c_attribution':'definition_attribution', 'item11c_membership_folds':'evidence_folds',
 'item11c_perception':'native_perception',
}
splits = {
 'item9b_capacity_contract': [
  ('admission_capacity', [0,1,2,3,4]), ('coordinate_registry',[5,6,7])],
 'item9b_corrections': [
  ('word_interpretation',[0,1]), ('interleave_lifecycle',[2]),
  ('coordinate_registry',[3,4,5,6]), ('occurrence_coordinates',[7,9,10,11]),
  ('byte_reconstruction',[8])],
 'item11c_review': [
  ('evidence_folds',[0]), ('definition_attribution',[1]), ('field_binding',[2]),
  ('field_learning',[3,4,5,6])],
 'item9b_followup': [
  ('interleave_lifecycle',[0,1,3,4,5,6,7,8,9,10,11]), ('discrimination_metric',[2])],
}

old_sources = {str(p.relative_to(ROOT)):p.read_text() for p in sorted((ROOT/'test').glob('test_item*.py'))
               if re.match(r'test_item(?:7|9b|11c)_', p.name)}
assert len(old_sources) == len(renames) + len(splits)
audit = {'source_files':old_sources, 'test_moves':[], 'deleted_duplicates':[],
         'reason':'Organize by executed behavior; no duplicate assertion deletions in item 8.'}
(HERE/'review-round-before.json').write_text(json.dumps(audit,indent=2)+'\n')
targets = defaultdict(list)
destinations = {}

def target_file(suffix):
    return 'test/test_' + suffix + '.py'

for old_file, source in old_sources.items():
    key = old_file.removeprefix('test/test_').removesuffix('.py')
    tree = ast.parse(source)
    tests = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
             and n.name.startswith(('test_', 'Test'))]
    if key in renames:
        dest = target_file(renames[key])
        targets[dest].append(source)
        for name in definitions(source):
            destinations[old_file+'::'+name] = dest+'::'+name
    else:
        shared = [n for n in tree.body if n not in tests and not
                  (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
        assert not any(isinstance(n,(ast.FunctionDef,ast.ClassDef)) for n in shared), old_file
        header = '\n'.join(body(source,n) for n in shared)
        selected=[]
        for dest_suffix, indices in splits[key]:
            dest=target_file(dest_suffix)
            nodes=[tests[i] for i in indices]
            selected.extend(indices)
            targets[dest].append(header+'\n\n'+'\n\n'.join(body(source,n) for n in nodes)+'\n')
            for node in nodes:
                destinations[old_file+'::'+node.name]=dest+'::'+node.name
        assert sorted(selected)==list(range(len(tests))),old_file

module_map={'test_'+old:'test_'+new for old,new in renames.items()}
# No imports point at split modules; the sole prose citation names its behavior.
module_map['test_item11c_review']='test_definition_attribution'
def rewrite(source):
    # Module names only, not archived item7_*.pt fixture names.
    for old,new in module_map.items():
        source=re.sub(r'\b'+re.escape(old)+r'\b',new,source)
    return source

new_sources={}
for dest,parts in targets.items():
    assert not (ROOT/dest).exists(),dest
    # Future imports must occur at the start; these source modules have none.
    assert not any('from __future__' in s for s in parts)
    new=rewrite('\n\n'.join(parts))
    nodes=ast.parse(new).body
    named=[n.name for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    assert len(named)==len(set(named)),(dest,named)
    new_sources[dest]=new

# Rewrite helper imports in other live tests, also keeping complete before/after files.
dependent={}
for p in (ROOT/'test').glob('*.py'):
    rel=str(p.relative_to(ROOT))
    if rel in old_sources:continue
    old=p.read_text();new=rewrite(old)
    if old!=new:
        ast.parse(new)
        dependent[rel]={'old':old,'new':new}

rows=json.loads((HERE/'ports-and-retirements.json').read_text())
by_id={r['old_id']:r for r in rows}
for old_file,source in old_sources.items():
    for name,old_body in definitions(source).items():
        if not name.split('::')[-1].startswith('test_'):continue
        old_id=old_file+'::'+name
        dest_id=destinations[old_id]
        dest_file,dest_name=dest_id.split('::',1)
        new_body=definitions(new_sources[dest_file])[dest_name]
        before_asserts=[ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(old_body)) if isinstance(n,ast.Assert)]
        after_asserts=[ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(new_body)) if isinstance(n,ast.Assert)]
        assert before_asserts==after_asserts,old_id
        audit['test_moves'].append({'old':old_id,'new':dest_id,'assertions_unchanged':True})
        if old_id in by_id:
            row=by_id[old_id]
            row.setdefault('intermediate_ports',[]).append({'id':old_id,'body':old_body})
            row['new']=[{'id':dest_id,'body':new_body}]
            row['reason']+=' Then relocated by behavior in suite-trim item 8.'
        else:
            row={'old_id':old_id,'old_body':old_body,'new':[{'id':dest_id,'body':new_body}],
                 'reason':'Move review-round case into the file for its behavior; same body and assertions except helper import module names.',
                 'evidence':['review-round-moves.json','review-round-before.json']}
            rows.append(row);by_id[old_id]=row
for rel,change in dependent.items():
    old_defs=definitions(change['old']);new_defs=definitions(change['new'])
    for name,old_body in old_defs.items():
        if name.split('::')[-1].startswith('test_') and new_defs[name]!=old_body:
            ident=rel+'::'+name
            if ident in by_id:
                row=by_id[ident]
                row.setdefault('intermediate_ports',[]).append({'id':ident,'body':old_body})
                row['new']=[{'id':ident,'body':new_defs[name]}]
            else:
                rows.append({'old_id':ident,'old_body':old_body,'new':[{'id':ident,'body':new_defs[name]}],
                  'reason':'Update import of relocated review-round helper; all assertions unchanged.',
                  'evidence':['review-round-dependent-imports.json']})
for row in rows:
    for new in row['new']:
        if new['id'] in destinations:
            new['id']=destinations[new['id']]
            file,name=new['id'].split('::',1)
            new['body']=definitions(new_sources[file])[name]

# All validation happens before publishing the move.
for path,source in new_sources.items():(ROOT/path).write_text(source)
for path,change in dependent.items():(ROOT/path).write_text(change['new'])
for path in old_sources:(ROOT/path).unlink()
(HERE/'ports-and-retirements.json').write_text(json.dumps(rows,indent=2)+'\n')
audit.pop('source_files')
audit.update(destination_files=list(new_sources),module_map=module_map)
(HERE/'review-round-moves.json').write_text(json.dumps(audit,indent=2)+'\n')
(HERE/'review-round-dependent-imports.json').write_text(json.dumps(dependent,indent=2)+'\n')
print(f'Moved {len(audit["test_moves"])} test definitions from {len(old_sources)} review files into {len(new_sources)} behavior files; no assertion deletions.')
