"""Freeze each recorded port's first old body and its current source body."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
records={}
for filename in ('test-ports.json','reader-test-ports.json','index-test-ports.json','closing-test-ports.json'):
    for item in json.loads((HERE/filename).read_text()):
        key=(item['file'],item['name'])
        if key not in records:
            records[key]=dict(file=item['file'],name=item['name'],before=item['before'],reasons=[],history=[])
        record=records[key]
        if item['reason'] not in record['reasons']:record['reasons'].append(item['reason'])
        record['history'].append(dict(receipt=filename,after=item['after']))
renames = {
    'test_word_symbol_defines_order0_native_features_and_object_stays_unwritten': 'test_interpreted_object_keeps_the_native_features_in_one_row',
    'test_explicit_word_triple_capacity_failure_is_atomic': 'test_explicit_word_row_capacity_failure_is_atomic',
    'test_objects_never_acquire_witnessed_kinds': 'test_interpretation_replaces_the_word_row_without_inventing_a_kind',
    'test_interpret_reuses_provisional_testimony_and_reverses_by_identity': 'test_interpret_reuses_definition_and_reverses_by_identity',
    'test_one_meta_generalizes_over_several_words_and_objects': 'test_definition_rows_generalize_over_several_words_and_objects',
    'test_meta_binding_does_not_discriminate_at_binding_time': 'test_definition_binding_preserves_ambiguous_objects_until_selection',
}
for (file,name),record in records.items():
    path=ROOT/file; source=path.read_text(); tree=ast.parse(source)
    found=[node for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.name==renames.get(name,name)]
    assert len(found)<=1,(file,name,len(found))
    record['after']=ast.get_source_segment(source,found[0]) if found else None
    if not found: assert record['history'][-1]['after'] is None,(file,name)
    if name in renames: assert found,(file,name)
    record['current_name']=found[0].name if found else None
    record['status']='renamed' if found and found[0].name!=name else ('ported' if found else 'retired')
    record['source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    record['changed_after_last_script']=record['after'] != record['history'][-1]['after']
    del record['history']
result=dict(protocol=__doc__,ports=list(records.values()))
(HERE/'final-test-ports.json').write_text(json.dumps(result,indent=2)+'\n')
print(len(records),'ports;',sum(r['changed_after_last_script'] for r in records.values()),'with final manual reconciliation')
