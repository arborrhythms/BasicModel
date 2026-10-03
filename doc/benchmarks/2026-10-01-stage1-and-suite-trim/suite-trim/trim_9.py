"""Use explicit eager execution where graph capture is not the test subject."""
import ast,json,re
from port_ledger import HERE,ROOT,definitions

baseline=json.loads((HERE/'prior-slowest-25.json').read_text())
selected={}
for row in baseline:
    file,name=row['nodeid'].split('::')
    file=file.replace('test_item9b_schedule','test_interleave_schedule')
    name=name.split('[')[0]
    if name in ('test_real_packed_ends_train_before_the_next_sentence',
                'test_real_aligned_loop_matches_prior_compiled_semantics_across_chunks'):
        continue
    selected.setdefault(file,set()).add(name)
# These files concentrate most of the remaining call time. Their subjects are
# prepared-answer ownership and supervised gradients, never graph compilation.
for file in ('test/test_prepared_answer_boundary.py','test/test_output_path_supervised.py'):
    source=(ROOT/file).read_text()
    for n in ast.parse(source).body:
        if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'):
            selected.setdefault(file,set()).add(n.name)

before={file:(ROOT/file).read_text() for file in selected}
before['test/test_compiled_word_chunk.py']=(ROOT/'test/test_compiled_word_chunk.py').read_text()
before['test/test_sentence_compose.py']=(ROOT/'test/test_sentence_compose.py').read_text()
# Save the accepted slow/timeout probes and source before any fixture repair.
(HERE/'slow-fixture-before.json').write_text(json.dumps({'probes':baseline,'files':before},indent=2)+'\n')
for file,names in selected.items():
    source=before[file]
    defs=definitions(source)
    for name in names:
        old=defs[name]
        new=re.sub(r'^(def\s+\w+\([^)]*)(\)\s*:)',
                   lambda m:m[1]+(', ' if m[1].rstrip()[-1]!='(' else '')+'eager_reading'+m[2],
                   old,count=1,flags=re.S|re.M)
        assert new!=old,(file,name)
        source=source.replace(old,new,1)
    (ROOT/file).write_text(source)

def change_body(file,name,replacements):
    path=ROOT/file;source=path.read_text();old=definitions(source)[name];new=old
    for prior,replacement in replacements:
        assert prior in new,(file,name,prior)
        new=new.replace(prior,replacement)
    source=source.replace(old,new,1);ast.parse(source);path.write_text(source)

# Minimum K=2-crossing example: three content words, one shorter row, W8.
for name in ('test_real_aligned_loop_matches_prior_compiled_semantics_across_chunks',
             'test_no_grad_fallback_retains_eager_stm_depth_semantics'):
    change_body('test/test_compiled_word_chunk.py',name,[
      ('_tiny_canonical_model(tmp_path, monkeypatch)',
       '_tiny_canonical_model(tmp_path, monkeypatch, input_width=16, word_buckets="8")'),
      ('["alpha beta gamma", "delta"]','["a b c", "d"]')])
change_body('test/test_sentence_compose.py','test_real_packed_ends_train_before_the_next_sentence',[
    ('concept_rows=256, part_rows=128,','concept_rows=128, part_rows=64, input_width=16,')])
change_body('test/test_interleave_schedule.py','test_native_interleave_supplies_context_then_reads_the_same_sentences',[
    ("concept_rows=128, input_width=64", "concept_rows=128, input_width=16"),
    ("texts = ('the wug sat', 'the wug flew')", "texts = ('a b', 'a c')")])
change_body('test/test_negative_expectation.py','test_native_unlabelled_batch_trains_the_same_chooser',[
    ('word_buckets="8", batch_size=1,','word_buckets="8", batch_size=1, input_width=16,'),
    ('("a bicycle has a wheel", "a wheel is round", "a bicycle is large")','("a b", "a c", "b c")')])
change_body('test/test_grammar_word_learning.py','test_normal_text_reconstruction_updates_the_grammar_chooser',[
    ('word_buckets="8,16"','word_buckets="8", input_width=16'),
    ('["a bicycle has a wheel", "a wheel belongs to a bicycle"]','["a b c", "d e f"]')])

rows=json.loads((HERE/'ports-and-retirements.json').read_text());byid={r['old_id']:r for r in rows}
changes=[]
for file,old in before.items():
    new=(ROOT/file).read_text();olddefs=definitions(old);newdefs=definitions(new)
    ast.parse(new)
    for name,oldbody in olddefs.items():
        if not name.startswith('test_') or newdefs[name]==oldbody:continue
        newbody=newdefs[name]
        assertions=lambda s:[ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Assert)]
        assert assertions(oldbody)==assertions(newbody),(file,name)
        ident=file+'::'+name
        payload=dict(id=ident,body=newbody)
        if ident in byid:
            row=byid[ident];row.setdefault('intermediate_ports',[]).append(dict(id=ident,body=oldbody));row['new']=[payload]
        else:
            rows.append(dict(old_id=ident,old_body=oldbody,new=[payload],
                reason='Item 9 performance port: plain eager loop for behavior tests; shortest sentences/capacities retaining the tested subject. All assertions unchanged; compiler cases still compile.',
                evidence=['slow-fixture-before.json','slow-fixture-changes.json']))
        changes.append(dict(id=ident,old=oldbody,new=newbody,assertions_unchanged=True))
(HERE/'ports-and-retirements.json').write_text(json.dumps(rows,indent=2)+'\n')
(HERE/'slow-fixture-changes.json').write_text(json.dumps(changes,indent=2)+'\n')
print('Ported',len(changes),'test definitions; unchanged assertions.')
