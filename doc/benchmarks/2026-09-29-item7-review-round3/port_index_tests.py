"""Final lexical fixture ports from separate word seats/caches to DEF rows."""
import ast,json
from pathlib import Path
here=Path('doc/benchmarks/2026-09-29-item7-review-round3')
files=['test/test_item9b_interpret.py','test/test_lexical_reference_orders.py','test/test_word_store.py','test/test_radix_layer_reverse.py','test/test_surface_grammar.py']
before={f:Path(f).read_text() for f in files}
def edit(f,a,b):
 p=Path(f);s=p.read_text();assert a in s,(f,a);p.write_text(s.replace(a,b))
edit(files[0],'''    witnessed = cs.new_concept()
    cs.add_part(witnessed, 8)
    cs._populate_concept_weights(witnessed)''','''    witnessed = interpret.forward(interpret.lookup_word([8], [], form='witnessed'))''')
edit(files[1],'''    ids = []
    for order in range(3):''','''    word, obj = cs.interpret_word([7], [], key='cat')
    ids = [obj]
    for order in range(1, 3):''')
edit(files[1],"        cs.bind_word_concept('cat', cid)","        cs.interpret.define(word, cid)\n        cs.bind_word_concept('cat', cid)")
edit(files[1],"    cs.remember_word_surface(row, b'cat', object_row=row, object_id=ids[0])\n",'')
edit(files[1],"assert cs.word_concepts('cat') == tuple(ids)","assert cs.word_concepts('cat') == tuple(sorted([cs.definitions.word(form='cat'), *ids]))")
edit(files[1],'''    saved = _model_with(cs, SimpleNamespace())._collect_structural_extras()
    restored = _cs(nS=256, order=3)
    _model_with(restored, SimpleNamespace())._restore_structural_extras(saved)
    assert restored.word_concepts('cat') == tuple(ids)''','''    source = _model_with(cs, SimpleNamespace())
    source.symbolSpace = SimpleNamespace(ltm_store=cs.definitions._store())
    saved = source._collect_structural_extras()
    restored = _cs(nS=256, order=3)
    target = _model_with(restored, SimpleNamespace())
    target.symbolSpace = SimpleNamespace(ltm_store=restored.definitions._store())
    target.symbolSpace.ltm_store.load_state_dict(source.symbolSpace.ltm_store.state_dict())
    target._restore_structural_extras(saved)
    assert restored.word_concepts('cat') == tuple(sorted([restored.definitions.word(form='cat'), *ids]))''')
edit(files[2],"        assert restored._row_surfaces == owner._row_surfaces", "        assert dict(restored.definitions._forms) == dict(owner.definitions._forms)\n        assert not hasattr(restored, '_row_surfaces')")
edit(files[3],"    cs.remember_word_surface(row, b'beta', object_row=row, object_id=obj)\n",'')
edit(files[4],"assert len(model._concept_owner()._row_surfaces) >= 178","assert len(model._concept_owner().definitions.word_ids) >= 178")
records=[]
for f,s in before.items():
 after=Path(f).read_text();old={n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
 for n in ast.parse(after).body:
  if isinstance(n,ast.FunctionDef) and n.name in old:
   new=ast.get_source_segment(after,n)
   if new!=old[n.name]:records.append(dict(file=f,name=n.name,before=old[n.name],after=new,reason='§17: spellings and object associations are derived from DEF rows; a supplied native witness is interpreted, and checkpoint fixtures carry the common truth store. Thresholds retained.'))
(here/'index-test-ports.json').write_text(json.dumps(records,indent=2)+'\n')
