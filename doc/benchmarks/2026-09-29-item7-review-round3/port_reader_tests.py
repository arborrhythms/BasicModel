"""Explicit second-pass fixture ports; original bodies retained for review."""
import ast,json
from pathlib import Path
root=Path.cwd();here=root/'doc/benchmarks/2026-09-29-item7-review-round3'
files=['test/test_word_store.py','test/test_iterated_symbolic_wave.py','test/test_cs_symbol_table.py','test/test_item9b_interpret.py','test/test_relevance_bases.py','test/test_sparse_concept_e2e.py','test/test_wholespace_property_migration.py','test/test_radix_layer_reverse.py']
before={f:(root/f).read_text() for f in files}
def edit(f,old,new):
 p=root/f;s=p.read_text();assert old in s,(f,old);p.write_text(s.replace(old,new,1))
def function(f,name,new):
 p=root/f;s=p.read_text();node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name);lines=s.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=[new.rstrip()+'\n'];p.write_text(''.join(lines))
edit(files[1],'    return cs','    from definition_fixtures import with_definitions\n    return with_definitions(cs)')
edit(files[1],'    A, _b = cs.interpret_word','    _word, A = cs.interpret_word')
edit(files[2],'    before = float(store.participation[row])','    # Exercise re-use after the ordinary value decay, not an already saturated gate.\n    store.participation[row] = .25\n    before = float(store.participation[row])')
edit(files[3],"        cs.interpret.define(cs.definitions.word(form=form), kind)","        cs._csw_concept_row(2, kind)  # the supplied kind owns a payload row\n        cs.interpret.define(cs.definitions.word(form=form), kind)")
edit(files[3],"    witnessed = cs.new_concept()","    cs._serial = False  # witness the later definition through the parallel field\n    witnessed = cs.new_concept()")
function(files[4],'test_symbol_history_projection','''def test_symbol_history_projection():
    """Heat follows a genuine sigma reference; DEF is not a part edge."""
    from test_cs_symbol_table import _cs_sparse_active
    cs = _cs_sparse_active()
    _, obj = cs.interpret_word([1], 2, key="w1")
    parent = cs.singleton_concept(obj)
    row_object = cs._csw_row_of(parent)
    assert row_object is not None
    p = cs.symbol_history_priority({int(obj): 5.0})
    assert p is not None and float(p[row_object]) >= 5.0
    assert cs.symbol_history_priority({}) is None
    assert cs.symbol_history_priority(None) is None
''')
edit(files[5],'    cs.interpret_word([1, 2], WORD, key="cat")\n    assert [id(p) for p in cs.getParameters()]','    # No admitted word: forced interpret now activates feature parameters.\n    assert [id(p) for p in cs.getParameters()]')
edit(files[6],'cs._maybe_autobind_meta(None, None)','cs._maybe_autobind_words(None, None)')
edit(files[7],"cs.remember_word_surface(cs._csw_row_of(word), b'beta', object_row=row, object_id=obj)","cs.remember_word_surface(row, b'beta', object_row=row, object_id=obj)")
function(files[0],'test_surface_bytes_belong_to_words_and_objects_follow_their_association','''def test_surface_bytes_belong_to_words_and_objects_follow_their_association():
    m = _surface_snapshot_model()
    try:
        isp, owner = m.inputSpace, m._concept_owner()
        row = int(isp._ar_word_object_rows[0, 0])
        obj = int(isp._ar_word_object_ids[0, 0])
        word = int(isp._ar_word_concept_ids[0, 0])
        other_word = int(isp._ar_word_concept_ids[0, 1])
        other_row = int(isp._ar_word_object_rows[0, 1])
        assert row != other_row
        assert owner._csw_row_of(word) is None
        reference = isp._ar_word_object_atoms
        target = m._byte_tables(*reference.shape[:2])
        before = _surface_byte_cost(m, reference, target)
        store = owner.definitions._store()
        old = owner.definitions.row(word, obj)
        store.origin[old] = store.ORIGIN_USER
        store._update_semantic_fingerprint(old)
        store.clear_origin(store.ORIGIN_USER)
        owner.interpret.define(other_word, obj)
        m._stage_snapshot_bytes()
        bank = isp._ar_concept_lookup_rows[0]
        object_col = int((bank == row).nonzero()[0])
        expected = owner.word_surface_for_row(other_row)
        assert bytes(isp._ar_bank_bytes[0, object_col, :len(expected)].tolist()) == expected
        assert _surface_byte_cost(m, reference, target)[0] > before[0] + 1.0
        store.clear_origin(store.ORIGIN_CONVERSATION)
        m._stage_snapshot_bytes()
        assert not bool(isp._ar_bank_valid[0, object_col].any())
    finally:
        m.End()
        m.symbolSpace.soft_reset()
''')
edit(files[0],'        owner._row_surfaces[word_row] = b"ZZ"\n        owner._object_word_concept[obj] = other_word', '''        word = int(isp._ar_word_concept_ids[0, 0])
        store = owner.definitions._store()
        old = owner.definitions.row(word, obj)
        store.origin[old] = store.ORIGIN_USER
        store._update_semantic_fingerprint(old)
        store.clear_origin(store.ORIGIN_USER)
        owner.interpret.define(other_word, obj)''')
edit(files[0],'        owner._row_surfaces.clear()','        owner.definitions._store().clear_origin(owner.definitions._store().ORIGIN_CONVERSATION)')
records=[]
for f,s in before.items():
 after=(root/f).read_text();old={n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
 for n in ast.parse(after).body:
  if isinstance(n,ast.FunctionDef) and n.name in old:
   new=ast.get_source_segment(after,n)
   if new!=old[n.name]:records.append(dict(file=f,name=n.name,before=old[n.name],after=new,reason='§17/§18: DEF identity lookup and object-owned inventory seats; explicit native fixtures replace retired META/cache assumptions. Numerical assertions retained.'))
(here/'reader-test-ports.json').write_text(json.dumps(records,indent=2)+'\n')
