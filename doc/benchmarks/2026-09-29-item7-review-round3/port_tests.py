"""Explicit ports for the §17 row replacement; original bodies are recorded."""
import ast, json, re
from pathlib import Path
HERE=Path(__file__).resolve().parent
records=[]

def edit(file, transform, reason):
    p=Path(file);old=p.read_text();new=transform(old)
    if new==old:return
    before={n.name:ast.get_source_segment(old,n) for n in ast.walk(ast.parse(old)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    after={n.name:ast.get_source_segment(new,n) for n in ast.walk(ast.parse(new)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    for name, body in before.items():
        if after.get(name)!=body:records.append(dict(file=file,name=name,reason=reason,before=body,after=after.get(name)))
    p.write_text(new)

def remove(source,names):
    lines=source.splitlines(keepends=True)
    tree=ast.parse(source)
    for n in sorted((n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names),key=lambda n:n.lineno,reverse=True):
        start=min([n.lineno]+[d.lineno for d in n.decorator_list])-1
        del lines[start:n.end_lineno]
    return ''.join(lines)

def functions(source,replacements):
    lines=source.splitlines(keepends=True)
    for n in sorted((n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name in replacements),key=lambda n:n.lineno,reverse=True):
        body=replacements[n.name]
        lines[n.lineno-1:n.end_lineno]=[body.rstrip()+'\n']
    return ''.join(lines)

# Bare ConceptualSpace fixtures explicitly supply the one definition owner.
# Clause-only tests retain their separately constructed target store: the
# vocabulary fixture is outside the closing whose row delta they assert.
Path('test/definition_fixtures.py').write_text('''"""Definition owner for isolated conceptual-inventory mechanism fixtures."""
from Layers import TernaryTruthStore


def with_definitions(cs):
    object.__setattr__(cs, '_definition_store', TernaryTruthStore(cs.nDim, capacity=1024))
    return cs
''')
for file in ('test/test_cs_sparse_weights.py','test/test_cs_symbol_table.py','test/test_concept_capacity_policy.py','test/test_sparse_concept_e2e.py','test/test_attention_promotion.py'):
 def fixture(s):
  s='from definition_fixtures import with_definitions\n'+s if not s.startswith('"""') else s
  # Imports follow future statements/docstring, avoiding statement reordering.
  s=s.replace('import torch\n','import torch\nfrom definition_fixtures import with_definitions\n',1)
  s=s.replace('    return cs\n','    return with_definitions(cs)\n')
  s=s.replace('return Spaces.ConceptualSpace([nP, _D], [nS, _D], [nS, _D])','return with_definitions(Spaces.ConceptualSpace([nP, _D], [nS, _D], [nS, _D]))')
  s=s.replace('    cs = Spaces.ConceptualSpace([nP, _D], [64, _D], [64, _D])   # NOT active','    cs = with_definitions(Spaces.ConceptualSpace([nP, _D], [64, _D], [64, _D]))')
  return s
 edit(file,fixture,'§17: standalone interpretation fixture supplies a truth store; no production fallback store.')

# Replace the eliminated third META return in existing mechanism tests.
for p in Path('test').glob('test_*.py'):
 if p.name=='test_item7_definitions.py':continue
 s=p.read_text()
 if 'interpret_word' not in s:continue
 def pairs(s):
  return re.sub(r'(?m)^(\s*)([A-Za-z_][\w]*),\s*([A-Za-z_][\w]*),\s*([A-Za-z_][\w]*)\s*=\s*(.*\.interpret_word\()',r'\1\2, \3 = \5',s)
 edit(str(p),pairs,'§17: interpret_word returns word/object identities; no third META exists.')

for file in ('test/test_word_store.py','test/test_autobind_from_cs.py'):
 edit(file,lambda s:s.replace('_maybe_autobind_meta','_maybe_autobind_words'),'§17.5: rename the retained category/recognition boundary after removing META.')
for file in ('test/test_mereology_word_binding.py','test/test_item7_taxonomy.py','test/test_cs_symbol_table.py'):
 edit(file,lambda s:s.replace('.word_references.deref(','.definitions.deref(').replace('.word_references.candidates(','.definitions.objects('),'§17.4: use the derived DEF-row index.')

edit('test/test_reference_table.py',lambda s:remove(s,{
 'test_nary_bindings_keep_each_selected_extent','test_full_rows_only','test_append_only','test_gate_license_required',
 'test_unknown_word_is_a_query_outcome','test_no_reverse_index_anywhere','test_search_is_object_side_scan_by_dominance',
 'test_minted_whole_is_searchable_by_its_parts','test_bind_gauge_orients_object_row','test_bottom_extent_cached_definable_but_empty','test_top_saturation_detected'
 }).replace('from References import ReferenceTable, symbol_code','from References import symbol_code'),
 'Retired by §17.4–5: ReferenceTable, vector-dominance search, binding gauge and extent caches are deleted; DEF tests 22/24/25/28/34 replace these contracts. Symbol-code assertions stay.')
edit('test/test_mereology_word_binding.py',lambda s:remove(s,{'test_one_word_object_meta_has_typed_native_members'}),
 'Retired by §17.5: META membership/order; replaced by test 22 of definition rows.')
edit('test/test_item7_taxonomy.py',lambda s:remove(s,{'test_retiring_one_testimony_object_preserves_its_shared_meta','test_restore_rebuilds_native_meta_membership_and_invalidates_direct_index'}),
 'Retired META lifecycle/checkpoint reader; new tests 28/32 cover common-store deletion and actual HEAD migration.')
edit('test/test_item7_taxonomy.py',lambda s:s.replace('    assert not cs.is_meta(parent)','    assert parent not in cs.definitions.word_ids'),
 '§17.5: the parent remains an object concept; the META predicate no longer exists.')
edit('test/test_item7_taxonomy.py',lambda s:functions(s,{
 'test_one_meta_generalizes_over_several_words_and_objects':'''def test_definition_rows_generalize_over_several_words_and_objects():
    cs, interpret = _operator()
    first = interpret.lookup_word([7], [1], form='bank')
    river = interpret.forward(first)
    second = interpret.lookup_word([8], [1], form='shore')
    money = interpret.forward(second)
    interpret.define(first, money)
    interpret.define(second, river)
    assert cs.definitions.objects(first) == (river, money)
    assert cs.definitions.deref(first, selected=river) == river
    assert cs.definitions.deref(second, selected=money) == money
    with pytest.raises(ValueError, match='selection'):
        cs.definitions.deref(first)
''',
 'test_meta_binding_does_not_discriminate_at_binding_time':'''def test_definition_binding_preserves_ambiguous_objects_until_selection():
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [1], form='bank')
    first = interpret.forward(word)
    second = cs.new_concept()
    _concept_alloc_of(cs).reference_orders[second] = 1
    interpret.define(word, second)
    assert interpret.forward(word, selected=second) == second
    assert interpret.forward(word, selected=first) == first
    assert cs.definitions.objects(word) == (first, second)
'''}),'§17.2/4: n-ary META membership is several DEF rows; ambiguity/selection assertions preserved.')

edit('test/test_item9b_interpret.py',lambda s:remove(s,{'test_unused_one_off_testimony_is_forgotten_at_the_boundary','test_native_parallel_pass_and_boundary_do_not_interpret_words'}),
 '§17.6/18: word-specific forgetting is deleted and every reading interprets; tests 27/28/31/33 replace these contrary contracts.')
edit('test/test_item9b_interpret.py',lambda s:s.replace('return cs, InterpretLayer(conceptualSpace=cs)','return cs, cs.interpret'),
 'One interpret writer per fixture, including its transient admission transaction.')
edit('test/test_item9b_interpret.py',lambda s:functions(s,{
 'test_interpret_reuses_provisional_testimony_and_reverses_by_identity':'''def test_interpret_reuses_definition_and_reverses_by_identity():
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [1], form='wug')
    obj = interpret.forward(word, occurrence=(0, 1))
    assert obj != word
    assert ('sym', word) not in cs.concept_parts(obj)
    assert cs._concept_source_order(obj) == 0
    row = cs._csw_row_of(obj)
    assert cs._csw_row_of(word) is None
    before = len(cs.definitions._store())
    for sentence in range(1, 20):
        assert interpret.forward(word, occurrence=(sentence, 1)) == obj
        interpret.boundary()
    assert len(cs.definitions._store()) == before
    assert interpret.reverse(obj) == word
    assert cs._csw_row_of(obj) == row
'''}),'§17.7: one object row with native parts/wholes replaces raised testimony and its word edge; repeated identity/inverse assertions retained.')
edit('test/test_item9b_interpret.py',lambda s:s.replace("cs.bind_word_concept('cat', kind)","interpret.define(word, kind)\n    cs.bind_word_concept('cat', kind)").replace('assert cs._concept_source_order(particular) == 1','assert cs._concept_source_order(particular) == 0').replace('interpret.forward(word, order=1) == particular','interpret.forward(word, order=0) == particular').replace("        cs.bind_word_concept(form, kind)","        cs.interpret.define(cs.definitions.word(form=form), kind)\n        cs.bind_word_concept(form, kind)").replace('cs._csw_row_of(word) for word in words','cs._csw_row_of(obj) for obj in objects'),
 '§17.3: supplied grammar associations use interpret.define; object occupies the word row at native order. Kind selection assertions remain.')
edit('test/test_item9b_corrections.py',lambda s:s.replace("    cs.bind_word_concept('cat', kind)","    interpret.define(word, kind)\n    cs.bind_word_concept('cat', kind)"),
 '§17.3: interpret is the sole definition writer; a native form annotation does not write a DEF row.')
edit('test/test_item7_acceptance.py',lambda s:s.replace('word_rows.append(self.cs._csw_row_of(wid))','word_rows.append(self.cs._csw_row_of(obj))'),
 '§17.7: AnswerProgram physical word row is now the object row; all closing assertions retained.')

edit('test/test_cs_symbol_table.py',lambda s:remove(s,{'test_meta_membership_persists_identity_resolution','test_meta_word_object_recovers_by_intersection','test_fresh_object_concept_reads_its_naming_word_at_order1'}),
 '§17.5 deletes META and the object-to-word part edge; tests 22/24/34 replace these assertions.')
edit('test/test_cs_symbol_table.py',lambda s:functions(s,{
 'test_interpret_word_structure':'''def test_interpret_word_structure():
    cs = _cs()
    A, B = cs.interpret_word([65, 66, 67], WORD)
    assert set(cs.concept_parts(A)) == {65, 66, 67}
    assert cs.concept_wholes(A) == [WORD]
    assert set(cs.concept_parts(B)) == {65, 66, 67}
    assert cs.concept_wholes(B) == [WORD]
    assert cs._csw_row_of(A) is None
    assert cs._csw_row_of(B) is not None
    assert cs.definitions.deref(A) == B
'''}),'§17.7: the object inherits the word row, not a word part-edge; parts, wholes and identity are still checked.')
# The following mechanisms exercise a concept inventory row, not a word's spelling record.
for file in ('test/test_cs_symbol_table.py','test/test_order0_percept_definitions.py','test/test_iterated_symbolic_wave.py'):
 def objects(s):
  s=s.replace('A, _B = cs.interpret_word','_word, A = cs.interpret_word')
  if file.endswith('test_order0_percept_definitions.py'):
   s=s.replace('A, _ = cs.interpret_word','word, A = cs.interpret_word').replace('cs.concept_parts(A) == letters','cs.concept_parts(word) == letters')
  s=s.replace('(A1, B1, C1) == (A2, B2, C2)','(A1, B1) == (A2, B2)').replace('(A, B, C)','(A, B)')
  s=s.replace('assert cs.concept_parts(B) == [("sym", A)] and cs.concept_wholes(B) == []','assert set(cs.concept_parts(B)) == {65, 66, 67} and cs.concept_wholes(B) == [WORD]')
  s=s.replace('assert set(cs.concept_parts(B)) == {("sym", A)}','assert set(cs.concept_parts(B)) == {1}').replace('assert set(cs.concept_wholes(B)) == set()','assert set(cs.concept_wholes(B)) == {2}')
  s=s.replace('assert cs.concept_weights(row) == [(cs._csw_row_of(A), 1.)]','assert cs._csw_row_of(A) is None and cs.definitions.deref(A) == B')
  return s
 edit(file,objects,'§17.7: mechanism addresses the object inventory row; no duplicated word row/META. Numerical edge and repeat-recognition assertions retained.')

edit('test/test_sparse_concept_e2e.py',lambda s:remove(s,{'test_meta_is_a_sigma_generalization_over_named_word_and_object_members'}),
 '§17.5: deleted META sigma fold; definition tests 22/24 replace this obsolete shape.')
edit('test/test_sparse_concept_e2e.py',lambda s:s.replace('a_row = cs._csw_concept_row(0, A)','a_row = cs._csw_row_of(B)').replace('    assert not any(row == b_row for row, _ in features._index)','    assert b_row == a_row and cs._csw_row_of(A) is None').replace('test_word_symbol_defines_order0_native_features_and_object_stays_unwritten','test_interpreted_object_keeps_the_native_features_in_one_row').replace('    assert getattr(cs, "_sparse_fam", None) is None    # nothing populated','    assert Spaces._concept_alloc_of(cs).layer().features.nnz > 0'),
 '§18/17.7: every read word has a native definition, including zero symbolic recursion; the object occupies that row.')
edit('test/test_reconstruction_bank_contract.py',lambda s:s.replace('owner._concept_allocator.word_obj_meta','owner.definitions.word_ids'),
 '§17.4: first-sight bank checks the derived definition index.')

# Codebook mask tests supply the interface they isolate, without the retired store.
edit('test/test_codebook_update_law.py',lambda s:remove(s,{'test_table_accessors'}).replace('from References import ReferenceTable\n','').replace('    table = ReferenceTable()\n    table.bind(word=1, obj=3, licensed=True)','    table = types.SimpleNamespace(bound_words=lambda: [1], bound_objects=lambda: [3])').replace('        t = ReferenceTable()\n        t.bind(word=2, obj=0, licensed=True)','        t = types.SimpleNamespace(bound_words=lambda: [2], bound_objects=lambda: [0])'),
 'Mask wiring is tested with an explicit index interface; deleted ReferenceTable accessors are covered by definition test 24.')

(HERE/'test-ports.json').write_text(json.dumps(records,indent=2)+'\n')
print(len(records),'recorded function changes')

# Capacity tests reserve physical rows and common-store rows, not a retired
# triple of identities. Their snapshot assertions still check atomic refusal.
edit('test/test_concept_capacity_policy.py',lambda s:s.replace('"word_obj_meta": dict(alloc.word_obj_meta),','"definitions": dict(cs.definitions._rows),'),
 '§17.4: snapshot the derived definition pairs instead of allocator META records.')
edit('test/test_concept_capacity_policy.py',lambda s:functions(s,{
 'test_explicit_word_triple_capacity_failure_is_atomic':'''def test_explicit_word_row_capacity_failure_is_atomic():
    cs = _cs(8)
    for row in range(8):
        assert cs._csw_concept_row(0, 1000 + row) == row
    before = _allocator_snapshot(cs)
    with pytest.raises(RuntimeError, match='capacity'):
        cs.interpret_word([10, 11], (0,), key='unseated')
    _assert_snapshot_equal(before, _allocator_snapshot(cs))
''',
 'test_automatic_capacity_mode_reuses_known_identity_without_recycling':'''def test_automatic_capacity_mode_reuses_known_identity_without_recycling():
    cs = _cs(8)
    known = cs.interpret_word([10], (0,), key='known')
    for row in range(1, 8):
        assert cs._csw_concept_row(0, 1000 + row) == row
    cs.retire_concept(1007)
    assert cs.interpret_word([10], (0,), key='known') == known
    before = _allocator_snapshot(cs)
    with pytest.raises(RuntimeError, match='capacity'):
        cs.interpret_word([20], (0,), key='unseen')
    _assert_snapshot_equal(before, _allocator_snapshot(cs))
    assert 1007 in cs._concept_allocator.retired
    before = _allocator_snapshot(cs)
    with pytest.raises(RuntimeError, match='capacity'):
        cs.interpret_word([10, 11], (0,), key='known')
    _assert_snapshot_equal(before, _allocator_snapshot(cs))
''',
 'test_rejected_word_does_not_consume_location_fallback_rows':'''def test_rejected_word_does_not_consume_location_fallback_rows(remaining):
    cs = _cs(8)
    for row in range(8 - remaining):
        cs._csw_concept_row(0, 1000 + row)
    store = cs.definitions._store()
    while len(store) < store.capacity:
        store.append_idea(torch.zeros(_D))
    before = _allocator_snapshot(cs)
    spans = torch.tensor([[[0, 2]]])
    cs._autobind_property_concepts(
        torch.tensor([[10, 11]]), torch.randn(1, 2, _D),
        torch.tensor([[0, 0]]), [['new']], [['new']],
        percept_where=torch.tensor([[0, 1]]), percept_when=None,
        tile_spans=None, percept_store=None, ws=_property_ws(spans))
    _assert_snapshot_equal(before, _allocator_snapshot(cs))
    assert cs.definitions.word(form='new') is None
    assert len(cs._concept_allocator.layer()._tensor_rows) == 8 - remaining
''',
 'test_rejected_mixed_type_word_suppresses_every_overlapping_ws_span':'''def test_rejected_mixed_type_word_suppresses_every_overlapping_ws_span():
    cs = _cs(8)
    for row in range(8):
        cs._csw_concept_row(0, 1000 + row)
    before = _allocator_snapshot(cs)
    ws = _property_ws(torch.tensor([[[0, 3], [3, 4]]]))
    cs._autobind_property_concepts(
        torch.tensor([[10, 11, 12, 13]]), torch.randn(1, 4, _D),
        torch.tensor([[0, 0, 0, 0]]), [['abc1']], [['abc1']],
        percept_where=torch.tensor([[0, 1, 2, 3]]), percept_when=None,
        tile_spans=[[(0, 4), (0, 4), (0, 4), (0, 4)]],
        percept_store=None, ws=ws)
    _assert_snapshot_equal(before, _allocator_snapshot(cs))
    assert cs.definitions.word(form='abc1') is None
'''}).replace('assert set(alloc.word_obj_meta) == {"ab", "cd"}','assert set(cs.definitions._forms) == {"ab", "cd"}').replace('assert alloc.next_id == 7','assert alloc.next_id == 5'),
 '§17.7: reserve one physical row and one definition row, with two identities. Full-row/full-store refusal and no location fallback remain strict.')

edit('test/test_structural_checkpoint.py',lambda s:s.replace('    alloc.word_obj_meta["cat"] = (word, obj, meta)\n','').replace('    assert restored_alloc.word_obj_meta["cat"] == (word, obj, meta)\n','').replace('    assert restored_cs._word_obj_meta is restored_alloc.word_obj_meta\n','').replace('        ("whole", ("sym", word)), ("part", ("sym", obj)),\n        ("word", ("sym", word)), ("object", ("sym", obj))]','        ("whole", ("sym", word)), ("part", ("sym", obj))]').replace('    assert restored_cs._percept_word_concept[7] == (word, obj)\n','    assert not hasattr(restored_cs, "_percept_word_concept")\n'),
 'Keep generic allocator/parameter restoration assertions; actual old-META migration and rebuilt DEF index are exercised by test 32, not reinstated retired host caches.')
edit('test/test_word_store.py',lambda s:s.replace('    wom = cs._word_obj_meta\n    A_zebra = wom["zebra"][0]\n    A_yak = wom["yak"][0]','    index = cs.definitions\n    A_zebra = index.deref(index.word(form="zebra"))\n    A_yak = index.deref(index.word(form="yak"))'),
 '§17.7: the recognised WORDS category refers to the object row, with forms resolved by DEF index.')
edit('test/test_autobind_from_cs.py',lambda s:s.replace('len(getattr(getattr(cs, "_concept_allocator", None), "lexical_words", {}))','len(cs.definitions.word_ids)'),
 '§17.4: boundary admission counts live defined words in the derived index.')
edit('test/test_attention_promotion.py',lambda s:functions(s,{
 'test_objects_never_acquire_witnessed_kinds':'''def test_interpretation_replaces_the_word_row_without_inventing_a_kind():
    cs, _ = _fixture()
    before = len(cs._concept_allocator.layer()._tensor_rows)
    word, obj = cs.interpret_word([1], 2, key='cat')
    assert cs._csw_row_of(word) is None
    row = cs._csw_row_of(obj)
    assert row in cs._witnessed_rows()
    assert len(cs._concept_allocator.layer()._tensor_rows) == before + 1
    assert cs.taxonomy_parents(obj) == []
'''}),'§17.7: object retains the witnessed native row; old unwitnessed extra object/META rows no longer exist. No invented taxonomy kind is asserted.')
edit('test/test_sparse_concept_e2e.py',lambda s:s.replace('csw = [ly.values for fam in cs._sparse_fam.values() for ly in fam\n           if ly is not None and ly.values is not None]','csw = [matrix.values for matrix in Spaces._concept_alloc_of(cs).layer().definition_matrices()\n           if matrix.values is not None]'),
 '§17.7: optimize actual native definition weights after retiring the META sigma edge; optimizer inclusion assertion stays.')
(HERE/'test-ports.json').write_text(json.dumps(records,indent=2)+'\n')
print(len(records),'total recorded function changes')
