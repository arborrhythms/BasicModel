"""Identity-based readers and checkpoint probes before repairing Stage B."""
from pathlib import Path
from types import SimpleNamespace
import torch
from test_item7_definitions import model, admit


def test_decoder_follows_the_definition_rows_after_reassociation_and_forgetting():
    m = model()
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    word, obj = admit(m, 'hello')
    other, _ = admit(m, 'there')
    row = cs._csw_row_of(obj)
    assert cs.word_surface_for_row(row) == b'hello'
    definition = cs.definitions.row(word, obj)
    store.origin[definition] = store.ORIGIN_USER
    store._update_semantic_fingerprint(definition)
    store.clear_origin(store.ORIGIN_USER)
    assert cs.word_surface_for_row(row) is None
    cs.interpret.define(other, obj)
    assert cs.word_surface_for_row(row) == b'there'


def test_pending_field_word_reservation_survives_checkpoint(tmp_path):
    from test_grounded_xor import grounded_model
    m, _ = grounded_model(tmp_path)
    cs = m._concept_owner()
    word = cs.interpret.lookup_word([48, 48], [1], form='00', word_reading=True)
    assert word in cs.interpret._field_pending
    saved = m._collect_structural_extras()
    fresh, _ = grounded_model(tmp_path)
    fresh.symbolSpace.ltm_store.load_state_dict(m.symbolSpace.ltm_store.state_dict())
    fresh._restore_structural_extras(saved)
    target = fresh._concept_owner()
    assert target.interpret._pending == cs.interpret._pending
    assert target.interpret._field_pending == cs.interpret._field_pending
    assert word in target.word_concepts('00')


def test_taxonomy_keeps_the_forms_derived_from_definition_rows():
    from test_cs_sparse_weights import _cs
    from ConceptIndex import index_part_row
    cs = _cs(nS=64, order=3)
    word, whole = cs.interpret_word([7], [1], key='animal')
    _, part = cs.interpret_word([8], [1], key='cat')
    parent = index_part_row(cs, part, whole)
    assert parent in cs.word_concepts('animal')
    assert cs._concept_source_order(parent) == cs._concept_source_order(part) + 1


def test_migrated_meta_has_no_live_identity_or_inventory_row():
    from test_cs_sparse_weights import _cs
    from test_structural_checkpoint import _model_with
    from Layers import TernaryTruthStore
    fixture = torch.load(Path('test/fixtures/item7_legacy_definition.pt'), weights_only=False)
    cs = _cs(nS=64, order=3)
    target = _model_with(cs, SimpleNamespace())
    target.symbolSpace = SimpleNamespace(ltm_store=TernaryTruthStore(cs.nDim, capacity=32))
    object.__setattr__(cs, '_model', target)
    target._restore_structural_extras(fixture['structural'])
    meta = fixture['meta']
    alloc = cs._concept_allocator
    assert meta not in alloc.placement
    assert cs._csw_row_of(meta) is None
    assert not alloc.records(meta)
