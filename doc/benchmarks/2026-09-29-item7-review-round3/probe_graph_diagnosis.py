"""Read-only observations separating setup capacity from closing identities."""
import json
from pathlib import Path
import pytest
from test_mm_xor import _fresh_model

HERE = Path(__file__).resolve().parent


def test_registry_capacity_is_exhausted_before_a_sentence_is_read():
    model, _, _ = _fresh_model(str(Path('data/matrix/MM_20M_grammar_wordstore.xml').resolve()))
    try:
        registry = model.grammatical_thoughts
        cs = registry.space
        store = model.symbolSpace.ltm_store
        result = dict(concept_capacity=cs.nVectors, order_capacities=cs._order_caps(),
            unavailable=registry.unavailable_operation_ids, reason=registry.unavailable_reason,
            physical_rows=dict(cs._csw_rows), next_identity=cs._concept_allocator.next_id,
            store_rows=len(store))
        (HERE/'graph-setup-diagnosis.json').write_text(json.dumps(result, indent=2)+'\n')
        assert len(store) == 0
        assert 'part' in registry.unavailable_operation_ids
        with pytest.raises(RuntimeError, match='snap block has 0 occupied rows and capacity 4'):
            registry.clause_reference('part')
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_repeated_sentences_record_their_actual_new_identities():
    model, _, _ = _fresh_model(str(Path('data/MM_ladder.xml').resolve()))
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    try:
        texts = ['hello world', 'hello there', 'loving world', 'loving there']
        cs, store = model._concept_owner(), model.symbolSpace.ltm_store
        snapshots = []
        for _ in range(2):
            model.forward(model.inputSpace.prepInput(texts))
            alloc = cs._concept_allocator
            snapshots.append(dict(next_identity=alloc.next_id,
                physical_rows=len(alloc.layer()._tensor_rows), store_rows=len(store),
                definitions=int((store.rel_type[:len(store)] == store.REL_DEF).sum()),
                idea_ids=store.row_ids[store.ideas()].tolist(),
                relations=store.row_ids[store.relations()].tolist()))
        (HERE/'sentence-identity-diagnosis.json').write_text(json.dumps(snapshots, indent=2)+'\n')
    finally:
        model.End()
        model.symbolSpace.soft_reset()
