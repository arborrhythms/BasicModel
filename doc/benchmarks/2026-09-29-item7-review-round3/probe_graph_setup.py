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
            physical_rows=dict(cs._csw_rows), next_identity=getattr(getattr(cs, '_concept_allocator', None), 'next_id', 1),
            store_rows=len(store))
        (HERE/'graph-setup-diagnosis.json').write_text(json.dumps(result, indent=2)+'\n')
        assert len(store) == 0
        assert 'part' in registry.unavailable_operation_ids
        with pytest.raises(RuntimeError, match='snap block has 0 occupied rows and capacity 4'):
            registry.clause_reference('part')
    finally:
        model.End()
        model.symbolSpace.soft_reset()

