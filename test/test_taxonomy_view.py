"""A bounded derived taxonomy view reads only conceptual reference records."""
from types import SimpleNamespace

import pytest
import torch

from test_cs_symbol_table import _cs


def _world():
    cs = _cs()
    return cs, tuple(cs.new_concept() for _ in range(3))


def _view(cs, **limits):
    from Taxonomy import capture_taxonomy
    return capture_taxonomy(cs, **limits)


def _ref(cid):
    return ("sym", cid)


def test_direct_taxonomic_links_change_partof_with_canonical_direction():
    cs, (a, b, _c) = _world()
    assert _view(cs).part_of(_ref(a), _ref(b))["support_true"] == 0
    cs.add_whole(a, _ref(b))
    result = _view(cs).part_of(_ref(a), _ref(b))
    assert result["support_true"] == 1
    assert result["support_false"] == 0
    assert result["domain"] == "conceptual-taxonomy"
    assert result["path"][0].owner == _ref(a)
    assert _view(cs).part_of(_ref(b), _ref(a))["support_true"] == 0


def test_reified_relation_is_visible_without_endpoint_mutation():
    cs, (a, b, _c) = _world()
    edge = cs.reify_concept(a, b)
    assert cs.concept_wholes(a) == cs.concept_parts(b) == []
    result = _view(cs).part_of(_ref(a), _ref(b))
    assert result["support_true"] == 1
    assert {item.owner for item in result["path"]} == {_ref(edge)}
    cs.retire_concept(edge)
    assert _view(cs).part_of(_ref(a), _ref(b))["support_true"] == 0


def test_percept_codes_vector_overlap_and_world_rows_are_not_taxonomy_edges():
    from Layers import TernaryTruthStore
    cs, (a, b, _c) = _world()
    cs.add_part(b, a)  # raw percept code, not a sym reference
    cs.add_whole(a, b)  # raw whole-space code, not a sym reference
    cs._ltm_store = TernaryTruthStore(4)
    cs._ltm_store.append_relation(torch.eye(4)[0], torch.eye(4)[2], torch.ones(4),
                                 rel_type=cs._ltm_store.REL_PARTOF, trust=1)
    cs.wholeSpace_ref = SimpleNamespace(taxonomy_parent=lambda _ref: b)
    result = _view(cs).part_of(_ref(a), _ref(b))
    assert result["support_true"] == result["support_false"] == 0


def test_snapshot_does_not_read_later_taxonomy_mutations():
    cs, (a, b, _c) = _world()
    prior = _view(cs)
    cs.add_whole(a, _ref(b))
    assert prior.part_of(_ref(a), _ref(b))["support_true"] == 0
    assert _view(cs).part_of(_ref(a), _ref(b))["support_true"] == 1


def test_taxonomy_cycles_do_not_loop_and_retired_references_are_unavailable():
    cs, (a, b, c) = _world()
    cs.add_whole(a, _ref(b))
    cs.add_whole(b, _ref(a))
    cs.add_whole(b, _ref(c))
    result = _view(cs).part_of(_ref(a), _ref(c), max_steps=8)
    assert result["support_true"] == 1
    assert len(result["path"]) == 2
    cs.retire_concept(c)
    unavailable = _view(cs).part_of(_ref(a), _ref(c))
    assert unavailable["support_true"] == 0
    assert "unavailable_reference" in unavailable["incomplete"]


def test_taxonomy_budgets_report_incomplete_without_fabricating_falsehood():
    cs, (a, b, _c) = _world()
    cs.reify_concept(a, b)
    view = _view(cs, max_nodes=1, max_records=1)
    result = view.part_of(_ref(a), _ref(b))
    assert result["support_true"] == result["support_false"] == 0
    assert result["incomplete"]
    assert view.nodes_scanned <= 1 and view.records_scanned <= 1
    limited = _view(cs).part_of(_ref(a), _ref(b), max_steps=1)
    assert limited["support_true"] == limited["support_false"] == 0
    assert "traversal_limit" in limited["incomplete"]


@pytest.mark.parametrize("bad", [1, ("ws", 1), ("sym", True), ("sym", 1.5)])
def test_queries_require_typed_concept_handles_not_numeric_row_features(bad):
    cs, (a, _b, _c) = _world()
    with pytest.raises((TypeError, ValueError)):
        _view(cs).part_of(bad, _ref(a))


def test_capture_is_a_pure_read_even_before_allocator_initialization():
    cs = SimpleNamespace()
    view = _view(cs)
    assert not hasattr(cs, "_concept_allocator")
    assert view.part_of(_ref(1), _ref(2))["support_true"] == 0


def test_duplicate_sources_cannot_inflate_taxonomy_support():
    cs, (a, b, _c) = _world()
    cs.add_whole(a, _ref(b))
    cs.add_part(b, _ref(a))
    result = _view(cs).part_of(_ref(a), _ref(b))
    assert result["support_true"] == 1
    assert len(result["path"]) == 1


def test_capture_charges_every_record_read_including_raw_domain_refs(monkeypatch):
    cs, (a, b, c) = _world()
    for ref in (42, _ref(b), _ref(c)):
        cs.add_part(a, ref)
    layer = cs._concept_allocator.store_of(a)
    original = layer.iter_constituents
    read = []
    def observe(cid):
        for item in original(cid):
            read.append((cid, item))
            yield item
    monkeypatch.setattr(layer, "iter_constituents", observe)
    view = _view(cs, max_nodes=8, max_records=1)
    assert len(read) == view.records_scanned == 1
    assert view.part_of(_ref(b), _ref(a))["support_true"] == 0
