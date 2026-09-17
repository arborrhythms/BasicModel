"""Taxonomy control/credit cannot regain legacy world-evidence shortcuts."""
from types import SimpleNamespace

import torch

from Layers import TernaryTruthStore
from reasoning import QuerySpec, TruthGroundedReasoner
from thinking import ThinkingKernel, Frame, Testimony, traces_from_store
from test_cs_symbol_table import _cs


def _setup():
    cs = _cs()
    a, b, c = (("sym", cs.new_concept()) for _ in range(3))
    store = TernaryTruthStore(8)
    kernel = ThinkingKernel(TruthGroundedReasoner(SimpleNamespace(conceptualSpace=cs), store=store))
    return cs, (a, b, c), store, kernel


def test_numeric_testimony_cannot_establish_taxonomic_inclusion():
    _cs, (a, b, _), store, kernel = _setup()
    spec = QuerySpec.from_surface("isPart", a, b)
    frame = Frame(spec)
    frame.bindings["testimony"] = [Testimony(spec, 1., "expert", source_trust=1.)]
    assert kernel._frame_interval(frame).upper == 0
    assert kernel.incorporate(frame.bindings["testimony"][0]) == -1
    assert len(store) == 0


def test_unrelated_true_child_cannot_establish_parent_relation():
    _cs, (a, b, _), store, kernel = _setup()
    fact = torch.eye(8)[0]
    store.append_idea(fact, trust=1.)
    frame = Frame(QuerySpec.from_surface("isPart", a, b))
    kernel._pool = 12
    kernel.execute(frame, {"op": "think", "target": QuerySpec.from_surface("exist", fact)})
    assert kernel._frame_interval(frame).upper == 0


def test_nested_taxonomy_result_keeps_each_record_source():
    cs, (a, b, c), _store, kernel = _setup()
    cs.add_whole(a[1], b)
    cs.add_whole(b[1], c)
    result = kernel.run(QuerySpec.from_surface("isPart", a, c))
    assert result.value == "true"
    def edges(value):
        from Taxonomy import TaxonomyEdge
        if isinstance(value, TaxonomyEdge):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from edges(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from edges(item)
    sources = set(edges(result.provenance))
    assert {(edge.part, edge.whole) for edge in sources} == {(a, b), (b, c)}


def test_legacy_trace_training_derives_targets_from_taxonomy_only():
    cs, (a, b, c), store, kernel = _setup()
    cs.add_whole(a[1], b)
    cs.add_whole(b[1], c)
    examples = traces_from_store(kernel)
    assert examples and any(op == "think" for _state, op in examples)
    assert len(store) == 0
    cs.retire_concept(b[1])
    x, y, z = torch.eye(8)[:3]
    store.append_relation(x, torch.zeros(8), y, rel_type=store.REL_PARTOF, trust=1.)
    store.append_relation(y, torch.zeros(8), z, rel_type=store.REL_PARTOF, trust=1.)
    assert traces_from_store(kernel) == []
