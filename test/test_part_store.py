"""LTM2 has one relation, with readings supplied by native argument types."""
import torch

from Layers import ConceptAllocator
from PartStore import PartStore


def test_taxonomy_and_nested_relations_share_symbol_rows():
    alloc = ConceptAllocator()
    x, y, z = (alloc.new_concept() for _ in range(3))
    alloc.add(y, 'part', ('sym', x))
    inner = alloc.parts.find(x, y)
    assert inner in alloc.placement
    outer = alloc.parts.add(inner, z, truth_plus=.7, truth_minus=.2)
    assert alloc.parts.refs[outer].tolist() == [inner, z]
    assert ('sym', x) in alloc.refs(y, 'part')
    assert alloc.parts.refs[inner].tolist() == [x, y]
    assert len(alloc.parts) == 2
    before = outer
    alloc.parts.retain([inner, outer])
    assert alloc.parts.find(inner, z) == before
    assert set(alloc.parts.row(inner)) == set(PartStore.columns)
    assert alloc.parts.refs.shape[-1] == 2
    assert not hasattr(alloc.parts, 'rel_type')


def test_pair_rewitnesses_both_lanes_without_a_relation_kind():
    store = PartStore(16)
    first = store.add(2, 3, truth_plus=.7, truth_minus=.1, when=torch.ones(4))
    assert store.add(2, 3, truth_plus=.2, truth_minus=.8) == first
    assert len(store) == 1 and store.access_count[first] == 2
    torch.testing.assert_close(store.truth_plus[first], torch.tensor(.7))
    torch.testing.assert_close(store.truth_minus[first], torch.tensor(.8))
    types = {2: {'situation'}, 3: {'situation', 'speech_individual'}}
    assert store.readings(first, types.get) == ('conditional', 'attribution')


def test_part_checkpoint_and_retention_keep_native_addresses():
    store = PartStore(16)
    a = store.add(2, 3, truth_plus=.6)
    b = store.add(a, 4, truth_minus=.4)
    clone = PartStore(4)
    clone.load_state_dict(store.state_dict())
    clone.retain([b])
    assert clone.find(a, 4) == b
    assert clone.find(2, 3) is None
    torch.testing.assert_close(clone.truth_minus[b], torch.tensor(.4))


def test_definition_cannot_answer_identity_query_with_its_singleton_word():
    from part_fixtures import native_store
    from Queries import ThoughtLTMCapability
    from QueryWork import QueryWorkBudget
    store, (obj, word) = native_store(points=torch.eye(4)[:2])
    owner = store._concept_owner()
    alloc = owner._concept_allocator
    alloc.symbol_types[obj].add('object')
    alloc.symbol_types[word].add('word')
    # A native form is a singleton without being identical to its referent.
    alloc.singletons.add(word)
    edge = store.parts.add(obj, word, truth_plus=1.)
    reader = ThoughtLTMCapability(store=lambda: store, equal=lambda *_: 0., tau_id=.9)
    def query():
        return reader.relation_evidence('equal', ('sym', obj), None,
                                       max_records=16, work=QueryWorkBudget(16))
    assert query()['frames'] == (), 'a DEF edge must not answer with the name of its object'
    assert store.part_row(edge)['readings'] == ('definition',)
    # The same ordered pair with an object singleton has the equality reading.
    alloc.symbol_types[word] = {'concept', 'object'}
    assert query()['frames'][0]['occurrence'] == ('sym', edge)
