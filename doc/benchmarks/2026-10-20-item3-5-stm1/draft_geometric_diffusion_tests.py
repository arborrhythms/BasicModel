"""The two priming channels share energy and never write placement."""
from types import SimpleNamespace

import torch

from PartStore import PartStore
from Priming import diffuse, overlap_edges, part_edges


def owner(points, *, capacity=2):
    reads = []
    def lookup(indices):
        reads.extend(indices.tolist())
        return points[indices]
    return SimpleNamespace(
        similarity_codebook=SimpleNamespace(W=points, lookup_rows=lookup),
        _concept_allocator=SimpleNamespace(parts=PartStore(256)),
        _closed_clause_store=lambda: None,
        _csw_row_of=lambda symbol: symbol - 1 if 0 < symbol <= len(points) else None,
        concept_id_at_row=lambda row: row + 1 if 0 <= row < len(points) else None,
        stm=SimpleNamespace(capacity=capacity), reads=reads)


def test_graph_and_meet_share_outflow_without_code_feedback_or_cross_streams():
    points = torch.tensor([[1., 0., 0.], [.5, .5, 0.], [0., 0., 1.]], requires_grad=True)
    native = owner(points)
    native._concept_allocator.parts.add(1, 2, symbol=100)
    before_codes, before_rng = points.detach().clone(), torch.get_rng_state().clone()
    surface = torch.tensor([[2., 1., 1.], [1., 1., 4.]])
    diffuse(native, surface, part_edges(native), .3, torch.tensor([True, False]),
            torch.tensor([[0, 1], [2, -1]]))
    # A unit Part edge and a half-covered conceptual edge divide the same .3.
    torch.testing.assert_close(surface, torch.tensor([[1.7, 1.3, 1.], [1., 1., 4.]]))
    audit = native._last_priming_diffusion
    assert audit['degree'][0, 0] == 1.5
    torch.testing.assert_close(audit['before'], audit['after'], rtol=0, atol=1e-6)
    torch.testing.assert_close(points, before_codes, rtol=0, atol=0)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert points.grad is None and not surface.requires_grad


def test_meet_reads_reach_bank_through_bounded_part_walk_not_full_scan():
    points = torch.zeros(100, 3)
    points[:, 0] = torch.linspace(.1, 1., 100)
    native = owner(points)
    parts = native._concept_allocator.parts
    parts.refs[101:200] = torch.stack((torch.arange(1, 100), torch.arange(2, 101)), -1)
    parts.count.fill_(99)
    surface = torch.ones(1, 100)
    surface[0, 0] = 2.
    edges, audits = overlap_edges(native, surface, torch.tensor([True]), torch.tensor([[0]]))
    audit = audits[0]
    assert audit['part_reads'] <= 4 * native.stm.capacity
    assert 0 < audit['comparisons'] < 99
    assert audit['edges'] <= native.stm.capacity
    assert audit['frontier_truncated']
    assert len(native.reads) < 100 and 99 not in native.reads
    assert len(edges[0]) > 0 and bool((edges[3] > 0).all())


def test_overlap_reads_recent_stm_symbols_in_their_own_stream():
    native = owner(torch.tensor([[1., 0.], [.5, .5], [.25, .75]]))
    native._closed_clause_store = lambda: SimpleNamespace(location=lambda _: None)
    native._model = SimpleNamespace(symbolSpace=SimpleNamespace(expectation=SimpleNamespace(
        _inter_context=[[], []], situation_references=lambda b: (
            SimpleNamespace(symbol_ids=(b + 2,), activations=(), row_id=-1),))))
    surface = torch.tensor([[2., 1., 1.], [2., 1., 1.]])
    (batch, source, target, weight), _ = overlap_edges(
        native, surface, torch.tensor([True, True]), torch.tensor([[0], [0]]))
    assert list(zip(batch.tolist(), source.tolist(), target.tolist())) == [(0, 0, 1), (1, 0, 2)]
    torch.testing.assert_close(weight, torch.tensor([.5, .25]))


def test_overlap_uses_meaning_even_when_the_forms_suggest_the_opposite_neighbour():
    native = owner(torch.tensor([[1.,0.,1.,0.],
                                 [1.,0.,0.,1.],
                                 [0.,1.,1.,0.]]))
    native.similarity_codebook.mereology = SimpleNamespace(
        percept_width=2, percept_event_width=2)
    surface = torch.tensor([[2.,1.,1.]])
    (batch, source, target, weight), _ = overlap_edges(
        native, surface, torch.tensor([True]), torch.tensor([[0,1,2]]))
    assert list(zip(batch.tolist(), source.tolist(), target.tolist())) == [(0,0,2)]
    torch.testing.assert_close(weight, torch.ones(1), rtol=0, atol=0)


def test_part_neighbour_index_rebuilds_after_forgetting_and_load():
    parts = PartStore(32)
    parts.add(2, 3, symbol=10)
    assert 10 in tuple(parts.neighbours(2))
    restored = PartStore(1)
    restored.load_state_dict(parts.state_dict())
    assert set(restored.neighbours(2)) == {3, 10}
    restored.remove(2, 3)
    assert tuple(restored.neighbours(2)) == ()


def test_hated_neighbours_cannot_overdraw_a_hub_and_create_energy_at_the_floor():
    native = owner(torch.zeros(41, 2))
    surface = torch.zeros(1, 41)
    surface[0, -1] = 1.
    edges = (torch.arange(40), torch.full((40,), 40), torch.ones(40))
    diffuse(native, surface, edges, .05, torch.tensor([True]), torch.tensor([[0]]))
    assert bool((surface >= 0).all())
    torch.testing.assert_close(surface.sum(), torch.tensor(1.), rtol=0, atol=1e-6)
    assert 0 < native._last_priming_diffusion['budget_fraction'][0] < 1
