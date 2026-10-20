"""Recorded Part structure diffuses energy without reading code placement."""
from types import SimpleNamespace

import torch

from PartStore import PartStore
from Priming import diffuse, part_edges


def owner(points):
    def lookup(_):
        raise AssertionError('Part diffusion must not read code placement')
    return SimpleNamespace(
        similarity_codebook=SimpleNamespace(W=points, lookup_rows=lookup),
        _concept_allocator=SimpleNamespace(parts=PartStore(256)),
        _closed_clause_store=lambda: None,
        _csw_row_of=lambda symbol: symbol - 1 if 0 < symbol <= len(points) else None)


def test_part_outflow_preserves_energy_codes_rng_and_inactive_streams():
    points = torch.tensor([[1., 0., 0.], [.5, .5, 0.], [0., 0., 1.]], requires_grad=True)
    native = owner(points)
    native._concept_allocator.parts.add(1, 2, symbol=100)
    before_codes, before_rng = points.detach().clone(), torch.get_rng_state().clone()
    surface = torch.tensor([[2., 1., 1.], [1., 1., 4.]])
    diffuse(native, surface, part_edges(native), .3, torch.tensor([True, False]))
    torch.testing.assert_close(surface, torch.tensor([[1.7, 1.3, 1.], [1., 1., 4.]]))
    audit = native._last_priming_diffusion
    assert audit['degree'][0, 0] == 1
    torch.testing.assert_close(audit['before'], audit['after'], rtol=0, atol=1e-6)
    torch.testing.assert_close(points, before_codes, rtol=0, atol=0)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert points.grad is None and not surface.requires_grad


def test_part_graph_follows_native_addresses_after_forgetting_and_load():
    native = owner(torch.zeros(3, 2))
    parts = native._concept_allocator.parts
    parts.add(1, 2, symbol=100)
    parts.add(2, 3, symbol=101)
    restored = PartStore(1)
    restored.load_state_dict(parts.state_dict())
    native._concept_allocator.parts = restored
    restored.remove(1, 2)
    source, target, weight = part_edges(native)
    assert list(zip(source.tolist(), target.tolist())) == [(1, 2), (2, 1)]
    torch.testing.assert_close(weight, torch.ones(2), rtol=0, atol=0)


def test_hated_neighbours_cannot_overdraw_a_hub_and_create_energy_at_the_floor():
    native = owner(torch.zeros(41, 2))
    surface = torch.zeros(1, 41)
    surface[0, -1] = 1.
    edges = (torch.arange(40), torch.full((40,), 40), torch.ones(40))
    diffuse(native, surface, edges, .05, torch.tensor([True]))
    assert bool((surface >= 0).all())
    torch.testing.assert_close(surface.sum(), torch.tensor(1.), rtol=0, atol=1e-6)
    assert 0 < native._last_priming_diffusion['budget_fraction'][0] < 1
