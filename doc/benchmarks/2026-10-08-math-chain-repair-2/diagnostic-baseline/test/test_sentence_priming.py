"""Sentence priming is row-local and uses the whole native definition graph."""
import torch
from test_priming_energy import _chain_cs


def test_serial_reading_diffuses_over_existing_definitions(monkeypatch):
    cs, first, _ = _chain_cs()
    monkeypatch.setattr(cs, '_sparse_active', lambda: False)
    cs.prime_seen(torch.tensor([[3]]))
    surface = cs.prime_seen(torch.tensor([[3]]))
    assert surface.shape == (1, cs._priming_dim())
    assert surface[0, first] > 1


def test_batch_rows_never_prime_each_other():
    cs, first, _ = _chain_cs()
    surface = cs.prime_seen(torch.tensor([[3], [6]]))
    assert surface.shape == (2, cs._priming_dim())
    assert surface[0, 3] == 2 and surface[1, 3] == 1
    assert surface[0, 6] == 1 and surface[1, 6] == 2
    surface = cs.prime_seen(torch.tensor([[3], [6]]))
    assert surface[0, first] > 1 and surface[1, first] == 1


def test_no_edge_cap_and_no_scalar_host_walk(monkeypatch):
    from types import SimpleNamespace
    cs, _, _ = _chain_cs()
    count = 4100
    layer = SimpleNamespace(nnz=count, nOutput=8192, values=torch.ones(count),
        _indices=lambda device: (torch.arange(count, device=device), torch.arange(count, device=device)+1))
    store = SimpleNamespace(part_matrices=lambda: [layer])
    monkeypatch.setattr(cs, '_concept_allocator', SimpleNamespace(_layers={0: store}))
    src, dst, weight = cs._priming_edges()
    assert src.numel() == dst.numel() == weight.numel() == 2 * count
    assert torch.equal(src[:count], dst[count:])
