"""Sentence priming is row-local and uses the placed conceptual neighbourhood."""
import torch
from test_priming_energy import _chain_cs


def test_serial_reading_diffuses_over_placed_rows(monkeypatch):
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
    assert surface[0, 3] == 1.5 and surface[1, 3] == 1
    assert surface[0, 6] == 1 and surface[1, 6] == 1.5
    surface = cs.prime_seen(torch.tensor([[3], [6]]))
    assert surface[0, first] > 1 and surface[1, first] == 1


def test_part_graph_uses_the_constructed_relation_capacity(monkeypatch):
    from types import SimpleNamespace
    from LTM1 import LTM1
    cs, _, _ = _chain_cs()
    count = 4100
    parts = LTM1(capacity=count + 2)
    parts.refs[1:count+1] = torch.stack((torch.arange(1, count+1), torch.arange(2, count+2)), -1)
    parts.ref_kinds[1:count+1].fill_(parts.SYMBOL)
    parts.count.fill_(count)
    monkeypatch.setattr(cs, '_concept_allocator', SimpleNamespace(parts=parts, relate_idx={('part-base', i): (2, i-1) for i in range(2,count+2)}))
    monkeypatch.setattr(cs, '_csw_row_of', lambda symbol: symbol - 1)
    from Priming import part_edges
    src, dst, weight = part_edges(cs)
    assert src.numel() == dst.numel() == weight.numel() == count
    assert torch.equal(src+1, dst)
