"""A scalar form audit must preserve native device/dtype and its exact report."""
import pytest
import torch
from SymbolCentroid import place

@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_form_audit_reports_cpu_double_without_moving_forms(device):
    values = torch.tensor([[1., 0., .1], [1., 1., .2], [0., 0., 1.]], device='cpu')
    parts = {0: {1}, 1: {1, 2}, 2: {3}}
    _, expected = place(dict(enumerate(values)), parts)
    forms = {row: value.to(device) for row, value in enumerate(values)}
    placed, actual = place(forms, parts)
    assert actual == expected
    assert actual['moved'] == 0 and actual['identity_recovered']
    for row in forms:
        assert placed[row].device == forms[row].device
        assert placed[row].dtype == forms[row].dtype
        torch.testing.assert_close(placed[row].cpu(), values[row], rtol=0, atol=0)
        torch.testing.assert_close(forms[row].cpu(), values[row], rtol=0, atol=0)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_priming_energy_audit_keeps_device_values_and_double_totals(device):
    from types import SimpleNamespace
    from Priming import diffuse
    surface = torch.tensor([[2., 1., 0.], [1., 1., 4.]], device=device)
    before = surface.cpu().double().sum(-1)
    edges = (torch.tensor([0, 1], device=device), torch.tensor([1, 0], device=device),
             torch.ones(2, device=device))
    owner = SimpleNamespace()
    diffuse(owner, surface, edges, .25, torch.tensor([True, False], device=device))
    torch.testing.assert_close(surface.cpu(), torch.tensor([[1.75, 1.25, 0.], [1., 1., 4.]],
        device='cpu'), rtol=0, atol=0)
    assert surface.device.type == device
    audit = owner._last_priming_diffusion
    torch.testing.assert_close(audit['before'], before, rtol=0, atol=0)
    torch.testing.assert_close(audit['after'], before, rtol=0, atol=0)
    assert audit['before'].dtype == audit['after'].dtype == torch.float64
    assert audit['degree'].device == audit['budget_fraction'].device == surface.device
