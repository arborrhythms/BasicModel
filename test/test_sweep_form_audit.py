"""A scalar form audit must preserve native device/dtype and its exact report."""
import pytest
import torch

@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_form_audit_reports_cpu_double_without_moving_forms(device):
    values = torch.tensor([[1., 0., .1], [1., 1., .2], [0., 0., 1.]], device='cpu')
    from test_review13_subspaces import fixture, word
    cs, cb, _, _ = fixture()
    rows = [word(cs, [(4, 1.)]), word(cs, [(4, 1.), (7, 1.)]), word(cs, [(8, 1.)])]
    cpu = dict(zip(rows, values))
    expected = cb.mereology.containment_audit(before=cpu, after=cpu)
    forms = {row: value.to(device) for row, value in cpu.items()}
    actual = cb.mereology.containment_audit(before=forms, after=forms)
    assert actual == expected
    for row in forms:
        assert forms[row].device.type == device
        assert forms[row].dtype == values.dtype
        torch.testing.assert_close(forms[row].cpu(), cpu[row], rtol=0, atol=0)



@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_priming_energy_audit_keeps_device_values_and_double_totals(device):
    from types import SimpleNamespace
    from Priming import propagate
    from test_priming_diffusion import owner as fixture_owner, directions
    surface = torch.tensor([[2., 1., 0.], [1., 1., 4.]], device=device)
    before = surface.cpu().double().sum(-1)
    owner = fixture_owner(directions([0,10,80],dtype=torch.float32).to(device), spread=.25)
    propagate(owner, surface, torch.tensor([True, False], device=device))
    torch.testing.assert_close(surface.cpu(), torch.tensor([[1.75, 1.25, 0.], [1., 1., 4.]],
        device='cpu'), rtol=0, atol=0)
    assert surface.device.type == device
    audit = owner._last_priming_diffusion
    torch.testing.assert_close(audit['before'], before, rtol=0, atol=0)
    torch.testing.assert_close(audit['after'], before, rtol=0, atol=0)
    assert audit['before'].dtype == audit['after'].dtype == torch.float64
    assert audit['receivers'].device == audit['budget_fraction'].device == surface.device
