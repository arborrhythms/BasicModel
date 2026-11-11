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
    cpu = {row: values[i % len(values)] for i, row in enumerate(cs.basic_percepts.index.values())}
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
    from Expectation import ActivationPredictor
    predictor=ActivationPredictor(3).to(device)
    keys=torch.tensor([[[1.,0.,.1],[1.,1.,.2],[0.,0.,1.]]],device=device,requires_grad=True)
    knowing=torch.zeros(1,3,2,device=device,requires_grad=True)
    context=torch.zeros(1,6,device=device,requires_grad=True)
    before=keys.detach().clone()
    predicted=predictor(keys,context,knowing).sigmoid()
    predicted.square().sum().backward()
    assert predicted.device.type==device and predicted.dtype==keys.dtype
    assert keys.grad is None and context.grad is None and knowing.grad is None
    torch.testing.assert_close(keys,before,rtol=0,atol=0)
