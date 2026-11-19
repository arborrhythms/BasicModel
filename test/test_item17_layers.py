"""The Pi and the Sigma of signed concepts and unsigned rows, and the old
classes beside them."""
import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.assessment_rung(1)

ROOT = Path(__file__).resolve().parents[1]
BEFORE = '1598a9bd46'
OLD_CLASSES = ('SigmaLayer', 'PiLayer', 'PiLayer2', 'SigmaLayer2')


def _class_sources(text):
    tree = ast.parse(text)
    return {node.name: ast.get_source_segment(text, node) for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in OLD_CLASSES}


@pytest.fixture(scope='module')
def before(tmp_path_factory):
    text = subprocess.run(['git', 'show', f'{BEFORE}:bin/Layers.py'], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout
    path = tmp_path_factory.mktemp('before')/'Layers_before.py'
    path.write_text(text)
    spec = importlib.util.spec_from_file_location('Layers_before', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['Layers_before'] = module
    spec.loader.exec_module(module)
    yield text, module
    sys.modules.pop('Layers_before', None)


def _signed_range(*shape):
    x = torch.rand(*shape)*2 - 1
    x.view(-1)[:4] = torch.tensor([-1., 1., 0., -1.])
    return x


def _unit_range(*shape):
    m = torch.rand(*shape)
    m.view(-1)[:3] = torch.tensor([0., 1., 1e-9])
    return m


# Certificate 1: the old classes are unchanged.

def test_the_old_classes_have_the_same_source(before):
    text, _ = before
    old = _class_sources(text)
    new = _class_sources((ROOT/'bin/Layers.py').read_text())
    assert set(old) == set(OLD_CLASSES)
    for name in OLD_CLASSES:
        assert new[name] == old[name], name


@pytest.mark.parametrize('name, build, domain', [
    ('SigmaLayer', lambda m: m.SigmaLayer(6, 6), _signed_range),
    ('SigmaLayer', lambda m: m.SigmaLayer(6, 6, nonlinear=False), _signed_range),
    ('PiLayer', lambda m: m.PiLayer(6, 6, nonlinear=True), _signed_range),
    ('PiLayer2', lambda m: m.PiLayer2(6, 6), _unit_range),
    ('SigmaLayer2', lambda m: m.SigmaLayer2(6, 6), _unit_range),
])
def test_the_old_classes_give_the_same_values(before, name, build, domain):
    import Layers
    _, old_module = before
    torch.manual_seed(0)
    new = build(Layers)
    torch.manual_seed(0)
    old = build(old_module)
    old.load_state_dict(new.state_dict())
    with torch.no_grad():
        for p in new.parameters():
            p.add_(torch.randn_like(p)*.3)
        old.load_state_dict(new.state_dict())
    x = domain(32, 6)
    torch.testing.assert_close(new(x), old(x), rtol=0, atol=0)
    y = new(x).detach()
    try:
        back = new.reverse(y)
    except AttributeError:
        with pytest.raises(AttributeError):
            old.reverse(y)
        return
    torch.testing.assert_close(back, old.reverse(y), rtol=0, atol=0)


# Certificate 2: the Pi.

def _pi(n=5, scale=None):
    from Layers import PiLayer3
    pi = PiLayer3(n, n)
    if scale is not None:
        with torch.no_grad():
            for layer in pi.lane_layers:
                for p in (layer.raw_L, layer.raw_U, layer.d, layer.biasWeight):
                    p.copy_(torch.randn_like(p)*scale)
    return pi


def test_pi_gives_pi2_values_on_its_domain():
    from Layers import PiLayer2
    torch.manual_seed(1)
    pi = _pi(6, scale=2.)
    m = _unit_range(40, 2, 6)
    out = pi(m)
    for lane, layer in enumerate(pi.lane_layers):
        old = PiLayer2(6, 6)
        old.layer.load_state_dict(layer.state_dict())
        torch.testing.assert_close(out[:, lane], old(m[:, lane]), rtol=0, atol=0)
        torch.testing.assert_close(pi.reverse(out)[:, lane], old.reverse(out[:, lane]),
                                   rtol=0, atol=0)


def test_a_row_requires_its_members():
    from Layers import EPS_LOG
    pi = _pi(3)
    with torch.no_grad():
        for layer in pi.lane_layers:
            layer.raw_L.fill_(-30.)
            layer.raw_U.fill_(-30.)
            layer.d.fill_(-30.)
            layer.biasWeight.fill_(-30.)
            # L[1, 0] = softplus(log(e - 1)) = 1: row 0 reads members 0 and 1.
            layer.raw_L[1, 0] = torch.log(torch.expm1(torch.tensor(1.)))
    m = torch.tensor([[[.8, .6, .9], [.5, .7, .3]]])
    out = pi(m)
    torch.testing.assert_close(out[..., 0], m[..., 0]*m[..., 1], rtol=1e-4, atol=1e-6)
    absent = m.clone()
    absent[0, :, 1] = 0.
    assert (pi(absent)[..., 0] <= EPS_LOG).all()
    assert (out <= m + 1e-7).all()
    print(dict(both_present=out[0, :, 0].tolist(), product=(m[0, :, 0]*m[0, :, 1]).tolist(),
               one_absent=pi(absent)[0, :, 0].tolist()))


@pytest.mark.parametrize('scale', [0., 3., 30.])
def test_the_pi_is_bounded_over_the_whole_range(scale):
    from Layers import PiLayer3, EPS_LOG
    torch.manual_seed(2)
    pi = _pi(8, scale=scale)
    m = _unit_range(64, 2, 8)
    out = pi(m)
    assert torch.isfinite(out).all()
    assert (out >= 0).all() and (out <= 1).all()
    assert (out <= m.clamp(EPS_LOG, 1) + 1e-7).all()
    free = PiLayer3(8, 5, free=True)
    with torch.no_grad():
        free.raw.mul_(1 + scale*10)
    out = free(m)
    assert torch.isfinite(out).all() and (out >= 0).all() and (out <= 1).all()


def test_the_pi_has_an_exact_way_back():
    from Layers import EPS_LOG
    torch.manual_seed(3)
    pi = _pi(8)
    with torch.no_grad():
        for layer in pi.lane_layers:
            for p in (layer.raw_L, layer.raw_U, layer.d):
                p.add_(torch.randn_like(p)*.5)
    m = torch.rand(64, 2, 8)*.95 + .05
    m.view(-1)[:2] = 1.
    torch.testing.assert_close(pi.reverse(pi(m)), m, rtol=0, atol=1e-6)
    corner = torch.full((1, 2, 8), EPS_LOG)
    torch.testing.assert_close(_pi(8).reverse(_pi(8)(corner)), corner, rtol=1e-3, atol=0)


def test_a_row_has_no_weight_outside_its_own_lane():
    from Layers import PiLayer3
    torch.manual_seed(4)
    for pi in (_pi(5, scale=2.), PiLayer3(5, 4, free=True)):
        m = torch.rand(3, 2, 5, requires_grad=True)
        out = pi(m)
        for lane in (0, 1):
            other = 1 - lane
            grad, = torch.autograd.grad(out[:, lane].sum(), m, retain_graph=True)
            assert torch.equal(grad[:, other], torch.zeros_like(grad[:, other]))
            assert (grad[:, lane] != 0).any()
        moved = m.detach().clone()
        moved[:, 1] = torch.rand(3, 5)
        torch.testing.assert_close(pi(moved)[:, 0], out[:, 0].detach(), rtol=0, atol=0)


def test_a_membership_of_one_passes_its_derivative_and_one_below_the_floor_none():
    from Layers import PiLayer2, PiLayer3
    for pi in (_pi(4), PiLayer3(4, 4, free=True)):
        m = torch.tensor([[[1., 1e-8, .5, 1.], [.3, 1., 1e-9, .7]]], requires_grad=True)
        grad, = torch.autograd.grad(pi(m).sum(), m)
        assert grad[0, 0, 0] != 0 and grad[0, 0, 3] != 0 and grad[0, 1, 1] != 0
        assert grad[0, 0, 1] == 0 and grad[0, 1, 2] == 0
    old = PiLayer2(4, 4)
    m = torch.tensor([[1., .5, .5, .5]], requires_grad=True)
    grad, = torch.autograd.grad(old(m).sum(), m)
    print(dict(pi2_derivative_at_one=float(grad[0, 0])))


# Certificate 3: the Sigma.

def test_a_concept_is_one_signed_number_and_its_lanes_are_its_parts():
    from Layers import SigmaLayer3
    lanes = SigmaLayer3.lanes(torch.tensor([.5, -.5, 0.]))
    torch.testing.assert_close(lanes, torch.tensor([[.5, 0., 0.], [0., .5, 0.]]))
    torch.manual_seed(5)
    sigma = SigmaLayer3(7, 9)
    with torch.no_grad():
        sigma.weight.sub_(.3)
    y = sigma(_signed_range(50, 7))
    assert y.shape == (50, 9)
    lanes = SigmaLayer3.lanes(y)
    assert (lanes >= 0).all()
    assert (torch.minimum(lanes[:, 0], lanes[:, 1]) == 0).all()
    torch.testing.assert_close(lanes[:, 0] - lanes[:, 1], y, rtol=0, atol=0)


def test_on_the_way_back_minus_half_reaches_the_against_lane():
    from Layers import SigmaLayer3
    torch.manual_seed(6)
    sigma = SigmaLayer3(4, 6)
    x = torch.tensor([[-.5, .3, 0., .8], [.5, -.2, -.9, 0.]])
    back = sigma.reverse(sigma(x))
    assert back.shape == (2, 2, 4)
    torch.testing.assert_close(back[0, :, 0], torch.tensor([0., .5]), rtol=0, atol=1e-5)
    torch.testing.assert_close(back[1, :, 0], torch.tensor([.5, 0.]), rtol=0, atol=1e-5)
    torch.testing.assert_close(back[:, 0] - back[:, 1], x, rtol=0, atol=1e-5)
    assert (back >= 0).all()
    assert (torch.minimum(back[:, 0], back[:, 1]) == 0).all()
    with pytest.raises(ValueError, match='too few concepts'):
        SigmaLayer3(6, 4).reverse(torch.zeros(1, 4))


@pytest.mark.parametrize('scale', [1., 100., 1e4])
def test_the_sigma_is_bounded_over_the_whole_range(scale):
    from Layers import SigmaLayer3
    torch.manual_seed(7)
    sigma = SigmaLayer3(8, 5)
    with torch.no_grad():
        sigma.weight.copy_(torch.randn_like(sigma.weight)*scale)
    y = sigma(_signed_range(64, 8))
    assert torch.isfinite(y).all() and (y.abs() <= 1).all()


def test_the_sigma_has_no_bias_and_an_entry_is_for_or_not():
    from Layers import SigmaLayer3
    torch.manual_seed(8)
    sigma = SigmaLayer3(6, 4)
    assert [name for name, _ in sigma.named_parameters()] == ['weight']
    assert (sigma.weight >= 0).all()
    assert torch.equal(sigma(torch.zeros(1, 6)), torch.zeros(1, 4))
    with torch.no_grad():
        sigma.weight.sub_(.5)
    assert ((sigma.for_entries() > 0) & (sigma.not_entries() > 0)).sum() == 0
    torch.testing.assert_close(sigma.for_entries() - sigma.not_entries(), sigma.weight,
                               rtol=0, atol=0)
    assert torch.equal(SigmaLayer3(6, 4, scale=0.).weight, torch.zeros(6, 4))
