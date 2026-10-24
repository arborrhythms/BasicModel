"""Surface inverses on the paired carrier, with nested leaf counts."""
import pytest
import torch

from test_meaning_kernels import paired


@pytest.mark.parametrize('compiled', (False, True))
@pytest.mark.parametrize('postposed', (False, True))
def test_surface_nested_operand_and_transpose_reverse_exactly(paired, compiled, postposed):
    from Language import SurfaceLayer, TransposeLayer
    from MeaningKernels import centroid
    layer = SurfaceLayer(12, 12).double()
    with torch.no_grad():
        layer.meaning_kernel.log_scale.fill_(.04)
        layer.meaning_kernel.bias.fill_(.01)
        layer.meaning_kernel.context.copy_(torch.eye(4, dtype=torch.float64)*.03)
    leaves = torch.linspace(.15, .7, 48, dtype=torch.float64).reshape(4, 12)
    content = torch.cat((centroid(tuple(leaves[:3, :4])), leaves[2, 4:]))
    marker = leaves[3]
    surface = (content, marker) if postposed else (marker, content)
    transpose = TransposeLayer()
    operands = transpose.compose(*surface) if postposed else surface
    def roundtrip(left, right):
        parent = layer.compose_blocks(layer.compose, left, right, counts=(1, 3))
        right_known = layer.meaning_head_role == 0
        witness = right if right_known else left
        recovered = layer.reverse_blocks(parent, witness, counts=(1, 3), reference_right=right_known)
        return parent, recovered
    run = torch.compile(roundtrip, backend='eager', fullgraph=True) if compiled else roundtrip
    parent, recovered = run(*operands)
    torch.testing.assert_close(parent[:4], leaves[:, :4].mean(0), atol=1e-12, rtol=0)
    for actual, expected in zip(recovered, operands):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    result = transpose.reverse(*recovered) if postposed else recovered
    for actual, expected in zip(result, surface):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize('compiled', (False, True))
def test_surface_absent_marker_uses_one_kernel_and_emits_one_operand(paired, compiled):
    from Language import SurfaceLayer, SurfaceUnaryLayer
    base = SurfaceLayer(12, 12).double()
    unary = SurfaceUnaryLayer(base).double()
    assert unary.meaning_kernel is base.meaning_kernel
    with torch.no_grad():
        base.meaning_kernel.log_scale.fill_(.03)
        base.meaning_kernel.bias.fill_(.02)
    value = torch.linspace(.1, .8, 24, dtype=torch.float64).reshape(2, 12)
    run = lambda point: unary.reverse(unary.compose(point))
    if compiled:
        run = torch.compile(run, backend='eager', fullgraph=True)
    actual = run(value)
    assert actual.shape == value.shape
    torch.testing.assert_close(actual, value, atol=1e-12, rtol=0)
