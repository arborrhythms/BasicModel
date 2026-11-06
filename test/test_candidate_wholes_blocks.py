"""§50 item105: candidate property minima are exact and bounded by 256 rows."""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from FormCentroids import CandidateWholes


def fixture(properties, dtype):
    forms = torch.tensor([[[0., 0., 1., 0.], [0., 1., 0., 0.],
                           [1., 0., 0., 0.]]], dtype=dtype).expand(2, -1, -1).clone()
    support = torch.zeros(2, 3, 256, dtype=torch.bool)
    support[:, 0, [1, 7]] = True
    support[:, 1, [2, 7]] = True
    # An unspelled row supplies no primitive support.
    props = (torch.arange(properties * 256).reshape(properties, 256) % 31).to(dtype) / 32
    bank = CandidateWholes(forms, support, torch.ones(2, 3, dtype=torch.bool),
                           props, torch.zeros(properties, 4, dtype=dtype))
    left, right = forms, forms[:, [1, 0, 2]]
    union = support | support[:, [1, 0, 2]]
    expected = torch.where(union[..., None, :], props, 1.).amin(-1) * union.any(-1)[..., None]
    return bank, left, right, expected


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('properties', (0, 9, 256, 4096))
@pytest.mark.parametrize('loop', (False, True))
def test_blocked_candidate_read_is_bitwise_dense_eager_and_inductor(properties, dtype, loop):
    # Each parametrization is a separate constructed bank, not a resized run.
    torch._dynamo.reset()
    bank, left, right, expected = fixture(properties, dtype)
    def read(a, b):
        if not loop:
            return bank._weights(a, b)
        def condition(index, value):
            return index < 2
        def body(index, value):
            return index + 1, bank._weights(a, b)
        initial = a.new_zeros((*a.shape[:-1], properties))
        return torch.while_loop(condition, body, (torch.zeros((), dtype=torch.int64), initial))[1]
    torch.testing.assert_close(read(left, right), expected, rtol=0, atol=0)
    compiled = torch.compile(read, backend='inductor', fullgraph=True, dynamic=False)
    torch.testing.assert_close(compiled(left, right), expected, rtol=0, atol=0)


def test_candidate_mask_temporary_has_at_most_256_properties():
    bank, left, right, expected = fixture(4096, torch.float32)
    widths = []
    class Inspect(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            if function._schema.name == 'aten::where' and result.ndim == 4:
                widths.append(result.shape[-2])
            return result
    with Inspect():
        actual = bank._weights(left, right)
    assert widths and max(widths) <= 256
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
