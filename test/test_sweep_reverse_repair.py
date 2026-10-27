"""A native percept inverse cannot use the raw-byte slab ratio as its width."""
from types import SimpleNamespace
import pytest
import torch
from Spaces import Space


@pytest.mark.parametrize('legacy_width', [0, 1])
def test_native_reverse_preserves_full_percept_field(legacy_width):
    # Two streams, three native events, eight content and two four-coordinate
    # percept bands. The preceding one-word shape is deliberately stale.
    event = torch.arange(2*3*16, dtype=torch.float32).reshape(2, 3, 16)
    target = SimpleNamespace(batch=None)
    space = SimpleNamespace(subspace=SimpleNamespace(getEncodedOutputSize=lambda: legacy_width),
        _percept_width=16, _pre_reshape_output=(1, 16), nOutputDim=16,
        nWhere=4, nWhen=4, _reverse_target=lambda value: target)
    carrier = SimpleNamespace(materialize=lambda: event, carrier_pure=True)
    actual = Space.reverseBegin(space, carrier, returnVectors=True)
    assert target.batch == 2 and space._reverse_carrier is carrier
    assert actual.shape == event.shape
    torch.testing.assert_close(actual, event, rtol=0, atol=0)
