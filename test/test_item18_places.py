"""Round A's place, level and bracket certificates (65--67 and 66a)."""
import torch


def test_level_chart_uses_supported_precision_on_mps():
    import pytest
    from NumericPercepts import feeling, number
    if not torch.backends.mps.is_available():
        pytest.skip('MPS is unavailable on this host')
    values = torch.tensor([0., 1., 2., 10., float('inf')], device='mps')
    rebuilt = number(feeling(values))
    torch.testing.assert_close(rebuilt, values, rtol=1e-6, atol=1e-6)


def test_65_67_image_uses_the_same_reader_and_recovers_every_pixel(monkeypatch):
    import time
    from NumericPercepts import NumericPercepts
    from recon_bench import _build_model
    from image_corpus import load
    from corpus_audit import peak_bytes
    images, _ = load('shapes', 'test')
    model, *_ = _build_model('data/MM_image.xml')
    assert isinstance(model.numeric_percepts, NumericPercepts)
    assert model.numeric_percepts.dimensions == 784
    assert model.inputSpace.inputShape == [28, 28]
    sample = images[:1]
    # An image is put back once its places are known: it is presented as
    # often as nominates a place's row.
    for _ in range(model._concept_owner().concept_mint_count):
        model.forward(sample)
    calls = []
    read = model.outputSpace.read_field
    monkeypatch.setattr(model.outputSpace, 'read_field', lambda *args: (calls.append(True), read(*args))[1])
    started = time.perf_counter()
    model.forward(sample)
    forward_seconds = time.perf_counter() - started
    forward_peak_bytes = peak_bytes()
    understood = model.understand(sample)
    started = time.perf_counter()
    rebuilt, _ = model.reverseReconstruct(understood)
    reverse_seconds = time.perf_counter() - started
    reverse_peak_bytes = peak_bytes()
    torch.testing.assert_close(rebuilt, sample, rtol=0, atol=1e-6)
    assert len(calls) == 2
    field = understood.reconstruction_carriers['field']
    base, end = model.where_registry.slices['numeric']
    assert end - base == 784
    addresses = field.where[0, :784, 0]
    # An image is a box: x the column, y the row, its addresses row by row.
    registry = model.where_registry
    place = registry.place(addresses)
    decoded = registry.decode(registry.encoding.encode_place(place)) - base
    assert decoded.tolist() == list(range(784))
    local = place - torch.tensor(registry.boxes['numeric'][0], device=place.device)
    assert local[:, 1].tolist() == [row for row in range(28) for _ in range(28)]
    assert local[:, 0].tolist() == list(range(28)) * 28
    assert (local[:, 2] == 0).all()
    text, scalar = (model.where_registry.slices[key] for key in ('text', 'scalar'))
    assert text[1] <= base < end <= scalar[0] < scalar[1]
    print(dict(forward_seconds=forward_seconds, reverse_seconds=reverse_seconds,
               forward_peak_bytes=forward_peak_bytes, reverse_peak_bytes=reverse_peak_bytes,
               places=784, percept_width=model._concept_owner().basic_percepts.width))
    xor, *_ = _build_model('data/MM_xor.xml')
    assert isinstance(xor.numeric_percepts, NumericPercepts)
    assert xor.numeric_percepts.dimensions == 2
    xor_read = xor.outputSpace.read_field
    xor_calls = []
    monkeypatch.setattr(xor.outputSpace, 'read_field',
        lambda *args: (xor_calls.append(True), xor_read(*args))[1])
    xor.forward(torch.tensor([[[0.], [1.]]]))
    assert xor_calls == [True]
    xor_range = xor.where_registry.slices['numeric']
    assert len({text, (base, end), xor_range}) == 3
    assert xor_range[1] - xor_range[0] == 2


def test_66_numeric_certificate_on_level_percepts():
    from numeric_xor import certificate
    certificate()






def test_level_percepts_have_native_bounds_and_distinct_places():
    """The same value at two places is one value's row: its code the bins at
    or below it, in order; its places are told by their wires."""
    from recon_bench import _build_model
    model, *_ = _build_model('data/MM_xor.xml')
    field = model.numeric_percepts.read(torch.ones(1, 2, 1))
    basic = model._concept_owner().basic_percepts
    assert not hasattr(basic, 'numeric_dimension')
    percepts = field.percepts[0].tolist()
    assert len(set(percepts)) == 1
    for axis, percept in enumerate(percepts):
        parts, wholes = basic.occurrences[percept]
        assert parts and wholes
        # Rule 55: in the parts half the bins at or below the value's
        # level, in order; in the wholes half its level's group; one table
        # of values for every place.
        felt = torch.tensor(1.)
        torch.testing.assert_close(basic.code(percept), torch.cat((
            basic.value_code(felt, basic.identity_width),
            basic.value_whole(felt))))
        torch.testing.assert_close(basic.felt([percept])[0], felt.to(basic.value_codes))
    assert basic.width == model.perceptualSpace.nWhat
