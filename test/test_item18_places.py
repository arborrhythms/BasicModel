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
    assert not model._concept_owner().conceptual_pi
    calls = []
    read = model.outputSpace.read_field
    monkeypatch.setattr(model.outputSpace, 'read_field', lambda *args: (calls.append(True), read(*args))[1])
    sample = images[:1]
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
    encoded = model.where_encoding.encode(addresses)
    decoded = model.where_encoding.decode_index(encoded) - base
    assert decoded.div(28, rounding_mode='floor').tolist() == [row for row in range(28) for _ in range(28)]
    assert decoded.remainder(28).tolist() == list(range(28)) * 28
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


def test_66a_numeric_definition_reads_the_whole_bracket():
    from recon_bench import _build_model
    model, *_ = _build_model('data/MM_xor.xml')
    values = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).unsqueeze(-1)
    owner = model._concept_owner()
    model.train()
    for _ in range(owner.concept_mint_count + 1):
        model.forward(values)
    field = model._numeric_field
    assert field.evidence.shape[2] == 3
    start, end = model.where_registry.slices['numeric']
    assert end - start == 2
    assert field.where[0].tolist() == [[start, start + 1], [start + 1, end], [start, end]]
    matrix = owner._concept_allocator.layer().conjunctive
    assert matrix.nnz == 2
    parent = owner.concept_id_at_row(matrix._rows[0])
    evidence = field.evidence[field.concept_ids == parent][0]
    assert not evidence[:, :2].any()
    torch.testing.assert_close(evidence[:, 2, 0], torch.tensor([0., 0., 0., 1.]), atol=0, rtol=0)
    assert not evidence[..., 1].any()
    for axis, row in enumerate(model.numeric_percepts.rows):
        value = field.evidence[field.concept_ids == owner.concept_id_at_row(row)][0]
        torch.testing.assert_close(value[:, axis, 0], values[:, axis, 0], atol=0, rtol=0)
    torch.testing.assert_close(model.numeric_percepts.reverse(field), values, atol=1e-6, rtol=0)


def test_bracket_closure_joins_only_contained_readings_on_both_lanes():
    from FieldProjection import close_bracket, close
    from FieldLogic import reduce
    from Understanding import ConceptualField
    from test_cs_sparse_weights import _cs, _mint_row
    owner = _cs(nS=16, order=2)
    a, b, parent = 11, 12, 13
    rows = [_mint_row(owner, 0, cid) for cid in (a, b, parent)]
    owner.add_concept_edge(rows[2], rows[0], weight=1., conjunctive=True)
    owner.add_concept_edge(rows[2], rows[1], weight=1., conjunctive=True)
    evidence = torch.tensor([[[[.4, .7], [.8, .2], [1., 1.]]],
                             [[[.6, .5], [.3, .9], [1., 1.]]],
                             [[[0., 0.], [0., 0.], [0., 0.]]]])
    where = torch.tensor([[[3, 4], [4, 5], [5, 6]]])
    field = ConceptualField(torch.tensor([a, b, parent]), evidence, where, torch.zeros_like(where), None)
    result = close_bracket(owner, field, torch.tensor([[3, 5]]))
    joined = reduce(evidence[:, :, :2], operation='union', dim=2)
    expected = close(owner, rows, joined[:, :, None])
    torch.testing.assert_close(result.evidence[:, :, -1:], expected, atol=0, rtol=0)
    torch.testing.assert_close(result.evidence[:, :, :-1], evidence, atol=0, rtol=0)
    assert result.where[0, -1].tolist() == [3, 5]


def test_level_percepts_have_native_bounds_and_distinct_places():
    from recon_bench import _build_model
    from NumericPercepts import feeling
    model, *_ = _build_model('data/MM_xor.xml')
    field = model.numeric_percepts.read(torch.ones(1, 2, 1))
    basic = model._concept_owner().basic_percepts
    assert not hasattr(basic, 'numeric_dimension')
    percepts = field.percepts[0].tolist()
    assert len(set(percepts)) == 2
    for axis, percept in enumerate(percepts):
        parts, wholes = basic.occurrences[percept]
        assert parts and wholes
        assert basic.places[percept] == model.where_registry.slices['numeric'][0] + axis
        torch.testing.assert_close(basic.part_codes(parts).amax(), feeling(torch.tensor(1.)))
        torch.testing.assert_close(basic.whole_codes(wholes).amax(), feeling(torch.tensor(1.)))
    assert basic.width == model.perceptualSpace.nWhat
