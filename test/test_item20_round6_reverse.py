"""The order-zero decoder's native representation and grammar-free boundary."""
from types import SimpleNamespace as NS
import pytest
import torch

from FieldReverse import NativeActivity, NativeDecode
from Layers import RadixLayer
from Understanding import ConceptualField


def test_vectorized_native_decode_matches_greedy_walk_without_resizing():
    radix = RadixLayer(4, initial_cap=16)
    for raw in (b'a', b'b', b'abc', b'de'):
        radix.insert(raw)
    radix.activity_decoder = NativeDecode(16, 40, 8)
    pointers = tuple(value.data_ptr() for value in radix.activity_decoder.buffers())
    activity = torch.rand(3, 40, len(radix))
    activity[:, 7::9] = 0
    strength, rows = activity.max(-1)
    expected = torch.zeros(3, 40, dtype=torch.long)
    for batch in range(3):
        position = 0
        while position < 40:
            raw = radix.bytes_for(int(rows[batch, position])) if strength[batch, position] > 0 else b''
            count = min(len(raw), 40 - position)
            expected[batch, position:position + count] = torch.tensor(list(raw[:count]))
            position += max(1, count)
    torch.testing.assert_close(radix.decode_activity(activity), expected)
    sparse = NativeActivity(rows, strength, torch.empty(3, 40, 4))
    torch.testing.assert_close(radix.decode_activity(sparse), expected)
    radix.insert(b'fg')
    radix.decode_activity(sparse)
    assert pointers == tuple(value.data_ptr() for value in radix.activity_decoder.buffers())
    with pytest.raises(OverflowError, match='position capacity'):
        radix.activity_decoder(torch.zeros(1, 41, dtype=torch.long), torch.ones(1, 41))


def test_field_without_perception_keeps_its_gradient():
    value = torch.tensor([[[[.4, .2]]]], requires_grad=True)
    field = ConceptualField(torch.tensor([19]), value,
        torch.tensor([[[0, 2]]]), torch.tensor([[[1, 2]]]), None)
    assert field.percepts is None
    assert torch.autograd.grad(field.evidence.sum(), value)[0].eq(1).all()


def test_owned_reverse_scores_native_bytes_and_percept_targets_in_their_own_spaces():
    from Models import BasicModel
    from Understanding import InputReconstruction, Understanding
    event = torch.tensor([[[97], [98]]])
    percepts = torch.ones(1, 2, 4)
    record = InputReconstruction(None, event, None, None, None, None, percepts)
    owner = NS(_query_sentence_depth=0, word_brackets=False,
        _reverse_event_loss=lambda prediction, target, **kw: (prediction-target).square().sum())
    understanding = Understanding(input_reconstruction=record)
    for target in (event, percepts):
        actual, cost = BasicModel.reverseReconstruct(owner, understanding, target=target)
        torch.testing.assert_close(actual, event)
        assert cost == 0


def test_reverse_is_grammar_free_and_missing_field_raises():
    from Models import BasicModel
    sentinel = object()
    calls = []
    def stage(name):
        def apply(value, **kwargs):
            calls.append((name, kwargs))
            return value
        return apply
    def forbidden(*args, **kwargs):
        raise AssertionError('symbolic reverse entered the decoder')
    model = NS(numeric_percepts=None, _reverse_body=stage('body'), _reverse_perceptual=stage('perceptual'),
        inputSpace=NS(reverse=stage('input')), _chart_generate_from_stm=forbidden)
    assert BasicModel.reverse(model, sentinel, field=sentinel) is sentinel
    assert calls == [('body', {'field': sentinel}), ('perceptual', {}), ('input', {})]
    with pytest.raises(ValueError, match='explicit conceptual field'):
        BasicModel._reverse_body(NS(concept_binding='aligned'), sentinel)
    model._reverse_perceptual = lambda _: (_ for _ in ()).throw(RuntimeError('broken native stage'))
    with pytest.raises(RuntimeError, match='broken native stage'):
        BasicModel.reverse(model, sentinel, field=sentinel)


def test_sparse_activity_at_8192_positions_never_crosses_the_native_inventory():
    from FieldReverse import reverse
    from PerceptProperties import PrimitiveProperties
    from torch.utils._python_dispatch import TorchDispatchMode
    radix = RadixLayer(3, initial_cap=8)
    a, b = radix.insert(b'a'), radix.insert(b'b')
    radix.activity_decoder = NativeDecode(8, 8192, 4)
    primitive = PrimitiveProperties(0)
    evidence = torch.tensor([[[[.7, 0.]]]], requires_grad=True)
    field = ConceptualField(torch.tensor([41]), evidence,
        torch.tensor([[[8191, 8192]]]), torch.tensor([[[1, 2]]]), None)
    cs = NS(_concept_allocator=NS(layer=lambda _: NS(_tensor_row_keys={0: 41})),
        cs_percept_attribution=lambda field, **kw: (torch.tensor([0]), field[..., :1], None),
        basic_percepts=NS(numeric_axes={}, is_value=lambda _row: False, occurrences=[((a,), ())]))
    ps = NS(percept_store=radix, subspace=NS(nWhat=3, nWhere=0, nWhen=0), _radix_where_indices=[])
    model = NS(_concept_owner=lambda: cs, perceptualSpace=ps,
               wholeSpace=NS(subspace=NS(what=NS(primitive_properties=primitive))))
    shapes = set()
    class Allocations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            value = func(*args, **(kwargs or {}))
            from torch.utils._pytree import tree_flatten
            shapes.update(tuple(t.shape) for t in tree_flatten(value)[0] if torch.is_tensor(t))
            return value
    pointers = tuple(value.data_ptr() for value in radix.activity_decoder.buffers())
    with Allocations():
        result = reverse(model, NS(carrier_like=NS), field)._radix_activity
        decoded = radix.decode_activity(result)
    assert (1, 8192, len(radix)) not in shapes
    assert isinstance(result, NativeActivity) and result.percepts.shape == (1, 8192, 3)
    assert decoded[0, 8191] == 97 and decoded[0, :8191].eq(0).all()
    assert tuple(value.data_ptr() for value in radix.activity_decoder.buffers()) == pointers
    gradient, = torch.autograd.grad(result.percepts.sum(), evidence)
    assert torch.isfinite(gradient).all()


def test_conceptual_projection_reverse_uses_its_declared_input_extent():
    from Spaces import ConceptualSpace, ProjectionBasis
    basis = ProjectionBasis()
    basis.create(nInput=2, nVectors=3, nDim=4)
    value = torch.rand(1, 3)
    carrier = NS(is_empty=lambda: False, materialize=lambda: value)
    owner = NS(subspace=NS(what=basis), inputShape=(2, 4), _right_half_dim=0,
        _adopt_reverse_carrier=lambda sub: sub, _reverse_stash=lambda sub: NS(),
        reverseBegin=lambda sub, **kw: None, reverseEnd=lambda result, **kw: result)
    result = ConceptualSpace.reverse(owner, carrier)
    assert result.shape == (1, 2, 4)
    torch.testing.assert_close(result, basis.reverse(value, V=2))


def test_admitted_word_identity_survives_an_empty_feature_gate(monkeypatch):
    from test_mm_xor import _fresh_model
    from Corpus import admit_training_forms
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _, _ = _fresh_model('data/XOR_grammar.xml')
    try:
        raw, word = admit_training_forms(model, ('shape',))[0]
        owner = model._concept_owner()
        row = owner._csw_row_of(word)
        codebook = owner.similarity_codebook
        identity = model.perceptualSpace.percept_store.identity
        with torch.no_grad():
            for (target, _), index in owner._concept_allocator.layer().features._index.items():
                if target == row:
                    owner._concept_allocator.layer().features.values[index] = 0.
            codebook.mereology._form_valid[row] = False
            code = codebook.lookup_rows(torch.tensor(row))
        assert identity.read(code[:identity.width]) == raw
        assert codebook.mereology._definitions([row])[row] == [(owner.basic_percepts.words[word], 0.)]
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_native_word_and_observed_promoted_part_have_the_same_byte_events():
    from FieldReverse import reverse
    from PerceptProperties import PrimitiveProperties
    from SentenceField import native_target
    radix = RadixLayer(4, initial_cap=16)
    letters = [radix.insert(bytes([byte])) for byte in b'abc']
    word = radix.insert(b'abc')
    radix.activity_decoder = NativeDecode(16, 8, 8)
    field = ConceptualField(torch.tensor([41]), torch.tensor([[[[1., 0.]]]]),
        torch.tensor([[[0, 3]]]), torch.tensor([[[1, 2]]]), None)
    cs = NS(_concept_allocator=NS(layer=lambda _: NS(_tensor_row_keys={0: 41})),
        cs_percept_attribution=lambda evidence, **kw: (torch.tensor([0]), evidence[..., :1], None),
        basic_percepts=NS(numeric_axes={}, is_value=lambda _row: False, occurrences=[((word,), ())]))
    ps = NS(percept_store=radix, subspace=NS(nWhat=4, nWhere=0, nWhen=0), _radix_where_indices=[],
        _forward_input={'native_indices': torch.tensor([[word]]), 'native_part_spans': field.where},
        _radix_part_events=lambda ids, offsets: radix._basis.lookup_rows(ids))
    model = NS(_concept_owner=lambda: cs, perceptualSpace=ps,
        wholeSpace=NS(subspace=NS(what=NS(primitive_properties=PrimitiveProperties(0)))))
    activity = reverse(model, NS(carrier_like=NS), field)._radix_activity
    expected = radix._basis.lookup_rows(torch.tensor([letters]))
    torch.testing.assert_close(activity.percepts, expected)
    torch.testing.assert_close(native_target(model, field), expected.detach())
    assert radix.decode_activity(activity).tolist() == [[97, 98, 99]]
