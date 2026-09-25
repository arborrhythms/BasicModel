"""Native concept identity, learned output, and its independent ordered inverse."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from ConceptLessons import teach_concept_lessons
from recon_bench import _build_model
from Spaces import OutputSpace, SubSpace


ROOT = Path(__file__).resolve().parents[1]


def _bytes(texts, width=16):
    result = torch.zeros(len(texts), 1, width, dtype=torch.long)
    for b, text in enumerate(texts):
        result[b, 0, :len(text)] = torch.tensor(list(text.encode()))
    return result


def test_concept_output_resolves_each_batch_identity_and_preserves_unknown():
    # Field rows exchange identities between batch rows. A both pair reads
    # positive evidence, never the both corner or a signed difference.
    carrier = SubSpace(inputShape=(1, 1), outputShape=(1, 1), nInputDim=1, nOutputDim=1)
    carrier._concept_ids = torch.tensor([[17, 23], [23, 17], [-1, -1]])
    carrier._concept_activations = torch.tensor([
        [[ [.8, .8], [.3, 0.] ], [[0., 0.], [0., 0.]]],
        [[ [0., 0.], [0., 0.] ], [[.2, 0.], [.6, .9]]],
        [[ [1., 1.], [1., 1.] ], [[1., 1.], [1., 1.]]],
    ], requires_grad=True)
    head = SimpleNamespace(concept_ids=(17, 23), subspace=SubSpace(
        inputShape=(1, 1), outputShape=(1, 1), nInputDim=1, nOutputDim=1))
    result = OutputSpace.read_concepts(head, carrier).materialize()
    torch.testing.assert_close(result[..., 0], torch.tensor([[.8, 0.], [.6, 0.]]))
    result.sum().backward()
    assert carrier._concept_activations.grad[..., 1].count_nonzero() == 0
    assert carrier._concept_activations.grad[2].count_nonzero() == 0


def test_reconstruction_without_owned_evidence_cannot_read_the_latest_field(monkeypatch):
    model, _, _, _ = _build_model(str(ROOT / 'data/XOR_exact.xml'))
    model.forward(_bytes(['01']))
    carrier = SubSpace(inputShape=(1, 1), outputShape=(1, 1), nInputDim=1, nOutputDim=1)
    expected = torch.full_like(model.conceptualSpaces[0].subspace.materialize(), .125)
    carrier.set_event(expected)
    carrier.carrier_pure = True

    def forbidden(_carrier):
        raise AssertionError('inverse read a live field absent from its understanding')

    for cs in model.conceptualSpaces:
        monkeypatch.setattr(cs, 'unbind', forbidden)
    recovered = model._reverse_body(carrier)
    torch.testing.assert_close(recovered.materialize(), expected)
    model.End()


def test_native_understanding_retains_field_evidence_not_percept_events():
    model, _, _, _ = _build_model(str(ROOT / 'data/XOR_exact.xml'))
    understanding = model.understand(_bytes(['01']))
    assert set(understanding.reconstruction_carriers) == {'field', 'ir_mask_positions'}
    field = understanding.reconstruction_carriers['field']
    assert field.concept_ids.dtype == torch.long
    assert field.position_evidence.shape[-1] == 2
    assert not hasattr(field, 'event')
    model.End()


def test_parallel_field_never_dispatches_grammar_lift_or_lower(monkeypatch):
    from Language import LiftLayer, LowerLayer
    model, _, _, _ = _build_model(str(ROOT / 'data/XOR_exact.xml'))

    def forbidden(*args, **kwargs):
        raise AssertionError('parallel field dispatched the grammar')

    monkeypatch.setattr(model.languageSpace, 'forward', forbidden)
    for operator in (LiftLayer, LowerLayer):
        monkeypatch.setattr(operator, 'forward', forbidden)
        monkeypatch.setattr(operator, 'reverse', forbidden)
    model.forward(_bytes(['01']))
    model._chart_compose_per_word()
    model._chart_compose_at_C()
    model._chart_generate_from_stm()
    model.End()


@pytest.mark.slow
def test_serial_grammar_never_executes_field_sigma_pi_or_not(tmp_path, monkeypatch):
    from test_reverse_traversal import _traversal_model, _run
    model = _traversal_model(tmp_path)
    assert model.serial

    def forbidden(*args, **kwargs):
        raise AssertionError('serial grammar executed a field evidence operation')

    for cs in model.conceptualSpaces:
        cs._symbolic_order = 3  # exclusion is by mode, not an empty field cap
        cs.conceptual_pi = True
        assert not cs._sparse_active()
        for name in ('cs_read_memberships', '_compose_order0', 'cs_forward_content'):
            monkeypatch.setattr(cs, name, forbidden)
    _run(model, ['12 plus 1', '3 plus 4'])
    model.End()
    model.symbolSpace.soft_reset()


@pytest.mark.parametrize('trial', range(3))
def test_native_cli_curriculum_learns_output_and_keeps_located_inverse(trial, monkeypatch):
    model, _, _, _ = _build_model(str(ROOT / 'data/XOR_exact.xml'))
    teach_concept_lessons(model)
    model.eval()
    model.set_sigma(0.)
    cs = model.conceptualSpaces[0]
    store = cs._concept_allocator.layer(0)
    output = cs._csw_row_of(20001)
    cases = model._concept_lesson_receipt['cases']
    assert len(cases) == 4
    assert all(value == 0 for _, value in cs.concept_weights(output))
    assert all(cs._order0_inventory_row(r) and cs._order0_inventory_row(c % (store.nOutput + 1))
               for r, c in store.conjunctive._index)
    assert not any(r == output for r, _ in store.conjunctive._index)
    assert all(len(brackets) > 0 for brackets in store.conjunctive.locations.values())
    x = _bytes(['00', '01', '10', '11'])
    target = torch.tensor([0., 1., 1., 0.])
    optimizer = torch.optim.Adam([store.values], lr=.03)
    initial = None
    for _ in range(40):
        model.End()
        optimizer.zero_grad()
        prediction = model.forward(x)[2][:, 0, 0]
        loss = (prediction - target).square().mean()
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
        store.project_parts()
    model.End()
    understanding = model.understand(x)
    prediction = understanding.execution[2][:, 0, 0]
    assert initial == .5
    torch.testing.assert_close(prediction, target, atol=1e-6, rtol=0)
    torch.testing.assert_close(cs._cs_extents, torch.tensor([[[0, 2]]]).expand(4, -1, -1))
    assert len([weight for _, weight in cs.concept_weights(output) if weight > 0]) == 2
    reverse, _ = model.reverseReconstruct(understanding)
    assert model._decode_reconstructed_inputs(reverse, ['00', '01', '10', '11']) == ['00', '01', '10', '11']
    field = understanding.reconstruction_carriers['field']
    # Zero forward evidence cannot recover bytes from perceptual context or
    # the execution adapter. Those products remain present in this record.
    blank = replace(field, evidence=torch.zeros_like(field.evidence),
                    position_evidence=torch.zeros_like(field.position_evidence))
    unknown = replace(understanding, reconstruction_carriers={'field': blank})
    absent, _ = model.reverseReconstruct(unknown)
    assert absent.count_nonzero() == 0
    # A later perception cannot replace the captured understanding's inverse.
    model.End()
    model.forward(_bytes(['AAAA', 'Z', '222222', '!!!']))
    cs._cs_field_rows = torch.full_like(cs._cs_field_rows, -1)
    cs._cs_feature_memberships = torch.full_like(cs._cs_feature_memberships, float('nan'))
    cs._cs_position_spans.zero_()
    def forbidden(*args, **kwargs):
        raise AssertionError('native inverse read a percept event or code neighbour')
    for owner in model.conceptualSpaces:
        monkeypatch.setattr(owner, 'unbind', forbidden)
    monkeypatch.setattr(model.perceptualSpace, '_decode_radix_meta', forbidden)
    monkeypatch.setattr(model.perceptualSpace.percept_store, 'associate_span', forbidden)
    repeated, _ = model.reverseReconstruct(understanding)
    torch.testing.assert_close(repeated, reverse)
    assert model._decode_reconstructed_inputs(repeated, ['wrong'] * 4) == ['00', '01', '10', '11']
    model.End()
    for texts in [['AA', 'ZZ', '22', '!!'], ['A', 'ZZZZ', '222222', '!!!!!!!!']]:
        assert model.forward(_bytes(texts))[2].count_nonzero() == 0
        model.End()
    print(dict(trial=trial, initial_mse=initial, output=prediction.detach().tolist(),
               conjunctions=len(cases), reconstructed=['00', '01', '10', '11']))


def test_contained_ordered_group_reconstructs_at_its_extent():
    model, _, _, _ = _build_model(str(ROOT / 'data/XOR_exact.xml'))
    teach_concept_lessons(model)
    texts = ['00', '01', '10', '11']
    cs = model.conceptualSpaces[0]
    radix = model.perceptualSpace.percept_store
    for index, text in enumerate(texts):
        row = cs._csw_concept_row(0, 30000 + index)
        group = tuple(radix.get_id(bytes([value])) for value in text.encode())
        cs.add_concept_feature(row, 'ps', group, 1.)
        assert radix.get_id(text.encode()) is None
    understanding = model.understand(_bytes(texts))
    # Isolate PartSpace attribution. Its group spans the word extent even
    # when WholeSpace divides it into single-position property runs.
    features = cs._concept_allocator.layer(0).features
    with torch.no_grad():
        for (_, column), index in features._index.items():
            if (column // 2) % 2:
                features.values[index] = 0
    reverse, _ = model.reverseReconstruct(understanding)
    assert model._decode_reconstructed_inputs(reverse, []) == texts
    model.End()


def test_property_reverse_attributes_only_written_members():
    from PerceptProperties import PrimitiveProperties
    primitive = PrimitiveProperties(1)
    primitive.teach(0, [49], [1.])
    evidence = torch.tensor([[[1.]]], requires_grad=True)
    support = primitive.reverse(evidence)
    torch.testing.assert_close(support[..., 49], evidence[..., 0])
    # Reconstruction cannot invent a membership on an unseen primitive.
    (support * torch.arange(256)).sum().backward()
    assert primitive.members.grad[0, 48] == 0
    assert evidence.grad.abs().sum() > 0
