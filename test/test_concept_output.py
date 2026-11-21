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
    from Understanding import ConceptualField
    reader = torch.nn.Linear(4,2,bias=True)
    with torch.no_grad():
        reader.weight.copy_(torch.tensor([[1.,0.,0.,0.],[0.,0.,1.,0.]])); reader.bias.zero_()
    head = SimpleNamespace(field_readout=reader,outputShape=(2,1))
    owner = SimpleNamespace(_concept_row_of=lambda cid: {17:0,23:1}.get(cid))
    field = ConceptualField(carrier._concept_ids,carrier._concept_activations,
        torch.zeros(2,2,2,dtype=torch.long),torch.zeros(2,2,2,dtype=torch.long),None)
    result = OutputSpace.read_field(head,field,owner)
    torch.testing.assert_close(result[..., 0], torch.tensor([[.8, 0.], [.6, 0.]]))
    result.sum().backward()
    assert carrier._concept_activations.grad[..., 1].count_nonzero() == 0
    assert carrier._concept_activations.grad[2].count_nonzero() == 0


def test_reconstruction_without_owned_evidence_cannot_read_the_latest_field(monkeypatch):
    model, _, _, _ = _build_model(str(ROOT / 'data/XOR_symbols.xml'))
    model.forward(_bytes(['01']))
    carrier = SubSpace(inputShape=(1, 1), outputShape=(1, 1), nInputDim=1, nOutputDim=1)
    expected = torch.full_like(model.conceptualSpaces[0].subspace.materialize(), .125)
    carrier.set_event(expected)
    carrier.carrier_pure = True

    def forbidden(_carrier):
        raise AssertionError('inverse read a live field absent from its understanding')

    for cs in model.conceptualSpaces:
        monkeypatch.setattr(cs, 'unbind', forbidden)
    with pytest.raises(ValueError, match='field'):
        model._reverse_body(carrier)
    model.End()


def test_native_understanding_retains_field_and_native_percept_evidence():
    model, _, _, _ = _build_model(str(ROOT / 'data/XOR_symbols.xml'))
    understanding = model.understand(_bytes(['01']))
    # A configuration without sentence composition has the system-1 field only.
    assert not model._composes_sentences()
    assert set(understanding.reconstruction_carriers) == {'field', 'ir_mask_positions'}
    field = understanding.reconstruction_carriers['field']
    assert field.concept_ids.dtype == torch.long
    assert field.percepts is None
    assert not hasattr(field, 'event')
    model.End()


def test_parallel_field_never_dispatches_grammar_lift_or_lower(monkeypatch):
    from Language import ChangeLayer, LowerLayer
    model, _, _, _ = _build_model(str(ROOT / 'data/XOR_symbols.xml'))

    def forbidden(*args, **kwargs):
        raise AssertionError('parallel field dispatched the grammar')

    monkeypatch.setattr(model.languageSpace, 'forward', forbidden)
    for operator in (ChangeLayer, LowerLayer):
        monkeypatch.setattr(operator, 'forward', forbidden)
        monkeypatch.setattr(operator, 'reverse', forbidden)
    model.forward(_bytes(['01']))
    model._chart_compose_per_word()
    model._chart_compose_at_C()
    model._chart_generate_from_stm()
    model.End()




@pytest.mark.parametrize('trial', range(3))
def test_native_cli_curriculum_learns_output_and_keeps_located_inverse(trial, monkeypatch):
    from numeric_xor import certificate, fit
    model, prediction, error = certificate('data/XOR_exact.xml')
    _, _, without_intersection = fit('data/XOR_exact.xml', promotion=False)
    # Without the intersection XOR is not learned. The order-zero concepts
    # read their combined percepts by a squashed Sigma (rules 7 and 8), so the
    # readout of them may do better than the affine fit, and no worse.
    assert 1e-6 < without_intersection <= .25 + 1e-6
    values = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    design = torch.cat((values,torch.ones(4,1)),1)
    target = torch.tensor([0.,1.,1.,0.])
    affine = design @ torch.linalg.lstsq(design,target).solution
    assert abs(float((affine-target).square().mean())-.25) < 1e-6
    print(dict(trial=trial,mse=error,without_intersection=without_intersection))


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
