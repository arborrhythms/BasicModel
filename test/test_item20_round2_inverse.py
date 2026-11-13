"""Recorded action witnesses and fail-loud reconstruction certificates."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ChunkJournal import action_context, value_record
from DerivationReconstruction import inverse_step
from Generative import native_recompose
from Layers import GrammarLayer
from MeaningKernels import MeaningAction


def action_fixture(kind):
    op = GrammarLayer(12, 12).double()
    op.meaning_layout = (4, 12)
    op.rule_name = 'adverb' if kind == 'adverb' else 'lift'
    op.meaning_action = MeaningAction(8, modifier=kind == 'adverb').double()
    with torch.no_grad():
        for index, parameter in enumerate(op.meaning_action.parameters()):
            parameter.add_((torch.arange(parameter.numel(), dtype=parameter.dtype)
                            .reshape_as(parameter) + index + 1) / 300)
    left = torch.linspace(.15, .7, 12, dtype=torch.float64)[None]
    right = left.flip(-1)
    context = dict(object_value=left.roll(1, -1), object_present=torch.tensor([kind == 'object']),
                   group=torch.tensor([0 if kind == 'group' else -1]))
    parent = op.compose_blocks(lambda a, b: a+b, left, right, action_context=context)
    return op, left, right, parent, context


@pytest.mark.parametrize('kind', ['object', 'group', 'adverb'])
def test_recorded_action_context_round_trips_without_changing_adverbs(kind):
    op, left, right, parent, context = action_fixture(kind)
    record = value_record(torch.stack((left, right), 1),
                          SimpleNamespace(candidate=parent, action_context=context))
    program = SimpleNamespace(action_records=record[:, 36:])
    witness = action_context(program, 0)
    language = SimpleNamespace(_tree_layer=lambda arity: SimpleNamespace(ops=[op]))
    recovered = inverse_step(language, 1, 0, parent, record[:, :36].reshape(3, 12),
                             action_context=witness)[0][0]
    torch.testing.assert_close(recovered, left, rtol=0, atol=2e-15)
    recomposed = native_recompose(op, parent, left, right, form_width=4, action_context=witness)
    torch.testing.assert_close(recomposed, parent, rtol=0, atol=2e-15)
    if kind == 'adverb':
        recovered = inverse_step(language, 1, 0, parent, record[:, :36].reshape(3, 12))[0][0]
        torch.testing.assert_close(recovered, left, rtol=0, atol=2e-15)
    else:
        with pytest.raises(ValueError, match='recorded forward context'):
            inverse_step(language, 1, 0, parent, record[:, :36].reshape(3, 12))


def test_action_record_keeps_forward_value_and_gradient_after_live_refresh():
    op, left, right, parent, context = action_fixture('object')
    original = context['object_value'].clone().requires_grad_()
    context['object_value'] = original
    record = value_record(torch.stack((left, right), 1),
                          SimpleNamespace(candidate=parent, action_context=context))
    frozen = original.detach().clone()
    with torch.no_grad():
        original.add_(.125)
    witness = action_context(SimpleNamespace(action_records=record[:, 36:]), 0)
    torch.testing.assert_close(witness['object_value'], frozen, rtol=0, atol=0)
    witness['object_value'].sum().backward()
    torch.testing.assert_close(original.grad, torch.ones_like(original), rtol=0, atol=0)
    assert action_context(SimpleNamespace(action_records=None), 0) is None


def test_decoder_stage_error_reaches_body_reconstruction_and_batch(monkeypatch):
    from test_mm_xor import _fresh_model
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _, data = _fresh_model()
    try:
        raw, target = next(iter(data.data_loader(split='train', num_streams=4)))
        batch = (model.inputSpace.prepInput(raw), model.outputSpace.prepOutput(target))
        model.runBatch(train=False, batchSize=4, split='train', batch_override=batch)
        understanding = replace(model._last_understanding, input_reconstruction=None)
        error = RuntimeError('injected decoder stage')
        def reject(*args, **kwargs):
            raise error
        monkeypatch.setattr(model.numeric_percepts, 'reverse', reject)
        for call in (lambda: model.reverse(model.conceptualSpace.subspace, field=understanding.reconstruction_carriers['field']),
                     lambda: model.reverseReconstruct(understanding),
                     lambda: model.runBatch(train=False, batchSize=4, split='train', batch_override=batch)):
            with pytest.raises(RuntimeError, match='injected decoder stage') as raised:
                call()
            assert raised.value is error
    finally:
        model.End()


def test_projection_basis_reverse_uses_recorded_input_shape(tmp_path):
    from pathlib import Path
    from test_conceptual_recurrence import _build
    from Spaces import ProjectionBasis, SubSpace
    source = Path('data/MM_text.xml').read_text().replace(
        '<ConceptualSpace>', '<ConceptualSpace><codebook>project</codebook>')
    path = tmp_path / 'projection.xml'
    path.write_text(source)
    model = _build(str(path))
    try:
        space = model.conceptualSpaces[0]
        assert isinstance(space.subspace.what, ProjectionBasis)
        shape = (1, *space.inputShape)
        values = torch.linspace(.1, .4, torch.tensor(shape).prod().item()).reshape(shape)
        encoded = space.subspace.what.forward(values)
        # The projection emits scalar activations, carried as values. Its
        # coefficients are not a selection in the event codebook.
        carrier = SubSpace(encoded.shape[1:], encoded.shape[1:])
        carrier.carrier_pure = True
        carrier.set_event(encoded)
        recovered = space.reverse(carrier).materialize()
        assert recovered.shape == values.shape and torch.isfinite(recovered).all()
    finally:
        model.End()
