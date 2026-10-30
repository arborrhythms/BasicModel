"""Paired surface and counted inverse through the configured native dispatch.

Operations are explicitly selected for this mechanism certificate. It does
not claim that the chooser learned an attachment or a part of speech.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('postposed', (False, True))
def test_native_surface_dispatch_keeps_nested_counts_and_inverts(postposed):
    from data import TheData
    from Models import BaseModel
    from DerivationReconstruction import inverse_step
    config = Path(__file__).resolve().parents[1] / 'data/MM_corpus_orders.xml'
    TheData.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                       for split in ('train', 'validation', 'test')})
    model = BaseModel.from_config(str(config), data=TheData)[0]
    language = model.languageSpace
    rules = language._compose_binary_rules
    indices = {rule.method_name: i for i, rule in enumerate(rules)}
    ops = language._tree_layer(2).ops
    surface = getattr(ops[indices['surface']], 'gl', ops[indices['surface']])
    assert surface.meaning_layout is not None
    assert not hasattr(surface, 'marker_map')
    form_width, width = surface.meaning_layout
    bank = model.perceptualSpace.percept_store.identity
    try:
        leaves = []
        for i, raw in enumerate((b'alpha', b'beta', b'gamma', b'marker')):
            bank.admit(raw)
            # Grammar receives the PS event, including the native where/when
            # band, not the narrower dictionary's what vector.
            row = torch.tensor([[bank.word_rows[raw]]])
            form = model.perceptualSpace.synthesize_word_parts(
                row, torch.ones_like(row, dtype=torch.bool),
                torch.full_like(row, i)).reshape(-1)
            assert form.numel() == form_width
            meaning = torch.linspace(.1 + i * .05, .5 + i * .05, width - form_width)
            leaves.append(torch.cat((form, meaning))[None])

        def forward(name, left, right, counts):
            context = language._structural_context(phase='compose',
                input_stream=torch.stack((left, right), 1))
            context = replace(context, leaf_counts=tuple(left.new_tensor([[n]]) for n in counts))
            return language.forward_binary_step(left, right, torch.tensor([indices[name]]),
                torch.ones(1, dtype=torch.bool), grammar_context=context)

        content = forward('sum', forward('sum', leaves[0], leaves[1], (1, 1)), leaves[2], (2, 1))
        marker = leaves[3]
        if postposed:
            import STMChunks as chunks
            from Spaces import LanguageOperationChoice
            state = list(model.conceptualSpace.CSLang.state)
            state[chunks.BUFFER][0, :2] = torch.cat((marker, content))
            state[chunks.STACK][0, :2] = torch.tensor([0, 1])
            state[chunks.DEPTH][0] = 2
            choice = LanguageOperationChoice(marker, torch.tensor([3]), torch.tensor([0]),
                torch.tensor([indices['transpose']]), torch.tensor([True]), torch.ones(1),
                torch.tensor([0]), torch.tensor([True]))
            swapped = model.conceptualSpace.apply_language_choice(tuple(state), choice)
            # The native 2-to-2 route changes stack order, not physical values.
            assert torch.equal(swapped[chunks.BUFFER], state[chunks.BUFFER])
            assert int(swapped[chunks.DEPTH][0]) == 2
            newer, older = chunks.stack_view(swapped)[0][0, :2]
            left, right = older[None], newer[None]
        else:
            left, right = marker, content
        parent = forward('surface', left, right, (1, 3))
        expected_form = torch.stack(leaves).mean(0)[..., :form_width]
        torch.testing.assert_close(parent[..., :form_width], expected_form, atol=1e-6, rtol=0)
        recovered = inverse_step(language, 1, indices['surface'], parent,
            torch.cat((left, right, parent)), counts=(1, 3))
        for (actual, valid), expected in zip(recovered, (left, right)):
            assert valid.all()
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
        # Invert the complete selected forest, including the transpose action
        # and the two counted inner folds, back to the four source leaves.
        from DerivationReconstruction import reconstruct
        actions = ([] if postposed else [(0, -1, 3)]) + [
            (0, -1, 0), (0, -1, 1), (1, indices['sum'], -1),
            (0, -1, 2), (1, indices['sum'], -1)]
        if postposed:
            actions += [(0, -1, 3), (3, indices['transpose'], -1)]
        actions += [(1, indices['surface'], -1)]
        stack, operation_values, operand_counts = [], [], []
        for kind, operation, word in actions:
            values, counts = parent.new_zeros(3, width), (1, 1)
            if kind == 0:
                stack.append((leaves[word], 1))
            elif kind == 3:
                stack[-2:] = reversed(stack[-2:])
            else:
                (b, nb), (a, na) = stack.pop(), stack.pop()
                counts = na, nb
                result = forward('surface' if operation == indices['surface'] else 'sum', a, b, counts)
                values = torch.cat((a, b, result))
                stack.append((result, na + nb))
            operation_values.append(values)
            operand_counts.append(counts)
        program = SimpleNamespace(actions=torch.tensor(actions),
            operation_values=torch.stack(operation_values), operand_counts=torch.tensor(operand_counts))
        record = SimpleNamespace(word_values=torch.stack(leaves, 1), word_valid=torch.ones(1, 4, dtype=torch.bool),
            end_depth=torch.tensor([1]), end_slots=parent[:, None])
        decoded, unavailable, defined = reconstruct(language, record, [program], lambda b: range(4))
        assert defined.all() and not unavailable.any()
        torch.testing.assert_close(decoded, record.word_values, atol=4e-6, rtol=0)
        # The inverse starts at the completed understanding. Source words
        # supply only output geometry here; they cannot select the recovery.
        # The certified co-operands are live occurrence witnesses (§38.3).
        original = record.word_values
        record.word_values = torch.zeros_like(original)
        root = parent.detach().clone().requires_grad_()
        record.end_slots = root[:, None]
        program.operation_values = program.operation_values.detach().requires_grad_()
        decoded, unavailable, defined = reconstruct(language, record, [program], lambda b: range(4))
        torch.testing.assert_close(decoded, original, atol=4e-6, rtol=0)
        root_grad, witness_grad = torch.autograd.grad(decoded.sum(),
            (root, program.operation_values), allow_unused=True)
        assert root_grad.abs().sum() > 0
        assert witness_grad is not None and torch.isfinite(witness_grad).all()
        # A changed understanding must change the recovered content, even
        # with the same source geometry and fixed committed operation path.
        changed = root.detach().clone()
        changed[:, form_width] += .001
        record.end_slots = changed[:, None]
        different, _, _ = reconstruct(language, record, [program], lambda b: range(4))
        assert not torch.equal(different, decoded)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
