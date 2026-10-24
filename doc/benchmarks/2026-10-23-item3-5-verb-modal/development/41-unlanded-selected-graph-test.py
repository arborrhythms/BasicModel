
def test_eager_selected_face_preserves_operand_and_kernel_gradients(paired):
    from Language import OperationSelectionLayer, ProductLayer, LowerLayer, _BinaryGrammarOpAdapter
    from Queries import ConceptualSpaceCapability, StructuralGrammarContext
    ops = [_BinaryGrammarOpAdapter(ProductLayer(12, 12)), _BinaryGrammarOpAdapter(LowerLayer(12, 12))]
    layer = OperationSelectionLayer(d_model=12, ops=ops, unary_ops=[])
    value = torch.linspace(.1, .8, 48).reshape(2, 2, 12).requires_grad_()
    with torch.no_grad():
        for op in ops:
            op.gl.meaning_kernel.context.fill_(.01)
    context = StructuralGrammarContext(word_stream=(), conceptual_space=ConceptualSpaceCapability(12),
        primed_symbols=(), phase='compose', leaf_counts=(torch.ones(2, 1), torch.ones(2, 1)))
    candidates = layer._stacked_reduced(value, context)
    choices = torch.arange(2)
    expected = candidates[choices, 0, choices]
    selected = layer._selected_grammar_candidate(value, expected.detach(),
        torch.ones(2, dtype=torch.long), choices, torch.zeros(2, dtype=torch.long),
        choices, 2, 2, 0, context, None)
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)
    inputs = (value, *tuple(parameter for op in ops for parameter in op.parameters()))
    old = torch.autograd.grad(expected.square().sum(), inputs, allow_unused=True)
    new = torch.autograd.grad(selected.square().sum(), inputs, allow_unused=True)
    for a, b in zip(old, new):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert new[0].abs().sum() > 0

