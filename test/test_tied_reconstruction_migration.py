"""Reviewer probes for the completed-sentence tied reconstruction migration."""
import torch
import pytest

from test_reverse_traversal import _traversal_model, _run


def _model(tmp_path):
    from Models import _ensure_grad_anchors
    _ensure_grad_anchors(torch.device("cpu"))
    return _traversal_model(tmp_path)


@pytest.mark.parametrize("all_empty", [False, True])
def test_empty_rows_and_padding_carry_no_reconstruction_cost(tmp_path, all_empty):
    model = _model(tmp_path)
    try:
        _run(model, ["aa bb", "cc dd"])
        reference = model._tensor_pushed_ideas.detach().clone()
        active = model.inputSpace._word_active_mask.clone()
        active[1] = False
        if all_empty:
            active[:] = False
        model.inputSpace._word_active_mask = active
        reference[~active] = float("nan")
        recovered, idea, byte_cost, truncated, costs = model._reconstruct_sentences(
            model._stm_single_S, reference, model._tensor_sentence_roots_live,
            model._tensor_sentence_roots_depth, model._tensor_final_end_slots,
            model._tensor_final_end_depth)
        empty = ~active.any(-1)
        assert torch.isfinite(idea).all() and torch.isfinite(byte_cost).all()
        assert not recovered[~active].any()
        assert not idea[empty].any() and not byte_cost[empty].any()
        assert not costs[empty].any() and not truncated[empty].any()
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_reverse_replays_pre_fold_after_popping_the_new_word(tmp_path):
    """A valid pre,push,seal trace must invert as unseal,pop,pre.

    The current canonical producer leaves pre slots inactive. This explicitly
    executes a valid retained three-leaf compose derivation to check the
    documented trace ABI, including traces from earlier producers.
    """
    model = _model(tmp_path)
    try:
        _run(model, ["aa bb cc"])
        reference = model._tensor_pushed_ideas
        active = model.inputSpace._word_active_mask.clone()
        assert int(active.sum()) >= 3
        active[:, 3:] = False
        model.inputSpace._word_active_mask = active
        trace = model._reconstruction_stack()
        rules, arities, masks = trace.choices()
        left_rows, right_rows = trace.operand_rows()
        binary = model.languageSpace._tree_layer(2)
        index = list(binary.op_names).index("sum")
        rule_id = int(model.languageSpace._cs_binary_rule_ids[index])
        op = getattr(binary.ops[index], "gl", binary.ops[index])
        # push leaf0; push leaf1; pre-fold; push leaf2; seal.
        root01 = op.compose(reference[:, 0], reference[:, 1])
        root = op.compose(root01, reference[:, 2])
        width = int(active.shape[1])
        rows = model._word_symbol_rows()
        with torch.no_grad():
            masks.zero_()
            arities.zero_()
            for slot in (6, 3 * width):
                rules[:, slot] = rule_id
                arities[:, slot] = 2
                masks[:, slot] = True
            left_rows[:, 6], right_rows[:, 6] = rows[:, 0], rows[:, 1]
            left_rows[:, 3 * width], right_rows[:, 3 * width] = -1, rows[:, 2]
        recovered, cost, _bytes, truncated = model._reconstruct_sentence_traversal(root, reference)
        assert not bool(truncated.any())
        torch.testing.assert_close(recovered[:, :3], reference[:, :3], atol=1e-6, rtol=1e-5)
        assert float(cost.max()) < 1e-10
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_traced_verb_reverse_uses_the_actual_verb_operand(tmp_path):
    """Verb composition is its spectral operation, not its inherited Sigma."""
    model = _model(tmp_path)
    try:
        language = model.languageSpace
        binary = language._tree_layer(2)
        index = list(binary.op_names).index("verb")
        op = getattr(binary.ops[index], "gl", binary.ops[index])
        width = int(model.conceptualSpace.stm.concept_dim)
        generator = torch.Generator().manual_seed(812)
        noun = torch.randn(2, width, generator=generator).tanh() * .3
        verb = torch.randn(2, width, generator=generator).tanh() * .4
        with torch.no_grad():
            op._verb_spec.weight.fill_(.2)
        parent = op.compose(noun, verb)
        left, right = language.reverse_binary_step(
            parent, torch.full((2,), index), torch.ones(2, dtype=torch.bool),
            reference=verb, reference_side="right")
        torch.testing.assert_close(left, noun, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(right, verb)
        torch.testing.assert_close(op.compose(left, right), parent, atol=2e-5, rtol=2e-5)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_eager_reverse_evaluates_only_selected_recorded_operators(tmp_path, monkeypatch):
    model = _model(tmp_path)
    try:
        language = model.languageSpace
        binary = language._tree_layer(2)
        chosen = list(binary.op_names).index("sum")
        width = int(model.conceptualSpace.stm.concept_dim)
        calls = []
        original = language._reverse_of_binary_op

        def record(op, *args, **kwargs):
            calls.append(getattr(op, "gl", op).rule_name)
            return original(op, *args, **kwargs)

        monkeypatch.setattr(language, "_reverse_of_binary_op", record)
        parent, reference = torch.randn(2, width), torch.randn(2, width)
        language.reverse_binary_step(
            parent, torch.full((2,), chosen), torch.tensor([True, False]), reference)
        assert calls == ["sum"]
        calls.clear()
        language.reverse_binary_step(
            parent, torch.full((2,), chosen), torch.zeros(2, dtype=torch.bool), reference)
        assert calls == []
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_repeated_row_references_keep_their_occurrence_activations(tmp_path):
    """The same symbol identity can be pushed with different signed values."""
    model = _model(tmp_path)
    try:
        _run(model, ["aa bb cc"])
        active = model.inputSpace._word_active_mask.clone()
        active[:, :3] = True
        active[:, 3:] = False
        model.inputSpace._word_active_mask = active
        reference = torch.zeros_like(model._tensor_pushed_ideas)
        atom = torch.linspace(-.1, .1, reference.shape[-1], device=reference.device)
        reference[:, :3] = torch.tensor([.2, 1.3, -.6], device=reference.device)[:, None] * atom
        model.inputSpace._ar_word_concept_rows = torch.full_like(active, 42, dtype=torch.long)
        model.inputSpace._ar_word_object_rows = torch.full_like(active, -1, dtype=torch.long)
        trace = model._reconstruction_stack()
        rules, arities, masks = trace.choices()
        left_rows, right_rows = trace.operand_rows()
        binary = model.languageSpace._tree_layer(2)
        index = list(binary.op_names).index("sum")
        rule_id = int(model.languageSpace._cs_binary_rule_ids[index])
        op = getattr(binary.ops[index], "gl", binary.ops[index])
        root = op.compose(op.compose(reference[:, 0], reference[:, 1]), reference[:, 2])
        width = int(active.shape[1])
        with torch.no_grad():
            masks.zero_()
            arities.zero_()
            left_rows.fill_(-1)
            right_rows.fill_(-1)
            for slot in (4, 3 * width):
                rules[:, slot] = rule_id
                arities[:, slot] = 2
                masks[:, slot] = True
            left_rows[:, 4], right_rows[:, 4] = 42, 42
            right_rows[:, 3 * width] = 42
        recovered, cost, _bytes, truncated = model._reconstruct_sentence_traversal(root, reference)
        assert not bool(truncated.any())
        torch.testing.assert_close(recovered[:, :3], reference[:, :3], atol=1e-6, rtol=1e-5)
        assert float(cost.max()) < 1e-10
        damaged, damaged_cost, _, _ = model._reconstruct_sentence_traversal(root + .05, reference)
        assert not torch.allclose(damaged[:, :3], recovered[:, :3])
        assert float(damaged_cost.min()) > 1e-5
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_functional_sigma_inverse_uses_its_learned_affine_bias():
    """A zero-initialized bias must still invert after ordinary learning."""
    from Layers import SigmaLayer
    layer = SigmaLayer(8, 8, invertible=True)
    with torch.no_grad():
        layer.layer.biasWeight.copy_(torch.linspace(-.2, .2, 8).reshape(1, 8))
    parent = torch.linspace(-.15, .2, 16).reshape(2, 8)
    left, right = layer.generate_functional(parent)
    torch.testing.assert_close(layer.compose(left, right), parent, atol=2e-5, rtol=2e-5)
    expected = layer.generate(parent)
    torch.testing.assert_close(left, expected[0], atol=2e-5, rtol=2e-5)


def _identity_surface_kernels(model, monkeypatch):
    # Keep the test about ownership/dispatch, independent of downstream
    # percept dimensions. These are the shared numerical realization seams.
    monkeypatch.setattr(model, "_reverse_body", lambda value: value)
    monkeypatch.setattr(model, "_reverse_perceptual", lambda value: value)
    monkeypatch.setattr(model.inputSpace, "reverse", lambda value: value)
    model.reconstruct_from_idea = True
    model.idea_decode = False


def test_owned_input_reconstruction_survives_later_staging(tmp_path, monkeypatch):
    model = _model(tmp_path)
    _identity_surface_kernels(model, monkeypatch)
    monkeypatch.setattr(model, "_chart_generate_from_stm", lambda value: None)
    try:
        execution = _run(model, ["aa bb cc"])
        understanding = model._capture_understanding(execution)
        first, _ = model.reverseReconstruct(understanding)
        expected = first.detach().clone()
        _run(model, ["dd ee ff gg"])
        again, _ = model.reverseReconstruct(understanding)
        torch.testing.assert_close(again, expected)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_input_reconstruction_does_not_enter_free_generate(tmp_path, monkeypatch):
    model = _model(tmp_path)
    _identity_surface_kernels(model, monkeypatch)

    def forbidden(_value):
        raise AssertionError("input reconstruction entered free generate")

    monkeypatch.setattr(model, "_chart_generate_from_stm", forbidden)
    try:
        execution = _run(model, ["aa bb cc"])
        understanding = model._capture_understanding(execution)
        reconstructed, _ = model.reverseReconstruct(understanding)
        assert reconstructed is not None
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_zero_recovered_values_are_still_one_completed_traversal(tmp_path, monkeypatch):
    model = _model(tmp_path)
    _identity_surface_kernels(model, monkeypatch)
    monkeypatch.setattr(model, "_chart_generate_from_stm", lambda value: None)
    original = model._reconstruct_sentences
    calls = []

    def observed(*args, **kwargs):
        calls.append(1)
        result = original(*args, **kwargs)
        # A valid completed inverse is permitted to contain zero values;
        # readiness cannot be inferred from its nonzero count.
        return (torch.zeros_like(result[0]), *result[1:])

    monkeypatch.setattr(model, "_reconstruct_sentences", observed)
    try:
        execution = _run(model, ["aa bb"])
        understanding = model._capture_understanding(execution)
        model.reverseReconstruct(understanding)
        model.reverseReconstruct(understanding)
        assert len(calls) == 1
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_compiled_reverse_does_not_backpropagate_through_an_unused_inverse(tmp_path):
    model = _model(tmp_path)
    try:
        language = model.languageSpace
        binary = language._tree_layer(2)
        names = list(binary.op_names)
        selected = names.index("sum")
        unused = getattr(binary.ops[names.index("lift")], "gl", binary.ops[names.index("lift")])
        with torch.no_grad():
            unused._sigma.layer.raw_L.fill_(float("nan"))
        width = int(model.conceptualSpace.stm.concept_dim)
        parent = torch.linspace(-.2, .3, 2 * width).reshape(2, width).requires_grad_()
        witness = torch.full_like(parent, .03)
        indices = torch.full((2,), selected, dtype=torch.long)
        live = torch.tensor([True, False])
        reverse = torch.compile(language.reverse_binary_step, backend="eager", fullgraph=True)
        left, right = reverse(parent, indices, live, witness)
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        gradient, = torch.autograd.grad(left.sum() + right.sum(), (parent,))
        assert torch.isfinite(gradient).all()
        expected = torch.tensor([1., 2.]).reshape(2, 1).expand_as(gradient)
        torch.testing.assert_close(gradient, expected)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_enabling_tied_reconstruction_retains_the_owned_ideas(tmp_path):
    from Models import _ensure_grad_anchors
    from test_meronomy_ladder import _build_ladder_variant
    _ensure_grad_anchors(torch.device("cpu"))
    model = _build_ladder_variant(tmp_path, "enable_tied", [])
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    assert not model.reconstruct_in_loop
    model.reconstruct_in_loop = True
    try:
        execution = _run(model, ["aa bb cc"])
        understanding = model._capture_understanding(execution)
        owned = understanding.input_reconstruction
        assert owned is not None
        active = model.inputSpace._word_active_mask
        # A lossy fold can report a missing constituent. Enabling the path
        # must retain the actual slab, including such explicit zero results,
        # instead of publishing the disabled diagnostic placeholder.
        assert owned.ideas.shape[:2] == active.shape
        assert bool((owned.ideas.norm(dim=-1)[active] > 0).any())
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_explicit_target_changes_the_score_but_not_the_owned_reconstruction(tmp_path, monkeypatch):
    model = _model(tmp_path)
    _identity_surface_kernels(model, monkeypatch)
    try:
        execution = _run(model, ["aa bb"])
        understanding = model._capture_understanding(execution)
        surface = understanding.input_reconstruction.event
        first, exact = model.reverseReconstruct(understanding, target=surface.detach())
        again, damaged = model.reverseReconstruct(understanding, target=surface.detach() + .3)
        torch.testing.assert_close(first, again)
        assert float(exact.detach()) < 1e-10
        assert float(damaged.detach()) > .001
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_completed_state_damage_reaches_the_actual_input_surface(tmp_path):
    """The numerical surface chain cannot replace the inverse with a cached input."""
    model = _model(tmp_path)
    try:
        _run(model, ["aa bb cc"])
        parameters = {name: id(value) for name, value in model.named_parameters()}
        with torch.no_grad():
            original = model._complete_input_reconstruction()
            model._recon_completed = False
            model._reconstruction_product = None
            model._stm_single_S = model._stm_single_S + .1
            changed_end = model._tensor_final_end_slots.clone()
            changed_end[:, 0] += .1
            model._tensor_final_end_slots = changed_end
            damaged = model._complete_input_reconstruction()
        assert torch.isfinite(original.event).all() and torch.isfinite(damaged.event).all()
        assert not torch.allclose(original.ideas, damaged.ideas)
        assert not torch.allclose(original.event, damaged.event)
        assert {name: id(value) for name, value in model.named_parameters()} == parameters
    finally:
        model.End()
        model.symbolSpace.soft_reset()
