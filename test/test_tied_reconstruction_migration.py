"""Reviewer probes for the completed-sentence tied reconstruction migration."""
import torch
import pytest

from test_reverse_traversal import _traversal_model, _run

pytestmark = pytest.mark.usefixtures("eager_reading")


def _model(tmp_path):
    from Models import _ensure_grad_anchors
    _ensure_grad_anchors(torch.device("cpu"))
    return _traversal_model(tmp_path)


@pytest.mark.parametrize("all_empty", [False, True])
@pytest.mark.slow
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


def _supplied_three_leaf_inverse(model, root, reference):
    """The supplied-reading inverse owns occurrence witnesses; free R does not."""
    from types import SimpleNamespace
    from DerivationReconstruction import reconstruct
    language = model.languageSpace
    index = list(language._tree_layer(2).op_names).index('sum')
    op = getattr(language._tree_layer(2).ops[index], 'gl', language._tree_layer(2).ops[index])
    first = op.compose(reference[:, 0], reference[:, 1])
    original = op.compose(first, reference[:, 2])
    actions = torch.tensor([[0,-1,0],[0,-1,1],[1,index,-1],[0,-1,2],[1,index,-1]])
    programs = []
    for row in range(len(reference)):
        frames = reference.new_zeros(5, 3, reference.shape[-1])
        frames[2] = torch.stack((reference[row,0], reference[row,1], first[row]))
        frames[4] = torch.stack((first[row], reference[row,2], original[row]))
        programs.append(SimpleNamespace(actions=actions, operation_values=frames))
    valid = torch.zeros(reference.shape[:2], dtype=torch.bool)
    valid[:, :3] = True
    record = SimpleNamespace(word_values=reference, word_valid=valid,
        end_depth=torch.ones(len(reference), dtype=torch.long), end_slots=root[:,None], primed=None)
    recovered, unavailable, _ = reconstruct(language, record, programs, lambda _: torch.arange(3))
    error = (recovered[:, :3] - reference[:, :3]).square().mean((-1, -2))
    return recovered, error, None, unavailable.any(-1)


def _explicit_terminal_field(model):
    """A three-slot fixed reading exercises ownership with the native decoder."""
    from test_output_walk import _stop
    _stop(model)
    _, codes, _, _ = model._snapshot_tables(model._tensor_pushed_ideas)
    positions = model.inputSpace._ar_grammar_leaf_mask[0].nonzero().flatten()[:3]
    slots = codes.index_select(1, positions).clone()
    model._tensor_final_end_slots = slots
    model._tensor_final_end_depth = torch.full((len(slots),), 3, dtype=torch.long)
    model._stm_single_S = slots[:, 0]
    model._tensor_sentence_roots_live[:, 0] = slots.flatten(1)
    model._tensor_sentence_roots_depth[:, 0] = 3
    model._recon_completed = False
    model._reconstruction_product = None


@pytest.mark.slow
def test_supplied_reading_inverts_a_pre_fold_then_the_new_word(tmp_path):
    """The supplied reader follows its three-leaf program, including the pre-fold."""
    model = _model(tmp_path)
    try:
        _run(model, ["aa bb cc"])
        reference = model._tensor_pushed_ideas
        active = model.inputSpace._word_active_mask.clone()
        assert int(active.sum()) >= 3
        active[:, 3:] = False
        model.inputSpace._word_active_mask = active
        binary = model.languageSpace._tree_layer(2)
        index = list(binary.op_names).index("sum")
        op = getattr(binary.ops[index], "gl", binary.ops[index])
        root01 = op.compose(reference[:, 0], reference[:, 1])
        root = op.compose(root01, reference[:, 2])
        recovered, cost, _bytes, truncated = _supplied_three_leaf_inverse(model, root, reference)
        assert not bool(truncated.any())
        torch.testing.assert_close(recovered[:, :3], reference[:, :3], atol=1e-6, rtol=1e-5)
        assert float(cost.max()) < 1e-10
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.slow
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
        noun = .5 + torch.randn(2, width, generator=generator).tanh() * .3
        verb = .5 + torch.randn(2, width, generator=generator).tanh() * .4
        with torch.no_grad():
            op._verb_spec.weight.fill_(.002)
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_repeated_row_references_keep_their_occurrence_activations(tmp_path):
    """The same symbol identity can be pushed with different occurrence values."""
    model = _model(tmp_path)
    try:
        _run(model, ["aa bb cc"])
        active = model.inputSpace._word_active_mask.clone()
        active[:, :3] = True
        active[:, 3:] = False
        model.inputSpace._word_active_mask = active
        reference = torch.zeros_like(model._tensor_pushed_ideas)
        atom = torch.linspace(-.1, .1, reference.shape[-1], device=reference.device)
        reference[:, :3] = .5 + torch.tensor([.2, 1.3, -.6], device=reference.device)[:, None] * atom
        model.inputSpace._ar_word_concept_rows = torch.full_like(active, 42, dtype=torch.long)
        model.inputSpace._ar_word_object_rows = torch.full_like(active, -1, dtype=torch.long)
        binary = model.languageSpace._tree_layer(2)
        index = list(binary.op_names).index("sum")
        op = getattr(binary.ops[index], "gl", binary.ops[index])
        root = op.compose(op.compose(reference[:, 0], reference[:, 1]), reference[:, 2])
        recovered, cost, _bytes, truncated = _supplied_three_leaf_inverse(model, root, reference)
        assert not bool(truncated.any())
        torch.testing.assert_close(recovered[:, :3], reference[:, :3], atol=1e-6, rtol=1e-5)
        assert float(cost.max()) < 1e-10
        damaged, damaged_cost, _, _ = _supplied_three_leaf_inverse(model, root + .05, reference)
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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

    try:
        execution = _run(model, ["aa bb"])
        model._recon_completed = False
        model._reconstruction_product = None
        model.__dict__.pop('_reconstruct_compiled', None)
        monkeypatch.setattr(model, "_reconstruct_sentences", observed)
        understanding = model._capture_understanding(execution[:4])
        model.reverseReconstruct(understanding)
        model.reverseReconstruct(understanding)
        assert len(calls) == 1
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.slow
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
        parent = torch.linspace(.2, .3, 2 * width).reshape(2, width).requires_grad_()
        witness = torch.full_like(parent, .03)
        indices = torch.full((2,), selected, dtype=torch.long)
        live = torch.tensor([True, False])
        reverse = torch.compile(language.reverse_binary_step, backend="eager", fullgraph=True)
        left, right = reverse(parent, indices, live, witness)
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        gradient, = torch.autograd.grad(left.sum() + right.sum(), (parent,))
        assert torch.isfinite(gradient).all()
        expected = torch.tensor([2., 2.]).reshape(2, 1).expand_as(gradient)
        torch.testing.assert_close(gradient, expected)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


@pytest.mark.slow
def test_enabling_tied_reconstruction_retains_the_owned_ideas(tmp_path):
    from Models import _ensure_grad_anchors
    from test_meronomy_ladder import _build_ladder_variant
    _ensure_grad_anchors(torch.device("cpu"))
    model = _build_ladder_variant(tmp_path, "enable_tied", [])
    model.reconstruct_in_loop = False
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    assert not model.reconstruct_in_loop
    model.reconstruct_in_loop = True
    try:
        execution = _run(model, ["aa bb cc"])
        _explicit_terminal_field(model)
        understanding = model._capture_understanding(execution[:4])
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_completed_state_damage_reaches_the_actual_input_surface(tmp_path):
    """The numerical surface chain cannot replace the inverse with a cached input."""
    model = _model(tmp_path)
    try:
        _run(model, ["aa bb cc"])
        _explicit_terminal_field(model)
        parameters = {name: id(value) for name, value in model.named_parameters()}
        with torch.no_grad():
            original = model._complete_input_reconstruction()
            model._recon_completed = False
            model._reconstruction_product = None
            model._stm_single_S = model._stm_single_S * .8 + .1
            changed_end = model._tensor_final_end_slots.clone()
            changed_end[:, 0] = changed_end[:, 0] * .8 + .1
            model._tensor_final_end_slots = changed_end
            model._tensor_sentence_roots_live[:, 0] = changed_end.flatten(1)
            damaged = model._complete_input_reconstruction()
        assert torch.isfinite(original.event).all() and torch.isfinite(damaged.event).all()
        assert not torch.allclose(original.ideas, damaged.ideas)
        assert not torch.equal(original.event, damaged.event)
        assert {name: id(value) for name, value in model.named_parameters()} == parameters
    finally:
        model.End()
        model.symbolSpace.soft_reset()
