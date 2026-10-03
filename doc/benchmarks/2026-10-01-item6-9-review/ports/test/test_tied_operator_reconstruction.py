"""Compose-operator and snapshot probes for tied input reconstruction."""
import pytest
import torch

from test_tied_reconstruction_migration import _model
from test_reverse_traversal import _run


def test_word_scoring_honors_existing_nul_termination():
    """A token's first NUL terminates scoring, including within stored spans."""
    from types import SimpleNamespace
    from Models import BasicModel

    owner = SimpleNamespace(_BYTE_ASSIGNMENT_TAU=.1)
    bank_n = torch.eye(2).unsqueeze(0)
    word = torch.tensor(0)
    expected = expected_gradient = None
    for target_junk, candidate_junk in ((False, False), (True, False),
                                         (False, True), (True, True)):
        idea = torch.tensor([[1., .25]], requires_grad=True)
        target = torch.tensor([[[97, 0, 99 if target_junk else 0]]])
        target_valid = torch.tensor([[[True, True, target_junk]]])
        bank_bytes = torch.tensor([[[97, 0, 120 if candidate_junk else 0], [97, 98, 0]]])
        bank_valid = torch.tensor([[[True, True, candidate_junk], [True, True, True]]])
        cost = BasicModel._byte_word_cost(
            owner, idea, word, bank_n, bank_bytes, bank_valid,
            target, target_valid, True)
        gradient, = torch.autograd.grad(cost.sum(), (idea,))
        if expected is None:
            expected, expected_gradient = cost.detach(), gradient
        else:
            torch.testing.assert_close(cost, expected)
            torch.testing.assert_close(gradient, expected_gradient)
        unknown = BasicModel._byte_word_cost(
            owner, idea, word, bank_n, bank_bytes, torch.zeros_like(bank_valid),
            target, target_valid, True)
        torch.testing.assert_close(unknown, torch.tensor([256.]).log())


@pytest.mark.parametrize("window", [1, 2])
def test_surface_snapshot_does_not_turn_a_longer_word_into_a_full_window_match(window):
    """Candidate truncation cannot manufacture a word end at the target limit."""
    from types import SimpleNamespace
    from Models import BasicModel

    surfaces = {4: b"a" * window, 5: b"a" * window + b"b"}
    lexicon = SimpleNamespace(word_surface_for_row=surfaces.get)
    staged = SimpleNamespace(
        _ar_concept_lookup_rows=torch.tensor([[4, 5]]),
        _ar_concept_lookup_atoms=torch.eye(2).unsqueeze(0),
        _ar_word_part_ids=torch.zeros(1, 1, window, dtype=torch.long))
    owner = SimpleNamespace(inputSpace=staged, _concept_owner=lambda: lexicon,
                            _BYTE_ASSIGNMENT_TAU=.1)
    BasicModel._stage_snapshot_bytes(owner)
    ready, atoms, values, valid = BasicModel._snapshot_tables(owner, torch.zeros(1, 1, 2))
    target = torch.full((1, 1, window), ord("a"), dtype=torch.long)
    def cost(idea):
        return BasicModel._byte_word_cost(
            owner, idea, torch.tensor(0), atoms, values, valid,
            target, torch.ones_like(target, dtype=torch.bool), ready)
    correct = cost(torch.tensor([[1., .01]]))
    longer_idea = torch.tensor([[.01, 1.]], requires_grad=True)
    longer = cost(longer_idea)
    assert correct < .01
    assert longer > correct + 1., "clipping a stored word must not invent termination"
    gradient, = torch.autograd.grad(longer.sum(), (longer_idea,))
    assert gradient.isfinite().all() and gradient.abs().sum() > 0


def test_byte_fidelity_penalizes_a_candidate_with_extra_suffix_bytes():
    """Matching the input prefix must not make a longer word a perfect match."""
    from types import SimpleNamespace
    from Models import BasicModel

    owner = SimpleNamespace(_BYTE_ASSIGNMENT_TAU=.1)
    atoms = torch.eye(2).unsqueeze(0)
    spellings = torch.tensor([[[97, 0], [97, 98]]])  # WORDs "a" and "ab"
    lengths = torch.tensor([[[True, False], [True, True]]])
    target = torch.tensor([[[97, 0]]])
    target_valid = torch.tensor([[[True, False]]])
    def cost(idea):
        return BasicModel._byte_word_cost(
            owner, idea, torch.tensor(0), atoms, spellings, lengths,
            target, target_valid, True)
    short = cost(torch.tensor([[1., .01]]))
    longer_idea = torch.tensor([[.01, 1.]], requires_grad=True)
    longer = cost(longer_idea)
    assert short < .01
    assert longer > short + 1., "the target's word end must be scored"
    gradient, = torch.autograd.grad(longer.sum(), (longer_idea,))
    assert gradient.isfinite().all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("stale_surface", [False, True])
def test_inactive_dictionary_row_cannot_poison_byte_loss_or_its_gradient(tmp_path, stale_surface):
    model = _model(tmp_path)
    try:
        isp = model.inputSpace
        isp._ar_concept_lookup_atoms = torch.tensor([[[1., 0.], [float("nan"), float("nan")]]])
        isp._ar_concept_lookup_rows = torch.tensor([[4, -1]])
        isp._ar_bank_bytes = torch.tensor([[[97, 98], [99, 100]]])
        isp._ar_bank_valid = torch.tensor([[[True, True], [stale_surface, stale_surface]]])
        _, atoms, values, valid = model._snapshot_tables(torch.zeros(1, 1, 2))
        idea = torch.tensor([[.7, .4]], requires_grad=True)
        cost = model._byte_word_cost(
            idea, torch.tensor(0), atoms, values, valid,
            torch.tensor([[[97, 98]]]), torch.ones(1, 1, 2, dtype=torch.bool), True)
        assert cost.isfinite().all()
        gradient, = torch.autograd.grad(cost.sum(), (idea,))
        assert gradient.isfinite().all() and gradient.abs().sum() > 0
        assert not valid[:, 1].any()
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize("name", ["lift", "lower"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_tied_affine_inverse_uses_the_retained_operand(tmp_path, name, side):
    model = _model(tmp_path)
    try:
        language = model.languageSpace
        binary = language._tree_layer(2)
        index = list(binary.op_names).index(name)
        op = getattr(binary.ops[index], "gl", binary.ops[index])
        inner = op._sigma if name == "lift" else op._pi
        width = int(model.conceptualSpace.stm.concept_dim)
        with torch.no_grad():
            inner.layer.biasWeight.fill_(.13)
        left = torch.linspace(-.2, .3, width).reshape(1, width)
        right = torch.linspace(.1, -.15, width).reshape(1, width)
        parent = op.compose(left, right).detach().requires_grad_()
        recovered = language.reverse_binary_step(
            parent, torch.tensor([index]), torch.tensor([True]),
            left if side == "left" else right, reference_side=side)
        torch.testing.assert_close(recovered[0], left, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(recovered[1], right, atol=2e-5, rtol=2e-5)
        gradients = torch.autograd.grad(
            sum(child.sum() for child in recovered), (parent, inner.layer.raw_L))
        assert all(g.isfinite().all() and g.abs().sum() > 0 for g in gradients)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_absorbed_marker_uses_its_own_reference(tmp_path):
    from Language import PartLayer, WholeLayer, PrepositionLayer, ContextualBindLayer
    model = _model(tmp_path)
    try:
        parent = torch.tensor([[.2, .7]])
        marker = torch.tensor([[-.1, .3]])
        for op, side in ((PartLayer(), "left"), (PrepositionLayer(), "left"),
                         (WholeLayer(), "right"), (ContextualBindLayer(), "right")):
            left, right = model.languageSpace.reverse_binary_step(
                parent, torch.tensor([0]), torch.tensor([True]), marker,
                reference_side=side, ops=[op])
            torch.testing.assert_close(left if side == "left" else right, marker)
            torch.testing.assert_close(right if side == "left" else left, parent)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_product_recovers_nonzero_operand_residual(tmp_path):
    from Language import ProductLayer
    model = _model(tmp_path)
    try:
        left, right = torch.tensor([[.2, -.3]]), torch.tensor([[-.4, .5]])
        recovered = model.languageSpace.reverse_binary_step(
            left * right, torch.tensor([0]), torch.tensor([True]), right,
            ops=[ProductLayer()])
        torch.testing.assert_close(recovered[0], left)
        torch.testing.assert_close(recovered[1], right)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_unselected_unary_is_not_evaluated(tmp_path, monkeypatch):
    from Language import NotLayer, NonLayer
    model = _model(tmp_path)
    unused = NonLayer()
    calls = []
    monkeypatch.setattr(unused, "reverse", lambda x: calls.append(True) or x)
    try:
        model.languageSpace.reverse_unary_step(
            torch.tensor([[.2, .7]]), torch.tensor([0]), torch.tensor([True]),
            ops=[NotLayer(), unused])
        assert calls == []
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_snapshot_owns_values_across_same_shape_dictionary_update(tmp_path):
    model = _model(tmp_path)
    try:
        _run(model, ["aa bb"])
        isp = model.inputSpace
        reference = model._tensor_pushed_ideas
        _, atoms, values, valid = model._snapshot_tables(reference)
        expected = (atoms.clone(), values.clone(), valid.clone())
        with torch.no_grad():
            isp._ar_concept_lookup_atoms.add_(.2)
            isp._ar_bank_bytes.fill_(17)
            isp._ar_bank_valid.logical_not_()
        for actual, saved in zip((atoms, values, valid), expected):
            torch.testing.assert_close(actual, saved)
        _, next_atoms, next_values, next_valid = model._snapshot_tables(reference)
        assert not torch.equal(next_atoms, atoms)
        assert not torch.equal(next_values, values)
        assert not torch.equal(next_valid, valid)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_snapshot_preserves_backward_after_same_shape_dictionary_update(tmp_path):
    model = _model(tmp_path)
    try:
        _run(model, ["aa bb"])
        isp = model.inputSpace
        reference = model._tensor_pushed_ideas.detach().clone().requires_grad_()
        _, bank, values, valid = model._snapshot_tables(reference)
        ready, targets, target_mask = model._byte_tables(*reference.shape[:2])
        word = torch.tensor(0)
        loss = model._byte_word_cost(
            reference[:, 0], word, bank, values, valid, targets, target_mask, ready).sum()
        expected, = torch.autograd.grad(loss, reference, retain_graph=True)
        with torch.no_grad():
            isp._ar_concept_lookup_atoms.add_(.2)
            isp._ar_bank_bytes.fill_(17)
            isp._ar_bank_valid.logical_not_()
        actual, = torch.autograd.grad(loss, reference)
        torch.testing.assert_close(actual, expected)
        assert actual.isfinite().all() and actual.abs().sum() > 0
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize("limit", [8, 16])
def test_candidate_search_executes_at_most_the_declared_pair_budget(tmp_path, limit):
    from Language import UnionLayer
    model = _model(tmp_path)
    try:
        class CountedUnion(UnionLayer):
            def compose(self, left, right):
                self.pairs = left.numel() // (left.shape[0] * left.shape[-1])
                return super().compose(left, right)

        op = CountedUnion(monotonic=True)
        basis = torch.rand(1, 40, 2) * .7
        model.languageSpace.reverse_binary_step(
            torch.tensor([[.2, .7]]), torch.tensor([0]), torch.tensor([True]),
            ops=[op], basis=basis, basis_valid=torch.ones(1, 40, dtype=torch.bool),
            candidate_limit=limit, return_status=True)
        assert op.pairs == limit * limit
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_missing_absorbed_operand_uses_bounded_compose_candidates(tmp_path):
    from Language import PartLayer
    model = _model(tmp_path)
    try:
        parent = torch.tensor([[.2, .7]])
        basis = torch.tensor([[[.1, .3], [.6, .2], [100., 100.]]])
        left, right, unavailable = model.languageSpace.reverse_binary_step(
            parent, torch.tensor([0]), torch.tensor([True]), ops=[PartLayer()],
            basis=basis, basis_valid=torch.tensor([[True, True, False]]),
            candidate_limit=2, return_status=True)
        assert not unavailable.any()
        # The parent contains no marker identity. This is the declared
        # ambiguous candidate mean, and never the parent's duplicate.
        torch.testing.assert_close(left, basis[:, :2].mean(1))
        torch.testing.assert_close(right, parent)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_unavailable_inverse_is_explicit_without_candidates(tmp_path):
    from Language import UnionLayer
    model = _model(tmp_path)
    try:
        parent = torch.tensor([[.2, .7]])
        left, right, unavailable = model.languageSpace.reverse_binary_step(
            parent, torch.tensor([0]), torch.tensor([True]),
            ops=[UnionLayer()], return_status=True)
        assert unavailable.all()
        assert not torch.equal(left, parent) and not torch.equal(right, parent)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_noninvertible_compose_transform_uses_the_bounded_candidate_path(tmp_path):
    from Language import LiftLayer
    model = _model(tmp_path)
    try:
        op = LiftLayer(nInput=2, nOutput=2, invertible=False)
        parent = torch.tensor([[.2, .7]], requires_grad=True)
        _, _, unavailable = model.languageSpace.reverse_binary_step(
            parent, torch.tensor([0]), torch.tensor([True]), ops=[op], return_status=True)
        assert unavailable.all()
        left, right, unavailable = model.languageSpace.reverse_binary_step(
            parent, torch.tensor([0]), torch.tensor([True]), ops=[op],
            basis=torch.tensor([[[.1, .3], [.6, .2]]]),
            basis_valid=torch.ones(1, 2, dtype=torch.bool), candidate_limit=2,
            return_status=True)
        assert not unavailable.any()
        # LinearLayer.forward uses W; its separately exposed forwardBias is
        # not called by this compose path. Test the actual shared operator.
        parameters = (op._sigma.layer.W,)
        gradients = torch.autograd.grad((left + right).sum(), (parent, *parameters))
        assert all(g.isfinite().all() for g in gradients)
        assert gradients[0].abs().sum() > 0
        assert any(g.abs().sum() > 0 for g in gradients[1:])
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_adverb_uses_its_own_bounded_inverse_and_shared_parameters(tmp_path):
    model = _model(tmp_path)
    try:
        binary = model.languageSpace._tree_layer(2)
        index = list(binary.op_names).index("adverb")
        op = getattr(binary.ops[index], "gl", binary.ops[index])
        width = int(model.conceptualSpace.stm.concept_dim)
        with torch.no_grad():
            op._adv_edit.weight.copy_(torch.eye(width) * .9)
        left = torch.linspace(.45, .65, width).reshape(1, width)
        right = torch.linspace(.2, .4, width).reshape(1, width)
        parent = op.compose(left, right).detach().requires_grad_()
        assert not torch.allclose(parent, left)
        a, b = model.languageSpace.reverse_binary_step(
            parent, torch.tensor([index]), torch.tensor([True]), right)
        torch.testing.assert_close(a, left, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(b, right)
        gradients = torch.autograd.grad(a.sum(), (parent, op._adv_edit.weight))
        assert all(g.isfinite().all() and g.abs().sum() > 0 for g in gradients)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_masked_candidate_values_do_not_poison_parent_gradient(tmp_path):
    from Language import UnionLayer
    model = _model(tmp_path)
    try:
        parent = torch.tensor([[.25, .7]], requires_grad=True)
        basis = torch.tensor([[[.1, .2], [.3, .8], [float("nan"), float("nan")]]])
        left, right, unavailable = model.languageSpace.reverse_binary_step(
            parent, torch.tensor([0]), torch.tensor([True]),
            ops=[UnionLayer(monotonic=True)], basis=basis,
            basis_valid=torch.tensor([[True, True, False]]), candidate_limit=3,
            return_status=True)
        assert not unavailable.any()
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        gradient, = torch.autograd.grad((left + right).sum(), (parent,))
        assert gradient.isfinite().all() and gradient.abs().sum() > 0
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_inactive_batch_row_does_not_poison_the_selected_inverse_gradient(tmp_path):
    from Language import LiftLayer
    model = _model(tmp_path)
    try:
        op = LiftLayer(nInput=2, nOutput=2)
        parent = torch.tensor([[.2, .5], [float("nan"), float("nan")]])
        left, right = model.languageSpace.reverse_binary_step(
            parent, torch.tensor([0, 0]), torch.tensor([True, False]), ops=[op])
        gradient, = torch.autograd.grad(left[0].sum() + right[0].sum(), (op._sigma.layer.raw_L,))
        assert gradient.isfinite().all()
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize("name", ["null", "exist", "tense", "morphology"])
def test_declared_identity_unary_is_available_on_opaque_concepts(tmp_path, name):
    from Language import NullLayer, ExistLayer, TenseLayer, MorphologyLayer
    model = _model(tmp_path)
    try:
        op = {"null": NullLayer, "exist": ExistLayer, "tense": TenseLayer,
              "morphology": MorphologyLayer}[name]()
        idea = torch.tensor([[.2, -.7, .4, .8]])
        torch.testing.assert_close(op.compose(idea), idea)
        actual, unavailable = model.languageSpace.reverse_unary_step(
            idea, torch.tensor([0]), torch.tensor([True]), ops=[op], return_status=True)
        torch.testing.assert_close(actual, idea)
        assert not unavailable.any()
    finally:
        model.End()
        model.symbolSpace.soft_reset()
@pytest.mark.parametrize("promoted", [b"alphabet", b"alph"])
def test_byte_targets_survive_whole_word_and_prefix_promotion(tmp_path, promoted):
    """Radix compression changes the constituent count, never the target bytes."""
    from test_tied_reconstruction_migration import _model
    from test_compiled_word_chunk import _stage_fullgraph_tensor_peer

    model = _model(tmp_path)
    raw = b"alphabet"
    try:
        model.perceptualSpace._online_learning_frozen = True
        store = model.perceptualSpace.percept_store
        counts = []
        for promote in (False, True):
            if promote:
                store.insert(promoted)
            _stage_fullgraph_tensor_peer(model, [raw.decode("ascii")])
            active = model.inputSpace._word_active_mask
            # Exercise the radix stem's legitimate longest-known-prefix
            # tiling directly. The ladder fixture's alternative analysis
            # tiling can retain its previously selected one-byte units.
            pids = store.spell_out(raw)
            width = max(3, len(pids))
            ids = torch.zeros(*active.shape, width, dtype=torch.long)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            ids[0, 0, :len(pids)] = torch.tensor(pids)
            mask[0, 0, :len(pids)] = True
            model.inputSpace._ar_word_part_ids = ids
            model.inputSpace._ar_word_part_mask = mask
            model._stage_snapshot_bytes()
            counts.append(int(model.inputSpace._ar_word_part_mask[0, 0].sum()))
            ready, target, valid = model._byte_tables(*active.shape)
            observed = target[0, 0][valid[0, 0]].tolist()
            assert ready and observed == list(raw), (promote, counts, observed)
            assert model.inputSpace._ar_bank_bytes.shape[-1] >= len(raw) + 1
            # Even without a candidate spelling, every promoted word remains
            # a scoreable target and pays the uniform byte/termination cost.
            idea = torch.zeros(1, 2)
            cost = model._byte_word_cost(
                idea, torch.tensor(0), torch.zeros(1, 1, 2),
                torch.zeros(1, 1, 1, dtype=torch.long),
                torch.zeros(1, 1, 1, dtype=torch.bool), target, valid, ready)
            torch.testing.assert_close(cost, torch.tensor([256.]).log())
            model.End()
            model.symbolSpace.soft_reset()
        assert counts[1] < counts[0]
    finally:
        model.End()
        model.symbolSpace.soft_reset()
