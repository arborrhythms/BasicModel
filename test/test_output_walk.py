"""Compiled reverse-loops plan, slice 3: the answer-side generate walk as
the second compiled loop (doc/plans/2026-09-12-compiled-reverse-loops.md)."""
import sys
from pathlib import Path

import torch
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "bin"), str(_ROOT / "test")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_meronomy_ladder import _build_ladder  # noqa: E402


def _model():
    import tempfile
    from pathlib import Path as _P
    from test_meronomy_ladder import _build_ladder_variant
    m = _build_ladder_variant(_P(tempfile.mkdtemp()), "walk", [
        ("<packSentences>false</packSentences>",
         "<packSentences>false</packSentences>\n      <outputInLoop>true</outputInLoop>")])
    m.output_in_loop = True
    return m


def _stamped_event(m, rule_name, N=4):
    """A ``[1, N, D]`` event whose top live slot is stamped with the
    global rule id of ``rule_name``; slots below it are terminals."""
    import Language
    language = m.languageSpace
    binary = language._tree_layer(2)
    names = list(binary.op_names)
    idx = names.index(rule_name)
    global_id = int(language._cs_binary_rule_ids[idx])
    gl = getattr(list(binary.ops)[idx], "gl", list(binary.ops)[idx])
    cw = int(m.symbolSpace.subspace.nWhat)                 # the stamp channel
    D = int(m.conceptualSpace.stm.concept_dim)
    torch.manual_seed(0)
    event = torch.zeros(1, N, D)
    live = 2
    event[0, :live, :cw] = torch.tanh(torch.randn(live, cw) * 0.5)
    event[0, 0, cw] = 0.0                                                   # a plain concept slot
    event[0, 1, cw] = float(Language.TheGrammar.where_id_for_rule(global_id))  # the rule on top
    return event, gl, cw, live


def _prefer(m, choice):
    """Bias the generate policy towards one choice (a rule index or stop)."""
    policy = m.languageSpace.generate_policy
    with torch.no_grad():
        policy.weight.zero_(); policy.bias.zero_(); policy.bias[choice] = 5.0


def _stop(m):
    _prefer(m, int(m.languageSpace.generate_policy.out_features) - 1)


def test_walk_unreduces_a_rule_stamped_top_and_emits_its_constituents():
    """A rule on top is un-reduced with the tied inverse; the completed
    constituents are then popped into the emitted sequence, left to right:
    the plain slot below, the left child, the right child."""
    m = _model()
    _stop(m)
    for rule in ("lift", "lower"):
        names = list(m.languageSpace._tree_layer(2).op_names)
        if rule not in names:
            continue
        event, gl, cw, live = _stamped_event(m, rule)
        out, n_emitted, truncated, _cost = m._output_generate_walk(event, budget=8)
        assert n_emitted.tolist() == [live + 1]
        assert not bool(truncated.any())
        parent = event[0, 1]
        below, left, right = out[0, 0], out[0, 1], out[0, 2]
        assert torch.equal(below, event[0, 0])                       # the terminal below, first
        assert float(left[cw]) == 0.0 and float(right[cw]) == 0.0    # children stamped empty
        recomposed = gl.compose(left.unsqueeze(0), right.unsqueeze(0))[0]
        assert torch.allclose(recomposed[:cw], parent[:cw], atol=1e-3), rule
        assert float(out[0, 3:].abs().sum()) == 0.0                  # nothing beyond the emitted


def test_walk_emits_terminals_in_order_and_reports_no_truncation():
    m = _model()
    _stop(m)
    event, gl, cw, live = _stamped_event(m, "lift")
    event[0, 1, cw] = 0.0                                   # no rule stamp on top
    out, n_emitted, truncated, _cost = m._output_generate_walk(event, budget=8)
    assert n_emitted.tolist() == [live] and not bool(truncated.any())
    assert torch.equal(out[0, :live], event[0, :live])       # left to right


def test_walk_reports_truncation_when_work_is_pending():
    """No free slot for the rule's second child and the budget runs out
    with the rule still on top: pending work, reported as truncation."""
    m = _model()
    _stop(m)
    event, gl, cw, live = _stamped_event(m, "lift", N=2)
    out, n_emitted, truncated, _cost = m._output_generate_walk(event, budget=3)
    assert bool(truncated.all()) and n_emitted.tolist() == [0]


def test_walk_compiles_fullgraph_and_matches_eager():
    m = _model()
    _stop(m)
    event, gl, cw, live = _stamped_event(m, "lift")
    eager = m._output_generate_walk(event, budget=8)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(m._output_generate_walk, backend="eager", fullgraph=True)
    try:
        out, n_emitted, truncated, cost = compiled(event, 8)
        assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) == 1
        assert torch.allclose(out, eager[0], atol=1e-6)
        assert torch.equal(n_emitted, eager[1]) and torch.equal(truncated, eager[2])
        assert torch.allclose(cost, eager[3], atol=1e-6)
    finally:
        torch._dynamo.reset()


def test_walk_handles_mixed_output_lengths_per_row():
    """Output gate: one row un-reduces (three words), the other emits its
    two terminals; per-row lengths differ and neither row truncates."""
    m = _model()
    _stop(m)
    e0, gl, cw, live = _stamped_event(m, "lift")
    e1 = e0.clone()
    e1[0, 1, cw] = 0.0                                      # row 1: terminal on top
    event = torch.cat((e0, e1), dim=0)
    out, n_emitted, truncated, _cost = m._output_generate_walk(event, budget=8)
    assert n_emitted.tolist() == [live + 1, live]
    assert not bool(truncated.any())
    assert torch.equal(out[1, :live], event[1, :live])
    assert float(out[0, 2].abs().sum()) > 0.0               # row 0 emitted a third word


def test_walk_is_invariant_to_reconstruction_only_state():
    """Output gate: the walk reads nothing of the input reconstruction
    (contract 5): changing the reconstruction results leaves it unchanged."""
    m = _model()
    _stop(m)
    event, gl, cw, live = _stamped_event(m, "lift")
    base = m._output_generate_walk(event, budget=8)
    D = int(m.conceptualSpace.stm.concept_dim)
    object.__setattr__(m, "_recon_ideas", torch.randn(1, 4, D))
    object.__setattr__(m, "_recon_cost", torch.tensor([7.0]))
    object.__setattr__(m, "_recon_truncated", torch.tensor([True]))
    again = m._output_generate_walk(event, budget=8)
    assert torch.equal(again[0], base[0])
    assert torch.equal(again[1], base[1]) and torch.equal(again[2], base[2])


def test_generate_policy_is_credited_by_the_stamped_rule_and_trains():
    """Contract 5: a stamped top credits the policy by imitation (positive,
    finite cross-entropy whose gradient reaches the policy's weights); the
    policy is the only parameter the walk owns."""
    m = _model()
    language = m.languageSpace
    event, gl, cw, live = _stamped_event(m, "lift")
    names = list(language._tree_layer(2).op_names)
    _prefer(m, names.index("lower"))                        # wrong preference
    m.zero_grad(set_to_none=True)
    out, n_emitted, truncated, cost = m._output_generate_walk(event, budget=8)
    assert cost.shape == (1,) and float(cost) > 1.0 and bool(torch.isfinite(cost))
    cost.sum().backward()
    assert language.generate_policy.weight.grad is not None
    assert float(language.generate_policy.weight.grad.abs().sum()) > 0
    fold = [p.grad for p in gl.parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert not fold                                          # imitation credit trains the policy only


def test_generate_policy_decides_an_unstamped_top():
    """Contract 5: without a rule stamp the policy decides; preferring a
    binary rule un-reduces (work pending after one trip), preferring stop
    emits the live slots as they are."""
    m = _model()
    language = m.languageSpace
    names = list(language._tree_layer(2).op_names)
    event, gl, cw, live = _stamped_event(m, "lift")
    event[0, 1, cw] = 0.0                                   # unstamped top
    _prefer(m, names.index("lift"))
    out, n_emitted, truncated, cost = m._output_generate_walk(event, budget=1)
    assert n_emitted.tolist() == [0] and bool(truncated.all())
    assert float(cost) == 0.0                               # nothing stamped to imitate
    _stop(m)
    out2, n_emitted2, truncated2, _c = m._output_generate_walk(event, budget=8)
    assert n_emitted2.tolist() == [live] and not bool(truncated2.any())
    assert torch.equal(out2[0, :live], event[0, :live])


def test_answer_materialises_as_its_own_conceptual_idea_and_realises_through_the_walk():
    """The answer-materialisation boundary (spec sections 1-2): after
    resolution the answer is its own conceptual idea, [B, 3, D] at the
    concept width (the three LTM slots), the operand of the output loop;
    for the present relation it is the understanding's own end state with
    the question conditioning the root slot (zero at initialisation);
    reverseOutput un-folds it and realises the words through the reverse
    chain, reporting the idea's sources."""
    import tempfile
    from pathlib import Path as _P
    from test_meronomy_ladder import _build_ladder_variant
    from test_compiled_word_chunk import _stage_fullgraph_tensor_peer
    from What import What
    m = _build_ladder_variant(_P(tempfile.mkdtemp()), "walk_e2e", [
        ("<packSentences>false</packSentences>",
         "<packSentences>false</packSentences>\n      <outputInLoop>true</outputInLoop>")])
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    _stage_fullgraph_tensor_peer(m, ["12 plus 1", "3 plus 4"])
    with torch.no_grad():
        out = m._forward_with_compiled_sentence_state(None)
        m._publish_compiled_sentence_state(out)
        u = m._capture_understanding(out[:4] if isinstance(out, tuple) else out)
        derivation = m._resolve_answer(u, What.supervised(0))
        idea, resolved, sources, targets = m._materialize_answer_idea(u, derivation, What.supervised(0))
    D = int(m.conceptualSpace.stm.concept_dim)
    assert tuple(idea.shape) == (2, 3, D) and bool(resolved.all())
    assert all(src.startswith("idea:") for src in sources)
    end = m._sentence_end_state(None)
    assert torch.allclose(idea, end)                        # zero-initialised conditioning
    T = m._walk_budget()
    assert torch.is_tensor(targets) and tuple(targets.shape) == (2, T)
    with torch.no_grad():
        words, n_emitted, truncated, cost = m._output_generate_walk(
            m._walk_operand(idea), T, False, targets)
    assert tuple(words.shape) == (2, T, D)
    assert bool((n_emitted >= 1).all())                     # each row emitted at least its root
    m.End(); m.symbolSpace.soft_reset()


def test_generate_policy_is_credited_by_the_derivation_on_conceptual_slots():
    """An opaque conceptual idea carries no rule stamp, so the walk's
    chooser is credited by the teacher actions of the idea's own
    derivation (the recorded folds of its sentence, seals first, then per
    word its unary, post-binary, pop and pre-binary): followed under
    teacher forcing the walk pops every word of the sentence without
    truncation, and the policy's parameters receive the gradient of the
    imitation cost."""
    from test_compiled_word_chunk import _stage_fullgraph_tensor_peer
    m = _model()
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    _stage_fullgraph_tensor_peer(m, ["12 plus 1", "3 plus 4"])
    with torch.no_grad():
        out = m._forward_with_compiled_sentence_state(None)
        m._publish_compiled_sentence_state(out)
    T = m._walk_budget()
    targets = m._derivation_targets(None, T)
    language = m.languageSpace
    R2 = int(language._cs_binary_rule_ids.numel())
    R1 = int(language._cs_unary_rule_ids.numel())
    stop = R2 + R1
    n_words = m.inputSpace._word_active_mask.to(torch.long).sum(1)
    assert tuple(targets.shape) == (2, T)
    assert torch.equal((targets == stop).sum(1), n_words)        # one pop per word
    assert bool((targets >= -1).all()) and bool((targets <= stop).all())
    assert bool((targets[:, -1] == -1).all())                     # exhausted well inside the budget
    idea = m._sentence_end_state(None)
    words, n_emitted, truncated, cost = m._output_generate_walk(
        m._walk_operand(idea), T, False, targets)
    assert torch.equal(n_emitted, n_words) and not bool(truncated.any())
    params = list(language.generate_policy.parameters())
    grads = torch.autograd.grad(cost.sum(), params, allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)
    m.End(); m.symbolSpace.soft_reset()


def _policy_training_probe(m, opt, questions):
    """Observe the real loss and backward; do not replace any loss term."""
    params = list(m.languageSpace.generate_policy.parameters())
    observed, recorded = {}, {}
    backward, record = m._backward_training_loss, m.record_loss

    def record_probe(name, value, **kwargs):
        recorded[name] = value
        return record(name, value, **kwargs)

    def backward_probe(total, objectives, optimizer, amp_scaler=None):
        cost = m._output_policy_cost
        observed["raw_grads"] = torch.autograd.grad(
            cost.sum(), params, retain_graph=True, allow_unused=True)
        observed["total_grads"] = torch.autograd.grad(
            total, params, retain_graph=True, allow_unused=True)
        credit = recorded.get("output_policy")
        if torch.is_tensor(credit) and credit.requires_grad:
            observed["credit_grads"] = torch.autograd.grad(
                credit, params, retain_graph=True, allow_unused=True)
            observed["row_credit"] = torch.autograd.grad(
                credit, cost, retain_graph=True, allow_unused=True)[0]
            target = m._last_answer_target
            pred = m._align_output_pred(m._last_answer_construction.actual, target)
            with torch.no_grad():
                observed["answer_errors"] = torch.stack([
                    m.loss.compute(pred[b:b + 1], target[b:b + 1])
                    for b in range(len(questions))])
        return backward(total, objectives, optimizer, amp_scaler)

    m.record_loss, m._backward_training_loss = record_probe, backward_probe
    before = [p.detach().clone() for p in params]
    try:
        batch = (m.inputSpace.prepInput(["12 plus 1", "3 plus 4"]),
                 torch.zeros(2, 1, 1))
        torch.manual_seed(11)
        m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch, questions=questions)
    finally:
        m.record_loss, m._backward_training_loss = record, backward
    observed["changed"] = [not torch.equal(p, old)
                           for p, old in zip(params, before)]
    return observed


@pytest.mark.parametrize("weight", [1.0, 0.0])
def test_runbatch_trains_generate_policy_only_with_nonzero_weight(weight):
    """Codex item 1: standalone credit, total-loss credit and optimizer
    ownership must all agree on a real supervised training batch."""
    from What import What
    m = _model()
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    m.output_policy_weight = weight
    opt = m.getOptimizer(lr=1e-3)
    try:
        got = _policy_training_probe(
            m, opt, (What.supervised(0), What.supervised(1)))
        assert any(g is not None and bool(g.abs().sum() > 0)
                   for g in got["raw_grads"])
        if weight:
            assert any(g is not None and bool(g.abs().sum() > 0)
                       for g in got["total_grads"])
            for total, credit in zip(got["total_grads"], got["credit_grads"]):
                torch.testing.assert_close(total, weight * credit)
            # The first return baseline is zero: supplied answer error,
            # not reconstruction or input-parse imitation, supplies credit.
            torch.testing.assert_close(got["row_credit"], -got["answer_errors"] / 2)
            assert all(got["changed"])
        else:
            assert all(g is None or not bool(g.abs().any())
                       for g in got["total_grads"])
            assert not any(got["changed"])
        owned = [id(p) for group in opt.param_groups for p in group["params"]]
        assert all(owned.count(id(p)) == 1
                   for p in m.languageSpace.generate_policy.parameters())
    finally:
        m.End()
        m.symbolSpace.soft_reset()


def test_runbatch_does_not_train_generate_policy_without_supplied_answers():
    """No supervised target means no output-policy update, including Adam
    momentum left by a preceding supervised batch."""
    from What import What
    m = _model()
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    m.output_policy_weight = 1.0
    opt = m.getOptimizer(lr=1e-3)
    data = m.inputSpace.data
    had_outputs = data.has_supervised_outputs
    try:
        learned = _policy_training_probe(
            m, opt, (What.supervised(0), What.supervised(1)))
        assert all(learned["changed"])
        data.has_supervised_outputs = False
        unlabelled = _policy_training_probe(
            m, opt, (What.supervised(0), What.supervised(1)))
        assert not any(unlabelled["changed"])
        assert all(g is None or not bool(g.abs().any())
                   for g in unlabelled["total_grads"])
        present = _policy_training_probe(
            m, opt, (What.present(0), What.present(1)))
        assert not any(present["changed"])
        # A batch that does not run reverseOutput must not reuse the last
        # batch's graph or policy credit.
        m.answer_synthesis = False
        before = [p.detach().clone() for p in m.languageSpace.generate_policy.parameters()]
        batch = (m.inputSpace.prepInput(["12 plus 1", "3 plus 4"]),
                 torch.zeros(2, 1, 1))
        m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch,
                   questions=(What.supervised(0), What.supervised(1)))
        assert m._output_policy_cost is None
        assert all(torch.equal(p, old) for p, old in zip(
            m.languageSpace.generate_policy.parameters(), before))
    finally:
        data.has_supervised_outputs = had_outputs
        m.End()
        m.symbolSpace.soft_reset()


def test_runbatch_generate_policy_masks_rows_without_supplied_answers():
    from What import What
    m = _model()
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    m.output_policy_weight = 1.0
    opt = m.getOptimizer(lr=1e-3)
    try:
        got = _policy_training_probe(
            m, opt, (What.supervised(0), What.inference(1, split="train")))
        row_credit = got["row_credit"]
        assert row_credit is not None and bool(row_credit[0].abs() > 0)
        assert float(row_credit[1]) == 0.0
        torch.testing.assert_close(row_credit[0], -got["answer_errors"][0])
        assert all(got["changed"])
    finally:
        m.End()
        m.symbolSpace.soft_reset()


def test_question_conditioners_persist_per_answer_width():
    """The symbol-width conditioner (the resolved symbol) and the
    concept-width one (the materialised idea) are kept side by side:
    switching widths returns the existing module with its weights, and
    both are in the state dict."""
    m = _model()
    dev, dt = torch.device("cpu"), torch.float32
    narrow = m._question_conditioner(136, device=dev, dtype=dt)
    with torch.no_grad():
        narrow.weight.fill_(0.5)
    wide = m._question_conditioner(1032, device=dev, dtype=dt)
    assert wide is not narrow and wide.out_features == 1032
    again = m._question_conditioner(136, device=dev, dtype=dt)
    assert again is narrow and float(again.weight.abs().sum()) > 0
    assert set(m.question_conditioners.keys()) == {"136", "1032"}
    keys = [k for k in m.state_dict() if k.startswith("question_conditioners.")]
    assert any(k.startswith("question_conditioners.136.") for k in keys)
    assert any(k.startswith("question_conditioners.1032.") for k in keys)
    m.End(); m.symbolSpace.soft_reset()


def _conditioner_pair(m):
    modules = [m._question_conditioner(width, device=torch.device("cpu"),
                                       dtype=torch.float32)
               for width in (136, 1032)]
    with torch.no_grad():
        for i, module in enumerate(modules):
            module.weight.fill_(0.125 * (i + 1))
    return modules


def test_question_conditioner_checkpoint_reloads_both_widths_strictly(tmp_path):
    """Codex item 5: reload both saved widths, including a checkpoint whose
    singular compatibility alias points to the last-created, wider module."""
    m = _model()
    modules = _conditioner_pair(m)
    m._materialize_answer_path()
    assert m.question_conditioner is modules[1]
    checkpoint = tmp_path / "conditioners.ckpt"
    m.save_weights(str(checkpoint))
    fresh = _model()
    try:
        assert fresh.load_weights(str(checkpoint), strict=True, require_match=True)
        assert set(fresh.question_conditioners) == {"136", "1032"}
        for module in modules:
            restored = fresh.question_conditioners[str(module.out_features)]
            torch.testing.assert_close(restored.weight, module.weight, rtol=0, atol=0)
        assert fresh.question_conditioner is fresh.question_conditioners["136"]
        assert set(fresh.state_dict()) == set(m.state_dict())
    finally:
        for model in (m, fresh):
            model.End()
            model.symbolSpace.soft_reset()


def test_question_conditioner_optimizer_steps_both_widths():
    """Both widths join the real runBatch optimizer once and can be stepped,
    even when only one width is used by that batch's answer path."""
    from What import What
    m = _model()
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    modules = _conditioner_pair(m)
    opt = m.getOptimizer(lr=1e-3)
    try:
        batch = (m.inputSpace.prepInput(["12 plus 1", "3 plus 4"]),
                 torch.zeros(2, 1, 1))
        m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch,
                   questions=(What.supervised(0), What.supervised(1)))
        owned = [id(p) for group in opt.param_groups for p in group["params"]]
        answer_only = [id(p) for p in m.synthesis_parameters()]
        for module in modules:
            assert owned.count(id(module.weight)) == 1
            assert answer_only.count(id(module.weight)) == 1
        before = [module.weight.detach().clone() for module in modules]
        opt.zero_grad()
        probe = sum(module(torch.ones(2, module.in_features)).square().mean()
                    for module in modules)
        probe.backward()
        opt.step()
        for module, previous in zip(modules, before):
            assert not torch.equal(module.weight, previous)
        m._collect_fresh_synthesis_modules()
        assert not m._fresh_synthesis_params
    finally:
        m.End()
        m.symbolSpace.soft_reset()


def test_question_conditioner_legacy_singular_checkpoint_reloads_strictly(tmp_path):
    m = _model()
    m._materialize_answer_path()
    with torch.no_grad():
        m.question_conditioner.weight.fill_(0.25)
    checkpoint = tmp_path / "legacy-conditioner.ckpt"
    m.save_weights(str(checkpoint))
    saved = torch.load(checkpoint, weights_only=False)
    state = saved["state_dict"]
    for key in list(state):
        if key.startswith("question_conditioners."):
            del state[key]
    torch.save(saved, checkpoint)
    fresh = _model()
    try:
        assert fresh.load_weights(str(checkpoint), strict=True, require_match=True)
        torch.testing.assert_close(fresh.question_conditioner.weight,
                                   m.question_conditioner.weight, rtol=0, atol=0)
        assert fresh.question_conditioner is fresh.question_conditioners["136"]
    finally:
        for model in (m, fresh):
            model.End()
            model.symbolSpace.soft_reset()


def test_materialised_idea_follows_the_symbol_rows():
    """The symbol table and the concept table share row indices, so the
    answer is materialised from its symbols' rows and its derivation: the
    concept dictionary rows at the rows, folded by the recorded derivation
    through the grammar's forward ops.  With the rows the forward pushed,
    the replay reproduces the forward's end state exactly; a different
    symbol at a word (its row exchanged with its neighbour's) gives a
    different idea, without any snap of a symbol vector."""
    from test_compiled_word_chunk import _stage_fullgraph_tensor_peer
    from What import What
    m = _model()
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    _stage_fullgraph_tensor_peer(m, ["12 plus 1", "3 plus 4"])
    with torch.no_grad():
        out = m._forward_with_compiled_sentence_state(None)
        m._publish_compiled_sentence_state(out)
        u = m._capture_understanding(out[:4] if isinstance(out, tuple) else out)
        derivation = m._resolve_answer(u, What.supervised(0))
        idea0, resolved, sources, targets = m._materialize_answer_idea(u, derivation, What.supervised(0))
        end = m._sentence_end_state(None)
        assert torch.allclose(idea0, end, atol=1e-4), float((idea0 - end).abs().max())
        rows = m._word_symbol_rows()
        assert bool((rows[:, :2] >= 0).all()) and bool((rows[:, 0] != rows[:, 1]).all())
        isp = m.inputSpace
        for name in ("_ar_word_object_rows", "_ar_word_concept_rows"):
            table = getattr(isp, name).clone()
            table[:, :2] = table[:, :2].flip(1)
            setattr(isp, name, table)
        assert torch.equal(m._word_symbol_rows()[:, :2], rows[:, :2].flip(1))
        idea1, _r, _s, targets1 = m._materialize_answer_idea(u, derivation, What.supervised(0))
    assert not torch.allclose(idea1, idea0, atol=1e-4)
    assert torch.equal(targets1, targets)                       # the derivation is unchanged
    m.End(); m.symbolSpace.soft_reset()
