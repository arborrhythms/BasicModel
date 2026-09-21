"""Declared output catalogues share numerical owners, never input choices."""
from types import SimpleNamespace

import pytest
import torch

from test_meronomy_ladder import _build_ladder_variant


def _model(tmp_path, output_loop):
    replacements = [("<training>", "<training>\n      <outputInLoop>true</outputInLoop>")] if output_loop else []
    return _build_ladder_variant(tmp_path, "generation_catalog", replacements)


@pytest.mark.parametrize("output_loop", [False, True])
def test_declared_generation_catalog_shares_one_optimizer_owner(tmp_path, output_loop):
    from Language import TheGrammar

    model = _model(tmp_path, output_loop)
    try:
        language = model.languageSpace
        names = language._generate_binary_names + language._generate_unary_names
        assert set(names) == {rule.method_name for rule in TheGrammar.rules_downward}
        assert bool(language.generate_policy is not None) is output_loop
        shared = model.symbolSpace.subspace._resolve_rule_layer("CS", "lift")
        assert language.resolve_generation_op("CS", "lift") is shared
        assert shared in language._generate_binary_ops
        opt = model.getOptimizer(lr=1e-3)
        owned = [id(p) for group in opt.param_groups for p in group["params"]]
        dedicated = {id(p) for p in model.synthesis_parameters()}
        for parameter in shared.parameters():
            assert owned.count(id(parameter)) == 1
            assert id(parameter) not in dedicated
        assert not any("generation_ops." in key for key in model.state_dict())
        with pytest.raises(KeyError, match="undeclared"):
            language.resolve_generation_op("CS", "not_a_declared_operator")
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize("output_loop", [False, True])
def test_shared_generation_weights_and_adam_survive_strict_reload(tmp_path, output_loop):
    model = _model(tmp_path, output_loop)
    fresh = None
    try:
        model._materialize_answer_path()
        shared = model.languageSpace.resolve_generation_op("CS", "lift")
        opt = model.getOptimizer(lr=.001)
        model._optimizer = opt
        sum(p.square().sum() + .25 * p.sum() for p in shared.parameters()).backward()
        opt.step()
        expected = {name: p.detach().clone() for name, p in shared.named_parameters()}
        moments = {name: {key: value.detach().clone() for key, value in opt.state[p].items()}
                   for name, p in shared.named_parameters()}
        checkpoint = tmp_path / "shared_catalog.ckpt"
        model.save_weights(str(checkpoint))
        fresh = _model(tmp_path, output_loop)
        assert fresh.load_weights(str(checkpoint), strict=True, require_match=True)
        restored = fresh.languageSpace.resolve_generation_op("CS", "lift")
        assert restored is fresh.symbolSpace.subspace._resolve_rule_layer("CS", "lift")
        restored_opt = fresh.getOptimizer(lr=.001)
        owned = [id(p) for group in restored_opt.param_groups for p in group["params"]]
        for name, parameter in restored.named_parameters():
            assert owned.count(id(parameter)) == 1
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
            for key, value in moments[name].items():
                torch.testing.assert_close(restored_opt.state[parameter][key], value, rtol=0, atol=0)
        restored_opt.zero_grad(set_to_none=True)
        sum(p.square().sum() for p in restored.parameters()).backward()
        restored_opt.step()
        assert any(not torch.equal(p, expected[name]) for name, p in restored.named_parameters())
        # The catalogue adds no numerical checkpoint keys. A missing host
        # parameter is still a strict error, never filled from a second copy.
        saved = torch.load(checkpoint, weights_only=False)
        parameter = next(restored.parameters())
        name = next(name for name, p in fresh.named_parameters() if p is parameter)
        del saved["state_dict"][name]
        torch.save(saved, checkpoint)
        assert fresh.load_weights(str(checkpoint), strict=True, require_match=True) is False
    finally:
        for instance in (model, fresh):
            if instance is not None:
                instance.End()
                instance.symbolSpace.soft_reset()


def test_alias_catalog_reordering_preserves_shared_identity_without_rng_or_state(monkeypatch):
    import Language
    import util

    source = torch.nn.Linear(4, 4)
    coordinator = SimpleNamespace(
        languageLayer=None, muxedSize=4, _grammar_is_default_only=False,
        _host_layer_registry={("CS", "sum"): source},
        _resolve_rule_layer=lambda role, name: source)
    grammar = SimpleNamespace(rules_upward=[], rule_table=[], rules_downward=[])
    monkeypatch.setattr(Language, "TheGrammar", grammar)
    monkeypatch.setattr(util.TheXMLConfig, "training", lambda name, default=None: False)

    def catalog(roles):
        grammar.rules_downward = [SimpleNamespace(
            space_role=role, method_name="sum", lhs="X,Y",
            canonical="X,Y = sum.reverse(Z)", width_min=None, width_max=None)
            for role in roles]
        before = torch.get_rng_state().clone()
        result = Language.LanguageSpace(SimpleNamespace(subspace=coordinator))
        assert torch.equal(torch.get_rng_state(), before)
        assert result.resolve_generation_op("CS", "sum") is source
        assert result.resolve_generation_op("SS", "sum") is source
        assert not list(result.parameters())
        assert not result.state_dict()
        return result

    first = catalog(("CS", "SS"))
    second = catalog(("SS", "CS"))
    assert first._generate_binary_ops == second._generate_binary_ops == (source, source)
    grammar.rules_downward.clear()
    assert first.resolve_generation_op("CS", "sum") is source


def test_default_catalog_does_not_borrow_a_natural_fold_from_another_space(monkeypatch):
    import Language
    import util

    shared = torch.nn.Linear(4, 4)
    rules = [SimpleNamespace(space_role="SS", method_name="sigma", lhs=role,
                             canonical=f"{role} -> sigma.reverse({role})",
                             width_min=None, width_max=None) for role in ("P", "S")]
    grammar = SimpleNamespace(rules_upward=[], rules_downward=rules,
                              rules=rules, rule_table=[])
    coordinator = SimpleNamespace(
        languageLayer=None, muxedSize=4, _grammar_is_default_only=True,
        _host_layer_registry={("SS", "sigma"): shared},
        _default_generate_rules=lambda: {"subsymbolic": [[0]], "SS": [[1]]},
        _resolve_rule_layer=lambda role, name: shared)
    monkeypatch.setattr(Language, "TheGrammar", grammar)
    monkeypatch.setattr(util.TheXMLConfig, "training", lambda name, default=None: False)
    language = Language.LanguageSpace(SimpleNamespace(subspace=coordinator))
    assert language._generate_unary_rule_ids.tolist() == [1]
    assert language._generate_unary_ops == (shared,)
    assert language.resolve_generation_op("SS", "sigma") is shared
    with pytest.raises(KeyError, match="undeclared"):
        language.resolve_generation_op("subsymbolic", "sigma")


def test_output_dispatch_scope_uses_declared_shared_inverse_and_resets_on_error(tmp_path, monkeypatch):
    from Language import TheGrammar
    from test_subspace_what_stm_contract import (
        _make_minimal_signal_router, _make_stack_subspace_with_where)

    model = _model(tmp_path, False)
    try:
        language = model.languageSpace
        source = model.symbolSpace.subspace._resolve_rule_layer("CS", "lift")
        monkeypatch.setattr(source, "reverse", lambda parent, **kwargs: (parent + .1, parent - .1))
        router = _make_minimal_signal_router(D=8)
        rule = next(i for i, r in enumerate(TheGrammar.rules_upward) if r.method_name == "lift")
        syntactic = SimpleNamespace(space_role="CS", _word_space=model.symbolSpace.subspace,
                                    _by_name={"lift": source})
        calls = []
        resolve = language.resolve_generation_op

        def record(role, name):
            calls.append((role, name))
            return resolve(role, name)

        monkeypatch.setattr(language, "resolve_generation_op", record)

        def read():
            sub = _make_stack_subspace_with_where(B=1, K=4, D=8, W=2)
            router.shift(sub, torch.zeros(1, 8), where_id=TheGrammar.where_id_for_rule(rule))
            router.unreduce(sub, syntactic, grammar=TheGrammar)
            return sub.materialize(mode="what")[0, 0]

        expected = read()
        assert not calls
        with pytest.raises(RuntimeError, match="probe output failure"):
            with language.generation_scope():
                torch.testing.assert_close(read(), expected)
                assert calls == [("CS", "lift")]
                with language.generation_scope():
                    assert language._generation_active
                assert language._generation_active
                raise RuntimeError("probe output failure")
        calls.clear()
        torch.testing.assert_close(read(), expected)
        assert not calls
        # Removing a declaration must fail in output; a compose registration
        # cannot silently provide an undeclared generation alternative.
        monkeypatch.delitem(language._generation_resolved, ("CS", "lift"))
        with language.generation_scope(), pytest.raises(KeyError, match="undeclared"):
            read()
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_ordinary_unary_output_keeps_the_shared_host_inverse(tmp_path, monkeypatch):
    from pathlib import Path
    from Language import TheGrammar
    from test_output_path_supervised import _build

    root = Path(__file__).resolve().parents[1]
    config = (root / "data/MM_xor.xml").read_text().replace(
        "<grammar>xor.grammar</grammar>",
        "<grammar><P>sigma(P)</P><C>pi(C)</C><S>sigma(S)</S></grammar>")
    config = config.replace("<architecture>", "<architecture><answerSynthesis>true</answerSynthesis>", 1)
    path = tmp_path / "flat_natural_folds.xml"
    path.write_text(config)
    model = _build(path, "xor")
    try:
        owner, language = model.symbolSpace.subspace, model.languageSpace
        assert owner._grammar_is_default_only and not TheGrammar.rules_downward
        count = 0
        for role, rows in owner._default_generate_rules().items():
            for rule_id in rows[0]:
                method = TheGrammar.rules[rule_id].method_name
                shared = owner._host_layer_registry.get((role, method))
                if shared is not None:
                    assert language.resolve_generation_op(role, method) is shared
                    count += 1
        assert count
        # Exercise the ordinary dispatcher with the actual shared sigma.
        # Its host adapter may implement two passes; output must retain it.
        from Language import SyntacticLayer
        shared = language.resolve_generation_op("SS", "sigma")
        calls = []
        layer = SyntacticLayer.__new__(SyntacticLayer)
        torch.nn.Module.__init__(layer)
        object.__setattr__(layer, "_word_space", owner)
        object.__setattr__(layer, "_by_name", {"sigma": shared})
        layer.space_role = "SS"
        object.__setattr__(layer, "_host_space", SimpleNamespace(
            _sigma_reverse=lambda value: calls.append("host") or value * 2))
        monkeypatch.setattr(layer, "_next_rule_name", lambda **kwargs: "sigma")
        monkeypatch.setattr(layer, "_read_subspace", lambda sub, **kwargs: sub.value)
        monkeypatch.setattr(layer, "_write_subspace", lambda sub, value, **kwargs: setattr(sub, "value", value))
        resolve = language.resolve_generation_op

        def record(role, name):
            calls.append((role, name))
            return resolve(role, name)

        monkeypatch.setattr(language, "resolve_generation_op", record)
        # The normal prepared-answer entry must open this scope itself.
        from What import What
        model.eval()
        with torch.no_grad():
            inputs = model.inputSpace.prepInput(list(model.inputSpace.data.train_input[:2]))
            understanding = model.understand(inputs)
            derivation = model.resolveAnswer(understanding, (What.supervised(0), What.supervised(1)))
            construction = model.reverseOutput(understanding, derivation)
        assert torch.isfinite(construction.actual).all()
        assert ("SS", "sigma") in calls
        assert not language._generation_active
        calls.clear()
        value = torch.full((1, 1, shared.nInput), .1)
        with language.generation_scope():
            got = layer.reverse(SimpleNamespace(value=value))
        torch.testing.assert_close(got.value, value * 2)
        assert calls == [("SS", "sigma"), "host"]
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize("output_loop", [False, True])
def test_normal_supervised_output_respects_gradient_contract(tmp_path, monkeypatch, output_loop):
    """A real runBatch/Adam step, with no forced generate actions or answer seeds."""
    import json
    import util
    from What import What
    from test_output_path_supervised import _native_answer_model

    monkeypatch.setattr(util, "TheCompileBackend", "none")
    model = _native_answer_model(tmp_path, output_loop)
    optimizer = model.getOptimizer(lr=.001)
    recorded, observed = {}, {}
    record, backward = model.record_loss, model._backward_training_loss

    def record_loss(name, value, **kwargs):
        recorded[name] = value
        return record(name, value, **kwargs)

    def check_gradients(total, amp_scaler=None):
        construction = model._last_answer_construction
        loss = recorded["output"]
        assert loss.requires_grad and float(loss.detach()) > 0
        sources = [program.end_state for program in construction.derivation.program]
        sources = [value for value in sources if value.requires_grad]
        assert sources, "the forward must carry a live conclusion for this boundary probe"
        groups = model._shared_operator_parameter_groups(optimizer)
        named = [(name, p) for name, params in groups.items() for p in params]
        conditioner = model.question_conditioners[str(model.conceptualSpace.stm.concept_dim)].weight
        gradients = torch.autograd.grad(loss, [conditioner] + sources + [p for _, p in named],
                                        retain_graph=True, allow_unused=True)
        assert gradients[0] is not None and bool(gradients[0].any())
        assert all(g is None or not bool(g.any()) for g in gradients[1:1 + len(sources)])
        reached = [(name, p, float(g.detach().norm())) for (name, p), g in
                   zip(named, gradients[1 + len(sources):]) if g is not None and bool(g.any())]
        # This fixture's ordinary path uses dedicated synthesis and a
        # parameter-free reverse. Its grammar walk executes shared maps.
        # Do not demand credit for an operator that never participated.
        if output_loop:
            assert reached, "the active output walk must train a shared numerical map"
        observed["reached"] = [(name, p, p.detach().clone(), norm) for name, p, norm in reached]
        observed["conditioner"] = (conditioner, conditioner.detach().clone())
        observed["loss"] = float(loss.detach())
        return backward(total, amp_scaler)

    monkeypatch.setattr(model, "record_loss", record_loss)
    monkeypatch.setattr(model, "_backward_training_loss", check_gradients)
    try:
        batch = (model.inputSpace.prepInput(["1 plus 2", "3 plus 4"]), torch.zeros(2, 1, 1))
        torch.manual_seed(11)
        model.runBatch(train=True, batchSize=2, split="train", optimizer=optimizer,
                       batch_override=batch, questions=(What.supervised(0), What.supervised(1)))
        owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
        stepped = [(name, norm) for name, p, before, norm in observed["reached"]
                   if not torch.equal(p, before) and optimizer.state.get(p)]
        if output_loop:
            assert stepped
        conditioner, before = observed["conditioner"]
        assert not torch.equal(conditioner, before) and optimizer.state.get(conditioner)
        assert all(owned.count(id(p)) == 1 for _, p, _, _ in observed["reached"])
        print("generation-gradient-evidence " + json.dumps({
            "output_loop": output_loop, "seed": 11, "output_loss": observed["loss"],
            "stepped_shared_groups": sorted({name for name, _ in stepped}),
            "conclusion_gradient": 0}))
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
