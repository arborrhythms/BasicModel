from Queries import ConceptualSpaceCapability
from AccessibleMind import Subsystem as Mind
"""Reviewer probes for item 0's single structural thought-operation catalog."""

from types import SimpleNamespace

import pytest
import torch

from Language import Grammar
from test_cs_symbol_table import _cs


def _family(*, compose=True, generate=True, thought=True,
            reverse_inputs=("I1", "I2")):
    """Return one minimal structural family, optionally selected for thought."""
    config = {}
    if compose:
        config["compose"] = {
            "rule": ["part_O1 = part.forward(part_I1, part_I2)"]}
    if thought:
        config["thought"] = {
            "rule": ["part_O1 = part.thought(part_I1, part_I2)"]}
    if generate:
        lhs = ", ".join("part_" + role for role in reverse_inputs)
        config["generate"] = {
            "rule": [lhs + " = part.reverse(part_O1)"]}
    return config


def _only_operation(grammar):
    operations = grammar.thought_operations
    assert isinstance(operations, tuple)
    assert len(operations) == 1
    return operations[0]


def test_structural_pair_derives_one_immutable_thought_operation_without_queries():
    """Forward/reverse structural declarations, not <Queries>, own the contract."""
    grammar = Grammar()
    grammar.configure(_family())

    operation = _only_operation(grammar)
    assert operation.semantic_id == "part"
    assert operation.operand_roles == ("I1", "I2")
    assert operation.result_role == "O1"
    assert operation.forward_rule_ids
    assert operation.reverse_rule_ids


@pytest.mark.parametrize(
    ("compose", "generate", "forward", "reverse"),
    [(True, False, True, False), (False, True, False, True)],
)
def test_a_structural_face_in_either_section_enters_the_same_boundary_catalog(
        compose, generate, forward, reverse):
    """A thought-capable family is not restricted to the input parse path."""
    grammar = Grammar()
    grammar.configure(_family(compose=compose, generate=generate))

    operation = _only_operation(grammar)
    assert operation.semantic_id == "part"
    assert bool(operation.forward_rule_ids) is forward
    assert bool(operation.reverse_rule_ids) is reverse


def test_paired_structural_faces_with_inconsistent_roles_fail_at_configuration():
    grammar = Grammar()
    with pytest.raises(ValueError, match="part|role|contract"):
        grammar.configure(_family(reverse_inputs=("I1",)))


def test_thought_section_selects_the_boundary_catalogue_not_all_structural_rules():
    """A model may compose two operators while authorizing only one to think."""
    grammar = Grammar()
    grammar.configure({
        "compose": {"rule": [
            "part_O1 = part.forward(part_I1, part_I2)",
            "equal_O1 = equal.forward(equal_I1, equal_I2)",
        ]},
        "thought": {"rule": [
            "equal_O1 = equal.thought(equal_I1, equal_I2)",
        ]},
        "generate": {"rule": [
            "part_I1, part_I2 = part.reverse(part_O1)",
            "equal_I1, equal_I2 = equal.reverse(equal_O1)",
        ]},
    })

    assert tuple(operation.semantic_id for operation in grammar.thought_operations) == (
        "equal",)


def test_structural_operator_without_thought_declaration_is_not_a_boundary_action():
    """Speech/understanding availability must not imply thought permission."""
    grammar = Grammar()
    grammar.configure(_family(thought=False))

    assert grammar.thought_operations == ()


def test_thought_role_contract_must_match_its_structural_operator():
    grammar = Grammar()
    with pytest.raises(ValueError, match="part|role|contract"):
        grammar.configure({
            "compose": {"rule": [
                "part_O1 = part.forward(part_I1, part_I2)",
            ]},
            "thought": {"rule": [
                "part_O1 = part.thought(part_I2, part_I1)",
            ]},
        })


def test_thought_only_operator_is_available_without_speech_or_understanding_faces():
    """Each grammar section has its own allowed operators and context."""
    from Queries import GrammaticalThoughtRegistry

    grammar = Grammar()
    grammar.configure({
        "thought": {"rule": [
            "part_O1 = part.thought(part_I1, part_I2)",
        ]},
    })

    assert tuple(item.semantic_id for item in grammar.thought_operations) == (
        "part",)
    operation = grammar.thought_operations[0]
    assert operation.forward_rule_ids == operation.reverse_rule_ids == ()
    assert operation.forms[0].forward_rule_ids == operation.forms[0].reverse_rule_ids == ()
    registry = GrammaticalThoughtRegistry.install(_cs(), grammar)
    assert registry.executable_operation_ids == ("part",)


def test_ispart_can_have_structural_and_thought_faces_under_one_exact_name():
    """The model may use `isPart` in compose and thought without aliases."""
    from Queries import GrammaticalThoughtRegistry

    grammar = Grammar()
    grammar.configure({
        "compose": {"rule": [
            "isPart_O1 = isPart.forward(isPart_I1, isPart_I2)",
        ]},
        "thought": {"rule": [
            "isPart_O1 = isPart.thought(isPart_I1, isPart_I2)",
        ]},
    })

    assert tuple(item.semantic_id for item in grammar.thought_operations) == (
        "isPart",)
    registry = GrammaticalThoughtRegistry.install(_cs(), grammar)
    assert registry.executable_operation_ids == ("isPart",)


def test_thought_selection_requires_a_checked_executor_at_registry_install():
    """A model cannot turn a structural-only fold into a thought action."""
    from Queries import GrammaticalThoughtRegistry

    grammar = Grammar()
    grammar.configure({
        "compose": {"rule": [
            "union_O1 = union.forward(union_I1, union_I2)",
        ]},
        "thought": {"rule": [
            "union_O1 = union.thought(union_I1, union_I2)",
        ]},
    })

    with pytest.raises(ValueError, match="union|executor|thought"):
        GrammaticalThoughtRegistry.install(_cs(), grammar)


@pytest.mark.parametrize(
    "config",
    [
        {"Queries": {"query": ["isPart(X, Y)"]}},
        {"Symbolic": {"Queries": {"query": ["isPart(X, Y)"]}}},
    ],
)
def test_queries_block_is_not_a_second_thought_catalogue(config):
    """Retirement applies to top-level and sectioned grammar syntax."""
    grammar = Grammar()
    with pytest.raises(ValueError, match="Queries|thought|catalogue"):
        grammar.configure(config)


def test_canonical_thought_executor_joins_only_a_declared_thought_form():
    """Table entries still require an exact model thought declaration."""
    from Queries import GrammaticalThoughtRegistry, THOUGHT_EXECUTORS

    grammar = Grammar()
    grammar.configure(_family())
    registry = GrammaticalThoughtRegistry.install(_cs(), grammar)

    assert registry.executable_operation_ids == ("part",)
    assert registry.operation_spec("part").semantic_id == "part"
    assert "part" in THOUGHT_EXECUTORS
    assert THOUGHT_EXECUTORS["isPart"].semantic_id == "isPart"
    with pytest.raises(ValueError, match="not registered|structural"):
        registry.operation_spec("equal")


def test_tiny_concept_inventory_keeps_thought_families_structural_not_partial():
    """VP setup is all-or-nothing; a small model never lazily mints one."""
    from Queries import GrammaticalThoughtRegistry

    space = _cs()
    # IDs start at one, so two missing VPs do not fit a two-row inventory.
    # The underlying fixture basis stays intact; this owns only the allocator's
    # physical-capacity boundary used by setup preflight.
    space.nVectors = 2
    grammar = Grammar()
    grammar.configure({"compose": {"rule": [
        "part_O1 = part.forward(part_I1, part_I2)",
        "equal_O1 = equal.forward(equal_I1, equal_I2)",
    ]}, "thought": {"rule": [
        "part_O1 = part.thought(part_I1, part_I2)",
        "equal_O1 = equal.thought(equal_I1, equal_I2)",
    ]}})

    registry = GrammaticalThoughtRegistry.install(space, grammar)

    assert tuple(item.semantic_id for item in registry.operations) == (
        "part", "equal")
    assert registry.executable_operation_ids == ()
    assert registry.unavailable_operation_ids == ("part", "equal")
    assert not any(name.startswith("grammatical-vp:")
                   for name in getattr(space, "_frozen_named", {}))
    with pytest.raises(RuntimeError, match="concept inventory exhausted"):
        registry.form("part", ("sym", 1), ("sym", 1))


def test_canonical_form_derives_closed_and_open_roles_without_alias_methods():
    """`part` owns both closed and open-role forms through its grammar roles."""
    from Queries import GrammaticalThoughtRegistry

    space = _cs()
    grammar = Grammar()
    grammar.configure(_family())
    registry = GrammaticalThoughtRegistry.install(space, grammar)
    part = ("sym", space.synthesize_higher_order([("sym", space.new_concept())]))
    whole = ("sym", space.new_concept())
    for reference in (part, whole):
        space._csw_concept_row(0, reference[1])

    closed = registry.form("part", part, whole)
    open_part = registry.form("part", whole, open_roles=("I1",))
    assert closed.mode == "interrogative"
    assert closed.role_mask.tolist() == [True, True, True]
    assert closed.role_refs[0] == part and closed.role_refs[2] == whole
    assert open_part.role_mask.tolist() == [False, True, True]
    assert open_part.role_refs[0] is None and open_part.role_refs[2] == whole
    with pytest.raises(ValueError, match="not registered|canonical|alias"):
        registry.form("parts", whole)


def test_open_part_form_executes_a_taxonomy_set_without_an_alias():
    """A grammar-open role selects the old neighbor result through ``part``."""
    from Queries import (
        GrammaticalThoughtRegistry, ThoughtConceptualCapability,
        ThoughtGrammarContext, ThoughtTaxonomyCapability,
    )
    from QueryWork import QueryWorkBudget

    space = _cs()
    grammar = Grammar()
    grammar.configure(_family())
    registry = GrammaticalThoughtRegistry.install(space, grammar)
    part, whole = ("sym", space.new_concept()), ("sym", space.new_concept())
    for reference in (part, whole):
        space._csw_concept_row(0, reference[1])
    space.add_whole(part[1], whole)
    # I1 is the unknown part; I2 carries the known whole.
    request = registry.form("part", whole, open_roles=("I1",))
    context = ThoughtGrammarContext(
        word_stream=(), conceptual_space=ThoughtConceptualCapability(
            space, lambda left, right: float(torch.allclose(left, right))),
        primed_symbols=(), ltm=object(), taxonomy=ThoughtTaxonomyCapability(space),
        work=QueryWorkBudget(16), continuation=None, boundary=lambda _row: None)

    result = registry.execute(request, context)
    assert result.result_kind == "set"
    assert result.evidence_kind == "taxonomy"
    assert any(item["reference"] == part for item in result.value)


def test_catalog_forms_unparsed_executable_candidates_without_a_reader_call():
    """The boundary menu comes from grammar, not the last parse face."""
    from Queries import GrammaticalThoughtRegistry

    space = _cs()
    grammar = Grammar()
    grammar.configure({"compose": {"rule": [
        "part_O1 = part.forward(part_I1, part_I2)",
        "equal_O1 = equal.forward(equal_I1, equal_I2)",
    ]}, "thought": {"rule": [
        "part_O1 = part.thought(part_I1, part_I2)",
        "equal_O1 = equal.thought(equal_I1, equal_I2)",
    ]}})
    registry = GrammaticalThoughtRegistry.install(space, grammar)
    part, whole = ("sym", space.new_concept()), ("sym", space.new_concept())
    for reference in (part, whole):
        space._csw_concept_row(0, reference[1])
    parsed = registry.form("part", part, whole)

    candidates = registry.controller_candidates(parsed, parsed, parsed)
    assert tuple(candidate.semantic_id for candidate in candidates[:2]) == (
        "part", "equal")
    assert {candidate.open_roles for candidate in candidates
            if candidate.semantic_id == "part"} == {(), ("I1",), ("I2",)}
    equal = next(candidate for candidate in candidates
                 if candidate.semantic_id == "equal")
    assert equal.request.role_refs[1] == registry._reference(
        ("conceptual-identity", "equal"))
    assert equal.request.role_refs[0] == part
    assert equal.request.role_refs[2] == whole
    assert equal.request.mode == "interrogative"


def test_thought_execution_receives_the_common_context_and_only_capability_views():
    """A boundary executor has no generic model/reasoner escape hatch."""
    from Queries import (
        GrammaticalThoughtRegistry, ThoughtGrammarContext, ThoughtResult,
    )
    from QueryWork import QueryWorkBudget

    class TaxonomyRead:
        def __init__(self):
            self.calls = []

        def evidence(self, left, right, **limits):
            self.calls.append((left, right, limits))
            return {"support_true": 1.0, "support_false": 0.0,
                    "candidates": ()}

    space = _cs()
    grammar = Grammar()
    grammar.configure(_family())
    registry = GrammaticalThoughtRegistry.install(space, grammar)
    part = ("sym", space.synthesize_higher_order([("sym", space.new_concept())]))
    whole = ("sym", space.new_concept())
    for reference in (part, whole):
        space._csw_concept_row(0, reference[1])
    request = registry.form("part", part, whole)
    taxonomy = TaxonomyRead()
    boundary_rows = []
    from Queries import ThoughtConceptualCapability
    from reasoning import TruthGroundedReasoner
    context = ThoughtGrammarContext(
        word_stream=("the", "part"), conceptual_space=ThoughtConceptualCapability(space, TruthGroundedReasoner.equal),
        primed_symbols=("recent",), ltm=object(), taxonomy=taxonomy,
        work=QueryWorkBudget(8), continuation=None,
        boundary=lambda row: boundary_rows.append(row), row=0,
    )

    result = registry.execute(request, context)
    assert isinstance(result, ThoughtResult)
    assert result.semantic_id == "part" and result.support_true == 1.0
    assert taxonomy.calls[0][0:2] == (part, whole)
    assert boundary_rows == [0]
    assert context.word_stream == ("the", "part")
    assert context.conceptual_space.matches(space)
    assert context.primed_symbols == ("recent",)
    assert not hasattr(context, "reasoner")


def test_thought_boundary_detaches_executor_operands_and_recorded_request():
    """The common thought signature crosses a real hard data boundary."""
    from Queries import (
        GrammaticalThoughtRegistry, ThoughtConceptualCapability,
        ThoughtExecutorDescriptor,
        ThoughtGrammarContext, ThoughtSignature,
    )
    from QueryWork import QueryWorkBudget

    captured = []

    def execute(_context, arguments):
        captured.append(arguments["I1"])
        return {"support_true": 1.0, "support_false": 0.0}

    descriptor = ThoughtExecutorDescriptor(
        "probe", "probe-domain", ("concept",), Mind.SERIAL,
        (Mind.KNOWING,), (Mind.SERIAL,), "probe", execute)
    operation = SimpleNamespace(semantic_id="probe", operand_roles=("I1",))
    signature = ThoughtSignature(operation, descriptor, ("I1",))
    value = torch.ones(8, requires_grad=True)
    stream = torch.ones(1, 8, requires_grad=True)
    primed = torch.ones(1, requires_grad=True)
    context = ThoughtGrammarContext(
        word_stream=stream, conceptual_space=object(), primed_symbols=primed, ltm=object(),
        taxonomy=object(), work=QueryWorkBudget(8), continuation=None,
        boundary=lambda _row: None)

    assert not context.word_stream.requires_grad
    assert not context.primed_symbols.requires_grad
    assert context.word_stream.data_ptr() != stream.data_ptr()
    assert context.primed_symbols.data_ptr() != primed.data_ptr()
    signature.invoke(context, value)
    assert captured and not captured[0].requires_grad
    assert captured[0] is not value

    space = _cs()
    grammar = Grammar()
    grammar.configure({
        "compose": {"rule": [
            "equal_O1 = equal.forward(equal_I1, equal_I2)"]},
        "thought": {"rule": [
            "equal_O1 = equal.thought(equal_I1, equal_I2)"]},
    })
    registry = GrammaticalThoughtRegistry.install(space, grammar)
    request = registry.form("equal", value, value)
    result = registry.execute(request, ThoughtGrammarContext(
        word_stream=value, conceptual_space=ThoughtConceptualCapability(
            space, lambda left, right: float(torch.allclose(left, right))),
        primed_symbols=(), ltm=object(), taxonomy=object(),
        work=QueryWorkBudget(8), continuation=None, boundary=lambda _row: None))
    assert not result.request.roles.requires_grad
    assert result.request.roles.data_ptr() != request.roles.data_ptr()


@pytest.mark.parametrize(
    ("filename", "required"),
    [
        ("complete.grammar", {"part", "equal", "exist"}),
        ("default.grammar", {"part", "equal"}),
        ("ladder.grammar", {"part", "equal", "exist"}),
    ],
)
def test_production_grammars_load_from_structural_thought_faces(filename, required):
    grammar = Grammar()
    grammar.load_from_grammar_file(filename)
    assert required.issubset({item.semantic_id for item in grammar.thought_operations})
    if filename in {"complete.grammar", "ladder.grammar"}:
        part = next(item for item in grammar.thought_operations
                    if item.semantic_id == "part")
        whole = next(form for form in part.forms
                     if form.structural_id == "whole")
        assert whole.permutation == ("I2", "I1")


def test_thought_only_structural_faces_are_pure_contextual_noops():
    """Boundary capabilities are never performed by composition itself."""
    import torch
    from Language import GRAMMAR_LAYER_CLASSES

    value = torch.randn(2, 8)
    other = torch.randn(2, 8)
    for name in ("quantize", "arma", "what"):
        layer = GRAMMAR_LAYER_CLASSES[name]()
        torch.testing.assert_close(layer.compose(value), value)
        torch.testing.assert_close(layer.generate(value), value)
    lookup = GRAMMAR_LAYER_CLASSES["lookup"]()
    torch.testing.assert_close(lookup.compose(value, other), value)


def test_structural_dispatch_receives_only_the_common_owned_context():
    """Every structural face has the same capability-limited call shape."""
    import torch
    from Language import invoke_structural_face
    from Queries import StructuralGrammarContext

    class Capture:
        arity = 2

        def __init__(self):
            self.calls = []

        def compose_from_grammar_context(self, operands, *, context):
            self.calls.append(("compose", tuple(operands), context))
            return operands[0] + operands[1]

        def generate_from_grammar_context(self, result, *, context):
            self.calls.append(("generate", result, context))
            return result, result

    space = _cs()
    context = StructuralGrammarContext(
        word_stream=("older", "newer"), conceptual_space=ConceptualSpaceCapability(4),
        primed_symbols=("recent",), phase="compose")
    layer = Capture()
    left, right = torch.ones(4), torch.full((4,), 2.0)

    torch.testing.assert_close(
        invoke_structural_face(layer, (left, right), context=context),
        torch.full((4,), 3.0))
    assert len(layer.calls) == 1
    phase, operands, received = layer.calls[0]
    assert phase == "compose" and operands[0] is left and operands[1] is right
    assert received is context
    assert not hasattr(context, "ltm") and not hasattr(context, "taxonomy")
    with pytest.raises(ValueError, match="phase|generate"):
        invoke_structural_face(layer, (left,), context=context, phase="generate")


def test_structural_dispatch_adapts_legacy_forward_reverse_behind_the_contract():
    """A legacy unary fold cannot bypass the public context signature."""
    import torch
    from Language import invoke_structural_face
    from Queries import StructuralGrammarContext

    class LegacyFold:
        arity = 1

        def __init__(self):
            self.calls = []

        def forward(self, value):
            self.calls.append("forward")
            return value + 1.0

        def reverse(self, value):
            self.calls.append("reverse")
            return value - 1.0

    space = _cs()
    value = torch.ones(4)
    layer = LegacyFold()
    compose = StructuralGrammarContext(
        word_stream=("word",), conceptual_space=ConceptualSpaceCapability(4),
        primed_symbols=(), phase="compose")
    generate = StructuralGrammarContext(
        word_stream=("emitted",), conceptual_space=ConceptualSpaceCapability(4),
        primed_symbols=None, phase="generate")

    torch.testing.assert_close(
        invoke_structural_face(layer, (value,), context=compose), value + 1.0)
    generated = invoke_structural_face(layer, (value,), context=generate)
    assert len(generated) == 1
    torch.testing.assert_close(generated[0], value - 1.0)
    assert layer.calls == ["forward", "reverse"]


def test_complete_grammar_exposes_its_declared_canonical_thought_operations():
    from Queries import GrammaticalThoughtRegistry

    grammar = Grammar()
    grammar.load_from_grammar_file("complete.grammar")
    registry = GrammaticalThoughtRegistry.install(_cs(), grammar)
    assert {"exist", "part", "equal", "lookup", "quantize", "arma", "what"}.issubset(
        registry.executable_operation_ids)
    # `true` needs the deferred two-truths sealed-clause representation; it
    # must not become an accidentally exposed boundary operation beforehand.
    assert "true" not in registry.executable_operation_ids


def test_owner_built_structural_context_freezes_only_owned_stream_and_priming():
    """Compose snapshots its input; generate never inherits a target teacher."""
    from types import SimpleNamespace
    import torch
    from Language import SymbolSubSpace

    input_stream = torch.arange(8.0, requires_grad=True).reshape(1, 2, 4)
    output_prefix = torch.full((1, 1, 4), 7.0)
    priming = torch.tensor([[1.0, 1.5, 1.0]])
    owner = SimpleNamespace(
        conceptualSpace=SimpleNamespace(
            subspace=SimpleNamespace(muxedSize=4)),
        taxonomy=SimpleNamespace(priming_mask=lambda: priming),
        _generated_word_stream=output_prefix,
    )

    compose = SymbolSubSpace._structural_grammar_context(
        owner, phase="compose", input_stream=input_stream)
    assert compose.phase == "compose"
    assert compose.word_stream is not input_stream
    assert compose.word_stream.requires_grad
    torch.testing.assert_close(compose.word_stream, input_stream.detach())
    assert compose.primed_symbols is not priming
    torch.testing.assert_close(compose.primed_symbols, priming)
    assert compose.conceptual_space.width == 4
    assert not hasattr(compose.conceptual_space, "symbolSpace")
    assert not hasattr(compose, "taxonomy") and not hasattr(compose, "ltm")

    with torch.no_grad():
        input_stream.add_(100.0)
        priming.add_(100.0)
    torch.testing.assert_close(
        compose.word_stream, torch.arange(8.0).reshape(1, 2, 4))
    torch.testing.assert_close(compose.primed_symbols, torch.tensor([[1.0, 1.5, 1.0]]))

    # The passed target is deliberately ignored for the structural generate
    # context; only output-owner state may be exposed to a generate face.
    target_teacher = torch.full((1, 9, 4), -3.0)
    generate = SymbolSubSpace._structural_grammar_context(
        owner, phase="generate", input_stream=target_teacher)
    assert generate.phase == "generate"
    assert generate.word_stream is not output_prefix
    torch.testing.assert_close(generate.word_stream, output_prefix)
    assert generate.word_stream.shape[1] == 1


def test_live_router_adapters_receive_the_owner_built_structural_context():
    """The real unary/binary router invokes structural faces through one API."""
    import torch
    import torch.nn as nn
    from Language import (
        LanguageLayer, _BinaryGrammarOpAdapter, _UnaryGrammarOpAdapter,
    )
    from Queries import StructuralGrammarContext

    class Unary(nn.Module):
        arity = 1
        nInput = 0

        def __init__(self):
            super().__init__()
            self.contexts = []

        def compose(self, value):
            return value + 1.0

        def compose_from_grammar_context(self, operands, *, context):
            self.contexts.append(context)
            return self.compose(*operands)

    class Binary(nn.Module):
        arity = 2

        def __init__(self):
            super().__init__()
            self.contexts = []

        def compose(self, left, right):
            return left + right

        def compose_from_grammar_context(self, operands, *, context):
            self.contexts.append(context)
            return self.compose(*operands)

    space = _cs()
    context = StructuralGrammarContext(
        word_stream=("older", "newer"), conceptual_space=ConceptualSpaceCapability(4),
        primed_symbols=("recent",), phase="compose")
    unary, binary = Unary(), Binary()
    router = LanguageLayer(
        n_input=4, n_output=4, hidden_dim=8, feature_dim=4,
        max_depth=2, temperature=1.0)
    router.attach_unary_ops(
        ops=[_UnaryGrammarOpAdapter(unary)], rule_ids=[0], space_role="CS")
    router.attach_layer_ops(
        ops=[_BinaryGrammarOpAdapter(binary)], rule_ids=[1], space_role="CS")
    router.compose(torch.randn(1, 2, 4), word_space=None,
                   grammar_context=context)

    assert unary.contexts == [context]
    assert binary.contexts == [context]


def test_syntactic_executor_uses_the_same_structural_context_dispatcher():
    """The non-router executor cannot bypass the common face signature."""
    import torch
    import torch.nn as nn
    from Language import Grammar, SyntacticLayer, TheGrammar
    from Queries import StructuralGrammarContext

    class WordSpace:
        def register_host_layer(self, *_args):
            pass

    class Capture(nn.Module):
        arity = 1

        def __init__(self):
            super().__init__()
            self.contexts = []

        def compose(self, value):
            return value + 2.0

        def compose_from_grammar_context(self, operands, *, context):
            self.contexts.append(context)
            return self.compose(*operands)

    saved = (list(TheGrammar.rules), list(TheGrammar.rules_upward),
             list(TheGrammar.rules_downward), list(TheGrammar.reverse_rules),
             TheGrammar._configured)
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.configure({"compose": {"symbols": {"rule": ["S = not(S)"]}}})
        capture = Capture()
        layer = SyntacticLayer("SS", WordSpace(), {"not": capture})
        context = StructuralGrammarContext(
            word_stream=("word",), conceptual_space=ConceptualSpaceCapability(4),
            primed_symbols=(), phase="compose")
        value = torch.ones(1, 4)
        torch.testing.assert_close(
            layer.execute(0, value, context=context), torch.full((1, 4), 3.0))
        assert capture.contexts == [context]
    finally:
        (TheGrammar.rules, TheGrammar.rules_upward, TheGrammar.rules_downward,
         TheGrammar.reverse_rules, TheGrammar._configured) = saved


def test_recorded_compose_step_keeps_the_same_structural_context_contract():
    """A forced/replayed compose rule cannot bypass the router face API."""
    from types import SimpleNamespace
    import torch
    import torch.nn as nn
    from Language import LanguageSpace, _BinaryGrammarOpAdapter
    from Queries import StructuralGrammarContext

    class Capture(nn.Module):
        arity = 2

        def __init__(self):
            super().__init__()
            self.contexts = []

        def compose(self, left, right):
            return left + right

        def compose_from_grammar_context(self, operands, *, context):
            self.contexts.append(context)
            return self.compose(*operands)

    capture = Capture()
    owner = SimpleNamespace(
        _last_structural_compose_context=StructuralGrammarContext(
            word_stream=("older", "newer"), conceptual_space=ConceptualSpaceCapability(4),
            primed_symbols=(), phase="compose"))
    binary = SimpleNamespace(ops=[_BinaryGrammarOpAdapter(capture)])
    language = LanguageSpace.__new__(LanguageSpace)
    nn.Module.__init__(language)
    object.__setattr__(language, "_symbol_space", SimpleNamespace(subspace=owner))
    object.__setattr__(
        language, "_language_layer_ref", SimpleNamespace(_binary_layers={"CS": binary}))

    parent = language.forward_binary_step(
        torch.ones(1, 4), torch.full((1, 4), 2.0),
        torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool))
    torch.testing.assert_close(parent, torch.full((1, 4), 3.0))
    assert capture.contexts == [owner._last_structural_compose_context]


def test_live_tree_choice_keeps_the_same_structural_context_contract():
    """The normal in-STM chooser is a structural caller too."""
    from types import SimpleNamespace
    import torch
    import torch.nn as nn
    from Language import (
        BinaryStructuredReductionLayer, LanguageSpace, _BinaryGrammarOpAdapter,
    )
    from Queries import StructuralGrammarContext

    class Capture(nn.Module):
        arity = 2

        def __init__(self):
            super().__init__()
            self.contexts = []

        def compose(self, left, right):
            return left + right

        def compose_from_grammar_context(self, operands, *, context):
            self.contexts.append(context)
            return self.compose(*operands)

    capture = Capture()
    context = StructuralGrammarContext(
        word_stream=("older", "newer"), conceptual_space=ConceptualSpaceCapability(4),
        primed_symbols=(), phase="compose")
    owner = SimpleNamespace(_last_structural_compose_context=context)
    reducer = BinaryStructuredReductionLayer(
        d_model=4, ops=[_BinaryGrammarOpAdapter(capture)], r_copy=1)
    language = LanguageSpace.__new__(LanguageSpace)
    nn.Module.__init__(language)
    object.__setattr__(language, "_symbol_space", SimpleNamespace(subspace=owner))
    object.__setattr__(
        language, "_language_layer_ref", SimpleNamespace(_binary_layers={"CS": reducer}))
    buffer = torch.randn(1, 2, 4)
    state = (
        buffer, torch.tensor([2]), torch.zeros(1, 2, dtype=torch.long),
        torch.zeros(1, 2, dtype=torch.long), torch.full((1, 2), -1, dtype=torch.long),
        torch.ones(1, 2),
    )
    choice = language.choose_sentence_seal_binary(
        state, torch.ones(1, dtype=torch.bool), base_tau=0.5)
    assert choice.parent.shape == (1, 4)
    assert capture.contexts == [context]


def test_canonical_thought_executors_use_only_their_named_capability_views():
    """New thought faces cannot recover a legacy generic reasoner/model."""
    import torch
    from Meaning import ConceptualMeaning
    from Queries import THOUGHT_EXECUTORS, ThoughtGrammarContext
    from QueryWork import QueryWorkBudget

    class ConceptualRead:
        width = 4

        def equal(self, left, right):
            return 1.0 if torch.equal(left, right) else 0.0

        def quantize(self, value, **limits):
            return {"value": value + 1.0, "reference": ("sym", 7),
                    "nodes_scanned": 1, "incomplete": ()}

    class LTMRead:
        def retrieve(self, *_args, **_kwargs):
            return {'frames': (), 'value': (), 'records_scanned': 0, 'incomplete': ()}
        def existence_evidence(self, meaning, **limits):
            return {"support_true": 1.0, "support_false": 0.0,
                    "candidates": (), "incomplete": ()}

        def lookup(self, left, right, **limits):
            return {"value": ((left, right),), "records_scanned": 1,
                    "incomplete": ()}

        def expectation(self, row, **limits):
            return torch.full((3, 4), float(row + 1))

    class TaxonomyRead:
        def evidence(self, left, right, **limits):
            return {"support_true": 0.5, "support_false": 0.0,
                    "candidates": (), "incomplete": ()}

    meaning = ConceptualMeaning(
        torch.zeros(3, 4), torch.tensor([True, True, True]),
        mode="interrogative")
    context = ThoughtGrammarContext(
        word_stream=("owned",), conceptual_space=ConceptualRead(),
        primed_symbols=(), ltm=LTMRead(), taxonomy=TaxonomyRead(),
        work=QueryWorkBudget(32), continuation=lambda child: {"child": child},
        boundary=lambda _row: None, row=2)
    value = torch.ones(4)

    assert THOUGHT_EXECUTORS["exist"].executor(context, {"I1": meaning})[
        "support_true"] == 1.0
    assert THOUGHT_EXECUTORS["part"].executor(context, {"I1": ("sym", 1), "I2": ("sym", 2)})[
        "support_true"] == 0.5
    assert THOUGHT_EXECUTORS["equal"].executor(context, {"I1": value, "I2": value})[
        "support_true"] == 1.0
    assert THOUGHT_EXECUTORS["lookup"].executor(context, {"I1": value, "I2": value})[
        "records_scanned"] == 1
    assert THOUGHT_EXECUTORS["quantize"].executor(context, {"I1": value})[
        "reference"] == ("sym", 7)
    arma = THOUGHT_EXECUTORS["arma"].executor(context, {"I1": meaning})
    torch.testing.assert_close(arma["value"], torch.full((3, 4), 3.0))
    assert THOUGHT_EXECUTORS["what"].executor(context, {"I1": meaning})["value"] == {"child": meaning}
    assert not hasattr(context, "reasoner")


def test_descriptor_scopes_hide_undeclared_capability_methods_at_execution():
    """A descriptor's declared reads, not the raw owner, define its surface."""
    from types import SimpleNamespace
    import torch
    from Queries import (
        ThoughtExecutorDescriptor, ThoughtGrammarContext, ThoughtSignature,
    )
    from QueryWork import QueryWorkBudget

    seen = {}

    def execute(context, _arguments):
        seen["facts"] = hasattr(context.ltm, "existence_evidence")
        seen["lookup"] = hasattr(context.ltm, "lookup")
        seen["equal"] = hasattr(context.conceptual_space, "equal")
        return {"value": None, "incomplete": ()}

    descriptor = ThoughtExecutorDescriptor(
        "probe", "ltm-facts", ("concept",), Mind.SERIAL,
        (Mind.LTM,), (Mind.SERIAL,), "probe", execute)
    operation = SimpleNamespace(
        semantic_id="probe", operand_roles=("I1",), result_role="O1")
    signature = ThoughtSignature(operation, descriptor, ("I1",))

    class LTM:
        def existence_evidence(self, *_args, **_kwargs):
            return {}

        def lookup(self, *_args, **_kwargs):
            return {}

    class Conceptual:
        def equal(self, *_args):
            return 1.0

    context = ThoughtGrammarContext(
        word_stream=(), conceptual_space=Conceptual(), primed_symbols=(),
        ltm=LTM(), taxonomy=object(), work=QueryWorkBudget(2),
        continuation=None, boundary=lambda _row: None)
    signature.invoke(context, torch.ones(4))
    assert seen == {"facts": True, "lookup": False, "equal": False}


def test_description_thought_preparation_uses_only_its_declared_ltm_view():
    """A description operand has no reasoner fallback before an exist call."""
    import torch
    from Meaning import ConceptualMeaning
    from Queries import GrammaticalThoughtRegistry, ThoughtGrammarContext
    from QueryWork import QueryWorkBudget

    grammar = Grammar()
    grammar.configure({
        "compose": {"rule": ["exist_O1 = exist.forward(exist_I1)"]},
        "thought": {"rule": ["exist_O1 = exist.thought(exist_I1)"]},
    })
    space = _cs()
    registry = GrammaticalThoughtRegistry.install(space, grammar)
    description = ConceptualMeaning(
        torch.ones(3, 8), torch.tensor([True, True, True]), mode="assertive")
    occurrence = ("ltm", "fixture", 1)

    class LTMRead:
        def __init__(self):
            self.calls = []

        def resolve_description(self, reference, **limits):
            self.calls.append(("description", reference, limits))
            assert reference == occurrence
            return description, 1

        def existence_evidence(self, value, **limits):
            self.calls.append(("exist", value, limits))
            assert value is not description
            assert not value.roles.requires_grad
            torch.testing.assert_close(value.roles, description.roles)
            return {"support_true": 1.0, "support_false": 0.0,
                    "candidates": (), "incomplete": ()}

        def lookup(self, *_args, **_kwargs):
            raise AssertionError("undeclared lookup capability escaped")

    ltm = LTMRead()
    context = ThoughtGrammarContext(
        word_stream=("finished",), conceptual_space=space, primed_symbols=(),
        ltm=ltm, taxonomy=object(), work=QueryWorkBudget(32),
        continuation=None, boundary=lambda _row: None)
    request = registry.form("exist", occurrence, context=context)
    result = registry.execute(request, context)

    assert result.support_true == 1.0
    assert [call[0] for call in ltm.calls] == ["description", "description", "exist"]


def test_grammar_declares_whole_as_one_part_family_with_a_role_permutation():
    """The converse is grammar metadata, not an executor alias table entry."""
    import torch
    from Queries import GrammaticalThoughtRegistry

    grammar = Grammar()
    grammar.configure({
        "compose": {"rule": [
            "part_O1 = part.forward(part_I1, part_I2)",
            {"_": "whole_O1 = whole.forward(whole_I1, whole_I2)",
             "family": "part", "permutation": "I2,I1"},
        ]},
        "thought": {"rule": [
            "part_O1 = part.thought(part_I1, part_I2)",
            {"_": "whole_O1 = whole.thought(whole_I1, whole_I2)",
             "family": "part", "permutation": "I2,I1"},
        ]},
        "generate": {"rule": [
            "part_I1, part_I2 = part.reverse(part_O1)",
            {"_": "whole_I1, whole_I2 = whole.reverse(whole_O1)",
             "family": "part", "permutation": "I2,I1"},
        ]},
    })
    assert tuple(item.semantic_id for item in grammar.thought_operations) == ("part",)

    space = _cs()
    registry = GrammaticalThoughtRegistry.install(space, grammar)
    whole, part = (("sym", space.new_concept()), ("sym", space.new_concept()))
    for reference in (whole, part):
        space._csw_concept_row(0, reference[1])
    forward = registry.form("part", part, whole)
    converse = registry.form("whole", whole, part)
    assert registry.operation_spec("whole").semantic_id == "part"
    assert registry.executable_operation_ids == ("part",)
    assert registry.identities == (("conceptual-taxonomy", "part"),)
    assert converse.role_refs == forward.role_refs
    torch.testing.assert_close(converse.roles, forward.roles)


def test_model_boundary_builds_the_common_thought_context_without_a_reasoner():
    """The production caller supplies capabilities, never its model graph."""
    from types import SimpleNamespace
    import torch
    from Meaning import ConceptualMeaning
    from Models import BasicModel
    from Queries import ThoughtGrammarContext
    from QueryWork import QueryWorkBudget

    space = _cs()
    model = BasicModel()
    model.spaces = []
    object.__setattr__(model, "conceptualSpace", space)
    primed = torch.tensor([[1.0, 1.5]])
    object.__setattr__(model, "symbolSpace", SimpleNamespace(
        taxonomy=SimpleNamespace(priming_mask=lambda: primed),
        discourse=None, what_memory=None))
    meaning = ConceptualMeaning(
        torch.ones(3, 8, requires_grad=True), torch.tensor([True, True, True]),
        mode="interrogative")
    with model._query_boundary_scope((0,)):
        context = model._thought_grammar_context(
            meaning, row=0, work=QueryWorkBudget(8), continuation=None)

    assert isinstance(context, ThoughtGrammarContext)
    assert context.conceptual_space.matches(space)
    assert context.conceptual_space.width == 8
    assert not context.word_stream.requires_grad
    assert context.primed_symbols is not primed
    torch.testing.assert_close(context.primed_symbols, primed)
    assert not hasattr(context, "reasoner")
    assert not hasattr(context.ltm, "reasoner")
    assert not hasattr(context.taxonomy, "reasoner")


def test_normal_boundary_execution_uses_the_thought_registry_and_common_context():
    """The production controller cannot retain the legacy QueryContext bridge."""
    from types import SimpleNamespace
    from Layers import WhatInteractionMemory
    from Models import BasicModel
    from Queries import GrammaticalThoughtRegistry

    grammar = Grammar()
    grammar.configure({
        "compose": {"rule": ["part_O1 = part.forward(part_I1, part_I2)"]},
        "thought": {"rule": ["part_O1 = part.thought(part_I1, part_I2)"]},
    })
    space = _cs()
    registry = GrammaticalThoughtRegistry.install(space, grammar)
    part, whole = (("sym", space.new_concept()), ("sym", space.new_concept()))
    for reference in (part, whole):
        space._csw_concept_row(0, reference[1])
    space.add_whole(part[1], whole)
    model = BasicModel()
    model.spaces = []
    memory = WhatInteractionMemory(batch=1, capacity=32, detach_mode="episode")
    object.__setattr__(model, "conceptualSpace", space)
    object.__setattr__(model, "symbolSpace", SimpleNamespace(
        what_memory=memory, taxonomy=None, discourse=None,
        grammatical_thoughts=registry))
    object.__setattr__(model, "grammatical_thoughts", registry)
    model.what_thinking_detach = "episode"

    request = registry.form("part", part, whole)
    with model._query_boundary_scope((0,)):
        result = model.run_selected_thought(
            request, row=0, work_budget=8, registry=registry)

    assert result.evidence["support_true"] == 1.0
    assert [record.kind for record in memory.thought_history() if record.kind != "cutoff"] == [
        "begin", "thought", "thought", "finish"]


def test_completed_structural_whole_and_what_program_forms_one_canonical_request():
    """Program recovery uses grammar forms, not `query` flags or alias tables."""
    import torch
    from Language import LanguageSpace
    from Queries import GrammaticalThoughtRegistry
    from Understanding import AnswerProgram

    grammar = Grammar()
    grammar.configure({
        "compose": {"rule": [
            "part_O1 = part.forward(part_I1, part_I2)",
            {"_": "whole_O1 = whole.forward(whole_I1, whole_I2)",
             "family": "part", "permutation": "I2,I1"},
            "what_O1 = what.forward(what_I1)",
        ]},
        "thought": {"rule": [
            "part_O1 = part.thought(part_I1, part_I2)",
            {"_": "whole_O1 = whole.thought(whole_I1, whole_I2)",
             "family": "part", "permutation": "I2,I1"},
            "what_O1 = what.thought(what_I1)",
        ]},
    })
    space = _cs()
    registry = GrammaticalThoughtRegistry.install(space, grammar)
    whole, part = (("sym", space.new_concept()), ("sym", space.new_concept()))
    for reference in (whole, part):
        space._csw_concept_row(0, reference[1])
    whole_rule = next(rule for rule in grammar.rules_upward
                      if rule.method_name == "whole")
    what_rule = next(rule for rule in grammar.rules_upward
                     if rule.method_name == "what")
    owner = LanguageSpace.__new__(LanguageSpace)
    object.__setattr__(owner, "_compose_binary_rules", (whole_rule,))
    object.__setattr__(owner, "_compose_unary_rules", (what_rule,))
    leaves = torch.stack((torch.full((8,), -0.25), torch.full((8,), 0.75)))
    program = AnswerProgram(
        rows=torch.tensor([3, 5]), word_rows=torch.tensor([7, 9]),
        activations=torch.tensor([-0.25, 0.75]), leaves=leaves,
        actions=torch.tensor([[0, -1, 0], [0, -1, 1], [1, 0, -1], [2, 0, -1]]),
        targets=torch.tensor([0, -1]), end_state=torch.zeros(3, 8),
        concept_ids=torch.tensor([whole[1], part[1]]))

    meaning = owner.program_meaning(program, registry)
    expected = registry.form("part", part, whole, mode="interrogative")
    assert meaning is not None and meaning.mode == "interrogative"
    assert meaning.role_refs == expected.role_refs
    torch.testing.assert_close(meaning.roles[0], leaves[1])
    torch.testing.assert_close(meaning.roles[2], leaves[0])
