"""Owned compose actions preserve semantic operands lost by their folded root."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import Language
from Models import BasicModel
from Queries import GrammaticalThoughtRegistry
from Understanding import AnswerProgram
from test_query_vp_boundaries import _world


def _program_owner(monkeypatch, *, face="part", interrogative=False):
    cs, grammar, _legacy_registry, a, b, _context = _world()
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    selected = next(
        index for index, rule in enumerate(grammar.rules_upward)
        if rule.space_role == "CS" and rule.method_name == face
        and rule.arity == 2)
    binary = tuple(
        index for index, rule in enumerate(grammar.rules_upward)
        if rule.space_role == "CS" and rule.arity == 2)
    unary = tuple(
        index for index, rule in enumerate(grammar.rules_upward)
        if rule.space_role == "CS" and rule.arity == 1)

    def bank(ids):
        return SimpleNamespace(ops=tuple(
            Language.GRAMMAR_LAYER_CLASSES[face]() if index == selected else None
            for index in ids))

    layer = SimpleNamespace(
        _binary_rule_ids={"CS": binary}, _unary_rule_ids={"CS": unary},
        _binary_layers={"CS": bank(binary)}, _unary_layers={"CS": bank(unary)})
    monkeypatch.setattr(Language, "TheGrammar", grammar)
    owner = Language.LanguageSpace(SimpleNamespace(
        subspace=SimpleNamespace(languageLayer=layer, muxedSize=0)))
    local = binary.index(selected)
    what_local = (next(
        index for index, rule in enumerate(
            tuple(grammar.rules_upward[rule_id] for rule_id in unary))
        if rule.method_name == "what") if interrogative else None)
    leaves = torch.stack((
        -.25 * registry._payload(a), .75 * registry._payload(b)
    )).detach().requires_grad_()
    folded = layer._binary_layers["CS"].ops[local].compose(
        leaves[0].reshape(1, 1, -1),
        leaves[1].reshape(1, 1, -1)).reshape(-1)
    end = torch.cat((folded.unsqueeze(0), leaves.new_zeros(2, leaves.shape[-1])))

    def program(values=leaves, refs=(a, b)):
        actions = torch.tensor(
            [[0, -1, 0], [0, -1, 1], [1, local, -1]], dtype=torch.long)
        if what_local is not None:
            actions = torch.cat((actions, torch.tensor(
                [[2, what_local, -1]], dtype=actions.dtype)))
        return AnswerProgram(
            rows=torch.tensor([3, 5]), word_rows=torch.tensor([7, 9]),
            activations=torch.tensor([-.25, .75]), leaves=values,
            actions=actions,
            targets=torch.tensor([local, -1]), end_state=end,
            concept_ids=torch.tensor([reference[1] for reference in refs]))

    return cs, grammar, registry, owner, leaves, program, a, b


@pytest.mark.parametrize(
    "face,order", [("part", (0, 1)), ("whole", (1, 0)), ("equal", (0, 1))])
@pytest.mark.parametrize("interrogative", [False, True])
def test_selected_binary_relation_preserves_signed_operands_middle_vp_and_mode(
        monkeypatch, face, order, interrogative):
    _cs, _grammar, registry, owner, leaves, program, a, b = _program_owner(
        monkeypatch, face=face, interrogative=interrogative)
    entry = program()
    before = entry.actions.clone()
    meaning = owner.program_meaning(entry, registry)
    canonical = registry.form(
        face, a, b, mode="interrogative" if interrogative else "assertive")
    assert meaning.role_refs == canonical.role_refs
    assert meaning.mode == canonical.mode and meaning.polarity
    assert meaning.role_mask.tolist() == [True, True, True]
    torch.testing.assert_close(meaning.roles[1], canonical.roles[1])
    torch.testing.assert_close(meaning.roles[0], leaves[order[0]])
    torch.testing.assert_close(meaning.roles[2], leaves[order[1]])
    torch.testing.assert_close(entry.actions, before)
    (meaning.roles[0].sum() + 2 * meaning.roles[2].sum()).backward()
    expected = torch.empty_like(leaves)
    expected[order[0]] = 1
    expected[order[1]] = 2
    torch.testing.assert_close(leaves.grad, expected)


def test_changed_part_operand_remains_distinguishable_when_folded_root_is_identical(
        monkeypatch):
    _cs, _grammar, _registry, owner, leaves, program, _a, _b = _program_owner(
        monkeypatch)
    other = leaves.detach().clone()
    other[0] *= 3
    first, second = program(), program(other)
    torch.testing.assert_close(first.end_state, second.end_state)
    left, right = (owner.program_meaning(item, _registry) for item in (first, second))
    assert not torch.equal(left.roles[0], right.roles[0])
    torch.testing.assert_close(left.roles[1:], right.roles[1:])


@pytest.mark.parametrize("face", ["part", "equal"])
def test_unreduced_lexical_relation_keeps_signed_operands_and_native_vp(
        monkeypatch, face):
    """A three-slot lexical relation is still one selected meaning.

    The parser may retain a complete relative sentence in STM rather than
    folding it through a binary structural face.  Its middle leaf is already
    the registry-owned native VP, while the two noun leaves remain live,
    signed semantic operands.  Recovering it must not fall back to its
    physical end-state or reinterpret a native address as a value.
    """
    _cs, _grammar, registry, owner, leaves, _program, a, b = _program_owner(
        monkeypatch, face=face)
    canonical = registry.form(face, a, b, mode="assertive")
    vp = canonical.role_refs[1]
    lexical = torch.stack((
        leaves[0], torch.randn_like(leaves[0]), leaves[1],
    )).detach().requires_grad_()
    entry = AnswerProgram(
        rows=torch.tensor([3, 4, 5]),
        word_rows=torch.tensor([7, 8, 9]),
        activations=torch.tensor([-.25, 1.0, .75]),
        leaves=lexical,
        actions=torch.tensor(
            [[0, -1, 0], [0, -1, 1], [0, -1, 2]], dtype=torch.long),
        targets=torch.tensor([-1]),
        end_state=torch.zeros(3, lexical.shape[-1]),
        concept_ids=torch.tensor([a[1], vp[1], b[1]]),
        lexical_forms=(None, face, None),
    )

    meaning = owner.program_meaning(entry, registry)

    assert meaning is not None
    assert meaning.mode == "assertive" and meaning.polarity
    assert meaning.role_refs == canonical.role_refs
    torch.testing.assert_close(meaning.roles[0], lexical[0])
    torch.testing.assert_close(meaning.roles[1], canonical.roles[1])
    torch.testing.assert_close(meaning.roles[2], lexical[2])
    (meaning.roles[0].sum() + 2 * meaning.roles[2].sum()).backward()
    expected = torch.zeros_like(lexical)
    expected[0] = 1
    expected[2] = 2
    torch.testing.assert_close(lexical.grad, expected)


def test_unreduced_lexical_converse_uses_its_anchored_grammar_form(
        monkeypatch):
    """A shared canonical VP must not erase a lexical converse's role order.

    ``part`` and ``whole`` deliberately share the canonical ``part`` VP and
    checked executor.  The retained middle anchor is grammar provenance, not
    a learned surface/row/ID feature: it tells recovery which declared form
    supplied the lexical infix so the form's canonical permutation can place
    the two signed leaves correctly.
    """
    _cs, _grammar, registry, owner, leaves, _program, whole, part = _program_owner(
        monkeypatch, face="whole")
    canonical = registry.form("whole", whole, part, mode="assertive")
    vp = canonical.role_refs[1]
    lexical = torch.stack((
        leaves[0], torch.randn_like(leaves[0]), leaves[1],
    )).detach().requires_grad_()
    entry = AnswerProgram(
        rows=torch.tensor([3, 4, 5]),
        word_rows=torch.tensor([7, 8, 9]),
        activations=torch.tensor([-.25, 1.0, .75]),
        leaves=lexical,
        actions=torch.tensor(
            [[0, -1, 0], [0, -1, 1], [0, -1, 2]], dtype=torch.long),
        targets=torch.tensor([-1]),
        end_state=torch.zeros(3, lexical.shape[-1]),
        concept_ids=torch.tensor([whole[1], vp[1], part[1]]),
        lexical_forms=(None, "whole", None),
    )

    meaning = owner.program_meaning(entry, registry)

    assert meaning is not None
    assert meaning.role_refs == canonical.role_refs
    torch.testing.assert_close(meaning.roles[0], lexical[2])
    torch.testing.assert_close(meaning.roles[2], lexical[0])
    (meaning.roles[0].sum() + 2 * meaning.roles[2].sum()).backward()
    expected = torch.zeros_like(lexical)
    expected[2] = 1
    expected[0] = 2
    torch.testing.assert_close(lexical.grad, expected)

    # A checkpoint made before form provenance existed must decline this
    # ambiguous shared-VP lexical relation rather than inventing the first
    # declaration's orientation.
    legacy = replace(entry, lexical_forms=None)
    assert owner.program_meaning(legacy, registry) is None


def test_selected_unary_concept_operation_keeps_its_live_leaf_and_question_mode(
        monkeypatch):
    """A selected unary thought form is not limited to binary relations.

    ``quantize`` is explicitly declared in ``<thought>`` and accepts one
    conceptual operand.  Its completed program has no physical middle VP
    leaf: recovery must obtain the grammar-owned native VP while retaining
    the actual signed operand leaf, then let the structural ``what`` wrapper
    supply interrogative mode.  It must not decline the program merely because
    its selected structural root is unary.
    """
    _cs, _grammar, registry, owner, leaves, _program, _a, _b = _program_owner(
        monkeypatch)
    quantize = next(
        index for index, rule in enumerate(owner._compose_unary_rules)
        if rule.method_name == "quantize")
    what = next(
        index for index, rule in enumerate(owner._compose_unary_rules)
        if rule.method_name == "what")
    leaf = leaves[:1].detach().clone().requires_grad_()
    entry = AnswerProgram(
        rows=torch.tensor([3]), word_rows=torch.tensor([7]),
        activations=torch.tensor([-.25]), leaves=leaf,
        actions=torch.tensor(
            [[0, -1, 0], [2, quantize, -1], [2, what, -1]],
            dtype=torch.long),
        targets=torch.tensor([quantize, what, -1]),
        end_state=torch.zeros(3, leaf.shape[-1]),
        concept_ids=torch.tensor([-1]), lexical_forms=(None,))

    meaning = owner.program_meaning(entry, registry)

    canonical = registry.form("quantize", leaf[0], mode="interrogative")
    assert meaning is not None
    assert meaning.mode == "interrogative" and meaning.polarity
    assert meaning.role_refs == canonical.role_refs
    assert meaning.role_mask.tolist() == [True, True, False]
    torch.testing.assert_close(meaning.roles[0], leaf[0])
    torch.testing.assert_close(meaning.roles[1], canonical.roles[1])
    (meaning.roles[0].sum()).backward()
    torch.testing.assert_close(leaf.grad, torch.ones_like(leaf))


def test_selected_unary_description_operation_does_not_invent_an_occurrence(
        monkeypatch):
    """A direct word leaf is never reinterpreted as an LTM/thought reference."""
    _cs, _grammar, registry, owner, leaves, _program, _a, _b = _program_owner(
        monkeypatch)
    arma = next(
        index for index, rule in enumerate(owner._compose_unary_rules)
        if rule.method_name == "arma")
    what = next(
        index for index, rule in enumerate(owner._compose_unary_rules)
        if rule.method_name == "what")
    entry = AnswerProgram(
        rows=torch.tensor([3]), word_rows=torch.tensor([7]),
        activations=torch.tensor([-.25]), leaves=leaves[:1],
        actions=torch.tensor(
            [[0, -1, 0], [2, arma, -1], [2, what, -1]], dtype=torch.long),
        targets=torch.tensor([arma, what, -1]),
        end_state=torch.zeros(3, leaves.shape[-1]),
        # This deliberately arbitrary ID must not become a durable occurrence.
        concept_ids=torch.tensor([917]), lexical_forms=(None,))

    assert owner.program_meaning(entry, registry) is None


def test_capture_reads_anchored_form_from_retained_word_rows():
    """WORD rows dispatch the forward's decision, even for a shared OBJECT."""
    owner = SimpleNamespace(
        inputSpace=SimpleNamespace(
            _ar_word_concept_rows=torch.tensor([[7, 11, -1], [11, 7, -1]]),
            _ar_word_object_rows=torch.tensor([[19, 19, -1], [19, 19, -1]]),
            _ar_word_lexical_forms={7: "whole", 11: "part"}),
    )

    forms = BasicModel._word_lexical_forms(owner, 2, 3)

    assert forms == (("whole", "part", None), ("part", "whole", None))
    owner.inputSpace._ar_word_lexical_forms[7] = "equal"
    assert forms[0][0] == "whole"


def test_capture_does_not_invent_anchoring_for_an_unresolved_word():
    owner = SimpleNamespace(
        inputSpace=SimpleNamespace(
            _ar_word_concept_rows=torch.tensor([[-1, 7]])),
        perceptualSpace=SimpleNamespace(
            _forward_input={"word_texts": [["wholeOf", "wholeOf"]]}),
        languageSpace=SimpleNamespace(_surface_anchors={"wholeof": "whole"}),
    )

    assert BasicModel._word_lexical_forms(owner, 1, 2) == ((None, None),)


@pytest.mark.parametrize("packed", [False, True])
def test_forward_anchor_capture_survives_text_and_grammar_changes(
        tmp_path, monkeypatch, packed):
    """Real staging freezes the form before any observation writer captures it."""
    from test_compiled_word_chunk import (
        _stage_fullgraph_tensor_peer, _tiny_canonical_model,
    )
    from test_reverse_traversal import _stage_packed

    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets="16")
    model.eval()
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    try:
        with torch.no_grad():
            if packed:
                _stage_packed(model, [["wholeOf", "partOf"], ["partOf"]])
            else:
                _stage_fullgraph_tensor_peer(model, ["wholeOf", "partOf"])
            output = model._forward_with_compiled_sentence_state(None)
            model._publish_compiled_sentence_state(output)
            # Neither the transient spelling nor a changed anchor table may
            # reinterpret the leaves the completed forward actually retained.
            model.perceptualSpace._forward_input.pop("word_texts", None)
            monkeypatch.setattr(model.languageSpace, "_surface_anchors", {})
            monkeypatch.setattr(Language.TheGrammar, "surface_anchors", {})
            current, sentences = model._capture_answer_programs()

        assert sentences[0][0].lexical_forms == ("whole",)
        assert sentences[0][1].lexical_forms == ("part",)
        if packed:
            assert sentences[1][0].lexical_forms == ("part",)
            assert sentences[1][1] is None
            assert current[0] is sentences[1][0]
        else:
            assert current[0] is sentences[0][0]
        assert current[1] is sentences[0][1]
        held = sentences[0][0].detached()
        with torch.no_grad():
            _stage_fullgraph_tensor_peer(model, ["other", "words"])
        assert all(form is None for row in model._word_lexical_forms(
            2, model.inputSpace._word_active_mask.shape[1]) for form in row)
        assert not set(held.word_rows.tolist()).intersection(
            model.inputSpace._ar_word_lexical_forms)
        staged = model.inputSpace._ar_word_lexical_forms
        assert staged
        model.inputSpace.Reset(hard=False)
        assert model.inputSpace._ar_word_lexical_forms is staged
        model.inputSpace.Reset(hard=True)
        assert model.inputSpace._ar_word_lexical_forms == {}
        assert held.lexical_forms == ("whole",)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_native_addresses_do_not_become_semantic_operand_values(monkeypatch):
    cs, _grammar, _registry, owner, _leaves, program, a, b = _program_owner(
        monkeypatch)
    aliases = []
    for original in (a, b):
        reference = ("sym", cs.new_concept())
        row = cs._csw_concept_row(0, reference[1])
        with torch.no_grad():
            cs.similarity_codebook.getW()[row].copy_(_registry._payload(original))
        aliases.append(reference)
    first = owner.program_meaning(program(), _registry)
    renamed = owner.program_meaning(program(refs=tuple(aliases)), _registry)
    torch.testing.assert_close(first.roles, renamed.roles)
    assert renamed.role_refs[0] == aliases[0]
    assert renamed.role_refs[2] == aliases[1]


def test_owned_local_rule_meaning_survives_global_grammar_reconfiguration(
        monkeypatch):
    _cs, grammar, registry, owner, leaves, program, a, b = _program_owner(
        monkeypatch, interrogative=True)
    entry = program()
    grammar.configure({"compose": {"S": ["sum(S,S)"]}})
    meaning = owner.program_meaning(entry, registry)
    assert meaning.mode == "interrogative"
    assert meaning.role_refs == registry.form("part", a, b).role_refs
    torch.testing.assert_close(meaning.roles[0], leaves[0])


@pytest.mark.parametrize("wrapper, expected", [("not", False), ("non", False)])
def test_selected_relation_preserves_declared_outer_polarity(
        monkeypatch, wrapper, expected):
    """A negating compose wrapper changes meaning metadata, not its operands."""
    _cs, _grammar, registry, owner, leaves, program, a, b = _program_owner(
        monkeypatch, interrogative=True)
    unary = next(
        index for index, rule in enumerate(owner._compose_unary_rules)
        if rule.method_name == wrapper)
    entry = program()
    actions = torch.cat((entry.actions, torch.tensor(
        [[2, unary, -1]], dtype=entry.actions.dtype)), dim=0)
    meaning = owner.program_meaning(replace(entry, actions=actions), registry)
    assert meaning.mode == "interrogative"
    assert meaning.polarity is expected
    assert meaning.role_refs == registry.form("part", a, b).role_refs
    torch.testing.assert_close(meaning.roles[0], leaves[0])
    torch.testing.assert_close(meaning.roles[2], leaves[1])


@pytest.mark.parametrize("path", ["pending", "packed"])
def test_observation_boundary_uses_selected_relation_before_prediction_and_ltm(
        monkeypatch, path):
    from Layers import TernaryTruthStore
    from Models import BasicModel

    cs, _grammar, registry, owner, leaves, program, _a, _b = _program_owner(
        monkeypatch, interrogative=True)
    entry = program()
    slots = entry.end_state.unsqueeze(0)
    calls = []

    def predict(depths, payloads, **kwargs):
        calls.append((depths, payloads, kwargs))

    discourse = SimpleNamespace(
        observe_stm_end_state=lambda *args: None,
        predict_and_observe_stm_end_state=predict,
        expectation_scope="structured")
    store = TernaryTruthStore(8, capacity=16)
    host = SimpleNamespace(
        conceptualSpace=SimpleNamespace(
            _ltm_consolidation=True,
            stm_end_state_trust=lambda *args: torch.tensor([.9])),
        symbolSpace=SimpleNamespace(ltm_store=store, discourse=discourse),
        languageSpace=owner, grammatical_thoughts=registry,
        _capture_answer_programs=lambda: ((entry,), {0: (entry,)}),
        _expectation_documents_for_slot=lambda *args: ["doc"])
    monkeypatch.setattr(
        registry, "execute",
        lambda *args, **kwargs: pytest.fail("query executed during observation"))
    if path == "pending":
        host._pending_stm_end_state = (
            slots, torch.tensor([1]), torch.tensor([True]))
        BasicModel._drain_pending_stm_end_state(host)
        BasicModel._drain_pending_stm_end_state(host)
    else:
        host._packed_sentence_roots = slots[:, :1]
        host._tensor_sentence_roots_live = slots.reshape(1, 1, -1)
        host._tensor_sentence_roots_depth = torch.tensor([[1]])
        host._tensor_final_end_slots = slots
        host._tensor_final_end_depth = torch.tensor([1])
        host.inputSpace = SimpleNamespace(
            _packed_sentence_slot_end_positions=torch.tensor([[1]]),
            _packed_sentence_slot_mask=torch.tensor([[True]]),
            _packed_sentence_counts_host=(1,))
        BasicModel._drain_packed_stm_end_states(host)
    assert len(store) == 1 and len(calls) == 1
    stored = store.meaning_of(0)
    assert stored.mode == "interrogative"
    assert stored.role_refs == registry.form("part", _a, _b).role_refs
    assert stored.role_mask.tolist() == [True, True, True]
    assert store.row(0)["kind"] in ("question", "observation")
    assert calls[0][0] == [3]
    assert calls[0][2]["layout"] == "infix"
    torch.testing.assert_close(calls[0][1][0][0], leaves[0])
    torch.testing.assert_close(calls[0][1][0][2], leaves[1])
    assert not stored.roles.requires_grad
    assert calls[0][1][0].requires_grad
