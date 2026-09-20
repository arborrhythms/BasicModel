"""Pure selected composition retains nested occurrences for the existing LTM."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Layers import TernaryTruthStore, WhatInteractionMemory
from Models import BasicModel, _append_observed_meaning
from test_selected_relation_meaning import _program_owner


def _nested(monkeypatch, *, outer="whole", question=False):
    cs, grammar, registry, language, leaves, program, part, whole = _program_owner(monkeypatch)
    unary = {rule.method_name: i for i, rule in enumerate(language._compose_unary_rules)}
    binary = {rule.method_name: i for i, rule in enumerate(language._compose_binary_rules)}
    entry = program()
    actions = entry.actions.tolist()
    if outer == "what":
        actions += [[2, unary["what"], -1], [2, unary["what"], -1]]
    else:
        actions += [[0, -1, 2], [1, binary[outer], -1]]
        if question:
            actions += [[2, unary["what"], -1]]
        entry = replace(entry,
            rows=torch.tensor([3, 5, 3]), word_rows=torch.tensor([7, 9, 7]),
            activations=torch.tensor([-.25, .75, -.25]),
            leaves=torch.cat((leaves, leaves[:1])),
            concept_ids=torch.tensor([part[1], whole[1], part[1]]),
            lexical_forms=(None, None, None))
    return cs, registry, language, replace(entry, actions=torch.tensor(actions)), leaves


def test_nested_compose_preserves_complete_child_roles_without_a_write(monkeypatch):
    _cs, registry, language, entry, leaves = _nested(monkeypatch)
    monkeypatch.setattr(registry, "execute", lambda *_a: pytest.fail("execution during compose"))
    meaning = language.program_meaning(entry, registry)
    assert meaning is not None
    assert len(meaning.constituents) == 1
    child = meaning.constituents[0]
    assert meaning.role_refs[2] == ("constituent", 0)  # whole reverses its roles
    torch.testing.assert_close(child.roles[0], leaves[0])
    torch.testing.assert_close(child.roles[2], leaves[1])
    child.roles.sum().backward()
    assert leaves.grad is not None and bool(leaves.grad.any())


def test_learned_lexical_policy_cannot_replace_an_explicit_nested_fold(monkeypatch):
    _cs, registry, language, entry, _leaves = _nested(monkeypatch)
    expected = language.program_meaning(entry, registry)
    language.configure_meaning_learning(registry, (int(entry.word_rows[0]),),
                                        entry.leaves[:1], hidden=8)
    monkeypatch.setattr(language.meaning_codec, 'compose',
                        lambda *_a: pytest.fail('explicit nested meaning was reinterpreted'))
    actual = language.program_meaning(entry, registry)
    assert actual.role_refs == expected.role_refs
    torch.testing.assert_close(actual.constituents[0].roles, expected.constituents[0].roles)


@pytest.mark.parametrize("path", ["eager", "pending", "packed"])
def test_nested_observation_writers_retain_children_without_certifying_them(monkeypatch, path):
    _cs, registry, language, entry, leaves = _nested(monkeypatch)
    store = _write_selected_observation(language, registry, entry, path)
    assert len(store) == 2
    child, parent = store.row(0), store.row(1)
    assert parent["meaning"].role_refs[2] == child["occurrence"]
    assert child["kind"] == "unverified" and child["trust"] == 0
    torch.testing.assert_close(child["meaning"].roles[0], leaves[0].detach())
    assert not child["meaning"].roles.requires_grad
    assert parent["kind"] == "observation"


def test_nested_what_from_a_completed_program_enters_the_same_controller(monkeypatch):
    cs, registry, language, entry, _leaves = _nested(monkeypatch, outer="what")
    model = BasicModel()
    model.spaces = []
    store = TernaryTruthStore(8, capacity=16)
    memory = WhatInteractionMemory(batch=1, capacity=64, detach_mode="episode")
    object.__setattr__(model, "conceptualSpace", cs)
    object.__setattr__(model, "languageSpace", language)
    object.__setattr__(model, "grammatical_thoughts", registry)
    object.__setattr__(model, "symbolSpace", SimpleNamespace(
        grammatical_thoughts=registry, what_memory=memory, ltm_store=store))
    with model._query_boundary_scope((0,)):
        results = model._run_selected_program_thoughts((entry,), work_budget=128)
    assert len(results) == 1
    result = results[0][1]
    assert result.result.result_kind == "subgoal"
    assert result.result.value.semantic_id == "part"
    assert any(record.kind == "descend" for record in result.records)
    assert result.work.spent == memory.thought_state().work_spent
    assert all(store.row(index)["kind"] != "fact" for index in range(len(store)))
    memory.end_what_episode()


def test_constituent_capacity_failure_and_invalid_local_ref_do_not_partially_write(monkeypatch):
    _cs, registry, language, entry, _ = _nested(monkeypatch)
    meaning = language.program_meaning(entry, registry)
    store = TernaryTruthStore(8, capacity=1)
    assert store.append_meaning(meaning, kind='observation') == -1
    assert len(store) == 0
    malformed = replace(meaning, role_refs=(None, meaning.role_refs[1], ('constituent', 7)))
    with pytest.raises(ValueError, match='constituent'):
        store.bind_constituents(malformed)
    assert len(store) == 0

def _write_selected_observation(language, registry, entry, path):
    store = TernaryTruthStore(8, capacity=16)
    slots = entry.end_state.unsqueeze(0)
    if path == "eager":
        meaning = language.program_meaning(entry, registry)
        assert meaning is not None
        _append_observed_meaning(store, entry.end_state, 1, meaning=meaning, trust=.9)
    else:
        discourse = SimpleNamespace(
            observe_stm_end_state=lambda *_a: None,
            predict_and_observe_stm_end_state=lambda *_a, **_k: None,
            expectation_scope="structured")
        host = SimpleNamespace(
            conceptualSpace=SimpleNamespace(_ltm_consolidation=True,
                stm_end_state_trust=lambda *_a: torch.tensor([.9])),
            symbolSpace=SimpleNamespace(ltm_store=store, discourse=discourse),
            languageSpace=language, grammatical_thoughts=registry,
            _capture_answer_programs=lambda: ((entry,), {0: (entry,)}),
            _expectation_documents_for_slot=lambda *_a: ["doc"])
        if path == "pending":
            host._pending_stm_end_state = (slots, torch.tensor([1]), torch.tensor([True]))
            BasicModel._drain_pending_stm_end_state(host)
        else:
            host._packed_sentence_roots = slots[:, :1]
            host._tensor_sentence_roots_live = slots.reshape(1, 1, -1)
            host._tensor_sentence_roots_depth = torch.tensor([[1]])
            host._tensor_final_end_slots = slots
            host._tensor_final_end_depth = torch.tensor([1])
            host.inputSpace = SimpleNamespace(
                _packed_sentence_slot_end_positions=torch.tensor([[2]]),
                _packed_sentence_slot_mask=torch.tensor([[True]]),
                _packed_sentence_counts_host=(1,))
            BasicModel._drain_packed_stm_end_states(host)
    return store
