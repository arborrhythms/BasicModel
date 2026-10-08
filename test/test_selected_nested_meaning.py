"""Pure selected composition retains nested occurrences for the existing LTM."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Layers import TernaryTruthStore, WhatInteractionMemory
from Models import BasicModel, _append_observed_meaning
from test_selected_relation_meaning import _program_owner
from reading_fixtures import commit_reading, sentence_state


def _nested(monkeypatch, *, outer="whole", question=False):
    cs, grammar, registry, language, leaves, program, part, whole = _program_owner(monkeypatch)
    unary = {rule.method_name: i for i, rule in enumerate(language._compose_unary_rules)}
    binary = {rule.method_name: i for i, rule in enumerate(language._compose_binary_rules)}
    entry = program()
    actions = entry.actions.tolist()
    if outer == "ask":
        actions += [[2, unary["ask"], -1], [2, unary["ask"], -1]]
    else:
        actions += [[0, -1, 2], [1, binary[outer], -1]]
        if question:
            actions += [[2, unary["ask"], -1]]
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


@pytest.mark.parametrize("path", ["forward", "batch_eval", "packed"])
def test_nested_observation_writers_retain_children_without_certifying_them(monkeypatch, path):
    _cs, registry, language, entry, leaves = _nested(monkeypatch)
    store = _write_selected_observation(language, registry, entry, path)
    assert len(store) == 4
    assert all(store.row(i)['kind']=='question' for i in (1,2,3))
    child, parent = store.row(0), store.row(1)
    assert parent["meaning"].role_refs[2] == child["occurrence"]
    assert child["kind"] == "unverified" and child["trust"] == 0
    torch.testing.assert_close(child["meaning"].roles[0], leaves[0].detach())
    assert not child["meaning"].roles.requires_grad
    assert parent["kind"] == "question"


def test_10_packed_pending_and_eager_ends_write_identical_rows(monkeypatch):
    _cs, registry, language, entry, _ = _nested(monkeypatch)
    stores = [_write_selected_observation(language, registry, entry, path)
              for path in ('forward', 'batch_eval', 'packed')]
    def addresses(store):
        def address(cid):
            row = store.index_of_row(cid)
            return ('concept', cid) if row is None else ('clause', row)
        return [tuple(address(int(cid)) for cid in store.refs[row]) for row in range(len(store))]
    for store in stores[1:]:
        assert addresses(store) == addresses(stores[0])
        for name in ('slots', 'rel_type', 'c_plus', 'c_minus', 'role_mask', 'record_kind', 'where', 'when'):
            torch.testing.assert_close(getattr(store, name), getattr(stores[0], name), rtol=0, atol=0)
        for row in range(len(store)):
            torch.testing.assert_close(store.meaning_of(row).roles,
                                       stores[0].meaning_of(row).roles)


def test_nested_what_from_a_completed_program_enters_the_same_controller(monkeypatch):
    cs, registry, language, entry, _leaves = _nested(monkeypatch, outer="ask")
    model = BasicModel()
    model.spaces = []
    store = TernaryTruthStore(8, capacity=16)
    memory = WhatInteractionMemory(batch=1, capacity=64, detach_mode="episode")
    object.__setattr__(model, "conceptualSpace", cs)
    object.__setattr__(model, "languageSpace", language)
    object.__setattr__(model, "grammatical_thoughts", registry)
    object.__setattr__(model, "symbolSpace", SimpleNamespace(
        grammatical_thoughts=registry, what_memory=memory, ltm_store=store))
    model.eval()
    from test_thought_model_fixture import force_requested_thought
    force_requested_thought(model)
    with model._query_boundary_scope((0,)):
        results = model._run_selected_sentence_thoughts((sentence_state(language, entry, registry),), work_budget=128)
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
    with pytest.raises(OverflowError, match='forgetting'):
        store.append_meaning(meaning, kind='observation')
    assert len(store) == 0
    malformed = replace(meaning, role_refs=(None, meaning.role_refs[1], ('constituent', 7)))
    with pytest.raises(ValueError, match='constituent'):
        store.bind_constituents(malformed)
    assert len(store) == 0

def _write_selected_observation(language, registry, entry, path):
    store = TernaryTruthStore(8, capacity=16)
    from ClauseRow import attach_clause_index
    host = SimpleNamespace(languageSpace=language, grammatical_thoughts=registry)
    attach_clause_index(host, store, registry.space)
    # Every public entry reaches this common closing; real entry-point parity
    # is exercised by test_sentence_boundary.
    commit_reading(language, registry, entry, store, owner=host)
    return store
