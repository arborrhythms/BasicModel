from dataclasses import replace
"""Normal TruthSet replacement and restore retain thought-owned roots."""

import torch

from Memory import LTM0
from Meaning import ConceptualMeaning
from test_ltm_consolidation import _ON_CONFIG, _make_model


def _seed(model):
    store = model.symbolSpace.ltm_store
    store.reset()
    meaning = replace(ConceptualMeaning.from_description(torch.ones(store.nDim)), scope={"source": "reported"})
    row = store.append_meaning(meaning, trust=.75)
    return store, store.occurrence_of(row)


def _pin(model, reference):
    memory = model.symbolSpace.what_memory
    memory.detach_mode = "episode"
    roles = torch.ones(3, model.symbolSpace.ltm_store.nDim, requires_grad=True)
    question = ConceptualMeaning(
        roles, torch.ones(3, dtype=torch.bool),
        role_refs=(reference, None, None), bindings={"claim": reference},
    )
    memory.begin_thought_episode(question, work_budget=8)
    return memory, roles


def test_normal_truthset_replacement_keeps_live_thought_constituents_and_credit(monkeypatch):
    model = _make_model(_ON_CONFIG)
    store, reference = _seed(model)
    memory, roles = _pin(model, reference)
    model._ltm_provisioned = True
    monkeypatch.setattr(model, "_ltm_ingest_truth_texts", lambda *args, **kwargs: [])
    model._store_truths_into_ltm(store, model.symbolSpace.truth_layer, [], [])
    assert len(store) == 1
    assert store.occurrence_of(0) == reference and store.row(0)["trust"] == .75
    assert int(model.symbolSpace.truth_layer.count) == 1
    memory.thought_state().contexts[0].meaning.roles.square().sum().backward()
    assert roles.grad is not None and roles.grad.abs().sum() > 0


def test_full_checkpoint_restores_occurrence_owners_and_trust(tmp_path):
    model = _make_model(_ON_CONFIG)
    store, reference = _seed(model)
    _pin(model, reference)
    path = tmp_path / "nested.ckpt"
    model.save_weights(path)
    restored = _make_model(_ON_CONFIG)
    assert restored.load_weights(path)
    restored_store = restored.symbolSpace.ltm_store
    assert len(restored_store) == 1 and restored_store.occurrence_of(0) == reference
    assert restored_store.row(0)["trust"] == .75
    assert restored_store.read_structure(reference)["incomplete"] == ()
    current = restored.symbolSpace.what_memory.thought_state().contexts[0].meaning
    assert current.role_refs[0] == reference and not current.roles.requires_grad


def test_full_checkpoint_keeps_unreferenced_trusted_content(tmp_path):
    model = _make_model(_ON_CONFIG)
    _seed(model)
    path = tmp_path / "unreferenced.ckpt"
    model.save_weights(path)
    restored = _make_model(_ON_CONFIG)
    assert restored.load_weights(path)
    assert len(restored.symbolSpace.ltm_store) == 1
    assert int(restored.symbolSpace.truth_layer.count) == 1


def test_tensor_only_restore_keeps_referents_and_authority():
    model = _make_model(_ON_CONFIG)
    store, reference = _seed(model)
    parent = replace(ConceptualMeaning.from_description(torch.ones(store.nDim)), role_refs=(reference, None, None))
    store.append_meaning(parent)
    restored = _make_model(_ON_CONFIG)
    restored.load_state_dict(model.state_dict())
    current = restored.symbolSpace.ltm_store
    assert len(current) == 2 and current.occurrence_of(0) == reference
    assert current.row(0)["trust"] == .75
    assert current.read_structure(current.occurrence_of(1))["incomplete"] == ()
    assert int(restored.symbolSpace.truth_layer.count) == 1
