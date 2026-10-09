"""September 20 review: one policy, complete context, honest grammar ownership."""
from dataclasses import replace

import pytest
import torch

from test_normal_thought_controller import _catalog_world


@pytest.mark.parametrize('change', [
    {'polarity': False}, {'mode': 'assertive'},
    {'bindings': {'subject': ('sym', 999)}}, {'scope': {'place': 'workshop'}},
])
def test_semantic_metadata_changes_the_controller_input(change):
    model, registry, _, part, whole = _catalog_world()
    original = registry.form('part', part, whole)
    changed = replace(original, **change)
    a = model._selected_thought_context(original, level=0, pressure=0)
    b = model._selected_thought_context(changed, level=0, pressure=0)
    assert not torch.equal(a, b)


def test_no_standalone_language_codec_or_second_policy():
    import Language
    import Models
    import reasoning
    from pathlib import Path
    root = Path(Models.__file__).parent
    assert not (root / 'LinguisticMeaning.py').exists()
    assert not (root / 'thinking.py').exists()
    assert not hasattr(Language, 'WhatStepChooser')
    assert not hasattr(Models.BasicModel, 'configure_meaning_learning')
    assert not hasattr(Models.BasicModel, '_answer_policy_loss')
    assert not hasattr(reasoning, 'InterveningIdeaGenerator')
    assert not any(name.startswith('legacy_') for name in vars(reasoning.TruthGroundedReasoner))


def test_metadata_preserves_binding_identity_without_allocator_magnitudes():
    model, registry, _, part, whole = _catalog_world()
    question = registry.form('part', part, whole, bindings={'x': part}, scope={'place': 'a'})
    renamed = replace(question, role_refs=(('sym', 991), ('sym', 883), ('sym', 775)),
                      bindings=dict(question.bindings, x=('sym', 991)))
    features = lambda q: model._selected_thought_context(q, level=0, pressure=0)
    torch.testing.assert_close(features(question), features(renamed))
    assert not torch.equal(features(question), features(replace(question, bindings={'x': whole})))
    assert not torch.equal(features(question), features(replace(question, scope={'place': 'b'})))


def test_old_context_checkpoint_resets_policy_and_its_optimizer_moments():
    from Language import SelectedThoughtChooser
    from checkpoint_migrations import build_optimizer_param_manifest, remap_optimizer_state_by_name
    from Models import BasicModel
    old = BasicModel()
    old.selected_thought_choosers = torch.nn.ModuleDict({
        '8': SelectedThoughtChooser(context_dim=9 * 8 + 15)})
    optimizer = torch.optim.Adam(old.parameters(), lr=.01)
    sum(p.sum() for p in old.parameters()).backward()
    optimizer.step()
    state = {key: value.detach().clone() for key, value in old.state_dict().items()}
    old_manifest = build_optimizer_param_manifest(optimizer, old.named_parameters())
    new, registry, _, part, whole = _catalog_world()
    new._materialize_answer_path_from_checkpoint(state)
    state.update({key: value.detach().clone() for key, value in new.state_dict().items()})
    new.load_state_dict(state, strict=True)
    chooser = new._selected_thought_chooser(registry.form('part', part, whole))
    assert not bool(chooser.chooser.mlp[-1].weight.any())
    new_optimizer = torch.optim.Adam(chooser.parameters(), lr=.01)
    manifest = build_optimizer_param_manifest(new_optimizer, new.named_parameters())
    migrated = remap_optimizer_state_by_name(optimizer.state_dict(), old_manifest,
        new_optimizer.state_dict(), manifest, reset_parameters=new._pending_thought_policy_reset)
    assert migrated.state['state'] == {}
    assert len(migrated.diagnostics.dropped_saved_states) == len(tuple(old.parameters()))


def test_memory_attention_is_bounded_metered_and_row_local():
    from Layers import WhatInteractionMemory, TernaryTruthStore
    from QueryWork import QueryWorkBudget
    model, registry, _, part, whole = _catalog_world()
    memory = WhatInteractionMemory(batch=2, capacity=64, detach_mode='episode')
    model.symbolSpace.what_memory = memory
    store = model.symbolSpace.ltm_store = TernaryTruthStore(8, capacity=16)
    question = registry.form('part', part, whole)
    fact = replace(question, mode='assertive')
    store.append_meaning(fact, kind='fact', trust=.8)
    private = store.append_meaning(replace(fact, roles=fact.roles + 30), kind='observation')
    memory.begin_thought_episode(question, b=0, work_budget=128)
    memory.begin_thought_episode(replace(question, roles=question.roles + 20), b=1, work_budget=128)
    with model._query_boundary_scope((0,)):
        work = QueryWorkBudget(128)
        first = model._selected_thought_memory(question, row=0, work=work)
        assert 0 < work.spent <= 8
        assert work.counts['context_record'] == work.spent
        assert bool(first[0].abs().any()) and not bool(first[1].abs().any())
        store.slots[private].fill_(500)
        memory.commit_thought(replace(question, roles=question.roles - 50), b=1,
                              operation='equal')
        second = model._selected_thought_memory(question, row=0, work=QueryWorkBudget(128))
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)
        store.slots[0].add_(1)
        third = model._selected_thought_memory(question, row=0, work=QueryWorkBudget(128))
        torch.testing.assert_close(first[1], third[1])  # a write alone grants no read
    for row in (0, 1):
        memory.finish_thought(question, b=row)
        memory.end_what_episode(row)


@pytest.mark.parametrize('depth', [1, 2])
def test_policy_selects_nested_what_part_and_receives_actual_episode_credit(depth, monkeypatch):
    """A real grammar softmax completes a departure inside nested asks."""
    from test_item6_2_thinking import credit_chain_world, run
    from Queries import ThoughtOperationCandidate
    from ThoughtReferences import needs_episode, open_slots
    import ThoughtStream
    model, registry, store, goal, menu = credit_chain_world(monkeypatch)
    base_menu = ThoughtStream.candidates
    outer = goal
    for _ in range(depth): outer = registry.form('ask', outer)
    def nested(registry, root, active, current, records, descriptions=()):
        if not needs_episode(current): return ()
        name = registry.signature_for(root, verify_reference=False).operation.semantic_id
        if name == 'ask':
            return (ThoughtOperationCandidate(registry.operation_spec('ask'), root, ()),)
        return base_menu(registry, root, active, current, records, descriptions)
    monkeypatch.setattr(ThoughtStream, 'candidates', nested)
    selected = run(model, outer, work_budget=128,
        score=lambda result: dict(reconstruction=float(needs_episode(result.meaning)), answer=0.))
    assert not open_slots(selected.meaning)
    assert max(record.level for record in selected.records) == depth
    assert sum(record.kind == 'descend' for record in selected.records) == depth
    assert selected.support_true == 1.
    assert model._last_thought_comparison['explore_kept']
    before = model.shared_grammar.chooser.mlp[-1].weight.detach().clone()
    optimizer = torch.optim.SGD(model.shared_grammar.parameters(), lr=.1)
    optimizer.zero_grad();model._last_thought_score_function['surrogate'].backward();optimizer.step()
    assert not torch.equal(before, model.shared_grammar.chooser.mlp[-1].weight)
    memory = model._what_memory()
    for record in selected.records:
        if record.kind == 'return':
            assert any(memory.thought_reference(record) in later.sources for later in selected.records)
