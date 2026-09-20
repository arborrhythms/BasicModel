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
                      bindings={'x': ('sym', 991)})
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
    new.load_state_dict(state, strict=True)
    chooser = new._selected_thought_chooser(registry.form('part', part, whole))
    assert not bool(chooser.mlp[-1].weight.any())
    new_optimizer = torch.optim.Adam(chooser.parameters(), lr=.01)
    manifest = build_optimizer_param_manifest(new_optimizer, new.named_parameters())
    migrated = remap_optimizer_state_by_name(optimizer.state_dict(), old_manifest,
        new_optimizer.state_dict(), manifest, reset_parameters=new._pending_thought_policy_reset)
    assert migrated.state['state'] == {}
    assert len(migrated.diagnostics.dropped_saved_states) == len(tuple(chooser.parameters()))


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
        assert bool(first[0].abs().any()) and bool(first[1].abs().any())
        store.slots[private].fill_(500)
        memory.commit_thought(replace(question, roles=question.roles - 50), b=1,
                              operation='equal')
        second = model._selected_thought_memory(question, row=0, work=QueryWorkBudget(128))
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)
        store.slots[0].add_(1)
        third = model._selected_thought_memory(question, row=0, work=QueryWorkBudget(128))
        assert not torch.equal(first[1], third[1])
    for row in (0, 1):
        memory.finish_thought(question, b=row)
        memory.end_what_episode(row)


@pytest.mark.parametrize('depth', [1, 2])
def test_policy_selects_nested_what_part_and_receives_actual_episode_credit(depth):
    """Mechanism only: teach a routing policy, then execute without a script.

    The descent is a real MLP argmax with policy log probability. The separate
    multi-seed utility study must determine whether such descent is useful.
    """
    torch.manual_seed(23)
    model, registry, memory, part, whole = _catalog_world()
    model.what_thinking_hidden = 48
    model.selected_thought_policy_weight = 1.
    model.eval()
    question = registry.form('part', part, whole)
    record = memory.begin_thought_episode(question, work_budget=128)
    actions = registry.controller_candidates(question, question, question,
        descriptions=((question, memory.thought_reference(record)),)) + (None,)
    memory.finish_thought(question)
    memory.end_what_episode()
    chooser = model._selected_thought_chooser(question)
    optimizer = torch.optim.Adam(chooser.parameters(), lr=.015)
    samples = []
    for level in range(depth + 1):
        for supported in (False, True):
            evidence = {'support_true': 1.} if supported else None
            contexts = torch.stack([model._selected_thought_context(question, question,
                question if action is None else action.request,
                level=level, pressure=0, evidence=evidence) for action in actions])
            wanted = None if supported else ('what' if level < depth else 'part')
            target = next(i for i, a in enumerate(actions)
                          if (a is None if wanted is None else a is not None and a.semantic_id == wanted))
            samples.append((contexts.detach(), target))
    wrapper = next(a.request for a in actions if a is not None and a.semantic_id == 'what')
    after_actions = (None,) + registry.controller_candidates(question, wrapper, wrapper)
    for level in range(depth + 1):
        contexts = torch.stack([model._selected_thought_context(question, wrapper,
            wrapper if action is None else action.request, level=level, pressure=0,
            evidence={'support_true': 1.}) for action in after_actions])
        samples.append((contexts.detach(), 0, after_actions))
    samples = [(sample[0], sample[1], actions) if len(sample) == 2 else sample
               for sample in samples]
    # Ignore the pressure/memory fields in this explicitly constructed mechanism
    # probe, so differing actual work cannot substitute for semantic action.
    width = question.roles.shape[-1]
    keep = torch.zeros_like(chooser.mlp[0].weight)
    keep[:, :9 * width + 9] = 1  # role values and masks
    keep[:, 9 * width + 9] = 1   # context level
    keep[:, 9 * width + 11:9 * width + 15] = 1  # actual evidence
    keep[:, -2:] = 1  # operation/conclude
    for _ in range(800):
        optimizer.zero_grad()
        loss = sum(torch.nn.functional.cross_entropy(
            chooser.logits(ctx, [a is None for a in menu])[None], torch.tensor([target]))
            for ctx, target, menu in samples)
        loss.backward()
        chooser.mlp[0].weight.grad.mul_(keep)
        optimizer.step()
        with torch.no_grad():
            chooser.mlp[0].weight.mul_(keep)
        if float(loss.detach()) < .015:
            break
    assert all(chooser.choose(ctx, [a is None for a in menu])[0] == target
               for ctx, target, menu in samples)
    with model._query_boundary_scope((0,)):
        selected = model.run_selected_thought(question, work_budget=256)
    assert max(record.level for record in selected.records) == depth, [(r.level, r.operation, r.support_true) for r in selected.records]
    assert sum(record.kind == 'descend' for record in selected.records) == depth
    assert selected.support_true == 1.
    assert any(record.operation == 'part' and record.level == depth for record in selected.records)
    assert len(model._selected_thought_policy_records) >= 2 * depth + 2
    before = chooser.mlp[-1].weight.detach().clone()
    optimizer.zero_grad()
    model._selected_thought_policy_baseline = 0.
    credit = model._selected_thought_policy_loss(torch.tensor(.3))
    credit.backward()
    optimizer.step()
    assert not torch.equal(before, chooser.mlp[-1].weight)
    returned = [record for record in selected.records if record.kind == 'return']
    for record in returned:
        assert any(memory.thought_reference(record) in later.sources for later in selected.records)
    memory.end_what_episode()
