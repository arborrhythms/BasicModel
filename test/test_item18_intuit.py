"""The learned field and the retained implication engine have separate reads."""
from dataclasses import replace

import pytest
import torch

pytestmark = pytest.mark.assessment_rung(1)

from test_thought_infer import setup, invoke, idea
from ThoughtChunks import put
from ThoughtReferences import bindings


def search(model, registry, premise):
    held = put(model._concept_owner().stm, premise, row=0)
    return registry.form('infer', held, open_roles=('argin2',))


def test_infer_ordinary_derivation_does_not_claim_a_knowing_deposit():
    from test_thought_infer import operands
    from Queries import THOUGHT_EXECUTORS
    from AccessibleMind import EffectRound, OperatorEffects, Subsystem, apply_thought_effect
    from QueryWork import QueryWorkBudget
    model, registry, store, refs = setup()
    premise, rule, _ = operands(model, registry, store, refs)
    result = invoke(model, registry, registry.form('infer', premise, rule))
    descriptor = THOUGHT_EXECUTORS['infer']
    effects = EffectRound(OperatorEffects(descriptor.read_scope, descriptor.write_scope), 'thought')
    apply_thought_effect(model, result, row=0, work=QueryWorkBudget(256), effect_round=effects)
    assert Subsystem.KNOWING not in effects.written


def test_85_infer_keeps_the_old_three_rule_supports():
    model, registry, store, refs = setup()
    owner = model._concept_owner()
    # This fixed input was also evaluated by the pre-move intuit face.
    with torch.no_grad():
        for ref, first in zip(refs, (.8, .3, .4)):
            value = owner.similarity_codebook.W[owner._csw_row_of(ref[1])]
            value.zero_()
            value[:2] = value.new_tensor((first, 1-first))
    for a, b, pair in ((refs[0], refs[1], (.8, .2)), (refs[0], refs[2], (.6, .3)),
                       (refs[1], refs[2], (.7, .4))):
        store.append_relation(a[1], b[1], truth_plus=pair[0], truth_minus=pair[1])
    from ConceptSpread import open_prediction
    open_prediction(model, torch.tensor([True]))
    result = invoke(model, registry, search(model, registry, idea(registry, refs[0])))
    assert (result.support_true, result.support_false) == pytest.approx((.8, .4))
    torch.testing.assert_close(torch.tensor(result.evidence['firings']),
                               torch.tensor([[.8, .2], [.6, .3], [.5, .4]]))
    expected = torch.zeros(8, 2)
    expected[:2] = torch.tensor([[.4, .4], [.7, .4]])
    torch.testing.assert_close(torch.tensor(bindings(result.evidence['meaning'])['_intuition_field']), expected)
    from Intuition import replay
    field, pair = replay(result.evidence['intuition_derivation'])
    torch.testing.assert_close(field, expected)
    torch.testing.assert_close(pair, torch.tensor([.8, .4]))
    assert result.evidence['evidence_kind'] == 'inference'
    print(dict(supports=(result.support_true, result.support_false),
               firings=result.evidence['firings'], field=field.tolist()))


def test_84_intuit_reads_once_at_observation_and_no_relation_row(tmp_path, monkeypatch):
    from test_item18_expect import lesson, read
    from Queries import ThoughtLTMCapability
    model = lesson(tmp_path, monkeypatch)
    calls = []
    owner = model._concept_owner()
    hook = owner.expect_stack.register_forward_hook(lambda *args: calls.append(True))
    try:
        read(model, 'alpha gamma')
        assert len(calls) == 1
        from IntuitionField import read as learned_read
        from QueryWork import QueryWorkBudget
        def forbidden(*args, **kwargs):
            raise AssertionError('learned intuition read an implication')
        with monkeypatch.context() as guard:
            guard.setattr(ThoughtLTMCapability, 'implication_rules', forbidden)
            guard.setattr(model.symbolSpace.ltm_store, 'part_row', forbidden)
            value = learned_read(owner, row=0, work=QueryWorkBudget(64))
            assert value is None
            from ThoughtFaces import intuit
            from types import SimpleNamespace
            # The face receives no LTM capability at all.
            context = SimpleNamespace(row=0, work=QueryWorkBudget(64),
                conceptual_space=SimpleNamespace(intuition=lambda **kw: learned_read(owner, **kw)))
            effect = intuit(context, {'argin1': None})
            assert effect['evidence_kind'] == 'intuition' and effect['learned_field']
            assert effect['value'] is None
            assert len(calls) == 1
            assert not effect['witnesses']
    finally:
        hook.remove()
        from assessment_support import record
        record(model, dict(word='alpha', words=['alpha', 'gamma']))
        model.End(); model.symbolSpace.soft_reset()


def test_83_an_unseen_conclusion_still_needs_inference():
    from Understanding import ConceptualField
    from FieldProjection import close
    from SentenceField import publish
    from IntuitionField import open_field
    from AccessibleMind import apply_thought_effect
    from QueryWork import QueryWorkBudget
    from Layers import Error
    model, registry, store, original = setup()
    owner = model._concept_owner()
    object.__setattr__(model, 'conceptualSpaces', (owner,))
    refs = [*original, *(('sym', owner.new_concept(order=0)) for _ in range(5))]
    rows = [owner._csw_row_of(ref[1]) for ref in refs]
    with torch.no_grad():
        owner.similarity_codebook.W[rows] = torch.eye(8)
        owner.similarity_codebook.W[rows[6]] = torch.eye(8)[0]+torch.eye(8)[1]
        owner.similarity_codebook.W[rows[7]] = torch.eye(8)[2]+torch.eye(8)[3]
    for target, members in ((6, (0, 1)), (7, (2, 3))):
        for member in members:
            owner.add_concept_edge(rows[target], rows[member], 1., conjunctive=True)
    for source, target in ((6, 4), (7, 5)):
        store.append_relation(refs[source][1], refs[target][1], truth_plus=1.)
    from ConceptSpread import open_prediction
    open_prediction(model, torch.tensor([True]))
    ids = torch.tensor([ref[1] for ref in refs])
    def observation(members):
        values = torch.zeros(8, 1, 1, 2)
        values[list(members), 0, 0, 0] = 1
        return ConceptualField(ids, close(owner, rows, values),
            torch.tensor([[[0, 1]]]), torch.tensor([[[0, 1]]]), None)
    # The independent thought teacher and its 200 teaching presentations
    # are retired by §68.53.4. The unseen-combination inference contract stays.
    from IntuitionField import read as deposit
    from SentenceField import knowing
    publish(model, open_field(model, observation((2, 3))))
    open_prediction(model, torch.tensor([True]))
    deposit(owner, row=0, work=QueryWorkBudget(64))
    unseen = knowing(model)
    unlearned = (unseen.concept_ids == refs[5][1]).nonzero().item()
    assert unseen.evidence[unlearned, 0, 0, 0] == 0
    publish(model, unseen)
    result = invoke(model, registry, search(model, registry, idea(registry, refs[7])))
    apply_thought_effect(model, result, row=0, work=QueryWorkBudget(256))
    assert owner.subspace._concept_activations[unlearned, 0, 0, 0] == 1
    print(dict(unseen=float(unseen.evidence[unlearned, 0, 0, 0].detach()),
               inferred=float(owner.subspace._concept_activations[unlearned, 0, 0, 0].detach())))
