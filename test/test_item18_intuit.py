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
            value = owner.similarity_codebook.W[owner._concept_row_of(ref[1])]
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
    from LookForecast import LookForecast
    import ConceptSpread
    open_calls, look_calls = [], []
    original_open, original_look = ConceptSpread.open_prediction, LookForecast.forecast
    def opened(*args, **kwargs):
        open_calls.append(True)
        return original_open(*args, **kwargs)
    def looked(path, extent, rows=None):
        if extent.any() or (rows is not None and rows.any()):
            look_calls.append(True)
        return original_look(path, extent, rows)
    monkeypatch.setattr(ConceptSpread, 'open_prediction', opened)
    monkeypatch.setattr(LookForecast, 'forecast', looked)
    spread = ConceptSpread.spread
    def counted(*args, **kwargs):
        calls.append(True)
        return spread(*args, **kwargs)
    monkeypatch.setattr(ConceptSpread, 'spread', counted)
    intuited = []
    hook = owner.intuit_map.register_forward_hook(lambda *args: intuited.append(True))
    try:
        read(model, 'alpha gamma')
        assert len(open_calls) == 1
        observed_calls = len(calls)
        assert observed_calls == 1 + len(look_calls)
        from IntuitionField import read as learned_read
        from QueryWork import QueryWorkBudget
        def forbidden(*args, **kwargs):
            raise AssertionError('learned intuition read an implication')
        with monkeypatch.context() as guard:
            guard.setattr(ThoughtLTMCapability, 'implication_rules', forbidden)
            guard.setattr(model.symbolSpace.ltm_store, 'part_row', forbidden)
            value = learned_read(owner, row=0, work=QueryWorkBudget(64))
            assert value is None
            # intuit reads its own map, and not expectation's spread.
            assert len(intuited) == 1 and len(calls) == observed_calls
            from ThoughtFaces import intuit
            from types import SimpleNamespace
            # The face receives no LTM capability at all.
            context = SimpleNamespace(row=0, work=QueryWorkBudget(64),
                conceptual_space=SimpleNamespace(intuition=lambda **kw: learned_read(owner, **kw)))
            effect = intuit(context, {'argin1': None})
            assert effect['evidence_kind'] == 'intuition' and effect['learned_field']
            assert effect['value'] is None
            assert len(calls) == observed_calls and len(intuited) == 2
            assert not effect['witnesses']
    finally:
        hook.remove()
        from assessment_support import record
        record(model, dict(word='alpha', words=['alpha', 'gamma']))
        model.End(); model.symbolSpace.soft_reset()


