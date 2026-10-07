"""Exhaustive, face-invariant operator effects and operand declarations."""
import pytest


def test_every_shipped_rule_has_exhaustive_effects_and_operand_kinds():
    from Language import Grammar
    from AccessibleMind import OperatorEffects
    grammar=Grammar();grammar.load_from_grammar_file('complete.grammar')
    for rule in grammar.rules + grammar.thought_rules + grammar.ps_rules:
        assert len(rule.operand_kinds) == rule.arity
        assert set(rule.operand_kinds) <= {'field','symbol'}
        effects=OperatorEffects(rule.effect_reads, rule.effect_writes)
        if 'field' in rule.operand_kinds:
            assert effects.reads, rule.canonical
        elif rule.method_name:
            assert effects.writes, rule.canonical


def test_aliases_and_faces_inherit_one_complete_contract():
    from Language import Grammar
    from AccessibleMind import OperatorEffects, Subsystem as S
    grammar=Grammar(); rules=[]
    for body in ('r.forward(r_I1, r_I2)', 'r.reverse(r_O1)'):
        grammar._fill_rule_list(rules, {'rule': {'_': 'r_O1 = '+body, 'implementation': 'part'}})
    assert rules[0].effect_writes == rules[1].effect_writes
    effects=OperatorEffects(rules[0].effect_reads, rules[0].effect_writes)
    assert S.SERIAL in effects.for_face('compose').writes
    assert S.PERCEPT in effects.for_face('generate').writes
    assert S.SERIAL not in effects.for_face('generate').writes
    assert S.LTM not in effects.writes


@pytest.mark.parametrize('writes', ['serial', 'serial,percept,budget,ltm', 'serial,serial'])
def test_an_explicit_write_list_cannot_omit_add_or_repeat_effects(writes):
    from Language import Grammar
    with pytest.raises(ValueError, match='writes'):
        Grammar()._fill_rule_list([], {'rule': {
            '_': 'x_O1 = x.forward(x_I1, x_I2)', 'implementation': 'sum', 'writes': writes}})


def test_field_eligibility_belongs_to_the_implementation_property():
    from Language import Grammar, GRAMMAR_LAYER_CLASSES, SumLayer
    class Ordered(SumLayer):
        field_eligible=False
    GRAMMAR_LAYER_CLASSES['opaque_extension']=Ordered
    try:
        with pytest.raises(ValueError, match='field'):
            Grammar()._fill_rule_list([], {'rule': {
                '_': 'x_O1 = x.forward(x_I1, x_I2)',
                'implementation': 'opaque_extension', 'operands': 'field,field'}})
    finally:
        del GRAMMAR_LAYER_CLASSES['opaque_extension']


def test_round_rejects_undeclared_and_second_writes_before_committing():
    from AccessibleMind import OperatorEffects, EffectRound, Subsystem as S
    round=EffectRound(OperatorEffects(('serial',), ('serial',)), 'thought')
    round.claim(S.SERIAL)
    with pytest.raises(ValueError, match='once'):
        round.claim(S.SERIAL)
    with pytest.raises(ValueError, match='undeclared'):
        round.claim(S.LTM)


def test_order_change_is_declared_and_survives_an_alias():
    from Language import Grammar
    grammar=Grammar();rules=[]
    grammar._fill_rule_list(rules, {'rule': {'_':'S2 = renamed.forward(N1, V1)', 'implementation':'lower'}})
    rule=rules[0]._replace(method_name='no_semantic_spelling')
    assert grammar._rule_order_signature(rule).order_delta == -1


def test_mean_bundling_is_not_a_field_operation():
    from Language import Grammar
    with pytest.raises(ValueError, match='field'):
        Grammar()._fill_rule_list([], {'rule': {
            '_':'bundle_O1 = bundle.forward(bundle_I1, bundle_I2)',
            'implementation':'disjunction', 'operands':'field,field'}})


def test_thought_result_and_effect_properties_survive_semantic_rename():
    from dataclasses import replace
    from Queries import THOUGHT_EXECUTORS
    for original in THOUGHT_EXECUTORS.values():
        renamed=replace(original, semantic_id='opaque')
        assert renamed.result_kind == original.result_kind
        assert renamed.effect_kind == original.effect_kind
        assert renamed.method_grants == original.method_grants
