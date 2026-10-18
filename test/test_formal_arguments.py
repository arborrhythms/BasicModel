"""Fixed formal argument basis; no learned prototypes or merges."""
import os
import sys
from collections import namedtuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from PartsOfSpeech import formal_arguments


def vocabulary(grammar):
    roles = list(formal_arguments(grammar.rules_upward))
    return roles, {name:i for i,name in enumerate(roles)}, len(roles)

# Minimal rule stand-in: only the fields the helper reads.
_Rule = namedtuple("_Rule", ["method_name", "rhs_symbols", "lhs"], defaults=["result"])


class _G:
    def __init__(self, rules):
        self.rules_upward = rules


def test_inputs_and_outputs_enumerated():
    g = _G([
        _Rule("isEqual", ["isEqual_argin1", "isEqual_argin2"]),
        _Rule("exist", ["exist_argin1"]),
    ])
    roles, idx, n = vocabulary(g)
    assert n == len(roles) == len(idx)
    # 3 input slots (isEqual_argin1/argin2, exist_argin1) + 2 output slots (isEqual_argout1, exist_argout1)
    assert set(roles) == {
        "isEqual_argin1", "isEqual_argin2", "exist_argin1", "isEqual_argout1", "exist_argout1"}
    assert n == 5
    # inputs sorted come before outputs; index map is a bijection over columns
    assert sorted(idx.values()) == list(range(n))


def test_methodless_projection_skipped():
    g = _G([
        _Rule("lift", ["lift_argin1", "lift_argin2"]),
        _Rule(None, ["NP"]),          # epsilon / passthrough -> no role
        _Rule("", ["X"]),             # falsy method -> skipped
    ])
    roles, idx, n = vocabulary(g)
    assert set(roles) == {"lift_argin1", "lift_argin2", "lift_argout1"}
    assert n == 3


def test_deterministic_order():
    g = _G([_Rule("b", ["b_argin1"]), _Rule("a", ["a_argin1", "a_argin2"])])
    r1, _, _ = vocabulary(g)
    r2, _, _ = vocabulary(g)
    assert r1 == r2                     # stable across calls
    # inputs (sorted by method,pos) precede outputs (sorted by method)
    assert r1 == ["a_argin1", "a_argin2", "a_argout1", "b_argin1", "b_argout1"]


def test_empty_grammar_is_zero_width():
    roles, idx, n = vocabulary(_G([]))
    assert roles == [] and idx == {} and n == 0


def test_aliases_and_all_outputs_have_distinct_arguments():
    from Language import Grammar
    from PartsOfSpeech import operation_name
    grammar = Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    names = [operation_name(rule) for rule in grammar.rules_upward if operation_name(rule)]
    assert len(names) == len(set(names))
    arguments = formal_arguments(grammar.rules_upward)
    assert {'lower_argin1','definite_argin1','universal_argin1',
            'transpose_argout1','transpose_argout2'} <= set(arguments)
