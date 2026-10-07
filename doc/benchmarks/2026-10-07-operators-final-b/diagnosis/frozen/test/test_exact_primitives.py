"""Exact primitives, lexer, numeral code and verifier (mathematical thinking
spec sections 3, 5, 6.3, 9; plan Phase 1).  Pure Python + torch, no model."""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from exact import (Bound, Equation, ExactLexer, ExactState, ExactVerifier,  # noqa: E402
                   MathProblemGenerator, PRIMITIVES, Unbound, eval_expr,
                   illumination, numeral_code, num, var)


# -- lexer ---------------------------------------------------------------------

def test_lexer_parses_the_spec_examples():
    eqs, q = ExactLexer.lex("a = 3 ; b = a + 4 ; c = 2 * b ; what is c ?")
    assert q == "c"
    assert [e.render(words=False) for e in eqs] == ["a = 3", "b = a + 4", "c = 2 * b"]
    assert [e.render() for e in eqs] == ["a equals 3", "b equals a plus 4", "c equals 2 times b"]
    # word surfaces lex identically to glyph surfaces
    assert ExactLexer.lex("a equals 3 ; b equals a plus 4 ; what is b") == ExactLexer.lex(
        "a = 3 ; b = a + 4 ; what is b ?")
    assert eqs[2].rhs == ("mul", num(2), var("b"))
    eqs, q = ExactLexer.lex("x + y = 12 ; y = 2 * x ; what is x ?")
    assert q == "x" and eqs[0].solved_var is None and eqs[1].solved_var == "y"


@pytest.mark.parametrize("stage", [1, 2])
def test_lexer_round_trips_generator_surfaces(stage):
    gen = MathProblemGenerator(seed=3, range=64, depths=(1, 2, 3), stage=stage)
    for p in gen.problems(40):
        eqs, q = ExactLexer.lex(p.surface())
        assert q == p.query
        assert eqs == p.equations


@pytest.mark.parametrize("bad", ["", "a = 3", "a = 3 ; what is", "a = 3 = 4 ; what is a",
                                 "a + = 3 ; what is a ?", "a = 3 ; what is 7 ?"])
def test_lexer_never_guesses(bad):
    with pytest.raises(ValueError):
        ExactLexer.lex(bad)


# -- primitives ----------------------------------------------------------------

def test_evaluate_reports_unbound_dependencies_then_resolves():
    state, q = ExactState.from_surface("a = 3 ; b = a + 4 ; c = 2 * b ; what is c ?")
    assert state.lookup("b") is None
    assert state.evaluate(2) == Unbound({"b"})
    assert state.evaluate(1) == Unbound({"a"})
    assert state.evaluate(0) == 3
    state.bind("a", 3)
    assert state.evaluate(1) == 7
    state.bind("b", 7)
    assert state.evaluate(2) == 14
    state.bind("c", 14)
    assert state.lookup(q) == 14


def test_substitute_and_constrain_solve_stage_two():
    state, q = ExactState.from_surface("x + y = 12 ; y = 2 * x ; what is x ?")
    assert state.constrain(0) == Unbound({"x", "y"})
    rewritten = state.substitute(0, 1)
    assert isinstance(rewritten, Equation) and "y" not in rewritten.vars
    bound = state.constrain(0)
    assert bound == Bound("x", 4)
    state.bind("x", 4)
    assert state.constrain(1) == Bound("y", 8)


def test_primitives_are_total_on_random_operands():
    gen = MathProblemGenerator(seed=11, range=32, depths=(1, 2, 3), stage=1)
    import random
    rng = random.Random(0)
    for p in gen.problems(20):
        state = ExactState(list(p.equations), range=p.range)
        names = sorted(set().union(*(e.vars for e in p.equations))) + ["zz"]
        for _ in range(40):
            op = rng.choice(PRIMITIVES)
            n = len(state.equations)
            if op == "lookup":
                operands = (rng.choice(names),)
            elif op == "bind":
                operands = (rng.choice(names), rng.randrange(-5, 40))
            elif op == "substitute":
                operands = (rng.randrange(-1, n + 1), rng.randrange(-1, n + 1))
            else:
                operands = (rng.randrange(-1, n + 1),)
            step = state.execute(op, *operands)
            res = step["result"]
            assert step["operation"] == f"exact:{op}"
            assert res is None or isinstance(res, (int, Unbound, Bound, Equation))
            if op == "bind":
                assert 0 <= res < p.range
        assert state.executions == 40 and len(state.trace) == 40


def test_execute_rejects_unknown_primitive():
    state, _ = ExactState.from_surface("a = 1 ; what is a")
    with pytest.raises(ValueError):
        state.execute("solve", 0)


# -- numeral code -------------------------------------------------------------

def test_numeral_code_is_exact_and_distinct():
    codes = [numeral_code(n, 14, bits=6) for n in range(64)]
    for n, c in enumerate(codes):
        assert c.shape == (14,)
        assert set(c[:6].tolist()) <= {-1.0, 1.0} and torch.all(c[6:] == 0)
        decoded = sum(int(c[k] > 0) << k for k in range(6))
        assert decoded == n
    assert len({tuple(c.tolist()) for c in codes}) == 64


# -- verifier and illumination ------------------------------------------------

@pytest.mark.parametrize("stage", [1, 2])
def test_verifier_accepts_solver_trace_and_rejects_edits(stage):
    gen = MathProblemGenerator(seed=5, range=64, depths=(2, 3), stage=stage)
    verifier = ExactVerifier()
    for p in gen.problems(12):
        bindings, trace = MathProblemGenerator.solve(p)
        assert bindings[p.query] == p.answer
        report = verifier.check(trace, p, answer=p.answer)
        assert report.rejected == 0 and report.accepted == len(trace)
        assert report.valid and report.final == p.answer
        # Edit one recorded bind value: the step is rejected (unjustified)
        # and the derivation is no longer valid.
        edited = [dict(s) for s in trace]
        idx = next(i for i, s in enumerate(edited) if s["operation"] == "exact:bind")
        v, n = edited[idx]["operands"]
        edited[idx]["operands"] = (v, (n + 1) % p.range)
        report = verifier.check(edited, p, answer=p.answer)
        assert report.rejected >= 1
        assert not report.valid
        # A trace that merely CLAIMS the answer without steps is not valid.
        assert not verifier.check([], p, answer=p.answer).valid


def test_illumination_is_monotone_along_the_solver_trace():
    gen = MathProblemGenerator(seed=9, range=16, depths=(2,), distractors=(0,), stage=1)
    p = gen.problem()
    _, trace = MathProblemGenerator.solve(p)
    report = ExactVerifier(measure_illumination=True).check(trace, p, answer=p.answer)
    series = report.illumination
    assert series[0] == 0.0 and series[-1] == 1.0
    assert all(a <= b + 1e-9 for a, b in zip(series, series[1:]))


def test_illumination_counts_applied_constraints_only():
    import math
    state, _ = ExactState.from_surface("a = 3 ; b = a + 4 ; what is b ?", range=16)
    from exact import Problem
    p = Problem(tuple(state.equations), "b", 7, 1, "x", 16, 1, frozenset("ab"), ("a", "b"))
    # Nothing applied: every value of b is a candidate.
    assert illumination(state, p) == 0.0
    # Applying the premise that defines b (without a bound) leaves a free:
    # b = a + 4 with a in [0, 16) gives 12 in-range candidates.
    state.evaluate(1)
    assert abs(illumination(state, p) - (1.0 - math.log(12) / math.log(16))) < 1e-9
    state.evaluate(0); state.bind("a", 3)
    assert illumination(state, p) == 1.0          # b is now determined
    state.bind("b", 7)
    assert illumination(state, p) == 1.0
