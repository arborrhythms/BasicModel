"""Exact arithmetic for the mathematical-thinking testbed.

Specification: ``doc/specs/2026-09-09-mathematical-thinking.md`` (sections
3-5, 9). This module supplies, parameter-free and total:

* ``ExactLexer`` -- the deterministic clause lexer that presents a problem
  surface (``a = 3 ; b = a + 4 ; what is b ?``) as typed equations and a
  query (spec 5.1);
* ``ExactState`` -- the model-owned scratchpad ``(E, beta)`` with the five
  primitives ``lookup / evaluate / bind / substitute / constrain`` (spec 5.2),
  each recorded as a replayable trace step (spec 5.3);
* ``numeral_code`` -- the fixed binary-digit fallback code for an integer
  answer symbol (spec 6.3);
* ``MathProblemGenerator`` -- stage 1 (dependency arithmetic) and stage 2
  (simultaneous linear constraints) problems with unique solutions, depth,
  structure hash and the solver's binding order (spec 4.2);
* ``ExactVerifier`` -- the evaluation-side replay that accepts or rejects
  each emitted step and validates the final answer (spec 9);
* ``illumination`` -- the oracle-side candidate measure ``I_t`` (spec 3).

The complete solver (``MathProblemGenerator.solve``) is available only to
generation, scoring and tests; the model never receives it.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

# -- expressions --------------------------------------------------------------
#
# Expr is a nested tuple:  ("num", n) | ("var", name) | (op, left, right)
# with op in {"add", "sub", "mul"}.  ``mul`` keeps one numeric side (the
# spec's linear grammar); the lexer accepts what the grammar produces.

Expr = Tuple[Any, ...]
_OPS = {"add": "+", "sub": "-", "mul": "*"}
_SYMBOLS = {v: k for k, v in _OPS.items()}
# Word surfaces (the presented form, Alec 2026-09-09): arithmetic is a
# syntax the grammar learns, so the operators are WORDS -- transitive verbs
# (``plus``) and the copula (``equals``) -- not glyphs, which the word lexer
# drops.  The lexer accepts both spellings.
_WORDS = {"add": "plus", "sub": "minus", "mul": "times"}
_WORD_SYMBOLS = {"plus": "+", "minus": "-", "times": "*", "equals": "="}


def num(n: int) -> Expr:
    return ("num", int(n))


def var(name: str) -> Expr:
    return ("var", str(name))


def render_expr(expr: Expr, words: bool = True) -> str:
    kind = expr[0]
    if kind == "num":
        return str(expr[1])
    if kind == "var":
        return expr[1]
    op = _WORDS[kind] if words else _OPS[kind]
    return f"{render_expr(expr[1], words)} {op} {render_expr(expr[2], words)}"


def expr_vars(expr: Expr) -> FrozenSet[str]:
    kind = expr[0]
    if kind == "num":
        return frozenset()
    if kind == "var":
        return frozenset((expr[1],))
    return expr_vars(expr[1]) | expr_vars(expr[2])


@dataclass(frozen=True)
class Unbound:
    """A primitive could not finish because these variables are unbound."""

    vars: FrozenSet[str]

    def __post_init__(self):
        object.__setattr__(self, "vars", frozenset(self.vars))


@dataclass(frozen=True)
class Bound:
    """``constrain`` determined one variable."""

    var: str
    value: int


PrimitiveResult = Union[int, None, Unbound, Bound, "Equation"]


def eval_expr(expr: Expr, bindings: Dict[str, int]) -> Union[int, Unbound]:
    """Exact evaluation under ``bindings``; unbound variables are reported."""
    missing = frozenset(v for v in expr_vars(expr) if v not in bindings)
    if missing:
        return Unbound(missing)

    def go(e):
        kind = e[0]
        if kind == "num":
            return e[1]
        if kind == "var":
            return int(bindings[e[1]])
        a, b = go(e[1]), go(e[2])
        if kind == "add":
            return a + b
        if kind == "sub":
            return a - b
        return a * b

    return int(go(expr))


def linearize(expr: Expr, bindings: Dict[str, int]):
    """``expr`` as ``(coefficients: {var: int}, constant: int)`` under the
    bindings, or ``None`` when the expression is not linear in the unbound
    variables (a product of two unbound variables)."""
    kind = expr[0]
    if kind == "num":
        return {}, int(expr[1])
    if kind == "var":
        name = expr[1]
        if name in bindings:
            return {}, int(bindings[name])
        return {name: 1}, 0
    left = linearize(expr[1], bindings)
    right = linearize(expr[2], bindings)
    if left is None or right is None:
        return None
    (lc, lk), (rc, rk) = left, right
    if kind in ("add", "sub"):
        sign = 1 if kind == "add" else -1
        coeffs = dict(lc)
        for v, c in rc.items():
            coeffs[v] = coeffs.get(v, 0) + sign * c
        return {v: c for v, c in coeffs.items() if c != 0}, lk + sign * rk
    # mul: at least one side must be constant
    if lc and rc:
        return None
    if lc:
        return {v: c * rk for v, c in lc.items() if c * rk != 0}, lk * rk
    return {v: c * lk for v, c in rc.items() if c * lk != 0}, lk * rk


@dataclass(frozen=True)
class Equation:
    lhs: Expr
    rhs: Expr

    def render(self, words: bool = True) -> str:
        copula = "equals" if words else "="
        return f"{render_expr(self.lhs, words)} {copula} {render_expr(self.rhs, words)}"

    @property
    def solved_var(self) -> Optional[str]:
        """The variable this equation defines when it is in solved form
        ``v = expr`` with ``v`` not occurring on the right."""
        if self.lhs[0] == "var" and self.lhs[1] not in expr_vars(self.rhs):
            return self.lhs[1]
        return None

    @property
    def vars(self) -> FrozenSet[str]:
        return expr_vars(self.lhs) | expr_vars(self.rhs)


# -- the lexer (spec 5.1) -----------------------------------------------------

class ExactLexer:
    """Deterministic clause lexer: surface -> (equations, query).

    Grammar (whitespace-tokenized):

        problem  := clause (';' clause)* ';' question
        question := 'what' 'is' VAR '?'?
        clause   := expr '=' expr
        expr     := term (('+' | '-') term)*
        term     := atom ('*' atom)*
        atom     := NUM | VAR

    Raises ``ValueError`` on anything else; the lexer never guesses.
    """

    @staticmethod
    def _tokens(text: str) -> List[str]:
        out = []
        for raw in text.replace("?", " ? ").split():
            out.append(_WORD_SYMBOLS.get(raw.lower(), raw))
        return out

    @classmethod
    def parse_expr(cls, tokens: Sequence[str]) -> Expr:
        pos = 0

        def atom():
            nonlocal pos
            if pos >= len(tokens):
                raise ValueError("expression ends early")
            tok = tokens[pos]
            pos += 1
            if tok.isdigit():
                return num(int(tok))
            if tok.isidentifier():
                return var(tok)
            raise ValueError(f"bad atom {tok!r}")

        def term():
            nonlocal pos
            node = atom()
            while pos < len(tokens) and tokens[pos] == "*":
                pos += 1
                node = ("mul", node, atom())
            return node

        def expr():
            nonlocal pos
            node = term()
            while pos < len(tokens) and tokens[pos] in ("+", "-"):
                op = _SYMBOLS[tokens[pos]]
                pos += 1
                node = (op, node, term())
            return node

        node = expr()
        if pos != len(tokens):
            raise ValueError(f"trailing tokens {tokens[pos:]!r}")
        return node

    @classmethod
    def parse_equation(cls, clause: str) -> Equation:
        clause = " ".join(cls._tokens(clause))
        if clause.count("=") != 1:
            raise ValueError(f"clause needs exactly one '=': {clause!r}")
        lhs, rhs = clause.split("=")
        return Equation(cls.parse_expr(cls._tokens(lhs)),
                        cls.parse_expr(cls._tokens(rhs)))

    QUERY_VAR = "_"        # the synthetic referent of a bare expression

    @classmethod
    def lex(cls, surface: str) -> Tuple[Tuple[Equation, ...], str]:
        clauses = [c.strip() for c in str(surface).split(";")]
        clauses = [c for c in clauses if c]
        if not clauses:
            raise ValueError("empty problem surface")
        if (len(clauses) == 1 and "=" not in cls._tokens(clauses[0])
                and not clauses[0].lower().startswith("what")):
            # Stage 0: a bare expression IS the question ("what is a + b");
            # present it as one solved-form premise ``_ = expr`` so the
            # same primitives (evaluate / bind) answer it exactly.
            expr = cls.parse_expr(cls._tokens(clauses[0]))
            return (Equation(var(cls.QUERY_VAR), expr),), cls.QUERY_VAR
        question = cls._tokens(clauses[-1])
        if question and question[-1] == "?":
            question = question[:-1]
        if (len(question) != 3 or question[0].lower() != "what"
                or question[1].lower() != "is" or not question[2].isidentifier()):
            raise ValueError(f"bad question clause {clauses[-1]!r}")
        equations = tuple(cls.parse_equation(c) for c in clauses[:-1])
        return equations, question[2]


# -- the scratchpad and primitives (spec 5.2 / 5.3) ---------------------------

PRIMITIVES = ("lookup", "evaluate", "bind", "substitute", "constrain")


@dataclass
class ExactState:
    """``sigma = (E, beta)``: the equations as presented (``substitute``
    rewrites them in place) and the partial binding map.  Every primitive
    is total: it returns a value, ``Unbound``, ``Bound``, an ``Equation`` or
    ``None`` and never raises on a well-typed operand.  ``execute`` records
    the spec 5.3 trace step and counts executions."""

    equations: List[Equation]
    bindings: Dict[str, int] = field(default_factory=dict)
    range: int = 64
    executions: int = 0
    trace: List[dict] = field(default_factory=list)
    # Premise indices the model has APPLIED (evaluate / substitute /
    # constrain); the oracle-side candidate measure counts only these.
    applied: set = field(default_factory=set)

    @classmethod
    def from_surface(cls, surface: str, range: int = 64) -> Tuple["ExactState", str]:
        equations, query = ExactLexer.lex(surface)
        return cls(list(equations), range=int(range)), query

    # primitives ---------------------------------------------------------

    def lookup(self, v: str) -> Optional[int]:
        return self.bindings.get(str(v))

    def evaluate(self, i: int) -> Union[int, Unbound]:
        eq = self._equation(i)
        if eq is None:
            return Unbound(frozenset())
        self.applied.add(int(i))
        return eval_expr(eq.rhs, self.bindings)

    def bind(self, v: str, n: int) -> Optional[int]:
        """Bind ``v`` to ``n`` (clamped into ``[0, range)``); returns the
        stored value.  Rebinding overwrites; justification is the
        verifier's concern, not the scratchpad's."""
        value = int(n)
        value = max(0, min(int(self.range) - 1, value))
        self.bindings[str(v)] = value
        return value

    def substitute(self, i: int, j: int) -> Union[Equation, Unbound]:
        """Rewrite ``E_i`` by replacing the variable ``E_j`` defines (solved
        form ``v = expr``) with that expression.  Returns the rewritten
        equation, or ``Unbound`` naming nothing when ``E_j`` is not in
        solved form / indices are invalid / ``v`` does not occur."""
        target, source = self._equation(i), self._equation(j)
        if target is None or source is None or i == j:
            return Unbound(frozenset())
        v = source.solved_var
        if v is None or v not in target.vars:
            return Unbound(frozenset())

        def rewrite(e):
            if e[0] == "var":
                return source.rhs if e[1] == v else e
            if e[0] == "num":
                return e
            return (e[0], rewrite(e[1]), rewrite(e[2]))

        new = Equation(rewrite(target.lhs), rewrite(target.rhs))
        self.equations[int(i)] = new
        self.applied.update((int(i), int(j)))
        return new

    def constrain(self, i: int) -> Union[Bound, Unbound]:
        """Apply ``E_i`` under the bindings: when exactly one variable is
        unbound and the equation is linear with an integral in-range
        solution, return it as ``Bound``; otherwise ``Unbound`` with the
        unbound variables (empty when nothing is left to determine)."""
        eq = self._equation(i)
        if eq is None:
            return Unbound(frozenset())
        self.applied.add(int(i))
        left = linearize(eq.lhs, self.bindings)
        right = linearize(eq.rhs, self.bindings)
        if left is None or right is None:
            return Unbound(frozenset(v for v in eq.vars if v not in self.bindings))
        (lc, lk), (rc, rk) = left, right
        coeffs = dict(lc)
        for v, c in rc.items():
            coeffs[v] = coeffs.get(v, 0) - c
        coeffs = {v: c for v, c in coeffs.items() if c != 0}
        constant = lk - rk                      # sum c_v v + constant = 0
        if len(coeffs) != 1:
            return Unbound(frozenset(coeffs))
        (v, c), = coeffs.items()
        if (-constant) % c != 0:
            return Unbound(frozenset((v,)))
        value = (-constant) // c
        if not (0 <= value < int(self.range)):
            return Unbound(frozenset((v,)))
        return Bound(v, int(value))

    # dispatch -----------------------------------------------------------

    def execute(self, op: str, *operands, iteration: int = 0,
                references: Sequence[Any] = ()) -> dict:
        """Run one named primitive and append its trace step."""
        if op not in PRIMITIVES:
            raise ValueError(f"unknown exact primitive {op!r}")
        result = getattr(self, op)(*operands)
        self.executions += 1
        step = {"operation": f"exact:{op}", "operands": tuple(operands),
                "result": result, "iteration": int(iteration),
                "references": tuple(references)}
        self.trace.append(step)
        return step

    def _equation(self, i) -> Optional[Equation]:
        try:
            i = int(i)
        except (TypeError, ValueError):
            return None
        if 0 <= i < len(self.equations):
            return self.equations[i]
        return None


# -- numeral code (spec 6.3 fallback) -----------------------------------------

def numeral_code(n: int, width: int, bits: Optional[int] = None,
                 answer_range: Optional[int] = None):
    """Fixed, parameter-free code for integer ``n`` in a ``width``-wide
    slot.  With ``answer_range`` given and ``answer_range <= width`` the code is ONE-HOT
    (+1 at coordinate ``n``, -1 elsewhere over the first ``range``
    coordinates): a linear output adapter can then realize the one-hot
    answer directly.  Otherwise the binary digits of ``n`` as +-1 over the
    first ``bits`` coordinates (LSB first), zero elsewhere.  Distinct
    integers give distinct codes; the model's answer symbol root slot
    receives it when no lexicon row exists for the numeral surface."""
    import torch

    width = int(width)
    if (answer_range is not None and 0 < int(answer_range) <= width
            and 0 <= int(n) < int(answer_range)):
        code = torch.zeros(width)
        code[: int(answer_range)] = -1.0
        code[int(n)] = 1.0
        return code
    if bits is None:
        bits = max(1, int(n).bit_length()) if n >= 0 else 1
        bits = max(bits, min(width, 8))
    bits = min(int(bits), width)
    code = torch.zeros(width)
    for k in range(bits):
        code[k] = 1.0 if (int(n) >> k) & 1 else -1.0
    return code


def referent_code(name: str, width: int, bits: int = 16):
    """Fixed, parameter-free code for a variable referent used as the root
    slot of a QUERY(v) symbol (spec 6.3): +-1 digits of a stable hash of
    the name over the first ``bits`` coordinates and, when room remains, a
    +1 interrogative marker in the last coordinate.  Never collides with a
    numeral code of the same width in practice (numeral codes are zero
    beyond their digit band)."""
    import torch

    width = int(width)
    digest = hashlib.sha1(str(name).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    bits = min(int(bits), width)
    code = torch.zeros(width)
    for k in range(bits):
        code[k] = 1.0 if (value >> k) & 1 else -1.0
    if width > bits:
        code[-1] = 1.0
    return code


# -- problems (spec 4) --------------------------------------------------------

@dataclass(frozen=True)
class Problem:
    equations: Tuple[Equation, ...]          # premises + distractors, shuffled
    query: str
    answer: int
    depth: int
    structure: str
    range: int
    stage: int
    chain: FrozenSet[str]                    # variables on the query's chain
    order: Tuple[str, ...]                   # the solver's binding order
    solution: Dict[str, int] = field(default_factory=dict, compare=False)
    expression: Optional[Expr] = None        # stage 0: the bare expression

    def surface(self, words: bool = True) -> str:
        """The presented surface, in WORDS by default (``3 plus 4``,
        ``b equals a plus 4 ; what is b``): arithmetic as a syntax."""
        if self.stage == 0 and self.expression is not None:
            # Stage 0 (direct arithmetic): the presented input IS the
            # expression; the answer is its value.
            return render_expr(self.expression, words)
        tail = f" ; what is {self.query}" + ("" if words else " ?")
        return " ; ".join(e.render(words) for e in self.equations) + tail

    def index_of(self, v: str) -> Optional[int]:
        for i, e in enumerate(self.equations):
            if e.solved_var == v:
                return i
        return None


_NAMES = tuple("abcdefghjkmnpqrstuvwxyz")
# Operand sentinels of the stage-1 chain builder: a fresh constant or the
# zero constant (never a string -- a chain VARIABLE may be named "c").
_CONST = object()
_ZERO = object()


class MathProblemGenerator:
    """Stage-0 direct arithmetic, stage-1 dependency chains and stage-2
    simultaneous linear systems.

    All draws come from one ``random.Random(seed)`` so a seed is a dataset.
    Variable names and constants are fresh per problem; premise order is
    shuffled; distractors are consistent equations over off-chain variables.
    """

    def __init__(self, seed: int = 0, range: int = 64,
                 depths: Sequence[int] = (1, 2, 3),
                 distractors: Sequence[int] = (0, 1, 2), stage: int = 1,
                 operators: Sequence[str] = ("add",)):
        self.rng = random.Random(int(seed))
        self.range = int(range)
        self.depths = tuple(int(d) for d in depths)
        self.distractors = tuple(int(d) for d in distractors)
        self.stage = int(stage)
        self.operators = tuple(str(o) for o in operators) or ("add",)

    # -- stage 0: direct arithmetic ------------------------------------------

    def _stage0(self) -> Problem:
        """One stochastic binary operation ``a op b`` with its value in
        ``[0, R)``: the presented input is the expression, the answer its
        value, depth 0, no variables, no chain (the direct-arithmetic
        curriculum stage that precedes any substitution)."""
        R = self.range
        op = self.rng.choice(self.operators)
        # Sample the ANSWER uniformly over [0, R) first, then the operands
        # that produce it: sampling a then b < R - a skews sums toward
        # R - 1 (13.5 % of a corpus at R = 32), and a majority-answer head
        # then matches that plateau without learning arithmetic.
        if op == "succ":
            # The SUCCESSOR (Alec 2026-09-09): "n plus one" -> the next number
            # noun.  Addition is iterated succession performed by the
            # thinking loop, so this is the rung below "a plus b": one fixed
            # symbol-to-symbol map the verb must learn.
            a = self.rng.randrange(0, R - 1)
            b, value, op = 1, a + 1, "add"
        elif op == "sub":
            value = self.rng.randrange(0, R)
            b = self.rng.randrange(0, R - value)
            a = value + b
        elif op == "mul":
            a = self.rng.randrange(0, R)
            b = self.rng.randrange(0, (R - 1) // max(1, a) + 1) if a else self.rng.randrange(0, R)
            value = a * b
        else:
            op = "add"
            value = self.rng.randrange(0, R)
            a = self.rng.randrange(0, value + 1)
            b = value - a
        expr = (op, num(a), num(b))
        structure = hashlib.sha1(f"s0|{op}".encode()).hexdigest()[:12]
        return Problem((), "", int(value), 0, structure, R, 0, frozenset(), (),
                       {}, expression=expr)

    # -- stage 1 ------------------------------------------------------------

    def _fresh_names(self, k: int) -> List[str]:
        names = list(_NAMES)
        self.rng.shuffle(names)
        if k <= len(names):
            return names[:k]
        return names + [f"{n}{i}" for i in range(k - len(names)) for n in names[:1]]

    def _stage1(self, depth: int, n_distractors: int) -> Problem:
        R = self.range
        names = self._fresh_names(depth + 1 + n_distractors)
        chain = names[:depth + 1]
        values: Dict[str, int] = {}
        eqs: List[Equation] = []
        skeleton: List[str] = []
        values[chain[0]] = self.rng.randrange(0, max(1, R // 4))
        eqs.append(Equation(var(chain[0]), num(values[chain[0]])))
        skeleton.append("c")
        for k in range(1, depth + 1):
            prev = chain[k - 1]
            choices = []
            pv = values[prev]
            # add constant
            if pv + 1 < R:
                choices.append(("add", _CONST))
            if pv > 0:
                choices.append(("sub", _CONST))
            if pv > 0 and 2 * pv < R:
                choices.append(("mul", _CONST))
            # add an earlier chain variable (structure variety)
            if k >= 2:
                other = chain[self.rng.randrange(0, k - 1)]
                if pv + values[other] < R:
                    choices.append(("add", other))
            if not choices:
                choices.append(("add", _ZERO))
            op, operand = self.rng.choice(choices)
            if operand is _CONST:
                if op == "add":
                    c = self.rng.randrange(1, R - pv)
                    rhs, val = ("add", var(prev), num(c)), pv + c
                elif op == "sub":
                    c = self.rng.randrange(1, pv + 1)
                    rhs, val = ("sub", var(prev), num(c)), pv - c
                else:
                    c = self.rng.randrange(2, max(3, R // max(1, pv)))
                    if c * pv >= R:
                        c = 2
                    rhs, val = (("mul", num(c), var(prev)) if self.rng.random() < 0.5
                                else ("mul", var(prev), num(c))), c * pv
                skeleton.append(f"{op}c{k - 1}")
            elif operand is _ZERO:
                rhs, val = ("add", var(prev), num(0)), pv
                skeleton.append(f"addz{k - 1}")
            else:
                rhs, val = ("add", var(prev), var(operand)), pv + values[operand]
                skeleton.append(f"addv{k - 1},{chain.index(operand)}")
            values[chain[k]] = val
            eqs.append(Equation(var(chain[k]), rhs))
        order = tuple(chain)
        # distractors: solved-form equations over off-chain variables only
        dnames = names[depth + 1:]
        for d, name in enumerate(dnames):
            if d > 0 and self.rng.random() < 0.5:
                prev = dnames[d - 1]
                c = self.rng.randrange(0, max(1, R - values[prev]))
                rhs, val = ("add", var(prev), num(c)), values[prev] + c
            else:
                val = self.rng.randrange(0, R)
                rhs = num(val)
            values[name] = val
            eqs.append(Equation(var(name), rhs))
        query = chain[-1]
        self.rng.shuffle(eqs)
        structure = hashlib.sha1(
            f"s1|{depth}|{'|'.join(skeleton)}|d{len(dnames)}".encode()).hexdigest()[:12]
        return Problem(tuple(eqs), query, values[query], depth, structure, R, 1,
                       frozenset(chain), order, dict(values))

    # -- stage 2 ------------------------------------------------------------

    def _stage2(self, depth: int, n_distractors: int) -> Problem:
        """``depth`` unknowns on the query's chain (2 or 3): ``depth - 1``
        solved-form relations ``x_k = a_k * x_1 (+ c_k)`` and one linear
        combination of all of them; unique, triangular solution."""
        R = self.range
        k = max(2, min(3, int(depth)))
        names = self._fresh_names(k + n_distractors)
        chain = names[:k]
        values: Dict[str, int] = {}
        eqs: List[Equation] = []
        skeleton: List[str] = []
        x1 = self.rng.randrange(1, max(2, R // 8))
        values[chain[0]] = x1
        for j in range(1, k):
            a = self.rng.randrange(1, 4)
            c = self.rng.randrange(0, 3)
            val = a * x1 + c
            if val >= R:
                a, c, val = 1, 0, x1
            values[chain[j]] = val
            rhs = ("mul", num(a), var(chain[0])) if a > 1 else var(chain[0])
            if c:
                rhs = ("add", rhs, num(c))
            eqs.append(Equation(var(chain[j]), rhs))
            skeleton.append(f"rel{a}{'c' if c else ''}")
        # the combination: sum a_i x_i = total (all coefficients positive)
        coeffs = [self.rng.randrange(1, 3) for _ in chain]
        total = sum(a * values[v] for a, v in zip(coeffs, chain))
        while total >= R and any(a > 1 for a in coeffs):
            coeffs[coeffs.index(max(coeffs))] = 1
            total = sum(a * values[v] for a, v in zip(coeffs, chain))
        if total >= R:
            # shrink x1 and rebuild deterministically
            self.rng = random.Random(self.rng.random())
            return self._stage2(depth, n_distractors)
        lhs = None
        for a, v in zip(coeffs, chain):
            term = ("mul", num(a), var(v)) if a > 1 else var(v)
            lhs = term if lhs is None else ("add", lhs, term)
        eqs.append(Equation(lhs, num(total)))
        skeleton.append("comb" + "".join(str(a) for a in coeffs))
        query = self.rng.choice(chain)
        order = (chain[0],) + tuple(v for v in chain[1:])
        dnames = names[k:]
        for name in dnames:
            val = self.rng.randrange(0, R)
            values[name] = val
            eqs.append(Equation(var(name), num(val)))
        self.rng.shuffle(eqs)
        structure = hashlib.sha1(
            f"s2|{k}|{'|'.join(skeleton)}|q{chain.index(query)}|d{len(dnames)}".encode()
        ).hexdigest()[:12]
        return Problem(tuple(eqs), query, values[query], k, structure, R, 2,
                       frozenset(chain), order, dict(values))

    # -- public -------------------------------------------------------------

    def problem(self, depth: Optional[int] = None,
                n_distractors: Optional[int] = None) -> Problem:
        depth = int(depth) if depth is not None else self.rng.choice(self.depths)
        n_d = (int(n_distractors) if n_distractors is not None
               else self.rng.choice(self.distractors))
        if self.stage == 0:
            return self._stage0()
        if self.stage == 2:
            return self._stage2(depth, n_d)
        return self._stage1(depth, n_d)

    def problems(self, n: int, **kw) -> List[Problem]:
        return [self.problem(**kw) for _ in range(int(n))]

    @staticmethod
    def solve(problem: Problem) -> Tuple[Dict[str, int], List[dict]]:
        """The complete solver (generation / scoring only): replays the
        problem with the primitives in the solver's order and returns the
        bindings plus the trace it produced."""
        state = ExactState(list(problem.equations), range=problem.range)
        if problem.stage == 1:
            for v in problem.order:
                i = problem.index_of(v)
                res = state.execute("evaluate", i)["result"]
                if isinstance(res, int):
                    state.execute("bind", v, res)
        else:
            comb = next(i for i, e in enumerate(problem.equations)
                        if e.solved_var is None)
            for v in problem.order[1:]:
                state.execute("substitute", comb, problem.index_of(v))
            res = state.execute("constrain", comb)["result"]
            if isinstance(res, Bound):
                state.execute("bind", res.var, res.value)
            for v in problem.order[1:]:
                res = state.execute("constrain", problem.index_of(v))["result"]
                if isinstance(res, Bound):
                    state.execute("bind", res.var, res.value)
        return dict(state.bindings), list(state.trace)

    @staticmethod
    def brute_force(problem: Problem) -> List[Dict[str, int]]:
        """Every assignment in ``[0, R)^vars`` satisfying all equations
        (tests only; exponential)."""
        names = sorted(set().union(*(e.vars for e in problem.equations)))
        out = []
        for values in itertools.product(range(problem.range), repeat=len(names)):
            b = dict(zip(names, values))
            if all(eval_expr(e.lhs, b) == eval_expr(e.rhs, b)
                   for e in problem.equations):
                out.append(b)
        return out


def split_by_surface(problems: Sequence[Problem], *, val_share: int = 1,
                     test_share: int = 2, modulus: int = 10):
    """Partition by the hash of the presented surface (stage 0: every
    problem shares one structure, so the held-out set is a set of UNSEEN
    operand pairs; a pair seen in train never appears in test)."""
    out = {"train": [], "validation": [], "test": []}
    seen = {}
    for p in problems:
        key = p.surface()
        if key in seen:
            out[seen[key]].append(p)
            continue
        h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % int(modulus)
        name = ("train" if h < modulus - val_share - test_share
                else "validation" if h < modulus - test_share else "test")
        seen[key] = name
        out[name].append(p)
    return out


def split_by_structure(problems: Sequence[Problem], *, train_depths: Sequence[int],
                       val_share: int = 1, test_share: int = 2, modulus: int = 10):
    """Partition by ``structure`` hash: hash mod ``modulus`` < ``modulus -
    val_share - test_share`` -> train, next ``val_share`` -> validation, rest
    -> test.  Problems deeper than ``max(train_depths)`` always go to test."""
    out = {"train": [], "validation": [], "test": []}
    max_train = max(int(d) for d in train_depths)
    for p in problems:
        if p.depth > max_train:
            out["test"].append(p)
            continue
        h = int(p.structure, 16) % int(modulus)
        if h < modulus - val_share - test_share:
            out["train"].append(p)
        elif h < modulus - test_share:
            out["validation"].append(p)
        else:
            out["test"].append(p)
    return out


# -- verification and illumination (spec 3, 9) --------------------------------

@dataclass
class VerifierReport:
    accepted: int = 0
    rejected: int = 0
    valid: bool = False
    final: Optional[int] = None
    steps: List[dict] = field(default_factory=list)
    illumination: List[float] = field(default_factory=list)


def _same_result(a, b) -> bool:
    if isinstance(a, Equation) or isinstance(b, Equation):
        return a == b
    return a == b


def illumination(state: ExactState, problem: Problem, *,
                 max_free: int = 2) -> Optional[float]:
    """``I = 1 - log|C(query)| / log R`` where ``C`` is the candidate set of
    the query under the constraints the model has APPLIED: its bindings plus
    the premises in ``state.applied``.  A premise merely present in the
    problem constrains nothing until a primitive touches it.  Brute force
    over at most ``max_free`` unbound variables of the applied premises;
    ``None`` when infeasible."""
    import math

    R = int(problem.range)
    if problem.query in state.bindings:
        return 1.0
    applied = [e for i, e in enumerate(state.equations) if i in state.applied]
    names = sorted(set().union(*(e.vars for e in applied)) | {problem.query})
    free = [v for v in names if v not in state.bindings]
    if len(free) > max_free:
        return None
    candidates = set()
    for values in itertools.product(range(R), repeat=len(free)):
        b = dict(state.bindings)
        b.update(zip(free, values))
        if all(eval_expr(e.lhs, b) == eval_expr(e.rhs, b) for e in applied):
            candidates.add(b[problem.query])
    if not candidates:
        return None
    return 1.0 - (math.log(len(candidates)) / math.log(R) if R > 1 else 0.0)


class ExactVerifier:
    """Replay an emitted trace on a fresh scratchpad (spec 9).

    A step is accepted iff (a) it names a primitive with well-typed
    operands, (b) for ``bind``, the value was produced for that variable by
    an earlier accepted ``evaluate`` / ``constrain`` step (justification),
    and (c) its recorded result equals the replay.  Rejected steps are not
    applied.  The derivation is valid iff the emitted answer equals the
    replayed ``beta(query)``.
    """

    def __init__(self, *, measure_illumination: bool = False):
        self.measure_illumination = bool(measure_illumination)

    def check(self, trace: Sequence[dict], problem: Problem,
              answer: Optional[int] = None) -> VerifierReport:
        state = ExactState(list(problem.equations), range=problem.range)
        report = VerifierReport()
        justified: Dict[str, set] = {}
        if self.measure_illumination:
            report.illumination.append(illumination(state, problem) or 0.0)
        for step in trace:
            op = str(step.get("operation", ""))
            if not op.startswith("exact:"):
                continue
            name = op[len("exact:"):]
            operands = tuple(step.get("operands", ()))
            ok = name in PRIMITIVES
            replay = None
            if ok:
                try:
                    if name == "bind":
                        v, n = str(operands[0]), int(operands[1])
                        ok = n in justified.get(v, set())
                        replay = state.bind(v, n) if ok else None
                    else:
                        replay = getattr(state, name)(*operands)
                except (TypeError, ValueError, IndexError):
                    ok = False
            if ok and not _same_result(replay, step.get("result")):
                ok = False
            if ok and name == "evaluate":
                eq = state._equation(operands[0])
                if isinstance(replay, int) and eq is not None and eq.solved_var:
                    justified.setdefault(eq.solved_var, set()).add(replay)
            if ok and name == "constrain" and isinstance(replay, Bound):
                justified.setdefault(replay.var, set()).add(replay.value)
            if ok:
                report.accepted += 1
            else:
                report.rejected += 1
            report.steps.append({**step, "accepted": ok})
            if self.measure_illumination:
                report.illumination.append(illumination(state, problem) or
                                           report.illumination[-1])
        report.final = state.bindings.get(problem.query)
        emitted = answer if answer is not None else report.final
        report.valid = (report.final is not None and emitted == report.final
                        and report.final == problem.answer)
        return report
