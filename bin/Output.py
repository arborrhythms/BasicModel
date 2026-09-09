"""Answer derivation and construction values for ``Model.reverseOutput()``.

What spec section 5.3: an answer is not a replay of the input surface.  A
question symbol is resolved (grammatical evaluation, lookup, binding, or
thinking) into an ``AnswerDerivation`` carrying the answer symbol, the
grammar trace, and the named synthesis references; only that answer symbol
descends through conceptual and perceptual synthesis to ``OutputSpace``.
Neither value carries a desired ``Data`` answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class StepChoice:
    """One row's hard resolve-step choice (mathematical thinking spec 6.3).

    ``kind`` is ``"answer"`` (answer the active question), ``"open"`` (defer
    it and pose the subquestion ``operand``), or -- only transiently inside
    the step loop -- ``"execute"``.  ``role`` says which question was
    active: ``"root"`` (the presented question), ``"pending"`` (the
    subquestion posed at the previous iteration) or ``"open"`` (the newest
    unanswered LTM input).  ``referent`` names it (``None`` for the root);
    ``question_rep`` is its QUERY symbol; ``value`` is the exact integer
    answered when one was bound.  ``log_prob`` is the chooser's log
    probability of this choice (a tensor while the graph is live) for the
    policy objective; ``candidates`` are the labels it chose among.
    """

    kind: str
    row: int = 0
    role: str = "root"
    referent: Optional[str] = None
    operand: Any = None
    value: Optional[int] = None
    question_rep: Any = field(default=None, repr=False, compare=False)
    log_prob: Any = field(default=None, repr=False, compare=False)
    index: int = 0
    candidates: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.kind not in ("answer", "open", "execute"):
            raise ValueError(f"unknown step kind {self.kind!r}")
        object.__setattr__(self, "candidates", tuple(self.candidates))


@dataclass(frozen=True)
class AnswerDerivation:
    """A resolved answer symbol plus the replayable derivation record.

    ``step`` carries one :class:`StepChoice` per batch row when the
    thinking resolve step ran (``None`` per row otherwise); ``exact_steps``
    are the spec 5.3 primitive execution records of this iteration.
    """

    answer_symbol: Any
    grammar_trace: Tuple[Any, ...] = field(default_factory=tuple)
    bindings: Mapping[str, Any] = field(default_factory=dict)
    synthesis_references: Tuple[int, ...] = field(default_factory=tuple)
    sentence_location: Optional[int] = None
    prefix: Any = None
    resolved: bool = True
    source: str = "identity"
    row_sources: Tuple[str, ...] = field(default_factory=tuple)
    step: Tuple[Optional[StepChoice], ...] = field(default_factory=tuple)
    exact_steps: Tuple[Any, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "grammar_trace", tuple(self.grammar_trace))
        object.__setattr__(self, "synthesis_references",
                           tuple(int(i) for i in self.synthesis_references))
        object.__setattr__(self, "bindings", dict(self.bindings or {}))
        object.__setattr__(self, "row_sources", tuple(self.row_sources))
        object.__setattr__(self, "step", tuple(self.step))
        object.__setattr__(self, "exact_steps", tuple(self.exact_steps))


@dataclass(frozen=True)
class AnswerConstruction:
    """The realized answer and every stage it passed through."""

    actual: Any
    derivation: AnswerDerivation
    concepts: Any = None
    percepts: Any = None
    surface: Any = None          # input-space realization (temporal answers)
    trace: Tuple[Any, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace", tuple(self.trace))
