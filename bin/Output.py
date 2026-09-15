"""Answer derivation and construction values for ``Model.reverseOutput()``.

Question resolution produces an ``AnswerDerivation`` with owned conceptual
ideas, the selected row program, target-free question context, and named
synthesis references. Thinking transforms these ideas before realization
through the answer path. ``answer_symbol`` remains the dense compatibility
seed for topologies without indexed programs. Neither value carries a
desired ``Data`` answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class StepChoice:
    """One row's hard resolve-step choice (mathematical thinking spec 6.3).

    ``kind`` is ``"answer"`` (answer the active question) or ``"open"``
    (defer it and pose the subquestion ``operand``, a presented referent).  ``role`` says which question was
    active: ``"root"`` (the presented question), ``"pending"`` (the
    subquestion posed at the previous iteration) or ``"open"`` (the newest
    unanswered LTM input).  ``referent`` names it (``None`` for the root);
    ``question_rep`` is its QUERY representation. ``log_prob`` is the chooser's log
    probability of this choice (a tensor while the graph is live) for the
    policy objective; ``candidates`` are the labels it chose among.
    """

    kind: str
    row: int = 0
    role: str = "root"
    referent: Optional[str] = None
    operand: Any = None
    question_rep: Any = field(default=None, repr=False, compare=False)
    log_prob: Any = field(default=None, repr=False, compare=False)
    index: int = 0
    candidates: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.kind not in ("answer", "open"):
            raise ValueError(f"unknown step kind {self.kind!r}")
        object.__setattr__(self, "candidates", tuple(self.candidates))


@dataclass(frozen=True)
class AnswerDerivation:
    """A resolved conceptual answer plus its owned derivation record.

    ``conceptual_answer`` retains the full-width result of resolution and
    thinking. ``answer_symbol`` serves only topologies without row programs.
    ``step`` carries one :class:`StepChoice` per batch row when the
    thinking resolve step ran (``None`` per row otherwise); ``exact_steps``
    is retained for trace-shape compatibility and is always empty (the
    runtime carries no exact primitives).
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
    program: Tuple[Any, ...] = field(default_factory=tuple, repr=False, compare=False)
    conditioning_context: Any = field(default=None, repr=False, compare=False)
    conceptual_answer: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "grammar_trace", tuple(self.grammar_trace))
        object.__setattr__(self, "synthesis_references",
                           tuple(int(i) for i in self.synthesis_references))
        object.__setattr__(self, "bindings", dict(self.bindings or {}))
        object.__setattr__(self, "row_sources", tuple(self.row_sources))
        object.__setattr__(self, "step", tuple(self.step))
        object.__setattr__(self, "exact_steps", tuple(self.exact_steps))
        object.__setattr__(self, "program", tuple(self.program))
        if self.conditioning_context is not None:
            object.__setattr__(self, "conditioning_context",
                               self.conditioning_context.clone())
        if self.conceptual_answer is not None:
            object.__setattr__(self, "conceptual_answer", self.conceptual_answer.clone())


@dataclass(frozen=True)
class AnswerConstruction:
    """The realized answer and every stage it passed through."""

    actual: Any
    derivation: AnswerDerivation
    concepts: Any = None
    percepts: Any = None
    surface: Any = None          # input-event realization for text scoring
    trace: Tuple[Any, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace", tuple(self.trace))
