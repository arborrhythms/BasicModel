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

import torch

from Meaning import ConceptualMeaning


def thought_answer_meanings(selected):
    """Adapt owned checked results without a reader or another execution.

    A set remains an ordered tuple of complete members. Codes are unary
    concepts; subgoals carry the child's typed result. Only the live selected
    truth meaning retains a structural gradient. Checked reader payloads and
    restored results are detached, including all members of a set.
    """
    from Queries import ThoughtResult
    from Layers import MeaningExpectation
    checked = getattr(selected, "result", None)
    meaning = getattr(selected, "meaning", None)
    seen = set()
    while isinstance(checked, ThoughtResult) and checked.result_kind == "subgoal":
        if id(checked) in seen or len(seen) >= 64:
            raise ValueError("cyclic or over-deep thought result")
        seen.add(id(checked))
        checked = checked.value
        meaning = checked.request if isinstance(checked, ThoughtResult) else None
    if not isinstance(checked, ThoughtResult):
        return ()
    request = checked.request
    width = int(request.roles.shape[-1])
    if checked.result_kind == "truth":
        if not isinstance(meaning, ConceptualMeaning):
            raise TypeError("truth answer requires its complete selected meaning")
        return (meaning,)
    if checked.result_kind == "prediction":
        value = checked.value
        if value is None:
            return ()
        if not isinstance(value, MeaningExpectation):
            raise TypeError("prediction answer requires a MeaningExpectation")
        if (value.roles.shape != request.roles.shape
                or value.presence_logits.shape != (3,)
                or not bool(torch.isfinite(value.presence_logits).all())):
            raise ValueError("prediction answer differs from the full role shape")
        return (ConceptualMeaning(value.roles.detach(),
                torch.ones(3, dtype=torch.bool, device=value.roles.device),
                mode="unspecified"),)
    if checked.result_kind in ("code", "concept"):
        value = checked.value
        if value is None:
            return ()
        if not torch.is_tensor(value) or value.shape != (width,):
            raise ValueError("code answer requires one full-width conceptual value")
        roles = torch.stack((value.detach(), torch.zeros_like(value),
                             torch.zeros_like(value)))
        return (ConceptualMeaning(
            roles, torch.tensor([True, False, False], device=value.device),
            mode="assertive", role_refs=(checked.evidence.get("reference"), None, None),
            bindings=request.bindings, scope=request.scope),)
    if checked.result_kind != "set":
        raise ValueError(f"unsupported thought answer kind {checked.result_kind!r}")
    answers = []
    for member in checked.value or ():
        stored = member.get("meaning")
        if isinstance(stored, ConceptualMeaning):
            if stored.roles.shape != request.roles.shape:
                raise ValueError("set member differs from the conceptual width")
            answers.append(stored.detached())
            continue
        value, reference = member.get("value"), member.get("reference")
        if not torch.is_tensor(value) or value.shape != (width,) or reference is None:
            raise ValueError("set answer member requires its owned payload and reference")
        opened = [slot for slot in (0, 2) if not bool(request.role_mask[slot])]
        if len(opened) != 1:
            raise ValueError("reference set answer requires one open operand role")
        slot = opened[0]
        roles, mask, refs = request.roles.clone(), request.role_mask.clone(), list(request.role_refs)
        roles[slot], mask[slot], refs[slot] = value.detach(), True, reference
        answers.append(ConceptualMeaning(roles, mask, mode="assertive",
            polarity=request.polarity, role_refs=tuple(refs),
            bindings=request.bindings, scope=request.scope).detached())
    return tuple(answers)


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
    # Presentation metadata is owned before realization. It carries no target
    # and does not cause another query or another conditioning pass.
    questions: Tuple[Any, ...] = field(default_factory=tuple, repr=False, compare=False)
    # Completed selected grammatical questions are ordinary-thought episodes,
    # not a second answer representation.  The live records remain owned by
    # WhatInteractionMemory; this tuple links a realized response to the
    # actual boundary execution that supplied its evidence.
    selected_thoughts: Tuple[Any, ...] = field(
        default_factory=tuple, repr=False, compare=False)
    # Per-row typed answer members, including every member of a checked set.
    # These are views of the prepared answer, not another memory owner.
    answer_meanings: Tuple[Tuple[ConceptualMeaning, ...], ...] = field(
        default_factory=tuple, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "grammar_trace", tuple(self.grammar_trace))
        object.__setattr__(self, "synthesis_references",
                           tuple(int(i) for i in self.synthesis_references))
        object.__setattr__(self, "bindings", dict(self.bindings or {}))
        object.__setattr__(self, "row_sources", tuple(self.row_sources))
        object.__setattr__(self, "step", tuple(self.step))
        object.__setattr__(self, "exact_steps", tuple(self.exact_steps))
        object.__setattr__(self, "program", tuple(self.program))
        object.__setattr__(self, "questions", tuple(self.questions))
        object.__setattr__(self, "selected_thoughts", tuple(self.selected_thoughts))
        object.__setattr__(self, "answer_meanings", tuple(
            tuple(row) for row in self.answer_meanings))
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
    texts: Tuple[Optional[str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace", tuple(self.trace))
        object.__setattr__(self, "texts", tuple(self.texts))
