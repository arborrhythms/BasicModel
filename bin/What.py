"""Shared ``what()`` question, answer, and thinking contracts.

The classes in this module are deliberately data/model neutral.  ``Data``
uses them to retrieve a desired answer while ``BasicModel`` uses the same
question to produce an answer.  Keeping the contract here avoids giving
either side access to the other's authority (in particular, model code never
receives a dataset target through a question).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple


class WhatRelation(str, Enum):
    """The answer family requested by a :class:`WhatQuestion`."""

    PRESENT = "present"
    PAST = "past"
    FUTURE = "future"
    SUPERVISED = "supervised"
    INFERENCE = "inference"


class WhatSlotOperation(str, Enum):
    """Replayable grammatical operation represented by one LTM slot."""

    OPEN = "open"
    COMPLETE = "complete"
    CLOSE = "close"


@dataclass(frozen=True)
class WhatQuestion:
    """A model-visible question with a zero-based presentation coordinate.

    ``offset`` is zero for present/non-temporal questions, negative for past
    questions, and positive for future questions.  The desired answer is
    intentionally absent: supervised answers remain on ``Data`` and cannot
    leak into the model call graph through this object.
    """

    when: int
    relation: WhatRelation = WhatRelation.PRESENT
    offset: int = 0
    split: str = "train"
    prompt: Any = None
    where: Any = None

    def __post_init__(self) -> None:
        relation = (self.relation if isinstance(self.relation, WhatRelation)
                    else WhatRelation(str(self.relation).lower()))
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "when", int(self.when))
        object.__setattr__(self, "offset", int(self.offset))
        object.__setattr__(self, "split", str(self.split))
        if self.when < 0:
            raise ValueError("what question 'when' must be zero based and non-negative")
        if self.where is not None:
            raise ValueError("what question 'where' is reserved for a later iteration")
        if relation is WhatRelation.PAST and self.offset >= 0:
            raise ValueError("past questions require a negative offset")
        if relation is WhatRelation.FUTURE and self.offset <= 0:
            raise ValueError("future questions require a positive offset")
        if relation not in (WhatRelation.PAST, WhatRelation.FUTURE) and self.offset != 0:
            raise ValueError(f"{relation.value} questions require offset=0")

    @property
    def target_when(self) -> int:
        """Return the requested presentation index (which may be invalid)."""
        return self.when + self.offset

    @classmethod
    def present(cls, when: int, *, split: str = "train", prompt: Any = None):
        return cls(when=when, relation=WhatRelation.PRESENT,
                   split=split, prompt=prompt)

    @classmethod
    def past(cls, when: int, offset: int = -1, *, split: str = "train",
             prompt: Any = None):
        return cls(when=when, relation=WhatRelation.PAST, offset=offset,
                   split=split, prompt=prompt)

    @classmethod
    def future(cls, when: int, offset: int = 1, *, split: str = "train",
               prompt: Any = None):
        return cls(when=when, relation=WhatRelation.FUTURE, offset=offset,
                   split=split, prompt=prompt)

    @classmethod
    def supervised(cls, when: int, *, split: str = "train", prompt: Any = None):
        return cls(when=when, relation=WhatRelation.SUPERVISED,
                   split=split, prompt=prompt)

    @classmethod
    def inference(cls, when: int, *, split: str = "runtime", prompt: Any = None):
        return cls(when=when, relation=WhatRelation.INFERENCE,
                   split=split, prompt=prompt)

    def context_values(self) -> Tuple[float, ...]:
        """Return a stable, target-free encoding for grammar context.

        The five relation bits and signed relative offset describe the
        question.  Neither the answer nor a dataset-derived feature is
        present, which makes this safe to install before model execution.
        Absolute presentation coordinates remain represented separately by
        the model's established ``.when`` machinery.
        """
        relations = tuple(
            1.0 if self.relation is candidate else 0.0
            for candidate in WhatRelation
        )
        signed_offset = float(self.offset)
        magnitude = abs(signed_offset)
        return relations + (signed_offset, magnitude)


# The design and specification use the concise ``What(present)`` spelling.
What = WhatQuestion


@dataclass(frozen=True)
class WhatAnswer:
    """The desired or produced ``what`` for one question."""

    question: WhatQuestion
    what: Any = None
    available: bool = True
    provenance: str = "model"
    source_when: Optional[int] = None
    reason: Optional[str] = None
    grammar_trace: Tuple[Any, ...] = field(default_factory=tuple)
    ltm_slot: Optional["LTMSlot"] = field(default=None, compare=False)
    execution: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.available and self.what is None:
            raise ValueError("an available what answer must contain a response")
        if not self.available and self.what is not None:
            raise ValueError("an unavailable what answer cannot contain a response")
        object.__setattr__(self, "provenance", str(self.provenance))
        object.__setattr__(self, "grammar_trace", tuple(self.grammar_trace))

    @classmethod
    def unavailable(cls, question: WhatQuestion, reason: str,
                    *, provenance: str = "data"):
        return cls(question=question, what=None, available=False,
                   provenance=provenance, reason=str(reason))


@dataclass(frozen=True)
class DataPresentation:
    """Stable input/output reservation at one dataset presentation index."""

    question: WhatQuestion
    input: Any
    output: Optional[WhatAnswer] = None

    @property
    def when(self) -> int:
        return self.question.when


@dataclass(frozen=True)
class LTMSlot:
    """One chronological LTM interaction with independently optional sides."""

    input: Any = None
    output: Any = None
    question: Optional[WhatQuestion] = None
    iteration: int = 0
    closure_pressure: float = 0.0
    forced: bool = False
    grammar_trace: Tuple[Any, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.input is None and self.output is None:
            raise ValueError("an LTM interaction slot cannot be empty")
        if int(self.iteration) < 0:
            raise ValueError("LTM slot iteration cannot be negative")
        pressure = float(self.closure_pressure)
        if pressure < 0.0:
            raise ValueError("LTM slot closure pressure cannot be negative")
        object.__setattr__(self, "iteration", int(self.iteration))
        object.__setattr__(self, "closure_pressure", pressure)
        object.__setattr__(self, "grammar_trace", tuple(self.grammar_trace))

    @property
    def operation(self) -> WhatSlotOperation:
        if self.input is not None and self.output is None:
            return WhatSlotOperation.OPEN
        if self.input is not None and self.output is not None:
            return WhatSlotOperation.COMPLETE
        return WhatSlotOperation.CLOSE


@dataclass(frozen=True)
class WhatThinkingResult:
    """Completed iterative-thinking result returned after LTM parity."""

    answer: WhatAnswer
    slots: Tuple[LTMSlot, ...]
    iterations: int
    forced_closures: int = 0
    closure_pressures: Tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", tuple(self.slots))
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "forced_closures", int(self.forced_closures))
        object.__setattr__(self, "closure_pressures",
                           tuple(float(p) for p in self.closure_pressures))


def question_batch(questions: Any) -> Tuple[WhatQuestion, ...]:
    """Normalize one question or an iterable without accepting raw targets."""
    if isinstance(questions, WhatQuestion):
        return (questions,)
    if questions is None:
        return ()
    normalized = tuple(questions)
    if not all(isinstance(question, WhatQuestion) for question in normalized):
        raise TypeError("questions must contain only WhatQuestion values")
    return normalized
