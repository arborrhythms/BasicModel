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
class AnswerDerivation:
    """A resolved answer symbol plus the replayable derivation record."""

    answer_symbol: Any
    grammar_trace: Tuple[Any, ...] = field(default_factory=tuple)
    bindings: Mapping[str, Any] = field(default_factory=dict)
    synthesis_references: Tuple[int, ...] = field(default_factory=tuple)
    sentence_location: Optional[int] = None
    prefix: Any = None
    resolved: bool = True
    source: str = "identity"
    row_sources: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "grammar_trace", tuple(self.grammar_trace))
        object.__setattr__(self, "synthesis_references",
                           tuple(int(i) for i in self.synthesis_references))
        object.__setattr__(self, "bindings", dict(self.bindings or {}))
        object.__setattr__(self, "row_sources", tuple(self.row_sources))


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
