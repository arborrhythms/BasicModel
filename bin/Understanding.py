"""The per-call ``Understanding`` produced by one bottom-up ``forward()``.

What spec Step 1 (doc/specs/2026-07-27-teaching-modes-and-next-iteration.md
section 5.1): one analysis pass yields one immutable value holding the live
perceptual context, the terminal conceptual state, the symbolic state, and
the routing/binding handles reconstruction needs to invert the analysis.
Both downward paths consume it -- ``Model.reverseReconstruct()`` (input-associated
inverse) and ``Model.reverseOutput()`` (answer synthesis) -- without either path
overwriting the other's carriers.

It contains no desired answer and no pre-analysis surface payload: the
forward input is a scoring target held by the caller, never a carrier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class AnswerProgram:
    """One sentence's owned row references and identified compose program.

    Tensor copies retain the current forward's gradients while insulating
    the record from subsequent staging. ``targets`` describe reconstruction
    only; output generation never follows them.
    """

    rows: Any
    activations: Any
    leaves: Any
    actions: Any
    targets: Any
    end_state: Any

    _tensor_fields = ("rows", "activations", "leaves", "actions", "targets", "end_state")

    def __post_init__(self) -> None:
        for name in self._tensor_fields:
            object.__setattr__(self, name, getattr(self, name).clone())

    def detached(self):
        """A durable recall record, without a previous brick's graph."""
        return type(self)(**{name: getattr(self, name).detach().to("cpu")
                             for name in self._tensor_fields})


@dataclass(frozen=True)
class Understanding:
    """Immutable logical products of one ``forward()`` call."""

    perceptual_context: Any = None
    conceptual_state: Any = None
    symbolic_state: Any = None
    reconstruction_carriers: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}))
    execution: Any = field(default=None, repr=False, compare=False)
    # The resolved INPUT symbol the answer path is seeded from (spec 5.3).
    # On the serial grammar path this is the grammar's root idea (the
    # STM-folded S, which varies with the sentence); the ``symbols``
    # tensor there is the symbol-space activation over a codebook that is
    # nearly empty at initialization and so is the same for every
    # sentence.  ``None`` means "use ``symbolic_state``".
    answer_seed: Any = field(default=None, repr=False, compare=False)
    answer_program: tuple = field(default_factory=tuple, repr=False, compare=False)
    sentence_programs: Mapping[int, tuple] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer_program", tuple(self.answer_program))
        object.__setattr__(self, "sentence_programs", MappingProxyType({
            int(slot): tuple(rows) for slot, rows in self.sentence_programs.items()}))
        carriers = self.reconstruction_carriers
        if not isinstance(carriers, MappingProxyType):
            object.__setattr__(
                self, "reconstruction_carriers",
                MappingProxyType(dict(carriers or {})))
        for forbidden in ("desired", "answer", "target", "surface"):
            if forbidden in self.reconstruction_carriers:
                raise ValueError(
                    f"reconstruction carriers may not hold {forbidden!r}")

    @property
    def has_conceptual_state(self) -> bool:
        return self.conceptual_state is not None
