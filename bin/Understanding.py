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

import torch


@dataclass(frozen=True)
class AnswerProgram:
    """One sentence's owned row references and identified compose program.

    Tensor copies retain the current forward's gradients while insulating
    the record from subsequent staging. ``targets`` describe reconstruction
    only; output generation never follows them. ``word_rows`` identify the
    presented WORDs separately from ``rows``, which identify their interpreted
    concepts (OBJECTs when associated).
    """

    rows: Any
    word_rows: Any
    activations: Any
    leaves: Any
    actions: Any
    targets: Any
    end_state: Any
    # Native allocator identities are addresses, never dictionary rows or
    # numerical semantic features. Older programs have unknown identities.
    concept_ids: Any = None
    # A lexical infix may name one grammar-spelled structural form even when
    # that form shares a canonical native VP with a converse.  This is frozen
    # grammar provenance (for example ``whole``), never a raw surface, row,
    # address, or numerical feature.  Older programs remain explicitly
    # unprovenanced rather than guessing a first form.
    lexical_forms: Any = None

    _tensor_fields = ("rows", "word_rows", "activations", "leaves", "actions", "targets", "end_state", "concept_ids")

    def __post_init__(self) -> None:
        ids = self.concept_ids
        if ids is None:
            ids = self.rows.new_full(self.rows.shape, -1)
        if (not torch.is_tensor(ids) or ids.dtype != torch.long
                or ids.shape != self.rows.shape
                or bool(((ids <= 0) & (ids != -1)).any())):
            raise ValueError("program concept IDs must be positive native addresses or -1, aligned to leaves")
        object.__setattr__(self, "concept_ids", ids)
        forms = self.lexical_forms
        if forms is None:
            forms = (None,) * int(self.rows.numel())
        else:
            try:
                forms = tuple(forms)
            except TypeError as error:
                raise TypeError(
                    "program lexical forms must be an aligned iterable") from error
            if len(forms) != int(self.rows.numel()):
                raise ValueError(
                    "program lexical forms must align to its retained leaves")
            if any(value is not None and not isinstance(value, str)
                   for value in forms):
                raise TypeError(
                    "program lexical forms must be grammar-form strings or None")
        object.__setattr__(self, "lexical_forms", forms)
        for name in self._tensor_fields:
            object.__setattr__(self, name, getattr(self, name).clone())

    def detached(self):
        """A durable recall record, without a previous brick's graph."""
        values = {name: getattr(self, name).detach().to("cpu")
                  for name in self._tensor_fields}
        values["lexical_forms"] = self.lexical_forms
        return type(self)(**values)


@dataclass(frozen=True)
class InputReconstruction:
    """Owned products of the completed input's single tied traversal.

    These are reconstructed values and scores, never input/answer targets.
    Clones preserve the current step's gradients and survive later staging.
    """

    ideas: Any
    event: Any
    idea_cost: Any
    byte_cost: Any
    truncated: Any
    sentence_costs: Any

    def __post_init__(self) -> None:
        for name in ("ideas", "event", "idea_cost", "byte_cost", "truncated", "sentence_costs"):
            value = getattr(self, name)
            object.__setattr__(self, name, value.clone() if value is not None else None)


@dataclass(frozen=True)
class Understanding:
    """Immutable logical products of one ``forward()`` call."""

    perceptual_context: Any = None
    conceptual_state: Any = None
    symbolic_state: Any = None
    reconstruction_carriers: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}))
    execution: Any = field(default=None, repr=False, compare=False)
    # Dense seed for topologies without indexed answer programs. The serial
    # path resolves its owned answer_program first and needs neither this
    # seed nor symbolic_state. None falls back to symbolic_state only in
    # the dense compatibility path.
    answer_seed: Any = field(default=None, repr=False, compare=False)
    answer_program: tuple = field(default_factory=tuple, repr=False, compare=False)
    sentence_programs: Mapping[int, tuple] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False)
    input_reconstruction: InputReconstruction | None = field(
        default=None, repr=False, compare=False)

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
