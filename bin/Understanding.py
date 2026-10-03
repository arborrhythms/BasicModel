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

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class AnswerProgram:
    """Temporary operations and values of one open sentence reading.

    Tensor copies retain the current forward's gradients while insulating
    the reading from subsequent staging. ``targets`` describe reconstruction
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
    # Word identity survives unary row replacement and synonymy during reading.
    word_ids: Any = None
    # A lexical infix may name one grammar-spelled structural form even when
    # that form shares a canonical native VP with a converse.  This is frozen
    # grammar provenance (for example ``whole``), never a raw surface, row,
    # address, or numerical feature.  Older programs remain explicitly
    # unprovenanced rather than guessing a first form.
    lexical_forms: Any = None
    # Grammar-selected identities are semantic addresses. Original ids and
    # leaves retain the tied reconstruction's presented-word provenance.
    reference_ids: Any = None
    reference_orders: Any = None
    symbol_where: Any = None
    symbol_when: Any = None
    reference_values: Any = None
    reference_relations: Any = None
    # Actual left/right operands and result, only while this reading is open.
    operation_values: Any = None
    operation_refs: Any = None
    operation_relations: Any = None
    leaf_orders: Any = None
    leaf_evidence: Any = None
    @property
    def _tensor_fields(self):
        core = ("rows", "word_rows", "activations", "leaves", "actions",
                "targets", "end_state", "concept_ids")
        return core + tuple(name for name in ("word_ids", "reference_ids", "reference_orders", "symbol_where", "symbol_when",
                                             "reference_values", "reference_relations", "operation_values", "operation_refs", "operation_relations", "leaf_orders", "leaf_evidence")
                            if getattr(self, name) is not None)

    def __post_init__(self) -> None:
        ids = self.concept_ids
        if ids is None:
            ids = self.rows.new_full(self.rows.shape, -1)
        if (not torch.is_tensor(ids) or ids.dtype != torch.long
                or ids.shape != self.rows.shape
                or bool(((ids <= 0) & (ids != -1)).any())):
            raise ValueError("program concept IDs must be positive native addresses or -1, aligned to leaves")
        object.__setattr__(self, "concept_ids", ids)
        words = self.word_ids
        if words is not None and (not torch.is_tensor(words) or words.dtype != torch.long
                                  or words.shape != ids.shape or bool(((words <= 0) & (words != -1)).any())):
            raise ValueError('program word IDs must be positive identities or -1, aligned to leaves')
        # Unresolved programs retain no second identity vector. Structural
        # edits can then change their leaves without carrying stale aliases.
        refs, orders = self.reference_ids, self.reference_orders
        if refs is not None and (not torch.is_tensor(refs) or refs.dtype != torch.long
                or refs.shape != ids.shape or bool(((refs <= 0) & (refs != -1)).any())):
            raise ValueError("program reference IDs must be native addresses or -1, aligned to leaves")
        if orders is not None and (not torch.is_tensor(orders) or orders.dtype != torch.long
                or orders.shape != ids.shape or bool((orders < -1).any())):
            raise ValueError("program reference orders must be nonnegative or -1, aligned to leaves")
        for name in ('symbol_where', 'symbol_when'):
            value = getattr(self, name)
            if value is not None and (not torch.is_tensor(value) or value.shape != (*ids.shape, 4)
                                      or not value.is_floating_point()):
                raise ValueError('symbol occurrence bands must align to program leaves')
        if self.reference_values is not None and self.reference_values.shape != self.leaves.shape:
            raise ValueError('semantic reference values must align with word leaves')
        if self.reference_relations is not None and (self.reference_relations.shape != ids.shape
                or self.reference_relations.dtype != torch.bool):
            raise ValueError('relation reference flags must align with native identities')
        if self.operation_values is not None and self.operation_values.shape != (
                self.actions.shape[0], 3, self.leaves.shape[-1]):
            raise ValueError('operation values must hold two operands and a result per action')
        if self.operation_refs is not None and (self.operation_refs.shape != (self.actions.shape[0], 2)
                or self.operation_refs.dtype != torch.long):
            raise ValueError('operation references must hold two operand addresses per action')
        if self.operation_relations is not None and (self.operation_relations.shape != (self.actions.shape[0],2)
                or self.operation_relations.dtype != torch.bool):
            raise ValueError('operation reference kinds must align to operand addresses')
        if self.leaf_orders is not None and (self.leaf_orders.shape != ids.shape
                or self.leaf_orders.dtype != torch.long or bool((self.leaf_orders < -1).any())):
            raise ValueError('leaf orders must align with the open reading')
        if self.leaf_evidence is not None and (self.leaf_evidence.shape != (*ids.shape, 2)
                or not bool(torch.isfinite(self.leaf_evidence).all())
                or bool(((self.leaf_evidence < 0) | (self.leaf_evidence > 1)).any())):
            raise ValueError('leaf evidence requires two finite poles in [0, 1]')
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
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else value.clone())

    def detached(self):
        """The open trial's values after its graph has trained and been freed."""
        values = {name: (None if getattr(self, name) is None else
                         getattr(self, name).detach().to("cpu"))
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
class SentenceEndState:
    """A completed field, independent of how its grammar produced it.

    An idea has one occupied slot; a relation has three. The two operand
    addresses cached by an idea do not restore its pre-fusion target.
    A selected question is a typed semantic request, never a compose program.
    """

    meaning: Any
    refs: tuple = (-1, -1, -1)
    row_id: int = -1
    where: Any = None
    when: Any = None
    query: Any = None
    order: int = 0
    evidence: tuple = (0., 0.)
    trust: float = 0.

    def __post_init__(self):
        from Meaning import ConceptualMeaning
        import math
        if type(self.order) is not int or self.order < -1:
            raise ValueError('completed field order must be nonnegative or unknown (-1)')
        if len(self.evidence) != 2 or any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in self.evidence):
            raise ValueError('completed evidence requires two finite poles in [0, 1]')
        if not math.isfinite(float(self.trust)) or not -1 <= float(self.trust) <= 1:
            raise ValueError('completed source trust must be a finite scalar in [-1, 1]')
        if not isinstance(self.meaning, ConceptualMeaning):
            raise TypeError('a sentence end state requires a conceptual field')
        if int(self.meaning.role_mask.sum()) not in (1, 3):
            raise ValueError('a sentence end state occupies one or three slots')
        if len(self.refs) != 3 or any(type(ref) is not int or ref == 0 or ref < -1 for ref in self.refs):
            raise ValueError('completed references must be native addresses or -1')
        if self.query is not None and (not isinstance(self.query, ConceptualMeaning)
                                      or self.query.mode != 'interrogative'):
            raise ValueError('a completed question must be an interrogative meaning')
        object.__setattr__(self, 'meaning', replace(self.meaning))
        if self.query is not None:
            object.__setattr__(self, 'query', replace(self.query))
        for name in ('where', 'when'):
            value = getattr(self, name)
            if value is not None:
                if not torch.is_tensor(value) or value.shape != (4,):
                    raise ValueError('a completed field band has four coordinates')
                object.__setattr__(self, name, value.clone())

    @property
    def end_state(self):
        return self.meaning.roles

    def detached(self):
        return type(self)(self.meaning.detached(), self.refs, self.row_id,
            None if self.where is None else self.where.detach(),
            None if self.when is None else self.when.detach(),
            None if self.query is None else self.query.detached(), self.order, self.evidence, self.trust)


@dataclass(frozen=True)
class ConceptualField:
    """One field's reading and native attribution, addressed by concept id.

    The where/when belong to the field. Percept events own their coordinates;
    no concept-by-position evidence or input byte trace is stored.
    """

    concept_ids: Any
    evidence: Any
    where: Any
    when: Any
    percepts: Any

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, getattr(self, name).clone())


@dataclass(frozen=True)
class Understanding:
    """Immutable logical products of one ``forward()`` call."""

    perceptual_context: Any = None
    conceptual_state: Any = None
    symbolic_state: Any = None
    reconstruction_carriers: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}))
    execution: Any = field(default=None, repr=False, compare=False)
    # The serial path owns completed fields. Other topologies retain their
    # dense answer seed.
    answer_seed: Any = field(default=None, repr=False, compare=False)
    sentence_states: tuple = field(default_factory=tuple, repr=False, compare=False)
    sentence_records: tuple = field(default_factory=tuple, repr=False, compare=False)
    sentence_fields: Mapping[int, tuple] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False)
    input_reconstruction: InputReconstruction | None = field(
        default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sentence_states", tuple(self.sentence_states))
        object.__setattr__(self, "sentence_records", tuple(self.sentence_records))
        object.__setattr__(self, "sentence_fields", MappingProxyType({
            int(slot): tuple(rows) for slot, rows in self.sentence_fields.items()}))
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
