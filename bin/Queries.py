"""Checked boundary-query contracts; no semantic memory or learned parameters.

Relation identities link grammatical compose/inverse faces and query interfaces.
These immutable definitions do not create a second VP embedding or a tool-only
semantic store. Native references are addresses; numerical payloads stay in CS.
"""
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Callable
import re

import torch

from Meaning import ConceptualMeaning
from QueryWork import QueryWorkBudget, QueryWorkExhausted, capture_limits
from Taxonomy import capture_taxonomy, concept_reference


def _snapshot_grammar_context_value(value):
    """Freeze owner-supplied stream/priming data before any face sees it."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, ConceptualMeaning):
        return value.detached()
    if isinstance(value, (dict, MappingProxyType)):
        return MappingProxyType({
            key: _snapshot_grammar_context_value(item)
            for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_snapshot_grammar_context_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_snapshot_grammar_context_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ConceptualSpaceCapability:
    """Read-only geometry contract for a structural grammar face.

    A structural operator already owns its parameterized tensor kernel.  It
    therefore needs only to know the full conceptual width of values it may
    receive, not the mutable ``ConceptualSpace`` / SymbolSpace graph that
    owns allocation, taxonomy, LTM, or a controller.  Keeping this as data
    rather than a back-reference makes the grammar context a real capability
    boundary instead of a conveniently named model escape hatch.
    """
    width: int

    def __post_init__(self):
        if type(self.width) is not int or self.width <= 0:
            raise ValueError('conceptual-space capability requires a positive width')

    def accepts(self, value):
        """Whether ``value`` is one or more full-width concept vectors."""
        return (torch.is_tensor(value) and value.ndim >= 1
                and int(value.shape[-1]) == self.width)


@dataclass(frozen=True)
class GrammarContext:
    """The immutable, owner-selected context common to every grammar face."""
    word_stream: object
    conceptual_space: object
    primed_symbols: object

    def __post_init__(self):
        if self.conceptual_space is None:
            raise ValueError('grammar context requires a conceptual-space capability')
        # These are owner-selected observations, not another route into the
        # structural graph or live taxonomy state.  Operators receive one
        # immutable snapshot regardless of whether a caller used the standard
        # SymbolSubSpace builder or constructed a context in a test/tool.
        object.__setattr__(
            self, 'word_stream', _snapshot_grammar_context_value(self.word_stream))
        object.__setattr__(
            self, 'primed_symbols', _snapshot_grammar_context_value(
                self.primed_symbols))


@dataclass(frozen=True)
class StructuralGrammarContext(GrammarContext):
    """Pure compose/generate context; it has no memory or controller access."""
    phase: str

    def __post_init__(self):
        super().__post_init__()
        if self.phase not in ('compose', 'generate'):
            raise ValueError('structural grammar context phase must be compose or generate')


@dataclass(frozen=True)
class ThoughtGrammarContext(GrammarContext):
    """Capability-scoped context for one checked completed-row thought call.

    There is intentionally no ``reasoner`` or model attribute.  The caller
    supplies narrowly shaped LTM/taxonomy views, a boundary permit and the
    one non-renewable work meter; a descriptor may use only its declared
    methods through those views.
    """
    ltm: object
    taxonomy: object
    work: QueryWorkBudget
    continuation: Callable | None
    boundary: Callable
    row: int = 0
    max_nodes: int = 256
    max_records: int = 1024
    max_steps: int = 8
    max_expansions: int = 1024

    def __post_init__(self):
        super().__post_init__()
        if self.ltm is None or self.taxonomy is None:
            raise ValueError('thought grammar context requires LTM and taxonomy capabilities')
        if not isinstance(self.work, QueryWorkBudget):
            raise TypeError('thought grammar context requires one QueryWorkBudget')
        if not callable(self.boundary):
            raise TypeError('thought grammar context requires a boundary permit')
        for name in ('row', 'max_nodes', 'max_records', 'max_steps', 'max_expansions'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'thought grammar context {name} must be a non-negative integer')

    def require_boundary(self):
        """Consume no semantic state; prove the owner opened this row's boundary."""
        self.boundary(self.row)


class _ThoughtCapabilityView:
    """A deliberately small callable surface over one owner-provided reader.

    The context passed to a descriptor is a normal Python object, not a
    security sandbox.  This facade is nevertheless important architectural
    ownership: its public attributes are exactly the named bound readers a
    descriptor declared.  In particular it never keeps a public ``owner``,
    ``model`` or ``reasoner`` escape hatch, and an undeclared reader fails as
    an ordinary missing attribute.
    """

    __slots__ = ('__methods',)

    def __init__(self, source, method_names):
        methods = {}
        for name in method_names:
            method = getattr(source, name, None)
            if callable(method):
                methods[name] = method
        object.__setattr__(self, '_ThoughtCapabilityView__methods',
                           MappingProxyType(methods))

    def __getattr__(self, name):
        methods = object.__getattribute__(
            self, '_ThoughtCapabilityView__methods')
        try:
            return methods[name]
        except KeyError as error:
            raise AttributeError(
                f'thought capability does not admit {name!r}') from error

    def __dir__(self):
        methods = object.__getattribute__(
            self, '_ThoughtCapabilityView__methods')
        return sorted(methods)


@dataclass(frozen=True)
class ThoughtResult:
    """Typed evidence produced by one checked thought operator."""
    semantic_id: str
    domain: str
    result_kind: str
    evidence_kind: str
    request: ConceptualMeaning
    evidence: object

    def __post_init__(self):
        if not isinstance(self.request, ConceptualMeaning):
            raise TypeError('thought result requires its complete request meaning')
        if not isinstance(self.evidence, MappingProxyType):
            raise TypeError('thought result evidence must be immutable')

    @property
    def value(self):
        return self.evidence.get('value')

    @property
    def support_true(self):
        return self.evidence.get('support_true', 0.0)

    @property
    def support_false(self):
        return self.evidence.get('support_false', 0.0)

    @property
    def incomplete(self):
        return self.evidence.get('incomplete', ())


@dataclass(frozen=True)
class ThoughtOperationCandidate:
    """One side-effect-free boundary action proposed by the grammar catalog.

    ``semantic_id`` and the structural open-role labels are dispatch metadata,
    never numerical policy features.  The chooser receives ``request``'s
    complete conceptual meaning instead.  Candidate formation may read the
    installed frozen VP payload, but it neither opens a reader nor consumes
    the shared episode meter; the selected action alone does that.
    """
    operation: object
    request: ConceptualMeaning
    open_roles: tuple = ()

    def __post_init__(self):
        semantic_id = getattr(self.operation, 'semantic_id', None)
        roles = tuple(getattr(self.operation, 'operand_roles', ()))
        if not isinstance(semantic_id, str) or not roles:
            raise ValueError('thought candidate requires a structural operation')
        if not isinstance(self.request, ConceptualMeaning):
            raise TypeError('thought candidate requires a complete request')
        if self.request.mode != 'interrogative':
            raise ValueError('thought candidate request must be interrogative')
        open_roles = tuple(self.open_roles)
        if (len(set(open_roles)) != len(open_roles)
                or not set(open_roles).issubset(roles)):
            raise ValueError('thought candidate has invalid open roles')
        object.__setattr__(self, 'open_roles', open_roles)

    @property
    def semantic_id(self):
        return self.operation.semantic_id


@dataclass(frozen=True)
class QueryContext:
    """One selected call's existing readers and controller continuation.

    Local read limits tighten a shared work meter; they never renew it. The
    boundary controller passes one meter through preparation, readers, and
    nested executors. Standalone evidence audits may omit that meter and
    retain their explicit local limits.
    ``schedule_subgoal`` hands the full question to that same controller.
    """
    reasoner: object
    schedule_subgoal: Callable | None = None
    row: int = 0
    max_nodes: int = 256
    max_records: int = 1024
    max_steps: int = 8
    max_expansions: int = 1024
    work: QueryWorkBudget | None = None

    def __post_init__(self):
        if self.work is not None and not isinstance(self.work, QueryWorkBudget):
            raise TypeError("query work requires one shared QueryWorkBudget")
        for name in ('row', 'max_nodes', 'max_records', 'max_steps', 'max_expansions'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'query {name} must be a non-negative integer')


def _validate_argument(value, kind):
    if kind == 'description':
        if not isinstance(value, ConceptualMeaning):
            raise TypeError('query description argument must preserve a complete ConceptualMeaning')
    elif kind == 'reference':
        concept_reference(value)
    elif kind == 'concept':
        if isinstance(value, tuple):
            concept_reference(value)
        elif not torch.is_tensor(value) or value.ndim != 1 or not value.numel():
            raise TypeError('query concept argument requires a typed reference or one full-width vector')
        elif not bool(torch.isfinite(value).all()):
            raise FloatingPointError('query concept argument must be finite')
    else:
        raise ValueError(f'unknown query argument type {kind!r}')


def _require_query_boundary(context):
    """Check model phase and row permission before query reads/execution.

    Standalone evidence readers have no BasicModel runtime. When a model
    supplies the guard, only its explicit completed-answer boundary may open a
    query row.
    """
    if isinstance(context, ThoughtGrammarContext):
        context.require_boundary()
        return
    guard = getattr(context.reasoner.model, '_assert_query_boundary', None)
    if callable(guard):
        guard(context.row)


@dataclass(frozen=True)
class QuerySignature:
    """One checked interface to a shared grammatical relation identity."""
    name: str
    semantic_id: str
    domain: str
    argument_roles: tuple
    argument_kinds: tuple
    occupied_roles: tuple
    open_roles: tuple
    result_roles: tuple
    result_kind: str
    read_scope: tuple
    write_scope: tuple
    evidence_kind: str
    compose_faces: tuple
    executor: Callable

    def __post_init__(self):
        if not callable(self.executor):
            raise ValueError('query signature requires an executor')
        if not self.semantic_id or not self.domain or not self.read_scope:
            raise ValueError('query signature requires identity, domain and read scope')
        for roles in (self.argument_roles, self.occupied_roles, self.open_roles, self.result_roles):
            if (not isinstance(roles, tuple) or len(set(roles)) != len(roles)
                    or any(type(role) is not int or role not in (0, 1, 2) for role in roles)):
                raise ValueError('query signature has invalid or ambiguous roles')
        if (len(self.argument_roles) != len(self.argument_kinds)
                or 1 in self.argument_roles or 1 not in self.occupied_roles
                or not set(self.argument_roles).issubset(self.occupied_roles)
                or set(self.open_roles) & set(self.occupied_roles)):
            raise ValueError('query signature has inconsistent grammatical roles')
        if any(kind not in ('reference', 'description', 'concept') for kind in self.argument_kinds):
            raise ValueError('query signature has an unsupported argument type')
        if self.result_kind not in ('truth', 'concept', 'set', 'code', 'prediction', 'subgoal'):
            raise ValueError('query signature has an unsupported result kind')

    @property
    def absent_roles(self):
        return tuple(role for role in range(3)
                     if role not in self.occupied_roles and role not in self.open_roles)

    def invoke(self, context, *arguments, domain=None):
        """Validate a selected call completely before invoking its executor."""
        if not isinstance(context, QueryContext):
            raise TypeError('query execution requires a QueryContext')
        _require_query_boundary(context)
        if domain is not None and domain != self.domain:
            raise ValueError(f'query {self.name} does not support domain {domain!r}')
        if len(arguments) != len(self.argument_kinds):
            raise ValueError(f'query signature {self.name} requires {len(self.argument_kinds)} arguments')
        for value, kind in zip(arguments, self.argument_kinds):
            _validate_argument(value, kind)
        shape = getattr(_concept_space(context), 'outputShape', None)
        if shape is not None:
            width = int(shape[-1])
            for value, kind in zip(arguments, self.argument_kinds):
                payload = value.roles if kind == 'description' else value
                if torch.is_tensor(payload) and payload.shape[-1] != width:
                    raise ValueError('query argument must have the full conceptual width')
        # Canonical operand order is independent of the surface interface.
        by_role = dict(zip(self.argument_roles, arguments))
        try:
            if context.work is not None:
                context.work.require("operation")
            result = self.executor(context, by_role)
        except QueryWorkExhausted:
            result = _work_exhausted()
        if not isinstance(result, dict):
            raise TypeError(f'query executor {self.name} returned an invalid result')
        return dict(result, result_kind=self.result_kind, evidence_kind=self.evidence_kind,
                    semantic_id=self.semantic_id, domain=self.domain)


def _work_exhausted():
    return {"value": None, "support_true": 0.0, "support_false": 0.0,
            "candidates": (), "incomplete": ("work_budget",)}


def _concept_space(context):
    if isinstance(context, ThoughtGrammarContext):
        return context.conceptual_space
    return getattr(context.reasoner.model, 'conceptualSpace', None)


def _argument(arguments, role):
    """Read a canonical thought role or its legacy numeric slot.

    New executor descriptors receive grammar role labels (``I1`` / ``I2``).
    The old query registry remains below during migration and still supplies
    the historical ``0`` / ``2`` slots, so this small bridge keeps one
    executor body without reintroducing aliases into the new catalogue.
    """
    if role in arguments:
        return arguments[role]
    slot = {'I1': 0, 'I2': 2}.get(role)
    if slot is None or slot not in arguments:
        raise ValueError(f'query argument {role!r} is unavailable')
    return arguments[slot]


def _existing_row(space, reference):
    """Read a concept row without lazy allocator creation or row assignment."""
    ref = concept_reference(reference)
    allocator = getattr(space, '_concept_allocator', None)
    if (allocator is None or ref[1] not in allocator.placement
            or ref[1] in allocator.retired):
        raise ValueError('query concept reference is unavailable')
    layer = allocator._layers.get(0)
    if layer is None:
        raise ValueError('query conceptual reference store is unavailable')
    # Row namespaces are addresses in the existing owner, not new semantics.
    for namespace in ('shared', 'snap', 'pool'):
        row = layer.row_of((namespace, ref[1]))
        if row is not None:
            return int(row)
    for order in range(1, len(space._order_caps())):
        row = layer.row_of((f'o{order}', ref[1]))
        if row is not None:
            return int(row)
    raise ValueError('query concept has no allocated payload row')


def _basis(space):
    book = getattr(space, 'similarity_codebook', None)
    if book is None:
        raise ValueError('query conceptual codebook is unavailable')
    read = getattr(book, 'active_prototypes', None)
    values = read() if callable(read) else book.getW()
    if not torch.is_tensor(values) or values.ndim != 2:
        raise ValueError('query conceptual codebook is unavailable')
    return values


def _detach_boundary_value(value):
    """Copy tensor-bearing reader output across the checked hard boundary."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, ConceptualMeaning):
        return value.detached()
    if isinstance(value, dict):
        return {key: _detach_boundary_value(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_boundary_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_boundary_value(item) for item in value)
    return value


def _freeze_boundary_value(value):
    """Detach a boundary value and make container evidence non-writable.

    A result can carry nested provenance records, so a top-level
    ``MappingProxyType`` alone is not a meaningful ownership boundary.  The
    public result still has the familiar mapping/tuple shape, but no executor
    tensor or mutable evidence container survives into history/controller
    code.
    """
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, ConceptualMeaning):
        return value.detached()
    if isinstance(value, (dict, MappingProxyType)):
        return MappingProxyType({
            key: _freeze_boundary_value(item)
            for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_boundary_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_boundary_value(item) for item in value)
    return value


class ThoughtConceptualCapability:
    """Narrow read-only conceptual-space capability for thought execution.

    It deliberately owns no allocator/write API.  The private native-space
    reference exists only to resolve already allocated typed references and
    never reaches an executor: ``ThoughtSignature`` places this behind its
    descriptor-specific facade before a call.
    """

    __slots__ = ('__space', '__equal', 'width')

    def __init__(self, space, equal):
        shape = getattr(space, 'outputShape', None)
        if shape is None or not shape:
            raise ValueError('thought conceptual capability requires a full-width space')
        width = int(shape[-1])
        if width <= 0 or not callable(equal):
            raise ValueError('thought conceptual capability is incomplete')
        object.__setattr__(self, '_ThoughtConceptualCapability__space', space)
        object.__setattr__(self, '_ThoughtConceptualCapability__equal', equal)
        object.__setattr__(self, 'width', width)

    def matches(self, space):
        """Private registry identity check; it is not semantic input."""
        return object.__getattribute__(
            self, '_ThoughtConceptualCapability__space') is space

    def payload(self, reference, *, work=None):
        space = object.__getattribute__(self, '_ThoughtConceptualCapability__space')
        row = _existing_row(space, reference)
        basis = _basis(space)
        if not 0 <= row < len(basis):
            raise ValueError('thought conceptual payload is not active')
        value = basis[row].detach().clone()
        if value.ndim != 1 or int(value.numel()) != self.width:
            raise ValueError('thought conceptual payload differs from the full width')
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('thought conceptual payload must be finite')
        return value

    def equal(self, left, right):
        if left.shape != right.shape or int(left.shape[-1]) != self.width:
            raise ValueError('thought equality requires equal full-width concepts')
        equal = object.__getattribute__(self, '_ThoughtConceptualCapability__equal')
        value = equal(left.detach(), right.detach())
        if torch.is_tensor(value):
            value = value.detach().item()
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError('thought equality capability returned an invalid score')
        return value

    def quantize(self, value, *, max_nodes, max_records, max_expansions, work):
        if type(max_nodes) is not int or max_nodes < 0:
            raise ValueError('thought quantize requires a non-negative node limit')
        if max_nodes == 0:
            return {'value': None, 'reference': None, 'nodes_scanned': 0,
                    'incomplete': ('capture_limit',)}
        if isinstance(value, tuple):
            return {'value': self.payload(value, work=work), 'reference': value,
                    'nodes_scanned': 1, 'incomplete': ()}
        if (not torch.is_tensor(value) or value.ndim != 1
                or int(value.numel()) != self.width):
            raise ValueError('thought quantize requires one full-width concept')
        space = object.__getattribute__(self, '_ThoughtConceptualCapability__space')
        allocator = getattr(space, '_concept_allocator', None)
        if allocator is None:
            return {'value': None, 'reference': None, 'nodes_scanned': 0,
                    'incomplete': ('unavailable_conceptual_codebook',)}
        # Native codebook access is a real node read.  Keep it inside the
        # charged loop so an exhausted selected operation cannot inspect the
        # basis merely while discovering that it has no allowance left.
        basis = None
        candidates, scanned, incomplete = [], 0, []
        for concept_id in allocator.placement:
            if scanned >= max_nodes:
                incomplete.append('capture_limit')
                break
            if work is not None and not work.consume('node'):
                incomplete.append('work_budget')
                break
            scanned += 1
            if basis is None:
                basis = _basis(space)
            if concept_id in allocator.retired:
                continue
            reference = ('sym', int(concept_id))
            try:
                row = _existing_row(space, reference)
            except ValueError:
                continue
            if not 0 <= row < len(basis):
                continue
            atom = basis[row].detach().clone()
            if not bool(torch.isfinite(atom).all()):
                raise FloatingPointError('thought conceptual payload must be finite')
            candidates.append((self.equal(value, atom), reference, atom))
        if not candidates:
            return {'value': None, 'reference': None, 'nodes_scanned': scanned,
                    'incomplete': tuple(incomplete)}
        score, reference, selected = max(candidates, key=lambda item: item[0])
        return {'value': selected, 'reference': reference, 'match': score,
                'nodes_scanned': scanned, 'incomplete': tuple(incomplete)}


class ThoughtTaxonomyCapability:
    """Read-only bounded taxonomy traversal over one native conceptual owner."""

    __slots__ = ('__space',)

    def __init__(self, space):
        object.__setattr__(self, '_ThoughtTaxonomyCapability__space', space)

    def evidence(self, part, whole, *, max_nodes, max_records, max_steps,
                 max_expansions, work):
        nodes, records = capture_limits(work, max_nodes, max_records)
        view = capture_taxonomy(
            object.__getattribute__(self, '_ThoughtTaxonomyCapability__space'),
            max_nodes=nodes, max_records=records, focus=(part, whole), work=work)
        return _detach_boundary_value(view.part_of(
            part, whole, max_steps=max_steps, max_expansions=max_expansions,
            work=work))

    def neighbors(self, reference, *, direction, max_nodes, max_records,
                  max_expansions, work):
        """Return bounded up/down neighbors for a grammar-open part role."""
        if direction not in ('up', 'down'):
            raise ValueError('thought taxonomy neighbor direction is invalid')
        nodes, records = capture_limits(work, max_nodes, max_records)
        view = capture_taxonomy(
            object.__getattribute__(self, '_ThoughtTaxonomyCapability__space'),
            max_nodes=nodes, max_records=records, focus=(reference,), work=work)
        edges = view.neighbors(reference, direction=direction)
        values, incomplete = [], list(view.incomplete)
        for index, edge in enumerate(edges):
            if index >= max_expansions:
                incomplete.append('traversal_limit')
                break
            if not work.consume('expansion'):
                incomplete.append('work_budget')
                break
            values.append({
                'reference': edge.whole if direction == 'up' else edge.part,
                'source': edge,
                'trust': 1.0,
            })
        return _detach_boundary_value({
            'value': tuple(values),
            'incomplete': tuple(dict.fromkeys(incomplete)),
            'nodes_scanned': view.nodes_scanned,
            'records_scanned': view.records_scanned,
            'edges_expanded': len(values),
        })


class ThoughtLTMCapability:
    """Narrow LTM/episode/prediction reader used only at a thought boundary."""

    __slots__ = ('__existence', '__store', '__equal', '__tau', '__memory',
                 '__discourse')

    def __init__(self, *, existence_evidence, store, equal, tau_id,
                 memory=None, discourse=None):
        if (not callable(existence_evidence) or not callable(store)
                or not callable(equal)):
            raise TypeError('thought LTM capability requires bounded reader callables')
        object.__setattr__(self, '_ThoughtLTMCapability__existence',
                           existence_evidence)
        object.__setattr__(self, '_ThoughtLTMCapability__store', store)
        object.__setattr__(self, '_ThoughtLTMCapability__equal', equal)
        object.__setattr__(self, '_ThoughtLTMCapability__tau', float(tau_id))
        object.__setattr__(self, '_ThoughtLTMCapability__memory', memory)
        object.__setattr__(self, '_ThoughtLTMCapability__discourse', discourse)

    def existence_evidence(self, description, *, max_records, work):
        read = object.__getattribute__(self, '_ThoughtLTMCapability__existence')
        return _detach_boundary_value(read(
            description.detached() if isinstance(description, ConceptualMeaning)
            else _detach_boundary_value(description),
            max_records=max_records, work=work))

    def resolve_description(self, reference, *, row, max_records, work):
        memory = object.__getattribute__(self, '_ThoughtLTMCapability__memory')
        if isinstance(reference, tuple) and reference and reference[0] == 'thought':
            if memory is None:
                raise ValueError('thought occurrence owner is unavailable')
            value, scanned = memory.resolve_thought(
                reference, b=row, max_records=max_records, work=work)
            return value.detached(), scanned
        if (not isinstance(reference, tuple) or len(reference) != 3
                or reference[0] != 'ltm' or not isinstance(reference[1], str)
                or type(reference[2]) is not int or reference[2] < 0):
            raise TypeError('description argument requires an existing occurrence reference')
        store = object.__getattribute__(self, '_ThoughtLTMCapability__store')()
        from Layers import TernaryTruthStore
        if (not isinstance(store, TernaryTruthStore)
                or bytes(store._occurrence_namespace.tolist()).hex() != reference[1]):
            raise ValueError('description occurrence namespace is unavailable')
        for index in range(min(len(store), max_records)):
            work.require('record')
            if int(store.occurrence_id[index]) == reference[2]:
                value = store.meaning_of(index)
                if value is None:
                    raise ValueError('description occurrence metadata is unavailable')
                return value.detached(), index + 1
        raise ValueError('description occurrence is unavailable within the query read limit')

    def lookup(self, left, right, *, max_records, max_expansions, work):
        store = object.__getattribute__(self, '_ThoughtLTMCapability__store')()
        from Layers import TernaryTruthStore
        if not isinstance(store, TernaryTruthStore):
            return {'value': (), 'records_scanned': 0,
                    'incomplete': ('unavailable_ltm',)}
        if store.nDim != left.numel():
            raise ValueError('thought lookup width differs from LTM')
        equal = object.__getattribute__(self, '_ThoughtLTMCapability__equal')
        tau = object.__getattribute__(self, '_ThoughtLTMCapability__tau')
        found, incomplete, scanned = [], [], 0
        count = min(len(store), max_records)
        if count < len(store):
            incomplete.append('capture_limit')
        for index in range(count):
            if not work.consume('record'):
                incomplete.append('work_budget')
                break
            scanned += 1
            record = store.row(index)
            meaning = record['meaning']
            if meaning is None or not record['metadata_complete']:
                incomplete.append('unavailable_metadata')
                continue
            if not bool(meaning.role_mask[0] and meaning.role_mask[2]):
                continue
            match = min(equal(left.detach().to(meaning.roles), meaning.roles[0]),
                        equal(right.detach().to(meaning.roles), meaning.roles[2]))
            if match >= tau:
                found.append(_detach_boundary_value(dict(record, match=float(match))))
        return {'value': tuple(found), 'records_scanned': scanned,
                'incomplete': tuple(dict.fromkeys(incomplete))}

    def expectation(self, row, *, work):
        discourse = object.__getattribute__(self, '_ThoughtLTMCapability__discourse')
        if discourse is None:
            return None
        if getattr(discourse, 'expectation_scope', None) != 'structured':
            raise ValueError('thought arma requires the full structured predictor')
        value = discourse.expect_next_meaning(int(row), record=False, work=work)
        if value is None:
            return None
        try:
            return type(value)(value.roles.detach().clone(),
                               value.presence_logits.detach().clone())
        except (AttributeError, TypeError):
            return _detach_boundary_value(value)


def _vector(context, value):
    if torch.is_tensor(value):
        return value
    if context.work is not None:
        context.work.require("payload")
    if isinstance(context, ThoughtGrammarContext):
        payload = getattr(context.conceptual_space, 'payload', None)
        if not callable(payload):
            raise ValueError('thought conceptual capability does not admit payload reads')
        vector = payload(value, work=context.work)
        if not torch.is_tensor(vector) or vector.ndim != 1:
            raise ValueError('thought conceptual payload capability returned no full vector')
        if not bool(torch.isfinite(vector).all()):
            raise FloatingPointError('thought native concept payload must be finite')
        return vector
    space = _concept_space(context)
    row = _existing_row(space, value)
    values = _basis(space)
    if not 0 <= row < len(values):
        raise ValueError('query concept payload is not active')
    vector = values[row].clone()
    if not bool(torch.isfinite(vector).all()):
        raise FloatingPointError('query native concept payload must be finite')
    return vector


def _exist(context, arguments):
    if isinstance(context, ThoughtGrammarContext):
        reader = getattr(context.ltm, 'existence_evidence', None)
        if not callable(reader):
            raise ValueError('thought LTM capability does not admit fact evidence')
        return reader(_argument(arguments, 'I1'), max_records=context.max_records,
                      work=context.work)
    keyword = {} if context.work is None else {"work": context.work}
    return context.reasoner.existence_evidence(
        _argument(arguments, 'I1'), max_records=context.max_records, **keyword)


def _part(context, arguments):
    if isinstance(context, ThoughtGrammarContext):
        if len(arguments) == 1:
            role, reference = next(iter(arguments.items()))
            if role not in ('I1', 'I2'):
                raise ValueError('thought open part role is invalid')
            neighbors = getattr(context.taxonomy, 'neighbors', None)
            if not callable(neighbors):
                raise ValueError('thought taxonomy capability does not admit neighbors')
            return neighbors(
                reference, direction='up' if role == 'I1' else 'down',
                max_nodes=context.max_nodes, max_records=context.max_records,
                max_expansions=context.max_expansions, work=context.work)
        return context.taxonomy.evidence(
            _argument(arguments, 'I1'), _argument(arguments, 'I2'),
            max_nodes=context.max_nodes,
            max_records=context.max_records, max_steps=context.max_steps,
            max_expansions=context.max_expansions, work=context.work)
    keyword = {} if context.work is None else {"work": context.work}
    return context.reasoner.taxonomy_evidence(
        arguments[0], arguments[2], max_nodes=context.max_nodes,
        max_records=context.max_records, max_steps=context.max_steps,
        max_expansions=context.max_expansions, **keyword)


def _neighbors(context, arguments):
    role = next(iter(arguments))
    reference = arguments[role]
    direction = 'up' if role == 0 else 'down'
    # Preserve incomplete diagnostics even when no neighbor is available.
    nodes, records = capture_limits(
        context.work, context.max_nodes, context.max_records)
    view = capture_taxonomy(
        _concept_space(context), max_nodes=nodes, max_records=records,
        focus=(reference,), work=context.work)
    edges = view.neighbors(reference, direction=direction)
    values, incomplete = [], list(view.incomplete)
    for index in range(len(edges)):
        if index >= context.max_expansions:
            incomplete.append('traversal_limit')
            break
        if context.work is not None and not context.work.consume('expansion'):
            incomplete.append('work_budget')
            break
        edge = edges[index]
        values.append({'reference': edge.whole if role == 0 else edge.part,
                       'source': edge, 'trust': 1.0})
    return {'value': values, 'incomplete': tuple(dict.fromkeys(incomplete)),
            'nodes_scanned': view.nodes_scanned,
            'records_scanned': view.records_scanned,
            'edges_expanded': len(values)}


def _equal(context, arguments):
    left, right = (_vector(context, _argument(arguments, role))
                   for role in ('I1', 'I2'))
    if left.shape != right.shape:
        raise ValueError('query equality requires equal full concept width')
    if isinstance(context, ThoughtGrammarContext):
        equal = getattr(context.conceptual_space, 'equal', None)
        if not callable(equal):
            raise ValueError('thought conceptual capability does not admit equality')
        score = equal(left, right)
    else:
        score = context.reasoner.equal(left, right)
    return {'support_true': score, 'support_false': 0.0, 'candidates': []}


def _lookup(context, arguments):
    """The distinct two-operand LTM read; retrieval never admits a fact."""
    left, right = (_vector(context, _argument(arguments, role))
                   for role in ('I1', 'I2'))
    if left.shape != right.shape:
        raise ValueError('query lookup requires equal full concept width')
    if isinstance(context, ThoughtGrammarContext):
        lookup = getattr(context.ltm, 'lookup', None)
        if not callable(lookup):
            raise ValueError('thought LTM capability does not admit lookup')
        return lookup(left, right, max_records=context.max_records,
                      max_expansions=context.max_expansions, work=context.work)
    store = context.reasoner.reasoning_store()
    from Layers import TernaryTruthStore
    if not isinstance(store, TernaryTruthStore):
        return {'value': (), 'records_scanned': 0, 'incomplete': ('unavailable_ltm',)}
    if store.nDim != left.numel():
        raise ValueError('query lookup width differs from LTM')
    found = []
    count = min(len(store), context.max_records)
    incomplete = ['capture_limit'] if count < len(store) else []
    scanned = 0
    for index in range(count):
        if context.work is not None and not context.work.consume("record"):
            incomplete.append("work_budget")
            break
        scanned += 1
        row = store.row(index)
        meaning = row['meaning']
        if meaning is None or not row['metadata_complete']:
            incomplete.append('unavailable_metadata')
            continue
        if not bool(meaning.role_mask[0] and meaning.role_mask[2]):
            continue
        match = min(context.reasoner.equal(left.to(meaning.roles), meaning.roles[0]),
                    context.reasoner.equal(right.to(meaning.roles), meaning.roles[2]))
        if match >= context.reasoner.tau_id:
            found.append(dict(row, match=match))
    return {'value': tuple(found), 'records_scanned': scanned,
            'incomplete': tuple(dict.fromkeys(incomplete))}


def _quantize(context, arguments):
    """Choose an allocated conceptual atom; SymbolSpace is never consulted."""
    if context.max_nodes == 0:
        return {'value': None, 'reference': None, 'nodes_scanned': 0,
                'incomplete': ('capture_limit',)}
    if isinstance(context, ThoughtGrammarContext):
        quantize = getattr(context.conceptual_space, 'quantize', None)
        if not callable(quantize):
            raise ValueError('thought conceptual capability does not admit quantization')
        return quantize(_argument(arguments, 'I1'), max_nodes=context.max_nodes,
                        max_records=context.max_records,
                        max_expansions=context.max_expansions, work=context.work)
    space = _concept_space(context)
    input_value = _argument(arguments, 'I1')
    if isinstance(input_value, tuple):
        value = _vector(context, input_value)
        return {'value': value, 'reference': input_value, 'nodes_scanned': 1,
                'incomplete': ()}
    vector = input_value
    allocator = getattr(space, '_concept_allocator', None)
    if allocator is None:
        return {'value': None, 'reference': None, 'nodes_scanned': 0,
                'incomplete': ('unavailable_conceptual_codebook',)}
    basis = None
    candidates, scanned, incomplete = [], 0, []
    for concept_id in allocator.placement:
        if scanned >= context.max_nodes:
            incomplete.append('capture_limit')
            break
        if context.work is not None and not context.work.consume("node"):
            incomplete.append("work_budget")
            break
        scanned += 1
        if basis is None:
            basis = _basis(space)
            if basis.shape[1] != vector.numel():
                raise ValueError('query quantize requires the full conceptual width')
        if concept_id in allocator.retired:
            continue
        reference = ('sym', int(concept_id))
        try:
            row = _existing_row(space, reference)
        except ValueError:
            continue
        if row < len(basis):
            atom = basis[row].clone()
            if not bool(torch.isfinite(atom).all()):
                raise FloatingPointError('query native concept payload must be finite')
            score = context.reasoner.equal(vector.to(atom), atom)
            candidates.append((score, reference, atom))
    if not candidates:
        return {'value': None, 'reference': None, 'nodes_scanned': scanned,
                'incomplete': tuple(incomplete)}
    score, reference, value = max(candidates, key=lambda item: item[0])
    return {'value': value, 'reference': reference, 'match': score,
            'nodes_scanned': scanned, 'incomplete': tuple(incomplete)}


def _arma(context, arguments):
    """Read the row's full prior estimate without staging an observation target."""
    if isinstance(context, ThoughtGrammarContext):
        predict = getattr(context.ltm, 'expectation', None)
        if not callable(predict):
            raise ValueError('thought LTM capability does not admit prediction')
        value = predict(context.row, work=context.work)
        return {'value': value,
                'incomplete': () if value is not None else ('cold_prediction',)}
    discourse = getattr(getattr(context.reasoner.model, 'symbolSpace', None), 'discourse', None)
    if discourse is None:
        return {'value': None, 'incomplete': ('unavailable_prediction',)}
    if getattr(discourse, 'expectation_scope', None) != 'structured':
        raise ValueError('query arma requires the full structured predictor')
    if arguments[0].roles.shape[-1] != discourse.concept_dim:
        raise ValueError('query arma description width differs from the predictor')
    keyword = {} if context.work is None else {"work": context.work}
    value = discourse.expect_next_meaning(context.row, record=False, **keyword)
    return {'value': value, 'incomplete': () if value is not None else ('cold_prediction',)}


def _what(context, arguments):
    question = _argument(arguments, 'I1')
    if question.mode != 'interrogative':
        raise ValueError('query what requires an interrogative conceptual question')
    if isinstance(context, ThoughtGrammarContext):
        if not callable(context.continuation):
            raise RuntimeError('thought what requires the active boundary controller')
        return {'value': context.continuation(question)}
    if not callable(context.schedule_subgoal):
        raise RuntimeError('query what requires the active boundary controller')
    return {'value': context.schedule_subgoal(question)}


@dataclass(frozen=True)
class ThoughtExecutorDescriptor:
    """Non-grammatical capability contract for one canonical thought face.

    Arity, role labels, open-role variants, converse form, and structural rule
    IDs deliberately do not live here.  They are supplied by the immutable
    ``Grammar.thought_operations`` family before this descriptor can be
    exposed at a boundary.
    """
    semantic_id: str
    domain: str
    argument_kinds: tuple
    result_kind: str
    read_scope: tuple
    write_scope: tuple
    evidence_kind: str
    executor: Callable

    def __post_init__(self):
        if (not isinstance(self.semantic_id, str)
                or re.fullmatch(r'[A-Za-z_]\w*', self.semantic_id,
                                re.ASCII) is None):
            raise ValueError('thought executor requires a canonical semantic id')
        if not isinstance(self.domain, str) or not self.domain:
            raise ValueError('thought executor requires a domain')
        if (not isinstance(self.argument_kinds, tuple)
                or any(kind not in ('reference', 'description', 'concept')
                       for kind in self.argument_kinds)):
            raise ValueError('thought executor has unsupported argument kinds')
        if self.result_kind not in ('truth', 'concept', 'set', 'code',
                                    'prediction', 'subgoal'):
            raise ValueError('thought executor has unsupported result kind')
        if (not isinstance(self.read_scope, tuple) or not self.read_scope
                or not isinstance(self.write_scope, tuple)
                or not isinstance(self.evidence_kind, str)
                or not self.evidence_kind or not callable(self.executor)):
            raise ValueError('thought executor has an incomplete capability contract')


# The canonical names are deliberately the structural grammar names.  The
# old is-/query-/plural spellings remain only in the pre-item-0 code below
# until every caller is migrated; they are not an authority for this table.
_thought_executors = (
    ThoughtExecutorDescriptor(
        'exist', 'ltm-facts', ('description',), 'truth',
        ('ltm.descriptions', 'ltm.facts'), (), 'fact', _exist),
    ThoughtExecutorDescriptor(
        'part', 'conceptual-taxonomy', ('reference', 'reference'), 'truth',
        ('conceptual.references',), (), 'taxonomy', _part),
    ThoughtExecutorDescriptor(
        'equal', 'conceptual-identity', ('concept', 'concept'), 'truth',
        ('conceptual.payloads',), (), 'conceptual-identity', _equal),
    ThoughtExecutorDescriptor(
        'lookup', 'ltm-lookup', ('concept', 'concept'), 'set',
        ('ltm.records',), (), 'retrieval', _lookup),
    ThoughtExecutorDescriptor(
        'quantize', 'conceptual-codebook', ('concept',), 'code',
        ('conceptual.codebook',), (), 'concept-codebook', _quantize),
    ThoughtExecutorDescriptor(
        'arma', 'discourse-prediction', ('description',), 'prediction',
        ('ltm.descriptions', 'ltm.prediction'), (), 'estimate', _arma),
    ThoughtExecutorDescriptor(
        'what', 'conceptual-subgoal', ('description',), 'subgoal',
        ('ltm.descriptions', 'question.meaning', 'episode.context'),
        ('episode.schedule',),
        'subgoal', _what),
)
THOUGHT_EXECUTORS = MappingProxyType(
    {descriptor.semantic_id: descriptor for descriptor in _thought_executors})


_THOUGHT_SCOPE_METHODS = MappingProxyType({
    # The name before the dot is a declared semantic capability, not a
    # concrete class.  The tuple says which context member receives which
    # callable names once an operation has been selected.
    'ltm.facts': ('ltm', ('existence_evidence',)),
    'ltm.descriptions': ('ltm', ('resolve_description',)),
    'ltm.records': ('ltm', ('lookup',)),
    'ltm.prediction': ('ltm', ('expectation',)),
    'conceptual.references': ('taxonomy', ('evidence', 'neighbors')),
    'conceptual.payloads': ('conceptual_space', ('payload', 'equal')),
    'conceptual.codebook': ('conceptual_space', ('quantize',)),
    # These scopes certify a completed meaning / episode context but do not
    # turn into a broad reader.  ``episode.schedule`` below is the one narrow
    # continuation capability and is intentionally write-scoped.
    'question.meaning': (None, ()),
    'episode.context': (None, ()),
})
_THOUGHT_WRITE_SCOPES = frozenset(('episode.schedule',))


def _descriptor_context(context, descriptor):
    """Return the descriptor-scoped view of a checked thought context.

    Reader presence is deliberately soft here: a selected executor gives the
    precise ``does not admit ...`` error when it actually needs a missing
    method.  That keeps a tensor-only operation usable with a minimal
    conceptual capability while still making every *available* method an
    explicit descriptor grant.
    """
    grants = {'conceptual_space': set(), 'ltm': set(), 'taxonomy': set()}
    for scope in descriptor.read_scope:
        surface = _THOUGHT_SCOPE_METHODS.get(scope)
        if surface is None:
            raise ValueError(
                f'thought executor {descriptor.semantic_id!r} declares '
                f'an unknown read scope {scope!r}')
        member, names = surface
        if member is not None:
            grants[member].update(names)
    unknown_writes = set(descriptor.write_scope).difference(_THOUGHT_WRITE_SCOPES)
    if unknown_writes:
        raise ValueError(
            f'thought executor {descriptor.semantic_id!r} declares unknown '
            f'write scope(s) {sorted(unknown_writes)!r}')
    return replace(
        context,
        conceptual_space=_ThoughtCapabilityView(
            context.conceptual_space, grants['conceptual_space']),
        ltm=_ThoughtCapabilityView(context.ltm, grants['ltm']),
        taxonomy=_ThoughtCapabilityView(context.taxonomy, grants['taxonomy']),
        continuation=(context.continuation
                      if 'episode.schedule' in descriptor.write_scope else None),
    )


@dataclass(frozen=True)
class ThoughtSignature:
    """One occupancy-specialized checked call of a grammar-owned operation."""
    operation: object
    descriptor: ThoughtExecutorDescriptor
    occupied_roles: tuple

    def __post_init__(self):
        roles = tuple(getattr(self.operation, 'operand_roles', ()))
        if (not roles or not set(self.occupied_roles).issubset(roles)
                or len(set(self.occupied_roles)) != len(self.occupied_roles)):
            raise ValueError('thought signature has invalid role occupancy')
        # A relation with one grammar-open side is the existing bounded
        # taxonomy-neighbor operation.  Other executors have no invented
        # partial-call meaning; their operand masks fail before a reader runs.
        open_roles = tuple(role for role in roles if role not in self.occupied_roles)
        if open_roles and not (
                self.descriptor.executor is _part and len(open_roles) == 1):
            raise ValueError('thought operation does not admit this open-role form')

    @property
    def open_roles(self):
        return tuple(role for role in self.operation.operand_roles
                     if role not in self.occupied_roles)

    @property
    def argument_kinds(self):
        kinds = dict(zip(self.operation.operand_roles,
                         self.descriptor.argument_kinds))
        return tuple(kinds[role] for role in self.occupied_roles)

    @property
    def result_kind(self):
        return 'set' if self.open_roles else self.descriptor.result_kind

    def invoke(self, context, *arguments):
        """Validate then run exactly one boundary thought capability."""
        if not isinstance(context, ThoughtGrammarContext):
            raise TypeError('thought execution requires a ThoughtGrammarContext')
        context.require_boundary()
        return self._invoke_permitted(context, *arguments)

    def _invoke_permitted(self, context, *arguments):
        """Run after the public caller has opened this row's boundary once.

        ``GrammaticalThoughtRegistry.execute`` must prove the permit before it
        resolves a description or reads a VP.  Direct signature users have no
        such entry point, so :meth:`invoke` performs that same check.  Sharing
        this body makes both routes obey the common thought context without
        turning one selected call into two observable boundary admissions.
        """
        if len(arguments) != len(self.occupied_roles):
            raise ValueError(
                f'thought operation {self.operation.semantic_id!r} requires '
                f'{len(self.occupied_roles)} bound operands')
        for value, kind in zip(arguments, self.argument_kinds):
            _validate_argument(value, kind)
        # The public thought context carries a capability, not the raw space.
        # It names its full width directly; a raw space remains supported only
        # for isolated legacy fixtures while migration completes.
        width = getattr(context.conceptual_space, 'width', None)
        if type(width) is not int:
            shape = getattr(context.conceptual_space, 'outputShape', None)
            width = int(shape[-1]) if shape is not None else None
        if type(width) is int and width > 0:
            for value, kind in zip(arguments, self.argument_kinds):
                payload = value.roles if kind == 'description' else value
                if torch.is_tensor(payload) and payload.shape[-1] != width:
                    raise ValueError('thought argument must have the full conceptual width')
        # An executor is a checked reader.  It cannot retain a live structural
        # operand (or mutate one through an alias) across the thought boundary.
        # The controller's learned policy uses its separately supplied selected
        # meanings; this call path is deliberately data-only.
        frozen_arguments = tuple(_freeze_boundary_value(value)
                                 for value in arguments)
        by_role = dict(zip(self.occupied_roles, frozen_arguments))
        try:
            context.work.require('operation')
            result = self.descriptor.executor(
                _descriptor_context(context, self.descriptor), by_role)
        except QueryWorkExhausted:
            result = _work_exhausted()
        if not isinstance(result, dict):
            raise TypeError(
                f'thought executor {self.operation.semantic_id!r} returned an invalid result')
        result = _freeze_boundary_value(dict(
            result, result_kind=self.result_kind,
            evidence_kind=self.descriptor.evidence_kind,
            semantic_id=self.operation.semantic_id,
            domain=self.descriptor.domain))
        if not isinstance(result, MappingProxyType):  # pragma: no cover - helper invariant
            raise AssertionError('thought result evidence must be frozen')
        return result


def _signature(name, semantic_id, domain, roles, kinds, result_kind, read_scope,
               evidence_kind, executor, *, open_roles=(), result_roles=(),
               write_scope=(), compose_faces=()):
    return QuerySignature(name, semantic_id, domain, roles, kinds,
                          tuple(sorted(set(roles) | {1})), open_roles,
                          result_roles, result_kind, read_scope, write_scope,
                          evidence_kind, compose_faces, executor)


_signatures = []
for name in ('isTrue', 'exist'):
    _signatures.append(_signature(name, 'exist', 'ltm-facts', (0,), ('description',),
        'truth', ('ltm.facts',), 'fact', _exist, compose_faces=('exist',)))
for names, roles in [(('isPart', 'part', 'queryPart', 'PartOf'), (0, 2)),
                     (('isWhole', 'whole'), (2, 0))]:
    for name in names:
        _signatures.append(_signature(name, 'part', 'conceptual-taxonomy', roles,
            ('reference', 'reference'), 'truth', ('conceptual.references',),
            'taxonomy', _part, compose_faces=('part', 'whole')))
for name, role, opened in [('parts', 2, 0), ('wholes', 0, 2)]:
    _signatures.append(_signature(name, 'part', 'conceptual-taxonomy', (role,),
        ('reference',), 'set', ('conceptual.references',), 'taxonomy', _neighbors,
        open_roles=(opened,), result_roles=(opened,), compose_faces=('part', 'whole')))
for name in ('isEqual', 'equal', 'queryEqual'):
    _signatures.append(_signature(name, 'equal', 'conceptual-identity', (0, 2),
        ('concept', 'concept'), 'truth', ('conceptual.payloads',),
        'conceptual-identity', _equal, compose_faces=('equal',)))
_signatures += [
    _signature('query', 'lookup', 'ltm-lookup', (0, 2), ('concept', 'concept'),
               'set', ('ltm.records',), 'retrieval', _lookup),
    _signature('quantize', 'quantize', 'conceptual-codebook', (0,), ('concept',),
               'code', ('conceptual.codebook',), 'concept-codebook', _quantize,
               result_roles=(2,)),
    _signature('arma', 'arma', 'discourse-prediction', (0,), ('description',),
               'prediction', ('external.prior-context',), 'estimate', _arma,
               result_roles=(2,)),
    _signature('what', 'what', 'conceptual-subgoal', (0,), ('description',),
               'subgoal', ('question.meaning', 'episode.context'), 'subgoal', _what,
               write_scope=('episode.schedule',), result_roles=(2,)),
]
BUILTIN_QUERIES = MappingProxyType({item.name: item for item in _signatures})


def _occurrence_description(context, reference):
    """Resolve an existing full description under the selected read limit.

    An occurrence is not a concept row. Never create one while constructing a
    candidate, reinterpret an unknown namespace, or fall back to its NP1.
    """
    if isinstance(context, ThoughtGrammarContext):
        reader = getattr(context.ltm, 'resolve_description', None)
        if not callable(reader):
            raise ValueError(
                'thought LTM capability does not admit description resolution')
        # Preparation is a real selected read.  The capability receives the
        # same meter so it can charge any records it actually scans; this
        # fixed charge accounts for entering the read rather than letting an
        # operand become free metadata.
        context.work.require('description')
        resolved = reader(reference, row=context.row,
                          max_records=context.max_records, work=context.work)
        if (not isinstance(resolved, tuple) or len(resolved) != 2
                or not isinstance(resolved[0], ConceptualMeaning)
                or type(resolved[1]) is not int or resolved[1] < 0
                or resolved[1] > context.max_records):
            raise ValueError(
                'thought LTM description capability returned invalid occurrence data')
        return resolved
    from Layers import TernaryTruthStore
    if not isinstance(context, QueryContext):
        raise TypeError('occurrence reference requires its query context')
    if isinstance(reference, tuple) and reference and reference[0] == 'thought':
        memory = getattr(getattr(context.reasoner.model, 'symbolSpace', None), 'what_memory', None)
        if memory is None:
            raise ValueError('thought occurrence owner is unavailable')
        # The existing interaction owner keeps current episode values live and
        # detaches them at the optimizer boundary. This does not copy a memory
        # store, create a new episode, or change the execution context level.
        return memory.resolve_thought(
            reference, b=context.row, max_records=context.max_records,
            work=context.work)
    if (not isinstance(reference, tuple) or len(reference) != 3
            or reference[0] != 'ltm' or not isinstance(reference[1], str)
            or type(reference[2]) is not int or reference[2] < 0):
        raise TypeError('description argument requires an existing occurrence reference')
    store = context.reasoner.reasoning_store()
    if (not isinstance(store, TernaryTruthStore)
            or bytes(store._occurrence_namespace.tolist()).hex() != reference[1]):
        raise ValueError('description occurrence namespace is unavailable')
    for index in range(min(len(store), context.max_records)):
        if context.work is not None:
            context.work.require("record")
        if int(store.occurrence_id[index]) == reference[2]:
            value = store.meaning_of(index)
            if value is None:
                raise ValueError('description occurrence metadata is unavailable')
            return value, index + 1
    raise ValueError('description occurrence is unavailable within the query read limit')


class GrammaticalQueryRegistry:
    """Derived bindings between grammatical VPs and checked boundary calls.

    Native named concepts own the VP payload and identity. This adapter owns
    neither a second embedding table nor a semantic store. ``install`` is an
    explicit setup/migration operation; forming and dispatching meanings never
    mint a concept, allocate a row, or write an LTM occurrence.
    """

    _NAME_PREFIX = 'grammatical-vp:'

    def __init__(self, space, grammar):
        self.space = space
        self.signatures = dict(getattr(grammar, 'query_signatures', {}))
        methods = {rule.method_name for rule in getattr(grammar, 'rules_upward', ())}
        self.compose_faces = {face for signature in BUILTIN_QUERIES.values()
                              for face in signature.compose_faces if face in methods}
        self.identities = tuple(dict.fromkeys(
            (signature.domain, signature.semantic_id)
            for signature in BUILTIN_QUERIES.values()
            if signature.name in self.signatures
            or self.compose_faces.intersection(signature.compose_faces)))

    @classmethod
    def install(cls, space, grammar):
        """Bind once at setup, reusing the existing checkpointed named concepts."""
        registry = cls(space, grammar)
        for identity in registry.identities:
            name = registry._name(identity)
            space.mint_frozen_concept(name)
            registry._reference(identity)  # reject an incomplete existing binding
        return registry

    @classmethod
    def _name(cls, identity):
        return cls._NAME_PREFIX + ':'.join(identity)

    def _reference(self, identity, *, work=None):
        if identity not in self.identities:
            raise ValueError('grammatical relation is not registered')
        identifier = getattr(self.space, '_frozen_named', {}).get(self._name(identity))
        if type(identifier) is not int or identifier < 1:
            raise ValueError('grammatical VP binding is unavailable')
        reference = ('sym', identifier)
        if work is not None:
            work.require("reference")
        _existing_row(self.space, reference)
        return reference

    def _payload(self, reference, *, work=None):
        if work is not None:
            work.require("payload")
        row = _existing_row(self.space, reference)
        basis = _basis(self.space)
        if not 0 <= row < len(basis):
            raise ValueError('grammatical VP or operand payload is unavailable')
        value = basis[row].clone()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('grammatical native payload must be finite')
        return value

    def form(self, name, *arguments, mode='interrogative', polarity=True,
             bindings=(), scope=(), context=None):
        """Construct canonical roles without executing a query or admitting truth.

        A referenced compound NP has a bounded full-role illumination summary;
        its authoritative content is the resolvable occurrence, including role
        order, bindings and scope. Executors resolve that content, never the
        summary. A caller must already own the occurrence before proposing it.
        """
        signature = BUILTIN_QUERIES.get(name)
        if signature is None or (name not in self.signatures
                and name not in self.compose_faces):
            raise ValueError(f'grammatical query interface {name!r} is not registered')
        if len(arguments) != len(signature.argument_roles):
            raise ValueError(f'query signature {name} requires {len(signature.argument_roles)} arguments')
        work = context.work if isinstance(context, QueryContext) else None
        vp = self._reference((signature.domain, signature.semantic_id), work=work)
        vp_payload = self._payload(vp, work=work)
        payloads = [torch.zeros_like(vp_payload), vp_payload, torch.zeros_like(vp_payload)]
        references = [None, vp, None]
        for role, kind, argument in zip(signature.argument_roles, signature.argument_kinds, arguments):
            if kind == 'description':
                description, _ = _occurrence_description(context, argument)
                # Full structure remains addressable, not inverted from this sum.
                value = description.roles.sum(0) / description.role_mask.sum().sqrt()
                references[role] = argument
            else:
                _validate_argument(argument, kind)
                if isinstance(argument, tuple):
                    value = self._payload(argument, work=work)
                    references[role] = argument
                else:
                    value = argument
            if value.shape != vp_payload.shape:
                raise ValueError('query operand must have the full conceptual width')
            payloads[role] = value.to(vp_payload)
        mask = torch.tensor([role in signature.occupied_roles for role in range(3)],
                            dtype=torch.bool, device=vp_payload.device)
        return ConceptualMeaning(torch.stack(payloads), mask, mode=mode,
                                 polarity=polarity, role_refs=tuple(references),
                                 bindings=bindings, scope=scope)

    def signature_for(self, meaning, *, work=None):
        """Derive an executable interface from the middle VP and bound/open roles."""
        if not isinstance(meaning, ConceptualMeaning) or meaning.mode != 'interrogative':
            raise ValueError('query execution requires a completed interrogative meaning')
        if not bool(meaning.role_mask[1]):
            raise ValueError('query meaning has no grammatical VP')
        reference = concept_reference(meaning.role_refs[1])
        identity = next((item for item in self.identities
                         if getattr(self.space, '_frozen_named', {}).get(self._name(item)) == reference[1]), None)
        if identity is None:
            raise ValueError('query VP is not a registered grammatical relation')
        self._reference(identity, work=work)
        occupied = tuple(meaning.role_mask.nonzero(as_tuple=True)[0].tolist())
        matches = [signature for signature in self.signatures.values()
                   if (signature.domain, signature.semantic_id) == identity
                   and signature.occupied_roles == occupied]
        if not matches:
            raise ValueError('query VP and occupied/open roles have no checked interface')
        # Alias names and converse surface order cannot change the canonical idea.
        signature = matches[0]
        if not meaning.polarity and signature.result_kind != 'truth':
            raise ValueError('negated query requires a truth-valued grammatical relation')
        return signature

    def execute(self, meaning, context):
        """Execute a selected completed question, retaining its entire proposition."""
        if not isinstance(context, QueryContext):
            raise TypeError('query execution requires a QueryContext')
        _require_query_boundary(context)
        if _concept_space(context) is not self.space:
            raise ValueError('query context belongs to a different conceptual space')
        if not isinstance(meaning, ConceptualMeaning):
            raise TypeError('query requires a complete ConceptualMeaning')
        if meaning.roles.shape[-1] != int(self.space.outputShape[-1]):
            raise ValueError('query meaning must have the full conceptual width')
        try:
            signature = self.signature_for(meaning, work=context.work)
        except QueryWorkExhausted:
            evidence = _work_exhausted()
            return dict(evidence, evidence=evidence, meaning=meaning,
                        query_signature=None, resolution_records_scanned=0)
        arguments, resolved = [], 0
        try:
            for role, kind in zip(signature.argument_roles, signature.argument_kinds):
                reference = meaning.role_refs[role]
                if kind == 'description':
                    value, scanned = _occurrence_description(context, reference)
                    resolved += scanned
                elif reference is not None:
                    if context.work is not None:
                        context.work.require("reference")
                    _existing_row(self.space, reference)
                    value = reference
                elif kind == 'concept':
                    value = meaning.roles[role]
                else:
                    raise ValueError('query operand requires a grounded conceptual reference')
                arguments.append(value)
            evidence = signature.invoke(context, *arguments)
        except QueryWorkExhausted:
            evidence = dict(_work_exhausted(), result_kind=signature.result_kind,
                            evidence_kind=signature.evidence_kind,
                            semantic_id=signature.semantic_id,
                            domain=signature.domain)
        result = dict(evidence, evidence=evidence, meaning=meaning,
                      query_signature=signature.name, resolution_records_scanned=resolved)
        if not meaning.polarity:
            result['support_true'], result['support_false'] = (
                evidence.get('support_false', 0.0), evidence.get('support_true', 0.0))
        return result


class GrammaticalThoughtRegistry:
    """Join canonical executors to the grammar-owned thought-operation tuple.

    This registry owns neither an alias menu nor a second structural catalogue.
    A descriptor without a declared structural family is absent; a structural
    family without a descriptor remains pure grammar.  Native VP identities
    stay checkpointed named concepts on the existing conceptual-space owner.
    """

    _NAME_PREFIX = 'grammatical-vp:'

    def __init__(self, space, grammar):
        self.space = space
        operations = tuple(getattr(grammar, 'thought_operations', ()))
        seen = set()
        for operation in operations:
            semantic_id = getattr(operation, 'semantic_id', None)
            if not isinstance(semantic_id, str) or semantic_id in seen:
                raise ValueError('grammar thought-operation catalogue is malformed')
            seen.add(semantic_id)
        self.operations = operations
        self._operations_by_id = {
            operation.semantic_id: operation for operation in operations}
        forms = {}
        for operation in operations:
            declared_forms = tuple(getattr(operation, 'forms', ()))
            if not declared_forms:
                raise ValueError(
                    f'thought operation {operation.semantic_id!r} has no structural form')
            for form in declared_forms:
                structural_id = getattr(form, 'structural_id', None)
                if not isinstance(structural_id, str) or structural_id in forms:
                    raise ValueError('grammar thought-operation forms are malformed')
                if (tuple(getattr(form, 'operand_roles', ()))
                        != tuple(operation.operand_roles)
                        or getattr(form, 'result_role', None) != operation.result_role
                        or not isinstance(getattr(form, 'permutation', ()), tuple)
                        or set(getattr(form, 'permutation', ()))
                        != set(operation.operand_roles)):
                    raise ValueError('grammar thought-operation form disagrees with its family')
                forms[structural_id] = (operation, form)
        self._forms_by_id = forms
        descriptors = {}
        for operation in operations:
            descriptor = THOUGHT_EXECUTORS.get(operation.semantic_id)
            if descriptor is None:
                continue
            if len(descriptor.argument_kinds) != len(operation.operand_roles):
                raise ValueError(
                    f'thought executor {operation.semantic_id!r} has an arity mismatch '
                    'with its structural role contract')
            descriptors[operation.semantic_id] = descriptor
        self.descriptors = descriptors
        self.executable_operation_ids = tuple(descriptors)
        self.identities = tuple(
            (descriptor.domain, semantic_id)
            for semantic_id, descriptor in descriptors.items())
        # Installation may find that a deliberately tiny ConceptualSpace cannot
        # reserve every native VP.  Keep the full structural catalogue intact,
        # but expose only the all-or-nothing setup-time subset that already has
        # a valid native binding.  No boundary call may mint a deferred VP.
        self._declared_identities = self.identities
        self.unavailable_operation_ids = ()
        self.unavailable_reason = None

    @classmethod
    def install(cls, space, grammar):
        """Bind executable operations once without partial capacity allocation.

        A small structural model may have fewer concept rows than the grammar's
        executable thought faces.  It must still construct and compose its
        grammar.  Preflight every *missing* VP as one group; if that group does
        not fit, retain only already checkpointed bindings and mark the other
        structural families unavailable for thought.  This is a setup-time
        capability decision, never a lazy allocation or an arbitrary
        declaration-order prefix.
        """
        registry = cls(space, grammar)
        bound, missing = [], []
        names = getattr(space, '_frozen_named', {})
        for identity in registry._declared_identities:
            if registry._name(identity) in names:
                # A checkpointed binding is authoritative but still must name
                # an active aligned native row before it becomes executable.
                registry._reference(identity)
                bound.append(identity)
            else:
                missing.append(identity)
        if missing:
            try:
                registry._preflight_missing_vps(len(missing))
            except RuntimeError as error:
                if 'concept inventory exhausted' not in str(error):
                    raise
                registry._limit_to_bound_identities(
                    bound, unavailable=missing, reason=str(error))
                return registry
            for identity in missing:
                space.mint_frozen_concept(registry._name(identity))
                registry._reference(identity)
            bound.extend(missing)
        registry._limit_to_bound_identities(bound)
        return registry

    def _preflight_missing_vps(self, count):
        """Check both ID and order-zero snap capacity before any VP mint.

        ``ConceptualSpace.nVectors`` is the physical inventory, while a
        non-serial grammar can reserve only its order-zero snap prefix for a
        freshly minted named VP.  The generic ID preflight correctly protects
        allocator IDs but cannot infer that all of these VPs request the snap
        region.  Count its occupied rows here so setup never mints a valid ID
        that has no aligned payload row.
        """
        count = int(count)
        caps_method = getattr(self.space, '_order_caps', None)
        rows = getattr(self.space, '_csw_rows', None)
        if callable(caps_method) and isinstance(rows, dict):
            caps = tuple(caps_method())
            if len(caps) > 1:
                snap_capacity = int(caps[0])
                occupied = {
                    int(row) for row in rows.values()
                    if 0 <= int(row) < snap_capacity}
                if len(occupied) + count > snap_capacity:
                    raise RuntimeError(
                        'ConceptualSpace concept inventory exhausted while '
                        f'allocating {count} id(s) for grammatical thought VPs: '
                        f'snap block has {len(occupied)} occupied rows and '
                        f'capacity {snap_capacity}. No concept was minted.')
        preflight = getattr(self.space, '_preflight_concept_allocation', None)
        if callable(preflight):
            preflight(count, context='grammatical thought VPs')

    def _limit_to_bound_identities(self, identities, *, unavailable=(), reason=None):
        """Expose only setup-time native VP bindings, preserving grammar order."""
        identities = tuple(identities)
        allowed = {semantic_id for _domain, semantic_id in identities}
        self.identities = identities
        self.descriptors = {
            semantic_id: descriptor
            for semantic_id, descriptor in self.descriptors.items()
            if semantic_id in allowed}
        self.executable_operation_ids = tuple(
            semantic_id for semantic_id in self.executable_operation_ids
            if semantic_id in allowed)
        self.unavailable_operation_ids = tuple(
            semantic_id for _domain, semantic_id in unavailable)
        self.unavailable_reason = reason

    @classmethod
    def _name(cls, identity):
        return cls._NAME_PREFIX + ':'.join(identity)

    def _reference(self, identity, *, work=None):
        if identity not in self.identities:
            raise ValueError('grammatical thought operation is not registered')
        identifier = getattr(self.space, '_frozen_named', {}).get(self._name(identity))
        if type(identifier) is not int or identifier < 1:
            raise ValueError('grammatical thought VP binding is unavailable')
        reference = ('sym', identifier)
        if work is not None:
            work.require('reference')
        _existing_row(self.space, reference)
        return reference

    def _operation_form(self, structural_id):
        pair = self._forms_by_id.get(structural_id)
        if pair is None:
            if structural_id in self._operations_by_id:
                # A canonical family always has its own structural form, but
                # retain a precise diagnosis for malformed external metadata.
                raise ValueError(
                    f'thought operation {structural_id!r} has no structural form')
            raise ValueError(
                f'thought operation {structural_id!r} is not registered by this grammar')
        return pair

    def operation_form(self, structural_id):
        """Return the canonical operation and the grammar-spelled form.

        This is structural metadata for program recovery; callers must not use
        it as a second executable catalogue.
        """
        return self._operation_form(structural_id)

    def operation_spec(self, semantic_id):
        """Return a canonical operation even when only a converse face exists.

        ``whole`` may be the only grammar-spelled structural form while its
        family identity remains canonical ``part``.  Boundary recovery has
        already decoded the native VP to that canonical identity, so it must
        not require an additional structural form literally named ``part``.
        """
        pair = self._forms_by_id.get(semantic_id)
        operation = (pair[0] if pair is not None
                     else self._operations_by_id.get(semantic_id))
        if operation is None:
            raise ValueError(
                f'thought operation {semantic_id!r} is not registered by this grammar')
        if operation.semantic_id not in self.descriptors:
            if operation.semantic_id in self.unavailable_operation_ids:
                raise RuntimeError(
                    f'thought operation {semantic_id!r} is unavailable: '
                    f'{self.unavailable_reason}')
            raise ValueError(
                f'structural thought operation {semantic_id!r} has no executor')
        return operation

    def _payload(self, reference, *, work=None):
        if work is not None:
            work.require('payload')
        row = _existing_row(self.space, reference)
        basis = _basis(self.space)
        if not 0 <= row < len(basis):
            raise ValueError('grammatical thought VP or operand payload is unavailable')
        value = basis[row].clone()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('grammatical thought native payload must be finite')
        return value

    @staticmethod
    def _slot_for_operand(role):
        slots = {'I1': 0, 'I2': 2}
        if role not in slots:
            raise ValueError(
                f'thought operation role {role!r} cannot fit the three-role meaning')
        return slots[role]

    def form(self, semantic_id, *arguments, open_roles=(),
             mode='interrogative', polarity=True, bindings=(), scope=(),
             context=None):
        """Form a canonical thought request without executing or allocating.

        Open roles are explicit structural role labels.  Thus a one-argument
        ``part`` call with ``open_roles=('I1',)`` is the former ``parts``
        behavior, but it retains the same semantic identity and VP.
        """
        operation, form = self._operation_form(semantic_id)
        descriptor = self.descriptors.get(operation.semantic_id)
        if descriptor is None:
            if operation.semantic_id in self.unavailable_operation_ids:
                raise RuntimeError(
                    f'thought operation {semantic_id!r} is unavailable: '
                    f'{self.unavailable_reason}')
            raise ValueError(
                f'structural thought operation {semantic_id!r} has no executor')
        surface_roles = tuple(form.operand_roles)
        try:
            open_roles = tuple(open_roles)
        except TypeError as error:
            raise TypeError('thought open roles must be an iterable of role labels') from error
        if (len(set(open_roles)) != len(open_roles)
                or not set(open_roles).issubset(surface_roles)):
            raise ValueError('thought open roles must be unique declared operand roles')
        surface_occupied_roles = tuple(
            role for role in surface_roles if role not in open_roles)
        if len(arguments) != len(surface_occupied_roles):
            raise ValueError(
                f'thought operation {semantic_id!r} requires {len(surface_occupied_roles)} '
                'bound operands for its requested open-role mask')
        supplied = dict(zip(surface_occupied_roles, arguments))
        source_by_canonical_role = dict(zip(operation.operand_roles,
                                            form.permutation))
        canonical_open_roles = tuple(
            role for role in operation.operand_roles
            if source_by_canonical_role[role] in open_roles)
        if canonical_open_roles and not (
                descriptor.executor is _part and len(canonical_open_roles) == 1):
            raise ValueError(
                f'thought operation {semantic_id!r} has no executable open-role form')
        occupied_roles = tuple(
            role for role in operation.operand_roles
            if role not in canonical_open_roles)
        work = getattr(context, 'work', None)
        if work is not None and not isinstance(work, QueryWorkBudget):
            raise TypeError('thought context must carry one QueryWorkBudget')
        identity = (descriptor.domain, operation.semantic_id)
        vp = self._reference(identity, work=work)
        vp_payload = self._payload(vp, work=work)
        payloads = [torch.zeros_like(vp_payload), vp_payload,
                    torch.zeros_like(vp_payload)]
        references = [None, vp, None]
        kinds_by_role = dict(zip(operation.operand_roles,
                                 descriptor.argument_kinds))
        preparation_context = (
            _descriptor_context(context, descriptor)
            if isinstance(context, ThoughtGrammarContext) else context)
        for role in occupied_roles:
            argument = supplied[source_by_canonical_role[role]]
            kind = kinds_by_role[role]
            if kind == 'description':
                if context is None:
                    raise TypeError('description thought operands require a ThoughtGrammarContext')
                description, _ = _occurrence_description(
                    preparation_context, argument)
                value = description.roles.sum(0) / description.role_mask.sum().sqrt()
                references[self._slot_for_operand(role)] = argument
            else:
                _validate_argument(argument, kind)
                if isinstance(argument, tuple):
                    value = self._payload(argument, work=work)
                    references[self._slot_for_operand(role)] = argument
                else:
                    value = argument
            if value.shape != vp_payload.shape:
                raise ValueError('thought operand must have the full conceptual width')
            payloads[self._slot_for_operand(role)] = value.to(vp_payload)
        mask = torch.tensor(
            [bool('I1' in occupied_roles), True, bool('I2' in occupied_roles)],
            dtype=torch.bool, device=vp_payload.device)
        return ConceptualMeaning(torch.stack(payloads), mask, mode=mode,
                                 polarity=polarity,
                                 role_refs=tuple(references),
                                 bindings=bindings, scope=scope)

    @staticmethod
    def _is_description_reference(reference):
        """Whether metadata names an existing owned occurrence, without reading it."""
        if not isinstance(reference, tuple) or not reference:
            return False
        if reference[0] == 'thought':
            return True
        return (len(reference) == 3 and reference[0] == 'ltm'
                and isinstance(reference[1], str)
                and type(reference[2]) is int and reference[2] >= 0)

    def _candidate_operand(self, sources, role, kind):
        """Pick one already-held operand without executing a reader.

        The order is current candidate, active frame, then root.  It is an
        ownership preference, not a learned position feature.  The returned
        full-width value keeps the selected signed operand visible to policy;
        execution later resolves typed references or descriptions through the
        descriptor-scoped context.
        """
        slot = self._slot_for_operand(role)
        for source in sources:
            if not bool(source.role_mask[slot]):
                continue
            value, reference = source.roles[slot], source.role_refs[slot]
            if kind == 'reference':
                try:
                    concept_reference(reference)
                except (TypeError, ValueError):
                    continue
            elif kind == 'description':
                if not self._is_description_reference(reference):
                    continue
            elif kind != 'concept':  # pragma: no cover - descriptor validation
                raise AssertionError(f'unknown thought operand kind {kind!r}')
            if (not torch.is_tensor(value) or value.ndim != 1
                    or not bool(torch.isfinite(value).all())):
                continue
            return value, reference
        return None

    def _form_candidate(self, operation, descriptor, bindings, *, source,
                        open_roles=()):
        """Build a request from held values only; no LTM/taxonomy read occurs."""
        open_roles = tuple(open_roles)
        if (len(set(open_roles)) != len(open_roles)
                or not set(open_roles).issubset(operation.operand_roles)):
            raise ValueError('thought candidate has invalid open roles')
        if open_roles and not (
                descriptor.executor is _part and len(open_roles) == 1):
            raise ValueError('thought candidate operation has no open-role executor')
        identity = (descriptor.domain, operation.semantic_id)
        vp = self._reference(identity)
        vp_payload = self._payload(vp)
        payloads = [torch.zeros_like(vp_payload), vp_payload,
                    torch.zeros_like(vp_payload)]
        references = [None, vp, None]
        kinds = dict(zip(operation.operand_roles, descriptor.argument_kinds))
        for role in operation.operand_roles:
            if role in open_roles:
                continue
            value, reference = bindings[role]
            kind = kinds[role]
            if kind == 'reference':
                concept_reference(reference)
            elif kind == 'description':
                if not self._is_description_reference(reference):
                    raise ValueError('thought candidate has no existing description')
            elif kind != 'concept':  # pragma: no cover - descriptor validation
                raise AssertionError(f'unknown thought operand kind {kind!r}')
            if value.shape != vp_payload.shape:
                raise ValueError('thought candidate operand differs from the full width')
            slot = self._slot_for_operand(role)
            payloads[slot] = value.to(vp_payload)
            # A concept value keeps a typed reference as provenance where one
            # exists, but ``execute`` uses its full-width signed value.
            references[slot] = reference
        mask = torch.tensor(
            [bool('I1' in operation.operand_roles and 'I1' not in open_roles),
             True,
             bool('I2' in operation.operand_roles and 'I2' not in open_roles)],
            dtype=torch.bool, device=vp_payload.device)
        return ConceptualMeaning(
            torch.stack(payloads), mask, mode='interrogative', polarity=True,
            role_refs=tuple(references), bindings=source.bindings,
            scope=source.scope)

    def controller_candidates(self, root, active, candidate):
        """Return catalog-derived executable requests without touching readers.

        The completed parse provides values and typed provenance, not a menu
        restriction: every declared executor whose closed role contract can
        be bound from the held root/active/current frame appears here, followed
        by each executable grammar-open form.  The current request is retained
        verbatim first so its exact outer polarity/bindings are not
        reconstructed from a folded root.
        """
        sources = (candidate, active, root)
        if not all(isinstance(item, ConceptualMeaning) for item in sources):
            raise TypeError('thought candidates require complete root/active/current meanings')
        if any(item.mode != 'interrogative' for item in sources):
            raise ValueError('thought candidates require interrogative meanings')
        requests = []
        try:
            signature = self.signature_for(candidate)
        except ValueError:
            signature = None
        if signature is not None:
            requests.append(ThoughtOperationCandidate(
                signature.operation, candidate, signature.open_roles))
        seen = {(item.semantic_id, item.open_roles) for item in requests}

        def append_form(operation, descriptor, open_roles):
            key = (operation.semantic_id, tuple(open_roles))
            if key in seen:
                return
            bindings = {}
            for role, kind in zip(operation.operand_roles,
                                  descriptor.argument_kinds):
                if role in open_roles:
                    continue
                bound = self._candidate_operand(sources, role, kind)
                if bound is None:
                    return
                bindings[role] = bound
            request = self._form_candidate(
                operation, descriptor, bindings, source=candidate,
                open_roles=open_roles)
            requests.append(ThoughtOperationCandidate(
                operation, request, tuple(open_roles)))
            seen.add(key)

        # Closed forms stay in grammar order, so an untrained controller's
        # deterministic first action remains the completed request when it is
        # valid.  Open forms are then available without becoming a second menu.
        for semantic_id in self.executable_operation_ids:
            operation = self._operations_by_id[semantic_id]
            append_form(operation, self.descriptors[semantic_id], ())
        for semantic_id in self.executable_operation_ids:
            operation = self._operations_by_id[semantic_id]
            descriptor = self.descriptors[semantic_id]
            if descriptor.executor is _part:
                for role in operation.operand_roles:
                    append_form(operation, descriptor, (role,))
        return tuple(requests)

    def signature_for(self, meaning, *, work=None, verify_reference=True):
        """Recover one checked operation from a canonical VP and role mask."""
        if (not isinstance(meaning, ConceptualMeaning)
                or meaning.mode != 'interrogative'):
            raise ValueError('thought execution requires a completed interrogative meaning')
        if not bool(meaning.role_mask[1]):
            raise ValueError('thought meaning has no grammatical VP')
        reference = concept_reference(meaning.role_refs[1])
        identity = next((identity for identity in self.identities
                         if getattr(self.space, '_frozen_named', {}).get(
                             self._name(identity)) == reference[1]), None)
        if identity is None:
            raise ValueError('thought VP is not a registered grammatical operation')
        if verify_reference:
            self._reference(identity, work=work)
        semantic_id = identity[1]
        operation = self.operation_spec(semantic_id)
        occupied = tuple(
            role for role in operation.operand_roles
            if bool(meaning.role_mask[self._slot_for_operand(role)]))
        allowed_slots = {1} | {self._slot_for_operand(role)
                              for role in operation.operand_roles}
        if any(bool(meaning.role_mask[index]) and index not in allowed_slots
               for index in range(len(meaning.role_mask))):
            raise ValueError('thought meaning has occupancy outside its structural role contract')
        signature = ThoughtSignature(operation, self.descriptors[semantic_id], occupied)
        if not meaning.polarity and signature.result_kind != 'truth':
            raise ValueError('negated thought requires a truth-valued operation')
        return signature

    def execute(self, meaning, context):
        """Execute one completed request through its descriptor-scoped context."""
        if not isinstance(context, ThoughtGrammarContext):
            raise TypeError('thought execution requires a ThoughtGrammarContext')
        # Forming a description operand may touch LTM.  The completed-row
        # permit therefore belongs at the registry entry point, before VP
        # validation, operand preparation, or the final descriptor call.
        # ``ThoughtSignature.invoke`` repeats this read-only assertion so a
        # direct signature call has the same protection.
        context.require_boundary()
        capability_matches = getattr(context.conceptual_space, 'matches', None)
        if (context.conceptual_space is not self.space
                and not (callable(capability_matches)
                         and bool(capability_matches(self.space)))):
            raise ValueError('thought context belongs to a different conceptual space')
        if not isinstance(meaning, ConceptualMeaning):
            raise TypeError('thought execution requires a complete ConceptualMeaning')
        if meaning.roles.shape[-1] != int(self.space.outputShape[-1]):
            raise ValueError('thought meaning must have the full conceptual width')
        resolved = 0
        try:
            signature = self.signature_for(meaning, work=context.work)
            preparation_context = _descriptor_context(
                context, signature.descriptor)
            arguments = []
            for role, kind in zip(signature.occupied_roles,
                                  signature.argument_kinds):
                slot = self._slot_for_operand(role)
                reference = meaning.role_refs[slot]
                if kind == 'description':
                    value, scanned = _occurrence_description(
                        preparation_context, reference)
                    resolved += scanned
                elif kind == 'concept':
                    # Concept operands are the selected full-width values.
                    # A typed reference remains provenance metadata; routing
                    # through it here would silently discard a signed leaf.
                    value = meaning.roles[slot]
                elif reference is not None:
                    context.work.require('reference')
                    _existing_row(self.space, reference)
                    value = reference
                else:
                    raise ValueError('thought operand requires a grounded conceptual reference')
                arguments.append(value)
            # ``execute`` already admitted this completed row before any
            # preparation/read above.  Avoid a second observable boundary
            # call while retaining :meth:`ThoughtSignature.invoke`'s guard
            # for direct checked calls.
            evidence = signature._invoke_permitted(context, *arguments)
        except QueryWorkExhausted:
            # Result metadata comes from the grammar-owned named VP identity;
            # do not retry its native row lookup after the shared meter has
            # explicitly refused that read.
            signature = self.signature_for(meaning, verify_reference=False)
            evidence = MappingProxyType(dict(
                _work_exhausted(), result_kind=signature.result_kind,
                evidence_kind=signature.descriptor.evidence_kind,
                semantic_id=signature.operation.semantic_id,
                domain=signature.descriptor.domain))
        evidence = _freeze_boundary_value(dict(
            evidence,
            resolution_records_scanned=resolved,
            records_scanned=evidence.get('records_scanned', 0)))
        if not meaning.polarity:
            swapped = dict(evidence)
            swapped['support_true'], swapped['support_false'] = (
                evidence.get('support_false', 0.0), evidence.get('support_true', 0.0))
            evidence = _freeze_boundary_value(swapped)
        return ThoughtResult(
            semantic_id=evidence['semantic_id'], domain=evidence['domain'],
            result_kind=evidence['result_kind'], evidence_kind=evidence['evidence_kind'],
            request=meaning.detached(), evidence=evidence)


def checked_query_declarations(declarations):
    """Validate static grammar declarations; this never parses input sentences."""
    if declarations is None:
        declarations = []
    if isinstance(declarations, str):
        declarations = [declarations]
    if not isinstance(declarations, list):
        raise ValueError('query declarations must be a signature or list of signatures')
    signatures, names = {}, []
    for text in declarations:
        if not isinstance(text, str):
            raise ValueError('query declaration must be a signature string')
        match = re.fullmatch(r'\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z_\w,\s]*)\)\s*', text, re.ASCII)
        if match is None:
            raise ValueError(f'malformed query signature {text!r}')
        name, arguments = match.groups()
        signature = BUILTIN_QUERIES.get(name)
        if signature is None:
            reason = 'deferred' if name == 'tense' else 'missing executor'
            raise ValueError(f'query {name!r}: {reason}')
        variables = [value.strip() for value in arguments.split(',')] if arguments.strip() else []
        if (len(variables) != len(signature.argument_roles)
                or any(re.fullmatch(r'[A-Za-z_]\w*', item, re.ASCII) is None for item in variables)):
            raise ValueError(f'query {name!r} has an invalid argument signature')
        if name in signatures:
            raise ValueError(f'duplicate query signature {name!r}')
        signatures[name] = signature
        names.append(name + '(' + ', '.join(variables) + ')')
    # Each grammar owns its catalog; the individual contracts remain frozen.
    # A mapping proxy here would break the existing grammar/model copy path.
    return names, signatures
