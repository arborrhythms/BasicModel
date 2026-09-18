"""Checked boundary-query contracts; no semantic memory or learned parameters.

Relation identities link grammatical compose/inverse faces and query interfaces.
These immutable definitions do not create a second VP embedding or a tool-only
semantic store. Native references are addresses; numerical payloads stay in CS.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable
import re

import torch

from Meaning import ConceptualMeaning
from QueryWork import QueryWorkBudget, QueryWorkExhausted, capture_limits
from Taxonomy import capture_taxonomy, concept_reference


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
    return getattr(context.reasoner.model, 'conceptualSpace', None)


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


def _vector(context, value):
    if torch.is_tensor(value):
        return value
    if context.work is not None:
        context.work.require("payload")
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
    keyword = {} if context.work is None else {"work": context.work}
    return context.reasoner.existence_evidence(
        arguments[0], max_records=context.max_records, **keyword)


def _part(context, arguments):
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
    left, right = (_vector(context, arguments[role]) for role in (0, 2))
    if left.shape != right.shape:
        raise ValueError('query equality requires equal full concept width')
    score = context.reasoner.equal(left, right)
    return {'support_true': score, 'support_false': 0.0, 'candidates': []}


def _lookup(context, arguments):
    """The distinct two-operand LTM read; retrieval never admits a fact."""
    left, right = (_vector(context, arguments[role]) for role in (0, 2))
    if left.shape != right.shape:
        raise ValueError('query lookup requires equal full concept width')
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
    space = _concept_space(context)
    if isinstance(arguments[0], tuple):
        value = _vector(context, arguments[0])
        return {'value': value, 'reference': arguments[0], 'nodes_scanned': 1,
                'incomplete': ()}
    vector = arguments[0]
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
    if arguments[0].mode != 'interrogative':
        raise ValueError('query what requires an interrogative conceptual question')
    if not callable(context.schedule_subgoal):
        raise RuntimeError('query what requires the active boundary controller')
    return {'value': context.schedule_subgoal(arguments[0])}


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
