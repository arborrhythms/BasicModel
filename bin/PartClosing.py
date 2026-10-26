"""Plan semantic compression from the kept journal; never retain its derivation."""
import torch
from dataclasses import replace
from ClauseRow import Composition
from Meaning import ConceptualMeaning
from LTM1 import OCCURRENCE, SYMBOL, RELATION
from Spaces import _concept_alloc_of


class PartClosing:
    _assertions = frozenset(('part', 'assertPart', 'whole', 'implies', 'equal'))

    def __init__(self, owner, store):
        self.owner, self.store = owner, store
        self.alloc = _concept_alloc_of(owner)
        self.parts = self.alloc.parts
        self._terms, self._planned, self._entries = {}, {}, {}
        self.symbols, self.edges = {}, []
        self.when = None
        self.content_key = None

    def _point(self, symbol):
        if symbol in (None, -1, 0):
            return None
        row = self.owner._csw_row_of(int(symbol))
        return None if row is None else self.owner.similarity_codebook.lookup_rows(row).detach()

    def _reference(self, value):
        reference = value.reference
        if reference in (-1, 0, None):
            reference = value.word_id
        if reference in (-1, 0, None):
            return None
        location = self.store.location(reference)
        if isinstance(location, int):
            return OCCURRENCE, int(self.store.address_keys[location])
        if isinstance(location, tuple):
            return RELATION, location[1]
        if reference > 0 and (self._point(reference) is not None or reference in self.alloc.symbol_types):
            return SYMBOL, reference
        return None

    def _plan_term(self, value, as_argument=False):
        """A pure journal walk: no argument mint, definition or composing edge."""
        if id(value) in self._planned:
            return self._planned[id(value)]
        children = tuple(self._plan_term(child, True) for child in value.children)
        reference = self._reference(value)
        if children and value.operation not in (*self._assertions, 'lift', 'verb'):
            if value.head_role:
                children = (children[value.head_role-1],)
            if len(children) == 1 and reference is None:
                reference = children[0]['reference']
        if value.binding is not None:
            children = (self._plan_term(value.binding, True),)
            reference = children[0]['reference']
        selected_bind = value.operation == 'bind' and reference is not None
        if children and not selected_bind:
            resolved = all(child['resolved'] for child in children)
        else:
            resolved = reference is not None and reference[0] == SYMBOL
            if reference is not None and reference[0] == OCCURRENCE:
                token = self.owner.mtm.token_of(('ltm', self.store.address_domain, reference[1]))
                resolved = token is not None and not self.owner.mtm.entries[token].pending
                if token is None:
                    row = self.store.index_of_row(reference[1])
                    marks = dict(self.store.raw_meaning_of(row).bindings)
                    resolved = not bool(marks.get('_open_references') or marks.get('_forward_references'))
            if resolved and reference[0] == SYMBOL:
                # A bare, uninterpreted word is a name, not its own object.
                resolved = not self.owner.interpret.no_object(reference[1]) and not (
                    'word' in self.alloc.symbol_types.get(reference[1], ()) and self._point(reference[1]) is None)
        identity = reference
        if value.operation in self._assertions and len(children) == 2:
            # A semantic pair has references and evidence, not an idea value.
            # Its containing pair therefore remains a chain as well.
            resolved = False
        if children and not selected_bind:
            child = children[-1 if value.operation == 'verb' else 0]
            identity = child.get('identity') or child['reference']
        record = dict(value=value, reference=reference, identity=identity,
                      children=children, resolved=resolved)
        self._planned[id(value)] = record
        return record

    def clause_term(self, clause):
        if clause.composition is not None:
            return clause.composition
        if clause.relation is None:
            reference = clause.refs[0]
            return Composition(clause.point, order=clause.order,
                reference=reference if type(reference) is int else -1)
        terms = []
        for role in (0, 2):
            ref = clause.refs[role]
            if isinstance(ref, tuple):
                child = clause.children[ref[1]]
                terms.append(self._terms.get(id(child)) or self.clause_term(child))
            else:
                point = self._point(ref)
                terms.append(Composition(clause.meaning.roles[role] if point is None else point,
                                         reference=ref, order=clause.order))
        return Composition(terms[1].point, tuple(terms), operation=clause.relation, order=clause.order)

    def prepare(self, clause):
        term = self._terms.setdefault(id(clause), self.clause_term(clause))
        record = self._plan_term(term)
        # A direct absolute field is already a completed idea, unless its
        # semantic binding explicitly declares a pending name or open slot.
        record['order'] = clause.order
        record['metadata'] = clause.meaning
        record['mention'] = clause.reference_request is not None
        marks = dict(clause.meaning.bindings)
        if clause.composition is None and clause.relation is None:
            record['resolved'] = not bool(marks.get('_forward_references') or marks.get('_open_references'))
        if clause.minted or (clause.reference_request or {}).get('mode') == 'mint':
            record['resolved'] = True
        if marks.get('_forward_references') or any(k != 'evidence' for k, _ in marks.get('_open_references', ())):
            record['resolved'] = False
        return record

    def preflight(self):
        # At worst every binary semantic term is a relation and every leaf or
        # compressed child an occurrence. Reserve the whole transaction first.
        relations = sum(len(r['children']) == 2 and (not r['resolved'] or r['value'].operation in self._assertions)
                        for r in self._planned.values())
        if int(self.alloc.next_id) + relations > self.parts.capacity:
            raise OverflowError('LTM1 construction capacity exhausted before close')
        self.owner._preflight_concept_allocation(0, context='semantic closing')

    def prediction_point(self, record):
        """One entry value: the compressed idea or the chain's head idea."""
        if record['resolved'] and record['value'].operation not in self._assertions:
            return record['value'].point
        for child in record['children']:
            point = self.prediction_point(child)
            if point is not None:
                return point
        reference = record['reference']
        if reference is not None:
            from ChainRead import point, head_idea
            return head_idea(self.store, reference[1]) if reference[0] == RELATION else point(self.store, reference)
        return record['value'].point if not record['children'] else None

    def _write(self, record, *, pair=(0., 0.), sentence=False, kind='unverified',
               metadata=None, address=None, stream=-1, surprise=-1., force=False, from_trust=False):
        term = record['value']
        # A sentence entry owns its kind, address and time even when its
        # projected head was already written as an embedded working mention.
        if id(term) in self._entries and not sentence:
            return self._entries[id(term)]
        children = record['children']
        if metadata is None:
            metadata = record.get('metadata')
        assertion = term.operation in self._assertions and len(children) == 2
        if (not sentence and not force and record['reference'] is not None
                and (not children or term.operation not in (*self._assertions, 'lift', 'verb'))):
            result = record['reference']
        elif (record['resolved'] and not assertion) or not children or (force and (term.binding is not None or record.get('mention'))):
            marks = {} if metadata is None else dict(metadata.bindings)
            reference = record['identity']
            if term.binding is not None:
                reference = self._entries.get(id(term.binding), reference)
            if term.operation in ('lift', 'verb') and children:
                child = children[0 if term.operation == 'lift' else -1]
                reference = child['reference']
                if reference is None or reference[0] != OCCURRENCE:
                    reference = self._entries.get(id(child['value'])) or reference
            if reference is not None and not (reference[0] == SYMBOL and self.owner.interpret.no_object(reference[1])):
                from ChainRead import envelope
                marks.setdefault('_identity', envelope(self.store, reference))
            value = replace(ConceptualMeaning.from_description(self.store._fit(term.point)),
                bindings=marks, scope=() if metadata is None else metadata.scope,
                polarity=True if metadata is None else metadata.polarity)
            if address is None:
                from Occurrence import native_content_key
                address = dict(document_key=('operand', self.content_key),
                    sentence_index=list(self._planned).index(id(term)), content_key=native_content_key(value))
            row = self.store.append_meaning(value, kind=kind, evidence=pair, order=record.get('order', term.order),
                stream=stream, surprise=surprise,
                when=self.when if sentence else False, **(address or {}))
            result = OCCURRENCE, int(self.store.address_keys[row])
        elif len(children) == 1:
            result = self._write(children[0], pair=pair, sentence=sentence, kind=kind,
                                 metadata=metadata, address=address, stream=stream, surprise=surprise,
                                 from_trust=from_trust)
        else:
            left, right = (self._write(child, stream=stream) for child in children)
            if term.operation == 'whole':
                left, right = right, left
            if from_trust and metadata is not None and not metadata.polarity:
                pair = pair[::-1]
            row = self.parts.add(left, right, truth_plus=pair[0], truth_minus=pair[1],
                when=self.when if sentence else None, surprise=surprise, witness=sentence)
            self.edges.append(row)
            if assertion and term.operation == 'equal':
                self.alloc.singletons.add(right[1] if right[0] == SYMBOL else right)
            result = RELATION, row
        self._entries[id(term)] = result
        self.symbols[id(term)] = result[1] if result[0] == SYMBOL else None
        return result

    def close(self, clause, *, pair=(0., 0.), sentence=False, kind='unverified',
              address=None, stream=-1, surprise=-1., force=False, from_trust=False):
        record = self._planned[id(self._terms[id(clause)])]
        return self._write(record, pair=pair, sentence=sentence, kind=kind,
            metadata=clause.meaning, address=address, stream=stream, surprise=surprise,
            force=force, from_trust=from_trust)
