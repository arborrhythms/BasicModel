"""Plan semantic compression from the kept journal; never retain its derivation."""
import torch
from dataclasses import replace
from ClauseRow import Composition
from Meaning import ConceptualMeaning
from Memory import OCCURRENCE, SYMBOL, RELATION, NULL_CONCEPT
from Spaces import _concept_alloc_of


class PartClosing:
    _assertions = frozenset(('part', 'assertPart', 'whole', 'implies', 'equal'))

    def __init__(self, owner, store):
        self.owner, self.store = owner, store
        self.alloc = _concept_alloc_of(owner)
        self.parts = self.alloc.parts
        self._terms, self._planned, self._entries = {}, {}, {}
        self.symbols, self.edges = {}, []
        self.occurrences = set()
        self.relation_dependencies = set()
        self.when = None
        self.content_key = None
        self.source_trust = 1.

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
        if value.operation == 'sign' or value.interpretation == 'no_op':
            record.update(reference=None, identity=None, resolved=True, order=0)
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
        if clause.relation == 'operator':
            record['operator'] = clause.refs[1]
        request = clause.reference_request or {}
        if request.get('sign_leaf') or request.get('no_op_leaf'):
            record['percept_value'] = request.get('percept')
            record.update(reference=None, identity=None, order=0)
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

    def _equality_member(self, reference):
        from ChainRead import envelope, point
        pool = self.owner.mtm
        token = pool.token_of(reference[1] if reference[0] == SYMBOL else envelope(self.store, reference))
        value = pool.value_of(token) if token is not None else point(self.store, reference)
        return token, value

    def _distinct_member(self, member, token, value, witness):
        other, other_token, previous = witness
        if member == other:
            return False
        if token is not None and other_token is not None:
            return token != other_token and not torch.equal(value.detach().to(previous), previous)
        start = self.store.when_start
        if token is None and other_token is None and member[0] == other[0] == OCCURRENCE and start is not None:
            return not torch.equal(value[start-4:start+4].detach().to(previous), previous[start-4:start+4])
        return False

    def observe_members(self, left, right, evidence_refs, words=(None, None)):
        """Both endpoints see the same admitted row, including a promoted token's origin."""
        from util import TheXMLConfig
        threshold = float(TheXMLConfig.space('ConceptualSpace', 'raiseEvidence', 0.))
        if not 0 <= threshold <= 1:
            raise ValueError('raiseEvidence must lie in [0, 1]')
        row = self.parts.find(*evidence_refs)
        if row is None:
            raise ValueError('equality observation requires its durable evidence')
        if left == right or not float(self.parts.truth_plus[row]) > threshold:
            return
        for member, side, word in ((left, right, words[1]), (right, left, words[0])):
            self._admit_member(member, side, word, evidence_refs, threshold)

    def _admit_member(self, member, side, word, evidence_refs, threshold):
        from ConceptIndex import index_part_row, unbound_label
        if (SYMBOL, NULL_CONCEPT) in (member, side):
            return
        order = self.owner._concept_source_order(member[1])
        if order < 0:
            return
        shared_order = self.owner._concept_source_order(side[1])
        if (side[0] == SYMBOL and ('part-base', side[1]) in self.alloc.relate_idx
                and order + 1 == shared_order):
            index_part_row(self.owner, member, side)
            return
        key = side[1] if side[0] == SYMBOL else side
        shared = side
        if side[0] == OCCURRENCE:
            shared_token, _ = self._equality_member(side)
            if shared_token is not None:
                key = shared_token
                shared = (OCCURRENCE, int(shared_token[-1]))
        parent = self.alloc.relate_idx.get(('part-kind', shared, order + 1))
        if parent is not None:
            index_part_row(self.owner, member, (SYMBOL, parent))
            return
        if order != shared_order:
            if not unbound_label(self.owner, side):
                return
            key = (side, order)
        token, value = self._equality_member(member)
        if value is None or (token is None and member[0] != OCCURRENCE):
            return
        # The first seat keeps its original checkpoint key. All additional
        # seats draw on the same fixed witness slab and can be renewed in place.
        empty = None
        members = []
        for i in range(self.alloc.raise_count - 1):
            seat = key if i == 0 else ('witness', key, i)
            witness = self.alloc.singleton_members.get(seat)
            if witness is not None:
                old_refs = self.alloc.singleton_rows.get(seat, (witness[0], side))
                old_row = self.parts.find(*old_refs)
                if (old_row is not None and float(self.parts.truth_plus[old_row]) > threshold
                        and self.owner._concept_source_order(witness[0][1]) == order):
                    if old_refs == evidence_refs or not self._distinct_member(member, token, value, witness):
                        return
                    members.append(witness[0])
                    continue
            if empty is None:
                empty = seat
        if empty is not None:
            self.alloc.remember_singleton(empty, member, token, value, replace=True)
            self.alloc.singleton_rows[empty] = evidence_refs
            return
        parent = index_part_row(self.owner, members[0], shared)
        for admitted in (*members[1:], member):
            if index_part_row(self.owner, admitted, shared) != parent:
                raise RuntimeError('equality witnesses disagree on their shared parent')
        names = (word,) if 'word' in self.alloc.symbol_types.get(word, ()) else ()
        if not names and side[0] == SYMBOL:
            names = ((side[1],) if 'word' in self.alloc.symbol_types.get(side[1], ())
                     else self.owner.definitions.words(side[1]))
        for name in names:
            row = self.parts.find(*evidence_refs)
            self.owner.interpret.define(name, parent, evidence=(float(self.parts.truth_plus[row]), float(self.parts.truth_minus[row])))

    def _observe_equality(self, left, right, record, pair):
        words = tuple(child['value'].word_id for child in record['children'])
        self.observe_members(left, right, (left, right), words)

    def _write_occurrence(self, record, *, pair, sentence, kind, metadata, address,
                          stream, surprise):
        """Write a mention's own idea without replacing its open chain."""
        term, children = record['value'], record['children']
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
            if reference[0] == RELATION:
                self.relation_dependencies.add(reference[1])
        content = term.point
        if 'percept_value' in record or term.operation == 'sign':
            content = record.get('percept_value')
            if content is None:
                content = torch.zeros_like(term.point)
        value = replace(ConceptualMeaning.from_description(self.store._fit(content)),
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
        self.occurrences.add(result[1])
        return result

    def _write(self, record, *, pair=(0., 0.), sentence=False, kind='unverified',
               metadata=None, address=None, stream=-1, surprise=-1., force=False, from_trust=False):
        term = record['value']
        forced_occurrence = force and (term.binding is not None or record.get('mention'))
        # A sentence entry owns its kind, address and time even when its
        # projected head was already written as an embedded working mention.
        # An earlier projection can also have written this term as a chain.
        # That chain cannot stand in for a determined mention's occurrence.
        if (id(term) in self._entries and not sentence
                and not (forced_occurrence and self._entries[id(term)][0] != OCCURRENCE)):
            return self._entries[id(term)]
        children = record['children']
        if metadata is None:
            metadata = record.get('metadata')
        assertion = term.operation in self._assertions and len(children) == 2
        if (not sentence and not force and record['reference'] is not None
                and (not children or term.operation not in (*self._assertions, 'lift', 'verb'))):
            result = record['reference']
        elif (record['resolved'] and not assertion) or not children or forced_occurrence:
            if from_trust:
                pair = tuple(pole * self.source_trust for pole in pair)
            result = self._write_occurrence(record, pair=pair, sentence=sentence, kind=kind,
                metadata=metadata, address=address, stream=stream, surprise=surprise)
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
            pair = tuple(pole * self.source_trust for pole in pair)
            operator = record.get('operator')
            operator = operator if type(operator) is int and operator > 0 else -1
            occupation_owner = getattr(self.store, '_occupation_owner', None)
            occupations = None if occupation_owner is None else occupation_owner()
            grammar_operation = None
            if occupations is not None:
                index = occupations.operation_index(term.formal_name or term.operation)
                if index >= 0:
                    grammar_operation = index
            row = self.parts.add(left, right, truth_plus=pair[0], truth_minus=pair[1],
                when=self.when if sentence else None, surprise=surprise, witness=sentence,
                operator=operator, grammar_operation=grammar_operation)
            self.edges.append(row)
            promoted = False
            if assertion and term.operation in ('equal', 'part', 'assertPart') and pair[0] > 0:
                from FamiliarTier import define_pending
                subject = children[0]['value']
                while len(subject.children) == 1 and subject.operation not in ('sign','bind'):
                    subject = subject.children[0]
                if subject.operation == 'sign' or subject.interpretation == 'no_op':
                    # Commit owns the new occurrence and opens its independent
                    # token before finish_definitions resolves this evidence.
                    other_word = children[1]['value'].word_id
                    self.owner.__dict__.setdefault('_kept_definition_witnesses', []).append(
                        (left, right, None, other_word, True))
                    promoted = True
                elif (term.operation == 'equal' or left[0] != SYMBOL or right[0] != SYMBOL
                      or not (self.owner.interpret.no_object(left[1])
                              and self.owner.interpret.no_object(right[1]))):
                    # Two bare word symbols can stand in a Part relation.
                    # Neither supplies a witnessed object for a definition.
                    word = children[0]['value'].word_id
                    word = None if word in (None, -1, 0) else word
                    promoted = define_pending(self.owner, left, right, word=word)
                    if not promoted:
                        self.owner.__dict__.setdefault('_kept_definition_witnesses', []).append((left, right, word, None, False))
            if assertion and term.operation == 'equal':
                self.alloc.singletons.add(right[1] if right[0] == SYMBOL else right)
                if not promoted:
                    self._observe_equality(left, right, record, pair)
            result = RELATION, row
        self._entries[id(term)] = result
        self.symbols[id(term)] = result[1] if result[0] == SYMBOL else None
        return result

    def close(self, clause, *, pair=(0., 0.), sentence=False, kind='unverified',
              address=None, stream=-1, surprise=-1., force=False, from_trust=False):
        record = self._planned[id(self._terms[id(clause)])]
        result = self._write(record, pair=pair, sentence=sentence, kind=kind,
            metadata=clause.meaning, address=address, stream=stream, surprise=surprise,
            force=force, from_trust=from_trust)
        self.parts.refresh_effective()
        return result
