"""Consume a kept composition into the common native-symbol Part store.

The temporary derivation supplies argument order. Neither that derivation nor
a working identity is retained on an LTM1 occurrence.
"""
import torch

from ClauseRow import Composition
from Layers import assert_cube
from Spaces import _concept_alloc_of


class PartClosing:
    def __init__(self, owner, store):
        self.owner, self.store = owner, store
        self.alloc = _concept_alloc_of(owner)
        self.parts = self.alloc.parts
        self.symbols = {}
        self.edges = []
        self._terms = {}
        self._plan = []
        self._planned = {}
        self._pairs = {}
        self._definitions = {}
        self._occupations = []
        self._individuals = set()
        self._next = int(self.alloc.next_id)
        self._ready = False
        self._committed = False
        self._seated = set()
        self.when = None

    def _point(self, symbol):
        row = self.owner._csw_row_of(int(symbol))
        return None if row is None else self.owner.similarity_codebook.lookup_rows(row).detach()

    @torch.no_grad()
    def _seat(self, symbol, term, types=()):
        if symbol in self._seated:
            return symbol
        self.alloc.symbol_types.setdefault(int(symbol), set()).update(types)
        self.alloc.symbol_types[int(symbol)].add('concept')
        self.alloc.reference_orders.setdefault(int(symbol), int(term.order))
        if int(term.order) == 1:
            self.alloc.individuals.add(int(symbol))
            self.alloc.singletons.add(int(symbol))
        row = self.owner._csw_row_of(int(symbol))
        if row is None:
            self.owner._preflight_concept_row(int(term.order))
            row = self.owner._csw_concept_row(int(term.order), int(symbol))
        if row is None:
            raise RuntimeError('a closing symbol has no native conceptual seat')
        point = assert_cube(term.point.detach())
        bank = self.owner.similarity_codebook
        if point.shape != bank.W[row].shape:
            raise ValueError('a minted composing argument requires its full concept-space value')
        bank.W[row].copy_(point.to(bank.W))
        self.alloc.settle(int(symbol))
        self._seated.add(symbol)
        return int(symbol)

    def _native(self, reference):
        reference = int(reference)
        return reference > 0 and reference in self.alloc.placement and reference not in self.alloc.retired

    def _reserve(self):
        symbol = self._next
        self._next += 1
        return symbol

    def _plan_term(self, value, as_argument):
        pending, active = [(value, as_argument, False)], set()
        while pending:
            term, argument, ready = pending.pop()
            if id(term) in self._planned:
                self._plan_one(term, argument)
                continue
            if ready:
                active.remove(id(term))
                self._plan_one(term, argument)
                continue
            if id(term) in active:
                raise ValueError('recursive composing object')
            active.add(id(term))
            pending.append((term, argument, True))
            dependencies = (*term.children, *((term.binding,) if term.binding is not None else ()))
            pending.extend((child, True, False) for child in reversed(dependencies))
        return self._planned[id(value)]['symbol']

    def _plan_one(self, value, as_argument):
        if id(value) in self._planned:
            record = self._planned[id(value)]
            if as_argument and record['symbol'] is None:
                record['symbol'] = self._reserve()
            record['seat'] |= (as_argument and value.binding is None
                               and self._point(record['symbol']) is None)
            return record['symbol']
        children = tuple(self._plan_term(child, True) for child in value.children)
        reference = (self._plan_term(value.binding, True) if value.binding is not None else
                     value.reference if self._native(value.reference) else None)
        edge, pair, types = None, None, ('object',)
        if not children:
            if reference is None and as_argument:
                reference = self._reserve()
        elif len(children) == 1:
            # A unary face changes the working concept/order. It supplies no
            # ordered pair; any encompassing binary operation mints that value
            # when it actually needs a distinct argument.
            if reference is None and as_argument:
                child = self._point(children[0])
                if child is not None and torch.equal(child.to(value.point), value.point):
                    reference = children[0]
                else:
                    reference = self._reserve()
                types = ('kind',) if value.order > 1 else ('object',)
            child_record = self._planned[id(value.children[0])]
            edge, pair = child_record['edge'], child_record['pair']
        elif len(children) == 2:
            part, whole = children
            if value.operation == 'whole':
                part, whole = whole, part
            pair = (part, whole)
            edge = self._pairs.get(pair, self.parts.find(*pair))
            if edge is None:
                edge = self._reserve()
                self._pairs[pair] = edge
            types = ('situation',) if value.operation in (
                'lift', 'part', 'assertPart', 'whole', 'implies', 'equal') else ('object',)
            if value.operation == 'verb':
                types = (*types, 'state')
                child_types = set(self.alloc.symbol_types.get(children[1], ()))
                for child_record in self._plan:
                    if child_record['symbol'] == children[1]:
                        child_types.update(child_record['types'])
                if 'situation' in child_types:
                    types = (*types, 'speech_individual')
            if reference is None:
                reference = edge
        else:
            raise ValueError('a composition has one or two arguments')
        record = dict(value=value, symbol=reference, edge=edge, pair=pair, types=types,
                      seat=as_argument and value.binding is None and reference is not None
                      and self._point(reference) is None)
        self._planned[id(value)] = record
        self._plan.append(record)
        return reference

    def _definition(self, symbol, word, evidence):
        if symbol is None or not self._native(word) or word == symbol:
            return
        interpreter = getattr(self.owner, 'interpret', None)
        if interpreter is not None and interpreter.no_object(symbol):
            # A selected existing marker is not a newly minted object. An
            # enclosing mint must not redefine that marker as its noun.
            return
        description = self.alloc.word_descriptions.get(word)
        if description is None and interpreter is not None:
            description = interpreter._pending.get(word)
        if description is None:
            raise ValueError('a word-derived mint requires its admitted word description')
        pair = symbol, word
        edge = self._pairs.get(pair, self.parts.find(*pair))
        if edge is None:
            edge = self._reserve()
            self._pairs[pair] = edge
        self._definitions[pair] = edge, description, evidence

    def prepare(self, clause):
        term = self._terms.setdefault(id(clause), self.clause_term(clause))
        pending, visited = [term], set()
        width = self.owner.similarity_codebook.W.shape[-1]
        while pending:
            value = pending.pop()
            if id(value) in visited:
                continue
            visited.add(id(value))
            if value.point.shape != (width,):
                raise ValueError('closing arguments require the full concept-space width')
            assert_cube(value.point)
            pending.extend(value.children)
        minted = clause.minted or bool(clause.reference_request and
                                      clause.reference_request.get('mode') == 'mint')
        symbol = self._plan_term(term, minted)
        if minted:
            self._individuals.add(symbol)
            self._definition(symbol, int(clause.subject_word_id), clause.evidence)
        for value in (self._planned[key] for key in visited):
            if not value['value'].children:
                # A kept word use can bind an existing object. Its word
                # link is testimony too; allocating a new seat is not the
                # condition for admitting that definition.
                self._definition(value['symbol'], value['value'].word_id, clause.evidence)

    def preflight(self):
        ref = self.store.__dict__.get('_occupation_owner')
        occupations = None if ref is None else ref()
        for record in self._plan:
            value = record['value']
            name = value.formal_name or value.operation
            if name is None or occupations is None:
                continue
            for role, child in enumerate(value.children, 1):
                symbol = self._planned[id(child)]['symbol']
                if symbol is not None and symbol != child.word_id:
                    argument = f'{name}_argin{role}'
                    if argument not in occupations.index:
                        raise ValueError(f'undeclared closing occupation: {argument}')
                    self._occupations.append((symbol, argument))
            if record['symbol'] is not None:
                argument = f'{name}_argout1'
                if argument in occupations.index:
                    self._occupations.append((record['symbol'], argument))
        count = self._next - int(self.alloc.next_id)
        self.owner._preflight_concept_allocation(count, context='complete Part closing')
        if self._next > self.parts.capacity:
            raise OverflowError('LTM2 symbol capacity exhausted before closing')
        # Reserve physical seats together. A collection of individually
        # possible admissions must not leave a prefix when their pool fills.
        layer, caps = self.alloc.layer(), self.owner._order_caps()
        occupied = set(layer._tensor_row_keys)
        planned = set()
        for record in self._plan:
            symbol, order = record['symbol'], int(record['value'].order)
            if not record['seat'] or symbol in planned:
                continue
            planned.add(symbol)
            if getattr(self.owner, '_concept_binding', 'mixing') == 'aligned':
                bases = ((0, int(self.owner.nVectors)),)
            elif order == 0:
                bases = ((0, caps[0]), (sum(caps), int(self.owner.nVectors)-sum(caps)))
            else:
                rung = min(order, len(caps)-1)
                bases = ((sum(caps[:rung]), caps[rung]),)
            seat = next((base+i for base, size in bases for i in range(size)
                         if base+i not in occupied), None)
            if seat is None:
                raise RuntimeError(
                    f'complete Part closing exceeds a concept row pool: order={order}, '
                    f'pools={bases}, occupied={len(occupied)}, planned={len(planned)}')
            occupied.add(seat)
        self._ready = True

    @torch.no_grad()
    def _commit(self):
        if not self._ready:
            raise RuntimeError('Part closing must preflight its complete derivation')
        if self._committed:
            return
        # Allocate handles in the order planned, before seating any content.
        start = int(self.alloc.next_id)
        for expected in range(start, self._next):
            if self.owner.new_concept() != expected:
                raise RuntimeError('native symbol allocation changed during closing')
        for record in self._plan:
            value, symbol, edge = record['value'], record['symbol'], record['edge']
            if record['pair'] is not None and len(value.children) == 2:
                self.parts.add(*record['pair'], symbol=edge, when=self.when)
                self.edges.append(edge)
                self.alloc.symbol_types.setdefault(edge, set()).update(record['types'])
                if value.operation == 'equal':
                    # Equality constructs membership in a singleton. The
                    # argument's type supplies that reading on later access.
                    self.alloc.singletons.add(record['pair'][1])
            if record['seat']:
                self._seat(symbol, value, record['types'])
            self.symbols[id(value)] = symbol
        for (symbol, word), (edge, description, evidence) in self._definitions.items():
            self.alloc.symbol_types.setdefault(symbol, set()).add('object')
            self.alloc.symbol_types.setdefault(word, set()).add('word')
            self.alloc.word_descriptions[word] = description
            self.parts.add(symbol, word, symbol=edge, when=self.when,
                           truth_plus=evidence[0], truth_minus=evidence[1], witness=False)
            self.owner.definitions.add(edge)
            interpreter = self.owner.interpret
            interpreter._pending.pop(word, None)
            interpreter._field_pending.discard(word)
        # A selected mint is an individual use at its actual conceptual
        # order; the order number is not the test for having an identity.
        self.alloc.individuals.update(self._individuals)
        self.alloc.singletons.update(self._individuals)
        ref = self.store.__dict__.get('_occupation_owner')
        occupations = None if ref is None else ref()
        if occupations is not None:
            for symbol, argument in self._occupations:
                occupations.add(symbol, argument)
        self._committed = True

    def clause_term(self, clause):
        if clause.composition is not None:
            return clause.composition
        # Direct callers supply their already-completed operands. They do not
        # get a fabricated derivation of an absolute point.
        if clause.relation is None:
            return Composition(clause.point, order=clause.order)
        terms = []
        for role in (0, 2):
            ref = clause.refs[role]
            if isinstance(ref, tuple):
                terms.append(self.clause_term(clause.children[ref[1]]))
            else:
                point = self._point(ref) if self._native(ref) else clause.meaning.roles[role]
                terms.append(Composition(point, reference=ref, order=clause.order))
        # Direct relation construction names its resulting native concept by
        # the encompassing argument, exactly the Part operator's point face.
        return Composition(terms[1].point, tuple(terms), operation=clause.relation,
                           order=clause.order)

    def close(self, clause, *, pair=(0., 0.), when=None, surprise=-1.):
        self._commit()
        term = self._terms[id(clause)]
        record = self._planned[id(term)]
        if record['edge'] is not None:
            self.parts.add(*record['pair'], truth_plus=pair[0], truth_minus=pair[1],
                           when=when, surprise=surprise, witness=False)
        return record['edge'] if clause.relation is not None else self.symbols[id(term)]
