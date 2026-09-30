"""Prepare a reviewable Z repair away from the still-frozen baseline tree.

This writes proposed source only. Application waits for all baseline trials.
"""
import ast
from pathlib import Path
R=Path.cwd(); H=Path(__file__).resolve().parent

def write(name, text):
    ast.parse(text)
    p=H/'z-proposal'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)

def change(text, before, after):
    assert text.count(before)==1, before[:100]
    return text.replace(before,after,1)

s=(R/'bin/ClauseRow.py').read_text()
a=s.index('@dataclass(frozen=True, eq=False)\nclass ClauseConcept:');b=s.index('\n\n@dataclass',a+1)
s=s[:a]+'''def predicate_identity(name):
    """A stable slot address for a grammar operation, never an inventory row.

    Only occurrences in the store give this address content. Its grammar name
    ties those occurrences together without looking at their vector values.
    The tag is disjoint from concept IDs and occurrence-addressed LTM rows.
    """
    import hashlib
    name = 'part' if name in ('whole', 'equal', 'generic') else str(name)
    return (1 << 61) | (int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'big') & ((1 << 61) - 1))


def predicate_relation(reference):
    for name in ('part', 'implies'):
        if reference == predicate_identity(name):
            return name
    return 'operator'


@dataclass(frozen=True, eq=False)
class ClausePredicate:
    """An open reading's predicate value and identity, awaiting its row slot."""
    point: Any = field(repr=False)
    name: str

    def __post_init__(self):
        if (not torch.is_tensor(self.point) or self.point.ndim != 1
                or not bool(torch.isfinite(self.point).all())):
            raise ValueError('a clause predicate requires one finite grammar value')
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('a clause predicate requires its grammar operation name')

    @property
    def identity(self):
        return predicate_identity(self.name)
''' + s[b:]
s=s.replace('isinstance(reference, ClauseConcept)', 'isinstance(reference, ClausePredicate)')
a=s.index('    def allocate(point):');b=s.index('\n    def concept_point',a)
s=s[:a]+'''    store_ref = weakref.ref(store)
    object.__setattr__(space, '_clause_store_ref', store_ref)

    def allocate(point):
        # As for DEF, the durable address is the occurrence about to be
        # written. A closed S is not an extra concept-inventory record.
        return (1 << 62) + int(store_ref()._next_occurrence)
''' +s[b:]
a=s.index('    def predicate_kind(reference):');b=s.index('\n    store.configure_clause_index',a)
s=s[:a]+'''    def predicate_kind(reference):
        kind = predicate_relation(reference)
        if kind != 'operator':
            return kind
        # Existing explicitly supplied thought predicates can still name a
        # part relation. Reading their identity does not request an executor.
        current = model_ref()
        registry = getattr(current, 'grammatical_thoughts', None)
        names = getattr(space_ref(), '_frozen_named', {})
        if registry is not None:
            for domain, operation in registry._declared_identities:
                if operation == 'part' and names.get(registry._name((domain, operation))) == reference:
                    return 'part'
        return 'operator'
''' +s[b:]
s=s.replace("context='clause endings'", "context='clause taxonomy'")
s=change(s,"        reference = field.refs[1]\n        result = resolve(reference)", "        reference = field.refs[1]\n        if isinstance(reference, ClausePredicate):\n            return predicate_relation(reference.identity)\n        result = resolve(reference)")
a=s.index('    def index_of_row(self, reference):');b=s.index('\n    def point_of_row',a)
s=s[:a]+'''    def _native_reference_indexes(self):
        """Derived identity-to-occurrence caches, invalidated by row edits."""
        count = len(self)
        token = (count, id(self.row_ids), self.row_ids._version,
                 id(self.refs), self.refs._version, self.rel_type._version)
        if self.__dict__.get('_native_reference_token') != token:
            rows, predicates = {}, {}
            for row, identity in enumerate(self.row_ids[:count].tolist()):
                if identity > 0:
                    rows.setdefault(identity, row)
            for row, references in enumerate(self.refs[:count].tolist()):
                for slot, identity in enumerate(references):
                    if (1 << 61) <= identity < (1 << 62):
                        predicates.setdefault(identity, (row, slot))
            object.__setattr__(self, '_native_row_index', rows)
            object.__setattr__(self, '_native_predicate_index', predicates)
            object.__setattr__(self, '_native_reference_token', token)
        return self._native_row_index, self._native_predicate_index

    def index_of_row(self, reference):
        return self._native_reference_indexes()[0].get(int(reference))
''' +s[b:]
s=change(s,"        resolve = getattr(self, '_concept_point', None)","        occurrence = self._native_reference_indexes()[1].get(int(reference))\n        if occurrence is not None:\n            return self.slots[occurrence]\n        resolve = getattr(self, '_concept_point', None)")
s=s.replace('isinstance(ref, ClauseConcept)', 'isinstance(ref, ClausePredicate)')
s=s.replace("raise ValueError('phrase admission requires the native concept index')", "raise ValueError('predicate identity requires the clause index')")
s=change(s,"        if check is not None:\n            check(identities + (0 if index_plan is None else index_plan.count))", "        if check is not None and index_plan is not None and index_plan.count:\n            check(index_plan.count)")
write('bin/ClauseRow.py',s)

s=(R/'bin/ClauseJournal.py').read_text().replace('from ClauseRow import Clause, ClauseConcept','from ClauseRow import Clause, ClausePredicate, predicate_identity')
a=s.index('        members = tuple(dict.fromkeys',s.index('    def concept(node):'));b=s.index('\n    def is_clause',a)
s=s[:a]+'''        # A composite phrase has no concept-inventory identity. When a
        # relative row needs its address, operand() writes its ended point.
        return -1

    def operation_concept(node, relation=None):
        """The selected grammar predicate lives only in its row occurrences."""
        from References import symbol_code
        identity = relation if relation is not None else name(node)
        code = predicate_identity(identity)
        point = symbol_code(code, program.leaves.shape[-1], n_where=0, n_when=0).to(program.leaves)
        return ClausePredicate(point, identity)
''' +s[b:]
s=change(s,"""            if relation is not None and registry is not None:
                try:
                    native = registry.clause_reference(relation)
                    predicate, predicate_ref = registry._payload(native), int(native[1])
                except (ValueError, KeyError):
                    pass
""", """            if relation is not None:
                predicate_ref = operation_concept(node, relation)
                predicate = predicate_ref.point
""")
write('bin/ClauseJournal.py',s)

s=(R/'bin/ConceptIndex.py').read_text()
s=s.replace('        self.terms = {}\n','')
s=s.replace('from ClauseRow import ClauseConcept','from ClauseRow import ClausePredicate')
s=s.replace('isinstance(ref, ClauseConcept)', 'isinstance(ref, ClausePredicate)')
s=change(s,'        self.validate_rows()','''        if not self.validate_rows():
            # Optional symbolization is all-or-nothing. The completed field
            # still writes its actual references and the reading goes on.
            self.definitions, self.edges, self.keys, self.forms = {}, set(), {}, []
            self.points = {cid: point for cid, point in self.points.items()
                           if (1 << 61) <= cid < (1 << 62)}''')
s=change(s,"        layer = alloc.layer(0)\n        trial = SimpleNamespace", "        if self.definitions:\n            _, end, capacity = cs._concept_capacity_window(len(self.definitions))\n            if capacity is not None and end > capacity:\n                return False\n        layer = alloc.layer(0)\n        trial = SimpleNamespace")
s=change(s,"                raise RuntimeError('native conceptual row capacity exhausted before clause admission')", "                return False\n        return True")
a=s.index('    def term(self, term):');b=s.index('\n    def symbol_at',a)
s=s[:a]+'''    def term(self, term):
        """Retain a predicate's live slot value until its occurrence is written."""
        cid = term.identity
        self.points[cid] = term.point
        return cid
''' +s[b:]
s=s.replace("            for member in (source if cid in self.terms else (source,)):", "            for member in (source,):")
s=change(s,"                if cid in self.terms and self.terms[cid].key[0] == 'grammar-predicate':\n                    cs.__dict__.setdefault('_frozen_concepts', set()).add(cid)\n",'')
write('bin/ConceptIndex.py',s)

s=(R/'bin/Spaces.py').read_text()
s=change(s,"        return _concept_alloc_of(self).order_of(concept_id, _seen)",'''        alloc = _concept_alloc_of(self)
        if int(concept_id) not in alloc.placement:
            store = self._closed_clause_store()
            row = None if store is None else store.index_of_row(concept_id)
            if row is not None:
                # Preserve the particular-reference order; the row's own
                # abstraction stamp separately governs decoding its point.
                return int(int(store.rel_type[row]) == store.REL_NONE)
        return alloc.order_of(concept_id, _seen)''')
s=change(s,"        return tuple(sorted(cid for cid in values if cid in alloc.placement and cid not in alloc.retired))",'''        store = self._closed_clause_store()
        return tuple(sorted(cid for cid in values if cid not in alloc.retired and
                            (cid in alloc.placement or
                             (store is not None and store.index_of_row(cid) is not None))))''')
s=change(s,'    def _definition_index(self):','''    def _closed_clause_store(self):
        model = getattr(self, '_model', None)
        symbol = getattr(model, 'symbolSpace', None) or getattr(self, 'symbolSpace', None)
        store = getattr(symbol, 'ltm_store', None)
        if store is not None:
            return store
        reference = self.__dict__.get('_clause_store_ref')
        return None if reference is None else reference()

    def _definition_index(self):''')
write('bin/Spaces.py',s)
print('Prepared four syntax-checked proposed files; active baseline source is unchanged.')
