"""Derived definition indexes over LTM1's Part(object, word) edges.

Word and object IDs address the indexed relation rows. Native text queries
first resolve a word through its percept. Appends update the index, and
load and compaction rebuild it from the surviving rows.
"""
import weakref

import torch


def form_key(value):
    return value.decode('utf-8', errors='surrogateescape') if isinstance(value, bytes) else value


class DefinitionIndex:
    def __init__(self, store, allocator):
        self._store = weakref.ref(store)
        self._allocator = weakref.ref(allocator)
        self._revision = None
        self.rebuild()

    def __deepcopy__(self, memo):
        import copy
        result = object.__new__(type(self))
        memo[id(self)] = result
        store = copy.deepcopy(self._store(), memo)
        result._store = weakref.ref(store)
        for name, value in self.__dict__.items():
            if name not in ('_store', '_allocator'):
                setattr(result, name, copy.deepcopy(value, memo))
        result._allocator = weakref.ref(copy.deepcopy(self._allocator(), memo))
        return result

    def _ensure(self):
        if self._revision != self._store().effective_revision:
            self.rebuild()

    def rebuild(self):
        self._forms, self._units, self._objects, self._words, self._rows = {}, {}, {}, {}, {}
        self._object_table = None
        self._descriptions = {}
        self._identities = {}
        store = self._store()
        store.refresh_effective()
        # Only an explicitly named pair can define a word. Look up those
        # typed endpoints in LTM1, preserving the physical read order even
        # when compaction has left holes or reused a relation handle.
        allocator = self._allocator()
        named = []
        for word in allocator.named_identities.keys() & allocator.word_descriptions.keys():
            if word > 0:
                for obj in allocator.named_identities[word]:
                    row = store.find(obj, word) if obj > 0 else None
                    if row is not None:
                        named.append((row, obj, word))
        self._index_named(sorted(named))
        self._revision = store.effective_revision

    def add(self, row):
        store = self._store()
        store.refresh_effective()
        (_, obj), (_, word) = store.references(row)
        self._index_named(((row, obj, word),))

    def _index_named(self, named):
        store, allocator = self._store(), self._allocator()
        bindings = []
        for row, obj, word in named:
            value = allocator.word_descriptions.get(word)
            if (value is None or obj not in allocator.named_identities.get(word, ())
                    or 'definition' not in store.readings(
                        row, lambda ref: allocator.symbol_types.get(ref[1], ()) if ref[0] == store.SYMBOL else ())):
                continue
            if word <= 0 or obj <= 0 or word == obj:
                raise ValueError('DEF requires distinct native word and object identities')
            # A name can be known before it is trusted as a denotation.
            self._descriptions[word] = value
            if value.get('identity_key') is not None:
                self._identities[value['identity_key']] = word
            for form in value['forms']:
                self._forms[form_key(form)] = word
            for parts in value['parts']:
                self._units[tuple(parts)] = word
            bindings.append((row, obj, word))
        if not bindings:
            return
        from ConceptTrust import definition_threshold
        threshold = definition_threshold()
        rows = store.refs.new_tensor([row for row, _, _ in bindings])
        strengths = store.effective_plus.index_select(0, rows).tolist()
        for (row, obj, word), strength in zip(bindings, strengths, strict=True):
            if strength < threshold:
                continue
            if (word, obj) in self._rows and self._rows[word, obj] != row:
                raise ValueError('duplicate word DEF object row')
            self._object_table = None
            self._rows[word, obj] = row
            self._objects[word] = tuple(sorted(set(self._objects.get(word, ())) | {obj}))
            self._words[obj] = tuple(sorted(set(self._words.get(obj, ())) | {word}))

    @property
    def word_ids(self):
        self._ensure()
        return self._descriptions.keys()

    @property
    def object_ids(self):
        self._ensure()
        return self._words.keys()

    def object_table(self, device):
        """Sorted word references and their first denotation in this live LTM index."""
        self._ensure()
        if getattr(self, '_object_table', None) is None or self._object_table[0].device != device:
            words = sorted(self._objects)
            alternatives = [(i, self._objects[w]) for i, w in enumerate(words) if len(self._objects[w]) > 1]
            columns = torch.tensor((words, [self._objects[w][0] for w in words]),
                                   dtype=torch.long, device=device)
            seats = columns.new_tensor([i for i, _ in alternatives])
            self._object_table = (*columns.unbind(), seats, tuple(group for _, group in alternatives))
        return self._object_table

    def bound_words(self):
        self._ensure()
        return tuple(self._objects)

    def bound_objects(self):
        self._ensure()
        return tuple(self._words)

    def word(self, *, form=None, unit=None, identity=None):
        self._ensure()
        if identity is not None:
            return self._identities.get(identity)
        if unit is not None:
            key = (int(unit),) if isinstance(unit, int) else tuple(unit)
            got = self._units.get(key)
            if got is not None:
                return got
        owner_ref = getattr(self._store(), '_concept_owner', None)
        owner = owner_ref() if owner_ref is not None else None
        if owner is not None and isinstance(form, (str, bytes)) and owner.interpret._native_identity() is not None:
            word = owner.interpret.word_for_form(form)
            return word if word in self._descriptions else None
        return self._forms.get(form_key(form)) if form is not None else None

    def objects(self, word):
        self._ensure()
        return self._objects.get(int(word), ())

    def words(self, obj):
        self._ensure()
        return self._words.get(int(obj), ())

    def row(self, word, obj):
        self._ensure()
        return self._rows.get((int(word), int(obj)))

    def description(self, word):
        self._ensure()
        return self._descriptions.get(int(word))

    def deref(self, word, *, selected=None):
        choices = self.objects(word)
        if selected is not None:
            if int(selected) not in choices:
                raise ValueError('selection is not a defined object')
            return int(selected)
        if len(choices) > 1:
            raise ValueError('ambiguous definition requires a selection')
        return choices[0] if choices else None
