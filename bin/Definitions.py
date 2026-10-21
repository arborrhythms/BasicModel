"""Derived definition indexes over LTM1's Part(object, word) edges.

All durable data lives on those rows. Lookups never scan, compare codes or
reconstruct a retired META. Appends update the index; load and compaction
rebuild it from the surviving rows.
"""
import weakref


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
        if self._revision != self._store().revision:
            self.rebuild()

    def rebuild(self):
        self._forms, self._units, self._objects, self._words, self._rows = {}, {}, {}, {}, {}
        self._descriptions = {}
        self._identities = {}
        store = self._store()
        for row in store.indices().tolist():
            self.add(row)
        self._revision = store.revision

    def add(self, row):
        store = self._store()
        obj, word = store.refs[row].tolist()
        allocator = self._allocator()
        value = allocator.word_descriptions.get(word)
        if (value is None or 'definition' not in store.readings(
                row, lambda ref: allocator.symbol_types.get(ref, ()))):
            return  # only an object -> word edge has the definition reading
        if word <= 0 or obj <= 0 or word == obj:
            raise ValueError('DEF requires distinct native word and object identities')
        if (word, obj) in self._rows and self._rows[word, obj] != row:
            raise ValueError('duplicate word DEF object row')
        self._rows[word, obj] = row
        self._objects[word] = tuple(sorted(set(self._objects.get(word, ())) | {obj}))
        self._words[obj] = tuple(sorted(set(self._words.get(obj, ())) | {word}))
        self._descriptions[word] = value
        self._revision = store.revision
        if value.get('identity_key') is not None:
            self._identities[value['identity_key']] = word
        for form in value['forms']:
            self._forms[form_key(form)] = word
        for parts in value['parts']:
            self._units[tuple(parts)] = word

    @property
    def word_ids(self):
        self._ensure()
        return self._objects.keys()

    @property
    def object_ids(self):
        self._ensure()
        return self._words.keys()

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
