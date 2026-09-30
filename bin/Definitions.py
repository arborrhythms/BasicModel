"""Derived identity indexes for the store's word DEF object rows.

All durable data lives on those rows. Lookups never scan, compare codes or
reconstruct a retired META. Appends update the index; load and compaction
rebuild it from the surviving rows.
"""
import weakref


def form_key(value):
    return value.decode('utf-8', errors='surrogateescape') if isinstance(value, bytes) else value


class DefinitionIndex:
    def __init__(self, store):
        self._store = weakref.ref(store)
        self.rebuild()

    def __deepcopy__(self, memo):
        import copy
        result = object.__new__(type(self))
        memo[id(self)] = result
        store = copy.deepcopy(self._store(), memo)
        result._store = weakref.ref(store)
        for name, value in self.__dict__.items():
            if name != '_store':
                setattr(result, name, copy.deepcopy(value, memo))
        return result

    def rebuild(self):
        self._forms, self._units, self._objects, self._words, self._rows = {}, {}, {}, {}, {}
        self._descriptions = {}
        store = self._store()
        alive = set()
        for row in range(len(store)):
            if int(store.rel_type[row]) == store.REL_DEF:
                self.add(row)
                alive.add(int(store.occurrence_id[row]))
        store._definition_rows = {key: value for key, value in store._definition_rows.items() if key in alive}

    def add(self, row):
        store = self._store()
        word, obj = (int(store.refs[row, role]) for role in (0, 2))
        value = store._definition_rows.get(int(store.occurrence_id[row]))
        if value is None:
            return  # state_dict precedes its required semantic sidecar
        if word <= 0 or obj <= 0 or word == obj:
            raise ValueError('DEF requires distinct native word and object identities')
        if (word, obj) in self._rows and self._rows[word, obj] != row:
            raise ValueError('duplicate word DEF object row')
        self._rows[word, obj] = row
        self._objects[word] = tuple(sorted(set(self._objects.get(word, ())) | {obj}))
        self._words[obj] = tuple(sorted(set(self._words.get(obj, ())) | {word}))
        self._descriptions[word] = value
        for form in value['forms']:
            self._forms[form_key(form)] = word
        for parts in value['parts']:
            self._units[tuple(parts)] = word

    @property
    def word_ids(self):
        return self._objects.keys()

    @property
    def object_ids(self):
        return self._words.keys()

    def bound_words(self):
        return tuple(self._objects)

    def bound_objects(self):
        return tuple(self._words)

    def word(self, *, form=None, unit=None):
        if unit is not None:
            key = (int(unit),) if isinstance(unit, int) else tuple(unit)
            got = self._units.get(key)
            if got is not None:
                return got
        return self._forms.get(form_key(form)) if form is not None else None

    def objects(self, word):
        return self._objects.get(int(word), ())

    def words(self, obj):
        return self._words.get(int(obj), ())

    def row(self, word, obj):
        return self._rows.get((int(word), int(obj)))

    def description(self, word):
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


def migrate_meta_definitions(cs, saved):
    """Load-only conversion of checkpoint META membership to ordinary rows."""
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(cs)
    pairs, metas = {}, set()
    forms = {}
    for form, (word, obj, meta) in saved.get('word_obj_meta', {}).items():
        pairs[word, obj] = True
        forms.setdefault(word, []).append(form_key(form))
        if meta is not None:
            metas.add(meta)
    for (word, _), (obj, meta) in saved.get('interpretations', {}).items():
        pairs[word, obj] = True
        if meta is not None:
            metas.add(meta)
    for cid in alloc.placement:
        words = [ref[1] for ref in alloc.refs(cid, 'word') if isinstance(ref, tuple) and ref[0] == 'sym']
        objects = [ref[1] for ref in alloc.refs(cid, 'object') if isinstance(ref, tuple) and ref[0] == 'sym']
        if words and objects:
            metas.add(cid)
            pairs.update(((word, obj), True) for word in words for obj in objects)
    missing = sum(cs.definitions.row(word, obj) is None for word, obj in pairs if word != obj)
    store = cs.definitions._store()
    if len(store) + missing > store.capacity:
        raise RuntimeError('definition store capacity exhausted while migrating META')
    layer = alloc.layer()
    for word, obj in pairs:
        if word in alloc.retired or obj in alloc.retired or word == obj:
            continue
        parts, wholes = cs.concept_parts(word), cs.concept_wholes(word)
        cs.interpret.define(word, obj, description=dict(forms=tuple(forms.get(word, ())),
                            parts=(tuple(parts),), wholes=tuple(wholes)))
        alloc.remove(obj, 'part', ('sym', word))
        word_row, object_row = cs._csw_row_of(word), cs._csw_row_of(obj)
        if word_row is not None:
            old_key = layer._tensor_row_keys.pop(word_row)
            del layer._tensor_rows[old_key]
            if object_row is None:
                new_key = (*old_key[:-1], obj) if isinstance(old_key, tuple) else obj
                layer._tensor_rows[new_key] = word_row
                layer._tensor_row_keys[word_row] = new_key
            else:
                # Existing learned object content retains its address. Its
                # native word evidence migrates from the retired word seat.
                for (row, column), pos in tuple(layer.features._index.items()):
                    if row == word_row:
                        layer.features.add_edge(object_row, column, float(layer.features.values[pos]))
                        layer.features.values.data[pos] = 0.
    for meta in metas:
        cs.retire_concept(meta)
        row = cs._csw_row_of(meta)
        if row is not None:
            key = layer._tensor_row_keys.pop(row)
            del layer._tensor_rows[key]
            for matrix in layer.definition_matrices():
                for (target, column), pos in matrix._index.items():
                    if target == row:
                        matrix.values.data[pos] = 0.
    cs.definitions.rebuild()
