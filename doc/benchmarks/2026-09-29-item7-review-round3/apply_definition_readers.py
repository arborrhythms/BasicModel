"""Recorded Stage B edit script; runtime code is in bin, not imported here."""
from pathlib import Path
p=Path('bin/Spaces.py');s=p.read_text()
s=s.replace('        object.__setattr__(host, "_word_obj_meta", alloc.word_obj_meta)\n','')
a=s.index('    def meta_word_object(');b=s.index('    def _sparse_active(',a);s=s[:a]+s[b:]
a=s.index('    def interpret_word(');b=s.index('    def _register_recognized_word(',a)
s=s[:a]+'''    def interpret_word(self, word_parts, word_whole, key=None, *, occurrence=None):
        """One eager admission; the object's row replaces the word's row."""
        word = self.interpret.lookup_word(word_parts, word_whole, form=key)
        if word is None:
            return None
        obj = self.interpret.forward(word, occurrence=occurrence)
        return word, obj

'''+s[b:]
a=s.index('    def _record_percept_concept(');b=s.index('    def bind_word_concept(',a)
s=s[:a]+'''    def concept_of_percept(self, pid):
        return self.definitions.word(unit=int(pid))

    def object_concept_of_percept(self, pid):
        word = self.definitions.word(unit=int(pid))
        return None if word is None else self.definitions.deref(word)

    def word_concept_of_object(self, cid):
        words = self.definitions.words(cid)
        return words[0] if len(words) == 1 else None

'''+s[b:]
s=s.replace('        return tuple(sorted(cid for cid in alloc.word_forms.get(str(form), ())\n                            if cid in alloc.placement and cid not in alloc.retired))','''        values = set(alloc.word_forms.get(str(form), ()))
        index = self._definition_index()
        word = None if index is None else index.word(form=form)
        if word is not None:
            values.update((word, *index.objects(word)))
        return tuple(sorted(cid for cid in values if cid in alloc.placement and cid not in alloc.retired))''')
a=s.index('    def is_meta(');b=s.index('    def index_part_row(',a)
s=s[:a]+'''    def _definition_index(self):
        model = getattr(self, '_model', None)
        symbol = getattr(model, 'symbolSpace', None) or getattr(self, 'symbolSpace', None)
        store = getattr(symbol, 'ltm_store', None)
        if store is None:
            store = self.__dict__.get('_definition_store')
        return None if store is None else store.definitions

    @property
    def definitions(self):
        index = self._definition_index()
        if index is None:
            raise RuntimeError('interpret requires the common truth store')
        return index

'''+s[b:]
a=s.index('        alloc = _concept_alloc_of(self)',s.index('    def ps_children_of_whole('));b=s.index('        return sorted(',a)
s=s[:a]+'''        words = self.definitions.words(int(concept))
'''+s[b:]
s=s.replace('word_rows = {self._csw_row_of(cid) for cid in _concept_alloc_of(self).lexical_words.values()}','index = self._definition_index()\n        word_rows = {self._csw_row_of(cid) for cid in (() if index is None else index.object_ids)}')
a=s.index('        objects = set(',s.index('    def _witnessed_rows'));b=s.index('        out = []',a)
s=s[:a]+s[b:];s=s.replace('            if cid in objects or cid in alloc.retired:', '            if cid in alloc.retired:')
a=s.index('                    bindings = _concept_alloc_of(self).word_obj_meta');b=s.index('                if formed is None:',a)
s=s[:a]+'''                    index = self.definitions
                    word = index.word(form=key)
                    obj = None if word is None else index.deref(word)
                    formed = None if obj is None else (word, obj)
                else:
                    word = self.interpret.lookup_word(parts, prop_rows, form=key, word_reading=True)
                    formed = None if word is None else (word, self.interpret.forward(word))
'''+s[b:]
s=s.replace('                A, _obj, _meta = formed','                A, obj = formed').replace('self._register_recognized_word(A, key, parts[0])','self._register_recognized_word(obj, key, parts[0])')
s=s.replace('''                        got = (getattr(self, "_word_obj_meta", None)
                               or {}).get(key)
                        loc_sym = got[0] if got else None''','''                        word = self.definitions.word(form=key)
                        loc_sym = None if word is None else self.definitions.deref(word)''')
s=s.replace('_maybe_autobind_meta','_maybe_autobind_words')
p.write_text(s)
p=Path('bin/ConceptIndex.py');s=p.read_text()
a=s.index('def meta_members(');b=s.index('def _symbol_at(',a);s=s[:a]+s[b:]
a=s.index('def bind_meta(');b=s.index('def index_part_row(',a);s=s[:a]+s[b:]
s=s.replace('words = set(alloc.lexical_words.values()) - {obj for obj, _ in alloc.interpretations.values()}','index = cs._definition_index()\n    words = set(() if index is None else index.word_ids)')
s=s.replace('''words = set(self.alloc.lexical_words.values()) - {
                    obj for obj, _ in self.alloc.interpretations.values()}''','''index = cs._definition_index()
                words = set(() if index is None else index.word_ids)''')
p.write_text(s)
p=Path('bin/References.py');s=p.read_text();s=s[:s.index('\n\nclass ReferenceTable:')];s=s.replace('Direct word-to-object candidates derived from native META membership.','Identity symbol encodings; definition lookup lives on the truth store.');p.write_text(s)
p=Path('bin/Models.py');s=p.read_text()
s=s.replace('''"relate_idx", "word_obj_meta", "word_forms",
            "lexical_words", "interpretations", "reference_orders", "testimony_seen",''','''"relate_idx", "word_forms", "reference_orders", "testimony_seen",''')
s=s.replace('''"_words_concept_id", "_percept_word_concept",
            "_object_word_concept", "_priming_bridge", "_frozen_concepts",''','''"_words_concept_id", "_priming_bridge", "_frozen_concepts",''')
s=s.replace('''        object.__setattr__(cs, "_word_obj_meta", alloc.word_obj_meta)
        from ConceptIndex import restore_word_index
        restore_word_index(cs)''','''        # Migration is deferred until the common store's semantic rows load.
        if saved.get('word_obj_meta') or saved.get('interpretations'):
            object.__setattr__(cs, '_legacy_definition_bindings', saved)''')
s=s.replace('''        what_history = extras.get("what_history")''','''        from Definitions import migrate_meta_definitions
        for cs in conceptual_spaces:
            legacy = cs.__dict__.pop('_legacy_definition_bindings', None)
            if legacy is not None:
                migrate_meta_definitions(cs, legacy)

        what_history = extras.get("what_history")''',1)
s=s.replace('        wom = getattr(alloc, "word_obj_meta", {}) if alloc is not None else {}','        definitions = owner.definitions')
s=s.replace('                triple = wom.get(key)','                word_id = definitions.word(form=key)\n                triple = None if word_id is None else (word_id, definitions.deref(word_id))')
s=s.replace('''                    wom = (getattr(alloc, "word_obj_meta", {})
                           if alloc is not None else wom)
''','')
s=s.replace('''                row = owner._csw_row_of(A)
                if row is None:
                    row = owner._csw_concept_row(
                        owner._concept_source_order(A), A)''','''                row = owner._csw_row_of(object_id)''')
s=s.replace('''        allocator = getattr(owner, "_concept_allocator", None)
        triple = getattr(allocator, "word_obj_meta", {}).get(word)
        if triple is None:
            return None
        row = owner._csw_row_of(int(triple[1]))''','''        word_id = owner.definitions.word(form=word)
        if word_id is None:
            return None
        row = owner._csw_row_of(owner.definitions.deref(word_id))''')
s=s.replace('if getattr(_c, "_percept_word_concept", None):','if _c._definition_index() is not None and _c.definitions.word_ids:')
p.write_text(s)
