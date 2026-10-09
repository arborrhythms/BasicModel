"""Unary word-to-object interpretation, with definitions in the common store."""
import torch
from Layers import GrammarLayer
from Definitions import form_key


def positive_evidence(presence):
    """A native positive identification, before any conceptual pole operation."""
    return torch.stack((presence.clamp(0, 1), torch.zeros_like(presence)), -1)


def activate_code(atoms, presence, form_width, *, evidence):
    """One signless code, two independent positive evidence lanes.

    Identification presence belongs only to form. The dictionary's for block
    is the concept's signless context code; its two occurrences have their own
    magnitudes. In particular both never becomes neither.
    """
    if evidence.shape != (*atoms.shape[:-1], 2):
        raise ValueError('interpretation requires an aligned evidence pair')
    torch._assert_async((presence >= 0).all(), 'identification presence must be nonnegative')
    width = (atoms.shape[-1] - form_width) // 2
    code = atoms[..., form_width:form_width + width]
    return torch.cat((atoms[..., :form_width] * presence.unsqueeze(-1),
                      code * evidence[..., :1], code * evidence[..., 1:2]), -1)


class InterpretLayer(GrammarLayer):
    rule_name = 'interpret'
    arity = 1
    space_role = 'SS'
    invertible = True
    mandatory = True

    def __init__(self, nInput=0, nOutput=0, conceptualSpace=None):
        super().__init__(nInput=nInput, nOutput=nOutput)
        object.__setattr__(self, '_conceptual_space', conceptualSpace)
        self._pending = {}
        self._field_pending = set()

    @property
    def owner(self):
        if self._conceptual_space is None:
            raise RuntimeError('interpret requires its shared ConceptualSpace')
        return self._conceptual_space

    def _preflight(self, *, inventory=True, word=None):
        store = self.owner.definitions._store()
        reserved = sum(cid != word for cid in self._pending)
        if len(store) + reserved >= store.capacity:
            raise RuntimeError('definition store capacity exhausted; no word was admitted')
        if inventory:
            self.owner._preflight_concept_row(0)

    def lookup_word(self, parts, wholes, *, form=None, word_reading=False):
        """Fuse and recognise before changing any identity or definition."""
        from Spaces import _concept_alloc_of
        cs = self.owner
        alloc, index = _concept_alloc_of(cs), cs.definitions
        parts = tuple(map(int, parts))
        wholes = (int(wholes),) if isinstance(wholes, int) else tuple(map(int, wholes or ()))
        if getattr(cs, '_online_learning_frozen', False):
            # Held-out reads may resolve existing definitions, but cannot
            # reserve identities, add decompositions or finish pending words.
            return index.word(unit=parts, form=form)
        model = getattr(cs, '_model', None)
        ps = getattr(model, 'perceptualSpace', None)
        native = getattr(ps, 'percept_store', None)
        identity = getattr(native, 'identity', None)
        identity_key = None
        if identity is not None and form is None and parts:
            atom_rows = set(identity.atoms.values())
            if all(part in atom_rows for part in parts):
                form = identity.read_parts(parts)
        if identity is not None and isinstance(form, (str, bytes)):
            from WordIdentity import bytes_of
            raw = bytes_of(form) if isinstance(form, str) else form
            parts = identity.admit(raw)
            self._sync_identity(identity)
            identity_key = identity.key(identity.form(raw)).hex()
        elif native is not None:
            parts = tuple(ps.fuse_parts(parts))
        cs._canonicalize_part_literals()
        form = form_key(form)
        word = index.word(unit=parts, form=form, identity=identity_key)
        for cid, value in self._pending.items():
            if parts in value['parts'] or (form is not None and form in value['forms']):
                word = cid
                break
        if word is not None:
            definition = index.description(word) or self._pending[word]
            previous = definition['parts']
            if any(tuple(ps.fuse_parts(p) if native is not None else p) == parts for p in previous):
                return word
            # Different parts, unlike moving property evidence, are an
            # alternative witnessed definition of this word's object.
            obj = index.deref(word)
            if word not in self._field_pending:
                try:
                    cs._preflight_concept_row(0)
                except RuntimeError:
                    if not word_reading:
                        raise
                    # An optional new decomposition cannot evict a known
                    # word or partially amend its admitted definition.
                    drops = cs.__dict__.setdefault('_concept_admission_drops', {})
                    reason = 'word alternative'
                    drops[reason] = drops.get(reason, 0) + 1
                    return word
                cs._populate_concept_weights(obj if obj is not None else word,
                    witness=(parts, wholes), word_reading=word_reading)
            for part in parts:
                cs.add_part(word, part)
            for whole in wholes:
                cs.add_whole(word, whole)
            value = dict(definition, parts=tuple(previous) + (parts,),
                         wholes=tuple(dict.fromkeys((*definition['wholes'], *wholes))))
            if word in self._pending:
                self._pending[word] = value
            for candidate in index.objects(word):
                row = index.row(word, candidate)
                store = index._store()
                store._definition_rows[int(store.address_keys[row])] = value
                store._update_semantic_fingerprint(row)
                index.add(row)
            return word
        # In the discovering field, interpretation names the case it admits.
        # It never allocates a competing concept row for the same unit.
        discovering = word_reading and cs._promotion_enabled and cs.conceptual_pi and cs._sparse_active()
        if discovering and getattr(ps, '_forward_input', None):
            # An empty field has no object knowledge from which to discover
            # a case. Temporary reading forcing then admits the definition
            # normally. Existing word objects do not turn that admission into
            # native discovery: identify them through the definition table.
            # A field with witnessed objects still owns its case rows, so a
            # word cannot occupy a competing row before its unit recurs.
            lexical = set(index.word_ids) | set(index.object_ids)
            discovering = any(cs.concept_id_at_row(row) not in lexical
                              for row in cs._witnessed_rows())
        if discovering:
            features = alloc.layer()
            case = next((row for (row, col) in features.features._index
                         if len(parts) == 1 and col == 4 * parts[0] and bool(features.assigned[row])), None)
            obj = None if case is None else cs.concept_id_at_row(case)
            try:
                self._preflight(inventory=False)
            except RuntimeError:
                return None
            word = alloc.new_concept()
            for part in parts:
                cs.add_part(word, part)
            for whole in wholes:
                cs.add_whole(word, whole)
            self._pending[word] = dict(forms=() if form is None else (form,), parts=(parts,), wholes=wholes)
            if obj is None:
                # The field has not admitted this object's case yet. The word
                # is a native concept; its reserved interpretation completes
                # on that admission, without occupying a competing row.
                self._field_pending.add(word)
            else:
                self.define(word, obj)
            return word
        try:
            self._preflight()
        except RuntimeError:
            if word_reading:
                return None
            raise
        # Two native identities share one physical row through the unary.
        # The second identity is allocated by forward, after this reservation.
        word = alloc.new_concept()
        for part in parts:
            cs.add_part(word, part)
        for whole in wholes:
            cs.add_whole(word, whole)
        alloc.singletons.add(word)
        cs._populate_concept_weights(word, witness=(parts, wholes), word_reading=True)
        cs._csw_concept_row(0, word)
        self._pending[word] = dict(forms=() if form is None else (form,), parts=(parts,), wholes=wholes)
        if identity_key is not None:
            self._pending[word]['identity_key'] = identity_key
        return word

    def _sync_identity(self, identity):
        """A collision extends the existing word's parts without changing its row."""
        if getattr(self, '_identity_revision', -1) == identity.revision:
            return
        from WordIdentity import bytes_of
        cs, index = self.owner, self.owner.definitions
        for word, old in tuple(self._pending.items()):
            if not old['forms']:
                continue
            raw = old['forms'][0]
            raw = bytes_of(raw) if isinstance(raw, str) else raw
            if raw not in identity.words:
                continue
            parts = tuple(identity.atoms[a] for a in identity.words[raw])
            self._pending[word] = dict(old, parts=(parts,), identity_key=identity.key(identity.form(raw)).hex())
            for part in parts:
                cs.add_part(word, part)
            if word not in self._field_pending:
                cs._populate_concept_weights(word, witness=(parts, old['wholes']), word_reading=True)
        for word in tuple(index.word_ids):
            old = index.description(word)
            if not old or not old['forms']:
                continue
            raw = old['forms'][0]
            raw = bytes_of(raw) if isinstance(raw, str) else raw
            if raw not in identity.words:
                continue
            parts = tuple(identity.atoms[a] for a in identity.words[raw])
            value = dict(old, parts=(parts,), identity_key=identity.key(identity.form(raw)).hex())
            for part in parts:
                cs.add_part(word, part)
            for obj in index.objects(word):
                cs._populate_concept_weights(obj, witness=(parts, old['wholes']), word_reading=True)
                row = index.row(word, obj)
                store = index._store()
                store._definition_rows[int(store.address_keys[row])] = value
                store._update_semantic_fingerprint(row)
        index.rebuild()
        self._identity_revision = identity.revision

    def binding_atoms(self, atoms):
        model = getattr(self.owner, '_model', None)
        store = getattr(getattr(model, 'perceptualSpace', None), 'percept_store', None)
        identity = getattr(store, 'identity', None)
        if identity is None or atoms.shape[-1] < identity.width:
            return atoms
        form = atoms[..., :identity.width]
        sparse = ((form == 0) | (form == 1)).all(-1)
        # A resolved live reference is already a conceptual direction. Only
        # sparse native forms take the word-identity projection.
        return torch.where(sparse[..., None], identity.binding(atoms), atoms)

    @torch.no_grad()
    def define(self, word, obj, *, description=None):
        """The sole writer of word DEF object. Operands name, never copy, codes."""
        from Spaces import _concept_alloc_of
        from Meaning import ConceptualMeaning
        from References import symbol_code
        from Occurrence import definition_key, sentence_key
        cs, word, obj = self.owner, int(word), int(obj)
        alloc, index = _concept_alloc_of(cs), cs.definitions
        store = index._store()
        if getattr(cs, '_online_learning_frozen', False):
            return index.row(word, obj)
        if word == obj or any(cid not in alloc.placement or cid in alloc.retired for cid in (word, obj)):
            raise ValueError('DEF requires distinct, live native identities')
        row = index.row(word, obj)
        if row is not None:
            store.write_timestamp(row)
            store.witness_count[row] += 1
            return row
        self._preflight(inventory=False, word=word)
        description = description or self._pending.get(word) or index.description(word)
        if description is None:
            description = dict(forms=(), parts=(tuple(cs.concept_parts(word)),),
                               wholes=tuple(cs.concept_wholes(word)))
        roles = store.slots.new_zeros(3, store.nDim)
        # Null operand vectors are intentional: refs is the identity column.
        roles[1] = symbol_code(store.REL_DEF, store.nDim, 0, 0).to(roles)
        meaning = ConceptualMeaning(roles, torch.ones(3, dtype=torch.bool, device=roles.device),
            mode='assertive', sentence_kind='relation', role_refs=(('sym', word), None, ('sym', obj)))
        row = store.append_meaning(meaning, kind='unverified', rel_type=store.REL_DEF,
                                   trust=0., evidence=(0., 0.), order=cs._concept_source_order(obj),
                                   document_key=('DEF', word, obj), content_key=sentence_key(()),
                                   address=definition_key(word, obj))
        store.refs[row] = store.refs.new_tensor([word, -1, obj])
        store.row_ids[row] = int(store.address_keys[row])
        store.when[row].zero_()  # Definitions are unlocated, never clock-stamped.
        store._definition_rows[int(store.address_keys[row])] = description
        store._update_semantic_fingerprint(row)
        index.add(row)
        self._pending.pop(word, None)
        self._field_pending.discard(word)
        return row

    def activate(self, atoms, presence, *, evidence):
        """Identify the form and carry both conceptual magnitudes intact."""
        bank = getattr(self.owner, 'similarity_codebook', None)
        derived = getattr(bank, 'mereology', None)
        width = atoms.shape[-1] if derived is None else derived.percept_event_width
        return activate_code(atoms, presence, width, evidence=evidence)

    def forward(self, word, *, order=None, occurrence=None, selected=None,
                object_atoms=None, presence=None, evidence=None):
        if torch.is_tensor(word):
            if object_atoms is None or presence is None or evidence is None:
                raise ValueError('interpret tensor face requires the resolved object bank')
            width = object_atoms.shape[-1]
            return torch.cat((self.activate(self.binding_atoms(object_atoms), presence,
                                           evidence=evidence), word[..., width:]), -1)
        from Spaces import _concept_alloc_of
        cs, word = self.owner, int(word)
        alloc, index = _concept_alloc_of(cs), cs.definitions
        candidates = index.objects(word)
        frozen = getattr(cs, '_online_learning_frozen', False)
        if frozen and not candidates:
            return None
        if word in self._field_pending:
            return self.admit_discovered(word)
        if selected is not None:
            obj = index.deref(word, selected=selected)
        elif candidates:
            choices = tuple(cid for cid in candidates if order is None or cs._concept_source_order(cid) == int(order))
            if not choices:
                choices = candidates
            if len(choices) != 1:
                return None  # Enumerate associations; the grammar chooses.
            obj = choices[0]
        else:
            if word not in self._pending:
                raise ValueError('interpret requires an admitted word')
            self._preflight(inventory=False, word=word)
            obj = alloc.new_concept()
            layer = alloc.layer()
            row = cs._csw_row_of(word)
            if row is None:
                raise ValueError('unary interpretation requires the word inventory row')
            key = layer._tensor_row_keys[row]
            new_key = (*key[:-1], obj) if isinstance(key, tuple) else obj
            del layer._tensor_rows[key]
            layer._tensor_rows[new_key] = row
            layer._tensor_row_keys[row] = new_key
            # The object retains that row's native definition, not an edge to
            # its word or an extra raised singleton/META fold.
            for role, ref in alloc.records(word):
                alloc.add(obj, role, ref)
            alloc.singletons.add(obj)
            alloc.reference_orders[obj] = cs._concept_source_order(word)
        if frozen:
            return obj
        self.define(word, obj)
        if occurrence is not None:
            seen = alloc.testimony_seen.setdefault(obj, set())
            token = tuple(occurrence)
            current = alloc.testimony_current.setdefault(obj, set())
            if token not in current:
                current.add(token)
                if len(seen) < 2:
                    seen.add(token)
                row = cs._csw_row_of(obj)
                if row is not None:
                    store = alloc.layer()
                    beta = cs.concept_use_ewma
                    store.participation[row] = beta * store.participation[row] + (1 - beta)
        return obj

    def admit_discovered(self, word=None):
        """Finish reserved definitions when the field's cases receive IDs."""
        from Spaces import _concept_alloc_of
        cs = self.owner
        if getattr(cs, '_online_learning_frozen', False):
            return None
        model = getattr(cs, '_model', None)
        ps = getattr(model, 'perceptualSpace', None)
        layer = _concept_alloc_of(cs).layer()
        result = None
        for pending in tuple(self._field_pending) if word is None else (word,):
            value = self._pending[pending]
            for parts in value['parts']:
                fused = tuple(ps.fuse_parts(parts)) if ps is not None else parts
                if len(fused) != 1:
                    continue
                row = next((r for (r, col) in layer.features._index
                            if col == 4 * fused[0] and bool(layer.assigned[r])), None)
                obj = None if row is None else cs.concept_id_at_row(row)
                if obj is not None:
                    self.define(pending, obj)
                    result = obj
                    break
        return result

    compose = forward

    def reverse(self, obj, *, word_atoms=None, presence=None, evidence=None):
        if torch.is_tensor(obj):
            if word_atoms is None or presence is None or evidence is None:
                raise ValueError('interpret reverse requires the owned word bank')
            width = word_atoms.shape[-1]
            return torch.cat((self.activate(word_atoms, presence, evidence=evidence), obj[..., width:]), -1)
        words = self.owner.definitions.words(int(obj))
        if len(words) != 1:
            raise ValueError('object requires a lexical selection')
        return words[0]

    generate = reverse

    def boundary(self):
        # Definitions share the common store's retention, without a second
        # word-specific forgetting policy or protected dictionary copy.
        from Spaces import _concept_alloc_of
        _concept_alloc_of(self.owner).testimony_current.clear()
