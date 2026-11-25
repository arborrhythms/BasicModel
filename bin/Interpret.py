"""Unary word-to-object interpretation, with definitions in the common store."""

from ReadingIndex import steps as recorded_steps
import torch
from Layers import GrammarLayer
from Definitions import form_key
from Memory import NULL_CONCEPT


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


def leaf_readings(language, actions, count, *, forced):
    """Read the two leaf decisions from the actual compose journal.

    A stack entry retains a leaf only until its first non-reading operation.
    The symbol and no-op are abstentions, so neither needs a grammar rule.
    """
    return _read_leaves(language, actions, count, forced=forced)[0]


def well_formed(language, actions, count, *, forced):
    """Whether every word left uninterpreted entered the tree as a literal:
    taken as an operand by a rule of the grammar. A word that no rule took
    stands alone, and the sentence is not well formed."""
    readings, stack = _read_leaves(language, actions, count, forced=forced)
    return not any(leaf is not None and readings[leaf] == 'no_op' for leaf in stack)


def _read_leaves(language, actions, count, *, forced):
    stack, readings, shifted = [], ['symbol' if forced else 'no_op'] * count, set()
    for kind, local, position in recorded_steps(actions):
        if kind < 0:
            break
        if kind == 0:
            if position in shifted or not 0 <= position < count:
                raise ValueError('a reading must shift each leaf exactly once')
            shifted.add(position)
            stack.append(position)
            continue
        if kind == 3:
            stack[-2:] = reversed(stack[-2:])
            continue
        rules = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
        rule = rules[local]
        name = getattr(rule, 'method_name', None)
        arity = 2 if kind == 1 and name != 'bind' else 1
        at = max(0, position) if kind == 2 else 0
        if at:
            stack.append(stack.pop(-1-at))
        operands = stack[-arity:]
        del stack[-arity:]
        leaf = operands[0] if arity == 1 else None
        if name == 'interpret':
            if leaf is None or readings[leaf] != 'no_op':
                raise ValueError('interpret requires an uninterpreted leaf')
            readings[leaf] = 'symbol'
        elif name == 'sign':
            if leaf is None or readings[leaf] != 'symbol':
                raise ValueError('sign requires an interpreted leaf')
            readings[leaf] = 'sign'
            leaf = None
        else:
            leaf = None
        stack.append(leaf)
        if at:
            stack.insert(len(stack)-1-at, stack.pop())
    if len(shifted) != count or forced and any(value == 'no_op' for value in readings):
        raise ValueError('every forced reading leaf must be a sign or a symbol')
    return tuple(readings), tuple(stack)


class SilenceLayer(GrammarLayer):
    """S -> NULL: the sentence whose read gives no well-formed formula.

    In no grammar file and chosen by nothing: the closing applies it where
    a sentence's trees did not reduce to one. Its row holds no word and no
    concept, and what it puts back is NULL at every place.
    """
    rule_name = 'silence'
    arity = 0
    space_role = 'SS'
    invertible = True
    mandatory = False

    def __init__(self, nInput=0, nOutput=0):
        super().__init__(nInput=nInput, nOutput=nOutput)

    def forward(self, like):
        """The empty sentence: NULL in the shape of ``like``."""
        return torch.zeros_like(like)

    compose = forward

    def reverse(self, like):
        """NULL put back at every place of ``like``."""
        return torch.zeros_like(like)

    @staticmethod
    def silent(depths, active):
        """The rows whose closing published no tree: depth below zero."""
        return active & (depths < 0)

    @staticmethod
    def write(store, like):
        """The silent sentence's row: no word, no concept, no trust; it
        carries only the present's stamp."""
        return store.append_idea(torch.zeros_like(like), trust=0.)


class SignLayer(GrammarLayer):
    """Keep the lexical carrier for generation. The kept site opens the percept."""
    rule_name = 'sign'
    arity = 1
    space_role = 'SS'
    invertible = True
    mandatory = False

    def __init__(self, nInput=0, nOutput=0):
        super().__init__(nInput=nInput, nOutput=nOutput)

    def forward(self, word):
        return word

    compose = forward

    def reverse(self, word):
        return word


class InterpretLayer(GrammarLayer):
    rule_name = 'interpret'
    arity = 1
    space_role = 'SS'
    invertible = True
    mandatory = True

    def __init__(self, nInput=0, nOutput=0, conceptualSpace=None):
        super().__init__(nInput=nInput, nOutput=nOutput)
        object.__setattr__(self, '_conceptual_space', conceptualSpace)
        self._description_revision = 0

    @property
    def owner(self):
        if self._conceptual_space is None:
            raise RuntimeError('interpret requires its shared ConceptualSpace')
        return self._conceptual_space

    def objectless(self):
        """Each admitted word that names no object yet, with its description
        (its forms, parts and wholes): a word not yet read, since a reading
        gives a word with no object its blank object (:meth:`blank`)."""
        from Spaces import _concept_alloc_of
        alloc, index = _concept_alloc_of(self.owner), self.owner.definitions
        named = index.word_ids
        return {word: value for word, value in alloc.word_descriptions.items()
                if word not in named and not alloc.named_identities.get(word)
                and word in alloc.concepts and word not in alloc.retired}

    def no_object(self, word):
        """An admitted form can request a referent without inventing an object."""
        word = int(word)
        return word in self.objectless() and not self.owner.definitions.objects(word)

    def word_for_form(self, form):
        """Read an admitted form, including one with no object yet."""
        form = form_key(form)
        word = self.owner.definitions.word(form=form)
        if word is not None:
            return word
        return next((word for word, value in self.objectless().items()
                     if form in value['forms']), None)

    def lexical_identities(self):
        """The current object meanings and objectless forms that can emit."""
        return tuple(dict.fromkeys((*self.owner.definitions.object_ids, *self.objectless())))

    def denotations(self, word):
        """All admitted objects of the word, without discarding lower orders."""
        return self.owner.definitions.objects(int(word))

    def preferred(self, word, *, batch=0, order=None):
        return self.preferred_objects(self.denotations(word), batch=batch, order=order)

    def preferred_objects(self, candidates, *, batch=0, order=None):
        """Choose within the highest expected order, or the highest order."""
        cs = self.owner
        candidates = tuple(candidates) if candidates is not None else ()
        if len(candidates) <= 1:
            return candidates
        if order is not None:
            matching = tuple(cid for cid in candidates if cs._concept_source_order(cid) == int(order))
            candidates = matching or candidates
            if len(candidates) == 1:
                return candidates
        scores = self._expected_objects(batch)
        if scores is None:
            return (max(candidates, key=lambda cid: (cs._concept_source_order(cid), -cid)),)
        return (int(self._preferred_id(candidates, scores)),)

    def _expected_objects(self, batch):
        expected = getattr(self.owner, '_expected_knowing', None)
        if (batch is None or not torch.is_tensor(expected) or not 0 <= batch < len(expected)
                or not expected.shape[1]):
            return None
        return expected[batch, :, 0].detach()

    def _preferred_id(self, candidates, scores):
        """Choose an object by positive expectation, order, score, then smallest ID."""
        cs = self.owner
        ids = torch.tensor(candidates, dtype=torch.long, device=scores.device)
        orders = ids.new_tensor([cs._concept_source_order(cid) for cid in candidates])
        rows = [cs._concept_row_of(cid) for cid in candidates]
        rows = ids.new_tensor([r if r is not None and 0 <= r < len(scores) else -1 for r in rows])
        values = torch.where(rows >= 0, scores[rows.clamp_min(0)], 0.)
        eligible = (values > 0) == (values > 0).any()
        eligible = eligible & (orders == orders.masked_fill(~eligible, torch.iinfo(orders.dtype).min).amax())
        first = eligible.int().argmax().reshape(1)
        # The scalar maximum keeps its first contender if that score is NaN.
        usable = eligible & ~values.isnan()
        best = values.masked_fill(~usable, -torch.inf).amax()
        winner = ids.masked_fill(~(usable & (values == best)), torch.iinfo(ids.dtype).max).amin()
        return torch.where(values.gather(0, first)[0].isnan(), ids.gather(0, first)[0], winner)

    def denoted_ids(self, ids, *, batch=0):
        """Resolve a field's word references in the current stream's expected context."""
        cs = self.owner
        words, objects, seats, alternatives = cs.definitions.object_table(ids.device)
        if not len(words):
            return ids
        if alternatives:
            scores = self._expected_objects(batch)
            if scores is None:
                chosen = ids.new_tensor([self.preferred_objects(group, batch=batch)[0] for group in alternatives])
            else:
                chosen = torch.stack([self._preferred_id(group, scores) for group in alternatives])
            objects = objects.index_copy(0, seats, chosen.to(objects))
        at = torch.searchsorted(words, ids.contiguous()).clamp_max(len(words) - 1)
        return torch.where(words[at] == ids, objects[at], ids)

    def remembered_objects(self, word):
        """Names retained in the concepts, independent of live definition rows."""
        alloc = self.owner._concept_allocator
        return tuple(sorted(cid for cid in alloc.named_identities.get(int(word), ())
                            if cid not in alloc.retired))

    def lexical_form(self, identity):
        if self.no_object(identity):
            value = self.objectless()[int(identity)]
        elif int(identity) in self.owner.definitions.word_ids:
            value = self.owner.definitions.description(int(identity))
        else:
            words = self.owner.definitions.words(int(identity))
            from PerceptCollections import said_by
            words = said_by(self.owner, identity, words)
            if len(words) != 1:
                return None
            value = self.owner.definitions.description(words[0])
        forms = value['forms']
        return forms[0] if len(forms) == 1 else None

    def recognized_objects(self, field, *, batch=0, position=None):
        """Read a percept's fully present classes, without a word or code search."""
        ids = field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:, batch]
        evidence = field.evidence[:, batch, :, 0]
        presence = evidence.amax(-1) if position is None else evidence[:, position]
        objects = self.owner.definitions.object_ids
        return tuple(int(cid) for cid, value in zip(ids.tolist(), presence.detach())
                     if cid in objects and float(value) >= 1. and self.lexical_form(cid) is not None)

    def lexical_atoms(self, rows, atoms, *, word_ids=None, stream=None):
        """Read a higher-order object's word form into an ephemeral lexical view.

        Concept rows keep only their own meaning. An observed word may name
        the chosen form explicitly; a free read requires its unique mapping.
        Order-zero percept collections retain their own native coordinates.
        """
        owner = self.owner
        derived = getattr(owner.similarity_codebook, 'mereology', None)
        if derived is None:
            return atoms
        def form_row(row, observed):
            if row < 0 or owner._row_order(row) <= 0:
                return None
            if observed is None:
                identity = owner.concept_id_at_row(row)
                candidates = () if identity is None else owner.definitions.words(identity)
                if len(candidates) != 1:
                    return None
                observed = candidates[0]
            return owner._concept_row_of(observed) if observed > 0 else None

        width = derived.percept_event_width
        if isinstance(rows, int) and (word_ids is None or isinstance(word_ids, int)):
            # The grammar has already resolved both references for this leaf.
            if atoms.ndim != 1:
                raise ValueError('lexical rows must align with their atom bank')
            row = form_row(rows, word_ids)
            if row is None:
                return atoms
            form = owner.similarity_codebook.lookup_rows(row, stream=0 if stream is None else stream)
            return torch.cat((form[:width], atoms[width:]))
        rows = torch.as_tensor(rows, device=atoms.device, dtype=torch.long)
        if tuple(rows.shape) != tuple(atoms.shape[:-1]):
            raise ValueError('lexical rows must align with their atom bank')
        if word_ids is not None and tuple(word_ids.shape) != tuple(rows.shape):
            raise ValueError('observed lexical identities must align with their rows')
        if word_ids is not None:
            # The observed lexical identity names the word directly. Its
            # coordinates are read only for a seated object above order zero.
            from Meaning import require_sentence_tensor
            require_sentence_tensor((rows < owner.nVectors).all(),
                                    'concept row is outside the constructed inventory', IndexError)
            words = word_ids.to(rows)
            word_rows = owner._concept_allocator.layer(0).concept_rows(words)
            mapped = torch.where((rows >= owner._order_caps()[0]) & (words > 0), word_rows, -1)
        else:
            # A free read can use only an unambiguous name of this object.
            mapped = [form_row(row, None) for row in rows.detach().reshape(-1).tolist()]
            mapped = rows.new_tensor([-1 if row is None else row for row in mapped]).reshape_as(rows)
        if not bool((mapped >= 0).any()):
            return atoms
        options = (dict(streams=True) if rows.ndim > 1 and stream is None else
                   {} if stream is None else dict(stream=stream))
        forms = owner.similarity_codebook.lookup_rows(mapped.clamp_min(0), **options)
        form = torch.where((mapped >= 0)[..., None], forms[..., :width], atoms[..., :width])
        return torch.cat((form, atoms[..., width:]), -1)

    def _preflight(self, *, inventory=True, word=None):
        from Spaces import _concept_alloc_of
        store = self.owner.definitions._store()
        alloc = _concept_alloc_of(self.owner)
        needed = 3 if inventory else 1  # word, object and its Part edge
        if (len(store) >= store.capacity
                or alloc.next_id + needed > store.capacity):
            raise RuntimeError('definition store capacity exhausted; no word was admitted')
        if inventory:
            self.owner._preflight_concept_row(0)

    def lookup_word(self, parts, wholes, *, form=None, word_reading=False, reading=None):
        """Fuse and recognise before changing any identity or definition.

        A form that names no word yet becomes one only when MTM nominates its
        basic percept, on its ``concept_mint_count``-th reading (``reading``
        names the reading); until then it is perceived and not found."""
        from Spaces import _concept_alloc_of
        cs = self.owner
        alloc, index = _concept_alloc_of(cs), cs.definitions
        parts = tuple(map(int, parts))
        wholes = (int(wholes),) if isinstance(wholes, int) else tuple(map(int, wholes or ()))
        if getattr(cs, '_online_learning_frozen', False):
            # Held-out reads may resolve existing definitions, but cannot
            # reserve identities, add decompositions or finish pending words.
            known = index.word(unit=parts, form=form)
            return known if known is not None else next((cid for cid, value in self.objectless().items()
                if parts in value['parts'] or (form is not None and form_key(form) in value['forms'])), None)
        model = getattr(cs, '_model', None)
        ps = getattr(model, 'perceptualSpace', None)
        native = getattr(ps, 'percept_store', None)
        identity = getattr(native, 'identity', None)
        identity_key = None
        if identity is not None and form is None and parts:
            native_rows = set(identity.atoms.values()) | set(identity.word_rows.values())
            if all(part in native_rows for part in parts):
                # The stored form whose code the parts make; a code that is
                # no stored form has no spelling and is fused as parts.
                form = identity.spelling(parts)
        if identity is not None and isinstance(form, (str, bytes)):
            from WordIdentity import bytes_of
            raw = bytes_of(form) if isinstance(form, str) else form
            parts = identity.admit(raw)
            self._sync_identity(identity)
            identity_key = identity.definition_key(raw)
        elif native is not None:
            parts = tuple(ps.fuse_parts(parts))
        form = form_key(form)
        word = index.word(unit=parts, form=form, identity=identity_key)
        objectless = self.objectless()
        for cid, value in objectless.items():
            if parts in value['parts'] or (form is not None and form in value['forms']):
                word = cid
                break
        if word is not None:
            if form is not None:
                cs.bind_word_concept(form, word)
            definition = index.description(word) or objectless[word]
            previous = definition['parts']
            if any(tuple(ps.fuse_parts(p) if native is not None else p) == parts for p in previous):
                return word
            # Different parts, unlike moving property evidence, are an
            # alternative witnessed definition of this word's object.
            obj = index.deref(word)
            if not cs.nominates(('alternative', word, parts), reading):
                return word
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
                witness=(parts, wholes), word_reading=True)
            for part in parts:
                cs.add_part(word, part)
            for whole in wholes:
                cs.add_whole(word, whole)
            value = dict(definition, parts=tuple(previous) + (parts,),
                         wholes=tuple(dict.fromkeys((*definition['wholes'], *wholes))))
            if word in objectless:
                alloc.word_descriptions[word] = value
                self._description_revision += 1
            for candidate in index.objects(word):
                row = index.row(word, candidate)
                store = index._store()
                from Spaces import _concept_alloc_of
                _concept_alloc_of(cs).word_descriptions[word] = value
                self._description_revision += 1
                index.add(row)
            return word
        if not cs.nominates(('form', form) if form is not None else ('parts', parts), reading):
            return None
        # In the discovering field, interpretation names the case it admits.
        # It never allocates a competing concept row for the same unit.
        discovering = word_reading and cs._promotion_enabled and cs._sparse_active()
        if discovering and getattr(ps, '_forward_input', None):
            # An empty field has no object knowledge from which to discover
            # a case. Temporary reading forcing then admits the definition
            # normally. Existing word objects do not turn that admission into
            # native discovery: identify them through the definition table.
            # A field with witnessed objects still owns its case rows, so a
            # word cannot occupy a competing row before its unit recurs.
            lexical = set(index.word_ids) | set(index.object_ids) | set(self.objectless())
            discovering = any(cs.concept_id_at_row(row) not in lexical
                              for row in cs._witnessed_rows())
        if discovering:
            store = alloc.layer()
            case = next((row for (row, percept, _) in alloc.order_zero_members
                         if len(parts) == 1 and percept in cs.basic_percepts.carrying('ps', parts[0]) and bool(store.assigned[row])), None)
            obj = None if case is None else cs.concept_id_at_row(case)
            try:
                self._preflight(inventory=False)
                cs._preflight_concept_row(0, context='word admission')
            except RuntimeError:
                return None
            word = alloc.new_concept()
            alloc.symbol_types.setdefault(word, set()).add('word')
            for part in parts:
                cs.add_part(word, part)
            for whole in wholes:
                cs.add_whole(word, whole)
            alloc.singletons.add(word)
            cs._populate_concept_weights(word, witness=(parts, wholes), word_reading=True)
            cs._csw_concept_row(0, word)
            alloc.word_descriptions[word] = dict(forms=() if form is None else (form,), parts=(parts,), wholes=wholes)
            cs.similarity_codebook.trust.remember(word, cs.similarity_codebook.lookup_rows(cs._concept_row_of(word)))
            self._description_revision += 1
            if form is not None:
                cs.bind_word_concept(form, word)
            if obj is not None:
                # The field's case is the word's object. Without one, the
                # word's reading gives it its blank object.
                self.define(word, obj)
            return word
        try:
            self._preflight()
        except RuntimeError:
            if word_reading:
                return None
            raise
        # The form owns a row before it has an object. A later witnessed
        # definition may supply one; interpretation cannot invent it.
        word = alloc.new_concept()
        alloc.symbol_types.setdefault(word, set()).add('word')
        for part in parts:
            cs.add_part(word, part)
        for whole in wholes:
            cs.add_whole(word, whole)
        alloc.singletons.add(word)
        cs._populate_concept_weights(word, witness=(parts, wholes), word_reading=True)
        cs._csw_concept_row(0, word)
        alloc.word_descriptions[word] = dict(forms=() if form is None else (form,), parts=(parts,), wholes=wholes)
        cs.similarity_codebook.trust.remember(word, cs.similarity_codebook.lookup_rows(cs._concept_row_of(word)))
        self._description_revision += 1
        if form is not None:
            cs.bind_word_concept(form, word)
        if identity_key is not None:
            alloc.word_descriptions[word]['identity_key'] = identity_key
            self._description_revision += 1
        return word

    def _sync_identity(self, identity):
        """A collision extends the existing word's parts without changing its row."""
        from Spaces import _concept_alloc_of
        if getattr(self, '_identity_revision', -1) == identity.revision:
            return
        from WordIdentity import bytes_of
        cs, index = self.owner, self.owner.definitions
        alloc = _concept_alloc_of(cs)
        for word, old in tuple(self.objectless().items()):
            if not old['forms']:
                continue
            raw = old['forms'][0]
            raw = bytes_of(raw) if isinstance(raw, str) else raw
            if raw not in identity.words:
                continue
            parts = identity.parts(raw)
            alloc.word_descriptions[word] = dict(old, parts=(parts,), identity_key=identity.definition_key(raw))
            self._description_revision += 1
            for part in parts:
                cs.add_part(word, part)
            cs._populate_concept_weights(word, witness=(parts, old['wholes']), word_reading=True, coactive=True)
        for word in tuple(index.word_ids):
            old = index.description(word)
            if not old or not old['forms']:
                continue
            raw = old['forms'][0]
            raw = bytes_of(raw) if isinstance(raw, str) else raw
            if raw not in identity.words:
                continue
            parts = identity.parts(raw)
            value = dict(old, parts=(parts,), identity_key=identity.definition_key(raw))
            for part in parts:
                cs.add_part(word, part)
            for obj in index.objects(word):
                cs._populate_concept_weights(obj, witness=(parts, old['wholes']), word_reading=True, coactive=True)
                row = index.row(word, obj)
                store = index._store()
                _concept_alloc_of(cs).word_descriptions[word] = value
                self._description_revision += 1
        index.rebuild()
        self._identity_revision = identity.revision

    @torch.no_grad()
    def define(self, word, obj, *, description=None, trust=None, evidence=None):
        """The sole writer of word DEF object. Operands name, never copy, codes."""
        from Spaces import _concept_alloc_of
        cs, word, obj = self.owner, int(word), int(obj)
        if NULL_CONCEPT in (word, obj):
            raise ValueError('the null-concept cannot define or be defined by a word')
        alloc, index = _concept_alloc_of(cs), cs.definitions
        store = alloc.parts
        if 'word' in alloc.symbol_types.get(obj, ()) or obj in index.word_ids:
            raise ValueError('a definition cannot name a word as its object')
        if getattr(cs, '_online_learning_frozen', False):
            return index.row(word, obj)
        if word == obj or any(cid not in alloc.concepts or cid in alloc.retired for cid in (word, obj)):
            raise ValueError('definition requires distinct, live native identities')
        pair = ((1. if trust is None else float(trust), 0.) if evidence is None
                else tuple(float(pole) for pole in evidence))
        import math
        if len(pair) != 2 or any(not math.isfinite(pole) or not 0 <= pole <= 1 for pole in pair):
            raise ValueError('definition evidence must be two poles in [0, 1]')
        if trust is None and evidence is None:
            cs.similarity_codebook.trust.read_object(obj)
        row = store.find(obj, word)
        if row is not None and obj in alloc.named_identities.get(word, ()):
            alloc.named_identities.setdefault(word, set()).add(obj)
            store.add(obj, word, truth_plus=pair[0], truth_minus=pair[1])
            index.add(row)
            return row
        self._preflight(inventory=False, word=word)
        description = description or alloc.word_descriptions.get(word) or index.description(word)
        if description is None:
            description = dict(forms=(), parts=(tuple(cs.concept_parts(word)),),
                               wholes=tuple(cs.concept_wholes(word)))
        alloc.word_descriptions[word] = description
        self._description_revision += 1
        alloc.symbol_types.setdefault(word, set()).add('word')
        alloc.symbol_types.setdefault(obj, set()).add('object')
        # A definition is an inner semantic relation. Its containing sentence,
        # when present, owns the percept clock stamp.
        alloc.named_identities.setdefault(word, set()).add(obj)
        row = store.add(obj, word, truth_plus=pair[0], truth_minus=pair[1])
        index.add(row)
        self._description_revision += 1
        self.member_of_objects(obj)
        return row

    def member_of_objects(self, obj):
        """An object that interpret finds or makes for a word is a member of
        the declared kind "objects", through the symbol that holds it (found
        first, made where none does: :meth:`ConceptualSpace.things_reading`).
        The kind stands one order above the objects of words; an object at
        or above its order is not one of its members."""
        from Spaces import _concept_alloc_of
        kind = _concept_alloc_of(self.owner).declared_classes.get('objects')
        return None if kind is None else self.owner.kind_entry(kind, obj)

    def compose(self, value, **interpretation):
        """The field chooser supplies its resolved row. A direct read uses the word API."""
        if torch.is_tensor(value) and not interpretation:
            return value
        return self.forward(value, **interpretation)

    def activate(self, atoms, presence, *, evidence):
        """Identify the form and carry both conceptual magnitudes intact."""
        bank = getattr(self.owner, 'similarity_codebook', None)
        derived = getattr(bank, 'mereology', None)
        width = atoms.shape[-1] if derived is None else derived.percept_event_width
        return activate_code(atoms, presence, width, evidence=evidence)

    def bind(self, value, *, identity, scope, candidate_ids, candidate_values,
             candidate_relations, available, associations, mode, active, witness=False):
        """Bind backward, or offer mint and forward to the ordinary chooser.

        Zero is the temporary open-reference marker (never a durable ID).
        Minus one requests admission at the kept closing. No word class,
        allocation, local softmax or random draw participates in this read.
        """
        from ClauseScope import ClauseScope
        matches = candidate_ids == identity[..., None]
        if associations is not None and associations.numel():
            source = identity[..., None] == associations[:, 0]
            target = candidate_ids[..., None] == associations[:, 1]
            matches = matches | (source[..., None, :] & target).any(-1)
        # A composed phrase, selected individual, or already-open operand
        # retains the binding its own operation made.
        explicit = mode in ('bind', 'pronoun')
        eligible = active & (identity > 0) & (scope.bitwise_and(
            ClauseScope.LOCAL | ClauseScope.DETERMINED) == 0)
        if explicit or mode == 'mint':
            eligible = active
        available = available & eligible[..., None]
        found = (available & matches).any(-1)
        # Identity is evidence for the shared scorer, never a decision to
        # bind. Slots and masks agree in eager and compiled execution.
        options = [(candidate_values[:, :, i], candidate_ids[:, :, i],
                    candidate_relations[:, :, i], available[:, :, i] & (mode != 'mint'), found)
                   for i in range(candidate_ids.shape[-1])]
        options.append((value, torch.full_like(identity, -1),
                        torch.zeros_like(active), eligible & (not explicit), found))
        options.append((value, torch.zeros_like(identity), torch.zeros_like(active),
                        eligible & (mode != 'mint') &
                        (~available.any(-1) if explicit else torch.ones_like(active)), found))
        options.append((value, identity, scope.bitwise_and(1) != 0, active & ~eligible, found))
        if witness:
            # Record the actual source branch, even when its address equals
            # the input's. Comparing addresses cannot recover this choice.
            options = [(*option, torch.full_like(active, i < candidate_ids.shape[-1]))
                       for i, option in enumerate(options)]
        return options

    def forward(self, word, *, order=None, occurrence=None, selected=None, batch=0,
                object_atoms=None, presence=None, evidence=None, binding=None):
        if binding is not None:
            return self.bind(word, **binding)
        if torch.is_tensor(word):
            if object_atoms is None or presence is None or evidence is None:
                raise ValueError('interpret tensor face requires the resolved object bank')
            width = object_atoms.shape[-1]
            return torch.cat((self.activate(object_atoms, presence,
                                           evidence=evidence), word[..., width:]), -1)
        from Spaces import _concept_alloc_of
        cs, word = self.owner, int(word)
        alloc, index = _concept_alloc_of(cs), cs.definitions
        candidates = self.preferred(word, batch=batch, order=order)
        frozen = getattr(cs, '_online_learning_frozen', False)
        if self.no_object(word):
            return word
        if frozen and not candidates:
            return None
        if selected is not None:
            obj = index.deref(word, selected=selected)
        elif candidates:
            choices = tuple(cid for cid in candidates if order is None or cs._concept_source_order(cid) == int(order))
            if not choices:
                choices = candidates
            if len(choices) != 1:
                return None  # Enumerate associations; the grammar chooses.
            obj = choices[0]
        elif word in index.word_ids:
            # The lexical identity survives while its definition is below
            # the authority threshold. Its blank object is read meanwhile,
            # as its reading reads it (``referent``); another object has no
            # usable denotation yet.
            blank = self.blank_of(word)
            return blank if blank is not None and not frozen else None
        else:
            raise ValueError('interpret requires an admitted word')
        # Selecting the association is a relation read, not new testimony.
        # Index/cache inspection alone remains free of activity effects.
        row = index.row(word, obj)
        if row is not None:
            from MemoryActivity import accessed
            accessed(cs._closed_clause_store().parts, row)
        if frozen:
            return obj
        if occurrence is not None:
            seen = alloc.testimony_seen.setdefault(obj, set())
            token = tuple(occurrence)
            current = alloc.testimony_current.setdefault(obj, set())
            if token not in current:
                current.add(token)
                if len(seen) < 2:
                    seen.add(token)
        return obj

    def _remembered_referents(self, word):
        remembered = self.remembered_objects(word)
        alloc = self.owner._concept_allocator
        return tuple(cid for cid in remembered if alloc.relate_idx.get(
            ('part-kind', (alloc.parts.SYMBOL, cid), alloc.order_of(cid) + 1)) not in remembered)

    def _ambiguous_referents(self, word, remembered):
        form = self.owner._concept_allocator.word_descriptions.get(word, {}).get('forms', ())
        return f'ambiguous identity reading for word {word} {form}: denotations {list(remembered)}'

    def referent(self, word, *, batch=0):
        """Resolve a word without learning, opening a token, or reading MTM."""
        word = int(word)
        objects = self.preferred(word, batch=batch)
        if objects:
            return objects[0]
        remembered = self._remembered_referents(word)
        if len(remembered) > 1:
            raise ValueError(self._ambiguous_referents(word, remembered))
        return remembered[0] if remembered else word

    def sentence_referents(self, words):
        """Resolve all positive names on each page using its own expected objects."""
        from Meaning import require_sentence_tensor
        defined = self.owner.definitions
        _, _, _, alternatives = defined.object_table(words.device)
        resolved = (torch.stack([self.denoted_ids(row, batch=b) for b, row in enumerate(words)])
                    if len(words) and alternatives else self.denoted_ids(words))
        # A word can retain its name after the live definition has expired.
        # Only those words need the fallback. A dormant ambiguous name has
        # no effect until this sentence actually reads it.
        remembered = []
        for word in sorted(self.owner._concept_allocator.named_identities.keys() - set(defined.bound_words())):
            names = self._remembered_referents(word)
            if len(names) == 1:
                remembered.append((word, names[0]))
            elif len(names) > 1:
                require_sentence_tensor((words != word).all(), self._ambiguous_referents(word, names))
        if remembered:
            keys, values = words.new_tensor(remembered).T.contiguous()
            at = torch.searchsorted(keys.contiguous(), words.contiguous()).clamp_max(len(keys) - 1)
            resolved = torch.where(keys[at] == words, values[at], resolved)
        return resolved

    def reading(self, word, *, batch=0):
        """Identify a read site's object; a word with no object is given its
        blank object (:meth:`blank`), and the word that names it is the meta
        concept.

        The grammar's forward operation remains a choice in thought. This
        eager word boundary supplies no grammatical order or operation.
        """
        word = int(word)
        if not self.mandatory:
            row = self.owner._concept_row_of(word)
            if row is None:
                raise RuntimeError('an identified word has no conceptual row')
            return word, row, self.owner._concept_source_order(word)
        obj = self.referent(word, batch=batch)
        if (self.denotations(word) or obj != word
                or getattr(self.owner, '_online_learning_frozen', False)):
            return obj, self.owner._concept_row_of(obj), self.owner._concept_source_order(obj)
        if self.owner._concept_row_of(word) is None:
            raise RuntimeError('an identified word has no conceptual row')
        blank = self.blank(word)
        obj = word if blank is None else blank
        return obj, self.owner._concept_row_of(obj), self.owner._concept_source_order(obj)

    def blank_of(self, word):
        """The word's blank object, where it has one, or ``None``."""
        from Spaces import _concept_alloc_of
        alloc = _concept_alloc_of(self.owner)
        return next((cid for cid in sorted(alloc.named_identities.get(int(word), ()))
                     if 'blank' in alloc.symbol_types.get(cid, ()) and cid not in alloc.retired), None)

    @torch.no_grad()
    def blank(self, word):
        """The blank object of a word that names no object: a new concept one
        order above the word, with no members, whose vector is zero, and the
        word's definition of it, the meta concept (``define``). A word that
        names an object is given none. Returns the blank, or ``None`` where
        its order has no room left (the word then reads as itself)."""
        from Spaces import _concept_alloc_of
        from ConceptPlacement import promotion_seat, positions
        cs, word = self.owner, int(word)
        alloc = _concept_alloc_of(cs)
        if self.denotations(word) or alloc.named_identities.get(word):
            return None
        order = cs._concept_source_order(word) + 1
        if not promotion_seat(cs, order, context='blank object'):
            return None
        cs._preflight_concept_allocation(1, context='blank object')
        cid = cs.new_concept(order=order)
        alloc.symbol_types.setdefault(cid, set()).update(('concept', 'object', 'blank'))
        alloc.individuals.add(cid)
        row = cs._concept_row_of(cid)
        # Magnitude is certainty in that space: the blank starts at nothing.
        positions(cs)[row].zero_()
        bank = cs.similarity_codebook
        if getattr(bank, 'mereology', None) is not None:
            bank.W[row].zero_()
        self.define(word, cid)
        return cid

    def reverse(self, obj, *, word_atoms=None, presence=None, evidence=None):
        if torch.is_tensor(obj):
            if word_atoms is None or presence is None or evidence is None:
                raise ValueError('interpret reverse requires the owned word bank')
            width = word_atoms.shape[-1]
            return torch.cat((self.activate(word_atoms, presence, evidence=evidence), obj[..., width:]), -1)
        if self.no_object(obj):
            return int(obj)
        from PerceptCollections import said_by
        words = said_by(self.owner, obj, self.owner.definitions.words(int(obj)))
        if len(words) != 1:
            raise ValueError('object requires a lexical selection')
        return words[0]

    generate = reverse

    def boundary(self):
        # Definitions share the common store's retention, without a second
        # word-specific forgetting policy or protected dictionary copy.
        from Spaces import _concept_alloc_of
        _concept_alloc_of(self.owner).testimony_current.clear()
