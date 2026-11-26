"""Unary word-to-object interpretation, with definitions in the common store."""

from ReadingIndex import steps as recorded_steps
import torch
from Layers import GrammarLayer
from Definitions import form_key
from Memory import NULL_CONCEPT


def preferred_ids(ids, orders, rows, scores):
    """Choose among each word's denotations for all streams' expectations."""
    live = ids >= 0
    valid = (rows >= 0) & (rows < scores.shape[1])
    padded = torch.nn.functional.pad(scores, (0, 1))
    values = padded[:, torch.where(valid, rows, scores.shape[1])]
    positive = live & (values > 0)
    eligible = live & (positive == positive.any(-1, keepdim=True))
    highest = orders.masked_fill(~eligible, torch.iinfo(orders.dtype).min).amax(-1, keepdim=True)
    eligible &= orders == highest
    first = eligible.int().argmax(-1, keepdim=True)
    usable = eligible & ~values.isnan()
    best = values.masked_fill(~usable, -torch.inf).amax(-1, keepdim=True)
    winner = ids.masked_fill(~(usable & (values == best)), torch.iinfo(ids.dtype).max).amin(-1)
    # Match the ordinary comparison when the first contender's score is NaN.
    first_id = ids.expand_as(values).gather(-1, first).squeeze(-1)
    return torch.where(values.gather(-1, first).squeeze(-1).isnan(), first_id, winner)


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
    from ParseForest import native
    parse = native(actions)
    if parse is not None:
        readings = ['symbol' if forced else 'no_op'] * count
        heads = {}
        for at in parse.visits:
            kind, local, word = parse.steps[at]
            if kind == 0:
                heads[at] = word
                continue
            children = parse.operands[at]
            leaf = heads[children[0]] if len(children) == 1 else None
            rule = (language._compose_binary_rules if kind == 1 else language._compose_unary_rules)[local]
            if rule.method_name == 'interpret':
                if leaf is None or readings[leaf] != 'no_op':
                    raise ValueError('interpret requires an uninterpreted leaf')
                readings[leaf] = 'symbol'
            elif rule.method_name == 'sign':
                if leaf is None or readings[leaf] != 'symbol':
                    raise ValueError('sign requires an interpreted leaf')
                readings[leaf] = 'sign'
                leaf = None
            else:
                leaf = None
            heads[at] = leaf
        return tuple(readings), tuple(heads[root] for root in parse.roots)
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
        from util import TheXMLConfig
        # True: a word's meta concept is made here, with a blank object where
        # the word names none. False: neither is made, and a word's object
        # is the other member of a unit across modality that holds the
        # word's symbol (:meth:`named`).
        self.create_meta_concepts = bool(TheXMLConfig.get('architecture.createMetaConcepts'))

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

    def named(self, word, *, batch=0):
        """Of a word, the other member of a unit across modality that holds
        the word's symbol (the one expected, where there are several:
        :meth:`preferred_objects`); the word itself where there is none."""
        from FamiliarTier import across
        found = self.preferred_objects(across(self.owner, word), batch=batch)
        return found[0] if found else int(word)

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
        scores = InterpretLayer._expected_objects(self, batch)
        if scores is None:
            return (max(candidates, key=lambda cid: (cs._concept_source_order(cid), -cid)),)
        return (int(InterpretLayer._preferred_id(self, candidates, scores)),)

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
        return preferred_ids(ids[None], orders[None], rows[None], scores[None])[0, 0]

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

    def spelling_seats(self, seats):
        """Follow each live object's emitting word. Direct word seats spell themselves."""
        cs = self.owner
        defined = cs.definitions
        # Different words can denote one object. Percept collections can
        # choose among them by their current ties, even without a new DEF.
        # Only the compact list of defined objects needs such a decision.
        pairs = []
        words = cs._word_definition_ids()
        for obj in defined.bound_objects():
            if obj in words:
                continue
            row = cs._concept_row_of(obj)
            if row is not None:
                word = cs.word_concept_of_object(obj)
                word_row = None if word is None else cs._concept_row_of(word)
                pairs.append((row, -1 if word_row is None else word_row))
        signature = tuple(sorted(pairs))
        cached = self.__dict__.get('_spelling_seats')
        if cached is None or cached[0] != signature or cached[1].device != seats.device:
            # An absent query uses the sentinel and retains its own address.
            table = seats.new_tensor((*signature, (-1, -1)))
            cached = signature, table[:, 0].contiguous(), table[:, 1].contiguous()
            self.__dict__['_spelling_seats'] = cached
        _, objects, words = cached
        at = torch.searchsorted(objects[:-1], seats.contiguous())
        return torch.where(objects[at] == seats, words[at], seats)

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
        return self._install_word(parts, wholes, form, identity_key, word_reading=word_reading)

    def install_nomination(self, word, seat, form_row, raw, wholes, initial, *, witnesses, holder_rows=None):
        """Publish the native records for a symbol already selected by MTM."""
        cs = self.owner
        identity = cs._model.perceptualSpace.percept_store.identity
        parts = identity.admit(raw, form_row=form_row, witnesses=witnesses, holder_rows=holder_rows)
        self._sync_identity(identity)
        return self._install_word(parts, tuple(wholes), form_key(raw), identity.definition_key(raw),
                                  word_reading=True, pool_word=word, seat=seat, initial=initial)

    def _install_word(self, parts, wholes, form, identity_key, *, word_reading, pool_word=None, seat=None, initial=None):
        from Spaces import _concept_alloc_of
        cs = self.owner
        alloc, index = _concept_alloc_of(cs), cs.definitions
        ps = getattr(getattr(cs, '_model', None), 'perceptualSpace', None)
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
        else:
            obj = None
            try:
                self._preflight()
            except RuntimeError:
                if word_reading:
                    return None
                raise
        # The form owns a row before it has an object. A later witnessed
        # definition may supply one; interpretation cannot invent it.
        word = alloc.new_concept() if pool_word is None else alloc.claim_word_symbol(pool_word)
        alloc.symbol_types.setdefault(word, set()).add('word')
        for part in parts:
            cs.add_part(word, part)
        for whole in wholes:
            cs.add_whole(word, whole)
        alloc.singletons.add(word)
        if seat is not None:
            cs._declare_concept_row(0, word, seat=seat, initial=initial)
        cs._populate_concept_weights(word, witness=(parts, wholes), word_reading=True)
        cs._csw_concept_row(0, word)
        alloc.word_descriptions[word] = dict(forms=() if form is None else (form,), parts=(parts,), wholes=wholes)
        cs.similarity_codebook.trust.remember(word, cs.similarity_codebook.lookup_rows(cs._concept_row_of(word)))
        self._description_revision += 1
        if form is not None:
            cs.bind_word_concept(form, word)
        if obj is not None and self.create_meta_concepts:
            # The field's case is the word's object. Without one, the
            # word's reading gives it its blank object.
            self.define(word, obj)
        if not discovering and identity_key is not None:
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
            return word if self.create_meta_concepts else self.named(word, batch=batch)
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
        alloc = self.owner._concept_allocator
        # With one remembered object, a live definition and an expired
        # definition resolve to the same identity. No strength or forecast
        # can change that choice. A raised kind can suppress a remembered
        # object, so defer those schemas to the full denotation read.
        named = alloc.named_identities
        if not named:
            return words
        if (all(len(objects) <= 1 for objects in named.values())
                and not any(key[0] == 'part-kind' for key in alloc.relate_idx)):
            pairs = tuple(sorted((word, obj) for word, objects in named.items() for obj in objects))
            if all(word > 0 and obj > 0 and obj not in alloc.retired for word, obj in pairs):
                if not pairs:
                    return words
                cached = getattr(self, '_sole_referents', None)
                if cached is None or cached[0] != pairs or cached[1].device != words.device:
                    cached = pairs, words.new_tensor(pairs).T.contiguous()
                    self._sole_referents = cached
                from WordConceptPool import basic_word_rows
                names, values = cached[1].unbind()
                objects = basic_word_rows(words, names, values)
                return torch.where(objects >= 0, objects, words)
        defined = self.owner.definitions
        keys, objects, seats, alternatives = defined.object_table(words.device)
        if alternatives:
            _, all_objects = defined.word_objects(words.device)
            ids = all_objects.index_select(0, seats)
            width = ids.shape[1]
            orders = words.new_tensor([[self.owner._concept_source_order(i) for i in group]
                                       + [-1] * (width - len(group)) for group in alternatives])
            rows = self.owner._concept_allocator.layer(0).concept_rows(ids)
            expected = getattr(self.owner, '_expected_knowing', None)
            if torch.is_tensor(expected) and expected.shape[1]:
                scores = expected[:len(words), :, 0].detach().to(words.device)
                scores = torch.nn.functional.pad(scores, (0, 0, 0, len(words) - len(scores)))
            else:
                scores = torch.zeros((len(words), 0), device=words.device)
            chosen = preferred_ids(ids, orders, rows, scores)
            table = objects.expand(len(words), -1).index_copy(1, seats, chosen)
            at = torch.searchsorted(keys, words.contiguous()).clamp_max(len(keys) - 1)
            resolved = torch.where(keys[at] == words, table.gather(1, at), words)
        elif len(keys):
            at = torch.searchsorted(keys, words.contiguous()).clamp_max(len(keys) - 1)
            resolved = torch.where(keys[at] == words, objects[at], words)
        else:
            resolved = words
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

    def _named_objects(self, words, objects):
        """``objects`` ``[B, W]`` with each word that has no object of its own
        given the one :meth:`named` gives it, by its stream. Only the words
        held by a unit across modality are looked at, so a model with no
        such unit reads nothing here."""
        from FamiliarTier import units
        from WordConceptPool import basic_word_rows
        held = {}
        for _, _, members in units(self.owner):
            for member in members:
                held.setdefault(int(member), []).extend(m for m in members if m != member)
        names = sorted(word for word in held if word in self.owner._concept_allocator.word_descriptions)
        if not names:
            return objects
        table = words.new_tensor([[(self.preferred_objects(tuple(dict.fromkeys(held[word])), batch=b) or (word,))[0]
                                   for word in names] for b in range(words.shape[0])])
        at = basic_word_rows(words.flatten(), words.new_tensor(names),
                             torch.arange(len(names), device=words.device)).reshape(words.shape)
        found = table.gather(1, at.clamp_min(0))
        return torch.where((at >= 0) & (objects == words), found, objects)

    def sentence_objects(self, words):
        """Resolve this presentation, allocating needed blank objects in reading order."""
        from WordConceptPool import basic_word_rows
        cs = self.owner
        alloc = cs._concept_allocator
        if self.mandatory:
            # With the key false no blank object is made (as in
            # :meth:`sentence_object`): a word's object is then the other
            # member of a unit across modality that holds it (:meth:`named`).
            if self.create_meta_concepts and not getattr(cs, '_online_learning_frozen', False):
                possible = sorted(name for name in alloc.word_descriptions
                                  if not alloc.named_identities.get(name))
                if possible:
                    names = words.new_tensor(possible)
                    places = torch.arange(words.numel(), device=words.device)
                    at = basic_word_rows(words.flatten(), names, torch.arange(len(names), device=words.device))
                    first = words.new_full((len(names) + 1,), words.numel())
                    first.scatter_reduce_(0, torch.where(at >= 0, at, len(names)), places, 'amin')
                    # Only first encounters can mint. Return those names in
                    # page order, even when the lexicon is much larger than
                    # the presentation. Repeated sites share the first blank.
                    encountered = (at >= 0) & (places == first[torch.where(at >= 0, at, len(names))])
                    new_names = words.flatten()[encountered]
                    for name in new_names.tolist():
                        if self.owner._concept_row_of(name) is None:
                            raise RuntimeError('an identified word has no conceptual row')
                        self.blank(name)
            objects = self.sentence_referents(words)
            if not self.create_meta_concepts:
                objects = self._named_objects(words, objects)
        else:
            objects = words
        matrix = alloc.layer(0)
        seats = matrix.concept_rows(objects)
        if not self.mandatory:
            from Meaning import require_sentence_tensor
            require_sentence_tensor(((words < 0) | (seats >= 0)).all(), 'an identified word has no conceptual row')
        # Seated concepts take the order of their physical block. A seated
        # relation instead takes the order of its denotation, and an old
        # occurrence name can refer back into the store.
        source_order = alloc.order_from_row
        if (getattr(source_order, '__self__', None) is cs and
                getattr(source_order, '__func__', None) is not None and
                source_order.__func__ is getattr(type(cs), '_seated_concept_order', None)):
            ends = seats.new_tensor(cs._order_caps()).cumsum(0)[:-1]
            orders = torch.where(seats >= 0, torch.bucketize(seats, ends, right=True), -1)
            handles = alloc.parts.__dict__.get('_sentence_handles')
            if handles is None:
                alloc.parts._index()
                handles = alloc.parts._endpoints
            unusual = sorted(key[1] for key in matrix._tensor_rows if isinstance(key, tuple)
                             and key[0] == 'concept' and
                             (key[1] in handles or key[1] not in alloc.concepts))
            if not unusual:
                return objects, seats, orders
            # Query only exceptional names this sentence actually addresses.
            exceptional = objects.new_tensor(unusual)
            which = basic_word_rows(objects, exceptional, exceptional)
            names = which[which >= 0].unique()
        else:
            names = objects[seats >= 0].unique()
            orders = torch.full_like(objects, -1)
        host = names.tolist()
        overrides = names.new_tensor([cs._concept_source_order(name) for name in host])
        at = torch.searchsorted(names, objects.contiguous())
        known = (at < len(names)) & (torch.nn.functional.pad(names, (0, 1), value=-1)[at] == objects)
        orders = torch.where(known, torch.nn.functional.pad(overrides, (0, 1), value=-1)[at], orders)
        return objects, seats, orders

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
        if self.create_meta_concepts:
            blank = self.blank(word)
            obj = word if blank is None else blank
        else:
            obj = self.named(word, batch=batch)
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
