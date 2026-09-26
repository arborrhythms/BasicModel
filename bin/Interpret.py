"""The mandatory serial word-to-object grammar face.

The eager admission face resolves stable identities before graph execution;
the tensor face transports the selected object through the word transaction.
Spelling belongs to the word's lookup/inverse, never to object resolution.
"""
import torch
from Layers import GrammarLayer


class InterpretLayer(GrammarLayer):
    rule_name = 'interpret'
    arity = 1
    space_role = 'SS'
    invertible = True
    mandatory = True

    def __init__(self, nInput=0, nOutput=0, conceptualSpace=None):
        super().__init__(nInput=nInput, nOutput=nOutput)
        object.__setattr__(self, '_conceptual_space', conceptualSpace)

    @property
    def owner(self):
        if self._conceptual_space is None:
            raise RuntimeError('interpret requires its shared ConceptualSpace')
        return self._conceptual_space

    def lookup_word(self, parts, wholes, *, form=None, reserve=1):
        """PartSpace lookup has already supplied these native word references."""
        from Spaces import _concept_alloc_of
        cs = self.owner
        alloc = _concept_alloc_of(cs)
        if isinstance(wholes, int):
            wholes = (wholes,)
        parts, wholes = tuple(map(int, parts)), tuple(map(int, wholes or ()))
        # Ordered native references identify the word. The form only owns its
        # spelling inverse and a cache of previously admitted lookups.
        key = parts
        word = alloc.lexical_words.get(key)
        prior = alloc.word_obj_meta.get(form) if form is not None else None
        if word is None and prior is not None:
            word = int(prior[0])
        if word is None:
            cs._preflight_concept_allocation(reserve,
                context='interpreted word' if reserve > 1 else 'word admission')
            word = cs.new_concept()
        alloc.lexical_words[key] = word
        for part in parts:
            cs.add_part(word, part)
        for whole in wholes:
            cs.add_whole(word, whole)
        alloc.singletons.add(word)
        cs._populate_concept_weights(word, witness=(parts, wholes))
        row = cs._csw_concept_row(0, word)
        if isinstance(form, (str, bytes)):
            raw = form if isinstance(form, bytes) else form.encode('utf-8')
            cs.__dict__.setdefault('_row_surfaces', {})[row] = raw
            cs.bind_word_concept(raw, word)
        return word

    def forward(self, word, *, order=None, occurrence=None,
                object_atoms=None, activation=None):
        if torch.is_tensor(word):
            if object_atoms is None or activation is None:
                raise ValueError('interpret tensor face requires the resolved object bank')
            width = object_atoms.shape[-1]
            return torch.cat((object_atoms * activation.unsqueeze(-1),
                              word[..., width:]), dim=-1)
        from Spaces import _concept_alloc_of
        cs = self.owner
        alloc = _concept_alloc_of(cs)
        word = int(word)
        order = None if order is None else int(order)
        if order not in (None, 1, 2):
            raise ValueError('interpret resolution must be particular (1) or kind (2)')
        if word not in alloc.placement or word in alloc.retired:
            raise ValueError('interpret requires an existing word concept')
        # The default describes a NEW testimony object, not an instruction to
        # replace an already known referent. Include witnessed associations,
        # not only the objects this operator originally minted.
        candidates = {obj for (w, _), (obj, _) in alloc.interpretations.items()
                      if w == word and obj not in alloc.retired}
        for identities in alloc.word_forms.values():
            if word in identities:
                candidates.update(cid for cid in identities if cid != word
                    and cid not in alloc.retired and cs._concept_source_order(cid) > 0)
        if len(candidates) > 1:
            matches = ({obj for w, obj, _ in alloc.word_obj_meta.values()
                        if w == word and obj in candidates} if order is None else
                       {cid for cid in candidates if cs._concept_source_order(cid) == order})
            if len(matches) != 1:
                raise ValueError('ambiguous word association requires a resolving grammar order')
            candidates = matches
        obj = next(iter(candidates)) if candidates else None
        order = cs._concept_source_order(obj) if obj is not None else (order or 1)
        key = (word, order)
        record = alloc.interpretations.get(key)
        if record is None:
            minted = obj is None
            if minted:
                obj, meta = cs._new_concepts(2, context='interpret testimony')
                alloc.singletons.add(obj)
            else:
                meta, = cs._new_concepts(1, context='interpret association')
            # Grammar resolves the referent's order; a testimony placeholder
            # has just the naming word as its literal, until the seal or a
            # witnessed field writes more of its definition.
            if minted:
                alloc.reference_orders[obj] = order
                cs.add_part(obj, ('sym', word))
            alloc.store_of(meta).embed_pair(meta, whole_ref=('sym', word),
                                           part_ref=('sym', obj))
            alloc.settle(meta)
            if minted:
                cs._populate_concept_weights(obj)
            # The association is structural; it owns no weighted dictionary row.
            row = cs._csw_concept_row(order, obj)
            store = alloc.layer()
            store.ensure_context()
            if minted:
                store.provisional[row] = True
                store.assigned[row] = True
                store.managed[row] = True
                store.participation[row] = 0.
            record = (obj, meta)
            alloc.interpretations[key] = record
            cs._record_percept_concept(word, obj, cs.concept_parts(word))
            for identities in alloc.word_forms.values():
                if word in identities:
                    identities.add(obj)
        obj, meta = record
        # Repeated eager reads of one occurrence do not inflate recurrence.
        if occurrence is not None:
            seen = alloc.testimony_seen.setdefault(obj, set())
            token = tuple(occurrence)
            current = alloc.testimony_current.setdefault(obj, set())
            if token not in current:
                current.add(token)
                if len(seen) < 2:
                    seen.add(token)
                row = cs._csw_row_of(obj)
                store = alloc.layer()
                beta = cs.concept_use_ewma
                store.participation[row] = beta * store.participation[row] + (1 - beta)
        return obj

    compose = forward

    def reverse(self, obj, *, word_atoms=None, activation=None):
        if torch.is_tensor(obj):
            if word_atoms is None or activation is None:
                raise ValueError('interpret reverse requires the owned word bank')
            width = word_atoms.shape[-1]
            return torch.cat((word_atoms * activation.unsqueeze(-1), obj[..., width:]), -1)
        word = self.owner.word_concept_of_object(int(obj))
        if word is None:
            raise ValueError('object has no owned lexical inverse')
        return word

    generate = reverse

    @torch.no_grad()
    def boundary(self):
        """Admit recurrent testimony; forget unused one-off placeholders.

        The same allocator is shared across stages. Only its owner performs
        lifecycle work, once per observation; other resets cannot age a row.
        Identities are never renumbered or recycled after retirement.
        """
        from Spaces import _concept_alloc_of
        cs = self.owner
        model = getattr(cs, '_model', None)
        if model is not None and callable(getattr(model, '_concept_owner', None)):
            if model._concept_owner() is not cs:
                return
            tick = getattr(model, '_word_observation_serial', 0)
            if self.__dict__.get('_boundary_tick') == tick:
                return
            self._boundary_tick = tick
        alloc = _concept_alloc_of(cs)
        store = alloc.layer()
        for key, (obj, meta) in list(alloc.interpretations.items()):
            row = cs._csw_row_of(obj)
            seen = alloc.testimony_seen.get(obj, ())
            used = bool(alloc.testimony_current.get(obj))
            if row is None or obj in getattr(cs, '_frozen_concepts', ()):
                continue
            if len(seen) > 1 and float(store.participation[row]) >= cs.concept_mint_threshold:
                store.provisional[row] = False
            elif (len(seen) == 1 and not used and bool(store.provisional[row])
                  and float(store.participation[row]) < cs.concept_recycle_threshold):
                # Keep old identity-to-spelling inverses for already captured
                # programs, but remove this placeholder from future knowing.
                for cid in (obj, meta):
                    address = cs._csw_row_of(cid)
                    if address is not None:
                        for matrix in store.definition_matrices():
                            for (target, column), pos in matrix._index.items():
                                if target == address:
                                    matrix.values[pos] = 0.
                        store.provisional[address] = False
                        store.assigned[address] = False
                        store.managed[address] = False
                        store.participation[address] = 0.
                    cs.retire_concept(cid)
                del alloc.interpretations[key]
                alloc.testimony_seen.pop(obj, None)
                for form, triple in list(alloc.word_obj_meta.items()):
                    if triple[1] == obj:
                        del alloc.word_obj_meta[form]
        alloc.testimony_current.clear()
