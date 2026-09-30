def forward(self, word, *, order=None, occurrence=None, selected=None,
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
