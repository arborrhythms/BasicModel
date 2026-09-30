import torch
def test_attended_word_discovery_diagnostic(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path, inventory=128)
    cs = model.conceptualSpaces[0]
    words = ('ant', 'bat', 'cat', 'dog', 'elk', 'fox', 'gnu', 'hare', 'ibex', 'jay', 'koala', 'lynx')
    for word in words:
        raw = torch.zeros(1, 1, 8, dtype=torch.long)
        raw[0, 0, :len(word)] = torch.tensor(list(word.encode()))
        model.forward(raw)
        cs.Reset(hard=True)
        model.End()
    for word in reversed(words):
        raw = torch.zeros(1, 1, 8, dtype=torch.long)
        raw[0, 0, :len(word)] = torch.tensor(list(word.encode()))
        model.forward(raw)
        from Spaces import _concept_alloc_of
        import json
        layer = _concept_alloc_of(cs).layer()
        print(json.dumps(dict(form=word, pending=cs.interpret._pending,
            field_pending=sorted(cs.interpret._field_pending), definitions=list(cs.definitions.bound_words()),
            native=cs._model.perceptualSpace._forward_input['native_indices'].tolist(),
            cases=[dict(row=r, col=c, assigned=bool(layer.assigned[r]),
                        provisional=bool(layer.provisional[r]), participation=float(layer.participation[r]))
                   for r,c in layer.features._index],
            ids=cs._cs_field_concept_ids.tolist()), default=str))
        cid, = cs.word_concepts(word)
        carrier = model._combine_last_cs_sub
        slot = (carrier._concept_ids == cid).nonzero().flatten()
        assert len(slot) == 1, word
        assert carrier._concept_activations[slot[0], 0, :, 0].max() > 0, word
        # Open attention retains every definition reached by native evidence.
        assert cs._cs_last_a0.shape[0] >= 8
        leg = model.symbolSpace.forward_concept_to_symbol(carrier)
        assert leg._symbol_indices[slot[0]].tolist() == [2 * cid, 2 * cid + 1]
        model.End()
    batch = torch.zeros(len(words), 1, 8, dtype=torch.long)
    for b, word in enumerate(words):
        batch[b, 0, :len(word)] = torch.tensor(list(word.encode()))
    model.forward(batch)
    carrier = model._combine_last_cs_sub
    leg = model.symbolSpace.forward_concept_to_symbol(carrier)
    for b, word in enumerate(words):
        from Spaces import _concept_alloc_of
        import json
        layer = _concept_alloc_of(cs).layer()
        print(json.dumps(dict(form=word, pending=cs.interpret._pending,
            field_pending=sorted(cs.interpret._field_pending), definitions=list(cs.definitions.bound_words()),
            native=cs._model.perceptualSpace._forward_input['native_indices'].tolist(),
            cases=[dict(row=r, col=c, assigned=bool(layer.assigned[r]),
                        provisional=bool(layer.provisional[r]), participation=float(layer.participation[r]))
                   for r,c in layer.features._index],
            ids=cs._cs_field_concept_ids.tolist()), default=str))
        cid, = cs.word_concepts(word)
        ids = carrier._concept_ids
        ids = ids[:, b] if ids.ndim == 2 else ids
        slot = (ids == cid).nonzero().flatten()
        assert len(slot) == 1, word
        assert carrier._concept_activations[slot[0], b, :, 0].max() > 0, word
        symbols = leg._symbol_indices
        pair = symbols[slot[0], b] if symbols.ndim == 3 else symbols[slot[0]]
        assert pair.tolist() == [2 * cid, 2 * cid + 1]
    model.End()
