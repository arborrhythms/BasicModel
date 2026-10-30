"""Read and score the binding left by a completed ordinary question.

No answer is decoded from a numeral, an address's magnitude or a nearest
numeric class. Text targets are consulted only after the candidate is fixed.
"""
import torch

from ThoughtReferences import bindings, open_slots


def references(meaning):
    if open_slots(meaning):
        return ()
    return _bound_references(meaning)


def _bound_references(meaning):
    roles = [role for kind, role in bindings(meaning).get('_bound_roles', ())
             if kind == 'referent']
    retained = dict(bindings(meaning).get('_bound_references', ()))
    return tuple((reference, meaning.roles[role]) for role in roles
                 if (reference := retained.get(role, meaning.role_refs[role])) is not None)


def matches(model, meaning, word):
    """Strict identity match against the existing lexical association index."""
    return _matches(model, meaning, word, references(meaning))


def _matches(model, meaning, word, values):
    known = set(model._concept_owner().word_concepts(word))
    if len(values) != 1 or not known:
        return False
    reference, _value = values[0]
    store = getattr(model.symbolSpace, 'ltm_store', None)
    seen = set()
    current = meaning
    for _ in range(64):
        if reference in seen:
            return False
        seen.add(reference)
        if reference[0] == 'sym':
            if reference[1] in known:
                return True
            if store is None or not store.parts.contains(reference[1]):
                return False
            from ChainRead import envelope
            # A chain exposes its actual head reference. An idea offers no
            # stored decomposition; finer identity requires reconstruction.
            reference = envelope(store, store.parts.references(reference[1])[0])
            continue
        if reference[0] == 'constituent':
            current = current.constituents[reference[1]]
        elif reference[0] in ('ltm', 'name') and store is not None:
            index = store._index_occurrences.get(reference)
            if index is None and reference[0] == 'ltm':
                return False
            owner = model._concept_owner()
            token = owner.mtm.token_of(reference)
            entry = owner.mtm.entries.get(token)
            if entry is not None and not entry.pending:
                from ChainRead import envelope
                # Follow the last actual assertion about this identity. A
                # later change ends this equality's authority over its value.
                event = next((event for event in reversed(entry.events) if event.role == 0), None)
                if event is not None and event.occurrence[0] == 'sym' and store.parts.contains(event.occurrence[1]):
                    row = store.part_row(event.occurrence[1])
                    if 'equality' in row['readings'] and row['evidence'][0] > 0:
                        left, right = (envelope(store, item) for item in store.parts.references(event.occurrence[1]))
                        if owner.mtm.token_of(left) == token:
                            reference = right
                            continue
            if reference[0] == 'name':
                if entry is None or entry.pending or entry.symbol is None:
                    return False
                reference = ('sym', entry.symbol)
                continue
            if int(store.row_ids[index]) in known:
                return True
            current = store.meaning_of(index)
        else:
            return False
        if current is None or int(current.role_mask.sum()) != 1:
            return False
        role = int(current.role_mask.nonzero(as_tuple=False)[0])
        reference = current.role_refs[role]
        if reference is None:
            return False
    return False


def cost(model, meaning, target, *, stream=0):
    """An exact identity costs zero; other bindings use content coverage.

    This is a loss-side answer term for the existing paired chooser rule.
    Missing target identities or an unbound answer cannot be scored as correct.
    """
    # R + A judges the answer's referent, irrespective of the evidence pair
    # or another still-free role. The committed-row verifier above is unchanged.
    values = _bound_references(meaning)
    if _matches(model, meaning, target, values):
        return meaning.roles.new_zeros(())
    if len(values) != 1:
        return meaning.roles.new_ones(())
    owner = model._concept_owner()
    rows = [owner._csw_row_of(identity) for identity in owner.word_concepts(target)]
    rows = [row for row in rows if row is not None]
    if not rows:
        return meaning.roles.new_ones(())
    basis = owner.similarity_codebook.lookup_rows(torch.tensor(
        rows, device=meaning.roles.device, dtype=torch.long),stream=stream).detach()
    content = values[0][1].detach()
    codes = getattr(owner.similarity_codebook, 'mereology', None)
    width = getattr(codes, 'context_width', content.shape[-1])
    start = getattr(codes, 'percept_event_width', 0)
    basis, content = basis[:, start:start+width], content[start:start+width]
    if not content.numel():
        return meaning.roles.new_ones(())
    from IdentityRows import meet_ratios
    coverage = meet_ratios(basis, content.expand_as(basis))[1]
    return 1. - coverage.amax()
