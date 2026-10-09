"""Read and score the binding left by a completed ordinary question.

No answer is decoded from a numeral, an address's magnitude or a nearest
numeric class. Text targets are consulted only after the candidate is fixed.
"""
import torch

from ThoughtReferences import bindings, open_slots


def references(meaning):
    if open_slots(meaning):
        return ()
    roles = [role for kind, role in bindings(meaning).get('_bound_roles', ())
             if kind == 'referent']
    return tuple((meaning.role_refs[role], meaning.roles[role]) for role in roles
                 if meaning.role_refs[role] is not None)


def _bound_references(meaning):
    roles = [role for kind, role in bindings(meaning).get('_bound_roles', ())
             if kind == 'referent']
    return tuple((meaning.role_refs[role], meaning.roles[role]) for role in roles
                 if meaning.role_refs[role] is not None)


def matches(model, meaning, word):
    """Strict identity match against the existing lexical association index."""
    known = set(model._concept_owner().word_concepts(word))
    values = references(meaning)
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
            return reference[1] in known
        if reference[0] == 'constituent':
            current = current.constituents[reference[1]]
        elif reference[0] == 'ltm' and store is not None:
            index = store._index_occurrences.get(reference)
            if index is None:
                return False
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
            return reference[1] in known
        if reference[0] == 'constituent':
            current = current.constituents[reference[1]]
        elif reference[0] == 'ltm' and store is not None:
            index = store._index_occurrences.get(reference)
            if index is None:
                return False
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


def cost(model, meaning, target):
    """An exact bound identity costs zero; other bindings receive code error.

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
        rows, device=meaning.roles.device, dtype=torch.long)).detach()
    distance = (basis - values[0][1].detach()).square().mean(-1).min()
    return .5 + .5 * distance / (1 + distance)
