"""Declared membership and counted evidence in the native concept hierarchy."""
import torch
from torch.nn.functional import normalize


def classes(owner):
    alloc = owner._concept_allocator
    return ({key[1] for key in alloc.relate_idx if key[0] == 'part-base'}
            | set(alloc.raised)
            | {cid for cid, kinds in alloc.symbol_types.items() if 'class' in kinds})


def members(owner, identity):
    """Persistent symbolic members, independent of grammar and epoch history."""
    alloc = owner._concept_allocator
    return tuple(ref[1] for ref in alloc.refs(identity,'part')
                 if isinstance(ref,tuple) and ref[0] == 'sym')


def direction(owner, identity):
    row = owner._concept_row_of(int(identity))
    if row is None:
        raise ValueError('a hierarchy member needs a seated concept')
    bank = owner.similarity_codebook
    derived = getattr(bank, 'mereology', None)
    value = derived.placement[row] if derived is not None else bank.W[row]
    return normalize(value.detach(), dim=-1)


def group_centroid(owner, identities, *, key=None):
    """A cached sum in the members' space, separate from their group's row."""
    identities = tuple(sorted(set(map(int, identities))))
    bank = owner.similarity_codebook
    derived = getattr(bank, 'mereology', None)
    positions = bank.W if derived is None else derived.placement
    if not identities:
        return positions.new_zeros(positions.shape[-1])
    orders = {owner._concept_allocator.order_of(identity) for identity in identities}
    if len(orders) != 1:
        raise ValueError('a centroid belongs to exactly one order')
    revision = (id(positions), positions._version)
    cache = owner.__dict__.setdefault('_membership_centroids', {})
    saved = None if key is None else cache.get(key)
    if saved is None or saved[:2] != (revision, identities):
        total = torch.stack([direction(owner, identity) for identity in identities]).sum(0)
        saved = (revision, identities, total)
        if key is not None:
            cache[key] = saved
    return saved[2] / len(identities)


def extension(owner, identity):
    return group_centroid(owner, members(owner, identity), key=('class',owner._concept_row_of(identity)))


@torch.no_grad()
def observe_member(owner, parent, member):
    """A kept participation teaches one class view, using its member's meaning."""
    from ConceptTrust import observe
    alloc, bank = owner._concept_allocator, owner.similarity_codebook
    if alloc.order_of(parent) != alloc.order_of(member) + 1:
        raise ValueError('a class must be one order above its member')
    row = owner._concept_row_of(parent)
    if row is None:
        raise ValueError('a hierarchy class needs a seated concept')
    if owner.is_frozen(parent):
        return
    alloc.symbol_types.setdefault(parent, set()).update(('concept', 'class'))
    derived = getattr(bank, 'mereology', None)
    alloc.store_of(parent).add_constituent(parent, 'part', ('sym', int(member)))
    owner._populate_concept_weights(parent, witness=({owner._concept_row_of(member): 1.}, ()))
    if derived is not None and not derived.meaning_pairs:
        return  # A form-only chart supplies no conceptual observation.
    view = bank.W.new_zeros(bank.W.shape[-1])
    start, width = ((derived.percept_event_width, derived.meaning_pairs)
                    if derived is not None and derived.meaning_pairs else (0, view.numel()))
    view[start:start+width] = direction(owner, member)
    bank.trust.allocated[row] = True
    observe(owner, parent, view, region=False)


def can_observe(owner):
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    return derived is None or derived.meaning_pairs > 0


def fit(owner, member, parent):
    """Candidate coverage: magnitude is commitment, never a class gain."""
    from IdentityRows import meet_ratios
    row = owner._concept_row_of(parent)
    value = extension(owner, parent) * owner.similarity_codebook.trust.lengths[row, 0]
    return meet_ratios(direction(owner, member), value)[0]
