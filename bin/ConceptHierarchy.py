"""Member centroids and higher-order context in the native concept hierarchy."""
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
    row = owner._csw_row_of(int(identity))
    if row is None:
        raise ValueError('a hierarchy member needs a seated concept')
    bank = owner.similarity_codebook
    derived = getattr(bank, 'mereology', None)
    value = derived.placement[row] if derived is not None else bank.W[row]
    return normalize(value.detach(), dim=-1)


@torch.no_grad()
def refresh(owner):
    """Derive all class directions bottom-up, preserving their learned radii."""
    bank, alloc = owner.similarity_codebook, owner._concept_allocator
    derived = getattr(bank, 'mereology', None)
    if derived is not None and not derived.meaning_pairs:
        return
    identities = sorted(classes(owner), key=alloc.order_of)
    if derived is not None and derived.meaning_pairs:
        derived.derive_class_placements()
    for identity in identities:
        row = owner._csw_row_of(identity)
        if row is None or owner.is_frozen(identity):
            continue
        if derived is not None and derived.meaning_pairs:
            start, width = derived.percept_event_width, derived.meaning_pairs
            value = derived.placement[row]
        else:
            start, width = 0, bank.W.shape[-1]
            seated = [member for member in members(owner, identity)
                      if owner._csw_row_of(member) is not None]
            if not seated:
                continue
            values = torch.stack([direction(owner, member) for member in seated])
            weights = (torch.stack([fit(owner, member, identity) for member in seated])
                       if bool(direction(owner, identity).any()) else values.new_ones(len(seated)))
            value = (values * weights[:, None]).sum(0)
        value = normalize(value, dim=-1)
        bank.W[row, start:start+width] = value * bank.trust.lengths[row, 0]
        bank.trust.content[row, start:start+width] = value


@torch.no_grad()
def observe_member(owner, parent, member):
    """A kept participation teaches one class view, using its member's meaning."""
    from ConceptTrust import observe
    alloc, bank = owner._concept_allocator, owner.similarity_codebook
    if alloc.order_of(parent) != alloc.order_of(member) + 1:
        raise ValueError('a class must be one order above its member')
    row = owner._csw_row_of(parent)
    if row is None:
        raise ValueError('a hierarchy class needs a seated concept')
    if owner.is_frozen(parent):
        return
    alloc.symbol_types.setdefault(parent, set()).update(('concept', 'class'))
    derived = getattr(bank, 'mereology', None)
    if derived is not None:
        derived.placement_derived[row] = True
    alloc.store_of(parent).add_constituent(parent, 'part', ('sym', int(member)))
    if derived is not None and not derived.meaning_pairs:
        return  # A form-only chart supplies no conceptual observation.
    view = bank.W.new_zeros(bank.W.shape[-1])
    start, width = ((derived.percept_event_width, derived.meaning_pairs)
                    if derived is not None and derived.meaning_pairs else (0, view.numel()))
    view[start:start+width] = direction(owner, member)
    bank.trust.allocated[row] = True
    observe(owner, parent, view)


@torch.no_grad()
def condition(owner, identities=None):
    """Pass class context down the hierarchy; change meanings, never forms or trust.

    Higher orders run first, so a class can pass its own parents' influence
    down to its members. Same-order updates share a snapshot. The ordinary
    placement objective and its negative samples supply the contrast.
    """
    from Priming import part_edges
    bank, alloc = owner.similarity_codebook, owner._concept_allocator
    derived = getattr(bank, 'mereology', None)
    if derived is None or not derived.meaning_pairs:
        return  # A chart without a meaning band cannot condition meaning alone.
    edges = part_edges(owner)
    if edges is None:
        return
    src, dst, weights = edges
    orders = {int(row): alloc.order_of(owner.concept_id_at_row(int(row)))
              for row in torch.cat((src, dst)).unique()}
    allowed = None if identities is None else {owner._csw_row_of(int(cid)) for cid in identities}
    targets = {int(row) for row in src if allowed is None or int(row) in allowed}
    # Ancestors may pass their own context down; unrelated members stay still.
    while True:
        parents = {int(parent) for child, parent in zip(src, dst) if int(child) in targets}
        expanded = targets | parents
        if expanded == targets:
            break
        targets = expanded
    for order in sorted({orders[row] for row in targets}, reverse=True):
        snapshot = derived.placement.clone()
        for row in sorted(row for row in targets if orders[row] == order):
            identity = owner.concept_id_at_row(row)
            if owner.is_frozen(identity) or not bool(derived.placement_witnesses[row]):
                continue
            selected = (src == row) & (weights > 0)
            parents, strength = dst[selected], weights[selected]
            higher = torch.tensor([orders[int(parent)] == order+1 for parent in parents],
                                  dtype=torch.bool, device=parents.device)
            parents, strength = parents[higher], strength[higher]
            if not len(parents):
                continue
            context = bank.W.new_zeros(len(bank.W))
            context[parents] = strength
            # Fellow members are positive context, never negative samples.
            context[src[torch.isin(dst, parents)]] = 1.
            negatives = derived._negatives(row, context, derived.placement_witnesses > 0,
                                          derived.content_key(row))
            value = derived.placement_step(snapshot[row], snapshot[parents], strength, snapshot[negatives])
            derived.placement[row].copy_(value)
            derived.placement_dirty[row] = True
            start, width = derived.percept_event_width, derived.meaning_pairs
            bank.W[row, start:start+width] = value * bank.trust.lengths[row, 0]
            raw = bank.trust.content[row, start:start+width]
            bank.trust.content[row, start:start+width] = value * raw.norm()


def can_observe(owner):
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    return derived is None or derived.meaning_pairs > 0


def fit(owner, member, parent):
    """Candidate coverage: magnitude is commitment, never a class gain."""
    from IdentityRows import meet_ratios
    row = owner._csw_row_of(parent)
    value = direction(owner, parent) * owner.similarity_codebook.trust.lengths[row, 0]
    return meet_ratios(direction(owner, member), value)[0]
