"""Initial places and same-order pulls; membership never copies a position."""
import torch
from torch.nn.functional import normalize


def seed(identity, like):
    generator = torch.Generator(device='cpu').manual_seed(int(identity))
    drawn = torch.randn(like.shape, generator=generator, device='cpu')
    return normalize(drawn.abs().to(like), dim=-1)


def positions(owner):
    bank = owner.similarity_codebook
    derived = getattr(bank, 'mereology', None)
    return bank.W if derived is None else derived.placement


@torch.no_grad()
def initialize(owner, identity, row):
    """Seat a new identity without borrowing the geometry of its members."""
    table = positions(owner)
    table[row].copy_(seed(identity, table[row]))
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    if derived is not None:
        derived.placement_dirty[row] = True


@torch.no_grad()
def identity_start(owner, identity, words):
    """Words select neighbours; only their objects at this order set the place."""
    alloc, table = owner._concept_allocator, positions(owner)
    row, order = owner._concept_row_of(identity), alloc.order_of(identity)
    count = owner.object_start_neighbours
    objects = set()
    for word in sorted(words):
        own = owner._concept_row_of(word)
        candidates = sorted(cid for cid, kinds in alloc.symbol_types.items()
            if 'word' in kinds and cid != word and owner._concept_row_of(cid) is not None
            and alloc.order_of(cid) == alloc.order_of(word))
        if own is None or not candidates:
            continue
        rows = [owner._concept_row_of(cid) for cid in candidates]
        scores = normalize(table[rows], dim=-1) @ normalize(table[own], dim=-1)
        selected = scores.argsort(descending=True, stable=True)[:count].tolist()
        objects.update(cid for index in selected
            for cid in alloc.named_identities.get(candidates[index], ())
            if alloc.order_of(cid) == order and owner._concept_row_of(cid) is not None)
    if objects:
        value = table[[owner._concept_row_of(cid) for cid in sorted(objects)]].mean(0)
        if bool(value.norm() > 0):
            table[row].copy_(normalize(value, dim=-1))
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    if derived is not None:
        derived.placement_dirty[row] = True


def promotion_seat(owner, order, *, context):
    """Automatic promotion may lapse in MTM; explicit allocations still fail."""
    from Spaces import ConceptRowCapacityError
    try:
        owner._preflight_concept_row(order, context=context)
    except ConceptRowCapacityError:
        import warnings
        counts = owner.__dict__.setdefault('_promotion_refusals', {})
        counts[order] = counts.get(order, 0) + 1
        warned = owner.__dict__.setdefault('_promotion_refusal_warned', set())
        if order not in warned:
            warned.add(order)
            caps = owner._order_caps()
            size = caps[order] if 0 <= order < len(caps) else 0
            warnings.warn(f'concept order {order} block ({size} rows) is full; '
                          f'{context} refused; entry remains in MTM until it lapses',
                          RuntimeWarning, stacklevel=3)
        return False
    return True


def meta_groups(owner, context):
    """Active classes and definition rows select same-order member groups."""
    from ConceptHierarchy import members
    alloc = owner._concept_allocator
    active = {owner.concept_id_at_row(row) for row in (context > 0).nonzero().flatten().tolist()}
    active.discard(None)
    groups = []

    def collect(identities, key, *, objects_only=False):
        by_order = {}
        for cid in identities:
            row = owner._concept_row_of(cid)
            if row is not None:
                by_order.setdefault(owner._row_order(row), []).append((cid, row))
        for order, entries in by_order.items():
            # One member at an order has no peer to pull it. In particular,
            # a word and its higher-order object cannot move one another.
            if len(entries) < 2:
                continue
            movers = tuple(row for cid, row in entries if cid in active and
                (not objects_only or 'object' in alloc.symbol_types.get(cid, ())))
            if movers:
                groups.append((movers, tuple(row for _, row in entries), (*key, order)))

    for parent in sorted(active):
        collect(members(owner, parent), ('meta', parent), objects_only=True)
    index = owner.definitions
    pairs = {(word, obj) for word in active for obj in index.objects(word)}
    pairs.update((word, obj) for obj in active for word in index.words(obj))
    for word, obj in sorted(pairs):
        collect((word, obj), ('definition', index.row(word, obj)))
    return groups
