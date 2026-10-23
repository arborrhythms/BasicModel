"""Read native concept kinds and retained cue regions beside MTM identities.

Current identity values live in MTM. These helpers own no working values;
retained concept caches and membership witnesses keep their existing owners.
"""
import torch


def allocator(owner):
    from Spaces import _concept_alloc_of
    return _concept_alloc_of(owner)


def individuals(owner):
    alloc = allocator(owner)
    return tuple(sorted(alloc.individuals - alloc.retired))


def cache_rows(owner):
    alloc = allocator(owner)
    concepts = {cid for cid, types in alloc.symbol_types.items() if 'concept' in types}
    return tuple(row for cid in concepts | alloc.individuals | set(alloc.form_regions.values())
                 if cid not in alloc.retired and (row := owner._csw_row_of(cid)) is not None)


def carrier(owner, point, *, region=True):
    """Validate the native carrier, excluding its separate location/time band."""
    like = owner.similarity_codebook.W
    if point is None:
        return like.new_zeros(like.shape[-1])
    value = point.detach().to(like).clone()
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    if derived is not None:
        value[..., derived.percept_width:derived.percept_event_width] = 0
    if not bool(torch.isfinite(value).all()) or (region and bool(((value < 0) | (value > 1)).any())):
        raise ValueError('identity content requires native concept coordinates in [0, 1]')
    return value


def coordinates(owner, point, *, region=True):
    """Read meanings only; the shared index is the link to perceptual form.

    Keep the carrier's width so narrowed rows and wide STM values retain
    their coordinate addresses. Form and location coordinates supply no
    conceptual mass, overlap, exclusion or union.
    """
    value = carrier(owner, point, region=region)
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    if derived is not None:
        value[..., :derived.percept_event_width] = 0
    return value


def point(owner, identity):
    row = owner._csw_row_of(int(identity))
    if row is None:
        return None
    return owner.similarity_codebook.W[row].detach()


def meet_ratios(query, candidate):
    """Candidate coverage and request coverage, without a distance term."""
    # A retained, unminted LTM0 concept has only its narrowed coordinates.
    # Neither operand acquires imaginary coordinates to match a wider one.
    width = min(query.shape[-1], candidate.shape[-1])
    query, candidate = query[..., :width], candidate[..., :width]
    meet = torch.minimum(query, candidate).sum(-1)
    mass_q, mass_c = query.sum(-1), candidate.sum(-1)
    # An empty request supplies no content evidence. Empty content cannot
    # establish identity merely by vacuous containment.
    return (torch.where(mass_c > 0, meet / mass_c.clamp_min(1e-12), 0.),
            torch.where(mass_q > 0, meet / mass_q.clamp_min(1e-12), 0.))


@torch.no_grad()
def mint(owner, content, *, form=None):
    """Allocate one ordinary concept seat only at a kept write."""
    try:
        owner._preflight_concept_row(1)
        owner._preflight_concept_allocation(1, context='identity fold cache')
    except RuntimeError:
        return None
    content = coordinates(owner, content)
    alloc = allocator(owner)
    identity = owner.new_concept()
    alloc.reference_orders[identity] = 1
    alloc.settle(identity)
    row = owner._csw_concept_row(1, identity)
    if row is None:
        raise RuntimeError('identity inventory changed after allocation preflight')
    owner.similarity_codebook.W[row].copy_(content)
    if form is None:
        alloc.individuals.add(identity)
    else:
        alloc.form_regions[int(form)] = identity
    return identity


def witnesses(owner, identity):
    parts = allocator(owner).parts
    rows = parts.indices()
    return tuple(rows[(parts.refs[rows] == int(identity)).any(-1)].tolist())


def witness_count(owner, identity):
    parts = allocator(owner).parts
    return sum(int(parts.access_count[row]) for row in witnesses(owner, identity))


def form_identity(owner, cue):
    """Only an exact native form identifies a retained region to read."""
    if cue is None or not bool(cue.any()):
        return None
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    width = cue.shape[-1] if derived is None else derived.percept_width
    for word in sorted(owner._word_definition_ids()):
        row = owner._csw_row_of(word)
        if row is None:
            continue
        value = owner.similarity_codebook.lookup_rows(row).detach().to(cue)
        if bool((value[:width] - cue[:width]).abs().max() <= 1e-5):
            return word
    return None


def form_region(owner, cue):
    form = form_identity(owner, cue)
    identity = allocator(owner).form_regions.get(form)
    return None if identity is None else coordinates(owner, point(owner, identity))


def validate_retained(owner):
    """Every load rejects non-native identity content, before any refold.

    This is a permanent representation boundary. Retiring a phase artifact
    does not make it eligible for a lossy conversion on a later load.
    """
    store = owner._closed_clause_store()
    if store is not None:
        for row in range(len(store)):
            meaning = store.raw_meaning_of(row)
            if meaning is not None:
                coordinates(owner, meaning.roles[meaning.role_mask])
    for row in cache_rows(owner):
        coordinates(owner, owner.similarity_codebook.W[row])


def migrate_checkpoint(model, state, live, drift, transforms):
    """Retire the pre-1b identity stores once, before strict loading.

    Native concept seats and occurrence points are retained. Signed chart
    vectors cannot be interpreted as cube regions and are never copied back.
    """
    from IndependentComponents import _pack_state, _unpack_state
    from IndependentComponents import IndependentComponents
    # Reject the retired representation before deleting any key or changing
    # any native owner. Even a chart vector that happens to be nonnegative
    # does not declare cube coordinates.
    if any('.nouns.directions.' in '.' + key for key in state):
        raise ValueError('retired signed-chart identity content requires its producing lesson')
    for name, component in model.named_modules(remove_duplicate=False):
        if not isinstance(component, IndependentComponents):
            continue
        prefix = name + '.'
        for key in tuple(state):
            if key.startswith(prefix+'nouns.'):
                del state[key]
                transforms[key] = None
                drift[key] = dict(action='retired identity bookkeeping; native content unchanged')
        key = prefix+'reference_close._extra_state'
        if key in state and _unpack_state(state[key]).get('version', 1) < 3:
            state[key] = _pack_state(dict(version=3))
            drift[key] = dict(action='retired count tables; participant cache rebuilds from retained rows')
        key = prefix+'reference_close.cue_policy'
        if key in state:
            del state[key]
            transforms[key] = None
            drift[key] = dict(action='retired cue matrix; native form regions learn from exposure')
        key = prefix+'reference_close.policy'
        if key in state and state[key].shape == (5,):
            previous = state[key]
            indices = (-1, 0, 1, -1, -1, 2, 3, 4)
            value = live[key].detach().to(previous).clone()
            for dst, src in enumerate(indices):
                if src >= 0:
                    value[dst] = previous[src]
            state[key] = value
            transforms[key] = (key, 0, indices)
            drift[key] = dict(action='content becomes meet ratios; retain expectation/recency/role weights')
    for key in tuple(state):
        if key.endswith('.identity.projection') and key not in live:
            del state[key]
            drift[key] = dict(action='retired signed form projection; composition reads native cube')
