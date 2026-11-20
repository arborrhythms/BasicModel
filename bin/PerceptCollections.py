"""Named order-zero collections of observed basic percepts, for any modality."""
from dataclasses import replace
import torch


def definitions(owner):
    alloc = owner._concept_allocator
    rows = {owner._concept_row_of(cid): cid for cid, kinds in alloc.symbol_types.items()
            if 'percept_collection' in kinds and cid not in alloc.retired}
    result = {row: [] for row in rows if row is not None}
    for (row, percept, negated), weight in alloc.order_zero_members.items():
        if row in result and not negated and weight > 0:
            result[row].append((percept, weight))
    return {row: edges for row, edges in result.items() if edges}


def read(owner, points):
    return {row: read_collection(owner, points, edges)
            for row, edges in definitions(owner).items()}


def read_collection(owner, points, edges):
    basic = owner.basic_percepts
    return basic.read_collection(basic.read_points(points, [p for p, _ in edges]),
                                 points.new_tensor([w for _, w in edges]))


@torch.no_grad()
def collect(owner, row, percepts):
    """Add witnessed basic percepts to one collection and refresh its bounds."""
    for percept in percepts:
        owner.add_concept_feature(row, 'bp', int(percept), 1.)
    owner.similarity_codebook.W[row].copy_(code(owner, row))


def code(owner, row):
    edges = definitions(owner)[row]
    point = owner.similarity_codebook.W.new_zeros(owner.similarity_codebook.W.shape[-1])
    bounds = owner.basic_percepts.collection_code([p for p, _ in edges])
    point[:len(bounds)] = bounds
    return point


def objects(owner, word):
    """The objects a word is tied to and not untied from: a tie whose
    certainty fell below the definition threshold no longer names it."""
    from WordTies import ties
    from ConceptTrust import definition_threshold
    alloc, held = owner._concept_allocator, ties(owner)
    least = definition_threshold()
    return tuple(cid for cid in owner.interpret.remembered_objects(word)
                 if 'percept_collection' in alloc.symbol_types.get(cid, ()) and cid not in alloc.retired
                 and (held.get(word, cid) is None or held.certainty(word, cid) >= least))


def words(owner, cid):
    """The words that name object ``cid``: those tied to it at the
    definition threshold or above."""
    from WordTies import ties
    from ConceptTrust import definition_threshold
    return ties(owner).words_of(cid, definition_threshold())


def said_by(owner, cid, words):
    """Of the words defined to object ``cid``, the one it is said by: where
    several words are tied to a percept collection, the word of its most
    certain tie (the first tie among equals); otherwise the words as given."""
    words = tuple(words)
    if len(words) < 2 or 'percept_collection' not in owner._concept_allocator.symbol_types.get(int(cid), ()):
        return words
    from WordTies import ties
    held = ties(owner)
    tied = [word for word in held.words_of(cid) if word in words]
    return tuple(sorted(tied, key=lambda word: -held.certainty(word, cid))[:1])


def holder(owner, percepts):
    """An object that already reads every one of ``percepts`` wholly, or
    ``None``: the first by row of those that do."""
    basic, W = owner.basic_percepts, owner.similarity_codebook.W
    points = W.new_zeros(1, len(percepts), W.shape[-1])
    for i, percept in enumerate(percepts):
        points[0, i, :basic.width] = basic.code(percept)
    held = [row for row, value in sorted(read(owner, points).items())
            if bool((value[0, :, 0] >= 1.).all()) and bool((value[0, :, 1] <= 0.).all())]
    return None if not held else owner.concept_id_at_row(held[0])


@torch.no_grad()
def observe(owner, word, percept, occurrence, *, stream):
    """Collect witnessed percepts in MTM; recurrence writes the shared feature
    table. An object that already reads the percept wholly is looked for
    before one is made, and the word is tied to it. The certainty of each
    tie of a word to an object is kept in MTM (:mod:`WordTies`), and the
    word's definition reads it."""
    from WordTies import ties as word_ties
    basic, alloc, pool = owner.basic_percepts, owner._concept_allocator, owner.mtm
    ties = word_ties(owner)
    existing = objects(owner, word)
    if len(existing) > 1:
        raise ValueError('a word has more than one perceptual definition')
    point = owner.similarity_codebook.W.new_zeros(owner.similarity_codebook.W.shape[-1])
    point[:basic.width] = basic.code(percept)
    if existing:
        cid, = existing
        row = owner._concept_row_of(cid)
        agreement = float(read(owner, point[None, None])[row][0, 0, 0] >= 1.)
        ties.start(word, cid)
    else:
        key = next((key for key, entry in pool.entries.items()
                    if entry.refinement.get('defining_word') == word), None)
        previous = None if key is None else pool.entries[key]
        members = () if previous is None else previous.refinement['members']
        if members:
            bounds = basic.read_points(point[None, None], [member[1][1] for member in members])
            agreement = float(basic.read_collection(bounds, point.new_ones(len(members)))[0, 0, 0] >= 1.)
        else:
            agreement = 1.
        key = pool.append_path(key, occurrence, kind='percept', when=0, where=None,
                               stream=stream, order=-1, concept=word)
        entry = pool.entries[key]
        pool.entries[key] = replace(entry, refinement=dict(entry.refinement, defining_word=word,
            members=tuple(dict.fromkeys((*members, (occurrence, ('bp', percept))))),
            readings=(*entry.refinement.get('readings', ()), agreement)))
        entry = pool.entries[key]
        if entry.count < owner.concept_mint_count:
            return None
        cid = holder(owner, [member[1][1] for member in entry.refinement['members']])
        if cid is None:
            owner._preflight_concept_row(0, context='percept collection')
            owner._preflight_concept_allocation(1, context='percept collection')
            cid = owner.new_concept(order=0)
            alloc.individuals.add(cid)
            alloc.symbol_types.setdefault(cid, set()).update(('concept', 'object', 'percept_collection'))
            row = owner._csw_concept_row(0, cid)
            # This row has perceptual bounds, not a placed meaning direction.
            # The certainty is the tie's, not the row's: scaling the bounds by
            # it would change which percepts the concept contains.
            owner.similarity_codebook.trust.allocated[row] = False
            pool.promote(key, cid)
            ties.origin[int(cid)] = int(word)
        else:
            row = owner._concept_row_of(cid)
            pool._release_id(pool.entries.pop(key))
            pool.accesses.pop(key, None)
        ties.start(word, cid, views=entry.count - 1, agreement=sum(entry.refinement['readings'][:-1]))
    # The object is widened by the readings of the word that made it; a
    # word tied to an object another word made only gains or loses certainty.
    if ties.origin.get(int(cid)) == int(word):
        collect(owner, row, (percept,))
    tie = ties.observe(word, cid, agreement)
    owner.interpret.define(word, cid, trust=tie.certainty)
    return cid
