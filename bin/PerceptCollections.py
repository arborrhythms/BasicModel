"""Named order-zero collections of observed basic percepts, for any modality."""
from dataclasses import replace
import math
import torch


def definitions(owner):
    alloc = owner._concept_allocator
    rows = {owner._csw_row_of(cid): cid for cid, kinds in alloc.symbol_types.items()
            if 'percept_collection' in kinds and cid not in alloc.retired}
    matrix = alloc.layer(0).features
    result = {row: [] for row in rows if row is not None}
    for (row, column), index in matrix._index.items():
        if row in result and column % 2 == 0 and float(matrix.values[index].detach()) > 0:
            result[row].append((column // 2, matrix.values[index]))
    return {row: edges for row, edges in result.items() if edges}


def read(owner, points):
    basic = owner.basic_percepts
    return {row: basic.read_collection(basic.read_points(points, [p for p, _ in edges]),
                                      torch.stack([w for _, w in edges]).to(points))
            for row, edges in definitions(owner).items()}


def read_native(owner, perception):
    if perception.boundaries is None:
        return {}
    return {row: owner.basic_percepts.read_collection(
                perception.boundaries.index_select(1, perception.selected([('bp', p) for p, _ in edges])),
                torch.stack([w for _, w in edges]).to(perception.values))
            for row, edges in definitions(owner).items()}


def code(owner, row):
    edges = definitions(owner)[row]
    point = owner.similarity_codebook.W.new_zeros(owner.similarity_codebook.W.shape[-1])
    bounds = owner.basic_percepts.collection_code([p for p, _ in edges])
    point[:len(bounds)] = bounds
    return point


def objects(owner, word):
    alloc = owner._concept_allocator
    return tuple(cid for cid in owner.interpret.remembered_objects(word)
                 if 'percept_collection' in alloc.symbol_types.get(cid, ()) and cid not in alloc.retired)


@torch.no_grad()
def observe(owner, word, percept, occurrence, *, stream):
    """Collect witnessed percepts in MTM; recurrence writes the shared feature table."""
    basic, alloc, pool = owner.basic_percepts, owner._concept_allocator, owner.mtm
    existing = objects(owner, word)
    if len(existing) > 1:
        raise ValueError('a word has more than one perceptual definition')
    point = owner.similarity_codebook.W.new_zeros(owner.similarity_codebook.W.shape[-1])
    point[:basic.width] = basic.code(percept)
    if existing:
        cid, = existing
        row = owner._csw_row_of(cid)
        agreement = float(read(owner, point[None, None])[row][0, 0, 0] >= 1.)
        previous = None
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
        owner._preflight_concept_row(0, context='percept collection')
        owner._preflight_concept_allocation(1, context='percept collection')
        cid = alloc.new_concept()
        alloc.reference_orders[cid] = 0
        alloc.individuals.add(cid)
        alloc.symbol_types.setdefault(cid, set()).update(('concept', 'object', 'percept_collection'))
        row = owner._csw_concept_row(0, cid)
        trust = owner.similarity_codebook.trust
        # This row has perceptual bounds, not a placed meaning direction.
        # Its definition owns the counted certainty; scaling the bounds by
        # that certainty would change which percepts the concept contains.
        trust.allocated[row] = False
        trust.views[row] = entry.count - 1
        trust.agreement[row] = sum(entry.refinement['readings'][:-1])
        pool.promote(key, cid)
    owner.add_concept_feature(row, 'bp', percept, 1.)
    owner.similarity_codebook.W[row].copy_(code(owner, row))
    trust = owner.similarity_codebook.trust
    trust.views[row] += 1
    trust.agreement[row] += agreement
    views = int(trust.views[row])
    certainty = float(trust.agreement[row]) / views * -math.expm1(-views / trust.trust_views)
    owner.interpret.define(word, cid, evidence=(certainty, 0.))
    return cid
