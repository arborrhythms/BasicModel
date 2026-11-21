"""The flat operated extension beside each durable idea."""
from dataclasses import replace
import torch
from Understanding import ConceptualField


def unkept(owner):
    """The concepts an entry does not keep: the declared classes "words"
    and "objects" of the conceptual space ``owner`` hold of every word and
    every object, so they say nothing of a moment."""
    if owner is None:
        return ()
    from Spaces import _concept_alloc_of
    declared=_concept_alloc_of(owner).declared_classes
    return tuple(int(identity) for identity in declared.values())


@torch.no_grad()
def uncertain(owner):
    """``{object: word}`` for each uncertain object of ``owner``: a blank
    object (§19.112, point 3). An uncertain object is not kept in LTM; the
    concept of the one word that names it is kept instead (``-1`` where no
    single word names it). *Provisional:* an object whose own certainty is
    below one half is the second case, not read here: no measure of an
    object's own certainty is yet kept (its place is zero until placed)."""
    if owner is None:
        return {}
    from Spaces import _concept_alloc_of
    alloc=_concept_alloc_of(owner)
    result={}
    for cid,types in alloc.symbol_types.items():
        if 'blank' not in types or cid in alloc.retired or owner._concept_row_of(int(cid)) is None:
            continue
        words=owner.definitions.words(int(cid))
        result[int(cid)]=int(words[0]) if len(words)==1 else -1
    return result


@torch.no_grad()
def select(field,capacity,present,*,unkept,uncertain=None):
    """The strongest field readings, over all orders, in one fixed list;
    the concepts of ``unkept`` are not kept, and an ``uncertain`` object's
    reading is kept as its word's (the stronger reading of a concept on a
    lane is the one kept)."""
    B=field.evidence.shape[1]
    ids=torch.full((B,capacity),-1,device=field.evidence.device,dtype=torch.long)
    values=field.evidence.new_zeros(B,capacity)
    if not capacity or not field.evidence.shape[2] or not len(field.concept_ids):
        return ids,values
    concepts=field.concept_ids[:,None].expand(-1,B) if field.concept_ids.ndim==1 else field.concept_ids
    strongest=field.evidence.amax(2).permute(1,0,2).reshape(B,-1)
    source=concepts.T[:,:,None].expand(-1,-1,2).reshape(B,-1)
    if uncertain:
        objects=torch.as_tensor(list(uncertain),dtype=source.dtype,device=source.device)
        words=torch.as_tensor(list(uncertain.values()),dtype=source.dtype,device=source.device)
        hit=source[...,None].eq(objects)
        source=torch.where(hit.any(-1),(hit.to(source.dtype)*words).sum(-1),source)
        strongest=strongest.masked_fill(source<0,0.)
        # One concept on one lane fills one slot, with its stronger reading.
        lane=torch.arange(source.shape[1],device=source.device).remainder(2)
        key=torch.where(source<0,-1-torch.arange(source.shape[1],device=source.device),source*2+lane)
        same=key[:,:,None].eq(key[:,None,:])
        order=torch.arange(source.shape[1],device=source.device)
        before=(strongest[:,None,:]>strongest[:,:,None])|(
            (strongest[:,None,:]==strongest[:,:,None])&(order[None,:]<order[:,None])[None])
        dropped=(same&before).any(-1)
        source=source.masked_fill(dropped,-1)
        strongest=strongest.masked_fill(dropped,0.)
    if unkept:
        left=torch.isin(source,torch.as_tensor(unkept,dtype=source.dtype,device=source.device))
        source=source.masked_fill(left,-1)
        strongest=strongest.masked_fill(left,0.)
    rank=strongest.masked_fill(source<0,-1).argsort(dim=-1,descending=True,stable=True)
    count=min(capacity,rank.shape[1]);rank=rank[:,:count]
    chosen=strongest.gather(1,rank)
    active=(chosen>0)&present[:,None]
    ids[:,:count]=torch.where(active,source.gather(1,rank),-1)
    values[:,:count]=torch.where(active,chosen*torch.where(rank.remainder(2)==0,1,-1),0.)
    return ids,values


def stage(model,sid,present,field):
    store=model.symbolSpace.ltm_store
    if store is None:
        return
    owner=model._concept_owner()
    model._sentence_episodes[int(sid)]=select(field,store.episodic_capacity,present,
        unkept=unkept(owner),uncertain=uncertain(owner))


@torch.no_grad()
def write(store,index,entry):
    if entry is None or not isinstance(index,int) or not store.episodic_capacity:
        return
    for name,value in zip(('episodic_concepts','episodic_values'),entry):
        getattr(store,name)[index].copy_(value.detach())
    count(store,'written',int(bool((entry[0]>=0).any())))


def matches(concepts,values,question_ids,question_values,question_orders):
    """One gather over the flat window, independently reduced at each order."""
    ids,permutation=question_ids.sort()
    q,orders=question_values[permutation],question_orders[permutation]
    K=max(1,int(orders.max())+1)
    positions=torch.searchsorted(ids,concepts).clamp_max(max(0,len(ids)-1))
    found_orders=orders[positions]
    valid=(concepts>=0)&(ids[positions]==concepts)&(found_orders>=0)
    lanes=values.lt(0).long();strength=values.abs()
    same=(concepts[:,:,None]==concepts[:,None,:])&(lanes[:,:,None]==lanes[:,None,:])
    slots=torch.arange(concepts.shape[1],device=concepts.device)
    better=(strength[:,None,:]>strength[:,:,None])|((strength[:,None,:]==strength[:,:,None])&(slots[None,:]<slots[:,None]))
    strongest=~(same&better).any(-1)
    product=q[positions].gather(-1,lanes[...,None]).squeeze(-1)*strength*valid*strongest
    numerator=values.new_zeros(len(values),K).scatter_add_(1,found_orders.clamp_min(0),product)
    denominator=q.new_zeros(K).scatter_add_(0,orders.clamp_min(0),q.square().sum(-1)*(orders>=0))
    return (numerator/denominator.clamp_min(torch.finfo(q.dtype).tiny)).clamp(max=1),denominator>0


def relevance(results, present):
    """Intersect result vectors; another result over rows joins this axis."""
    from FieldLogic import reduce
    poles = torch.stack((results, torch.zeros_like(results)), -1)
    return reduce(poles, operation='intersection', dim=1, mask=present)[..., 0]


def query(store, cue, *, stream=0, max_records, work=None, references=(), excluded=()):
    """Rank the window by per-order content, then materialize selected rows;
    the rows ``excluded`` are left out."""
    from ExtendedRecall import field_forest, matches as extend_matches
    forest = field_forest(cue,stream) if isinstance(cue,ConceptualField) else cue
    if not forest or not len(store) or not store.episodic_capacity:
        return dict(value=(),records_scanned=0,incomplete=())
    start = max(0,len(store)-store.ltm_window)
    result = extend_matches(store,forest,start=start)
    scores = result.values[:,0]
    if excluded:
        left = torch.tensor([row-start for row in excluded if start <= row < len(store)],
                            dtype=torch.long,device=scores.device)
        scores = scores.index_fill(0,left,0.) if len(left) else scores
    # Stable ordering starts newest first, then sorts by descending relevance.
    ranked = torch.argsort(scores.flip(0), descending=True, stable=True)
    selected = (len(scores)-1-ranked)[scores.flip(0)[ranked] > 0][:max_records]
    frames, incomplete = [], []
    for offset in selected.tolist():
        if work is not None and not work.consume('record'):
            incomplete.append('episode_read_limit')
            break
        row = start + offset
        from MemoryActivity import accessed
        accessed(store,row)
        frames.append(dict(store.row(row), index=row, match=float(scores[offset])))
    return dict(value=tuple(frames), records_scanned=len(frames), incomplete=tuple(incomplete))


def count(store, name, amount=1):
    """Passive counts of committed input closes and their episode rows."""
    ref = store.__dict__.get('_concept_owner')
    owner = None if ref is None else ref()
    model = None if owner is None else owner._model
    if model is not None:
        model._episode_counts[name] += amount






@torch.no_grad()
def for_meaning(store,meaning):
    """A kept idea's extension, from its kept tree or the native generate walk."""
    if not store.episodic_capacity:
        return None
    owner=store.owner()
    if owner is None:
        return None
    from ExtendedRecall import forest
    from Extension import joined,ConceptSet,as_field
    roots=forest(owner,meaning)
    if not roots:
        return None
    return tree_entry(store, roots)


def tree_entry(store, roots):
    """Copy an explicit extension into the entry's constructed signed slots."""
    from Extension import joined, ConceptSet, as_field
    empty=ConceptSet(store.episodic_concepts.new_empty(0),store.slots.new_empty(0,2))
    field=as_field((joined(roots,empty),),where=store.slots.new_zeros(1,1,2),when=store.slots.new_zeros(1,1,2))
    owner=store.owner()
    ids,values=select(field,store.episodic_capacity,store.slots.new_ones(1,dtype=torch.bool),
                      unkept=unkept(owner),uncertain=uncertain(owner))
    return ids[0],values[0]


def entry_set(store, row):
    """Read a flat entry as concepts with independent for and against lanes."""
    return _flat_set(store.episodic_concepts[row], store.episodic_values[row])


def held_rows(store, stm, row):
    """The rows of ``store`` that a chunk on the stack of STM's stream ``row``
    holds."""
    if stm is None or not hasattr(stm, 'state'):
        return ()
    from Memory import STMChunks as chunks
    state = stm.state
    if not 0 <= row < state[0].shape[0]:
        return ()
    refs, read = state[chunks.REFS][row, :, 0], state[chunks.READ][row]
    stacked = state[chunks.STACK][row] >= 0
    held = []
    for (kind, name), count, on in zip(refs.tolist(), read.tolist(), stacked.tolist()):
        if count > 0 and on and kind == chunks.OCCURRENCE:
            location = store.location(('ltm', store.address_domain, int(name)))
            if isinstance(location, int):
                held.append(location)
    return tuple(dict.fromkeys(held))


def stm_key(store, stm, row, *, work=None):
    """The entry of concepts of the top chunk of STM's stack, as a cue of one
    node, or ``()`` where the stack is empty or the chunk has no entry.

    A chunk that points at a row of ``store`` (a recalled row, or an idea
    kept at a sentence's close) is that row's entry; a chunk of a thought is
    its idea's extension made into an entry as a row's is made."""
    if stm is None or not hasattr(stm, 'state') or not store.episodic_capacity:
        return ()
    from Memory import STMChunks as chunks
    from Extension import Node
    state = stm.state
    if not 0 <= row < state[0].shape[0]:
        return ()
    top = int(chunks.at(state)[row])
    if top < 0:
        return ()
    kind, name = state[chunks.REFS][row, top, 0].tolist()
    entry = None
    if kind == chunks.OCCURRENCE:
        location = store.location(('ltm', store.address_domain, int(name)))
        if isinstance(location, int):
            entry = entry_set(store, location)
    else:
        contents = getattr(stm, '_thought_contents', ())
        item = contents[row][top] if row < len(contents) and top < len(contents[row]) else None
        if item is not None and torch.equal(state[chunks.WHEN][row, top],
                                            state[chunks.WHEN].new_tensor(item[0])):
            meaning = item[1].read()
            owner = store.owner()
            from ExtendedRecall import forest
            roots = meaning.extension or (() if owner is None else forest(owner, meaning, work=work))
            if roots:
                ids, values = tree_entry(store, roots)
                entry = _flat_set(ids, values)
    if entry is None or not len(entry.ids) or not bool(entry.values.any()):
        return ()
    return (Node(entry),)


def _flat_set(ids, values):
    from Extension import ConceptSet
    valid = ids >= 0
    unique, inverse = ids[valid].unique(return_inverse=True)
    lanes = torch.stack((values[valid].clamp_min(0), (-values[valid]).clamp_min(0)), -1)
    combined = lanes.new_zeros(len(unique), 2)
    combined.scatter_reduce_(0, inverse[:, None].expand(-1, 2), lanes, reduce='amax')
    return ConceptSet(unique, combined)
