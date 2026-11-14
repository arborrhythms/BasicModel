"""The flat operated extension beside each durable idea."""
from dataclasses import replace
import torch
from Understanding import ConceptualField


@torch.no_grad()
def select(field,capacity,present):
    """The strongest field readings, over all orders, in one fixed list."""
    B=field.evidence.shape[1]
    ids=torch.full((B,capacity),-1,device=field.evidence.device,dtype=torch.long)
    values=field.evidence.new_zeros(B,capacity)
    if not capacity or not field.evidence.shape[2] or not len(field.concept_ids):
        return ids,values
    concepts=field.concept_ids[:,None].expand(-1,B) if field.concept_ids.ndim==1 else field.concept_ids
    strongest=field.evidence.amax(2).permute(1,0,2).reshape(B,-1)
    source=concepts.T[:,:,None].expand(-1,-1,2).reshape(B,-1)
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
    model._sentence_episodes[int(sid)]=select(field,store.episodic_capacity,present)


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


def query(store, cue, *, stream=0, max_records, work=None, references=()):
    """Rank the window by per-order content, then materialize selected rows."""
    from ExtendedRecall import field_forest, matches as extend_matches
    forest = field_forest(cue,stream) if isinstance(cue,ConceptualField) else cue
    if not forest or not len(store) or not store.episodic_capacity:
        return dict(value=(),records_scanned=0,incomplete=())
    start = max(0,len(store)-store.ltm_window)
    result = extend_matches(store,forest,start=start)
    scores = result.values[:,0]
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
def reinstate(store, frames, stream):
    """Publish the remembered concepts, without reinterpreting old percepts."""
    ref = store.__dict__.get('_concept_owner')
    owner = None if ref is None else ref()
    if owner is None or owner._model is None:
        return False
    model = owner._model
    model._episode_counts.setdefault('content_recalls', 0)
    model._episode_counts['content_recalls'] += 1
    indices = [frame['index'] for frame in frames if isinstance(frame['index'], int)
               and bool((store.episodic_concepts[frame['index']] >= 0).any())]
    if not indices:
        return False
    entries = []
    for i in indices:
        for k, cid in enumerate(store.episodic_concepts[i].flatten().tolist()):
            if cid < 0:
                continue
            if owner._csw_row_of(cid) is None:
                model._episode_counts['ids_skipped'] += 1
            else:
                entries.append((i, k, cid))
    from SentenceField import knowing, publish
    current = knowing(model)
    B = current.evidence.shape[1]
    ids = sorted(set(current.concept_ids.flatten().tolist()) | {cid for _, _, cid in entries})
    lookup = {cid: i for i, cid in enumerate(ids)}
    R = max(current.evidence.shape[2], len(entries))
    evidence = current.evidence.new_zeros(len(ids), B, R, 2)
    where = current.where.new_zeros(B, R, 2)
    when = current.when.new_zeros(B, R, 2)
    for b in range(B):
        if b == stream:
            continue
        column = current.concept_ids if current.concept_ids.ndim == 1 else current.concept_ids[:, b]
        for old, cid in enumerate(column.tolist()):
            evidence[lookup[cid], b, :current.evidence.shape[2]] = current.evidence[old, b]
        where[b, :current.where.shape[1]] = current.where[b]
        when[b, :current.when.shape[1]] = current.when[b]
    from RowTime import encoder
    cursor = 0
    for reading, (i, k, cid) in enumerate(entries):
        value = store.episodic_values[i].flatten()[k]
        evidence[lookup[cid], stream, reading] = torch.stack((value.clamp_min(0), (-value).clamp_min(0)))
        raw = owner.word_surface_for_row(owner._csw_row_of(cid))
        if raw is None:
            model._episode_counts['surfaces_missing'] += 1
        length = 0 if raw is None else len(raw)
        where[stream, reading] = where.new_tensor((cursor, cursor + length))
        cursor += length + 1
        tick = encoder().decode_index(store.entry_time(i))
        when[stream, reading] = torch.stack((tick, tick + 1))
    field = ConceptualField(current.concept_ids.new_tensor(ids), evidence, where, when, None)
    owner._episode_review_field = field
    owner._cs_field_concept_ids = field.concept_ids
    owner._cs_field_where, owner._cs_field_when = where, when
    publish(model, field)
    perception = owner._percept_field
    if perception is not None:
        def emptied(value, axis):
            if value is None:
                return None
            result = value.clone()
            result.select(axis, stream).zero_()
            return result
        owner._percept_field = replace(perception,
            values=emptied(perception.values, 1), events=emptied(perception.events, 1),
            spans=emptied(perception.spans, 0), where=emptied(perception.where, 0),
            when=emptied(perception.when, 0), when_band=emptied(perception.when_band, 0),
            boundaries=emptied(perception.boundaries, 2))
    model._episode_counts['reinstatements'] += 1
    return True


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
    empty=ConceptSet(store.episodic_concepts.new_empty(0),store.slots.new_empty(0,2))
    field=as_field((joined(roots,empty),),where=store.slots.new_zeros(1,1,2),when=store.slots.new_zeros(1,1,2))
    ids,values=select(field,store.episodic_capacity,store.slots.new_ones(1,dtype=torch.bool))
    return ids[0],values[0]
