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
    ids,values=select(field,store.episodic_capacity,store.slots.new_ones(1,dtype=torch.bool))
    return ids[0],values[0]


def entry_set(store, row):
    """Read a flat entry as concepts with independent for and against lanes."""
    from Extension import ConceptSet
    ids, values = store.episodic_concepts[row], store.episodic_values[row]
    valid = ids >= 0
    unique, inverse = ids[valid].unique(return_inverse=True)
    lanes = torch.stack((values[valid].clamp_min(0), (-values[valid]).clamp_min(0)), -1)
    combined = lanes.new_zeros(len(unique), 2)
    combined.scatter_reduce_(0, inverse[:, None].expand(-1, 2), lanes, reduce='amax')
    return ConceptSet(unique, combined)
