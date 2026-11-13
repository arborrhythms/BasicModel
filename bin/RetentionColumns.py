"""Write-maintained luminosity and marginal coverage of indexed contents."""
import heapq
import torch
from torch.nn import functional as F


class RetentionColumns:
    def __init__(self, store):
        self.store = store
        self.records, self.index, self.coordinates = {}, {}, {}
        self.similarity, self.coverage, self.coordinate_contributions = {}, {}, {}
        self.coverage_sum = 0.
        self.coordinate_totals = {}
        self.coordinate_reduced = {}
        self.deducibility = {}
        self.pairs = {}
        self.heaps = {}
        self.revision = 0

    def _candidates(self, record):
        if not record['sign']:
            return set()
        return set().union(*(self.index.get((record['scope'],record['order'],-record['sign'],token),set())
                            for token in record['tokens'])) if record['tokens'] else set()

    def _refresh(self, keys):
        for key in keys:
            if key not in self.records:
                continue
            record = self.records[key]
            candidates = [other for other in self._candidates(record) if other != key
                and self.records[other]['scope']==record['scope']
                and self.records[other]['order']==record['order']
                and self.records[other]['sign']*record['sign']==-1
                and self.records[other]['trust']>record['trust']]
            if candidates:
                self.similarity[key]=max(self.pairs[key][other] for other in candidates)
            else:
                self.similarity[key]=-1.

    def _coordinate(self, coordinate):
        self.coverage_sum -= self.coordinate_totals.pop(coordinate, 0.)
        previous=self.coordinate_contributions.pop(coordinate,{})
        for key,value in previous.items():
            self.coverage[key]=self.coverage.get(key,0.)-value
        items=self.coordinates.get(coordinate,{})
        if not items:
            self.coordinate_reduced.pop(coordinate,None)
            return
        def leaders(pole):
            heap=self.heaps[coordinate][pole]
            def prune():
                while heap and (heap[0][1] not in items or
                        items[heap[0][1]][2] != heap[0][2]):
                    heapq.heappop(heap)
            prune()
            first=heapq.heappop(heap)
            prune()
            second=heap[0] if heap else (0.,None,None)
            heapq.heappush(heap,first)
            return [(-first[0],first[1]),(-second[0],second[1])]
        positive,negative=leaders(0),leaders(1)
        a,b=positive[0][0],negative[0][0]
        values=self.store.slots.new_tensor([a,b])
        full=float(values[0]-values[1]-values.min())
        self.coordinate_totals[coordinate] = full / self.store.nDim
        self.coverage_sum += self.coordinate_totals[coordinate]
        contributions={}
        reduced={}
        for key in {positive[0][1],negative[0][1]}:
            pa=positive[1][0] if len(positive)>1 else 0.
            pb=negative[1][0] if len(negative)>1 else 0.
            pa=pa if positive[0][1]==key else a
            pb=pb if negative[0][1]==key else b
            values=self.store.slots.new_tensor([pa,pb])
            reduced[key]=float(values[0]-values[1]-values.min())
            delta=(full-reduced[key])/self.store.nDim
            contributions[key]=delta
            self.coverage[key]=self.coverage.get(key,0.)+delta
        self.coordinate_contributions[coordinate]=contributions
        self.coordinate_reduced[coordinate]=reduced

    def remove(self, key):
        previous=self.records.pop(key,None)
        if previous is None:
            return
        affected=self._candidates(previous)
        for other in self.pairs.pop(key, {}):
            self.pairs[other].pop(key, None)
        for token in previous['tokens']:
            posting = (previous['scope'],previous['order'],previous['sign'],token)
            self.index[posting].discard(key)
            if not self.index[posting]:del self.index[posting]
        for coordinate in previous['coordinates']:
            self.coordinates[coordinate].pop(key,None)
            self._coordinate(coordinate)
        self.similarity.pop(key,None);self.coverage.pop(key,None);self.deducibility.pop(key,None)
        self._refresh(affected)

    def write(self, row):
        store=self.store
        key=int(store.address_keys[row])
        self.remove(key)
        point=store.slots[row].detach().clone()
        positive,negative=float(store.truth_plus[row]),float(store.truth_minus[row])
        coordinates=point.ne(0).nonzero().flatten().tolist()
        context=store._semantic_rows.get(key,{})
        # Content coordinates are the index fallback for unnamed ideas. They
        # guarantee that a positive cosine candidate cannot be missed.
        tokens={('coordinate',int(i)) for i in coordinates}
        tokens.update(('reference',repr(ref)) for ref in store._context_occurrence_references(context))
        record=dict(point=F.normalize(point,dim=-1),tokens=tokens,coordinates=coordinates,
            trust=max(positive,negative),sign=int(positive>0 and negative==0)-int(negative>0 and positive==0),
            order=store.order_of_row(row),scope=repr((context.get('bindings',()),context.get('scope',()))))
        self.records[key]=record
        for token in tokens:
            self.index.setdefault((record['scope'],record['order'],record['sign'],token),set()).add(key)
        candidates = self._candidates(record) - {key}
        self.pairs[key] = {}
        for other in candidates:
            similarity = float((record['point'] * self.records[other]['point']).sum())
            self.pairs[key][other] = similarity
            self.pairs[other][key] = similarity
        self.revision += 1
        for coordinate in coordinates:
            strength=point[coordinate].abs().clamp(0,1)
            positive_at=float(strength*store.truth_plus[row]);negative_at=float(strength*store.truth_minus[row])
            self.coordinates.setdefault(coordinate,{})[key]=(positive_at,negative_at,self.revision)
            heaps=self.heaps.setdefault(coordinate,([],[]))
            for pole,value in enumerate((positive_at,negative_at)):
                heapq.heappush(heaps[pole],(-value,key,self.revision))
            self._coordinate(coordinate)
        self.deducibility[key]=tuple(store._context_occurrence_references(context))
        self._refresh(self._candidates(record) | {key})

    def values(self, config, *, alive=None):
        keys=self.store.address_keys[:len(self.store)].tolist()
        if any(key not in self.records for key in keys):
            raise RuntimeError('retention columns are missing a committed write')
        selected=set(keys) if alive is None else {keys[i] for i in alive}
        if selected != set(keys):
            # A hypothetical retained subset uses cached similarities and
            # coordinate contributions, never another pairwise product.
            result={key:float(not any(other in selected and
                self.records[other]['trust']>record['trust'] and similarity>config.similarity
                for other,similarity in self.pairs[key].items()))
                for key,record in self.records.items()}
            if config.coverage and selected:
                result.update({key:0. for key in selected})
                totals=self.store.slots.new_zeros(self.store.nDim);overrides={}
                for coordinate,items in self.coordinates.items():
                    poles=[sorted(((pair[pole],key) for key,pair in items.items() if key in selected),reverse=True)
                           for pole in (0,1)]
                    if not poles[0]:continue
                    a,b=poles[0][0][0],poles[1][0][0]
                    values=totals.new_tensor([a,b]);totals[coordinate]=values[0]-values[1]-values.min()
                    for key in {poles[0][0][1],poles[1][0][1]}:
                        reduced=[(entries[1][0] if len(entries)>1 else 0.) if entries[0][1]==key else entries[0][0] for entries in poles]
                        values=totals.new_tensor(reduced)
                        overrides.setdefault(key,{})[coordinate]=values[0]-values[1]-values.min()
                full=totals.mean()
                for key,items in overrides.items():
                    without=totals.clone()
                    for coordinate,value in items.items():without[coordinate]=value
                    result[key]=float((full-without.mean()).clamp(0,1))
                result={key:max(0.,min(1.,value)) for key,value in result.items()}
            return self.store.slots.new_tensor([result[key] for key in keys])
        if config.coverage:
            # Float32's global mean/subtraction must have the same ties as
            # the former pass. Only coordinate leaders can contribute; at
            # most 2*nDim rows need this read of maintained coordinate values.
            totals=self.store.slots.new_zeros(self.store.nDim)
            for coordinate,value in self.coordinate_totals.items():totals[coordinate]=value*self.store.nDim
            full=totals.mean();overrides={}
            for coordinate,items in self.coordinate_reduced.items():
                for key,value in items.items():overrides.setdefault(key,{})[coordinate]=value
            values={}
            for key,items in overrides.items():
                without=totals.clone()
                for coordinate,value in items.items():without[coordinate]=value
                values[key]=float((full-without.mean()).clamp(0,1))
            return self.store.slots.new_tensor([values.get(key,0.) for key in keys])
        return self.store.slots.new_tensor([float(self.similarity[key]<=config.similarity) for key in keys])


def rebuild(store):
    store._retention_columns=RetentionColumns(store)
    for row in range(len(store)):
        store._retention_columns.write(row)


def reindex(store):
    """Compaction preserves stable keys and cached pairwise comparisons."""
    columns = getattr(store, '_retention_columns', None)
    if columns is None:
        rebuild(store)
        return
    keys = set(store.address_keys[:len(store)].tolist())
    for key in set(columns.records) - keys:
        columns.remove(key)
    for row, key in enumerate(store.address_keys[:len(store)].tolist()):
        if key not in columns.records:
            columns.write(row)


def written(store, row):
    if getattr(store,'_retention_columns',None) is None:
        store._retention_columns=RetentionColumns(store)
    store._retention_columns.write(row)
