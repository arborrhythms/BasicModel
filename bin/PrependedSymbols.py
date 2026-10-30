"""Construction-bounded symbol arrays with a leading MTM field."""
import torch
from torch import nn

WORD_FIELDS = ('concept_ids', 'concept_rows', 'concept_orders', 'object_ids',
               'object_rows', 'object_orders', 'symbol_where', 'symbol_when')
LOOKUP_FIELDS = ('lookup_rows', 'lookup_atoms', 'lookup_sentence_ids')


def word_view(source,name):
    """Word-aligned view; plain metadata fixtures need no runtime field owner."""
    entries = getattr(source,'symbol_entries',None)
    return entries.read(name,words=True) if entries is not None else getattr(source,'_ar_word_'+name,None)


def lookup_view(source,name):
    """The lexical inverse bank excludes entries with no independent spelling."""
    entries = getattr(source,'symbol_entries',None)
    return entries.read(name,words=True) if entries is not None else getattr(source,'_ar_concept_'+name,None)


class SymbolEntries(nn.Module):
    def __init__(self, streams, words, prepend, width):
        super().__init__()
        self.streams, self.words, self.prepend = int(streams), int(words), int(prepend)
        if min(self.streams, self.words, self.prepend) < 1:
            raise ValueError('symbol field requires positive construction bounds')
        self.extents = {}
        # B=nStreams; base=mtmPrepend (default 64); W is the largest
        # constructed serial-word bucket. The existing inverse bank contains
        # two W blocks (word and object), so its extent is base+2W.
        for name in WORD_FIELDS:
            band = name in ('symbol_where', 'symbol_when')
            shape = (self.streams, self.prepend+self.words, *((4,) if band else ()))
            self.register_buffer(name, torch.full(shape, 0. if band else -1,
                dtype=torch.float32 if band else torch.long), persistent=False)
        for name in LOOKUP_FIELDS:
            atoms = name == 'lookup_atoms'
            shape = (self.streams, self.prepend+2*self.words, *((int(width),) if atoms else ()))
            self.register_buffer(name, torch.full(shape, 0. if atoms else -1,
                dtype=torch.float32 if atoms else torch.long), persistent=False)
        self.register_buffer('timelines', torch.zeros(self.streams, self.prepend, dtype=torch.long), persistent=False)

    def stage_words(self, name, value):
        target = getattr(self, name)
        fill = 0 if target.is_floating_point() else -1
        if value is None:
            target.detach_()
            target.fill_(fill)
            self.extents.pop(name, None)
            return
        batch, count = value.shape[:2]
        if batch > self.streams or count+self.prepend > target.shape[1] or value.shape[2:] != target.shape[2:]:
            raise ValueError('symbol field exceeds its constructed base + W extent')
        if target.device != value.device or (target.is_floating_point() and target.dtype != value.dtype):
            target = target.to(value)
            setattr(self, name, target)
        target.detach_()
        target.fill_(fill)
        target[:batch, self.prepend:self.prepend+count].copy_(value)
        self.extents[name] = (batch,count)

    def read(self, name, *, words=False):
        extent = self.extents.get(name)
        if extent is None:
            return None
        batch, count = extent
        return getattr(self, name)[:batch, self.prepend if words else 0:self.prepend+count]


def symbolize(owner, store, key, token, timeline=0):
    """The name/mention form beside the selected timeline's working meaning."""
    from IdentityRows import point
    if token.order is not None:
        from FamiliarTier import centroid
        return centroid(owner, token.support)
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    # Legacy numerical fixtures have no separate form/meaning chart; their
    # entire point is content. The paired model supplies both explicit widths.
    paired = derived is not None and derived.meaning_pairs > 0
    width = derived.percept_event_width if paired else 0
    form_width = derived.percept_width if paired else 0
    form = None
    if key[:1] == ('name',):
        name = point(owner, key[1])
        if name is not None:
            form = name[:form_width]
    for occurrence in reversed(token.occurrences):
        reading = owner.mtm.reading_of(occurrence)
        if reading is not None:
            name = point(owner, reading)
            if name is not None:
                form = name[:form_width]
                break
    if form is None:
        forms = [store.slots[row, :form_width] for occurrence in token.occurrences
                 if isinstance((row := store.location(occurrence)), int)]
        form = torch.stack(forms).mean(0) if forms else token.value.new_zeros(form_width)
    value = token.timelines[timeline].value.detach().clone()
    value[:form_width] = form.to(value)
    value[form_width:width] = 0
    return value


def chunk_token(pool, store, state, batch, index):
    """Resolve a seat through its retained occurrence before a recycled id.

    A published identity uses its second reference for the pool origin. Recent
    entries already carry an occurrence in their first reference. No epoch or
    additional identity/value slab is needed to reject a reused transient id.
    """
    import STMChunks as chunks
    refs = state[chunks.REFS][batch,index]
    if int(refs[1,0]) == chunks.IDENTITY_REF:
        code = int(refs[1,1])
        key = ('candidate', code) if code >= 0 else ('name', -code-1)
        return key if key in pool.entries else None
    for position in (1,0):
        if int(refs[position,0]) == chunks.OCCURRENCE:
            occurrence = ('ltm',store.address_domain,int(refs[position,1]))
            return pool.token_of(occurrence)
    identity = int(state[chunks.IDENTITY][batch,index])
    return pool.token_of(identity) if identity >= 0 else None


def reconcile(state, pool, store):
    """Retire expired field identities and remap promoted seats at staging."""
    import STMChunks as chunks
    result = [value.clone() for value in state]
    for b in range(state[0].shape[0]):
        for index in range(state[0].shape[1]):
            refs = state[chunks.REFS][b,index]
            if not bool((refs[:,0] == chunks.OCCURRENCE).any()) and int(refs[1,0]) != chunks.IDENTITY_REF:
                continue
            key = chunk_token(pool,store,state,b,index)
            entry = pool.entries.get(key)
            identity = -1 if entry is None else entry.symbol if entry.symbol is not None else entry.transient
            result[chunks.IDENTITY][b,index] = identity
            if int(refs[0,0]) in (chunks.SYMBOL,chunks.IDENTITY_REF):
                if entry is None:
                    result[chunks.REFS][b,index,0] = 0
                else:
                    result[chunks.REFS][b,index,0,0] = chunks.SYMBOL if entry.symbol is not None else chunks.IDENTITY_REF
                    result[chunks.REFS][b,index,0,1] = identity
    return tuple(result)


def stage(model, state, present):
    """Host boundary: prepend live identities, then deposit them off stack."""
    import STMChunks as chunks
    from RowTime import encoder
    owner, isp = model._concept_owner(), model.inputSpace
    entries = isp.symbol_entries
    store, pool = model.symbolSpace.ltm_store, owner.mtm
    state = reconcile(state,pool,store)
    resolver = owner.components.reference_close
    stamp = encoder().encode(int(store.model_clock)).to(state[0])
    for field in (*WORD_FIELDS, *LOOKUP_FIELDS):
        value = getattr(entries, field)
        value[:, :entries.prepend].fill_(0 if value.is_floating_point() else -1)
    entries.timelines.zero_()
    staged = torch.zeros_like(state[chunks.READ], dtype=torch.bool)
    for b in range(state[0].shape[0]):
        if not bool(present[b]):
            continue
        records = resolver.accessible(tuple(value[b:b+1] for value in state), stream=b, include_pool=True)
        keys = {record.get('token') for _, record in records.values()} - {None}
        keys.update(key for key, token in pool.entries.items() if (token.order is not None or key[0] == 'name')
                    and (not pool.accesses.get(key) or b in pool.accesses[key]))
        keys = sorted(keys, key=lambda key: (max((event.when for timeline in pool.entries[key].timelines
            for event in timeline.events), default=pool.entries[key].when), pool.entries[key].when), reverse=True)
        choices = [(key, timeline) for key in keys for timeline in range(len(pool.entries[key].timelines))]
        for column, (key, timeline) in enumerate(choices[:entries.prepend]):
            token = pool.entries[key]
            identity = token.symbol if token.symbol is not None else token.transient
            if identity is None:
                raise RuntimeError('a prepended identity has neither a transient id nor a symbol')
            row = owner._csw_row_of(identity) if token.symbol is not None else None
            order = (token.order if token.order is not None else owner._row_order(row) if row is not None
                     else owner._concept_source_order(key[1]) if key[0] == 'name' else 1)
            value = symbolize(owner, store, key, token, timeline).to(state[0])
            where = token.where.to(value)
            if store.when_start is not None and token.order is None:
                value[store.when_start:store.when_start+4] = stamp
                if store.when_start >= 4:
                    value[store.when_start-4:store.when_start] = where
            for field, member in (('concept_ids',identity), ('object_ids',identity),
                    ('concept_rows',-1 if row is None else row), ('object_rows',-1 if row is None else row),
                    ('concept_orders',order), ('object_orders',order), ('symbol_where',where), ('symbol_when',stamp),
                    ('lookup_rows',-1 if row is None else row), ('lookup_atoms',value),
                    ('lookup_sentence_ids',model._open_sentence_slot)):
                getattr(entries, field)[b,column] = member
            entries.timelines[b,column] = timeline
            gate = torch.arange(len(present),device=state[0].device) == b
            refs = state[chunks.REFS].new_zeros(len(present),2,2)
            refs[:,0,0] = chunks.SYMBOL if token.symbol is not None else chunks.IDENTITY_REF
            refs[:,0,1] = identity
            refs[:,1,0] = chunks.IDENTITY_REF if token.order is not None or key[0] == 'name' else chunks.OCCURRENCE
            refs[:,1,1] = -key[1]-1 if key[0] == 'name' else key[-1]
            # Copying the working value from MTM is an arrival/read. Merely
            # enumerating the subsequent bind menu does not read this seat.
            match = ((state[chunks.IDENTITY][b] == identity) & (state[chunks.TIMELINE][b] == timeline)
                     & (state[chunks.STACK][b] < 0)
                     & ((state[chunks.REFS][b,:,0,0] == chunks.SYMBOL)
                        | (state[chunks.REFS][b,:,0,0] == chunks.IDENTITY_REF)))
            if bool(match.any()):
                index = torch.full((len(present),), int(match.long().argmax()), device=state[0].device)
                result = list(state)
                for field, member in ((chunks.BUFFER,value), (chunks.WHERE,where), (chunks.WHEN,stamp), (chunks.REFS,refs[b])):
                    result[field] = chunks.write(result[field], index, member.expand(len(present),*member.shape), gate)
                state = tuple(result)
                state = chunks.read(state,index,gate)
            else:
                state, index = chunks.deposit(state, value.expand(len(present),-1), gate,
                    order=order, row=-1 if row is None else row, refs=refs, identity=identity, timeline=timeline,
                    where=where, when=stamp)
            if int(index[b]) >= 0:
                staged[b,index[b]] = True
    # The leading field arrives at one boundary. Its order in the array is
    # event order, not an artificial sequence of attention reads.
    indices = torch.arange(state[0].shape[1],device=state[0].device)[None].expand_as(staged)
    state = chunks.read(state,indices,staged)
    return state
