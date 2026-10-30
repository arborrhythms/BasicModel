"""Grammar provenance for the shared chunks, in the existing scope and journal.

There is no identity value bank here. Current values and alternative timelines
are ordinary STM chunks; old numerical operands are the operation journal.
Only a kept Reading is translated to the closing transaction's host record.
"""
import torch
from torch import nn

import STMChunks as chunks

# The first two scope columns remain ClauseScope's flags/reference. The rest
# describe grammatical heads, not a second store of their numerical values.
IDENTITY, VERB, OBJECT, RESOLVED, WORD, SOURCE, LEAVES, TIMELINE, MODAL, EVENT, UNLOWERED, ORIGIN = range(2, 14)
SCOPE_WIDTH = 14
# refs[2], transaction[6], timeline[4], two operands' provenance[12],
# and one publication-kind code. The last code recovers opening/equality/part values
# from the numerical journal even after their working chunk is evicted.
RECORD_WIDTH = 2 + 6 + 4 + 2*(SCOPE_WIDTH-2) + 1


def empty_scope(like):
    from ClauseScope import ClauseScope
    return torch.cat((ClauseScope.empty(like), torch.zeros(
        *like.shape[:2], SCOPE_WIDTH-2, dtype=torch.long, device=like.device)), -1)


def read_identity(state, identities, timelines):
    """Prefer the off-stack field entry; an on-stack bound value is usable too."""
    match = ((identities[..., None] == state[chunks.IDENTITY][:, None])
             & (timelines[..., None] == state[chunks.TIMELINE][:, None])
             & (identities[..., None] >= 0))
    off = match & (state[chunks.STACK][:, None] < 0)
    field = off & ((state[chunks.REFS][:,None,:,0,0] == chunks.SYMBOL)
                   | (state[chunks.REFS][:,None,:,0,0] == chunks.IDENTITY_REF))
    off = torch.where(field.any(-1)[...,None],field,off)
    match = torch.where(off.any(-1)[..., None], off, match)
    indices = match.long().argmax(-1)
    return chunks.gather(state[0], indices), match.any(-1), indices


def fresh_identity(state, records, free_ids, *, trial_base=None, trial_limit=None):
    """Reserve a free pool id in this fork's chunks/journal, never in MTM."""
    used = (state[chunks.IDENTITY][..., None] == free_ids).any(1)
    # A journal can refer to an evicted introduction, so its id cannot be
    # reused within the same open reading. All six integer event fields are
    # checked; false conflicts with predicate symbols only withhold a seat.
    used |= (records[..., 2:8, None] == free_ids).any(1).any(1)
    used |= (records[:, :, (12,24), None] == free_ids).any(1).any(1)
    available = (free_ids >= 0) & ~used
    index = available.long().argmax(-1)
    candidate = free_ids.expand(used.shape[0],-1).gather(1,index[:,None])[:,0]
    fallback = torch.full_like(candidate,-1)
    if trial_base is not None:
        # One maximum over provenance replaces a quadratic candidate matrix
        # for journal-local names. Pool candidates remain bounded by mtmCapacity.
        latest = torch.maximum(state[chunks.IDENTITY].amax(-1),
            records[:,:,(2,3,4,6,12,24)].flatten(1).amax(-1))
        fallback = (latest+1).clamp_min(trial_base)
        fallback = torch.where(fallback < trial_limit,fallback,-1)
    return torch.where(available.any(-1),candidate,fallback)


def push(state, scope, gate, *, word, index, resolved, free_ids, records, trial_base=None, trial_limit=None, pending_ids=None, lexical_word=None):
    """Give a shifted individual a trial identity without admitting a concept."""
    physical = chunks.at(state)
    order = chunks.gather(state[chunks.ORDERS], physical.clamp_min(0)[:,None])[:,0]
    pending = (torch.zeros_like(word, dtype=torch.bool) if pending_ids is None else
               (word[:, None] == pending_ids[None]).any(-1) & word.ge(0))
    named = gate.reshape(-1).bool() & ((order == 1) | pending)
    identity = fresh_identity(state, records, free_ids,trial_base=trial_base,trial_limit=trial_limit)
    torch._assert_async((~named | (identity >= 0)).all(), 'open reading exhausted its constructed transient id pool')
    identity = torch.where(pending, word, torch.where(named, identity, -1))
    resolved = resolved & ~pending
    if lexical_word is not None:
        word = torch.where(pending & (lexical_word >= 0), lexical_word, word)
    result = list(state)
    result[chunks.IDENTITY] = chunks.write(state[chunks.IDENTITY], physical, identity, gate)
    refs = state[chunks.REFS].new_zeros(len(word),2,2)
    refs[:,0,0] = torch.where(resolved, chunks.SYMBOL, chunks.ACTION)
    refs[:,0,1] = torch.where(resolved, word, -index.expand_as(word)-1)
    result[chunks.REFS] = chunks.write(state[chunks.REFS], physical, refs, gate)
    member = scope.new_zeros(len(word), SCOPE_WIDTH-2)
    for field,value in ((IDENTITY,identity),(VERB,word),(RESOLVED,resolved.long()),
                        (WORD,word),(SOURCE,-index.expand_as(word)-1),(LEAVES,torch.ones_like(word)),
                        (UNLOWERED,order)):
        member[:,field-2] = value
    metadata = torch.cat((member[:,None], scope[:, 1:, 2:]), 1)
    return tuple(result), torch.cat((scope[:,:,:2], torch.where(gate[:,None,None],metadata,scope[:,:,2:])), -1)


class ChunkJournal(nn.Module):
    def __init__(self, binary, unary, *, pool_capacity=64, percept_band=None):
        super().__init__()
        self.percept_band = percept_band
        # At most mtmCapacity unresolved pool identities can be prepended.
        # This read-only sentence snapshot carries ids, never working values.
        self.register_buffer('pending_ids',torch.full((int(pool_capacity),),-1,dtype=torch.long),persistent=False)
        self.trial_base = self.trial_limit = None
        def table(rules):
            return torch.tensor([(
                getattr(rule,'head_role',0),
                getattr(rule,'determiner_mode',None) in ('mint','bind'),
                rule.method_name == 'bind',
                getattr(rule,'clause_form',None) == 'VP',
                getattr(rule,'clause_form',None) == 'S',
                getattr(rule,'relation_kind',None) == 'equal',
                getattr(rule,'relation_kind',None) in ('part','whole','equal'),
                getattr(rule,'relation_kind',None) in ('part','whole'),
                rule.method_name == 'adverb',
            ) for rule in rules] or [(0,)*9], dtype=torch.long)
        self.register_buffer('binary',table(binary),persistent=False)
        self.register_buffer('unary',table(unary),persistent=False)

    def rule(self, choice):
        return torch.where((choice.kind == 1)[:,None],
            self.binary[choice.local_op.clamp(0,len(self.binary)-1)],
            self.unary[choice.local_op.clamp(0,len(self.unary)-1)])

    @staticmethod
    def operands(scope, choice):
        unary = chunks.gather(scope, choice.position.clamp(0,scope.shape[1]-1)[:,None])[:,0]
        pair = choice.kind == 1
        if choice.stack_arity is not None:
            pair &= choice.stack_arity == 2
        left = torch.where(pair[:,None],scope[:,min(1,scope.shape[1]-1)],unary)
        return left, torch.where(pair[:,None],scope[:,0],unary)

    def choice(self, scope, selected):
        rule = self.rule(selected)
        left,right = self.operands(scope,selected)
        head = torch.where((rule[:,0] == 2)[:,None],right,left)
        event = rule[:,4].bool() | rule[:,5].bool() | rule[:,7].bool()
        post = rule[:,8].bool() & (left[:,EVENT] > 0)
        modal = torch.where(event,right[:,MODAL].bool() & rule[:,4].bool(),head[:,MODAL].bool()) | rule[:,8].bool()
        timeline = torch.where(event,left[:,TIMELINE],head[:,TIMELINE])
        timeline = torch.where(modal,torch.ones_like(timeline),timeline)
        unlowered = torch.where(rule[:,4].bool(),left[:,UNLOWERED],selected.order)
        unlowered = torch.where(post,left[:,UNLOWERED],unlowered)
        order = torch.where(rule[:,4].bool() & (modal | (right[:,TIMELINE] > 0)),unlowered,selected.order)
        order = torch.where(post,unlowered,order)
        return selected._replace(order=order,timeline=torch.stack((timeline,modal.long(),head[:,EVENT],unlowered),-1))

    def publish(self, state, identity, timeline, value, active, *, resolved=True):
        match = ((state[chunks.IDENTITY] == identity[:,None]) & (state[chunks.TIMELINE] == timeline[:,None])
                 & (state[chunks.STACK] < 0) & (identity[:,None] >= 0))
        field = match & ((state[chunks.REFS][:,:,0,0] == chunks.SYMBOL)
                         | (state[chunks.REFS][:,:,0,0] == chunks.IDENTITY_REF))
        match = torch.where(field.any(-1)[:,None],field,match)
        found = match.any(-1)
        index = match.long().argmax(-1)
        # A new identity inherits the mentioning head's percept stamp. A
        # refresh keeps the field entry's stamp; it does not enter STM again.
        head = chunks.at(state).clamp_min(0)
        source = torch.where(found,index,head)
        where = chunks.gather(state[chunks.WHERE],source[:,None])[:,0]
        when = chunks.gather(state[chunks.WHEN],source[:,None])[:,0]
        if self.percept_band is not None and value.shape[-1] == self.percept_band[2]:
            start,end,_ = self.percept_band
            value = torch.cat((value[:,:start],where,when,value[:,end:]),-1)
        result = list(state)
        result[0] = chunks.write(state[0],index,value,active & found)
        state = chunks.read(tuple(result),index,active & found)
        refs = state[chunks.REFS].new_zeros(len(identity),2,2)
        refs[:,0,0],refs[:,0,1] = chunks.IDENTITY_REF,identity
        return chunks.deposit(state,value,active & ~found & (identity >= 0),identity=identity,
            timeline=timeline,refs=refs,order=torch.ones_like(identity),where=where,when=when)[0]

    def apply(self, state, scope, selected, operands, references, records, values, *, column, free_ids, clause_scope):
        """Apply one kept operation inside a trial and journal its transaction."""
        batch,size = scope.shape[:2]
        rule = self.rule(selected)
        left,right = self.operands(scope,selected)
        active = selected.applied & ((selected.kind == 1) | (selected.kind == 2))
        # Determination opens an individual only when its looked-up member
        # order is one. An order-zero event stays an event; allocating a
        # token there would leave a later bind without a closing introduction.
        bind,determined = active & rule[:,2].bool(), active & rule[:,1].bool() & selected.order.eq(1)
        pair = active & (selected.kind == 1) & ~bind
        head = torch.where((rule[:,0] == 2)[:,None],right,left)
        parent = torch.where(((selected.kind == 2) | (rule[:,0] > 0))[:,None],head,0)
        chosen = torch.full_like(left[:,IDENTITY],-1) if selected.identity is None else selected.identity
        fresh = fresh_identity(state,records,free_ids,trial_base=self.trial_base,trial_limit=self.trial_limit)
        identity = torch.where(bind & (chosen > 0),chosen,
                    torch.where(bind & (head[:,IDENTITY] >= 0),head[:,IDENTITY],fresh))
        opening = bind | determined
        torch._assert_async((~opening | (identity >= 0)).all(), 'open reading exhausted its constructed transient id pool')
        parent = parent.clone()
        parent[:,IDENTITY] = torch.where(opening,identity,parent[:,IDENTITY])
        provenance = torch.cat((scope[:,:,2:],records[:,:,12:24],records[:,:,24:36]),1)
        same = provenance[:,:,IDENTITY-2] == chosen[:,None]
        known = (same & provenance[:,:,RESOLVED-2].eq(1)).any(-1)
        changed_before = ((records[:,:,3] == chosen[:,None]) & records[:,:,7].eq(1)
                          & records[:,:,11].eq(0)).any(-1)
        staged_resolved = ~(chosen[:,None] == self.pending_ids[None]).any(-1)
        resolved = torch.where(same.any(-1),known,staged_resolved) | changed_before
        parent[:,RESOLVED] = torch.where(bind & (chosen > 0),resolved.long(),parent[:,RESOLVED])
        parent[:,SOURCE] = column+1
        parent[:,LEAVES] = torch.where(pair,left[:,LEAVES]+right[:,LEAVES],left[:,LEAVES])
        timeline = head[:,TIMELINE]
        if selected.chunk is not None:
            selected_timeline = chunks.gather(state[chunks.TIMELINE],selected.chunk.clamp_min(0)[:,None])[:,0]
            timeline = torch.where(bind & (selected.chunk >= 0),selected_timeline,timeline)
        modal = head[:,MODAL]
        origin = head[:,ORIGIN]
        unlowered = selected.order
        if selected.timeline is not None:
            timeline = torch.where(bind,timeline,selected.timeline[:,0])
            modal,unlowered = selected.timeline[:,1],selected.timeline[:,3]
        reset = bind | determined
        modal = torch.where(reset,0,modal)
        vp = active & rule[:,3].bool()
        parent[:,IDENTITY] = torch.where(vp,-1,parent[:,IDENTITY])
        parent[:,VERB] = torch.where(vp,left[:,VERB],parent[:,VERB])
        parent[:,OBJECT] = torch.where(vp,right[:,IDENTITY],parent[:,OBJECT])
        parent[:,RESOLVED] = torch.where(vp,left[:,RESOLVED] & right[:,RESOLVED],parent[:,RESOLVED])
        sentence = active & (rule[:,4].bool() | rule[:,5].bool() | rule[:,7].bool())
        post = active & rule[:,8].bool() & (left[:,EVENT] > 0)
        inherited = active & ~reset & ~sentence & (head[:,EVENT] > 0) & ((rule[:,0] > 0) | (selected.kind == 2))
        # Ordinary surface attachment carries the event's provenance, not a
        # new state change. Only a wrapper of an alternative updates that
        # alternative, as the retired timeline owner did.
        wrapped = inherited & (timeline > 0)
        event = sentence | post | wrapped
        subject = torch.where(sentence,left[:,IDENTITY],head[:,IDENTITY])
        obj = torch.where(rule[:,5].bool(),right[:,IDENTITY],right[:,OBJECT])
        verb = right[:,VERB]
        obj = torch.where(sentence,obj,head[:,OBJECT])
        verb = torch.where(sentence,verb,head[:,VERB])
        parent[:,IDENTITY] = torch.where(event,subject,parent[:,IDENTITY])
        parent[:,OBJECT] = torch.where(event,obj,parent[:,OBJECT])
        parent[:,VERB] = torch.where(event,verb,parent[:,VERB])
        parent[:,RESOLVED] = torch.where(sentence,right[:,RESOLVED],parent[:,RESOLVED])
        origin = torch.where(sentence,column+1,origin)
        parent[:,EVENT] = torch.where(event,column+1,torch.where(reset,0,parent[:,EVENT]))
        parent[:,ORIGIN] = origin
        parent[:,TIMELINE],parent[:,MODAL],parent[:,UNLOWERED] = timeline,modal,unlowered
        result = torch.where(rule[:,5,None].bool(),operands[:,1],selected.candidate)
        result = torch.where(rule[:,7,None].bool(),torch.maximum(operands[:,0],operands[:,1]),result)
        update = event & (subject >= 0) & parent[:,RESOLVED].bool()
        state = chunks.apply_choice(state,selected,operand_actions=torch.stack((left[:,SOURCE],right[:,SOURCE]),-1))
        physical = chunks.at(state,torch.where(pair,0,selected.position))
        changed = list(state)
        if self.percept_band is not None and state[0].shape[-1] == self.percept_band[2]:
            start,end,_ = self.percept_band
            source = physical.clamp_min(0)[:,None]
            value = chunks.gather(state[0],source)[:,0]
            where = chunks.gather(state[chunks.WHERE],source)[:,0]
            when = chunks.gather(state[chunks.WHEN],source)[:,0]
            changed[0] = chunks.write(state[0],physical,
                torch.cat((value[:,:start],where,when,value[:,end:]),-1),active)
        for field,member in ((chunks.IDENTITY,parent[:,IDENTITY]),(chunks.TIMELINE,timeline)):
            changed[field] = chunks.write(state[field],physical,member,active)
        state = tuple(changed)
        # An S modifier moves its covered actual event into a second chunk.
        # The earlier operand remains in the operation journal, so rollback
        # needs no live shadow/before value arrays.
        prior = (left[:,EVENT]-1).clamp(0,records.shape[1]-1)
        full_values,local_values,local_columns = values
        prior_values = chunks.gather(full_values,prior[:,None])[:,0]
        local_match = local_columns == prior[:,None]
        local_prior = chunks.gather(local_values,local_match.long().argmax(-1)[:,None])[:,0]
        prior_values = torch.where(local_match.any(-1)[:,None],local_prior,prior_values).reshape(batch,3,-1)[:,0]
        later_actual = ((torch.arange(records.shape[1],device=records.device)[None] > prior[:,None])
            & (records[:,:,3] == subject[:,None]) & records[:,:,7].eq(1)
            & records[:,:,9].eq(0) & records[:,:,11].eq(0)).any(-1)
        state = self.publish(state,subject,torch.zeros_like(timeline),prior_values,
            post & (left[:,TIMELINE] == 0) & ~later_actual)
        suppress = (post | wrapped) & (head[:,EVENT] > 0)
        mask = suppress[:,None] & (torch.arange(records.shape[1],device=records.device)[None] == head[:,EVENT,None]-1)
        records = records.clone()
        records[:,:,11] = torch.where(mask,1,records[:,:,11])
        # Bind's result reads the chosen value. Publication joins the new
        # description into that identity before the next operand refresh.
        joined = torch.where((bind & (chosen > 0) & (timeline == 0))[:,None],
                             torch.maximum(selected.candidate,operands[:,0]),selected.candidate)
        state = self.publish(state,identity,timeline,joined,opening)
        state = self.publish(state,subject,timeline,result,update)
        event_record = torch.stack((parent[:,IDENTITY],torch.where(event,subject,-1),
            torch.where(event,obj,-1),torch.where(event,verb,-1),torch.where(bind,chosen,-1),update.long()),-1)
        timeline_record = torch.stack((selected.order,timeline,origin,torch.zeros_like(origin)),-1)
        publication = torch.where(opening,3,torch.where(rule[:,5].bool(),1,torch.where(rule[:,7].bool(),2,0)))
        record = torch.cat((references,event_record,timeline_record,left[:,2:],right[:,2:],publication[:,None]),-1)
        records = chunks.write(records,column,record,selected.valid)
        base,ended = clause_scope.apply(scope[:,:,:2],selected,column)
        metadata = scope[:,:,2:]
        rewritten = torch.where((active & ~pair)[:,None,None] &
            (torch.arange(size,device=scope.device)[None,:,None] == selected.position[:,None,None]),parent[:,None,2:],metadata)
        folded = torch.cat((parent[:,None,2:],metadata[:,2:],torch.zeros_like(metadata[:,:1])),1)[:,:size]
        metadata = torch.where(pair[:,None,None],folded,rewritten)
        swapped = torch.cat((scope[:,1:2,2:],scope[:,:1,2:],scope[:,2:,2:]),1)
        metadata = torch.where((selected.applied & (selected.kind == 3))[:,None,None],swapped,metadata)
        return state,torch.cat((base,metadata),-1),records,ended


def leaf_orders(scope,records,*,batch,positions,orders):
    """Read the shifted order before lexical reference admission can change it."""
    members = torch.cat((records[batch,:,12:24],records[batch,:,24:36],scope[batch,:,2:]),0)
    stamps = positions.new_full(positions.shape,-1) if orders is None else orders.clone()
    for index,word in enumerate(positions.tolist()):
        shifted = members[(members[:,SOURCE-2] == -word-1) & (members[:,LEAVES-2] > 0)]
        if len(shifted):
            stamps[index] = shifted[0,UNLOWERED-2]
    return stamps


def capture(owner, state, scope, records, values, *, batch, positions, columns, actions):
    """Translate the kept journal to closing handles, after the tensor walk.

    These compact host records have the Reading's lifetime. Their values are
    read from chunks or the existing numerical journal, never replayed.
    """
    from IdentityRows import allocator
    pool = owner.mtm
    alloc = allocator(owner)
    row = records[batch]
    selected = row.index_select(0,columns.clamp_min(0))
    metadata = torch.cat((row[:,12:24],row[:,24:36],scope[batch,:,2:]),0)
    ids = []
    for member in metadata[:,IDENTITY-2].tolist():
        if member > 0 and member not in ids:
            ids.append(member)
    for member in selected[:,(2,3,4,6)].reshape(-1).tolist():
        if member > 0 and member not in ids:
            ids.append(member)
    ids.sort(key=lambda identity: pool.token_of(identity) is None)
    handles = {identity:i+1 for i,identity in enumerate(ids)}
    origins = row.new_zeros(len(ids),3)
    actual,alternative = [],[]
    by_column = {int(column):index for index,column in enumerate(columns.tolist())
                 if int(actions[index,0]) in (1,2)}
    pos = {int(position):index for index,position in enumerate(positions.tolist())}
    leaves = row.new_zeros(len(positions))
    for identity in ids:
        index = handles[identity]-1
        members = metadata[metadata[:,IDENTITY-2] == identity]
        token = pool.token_of(identity)
        introduction = None
        if token is not None:
            entry = pool.entries[token]
            # Candidate counters are pool keys, never LTM occurrence addresses
            # (in particular, candidate zero is a valid retained candidate).
            origins[index] = row.new_tensor((5 if entry.order is not None else 6 if token[0] == 'name' else 2,
                token[-1], not entry.pending))
            if entry.pending and entry.order is None:
                for shifted in members[members[:,SOURCE-2] < 0]:
                    source = pos.get(-int(shifted[SOURCE-2])-1)
                    if source is not None:
                        leaves[source] = index+1
        elif identity in alloc.placement and identity not in alloc.retired:
            # A durable symbol can be recalled without a current MTM token.
            # Its first bind in this journal is a use, not an introduction;
            # the closing must keep that symbol even if its chunk is evicted.
            origins[index] = row.new_tensor((1, identity, 1))
        else:
            shifted = members[members[:,SOURCE-2] < 0]
            if len(shifted):
                source = int(shifted[0,SOURCE-2])
                word = -source-1
                introduction = pos.get(word)
                if introduction is not None:
                    leaves[introduction] = index+1
                origins[index] = row.new_tensor((4,word,int(shifted[0,RESOLVED-2])))
            else:
                introduced = ((selected[:,2] == identity) & (selected[:,3] < 0)).nonzero().flatten()
                action = int(introduced[0]) if len(introduced) else 0
                origins[index] = row.new_tensor((3,action,int(members[:,RESOLVED-2].any()) if len(members) else 0))
        base = (pool.entries[token].value.to(values) if token is not None else values.new_zeros(values.shape[-1]//3))
        branch = (pool.entries[token].timelines[-1].value.to(values)
                  if token is not None and len(pool.entries[token].timelines)>1 else base)
        for column in range(row.shape[0]):
            transaction = row[column]
            if int(transaction[2]) == identity and int(transaction[36]) == 3:
                operands = values[batch,column].reshape(3,-1)
                base = torch.maximum(operands[0],operands[2]) if int(transaction[6]) > 0 else operands[2]
            if int(transaction[3]) == identity and bool(transaction[7]) and not bool(transaction[11]):
                operands = values[batch,column].reshape(3,-1)
                publication = int(transaction[36])
                current = (operands[1] if publication == 1 else
                    torch.maximum(operands[0],operands[1]) if publication == 2 else operands[2])
                if int(transaction[9]) > 0:
                    branch = current
                else:
                    base = current
        # Publication can encode equality/part differently from its ordinary
        # grammatical result. The resident chunk is that exact working value.
        for timeline in (0,1):
            match = ((state[chunks.IDENTITY][batch] == identity) & (state[chunks.TIMELINE][batch] == timeline)
                     & (state[chunks.STACK][batch] < 0))
            field = match & ((state[chunks.REFS][batch,:,0,0] == chunks.SYMBOL)
                            | (state[chunks.REFS][batch,:,0,0] == chunks.IDENTITY_REF))
            if bool(field.any()):
                match = field
            if bool(match.any()):
                current = state[0][batch,int(match.long().argmax())]
                if timeline:
                    branch = current
                else:
                    base = current
        actual.append(base)
        alternative.append(branch)
    events = selected[:,2:8].clone()
    for column in (0,1,2,4):
        events[:,column] = events.new_tensor([handles.get(int(identity),0 if column != 4 or int(identity)==0 else -1)
                                              for identity in events[:,column]])
    events = torch.where((actions[:,0] > 0)[:,None],events,0)
    timeline = selected[:,8:12].clone()
    timeline[:,2] = timeline.new_tensor([by_column.get(int(origin)-1,-1)+1 for origin in timeline[:,2]])
    timeline = torch.where((actions[:,0] > 0)[:,None],timeline,0)
    width = values.shape[-1]//3
    return dict(identity_ids=row.new_tensor(ids),identity_origins=origins,identity_values=torch.stack(actual) if actual else values.new_zeros(0,width),
        identity_actions=events,identity_leaves=leaves,timeline_actions=timeline,
        timeline_values=torch.stack(alternative) if alternative else values.new_zeros(0,width))
