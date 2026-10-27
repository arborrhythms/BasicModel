"""Fixed choices through the production chunk and journal operations.

The capacities come from each fixture's declared words/actions and eight
shared chunks. These helpers retain no independent identity value bank.
"""
from dataclasses import replace
from types import SimpleNamespace
import torch

import STMChunks as chunks
from ChunkJournal import RECORD_WIDTH, ChunkJournal, empty_scope, push, capture, read_identity
from ClauseScope import ClauseScope
from Spaces import LanguageOperationChoice


def context(resolver, frames=()):
    if len(frames) == 13 and torch.is_tensor(frames[0]):
        return frames
    store, pool = resolver.store, resolver.components.owner.mtm
    state = chunks.empty(1,8,store.nDim,device=store.slots.device,dtype=store.slots.dtype)
    for frame in frames:
        if not hasattr(frame,'row_id'):
            frame = frame.read(store)
        if frame is None:
            continue
        from IdentityRows import point as identity_point
        for symbol in (() if getattr(frame,'activations',()) or store.location(frame.row_id) is not None
                       else getattr(frame,'symbol_ids',())):
            value = identity_point(resolver.components.owner,symbol)
            if value is not None:
                state,_ = chunks.deposit(state,value[None],torch.tensor([True]),identity=symbol,
                    refs=torch.tensor([[[chunks.SYMBOL,symbol],[0,0]]]),order=1)
        location = store.location(frame.row_id)
        if location is None:
            continue
        occurrence = store.occurrence_of(location)
        from ChainRead import head_idea
        point = head_idea(store,location[1]) if isinstance(location,tuple) else store.slots[location]
        if point is None:
            point = store.slots.new_zeros(store.nDim)
        references = torch.tensor([[[chunks.RELATION if isinstance(location,tuple) else chunks.OCCURRENCE,
                                    occurrence[-1]],[0,0]]])
        token = pool.token_of(occurrence)
        for reference, _, role in getattr(frame,'activations',()):
            if role == 0 and pool.token_of(reference) is not None:
                token = pool.token_of(reference)
                break
        entry = pool.entries.get(token)
        if entry is not None:
            references[0,1] = references.new_tensor((chunks.OCCURRENCE,token[-1]))
        identity = -1 if entry is None else entry.symbol if entry.symbol is not None else entry.transient
        state,_ = chunks.deposit(state,point[None],torch.tensor([True]),refs=references,
            identity=identity,order=frame.order,when=store.entry_when(location))
        # An object participant is not the sentence's subject chunk. Recall
        # it into its own shared seat, as production query/prepend does.
        for reference, _, role in getattr(frame,'activations',()):
            key = pool.token_of(reference)
            if role == 0 or key is None:
                continue
            entry = pool.entries[key]
            identity = entry.symbol if entry.symbol is not None else entry.transient
            if bool((state[chunks.IDENTITY] == identity).any()):
                continue
            refs = torch.tensor([[[chunks.SYMBOL if entry.symbol is not None else chunks.IDENTITY_REF,identity],
                                  [chunks.OCCURRENCE,key[-1]]]])
            state,_ = chunks.deposit(state,entry.value[None],torch.tensor([True]),refs=refs,
                identity=identity,order=1,where=entry.where,when=store.entry_when(location))
    return state


def recall(resolver, frames):
    """Explicit query-style recall before a fixture reads an unpooled row."""
    from QueryWork import QueryWorkBudget
    rows = [resolver.store.row(resolver.store.location(frame.row_id)) for frame in frames]
    # Each input has at most the fixture's whole fixed store of references.
    work = QueryWorkBudget(max(1,len(rows)*(resolver.store.capacity+resolver.store.parts.capacity)))
    resolver.store.recall_identities(rows,work=work)
    return context(resolver,frames)


def prepend(f, frames, words):
    from PrependedSymbols import SymbolEntries, stage
    resolver = f.cs.components.reference_close
    state = context(resolver,frames)
    entries = SymbolEntries(1,max(1,words),8,f.store.nDim)
    for name in ('concept_ids','concept_rows','concept_orders','object_ids','object_rows','object_orders'):
        entries.stage_words(name,torch.full((1,max(1,words)),-1,dtype=torch.long))
    model = SimpleNamespace(_concept_owner=lambda:f.cs,inputSpace=SimpleNamespace(symbol_entries=entries),
                            symbolSpace=SimpleNamespace(ltm_store=f.store),_open_sentence_slot=0)
    return stage(model,state,torch.tensor([True]))


class Walk:
    def __init__(self, binary, unary, *, capacity=64, first_id=512):
        self.binary,self.unary = binary,unary
        self.journal = ChunkJournal(binary,unary)
        self.scope = ClauseScope(binary,unary)
        self.free = torch.arange(first_id,first_id+capacity)
        self.journal.trial_base = first_id+capacity
        self.journal.trial_limit = first_id+capacity+2*capacity

    def empty(self, like, columns, *, state=None):
        state = chunks.empty(len(like),8,like.shape[-1],device=like.device,dtype=like.dtype) if state is None else state
        # Two references, six transaction fields, four timeline fields and
        # two twelve-column provenance records per declared action.
        records = torch.full((len(like),columns,RECORD_WIDTH),-1,dtype=torch.long,device=like.device)
        return state,empty_scope(state[0]),records,like.new_zeros(len(like),columns,3*like.shape[-1])

    def shift(self,state,value,*,word,index,order=1,resolved=True):
        stm,scope,records,values = state
        active = torch.ones(len(value),dtype=torch.bool,device=value.device)
        word = torch.as_tensor(word,device=value.device).expand(len(value))
        stm,_ = chunks.deposit(stm,value,active,order=order,grammar_order=0,stack=True)
        scope = self.scope.push(scope,active,word)
        stm,scope = push(stm,scope,active,word=word,index=torch.as_tensor(index,device=value.device),
            resolved=torch.as_tensor(resolved,device=value.device).expand(len(value)),free_ids=self.free,records=records,trial_base=self.journal.trial_base,trial_limit=self.journal.trial_limit)
        return stm,scope,records,values

    def apply(self,state,selected,operands,column,refs=None):
        stm,scope,records,values = state
        selected = self.journal.choice(scope,selected)
        index = torch.as_tensor(column,device=stm[0].device).expand(len(stm[0]))
        refs = records.new_full((len(stm[0]),2),-1) if refs is None else refs
        values = chunks.write(values,index,torch.cat((operands,selected.candidate[:,None]),1).flatten(1),selected.valid)
        stm,scope,records,_ = self.journal.apply(stm,scope,selected,operands,refs,records,
            (values,values,torch.arange(values.shape[1],device=values.device)[None].expand(len(values),-1)),
            column=index,free_ids=self.free,clause_scope=self.scope)
        return stm,scope,records,values


def choice(value, *, kind=1, local=0, identity=0, chunk=-1, bind=False, order=1, head=1):
    ids = torch.zeros(len(value),dtype=torch.long,device=value.device)
    active = torch.ones(len(value),dtype=torch.bool,device=value.device)
    return LanguageOperationChoice(value,ids+kind,ids,ids+local,active,value.new_ones(len(value)),
        ids,active,order=ids+order,identity=ids+identity,chunk=ids+chunk,
        stack_arity=ids+(1 if bind or kind == 2 else 2),head_role=ids+head)


def value(state, identity, timeline=0):
    stm = state[0]
    ids = stm[chunks.IDENTITY].new_full((len(stm[0]),1),identity)
    values,found,_ = read_identity(stm,ids,torch.full_like(ids,timeline))
    torch._assert_async(found.all(),'fixture identity must remain in the shared working set')
    return values[:,0]


def record(f,tree,*,frames=(),dummy_binds=(),branch_binds=(),names=('subject','alias'),leaf_orders=None):
    from ReferenceContext import denoted_order
    # These fixed source readings assert their event; ask at the head still
    # removes that evidence through the production journal.
    entry = f.program(tree)
    entry = replace(entry, leaf_evidence=entry.leaves.new_tensor([[1., 0.]]).expand(len(entry.leaves), 2))
    named = {f.words[name][0] for name in names if name in f.words}
    orders = (torch.tensor([1 if int(word) in named else 0 for word in entry.word_ids])
              if leaf_orders is None else torch.tensor(leaf_orders))
    # These fixed-operation identity certificates supply kinds whose selected members are individuals;
    # the retired determiner constant no longer manufactures that order.
    from order_fixtures import force_order
    if leaf_orders is None:
        stack = []
        for kind, local, position in entry.actions.tolist():
            if kind == 0:
                stack.append((position,))
                continue
            if kind == 3:
                stack[-2:] = reversed(stack[-2:])
                continue
            rule = (f.binary if kind == 1 else f.unary)[local]
            arity = 2 if kind == 1 and rule.method_name != 'bind' else 1
            operands = stack[-arity:]
            del stack[-arity:]
            head = (getattr(rule, 'head_role', 0) or 1)-1
            carried = operands[head] if head < arity else operands[0]
            if getattr(rule, 'determiner_mode', None) in ('mint', 'bind'):
                for index in carried:
                    if int(orders[index]) == 0:
                        orders[index] = 2
            stack.append(carried if getattr(rule, 'head_role', 0) else tuple(i for part in operands for i in part))
    for index, order in enumerate(orders.tolist()):
        if order != f.cs._concept_source_order(int(entry.concept_ids[index])):
            force_order(f.cs, int(entry.concept_ids[index]), order)
    state = prepend(f,frames,len(entry.leaves))
    walk = Walk(f.binary,f.unary,capacity=f.cs.mtm.capacity,first_id=int(f.cs.mtm.transient_ids[0]))
    walk.free = torch.where(f.cs.mtm.transient_used,-1,f.cs.mtm.transient_ids)
    walk.journal.trial_base = int(f.cs.mtm.transient_ids[-1])+1
    walk.journal.trial_limit = walk.journal.trial_base+len(entry.leaves)+len(entry.actions)
    pending = [entry.symbol if entry.symbol is not None else entry.transient
               for entry in f.cs.mtm.entries.values() if entry.pending]
    walk.journal.pending_ids[:len(pending)] = walk.free.new_tensor(pending)
    state = walk.empty(entry.leaves[None,0],len(entry.actions),state=state)
    first,binds,inputs = None,0,[]
    for column,(kind,local,word) in enumerate(entry.actions.tolist()):
        if kind == 0:
            state = walk.shift(state,entry.leaves[word][None],word=int(entry.concept_ids[word]),index=word,
                               order=int(orders[word]))
            continue
        stm,scope,records,values = state
        stm,live = chunks.refresh(stm,chunks.stack_indices(stm))
        state = stm,scope,records,values
        rule = (f.binary if kind == 1 else f.unary)[local]
        bind = rule.method_name == 'bind'
        arity = 2 if kind == 1 and not bind else 1
        operands = live[:,[1,0]] if arity == 2 else torch.stack((live[:,0],live[:,0]*0),1)
        identity,index = 0,-1
        refs = records.new_full((1,2),-1)
        if bind:
            offered,ids,indices,valid = chunks.bind_menu(stm,live[:,:1])
            if binds not in dummy_binds:
                eligible = valid[0,0] & (ids[0,0] > 0)
                if first is not None:
                    eligible &= ids[0,0] == first
                wanted = 1 if binds in branch_binds else 0
                timelines = torch.cat((torch.zeros(1,dtype=torch.long),stm[chunks.TIMELINE][0]))
                timeline_eligible = eligible & (timelines == wanted)
                eligible = timeline_eligible if bool(timeline_eligible.any()) else eligible & (timelines == 0)
                matches = eligible.nonzero().flatten()
                pick = int(matches[0]) if len(matches) else 0
            else:
                pick = 0
            identity,index = int(ids[0,0,pick]),int(indices[0,0,pick])
            operands = torch.stack((live[:,0],offered[:,0,pick]),1)
            result = f.operation(rule).compose(operands[:,0],operands[:,1],dummy=torch.tensor([pick == 0]))
            refs[:,1] = identity
            binds += 1
        else:
            result = f.operation(rule).compose(*(operands[:,i] for i in range(arity))).reshape(1,-1)
        stamps = chunks.stack_view(stm)[2][0,:arity].tolist()
        order = stamps[0] if bind else denoted_order(rule,tuple(reversed(stamps)), (), None)
        selected = choice(result,kind=kind,local=local,identity=identity,chunk=index,bind=bind,
                          order=order,head=getattr(rule,'head_role',0))
        inputs.append((rule.method_name,operands.clone(),result.clone()))
        state = walk.apply(state,selected,operands,column,refs)
        if first is None and (bind or getattr(rule,'determiner_mode',None) in ('mint','bind')):
            first = int(state[0][chunks.IDENTITY][0,int(chunks.at(state[0]))])
    stm,scope,records,values = state
    identity = capture(f.cs,stm,scope,records,values,batch=0,positions=torch.arange(len(entry.leaves)),
                       columns=torch.arange(len(entry.actions)),actions=entry.actions)
    end = chunks.stack_view(stm)[0][0,0]
    result = replace(entry,leaf_orders=orders,close_references=True,
        operation_values=values[0].reshape(-1,3,end.numel()),operation_refs=records[0,:,:2],
        end_state=torch.stack((end,end*0,end*0)),**identity)
    handle = 0 if first is None else int((result.identity_ids == first).long().argmax())+1
    return result,inputs,handle


def from_stack(buffer, depth, orders, grammar, rows, activations):
    # The supplied B, K and D define the allocation; no live state is resized.
    batch, capacity, width = buffer.shape
    state = list(chunks.empty(batch, capacity, width, device=buffer.device, dtype=buffer.dtype))
    positions = torch.arange(capacity, device=buffer.device)[None].expand(batch, -1)
    valid = positions < depth[:, None]
    state[chunks.BUFFER] = torch.where(valid[..., None], buffer, 0)
    state[chunks.DEPTH] = depth.clone()
    for field, value in ((chunks.ORDERS, orders), (chunks.GRAMMAR, grammar), (chunks.ROWS, rows)):
        state[field] = torch.where(valid, value, -1)
    state[chunks.ACTIVATIONS] = torch.where(valid, activations, 0)
    state[chunks.STACK] = torch.where(valid, positions, -1)
    state[chunks.READ] = valid.long()
    return tuple(state)
