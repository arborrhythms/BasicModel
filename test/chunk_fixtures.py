"""Fixed choices through the production chunk and journal operations.

The capacities come from each fixture's declared words/actions and eight
shared chunks. These helpers retain no independent identity value bank.
"""
from dataclasses import replace
from types import SimpleNamespace
import torch

from Memory import STMChunks as chunks
from ChunkJournal import RECORD_WIDTH, ChunkJournal, empty_scope, push, capture, read_identity
from ClauseScope import ClauseScope
from Spaces import LanguageOperationChoice


def context(resolver, frames=()):
    if len(frames) == len(chunks.FIELDS) and torch.is_tensor(frames[0]):
        return frames
    store, pool = resolver.store, resolver.components.owner.mtm
    state = chunks.empty(1,8,store.nDim,device=store.slots.device,dtype=store.slots.dtype)
    for frame in frames:
        if not hasattr(frame,'row_id'):
            frame = frame.read(store)
        if frame is None:
            continue
        from IdentityRows import point as identity_point
        for symbol in (() if getattr(frame,'activations',()) or store.location(getattr(frame,'occurrence',None)) is not None
                       else getattr(frame,'symbol_ids',())):
            value = identity_point(resolver.components.owner,symbol)
            if value is not None:
                state,_ = chunks.deposit(state,value[None],torch.tensor([True]),identity=symbol,
                    refs=torch.tensor([[[chunks.SYMBOL,symbol],[0,0]]]),order=1)
        location = store.location(getattr(frame,'occurrence',None))
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
            # §50: recalled identity chunks carry the computed current read,
            # as production prepending does, rather than the earlier clause.
            point = pool.value_of(token)
        identity = -1 if entry is None else entry.symbol if entry.symbol is not None else entry.transient
        state,_ = chunks.deposit(state,point[None],torch.tensor([True]),refs=references,
            identity=identity,order=frame.order,when=store.entry_time(location))
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
            state,_ = chunks.deposit(state,pool.value_of(key)[None],torch.tensor([True]),refs=refs,
                identity=identity,order=1,where=entry.where,when=store.entry_time(location))
    return state


def recall(resolver, frames, *, stream=None):
    """Explicit query-style recall before a fixture reads an unpooled row."""
    from QueryWork import QueryWorkBudget
    rows = [resolver.store.row(resolver.store.location(frame.occurrence)) for frame in frames]
    # Each input has at most the fixture's whole fixed store of references.
    work = QueryWorkBudget(max(1,len(rows)*(resolver.store.capacity+resolver.store.parts.capacity)))
    resolver.store.recall_identities(rows,work=work,stream=stream)
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
    from clock_fixtures import install
    install(model)
    model._open_sentence_slot = 0
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


def higher_word(f, word):
    """Declare a class sense above a fixture's existing named member."""
    wid, member = f.words[word]
    parent = f.cs.new_concept(order=f.cs._concept_source_order(member)+1)
    f.cs.add_part(parent, member)
    with torch.no_grad():
        f.cs.similarity_codebook.W[f.cs._concept_row_of(parent)].copy_(
            f.cs.similarity_codebook.W[f.cs._concept_row_of(member)])
    f.cs.interpret.define(wid, parent)
    f.words[word] = wid, parent
    return parent


def record(f,tree,*,frames=(),dummy_binds=(),branch_binds=(),bind_targets=None,names=('subject','alias'),leaf_orders=None):
    from ReferenceContext import denoted_order
    # These fixed source readings assert their event; ask at the head still
    # removes that evidence through the production journal.
    # Fixed source orders are supplied before seating the fixture's objects.
    # A concept's block is its order; a test cannot move an existing row by
    # changing a sidecar after interpreting its word.
    leaf_names, requested = [], []
    def declare(node):
        if isinstance(node, str):
            index = len(leaf_names)
            leaf_names.append(node)
            requested.append(f.cs._concept_source_order(f.words[node][1]) if node in f.words
                             else 1 if node in names else 0)
            return (index,)
        name, *children = node
        operands = [declare(child) for child in children]
        catalog = f.binary if len(children) == 2 or name == 'bind' else f.unary
        rule = next(rule for rule in catalog if rule.method_name == name)
        head = (getattr(rule, 'head_role', 0) or 1)-1
        carried = operands[head] if head < len(operands) else operands[0]
        if getattr(rule, 'determiner_mode', None) in ('mint', 'bind'):
            for index in carried:
                if requested[index] == 0:
                    requested[index] = 2
        return carried if getattr(rule, 'head_role', 0) else tuple(i for part in operands for i in part)
    declare(tree)
    orders = torch.tensor(requested if leaf_orders is None else leaf_orders)
    for name, order in zip(leaf_names, orders.tolist()):
        cid = f.noun(name, order=order)
        assert f.cs._concept_source_order(cid) == order, 'declare the fixture object at its source order'
    entry = f.program(tree)
    entry = replace(entry, leaf_evidence=entry.leaves.new_tensor([[1., 0.]]).expand(len(entry.leaves), 2))
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
                target = (bind_targets or {}).get(binds, first)
                if target is not None:
                    eligible &= ids[0,0] == target
                wanted = 1 if binds in branch_binds else 0
                timelines = torch.cat((torch.zeros(1,dtype=torch.long),stm[chunks.TIMELINE][0]))
                timeline_eligible = eligible & (timelines == wanted)
                eligible = timeline_eligible if bool(timeline_eligible.any()) else eligible & (timelines == 0)
                matches = eligible.nonzero().flatten()
                if binds in (bind_targets or {}):
                    assert len(matches), 'the fixed bind target must be in the native menu'
                pick = int(matches[0]) if len(matches) else 0
            else:
                pick = 0
            identity,index = int(ids[0,0,pick]),int(indices[0,0,pick])
            operands = torch.stack((live[:,0],offered[:,0,pick]),1)
            result = f.operation(rule).compose(operands[:,0],operands[:,1],dummy=torch.tensor([pick == 0]))
            refs[:,1] = identity
            binds += 1
        else:
            if getattr(rule, 'order_delta', 0) == -1:
                # Fixed choices use the same existing-word interpretation as
                # the live operand menu before composing unary lower.
                from ReferenceContext import ReferenceTypes, resolve_order
                ids = tuple(f.words[name][0] for name in f.words)
                definitions = [f.cs.definitions.objects(wid) for wid in ids]
                order_ids = torch.arange(len(f.cs._order_caps()))
                table = torch.full((1, len(ids), len(order_ids)), -1, dtype=torch.long)
                points = operands.new_zeros(1, len(ids), len(order_ids), operands.shape[-1])
                for i, objects in enumerate(definitions):
                    for cid in objects:
                        order = f.cs._concept_source_order(cid)
                        table[0, i, order] = cid
                        points[0, i, order] = f.registry._payload(('sym', cid))
                types = ReferenceTypes(torch.tensor([ids]), order_ids, table, points)
                source = chunks.stack_view(stm)[2][:, :1]
                native = stm[chunks.REFS].gather(1, chunks.stack_indices(stm)[:, :1, None, None].expand(-1, -1, 2, 2))[:, :, 0, 1]
                point, reference = resolve_order(operands[:, :1], native, source-1, types)
                operands = torch.cat((point, operands[:, 1:]), 1)
                refs[:, 0] = reference[:, 0]
            result = f.operation(rule).compose(*(operands[:,i] for i in range(arity))).reshape(1,-1)
        stamps = chunks.stack_view(stm)[2][0,:arity].tolist()
        if bind:
            target_order = int(stm[chunks.ORDERS][0, index]) if index >= 0 else stamps[0]
            order = denoted_order(rule, (stamps[0], target_order), (), None)
        else:
            order = denoted_order(rule, tuple(reversed(stamps)), (), None)
        selected = choice(result,kind=kind,local=local,identity=identity,chunk=index,bind=bind,
                          order=order,head=getattr(rule,'head_role',0))
        inputs.append((rule.method_name,operands.clone(),result.clone()))
        state = walk.apply(state,selected,operands,column,refs)
        if first is None and (bind or getattr(rule,'order_delta',0) or getattr(rule,'determiner_mode',None) in ('mint','bind')):
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
