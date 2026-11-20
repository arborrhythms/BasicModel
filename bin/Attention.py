"""One typed perceptual bracket table, used by the operation chooser.

The table and its reads are tensors. Native identities and retained part
witnesses are staged by the owners before this body; no allocation occurs here.
"""
from typing import NamedTuple
from dataclasses import dataclass, replace
import torch
from torch import nn
import torch.nn.functional as F
from FieldLogic import reduce as reduce_field

SPACE_INPUT, SPACE_STM, SPACE_LTM, SPACE_PART, SPACE_WHOLE, SPACE_SYMBOL = range(6)
LEVEL_BYTE, LEVEL_WORD, LEVEL_SENTENCE, LEVEL_ROW = range(4)


class BracketTable(NamedTuple):
    intervals: torch.Tensor
    valid: torch.Tensor
    space: torch.Tensor
    level: torch.Tensor
    done: torch.Tensor
    spent: torch.Tensor
    limit: torch.Tensor = None

    @classmethod
    def open(cls, lengths, *, budget):
        if int(budget) < 1:
            raise ValueError('attentionBudget must be positive')
        B,K=lengths.shape[0],int(budget)
        index=torch.arange(K,device=lengths.device)[None]
        intervals=torch.stack((torch.zeros_like(index*lengths[:,None]),
                               torch.where(index==0,lengths[:,None],0)), -1)
        valid=(index==0)&(lengths[:,None]>0)
        return cls(intervals,valid,torch.zeros_like(intervals[...,0]),
                   torch.full_like(intervals[...,0],LEVEL_SENTENCE),torch.zeros_like(valid),torch.zeros_like(lengths),
                   torch.full_like(lengths, K))

    @property
    def remaining(self):
        limit = self.intervals.shape[1] if self.limit is None else self.limit
        return (limit-self.spent).clamp_min(0)

    def split(self, slot, cut, *, space=None, descend=False, active=None):
        B,K,_=self.intervals.shape
        rows=torch.arange(B,device=slot.device)
        old=self.intervals[rows,slot]
        free=(~self.valid).long().argmax(-1)
        ok=(self.remaining>0)&(~self.valid).any(-1)&(cut>old[:,0])&(cut<old[:,1])
        if active is not None:ok=ok&active
        current_space=self.space[rows,slot] if space is None else space
        current_level=self.level[rows,slot]-(1 if descend else 0)
        mask=torch.arange(K,device=slot.device)[None]
        left=(mask==slot[:,None])&ok[:,None]
        right=(mask==free[:,None])&ok[:,None]
        intervals=torch.where(left[...,None],torch.stack((old[:,0],cut),-1)[:,None],self.intervals)
        intervals=torch.where(right[...,None],torch.stack((cut,old[:,1]),-1)[:,None],intervals)
        return BracketTable(intervals,self.valid|right,
                            torch.where(left|right,current_space[:,None],self.space),
                            torch.where(left|right,current_level.clamp_min(0)[:,None],self.level),
                            self.done&~(left|right),self.spent+ok.long(),self.limit)


class BracketRead(NamedTuple):
    poles: torch.Tensor  # [batch,concept,bracket,pole]
    pure: torch.Tensor
    both: torch.Tensor
    neither: torch.Tensor
    singular: torch.Tensor


def pooled_read(poles,positions,brackets,*,tolerance=0.):
    from ConceptEvidence import in_extents
    result,_=in_extents(poles,positions,brackets)
    result=result.permute(1,0,2,3)
    support=result>tolerance
    positive=support[...,0].any(1);negative=support[...,1].any(1)
    both=positive&negative
    neither=~(positive|negative)
    singular=support.any(-1).sum(1)==1
    return BracketRead(result,positive^negative,both,neither,singular)


def narrowing_mask(pure,both,neither,*,singular,word,can_split,has_parts):
    """Pinned word stop: only unknown words may descend below it."""
    divide=both&can_split&~word
    descend=has_parts&(~word|neither)
    interpret=pure&singular&word
    return torch.stack((divide,descend,interpret),-1)


def pooled_keys(event,spans):
    """Detached content inside each bracket, including an exact empty mask."""
    B,N,D=event.shape
    pos=torch.arange(N,device=event.device)[None,None]
    mask=(pos>=spans[...,0,None])&(pos<spans[...,1,None])
    weights=mask.to(event)/mask.sum(-1,keepdim=True).clamp_min(1)
    return weights@event.detach()


def read_code_field(codes, valid=None):
    """Nonlinear order-independent field read with the original word carrier.

    Both Boolean reductions are available in alternating code coordinates.
    The fixed coordinate packing introduces no weights or extra capacity;
    each binding learns its own projections before this paired field read.
    """
    bounded=codes.tanh()
    valid=(codes.detach().abs().sum(-1)>0) if valid is None else valid
    pos=bounded.clamp_min(0);neg=(-bounded).clamp_min(0)
    # Silent padding is absent, not false evidence for every word.
    conjunction=torch.stack((torch.where(valid[...,None],pos,1.).amin(1),
                             torch.where(valid[...,None],neg,0.).amax(1)),-1)
    disjunction=torch.stack((torch.where(valid[...,None],pos,0.).amax(1),
                             torch.where(valid[...,None],neg,1.).amin(1)),-1)
    pair=torch.where((torch.arange(codes.shape[-1],device=codes.device)%2==0)[None,:,None],conjunction,disjunction)
    read=(pair[...,0]-pair[...,1])*valid.any(1)[:,None]
    return bounded+read[:,None]


class NarrowedWords(NamedTuple):
    table: BracketTable
    values: torch.Tensor
    accepted: torch.Tensor
    descended: torch.Tensor
    actions: torch.Tensor
    alternatives: torch.Tensor
    probabilities: torch.Tensor = None
    alternative_counts: torch.Tensor = None
    poles: torch.Tensor = None
    pole_changes: torch.Tensor = None
    round_words: torch.Tensor = None
    field_scopes: torch.Tensor = None


@dataclass
class NarrowingState:
    """One suspended bracket walk; closing reads it without advancing it."""
    chooser: object
    keys: object
    spans: object
    known: object
    budget: object
    exploit: object
    departure: object
    identities: object
    poles: object
    lengths: object
    table: object
    accepted: object
    descended: object
    values: object
    actions: object
    alternatives: object
    probabilities: object
    counts: object
    reductions: object
    field_values: object
    field_poles: object
    field_valid: object
    handed_poles: object
    pole_changes: object
    round_words: object
    field_scopes: object
    field_operations: object
    declared: object
    rows: object
    slots: object
    focus_frames: object
    focus_fields: object
    can_advance: object = None
    replay_until: object = None
    steps_taken: int = 0
    look_slot: object = None
    look_active: object = None
    glance_keys: object = None


def open_narrowing(chooser,keys,spans,known,*,budget,exploit=None,departure=None,identities=None,poles=None,spent=None,replay_until=None,limit=None):
    """Stage the original walk, without choosing any action."""
    B,W,D=keys.shape;K=int(budget);A=6
    lengths=spans[...,1].amax(1) if W else spans.new_zeros(B)
    table=BracketTable.open(lengths,budget=K)
    if spent is not None and W:table=table._replace(spent=spent)
    if limit is not None:table=table._replace(limit=limit)
    accepted=torch.zeros_like(known);descended=torch.zeros_like(known)
    values=torch.zeros_like(keys);actions=[];alternatives=[];probabilities=[];counts=[]
    reductions=torch.zeros(B,K,3,dtype=torch.bool,device=keys.device)
    field_values=keys.new_zeros(B,K,D)
    field_poles=keys.new_zeros(B,K,2)
    field_valid=torch.zeros(B,K,dtype=torch.bool,device=keys.device)
    handed_poles=(torch.stack((known,torch.zeros_like(known)),-1).to(keys)
                  if poles is None else poles.detach().clone())
    pole_changes=torch.zeros_like(known)
    round_words=[]; field_scopes=[]; field_operations=[]
    declared=torch.tensor([i in chooser.attention_operations for i in range(A)],device=keys.device)
    rows=torch.arange(B,device=keys.device)
    slots=torch.arange(K,device=keys.device)[None]
    return NarrowingState(
        chooser, keys, spans, known, budget, exploit, departure, identities, poles,
        lengths, table, accepted, descended, values, actions, alternatives, probabilities, counts, reductions,
        field_values, field_poles, field_valid, handed_poles, pole_changes, round_words, field_scopes, field_operations, declared, rows, slots,
        spans.new_full((B,K,9), -1), keys.new_zeros(B,K,D+2),
        replay_until=departure if replay_until is None else replay_until)


def last_focus(state):
    """Return the most recent reversible division or descent in each stream."""
    frames = state.focus_frames
    at = torch.where(frames[..., 0] >= 0, state.slots, -1).amax(-1)
    frame = frames[state.rows, at.clamp_min(0)]
    lo, hi = state.spans.unbind(-1)
    inside = (lo >= frame[:, 2:3]) & (hi <= frame[:, 3:4]) & (hi > lo)
    legal = (at >= 0) & ~(inside & state.accepted).any(-1)
    return at, frame, legal


def focus_history(state, original, table, slot, split, entered, undo):
    """Undo the last focus locally so an independent sibling keeps its work."""
    at, frame, _ = last_focus(state)
    rows, slots = state.rows, state.slots
    parent = (slots == frame[:, 0:1]) & undo[:, None]
    child = (slots == frame[:, 1:2]) & undo[:, None]
    interval = torch.where(parent[..., None], frame[:, None, 2:4], table.intervals)
    table = table._replace(intervals=torch.where(child[..., None], 0, interval),
        valid=table.valid & ~child, done=table.done & ~(child|parent),
        level=torch.where(parent, frame[:, 4:5], torch.where(child, LEVEL_SENTENCE, table.level)),
        space=torch.where(parent, frame[:, 5:6], torch.where(child, SPACE_INPUT, table.space)),
        spent=table.spent + undo.long())
    restored_field = state.focus_fields[rows, at.clamp_min(0)]
    # The entered extent identifies the witness. Repeated identities share
    # the native descent, exactly as on entry. A split records no descent.
    prior_witness = ((state.spans == frame[:, None, 2:4]).all(-1)
                     & frame[:, 8:9].bool() & (at >= 0)[:, None])
    if state.identities is not None:
        identity = state.identities.gather(1, prior_witness.long().argmax(-1, keepdim=True))
        prior_witness = prior_witness.any(-1, keepdim=True) & (identity >= 0) & (state.identities == identity)
    # A separate occurrence can already have joined grammar on the same
    # identity. Its recorded interpretation keeps its own evidence.
    state.descended = state.descended & ~(prior_witness & undo[:, None] & ~state.accepted)
    saved = torch.cat((state.field_values[rows,slot], state.field_poles[rows,slot]),-1)
    reductions = state.reductions[rows,slot].long()
    bits = reductions[:,0] + 2*reductions[:,1] + 4*reductions[:,2]
    entry = torch.stack((slot, torch.where(split,(~original.valid).long().argmax(-1),-1),
        *original.intervals[rows,slot].unbind(-1), original.level[rows,slot], original.space[rows,slot],
        state.field_valid[rows,slot].long(),bits,entered.long()),-1)
    push = (state.slots == state.steps_taken) & (split|entered)[:, None]
    pop = (state.slots == at[:, None]) & undo[:, None]
    state.focus_frames = torch.where(push[..., None], entry[:,None],
                                     torch.where(pop[..., None], -1, state.focus_frames))
    state.focus_fields = torch.where(push[...,None], saved[:,None], state.focus_fields)
    return table, parent, child, frame, restored_field


def narrowing_view(state):
    """Read the current brackets; grammar's option is separate from attention."""
    table, keys, spans, known = state.table, state.keys, state.spans, state.known
    B,W,D=keys.shape; K=int(state.budget)
    lengths, poles, descended = state.lengths, state.poles, state.descended
    accepted, reductions, declared = state.accepted, state.reductions, state.declared
    field_valid, field_poles, field_values = state.field_valid, state.field_poles, state.field_values
    lo,hi=table.intervals.unbind(-1)
    covered=(spans[:,None,:,0]>=lo[:,:,None])&(spans[:,None,:,1]<=hi[:,:,None])&(spans[:,None,:,1]>spans[:,None,:,0])
    count=covered.sum(-1)
    left=torch.where(covered,spans[:,None,:,0],lengths[:,None,None]).amin(-1)
    right=torch.where(covered,spans[:,None,:,1],0).amax(-1)
    table=table._replace(intervals=torch.where((count>0)[...,None],torch.stack((left,right),-1),table.intervals),
                         level=torch.where(count==1,LEVEL_WORD,LEVEL_SENTENCE))
    word=count==1
    evidence=(torch.stack((known,torch.zeros_like(known)),-1).to(keys) if poles is None else poles)
    evidence=torch.where(descended[...,None],torch.stack((torch.ones_like(known),torch.zeros_like(known)),-1).to(keys),evidence)
    support=covered[...,None] & evidence[:,None].ne(0)
    positive=support[...,0].any(-1);negative=support[...,1].any(-1)
    pooled_poles=torch.where(covered[...,None],evidence[:,None],0.).amax(2)
    pooled_poles=torch.where(field_valid[...,None],field_poles,pooled_poles)
    positive=pooled_poles[...,0]>0;negative=pooled_poles[...,1]>0
    both=positive&negative;pure=positive^negative;neither=~(positive|negative)
    singular=support.any(-1).sum(-1)==1
    live=table.valid&~table.done&(count>0)&(table.remaining>0)[:,None]
    room=(~table.valid).any(-1)[:,None]
    progress=narrowing_mask(pure,both,neither,singular=singular,word=word,
                            can_split=(count>1)&room,has_parts=(count>0)&(word|room))
    field=((both | (count>1))[:,:,None]&~reductions)
    # Only attention's own focus/descent steps spend its allowance.
    pending=(spans[...,1]>spans[...,0])&~accepted
    minimum=pending.sum(-1)-live.sum(-1)+(pending&~evidence.any(-1)&~descended).sum(-1)
    field=field & (table.remaining>minimum)[:,None,None]
    # Address 1 focuses a single word. An unknown word descends;
    # a pure word is looked at in place and grammar can interpret it.
    _, frame, can_undo = last_focus(state)
    unfocus = (state.slots == frame[:, :1]) & can_undo[:, None]
    attention_progress=torch.stack((progress[...,0],progress[...,1] | progress[...,2], unfocus),-1)
    legal=torch.cat((attention_progress,field),-1)&live[...,None]&declared[None,None]
    pooled=(covered.to(keys)@keys)/count.clamp_min(1)[...,None]
    pooled=torch.where(field_valid[...,None],field_values,pooled)
    return (table,covered,count,evidence,pooled_poles,neither,live,progress,legal,pooled)


def interpretation_at(state, slot):
    """The original pure single-word rule, without an attention accept act."""
    table,covered,count,_,_,_,_,progress,_,_=narrowing_view(state)
    rows=state.rows
    allowed=table.valid[rows,slot]&~table.done[rows,slot]&progress[rows,slot,2]
    selected=covered[rows,slot]
    return selected.long().argmax(-1), allowed


def record_interpretation(state, slot, active):
    """Grammar consumes a word bracket without spending an attention act."""
    table,covered,_,_,_,_,_,progress,_,_=narrowing_view(state)
    rows=state.rows
    allowed=table.valid[rows,slot]&~table.done[rows,slot]&progress[rows,slot,2]
    torch._assert_async((~active|allowed).all(),'interpret requires one pure word')
    picked=covered[rows,slot]&active[:,None]
    state.accepted=state.accepted|picked
    state.values=torch.where(picked[...,None],state.keys,state.values)
    selected=(state.slots==slot[:,None])&active[:,None]
    state.table=table._replace(done=table.done|selected)
    state.field_valid=state.field_valid&~selected
    state.can_advance=None
    return state


def focus_cuts(selected, spans, evidence):
    """Byte boundaries of divide and descend for a batch of selected extents."""
    lo, hi = spans.unbind(-1)
    end = hi.amax(-1, keepdim=True)
    first = selected.long().argmax(-1, keepdim=True)
    first_for = evidence[..., 0].gather(1, first) > 0
    opposite = torch.where(first_for, evidence[..., 1] > 0, evidence[..., 0] > 0)
    disagreement = torch.where(selected & opposite, lo, end).amin(-1)
    descend = torch.where(selected, hi, end).amin(-1)
    left = torch.where(selected, lo, end).amin(-1)
    right = torch.where(selected, hi, 0).amax(-1)
    interior = (disagreement > left) & (disagreement < right)
    return torch.where(interior, disagreement, descend), descend


def narrow_step(state, *, context=None, lesson_action=None):
    """Take one original batched step; no row loop or word composition."""
    chooser=state.chooser
    keys=state.keys
    spans=state.spans
    known=state.known
    budget=state.budget
    exploit=state.exploit
    departure=state.departure
    identities=state.identities
    poles=state.poles
    lengths=state.lengths
    table=state.table
    accepted=state.accepted
    descended=state.descended
    values=state.values
    actions=state.actions
    alternatives=state.alternatives
    probabilities=state.probabilities
    counts=state.counts
    reductions=state.reductions
    field_values=state.field_values
    field_poles=state.field_poles
    field_valid=state.field_valid
    handed_poles=state.handed_poles
    pole_changes=state.pole_changes
    round_words=state.round_words
    field_scopes=state.field_scopes
    field_operations=state.field_operations
    declared=state.declared
    rows=state.rows
    slots=state.slots
    B,W,D=keys.shape;K=int(budget);A=6
    if W == 0 or len(actions) >= K:
        return state
    step=state.steps_taken
    table,covered,count,evidence,pooled_poles,neither,live,progress,legal,pooled=narrowing_view(state)
    lo,hi=table.intervals.unbind(-1)
    word=count==1
    state.can_advance=legal.any()
    mask=None if exploit is None else torch.where(departure==step,exploit.actions[:,step],-1)
    replay=None if exploit is None else torch.where(step<state.replay_until,exploit.actions[:,step],-1)
    if lesson_action is not None:
        replay = torch.where(lesson_action >= 0, lesson_action, -1 if replay is None else replay)
    if state.glance_keys is not None:
        context = covered.to(keys) @ state.glance_keys
    action,credit,alternate,detail=chooser.attend(pooled,legal,table.space,
        masked_action=mask,replay_action=replay,return_details=True,slot_limit=min(K,W),
        **({} if context is None else {'context': context}))
    if lesson_action is not None:
        taught = lesson_action >= 0
        state.lesson_log_probability = torch.where(taught,
            detail['logits'].log_softmax(-1).gather(1, action[:, None])[:, 0], 0.)
        alternate = alternate & ~taught
    slot=(action//A).clamp_max(K-1);op=action%A
    chosen=live[rows,slot]&(action<A*K)
    selected=covered[rows,slot]&chosen[:,None]
    selected_word=word[rows,slot]
    unknown=(op==1)&selected_word&neither[rows,slot]&chosen
    witness=selected&unknown[:,None]
    if identities is not None:
        witness=(witness[:,:,None] & (identities[:,:,None]==identities[:,None,:]) & (identities[:,None,:]>=0)).any(1)
    # Divide at the first pole disagreement, descend at a retained part.
    first=selected.long().argmax(-1)
    divide, descend_cut = focus_cuts(selected, spans, evidence)
    cut=torch.where(op==0,divide,descend_cut)
    split=(op<2)&~selected_word&chosen
    original = table
    table=table.split(slot,cut,descend=True,active=split)
    split = table.spent > original.spent
    undo = (op == 2) & chosen
    table, parent, child, frame, restored = focus_history(
        state, original, table, slot, split, unknown, undo)
    descended = (state.descended | witness)
    field_action=(op>=3)&chosen
    table=table._replace(spent=table.spent+((op==1)&selected_word&chosen).long()+field_action.long())
    where=(slots==slot[:,None])&field_action[:,None]
    which=torch.arange(3,device=keys.device)[None,None]==(op-3)[:,None,None]
    reductions=reductions|(where[...,None]&which)
    positive_keys=keys.clamp_min(0);negative_keys=(-keys).clamp_min(0)
    conjunction=torch.where(covered[...,None],positive_keys[:,None],torch.inf).amin(2)-torch.where(covered[...,None],negative_keys[:,None],0.).amax(2)
    disjunction=torch.where(covered[...,None],positive_keys[:,None],0.).amax(2)-torch.where(covered[...,None],negative_keys[:,None],torch.inf).amin(2)
    changed=torch.where((op==3)[:,None,None],conjunction,torch.where((op==4)[:,None,None],disjunction,pooled))
    # Empty padding contributes no field. In particular, never multiply
    # its reduction sentinel (inf) by a live straight-through credit.
    changed=torch.where((count>0)[...,None],changed,0.)
    field_values=torch.where(where[...,None],changed*credit[:,None,None],field_values)
    conjunctive=reduce_field(evidence[:,None].expand(-1,K,-1,-1), operation='intersection', dim=2, mask=covered)
    disjunctive=reduce_field(evidence[:,None].expand(-1,K,-1,-1), operation='union', dim=2, mask=covered)
    changed_poles=torch.where((op==3)[:,None,None],conjunctive,
        torch.where((op==4)[:,None,None],disjunctive,pooled_poles.flip(-1)))
    field_poles=torch.where(where[...,None],changed_poles,field_poles)
    # Preserve the operated extent after splitting. Resolve its word
    # witnesses at the end of the walk: an unknown word can acquire its
    # native identification by descent after its parent was operated on.
    # Freezing the parent's pre-descent (0,0) would erase that evidence.
    reached=selected & field_action[:,None]
    field_scopes.append(reached)
    field_operations.append(op)
    pole_changes = pole_changes | reached
    round_words.append(torch.where(chosen,first,-1))
    # A child is a fresh reading of its retained parts. A parent's
    # temporary Boolean aggregate must not become the child's evidence.
    reset=(slots==slot[:,None])&(split|unknown)[:,None]
    field_valid=(field_valid|where)&~reset
    actions.append(torch.where(live.any(-1),action,-1));alternatives.append(alternate&live.any(-1))
    probabilities.append(detail['probability'])
    counts.append(torch.where(live.any(-1),detail['alternative_count'],0))
    state.table=table
    state.accepted=accepted
    state.descended=descended
    state.values=values
    restored_bits = (frame[:, 7:8] & frame.new_tensor([1, 2, 4])) != 0
    state.reductions=torch.where(parent[...,None], restored_bits[:,None], reductions & ~child[...,None])
    state.field_values=torch.where(parent[...,None], restored[:,None,:D], field_values)
    state.field_poles=torch.where(parent[...,None], restored[:,None,D:], field_poles)
    state.field_valid=torch.where(parent, frame[:,6:7].bool(), field_valid & ~child)
    state.pole_changes=pole_changes
    state.steps_taken += 1
    state.look_slot, state.look_active = slot, chosen
    return state


_NARROW_CARRY = ('accepted','descended','values','reductions','field_values',
                 'field_poles','field_valid','pole_changes','focus_frames','focus_fields')
_NARROW_HISTORY = ('actions','alternatives','probabilities','counts','round_words',
                   'field_scopes','field_operations')


class _CapturedAttentionChooser:
    """Keep an unused stop-score input outside conditional autograd.

    OperationSelectionLayer.attend discards the learned stop score: its
    stop logit is the fixed legal-menu sentinel. Cond otherwise turns that
    unused parameter's None gradient into zero, allowing momentum updates.
    Detach before entering cond, while reusing the exact attend method.
    """
    def __init__(self, chooser):
        self._source=chooser
        self.stop_anchor=chooser.stop_anchor.detach()

    def __getattr__(self, name):
        return getattr(self._source,name)

    def attend(self, *args, **kwargs):
        return type(self._source).attend(self,*args,**kwargs)


def _captured_narrowing(state):
    """Tensor-conditional lowering of the same stop for fullgraph callers.

    Captured execution cannot read a tensor stop on the host. Its fixed
    carrier uses cond to skip the chooser/body after the first inert round;
    it does not retain the old budget-many chooser evaluations.
    """
    def tensors(walk):
        return (
            *walk.table, *(getattr(walk,name) for name in _NARROW_CARRY),
            *(getattr(walk,name)[-1] for name in _NARROW_HISTORY), walk.can_advance)

    narrow_step(state)
    templates=tensors(state)

    def pack(walk):
        # Cond's AOT backward in this stack passes tangents for integer/bool
        # outputs although its joint graph only accepts floating tangents.
        # Carry discrete bits through floating storage, without a numeric
        # integer cast (which would lose large address bits). Unpack before
        # every use. Boolean carriers are exactly zero or one.
        return tuple((value.to(torch.float32) if value.dtype == torch.bool else
                      value.contiguous().view(torch.float32) if not value.is_floating_point()
                      else value).clone() for value in tensors(walk))

    def unpack(values):
        return tuple(value.to(torch.bool) if template.dtype == torch.bool else
                     value.view(template.dtype).reshape(template.shape) if not template.is_floating_point()
                     else value if template.requires_grad else value.detach()
                     for value,template in zip(values,templates))

    carried=pack(state)
    width=len(BracketTable._fields)+len(_NARROW_CARRY)
    from Language import OperationSelectionLayer
    chooser=(_CapturedAttentionChooser(state.chooser)
             if type(state.chooser) is OperationSelectionLayer else state.chooser)
    for turn in range(len(state.actions),int(state.budget)):
        def advance(*values):
            values=unpack(values)
            current=replace(state,chooser=chooser,table=BracketTable(*values[:len(BracketTable._fields)]),steps_taken=turn,
                **dict(zip(_NARROW_CARRY,values[len(BracketTable._fields):width])),
                **{name: [] for name in _NARROW_HISTORY})
            narrow_step(current)
            return pack(current)

        def stopped(*values):
            return tuple(value.clone() for value in values)

        active=carried[-1].to(torch.bool)
        state.steps_taken=state.steps_taken+active.long()
        carried=torch.cond(active,advance,stopped,carried)
        values=unpack(carried)
        state.table=BracketTable(*values[:len(BracketTable._fields)])
        for name,value in zip(_NARROW_CARRY,values[len(BracketTable._fields):width]):
            setattr(state,name,value)
        for name,value in zip(_NARROW_HISTORY,values[width:-1]):
            getattr(state,name).append(value)
        state.can_advance=values[-1]
    return state


def finish_narrowing(state):
    """Finish examination decisions and pad the report without drawing again.

    Valid brackets can have no legal act (for example a contradictory word
    after its field acts are spent). Such a round is just as inert as a
    table with no live brackets. Keep its original stop/-1 distinction.
    """
    if state.keys.shape[1] == 0 or len(state.actions) >= int(state.budget):
        return state
    if torch.compiler.is_compiling():
        return _captured_narrowing(state)
    while len(state.actions) < int(state.budget):
        if state.can_advance is not None and not bool(state.can_advance):
            missing=int(state.budget)-len(state.actions)
            for name in ('actions','alternatives','probabilities','counts','round_words','field_scopes'):
                values=getattr(state,name)
                values.extend([values[-1]]*missing)
            break
        narrow_step(state)
    return state


def close_narrowing(state):
    """Resolve the recorded field acts and publish the original carrier."""
    keys=state.keys
    known=state.known
    spans=state.spans
    budget=state.budget
    table=state.table
    values=state.values
    accepted=state.accepted
    descended=state.descended
    handed_poles=state.handed_poles
    pole_changes=state.pole_changes
    actions=state.actions
    alternatives=state.alternatives
    probabilities=state.probabilities
    counts=state.counts
    round_words=state.round_words
    field_scopes=state.field_scopes
    field_operations=state.field_operations
    B,W,D=keys.shape;K=int(budget)
    if W == 0:
        return NarrowedWords(table,keys,known,known,spans.new_full((B,K),-1),
            torch.zeros(B,K,device=keys.device,dtype=torch.bool),
            keys.new_zeros(B,K),spans.new_zeros(B,K))
    handed_poles=torch.where(descended[...,None],
        torch.stack((torch.ones_like(known),torch.zeros_like(known)),-1).to(keys),handed_poles)
    # Only the evidence handoff is evaluated here; eligibility, action draws,
    # the progress budget and local value reads above are unchanged. Earlier
    # fields supply the evidence for later fields on overlapping brackets.
    for reached,op in zip(field_scopes,field_operations):
        conjunction=reduce_field(handed_poles, operation='intersection', dim=1, mask=reached)
        disjunction=reduce_field(handed_poles, operation='union', dim=1, mask=reached)
        pooled=torch.where(reached[...,None],handed_poles,0.).amax(1)
        pair=torch.where((op==3)[:,None],conjunction,
                         torch.where((op==4)[:,None],disjunction,pooled.flip(-1)))
        handed_poles=torch.where(reached[...,None],pair[:,None],handed_poles)
    def history(values, empty):
        return torch.stack(values,1) if values else empty
    return NarrowedWords(table,values,accepted,descended,history(actions,spans.new_empty(B,0)),
        history(alternatives,known.new_empty(B,0)),history(probabilities,keys.new_empty(B,0)),
        history(counts,spans.new_empty(B,0)),handed_poles.detach(),pole_changes,
        history(round_words,spans.new_empty(B,0)),history(field_scopes,known.new_empty(B,0,W)))


def narrow_words(chooser,keys,spans,known,*,budget,exploit=None,departure=None,identities=None,poles=None,spent=None,limit=None,glance_keys=None):
    """The complete examination reading, composed from its public steps."""
    state=open_narrowing(chooser,keys,spans,known,budget=budget,
        exploit=exploit,departure=departure,identities=identities,poles=poles,spent=spent,limit=limit)
    state.glance_keys = glance_keys
    finish_narrowing(state)
    return close_narrowing(state)


class BracketKeys:
    """Pure detached read helpers shared by typed brackets and answer recall."""
    @staticmethod
    def _pool(sub):
        """Detached mean-over-slots content vector ``[B, D]`` (or ``None``)
        from a SubSpace / event tensor. Empty / shapeless -> ``None``."""
        if sub is None:
            return None
        ev = sub.materialize() if hasattr(sub, "materialize") else sub
        if ev is None or not torch.is_tensor(ev):
            return None
        if ev.dim() == 2:
            ev = ev.unsqueeze(1)
        if ev.dim() != 3 or int(ev.shape[1]) == 0:
            return None
        return ev.detach().mean(dim=1)                       # [B, D]

    @staticmethod
    def _cos(q, keys):
        """Row-wise cosine of a query ``[B, Dq]`` against per-span keys
        ``[B, K, Dk]`` -> ``[B, K]`` (sliced to the common trailing width).
        ``None`` when either side is absent."""
        if q is None or keys is None:
            return None
        d = min(int(q.shape[-1]), int(keys.shape[-1]))
        if d == 0:
            return None
        # DETACH both sides: the query IS the primed symbols (derived from the
        # codebooks) and the keys are codebook-content -- the gradient must NOT
        # flow back into either (orders.md §6 "Learning"; C-9/C-11). Defense in
        # depth: the caller already detaches via ``_pool`` / ``_span_keys``.
        qn = q.detach()[..., :d]
        kn = keys.detach()[..., :d]
        qn = qn / qn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        kn = kn / kn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.einsum('bd,bkd->bk', qn, kn)            # [B, K]


    @staticmethod
    def _span_keys(percept_ev, spans):
        """Per-span detached pooled percept content ``[B, K, D]`` from the
        ``[start, end)`` atom brackets ``spans`` ``[B, K, 2]``."""
        Np = int(percept_ev.shape[1])
        pos = torch.arange(Np, device=percept_ev.device)     # [Np]
        s = spans[..., 0:1]                                  # [B, K, 1]
        e = spans[..., 1:2]
        mask = (pos.view(1, 1, Np) >= s) & (pos.view(1, 1, Np) < e)  # [B,K,Np]
        m = mask.to(percept_ev.dtype)
        denom = m.sum(dim=-1, keepdim=True).clamp_min(1.0)   # [B, K, 1]
        keys = torch.einsum('bkn,bnd->bkd', m,
                            percept_ev.detach()) / denom      # [B, K, D]
        return keys

    @staticmethod
    def superposition_scale(temperature):
        """Attention-only logit scale; compose exploration does not use it."""
        return 1.0 - min(1.0, max(0.0, float(temperature or 0.0)))


class PrimedSymbolReader(nn.Module):
    """Free, content/relation-driven attention over a TYPED, addressable space
    (doc/specs/reading-attention.md "(B) Global attention"; orders.md §6 "Two
    kinds of `.where`").

    Where reading attention (A) is *local* (next-word, monotonic, supervised),
    global attention is *free*: it ranges over a registry of **addressable
    spaces** -- the input window, STM, LTM, and the THREE tower codebooks
    (PartSpace part-percepts / WholeSpace whole-percepts + meronomy + taxonomy /
    SymbolSpace symbols) -- each a ``[B, M, D]`` set of candidate keys
    with a per-candidate normalized ``[start, end]`` bracket. ONE distribution
    competes ACROSS all spaces (no monotonic mask -- it can land anywhere,
    including the abstract relations that have no environmental `.where`),
    emitting a **typed** ``.where`` (which space + the bracket) and a **soft-read
    content** ``Σ αₖ·keyₖ``. Pointing the ``.where`` at the codebook/LTM is
    *introspection / recall*; at the input window it is *reading / search* --
    one mechanism, the type tag says which.

    The **stochastic element** (the two-pass superposition ``temperature``,
    :meth:`BracketKeys.superposition_scale`) flattens the distribution on
    the explore pass so a downstream task error -- NOT a next-word target --
    can shape where free attention lands (it has no supervised signal to break
    symmetry by itself; orders.md §6). Gradient stops at the keys: the
    codebook/LTM rows are EMA/persistent and DETACHED upstream, so only the
    scorer readout trains (the soft-read is differentiable through ``α`` only).
    Used for thought and generation. Numeric answer heads use only their
    affine record reader (6.8 review §11.2)."""

    SPACE_INPUT = 0
    SPACE_STM = 1
    SPACE_LTM = 2
    # Address spaces. SPACE_PART / SPACE_WHOLE appear whenever their tower
    # exposes a codebook; SPACE_SYMBOL is keyed by the primed bank's concept
    # codes (a symbol has no vector of its own).
    # Address-space model: doc/Architecture.md "Addressable attention".
    SPACE_PART = 3      # PartSpace codebook
    SPACE_WHOLE = 4     # WholeSpace codebook
    SPACE_SYMBOL = 5    # the primed bank, by its concepts' codes
    _N_SPACES = 6
    _N_FEATURES = 4   # cos(concept), cos(symbol), codebook boost, space id

    def __init__(self, hidden=16):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(self._N_FEATURES, int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), 1),
        )
        # A learned per-space prior (which address space to prefer a priori).
        self.space_bias = nn.Parameter(torch.zeros(self._N_SPACES))
        # The CONSUMER gate (zero-init): how much of the soft-read to inject back
        # into a thought/generation operand. At init the consume is a no-op
        # residual; the answer loss trains it (and, through the read, the scorer)
        # to retrieve content that lowers the loss.
        self.consume_gate = nn.Parameter(torch.zeros(1))

    def consume(self, symbols, content, *, identity=True):
        """Feed the soft-read ``content`` ``[B, Dc]`` into an owned operand
        ``symbols`` (``[B, N, D]`` or ``[B, D]``) as a zero-init chart translation
        on the common leading width. The identity operand stays in the cube.

        This closes global attention's loop (reading-attention.md "(B)"): the
        answer/output loss backprops through ``symbols`` → ``content``
        (``Σ αₖ·keyₖ``) → ``α`` → the scorer (+ this gate), so retrieval that
        helps the answer is rewarded. The keys are detached upstream, so the
        codebook / LTM / percept stores receive no gradient. Returns ``symbols``
        unchanged when there is nothing to read."""
        if (content is None or not torch.is_tensor(content)
                or symbols is None or not torch.is_tensor(symbols)):
            return symbols
        d = min(int(symbols.shape[-1]), int(content.shape[-1]))
        if d == 0:
            return symbols
        add = self.consume_gate * content[..., :d].to(symbols.dtype)   # [B, d]
        if symbols.dim() == 3:
            add = add.unsqueeze(1)                                     # [B,1,d]
        out = symbols.clone()
        if identity:
            from Layers import cube_translate
            out[..., :d] = cube_translate(out[..., :d], add)
        else:
            out[..., :d] = out[..., :d] + add
        return out

    @staticmethod
    def _cos_keys(q, keys, shared):
        """Row-wise cosine of a query ``[B, Dc]`` against ``keys`` -- ``[M, Dc]``
        when ``shared`` (one store for the whole batch -- codebook / LTM; a
        matmul, NO ``[B, M, Dc]`` materialization) or ``[B, M, Dc]`` per-batch
        (input / STM). Returns ``[B, M]`` (or ``None`` when ``q`` is absent)."""
        if q is None:
            return None
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)      # [B, Dc]
        kn = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if shared:
            return qn @ kn.t()                                     # [B, M]
        return torch.einsum('bd,bmd->bm', qn, kn)                  # [B, M]

    def forward(self, *, concept_q, symbol_q, spaces, temperature=0.0, chooser=None, work=None):
        """Score and select across the addressable ``spaces``.

        ``spaces`` is a list of dicts, each describing one address space:
          * ``id``    -- the space-type id (``SPACE_INPUT`` / ``_STM`` /
                         ``_LTM`` / ``_CODEBOOK``);
          * ``keys``  -- candidate content, DETACHED upstream: ``[M, D]`` for a
                         SHARED store (codebook / LTM -- matmul'd, never
                         broadcast to ``[B, M, D]``) or ``[B, M, D]`` per-batch
                         (input window / STM);
          * ``where`` -- ``[B, M, 2]`` or ``[M, 2]`` normalized brackets;
          * ``boosts``-- ``[M]`` per-row intent boosts (codebook only) or None;
          * ``valid`` -- ``[B, M]`` / ``[M]`` bool mask of real candidates.
          * ``score_keys`` -- optional age-relative view of located keys;
                         the read still returns their original content.

        Returns ``None`` when no space has candidates, else a dict:
          ``space_id`` ``[B]``, ``where`` ``[B, 2]``, ``content`` ``[B, Dc]``
          (the soft read), ``alpha`` ``[B, Mtot]``, ``space_of`` ``[Mtot]``."""
        if not spaces:
            return None
        usable = []
        for s in spaces:
            k = s.get("keys")
            if (k is not None and torch.is_tensor(k) and k.dim() in (2, 3)
                    and int(k.shape[-2]) > 0):
                usable.append(s)
        if not usable:
            return None
        # Batch size from a per-batch space, else from the query, else 1.
        B = 1
        for s in usable:
            if s["keys"].dim() == 3:
                B = int(s["keys"].shape[0]); break
        else:
            if concept_q is not None:
                B = int(concept_q.shape[0])
            elif symbol_q is not None:
                B = int(symbol_q.shape[0])
        # Common content width (input/STM/LTM/codebook differ, e.g. 1024 vs
        # 1020) -- slice to the min so the per-space soft reads sum cleanly.
        Dc = min(int(s["keys"].shape[-1]) for s in usable)
        if concept_q is not None:
            Dc = min(Dc, int(concept_q.shape[-1]))
        if symbol_q is not None:
            Dc = min(Dc, int(symbol_q.shape[-1]))
        if Dc <= 0:
            return None
        cq = None if concept_q is None else concept_q.detach()[..., :Dc]
        sq = None if symbol_q is None else symbol_q.detach()[..., :Dc]
        dtype = usable[0]["keys"].dtype
        dev = usable[0]["keys"].device
        prepared, logit_parts, where_parts, valid_parts, space_ids = (
            [], [], [], [], [])
        for s in usable:
            keys = s["keys"].detach()[..., :Dc]
            score_keys = s.get('score_keys', s['keys']).detach()[..., :Dc]
            shared = (keys.dim() == 2)
            M = int(keys.shape[0] if shared else keys.shape[1])
            sid = int(s["id"])
            zeros_bm = torch.zeros(B, M, dtype=dtype, device=dev)
            cos_c = self._cos_keys(cq, score_keys, shared)
            cos_s = self._cos_keys(sq, score_keys, shared)
            cos_c = zeros_bm if cos_c is None else cos_c.to(dtype)
            cos_s = zeros_bm if cos_s is None else cos_s.to(dtype)
            boosts = s.get("boosts")
            if boosts is not None and torch.is_tensor(boosts):
                bv = boosts.detach().to(device=dev, dtype=dtype)
                if bv.ndim == 1:
                    bv = bv[None].expand(B, -1)
                if bv.shape[0] != B:
                    raise ValueError('attention boosts must belong to each batch row')
                boost_feat = F.pad(bv[:, :M], (0, max(0, M-bv.shape[1])))
            else:
                boost_feat = zeros_bm
            sid_feat = torch.full((B, M), sid / max(self._N_SPACES - 1, 1),
                                  dtype=dtype, device=dev)
            feats = torch.stack([cos_c, cos_s, boost_feat, sid_feat], dim=-1)
            logit = self.scorer(feats).squeeze(-1) + self.space_bias[sid]
            where = s.get("where")
            if where is None or not torch.is_tensor(where):
                idx = torch.arange(M, device=dev, dtype=dtype)
                where = torch.stack([idx / M, (idx + 1) / M], dim=-1)  # [M, 2]
            where = where.to(dtype)
            if where.dim() == 2:
                where = where.view(1, M, 2).expand(B, M, 2)
            valid = s.get("valid")
            if valid is None:
                valid = torch.ones(B, M, dtype=torch.bool, device=dev)
            else:
                valid = valid.to(torch.bool)
                if valid.dim() == 1:
                    valid = valid.view(1, M).expand(B, M)
            prepared.append((keys, shared, M))
            logit_parts.append(logit)
            where_parts.append(where)
            valid_parts.append(valid)
            space_ids.append(torch.full((M,), sid, dtype=torch.long, device=dev))
        logits = torch.cat(logit_parts, dim=1)               # [B, Mtot]
        where_all = torch.cat(where_parts, dim=1)            # [B, Mtot, 2]
        valid_all = torch.cat(valid_parts, dim=1)            # [B, Mtot]
        space_of = torch.cat(space_ids, dim=0)               # [Mtot]
        if work is not None:
            if len(work) != B:raise ValueError('one attention allowance is required per row')
            allowed=torch.tensor([meter.remaining > 0 for meter in work],device=dev,dtype=torch.bool)
            valid_all=valid_all & allowed[:,None]
        # Stochastic element: scale the PREFERENCE before masking (t=0 -> sharp;
        # t=1 -> flat over the LEGAL candidates -- the explorer).
        logits = logits * BracketKeys.superposition_scale(temperature)
        masked = logits.masked_fill(~valid_all, -torch.inf)
        live=valid_all.any(-1)
        # STOP is the sole eligible choice on an empty or exhausted row.
        stop=torch.where(live, -torch.inf, 0.)[:,None]
        all_logits=torch.cat((masked,stop),-1)
        if chooser is None:
            probability=all_logits.softmax(-1)
            selected=all_logits.argmax(-1)
        else:
            selected,probability,_=chooser.select_logits(all_logits,
                structural=(True,)*all_logits.shape[-1])
        alpha = probability[:,:-1]
        # Soft read accumulated PER SPACE (so a shared store stays [M, Dc] and
        # is matmul'd, never broadcast to [B, M, Dc]).
        content = torch.zeros(B, Dc, dtype=alpha.dtype, device=dev)
        off = 0
        for keys, shared, M in prepared:
            a_s = alpha[:, off:off + M]                      # [B, M]
            if shared:
                content = content + a_s @ keys               # [B,M]@[M,Dc]
            else:
                content = content + torch.einsum('bm,bmd->bd', a_s, keys)
            off += M
        sel = selected.clamp_max(alpha.shape[-1]-1)
        space_id = torch.where(live,space_of.to(dev)[sel],-1)
        where_sel = torch.where(live[:,None],where_all[torch.arange(B, device=dev), sel],0.)
        if work is not None:
            for row,meter in enumerate(work):
                if bool(live[row]):meter.require('space-read')
        return {
            "space_id": space_id,
            "where": where_sel,
            "content": content,
            "alpha": alpha,
            "space_of": space_of,
        }
