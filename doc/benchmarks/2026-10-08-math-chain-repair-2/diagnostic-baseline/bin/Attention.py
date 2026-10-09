"""One typed perceptual bracket table, used by the operation chooser.

The table and its reads are tensors. Native identities and retained part
witnesses are staged by the owners before this body; no allocation occurs here.
"""
from typing import NamedTuple
import torch
from torch import nn
import torch.nn.functional as F

SPACE_INPUT, SPACE_STM, SPACE_LTM, SPACE_PART, SPACE_WHOLE, SPACE_SYMBOL = range(6)
LEVEL_BYTE, LEVEL_WORD, LEVEL_SENTENCE, LEVEL_ROW = range(4)


class BracketTable(NamedTuple):
    intervals: torch.Tensor
    valid: torch.Tensor
    space: torch.Tensor
    level: torch.Tensor
    done: torch.Tensor
    spent: torch.Tensor

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
                   torch.full_like(intervals[...,0],LEVEL_SENTENCE),torch.zeros_like(valid),torch.zeros_like(lengths))

    @property
    def remaining(self):
        return (self.intervals.shape[1]-self.spent).clamp_min(0)

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
                            self.done&~(left|right),self.spent+ok.long())

    def accept(self,slot,*,active=None):
        rows=torch.arange(self.valid.shape[0],device=slot.device)
        ok=(self.remaining>0)&self.valid[rows,slot]&~self.done[rows,slot]
        if active is not None:ok=ok&active
        selected=(torch.arange(self.valid.shape[1],device=slot.device)[None]==slot[:,None])&ok[:,None]
        return self._replace(done=self.done|selected,spent=self.spent+ok.long())


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
    gloss=pure&singular&word
    return torch.stack((divide,descend,gloss),-1)


def field_reduce(poles,*,operation,dim):
    """Associative, commutative paired Boolean fields, with exact silence.

    Conjunction intersects positives and unions counterevidence; disjunction
    does the dual. Negation exchanges the two observed poles.
    """
    if operation=='not':return poles.flip(-1)
    positive,negative=poles.unbind(-1)
    if operation=='and':return torch.stack((positive.amin(dim),negative.amax(dim)),-1)
    if operation=='or':return torch.stack((positive.amax(dim),negative.amin(dim)),-1)
    raise ValueError('undeclared field operation')


def pooled_keys(event,spans):
    """Detached content inside each bracket, including an exact empty mask."""
    B,N,D=event.shape
    pos=torch.arange(N,device=event.device)[None,None]
    mask=(pos>=spans[...,0,None])&(pos<spans[...,1,None])
    weights=mask.to(event)/mask.sum(-1,keepdim=True).clamp_min(1)
    return weights@event.detach()


def priming_prior(keys,prototypes,boosts):
    """Intent's codebook prior: max_v cosine(key,row_v) × boost_v."""
    keys=nn.functional.normalize(keys.detach(),dim=-1)
    prototypes=nn.functional.normalize(prototypes.detach(),dim=-1)
    cosine=keys@prototypes.transpose(-1,-2)
    return (cosine*boosts.detach().unsqueeze(-2)).amax(-1)


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


def narrow_words(chooser,keys,spans,known,*,budget,prior=None,exploit=None,departure=None,identities=None,poles=None,spent=None):
    """Batched pinned narrowing, with the field's observed poles as its mask.

    The six candidate meanings are declared by the grammar. Field reductions
    alter the pooled reading used by the next choice; only pure singular word
    brackets may project a native symbol. No row loops or allocation occur.
    """
    B,W,D=keys.shape;K=int(budget);A=6
    if W==0:
        table=BracketTable.open(spans.new_zeros(B),budget=K)
        return NarrowedWords(table,keys,known,known,spans.new_full((B,K),-1),
            torch.zeros(B,K,device=keys.device,dtype=torch.bool),
            keys.new_zeros(B,K),spans.new_zeros(B,K))
    lengths=spans[...,1].amax(1)
    table=BracketTable.open(lengths,budget=K)
    if spent is not None:table=table._replace(spent=spent)
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
    for step in range(K):
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
        progress=narrowing_mask(pure,both,neither,singular=singular,word=word,
                                can_split=count>1,has_parts=count>0)
        field=((both | (count>1))[:,:,None]&~reductions)
        # As in compose, reserve the remaining progress rounds. Optional
        # field rewrites cannot starve a word's divide/descent/gloss sequence.
        pending=(spans[...,1]>spans[...,0])&~accepted
        minimum=2*pending.sum(-1)-live.sum(-1)+(pending&~evidence.any(-1)&~descended).sum(-1)
        field=field & (table.remaining>minimum)[:,None,None]
        legal=torch.cat((progress,field),-1)&live[...,None]&declared[None,None]
        pooled=(covered.to(keys)@keys)/count.clamp_min(1)[...,None]
        pooled=torch.where(field_valid[...,None],field_values,pooled)
        local_prior=None if prior is None else (covered.to(keys)@prior[...,None])[...,0]/count.clamp_min(1)
        mask=None if exploit is None else torch.where(departure==step,exploit.actions[:,step],-1)
        replay=None if exploit is None else torch.where(step<departure,exploit.actions[:,step],-1)
        action,credit,alternate,detail=chooser.attend(pooled,legal,table.space,prior=local_prior,
            masked_action=mask,replay_action=replay,return_details=True)
        slot=(action//A).clamp_max(K-1);op=action%A
        chosen=live[rows,slot]&(action<A*K)
        selected=covered[rows,slot]&chosen[:,None]
        selected_word=word[rows,slot]
        gloss=(op==2)&selected_word&chosen
        unknown=(op==1)&selected_word&neither[rows,slot]&chosen
        picked_gloss=selected&gloss[:,None]
        accepted=accepted|picked_gloss
        witness=selected&unknown[:,None]
        if identities is not None:
            witness=(witness[:,:,None] & (identities[:,:,None]==identities[:,None,:]) & (identities[:,None,:]>=0)).any(1)
        descended=descended|witness
        values=torch.where(picked_gloss[...,None],keys*credit[:,None,None],values)
        # Divide at the first pole disagreement, descend at a retained part.
        first=selected.long().argmax(-1)
        first_positive=evidence[rows,first,0]>0
        disagree=selected & torch.where(first_positive[:,None],evidence[...,1]>0,evidence[...,0]>0)
        boundary=torch.where(disagree,spans[...,0],lengths[:,None]).amin(-1)
        part_cut=torch.where(selected,spans[...,1],lengths[:,None]).amin(-1)
        # A first word with both poles disagrees with itself. Its left edge
        # is not a split: isolate that word at its right edge so the other
        # words remain reachable without accepting the contradictory one.
        interior=(boundary>lo[rows,slot])&(boundary<hi[rows,slot])
        cut=torch.where((op==0)&disagree.any(-1)&interior,boundary,part_cut)
        split=(op<2)&~selected_word&chosen
        table=table.split(slot,cut,descend=(True),active=split)
        table=table.accept(slot,active=gloss)
        field_action=(op>=3)&chosen
        table=table._replace(spent=table.spent+unknown.long()+field_action.long())
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
        conjunctive=torch.stack((
            torch.where(covered,evidence[:,None,:,0],1.).amin(2),
            torch.where(covered,evidence[:,None,:,1],0.).amax(2)),-1)
        disjunctive=torch.stack((
            torch.where(covered,evidence[:,None,:,0],0.).amax(2),
            torch.where(covered,evidence[:,None,:,1],1.).amin(2)),-1)
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
        pole_changes |= reached
        round_words.append(torch.where(chosen,first,-1))
        # A child is a fresh reading of its retained parts. A parent's
        # temporary Boolean aggregate must not become the child's evidence.
        reset=(slots==slot[:,None])&(split|unknown|gloss)[:,None]
        field_valid=(field_valid|where)&~reset
        actions.append(torch.where(live.any(-1),action,-1));alternatives.append(alternate&live.any(-1))
        probabilities.append(detail['probability'])
        counts.append(torch.where(live.any(-1),detail['alternative_count'],0))
    handed_poles=torch.where(descended[...,None],
        torch.stack((torch.ones_like(known),torch.zeros_like(known)),-1).to(keys),handed_poles)
    # Only the evidence handoff is evaluated here; eligibility, action draws,
    # the progress budget and local value reads above are unchanged. Earlier
    # fields supply the evidence for later fields on overlapping brackets.
    for reached,op in zip(field_scopes,field_operations):
        positive,negative=handed_poles.unbind(-1)
        conjunction=torch.stack((torch.where(reached,positive,torch.inf).amin(1),
                                 torch.where(reached,negative,0.).amax(1)),-1)
        disjunction=torch.stack((torch.where(reached,positive,0.).amax(1),
                                 torch.where(reached,negative,torch.inf).amin(1)),-1)
        pooled=torch.where(reached[...,None],handed_poles,0.).amax(1)
        pair=torch.where((op==3)[:,None],conjunction,
                         torch.where((op==4)[:,None],disjunction,pooled.flip(-1)))
        handed_poles=torch.where(reached[...,None],pair[:,None],handed_poles)
    return NarrowedWords(table,values,accepted,descended,torch.stack(actions,1),
        torch.stack(alternatives,1),torch.stack(probabilities,1),torch.stack(counts,1),
        handed_poles.detach(),pole_changes,torch.stack(round_words,1))


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
    def _codebook_retrieval_prior(keys, codebook_rows, intent, external_boosts):
        """The **subsymbolic** score term -- the codebook-retrieval prior, the
        literal ``intent_boosts`` / ``selection_boost_fn`` path
        (orders.md §6 "Attention is connectionist spreading activation").

        Routes each span through the codebook: a span scores high when its
        content snaps near a prototype the intent has primed --

            ``prior_k = max_v( cos(key_k, row_v) · boost_v )``

        the same ``(sim · boosts).amax`` reduction ``WholeSpace._topk_priming
        _mask`` uses. ``boost_v`` ``[V]`` is the intent's graded similarity to
        each codebook row (``intent_priming_weights``; ``1.0`` = neutral): the
        externally-primed ``external_boosts`` (a ``set_intent`` already on the
        tower) when present, else derived from the query ``intent`` (prime from
        the current concept). Returns ``[B, K]`` (``None`` when no codebook is
        available -> the caller falls back to the concept-content cosine).

        Fully DETACHED (keys, rows, intent, boosts): the gradient never reaches
        the EMA-only codebooks (C-9/C-11)."""
        if (keys is None or codebook_rows is None
                or not torch.is_tensor(codebook_rows)
                or codebook_rows.numel() == 0):
            return None
        W = codebook_rows.detach()
        # Slice to the common LEADING width (the same idiom as ``_cos`` /
        # ``intent_priming_weights`` / ``_topk_priming_mask``). This is
        # CONTENT-vs-CONTENT by construction: the muxed percept key carries its
        # content first with the .where/.when columns appended at the TAIL
        # (negative indices), while the codebook rows are content-only and
        # NARROWER (e.g. key 1024 = 1020 content + 2 where + 2 when, W 1020), so
        # the min-slice drops the key's where/when tail and compares prototypes
        # to span content -- the intended semantics, not a lossy truncation.
        d = min(int(keys.shape[-1]), int(W.shape[-1]))
        if d == 0:
            return None
        # [V] intent boosts: the primed-intent state if the tower carries one,
        # else the graded intent->rows similarity of the current concept.
        boosts = external_boosts
        if boosts is None and intent is not None:
            from Spaces import intent_priming_weights
            boosts = intent_priming_weights(intent.detach(), W)
        kn = keys.detach()[..., :d]
        Wn = W[:, :d]
        kn = kn / kn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        Wn = Wn / Wn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        sim = torch.einsum('bkd,vd->bkv', kn, Wn)            # [B, K, V]
        if boosts is not None and torch.is_tensor(boosts):
            bv = boosts.detach().to(sim.dtype)
            if bv.ndim not in (1, 2) or (bv.ndim == 2 and bv.shape[0] != keys.shape[0]):
                raise ValueError('retrieval priming surface differs from the current batch; broadcasting is forbidden')
            V = int(Wn.shape[0])
            bv = F.pad(bv[..., :V], (0, max(0, V-bv.shape[-1])), value=1.)
            sim = sim * (bv[:,None] if bv.ndim == 2 else bv)
        return sim.amax(dim=-1)                              # [B, K]

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
    # The three tower codebooks, each a distinct address space. SPACE_PART /
    # SPACE_WHOLE appear whenever their tower exposes a codebook; SPACE_SYMBOL
    # only under <symbolTower> (the SS ``.what`` is an empty Basis otherwise).
    # Address-space model: doc/Architecture.md "Addressable attention".
    SPACE_PART = 3      # PartSpace codebook
    SPACE_WHOLE = 4     # WholeSpace codebook
    SPACE_SYMBOL = 5    # SymbolSpace symbol codebook (SS.subspace.what)
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

    def consume(self, symbols, content):
        """Feed the soft-read ``content`` ``[B, Dc]`` into an owned operand
        ``symbols`` (``[B, N, D]`` or ``[B, D]``) as a zero-init gated residual
        on the common leading width: ``symbols[..., :d] += gate · content``.

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
            shared = (keys.dim() == 2)
            M = int(keys.shape[0] if shared else keys.shape[1])
            sid = int(s["id"])
            zeros_bm = torch.zeros(B, M, dtype=dtype, device=dev)
            cos_c = self._cos_keys(cq, keys, shared)
            cos_s = self._cos_keys(sq, keys, shared)
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
