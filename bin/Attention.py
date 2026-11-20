"""A sentence's reading: the words its looks interpreted, look by look, and
the draw that takes one of several equally active words.

A look is a request over concepts carried down onto the sentence's percept
bank (:mod:`Salience`, :mod:`SentenceLooks`); what it returns is read by
``interpret``. The record here is what the readers of a reading read: which
words were interpreted and with what key, which looks drew among equals, and
how many looks were taken.
"""
from typing import NamedTuple
from dataclasses import dataclass, field
import torch
from torch import nn
import torch.nn.functional as F


class Reading(NamedTuple):
    """A sentence's reading. ``actions`` holds the word each look took (-1
    where it took none); ``alternatives`` marks a look whose word was drawn
    among equally active words, a departure's menu."""
    values: torch.Tensor              # [B, W, D] each interpreted word's key
    accepted: torch.Tensor            # [B, W] interpreted
    descended: torch.Tensor           # [B, W] read whole with no word concept
    actions: torch.Tensor             # [B, looks] the word each look took
    alternatives: torch.Tensor        # [B, looks] taken by a draw among equals
    probabilities: torch.Tensor       # [B, looks] the draw's probability
    alternative_counts: torch.Tensor  # [B, looks] the other equals of the draw
    poles: torch.Tensor               # [B, W, 2] each word's evidence
    looks: torch.Tensor               # [B] looks taken


@dataclass
class SentenceReading:
    """One path's reading of a sentence, advanced look by look."""
    chooser: object
    keys: torch.Tensor
    spans: torch.Tensor
    known: torch.Tensor
    poles: torch.Tensor
    limit: torch.Tensor               # [B] looks allowed
    width: int
    budget: int = 0                   # looks a reading's history is padded to
    exploit: object = None            # the reading a departure departs from
    departure: torch.Tensor = None    # [B] the look that departs, -1 none
    replay_until: torch.Tensor = None  # [B] looks before it replay the exploit
    accepted: torch.Tensor = None
    descended: torch.Tensor = None
    values: torch.Tensor = None
    actions: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)
    probabilities: list = field(default_factory=list)
    counts: list = field(default_factory=list)

    def __post_init__(self):
        self.accepted = torch.zeros_like(self.known)
        self.descended = torch.zeros_like(self.known)
        self.values = torch.zeros_like(self.keys)
        B = self.keys.shape[0]
        if self.departure is None:
            self.departure = torch.full((B,), -1, dtype=torch.long, device=self.keys.device)
        if self.replay_until is None:
            self.replay_until = self.departure.clone()

    @property
    def taken(self):
        return len(self.actions)

    def record(self, index, interpreted):
        """The word ``index`` interpreted where ``interpreted``."""
        picked = (torch.arange(self.known.shape[1], device=index.device)[None] == index[:, None]) & interpreted[:, None]
        self.accepted = self.accepted | picked
        self.values = torch.where(picked[..., None], self.keys, self.values)

    def reading(self):
        B, W = self.known.shape
        length = max(int(self.budget), len(self.actions))
        def history(values, like, fill=0):
            value = torch.stack(values, 1) if values else like.new_full((B, 0), fill)
            # Paths take different numbers of looks; their histories are
            # kept at the same addresses for replay and selection.
            return torch.cat((value, value.new_full((B, length - value.shape[1]), fill)), 1)
        poles = torch.where(self.descended[..., None],
            torch.stack((torch.ones_like(self.known), torch.zeros_like(self.known)), -1).to(self.keys), self.poles)
        looks = torch.stack(self.actions, 1).ge(0).sum(1) if self.actions else self.spans.new_zeros(B)
        reading = Reading(self.values, self.accepted, self.descended,
            history(self.actions, self.spans, -1), history(self.alternatives, self.known),
            history(self.probabilities, self.keys), history(self.counts, self.spans), poles.detach(), looks)
        pad = self.width - W
        return reading._replace(values=F.pad(reading.values, (0, 0, 0, pad)),
            accepted=F.pad(reading.accepted, (0, pad)), descended=F.pad(reading.descended, (0, pad)),
            poles=F.pad(reading.poles, (0, 0, 0, pad)))


def draw_among(activity, *, replay=None, masked=None):
    """The most active of ``activity`` ``[B, W]`` (``-inf`` where nothing is
    found), exactly; where several are equally the most active, one is taken
    by a draw. ``replay`` ``[B]`` takes a given word where it is not -1;
    ``masked`` ``[B]`` departs from the given word to another of its equals.
    Returns the word ``[B]`` (-1 where nothing is found), whether it was
    drawn among equals, the draw's probability and the count of the other
    equals."""
    found = torch.isfinite(activity)
    best = torch.where(found, activity, -torch.inf).amax(-1, keepdim=True)
    tied = found & (activity == best)
    count = tied.sum(-1)
    chosen = tied
    if masked is not None:
        other = tied & (torch.arange(tied.shape[1], device=tied.device)[None] != masked[:, None])
        chosen = torch.where(((masked >= 0) & other.any(-1))[:, None], other, tied)
    draw = torch.rand(tied.shape, device=tied.device).masked_fill(~chosen, -1.)
    index = torch.where(chosen.any(-1), draw.argmax(-1), -1)
    if replay is not None:
        index = torch.where(replay >= 0, replay, index)
    probability = torch.where(count > 0, 1. / count.clamp_min(1).to(activity.dtype), 0.)
    return index, count > 1, probability, (count - 1).clamp_min(0)


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
