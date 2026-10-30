"""The shared, bounded inverse menu for stored ideas and sentence generation.

The word bank supplies terminals. Optional structural candidates supply
nonterminals; neither a compose journal nor a desired output enters a menu.
All binary hypotheses are checked through their actual forward kernel.
"""
from typing import NamedTuple
from functools import wraps
import torch


class InverseMenu(NamedTuple):
    left: torch.Tensor
    right: torch.Tensor
    admissible: torch.Tensor
    exact: torch.Tensor
    form_residual: torch.Tensor
    meaning_residual: torch.Tensor
    stack_effect: torch.Tensor

    @property
    def features(self):
        return torch.cat((self.form_residual[..., None], self.meaning_residual,
            self.exact.to(self.left)[..., None], self.stack_effect[..., None]), -1).detach()


def relative_residual(parent, recovered, form_width=None):
    """Independent full-parent form/for/against errors; no signed cancellation."""
    width = parent.shape[-1] if form_width is None else form_width
    def error(start, end):
        value, restored = parent[..., start:end], recovered[..., start:end]
        scale = value.square().mean(-1)
        return (restored-value).square().mean(-1) / torch.where(scale > 0, scale, 1.)
    form = error(0, width)
    if width == parent.shape[-1]:
        return form, torch.zeros(*form.shape, 2, device=form.device, dtype=form.dtype)
    middle = width + (parent.shape[-1]-width)//2
    return form, torch.stack((error(width,middle), error(middle,parent.shape[-1])), -1)


def in_cube(value):
    return (torch.isfinite(value).all(-1) & ((value >= 0) & (value <= 1)).all(-1))


def empty_menu(parent, lefts, rights, available):
    zeros = parent.new_zeros(available.shape)
    return InverseMenu(torch.stack(lefts,1), torch.stack(rights,1),
        torch.zeros_like(available), torch.zeros_like(available), zeros,
        parent.new_zeros(*available.shape,2), zeros)


def native_recompose(op, parent, left, right=None, *, form_width=None,
                     whole_bank=None, case_bank=None):
    """The proposal's actual forward map, with its native count witnesses."""
    op = getattr(op,'gl',op)
    if form_width is not None:
        if right is None:
            whole = None if whole_bank is None else whole_bank.read(left)
            view = (case_bank.for_values(left,form_width=form_width) if case_bank is not None
                and callable(getattr(op,'compose_case_meaning',None)) else None)
            return op.compose_blocks(op.compose,left,whole=whole,case_view=view)
        from FormCentroids import fit
        ratio, whole = fit(parent[..., :form_width],left,right,whole_bank)
        view = (case_bank.for_values(right,form_width=form_width) if case_bank is not None
            and callable(getattr(op,'compose_case_meaning',None)) else None)
        return op.compose_blocks(op.compose,left,right,counts=(ratio,1-ratio),whole=whole,case_view=view)
    if right is None:
        return op.compose(left)
    if getattr(op,'inverse_kind',None)=='case_search':
        if case_bank is None:
            # No proposal exists in this domain. The inverse's unavailable
            # flag excludes this row; no case code is invented for scoring.
            return torch.zeros_like(parent)
        from CaseSelection import select_cases, fold_cases
        codes=case_bank.codes if case_bank.codes.ndim==3 else case_bank.codes[None]
        li=(codes-left[:,None]).square().sum(-1).argmin(-1)
        ri=(codes-right[:,None]).square().sum(-1).argmin(-1)
        return fold_cases(select_cases(case_bank,case_bank.ids[li],case_bank.ids[ri]))
    folded=left if getattr(op,'reconstructs_left',False) else op.compose(left,right)
    if getattr(op,'same_reference_idempotent',False):
        from MeaningCodes import repeated_symbol
        folded=torch.where(left.eq(right).all(-1)[:,None],repeated_symbol(left,form_width),folded)
    return folded


def eligibility_menu(parent,lefts,rights,available,binary,unary,proposals,recomposed,
                     exact_pairs,singular,terminal,terminal_available,form_width,*,
                     whole_bank=None,case_bank=None,cube_domain=True):
    """Admissibility preserves domain and witnesses; exactness measures error."""
    forms, meanings = [], []
    for value in recomposed:
        f,m=relative_residual(parent,value,form_width)
        forms.append(f); meanings.append(m)
    exact = [exact_pairs[:,i] for i in range(len(binary))]
    for offset,wrapped in enumerate(unary):
        index=len(binary)+offset
        value=lefts[index]
        domain=in_cube(value) if cube_domain else torch.isfinite(value).all(-1)
        valid=available[:,index]&domain&value.ne(parent).any(-1)
        safe=torch.where(valid[:,None],value,.5)
        folded=native_recompose(wrapped,parent,safe,form_width=form_width,
            whole_bank=whole_bank,case_bank=case_bank)
        f,m=relative_residual(parent,folded,form_width)
        valid=valid & torch.isfinite(f) & torch.isfinite(m).all(-1)
        proposals.append(valid); forms.append(f); meanings.append(m)
        exact.append(valid & (f <= 1e-8) & (m <= 1e-8).all(-1))
    # An exact terminal retains its original activation and both meaning lanes.
    # An approximate STOP emits an admitted native code, never the arbitrary top.
    terminal=torch.where(singular[:,None],parent,terminal)
    lefts=list(lefts); lefts[-1]=terminal
    domain=in_cube(terminal) if cube_domain else torch.isfinite(terminal).all(-1)
    valid=terminal_available & domain & (parent.ne(0).any(-1) if cube_domain else True)
    f,m=relative_residual(parent,terminal,form_width)
    proposals.append(valid); forms.append(f); meanings.append(m); exact.append(singular)
    admissible=torch.stack(proposals,1)&available
    exact=torch.stack(exact,1)&admissible
    # Preserve the existing pair-first exact reading. Approximate alternatives
    # are available only when no exact pair or terminal already explains it.
    compound=exact_pairs.any(-1)
    resolved=compound|singular
    strict=torch.cat((exact_pairs,torch.zeros_like(available[:,len(binary):-1]),
                      (singular&~compound)[:,None]),-1)
    admissible=torch.where(resolved[:,None],strict,admissible)
    effects=parent.new_zeros(available.shape)
    effects[:,:len(binary)]=1; effects[:,-1]=-1
    return InverseMenu(torch.stack(lefts,1),torch.stack(rights,1),admissible,exact,
        torch.stack(forms,1),torch.stack(meanings,1),effects)


def selected_residual(language,parent,left,right,choice,active,*,form_width=None,
                      whole_bank=None,case_bank=None):
    """A fresh pullback through only the selected native recomposition."""
    binary,unary=language._generate_binary_ops,language._generate_unary_ops
    total=parent.new_zeros(len(parent),3)
    compiling=torch.compiler.is_compiling()
    indices=range(len(binary)+len(unary)) if compiling else choice[active].unique().tolist()
    for index in indices:
        if index >= len(binary)+len(unary):
            continue
        use=active & (choice==index)
        op=(binary if index<len(binary) else unary)[index if index<len(binary) else index-len(binary)]
        def apply(p,a,b,mask):
            # Unselected rows may be padding, outside this operator's domain.
            a=torch.where(mask[:,None],a,.5); b=torch.where(mask[:,None],b,.5)
            folded=native_recompose(op,p,a,b if index<len(binary) else None,
                form_width=form_width,whole_bank=whole_bank,case_bank=case_bank)
            f,m=relative_residual(p,folded,form_width)
            return torch.where(mask[:,None],torch.cat((f[:,None],m),-1),0.)
        total=total+(torch.cond(use.any(),apply,lambda p,a,b,mask:p.new_zeros(len(p),3),
            (parent,left,right,use)) if compiling else apply(parent,left,right,use))
    f,m=relative_residual(parent,left,form_width)
    return total+torch.where((active & (choice==len(binary)+len(unary)))[:,None],
        torch.cat((f[:,None],m),-1),0.)


def walk_residual(steps, frontier, initial, depth, *, form_width=None):
    """Sum every local error plus unresolved frontier energy, without dilution."""
    valid=torch.arange(frontier.shape[1],device=frontier.device)[None]<depth[:,None]
    frontier=torch.where(valid[...,None],frontier,0.)
    width=initial.shape[-1] if form_width is None else form_width
    blocks=[(0,width)]
    if width<initial.shape[-1]:
        middle=width+(initial.shape[-1]-width)//2
        blocks.extend(((width,middle),(middle,initial.shape[-1])))
    energy=initial.new_zeros(len(initial))
    for start,end in blocks:
        baseline=initial[...,start:end].square().mean(-1).sum(-1)
        energy=energy+frontier[...,start:end].square().mean(-1).sum(-1)/torch.where(baseline>0,baseline,1.)
    return steps.sum((1,2))+energy


@torch.no_grad()
def lexical_candidates(owner,root,*,limit,chunk_size=256):
    """One root-conditioned scan; bounded chunks/top-k, never a per-step scan."""
    from SentenceUnderstanding import PrimedSymbols,readback_scores
    if limit<1 or chunk_size<1:
        raise ValueError('lexical retrieval needs positive bounds')
    if root.ndim==3:
        mask=root.ne(0).any(-1)
        root=root.sum(1)/mask.sum(-1,keepdim=True).clamp_min(1)
    B,D=root.shape
    rows=sorted({row for cid in owner.interpret.lexical_identities()
                 if (row:=owner._csw_row_of(cid)) is not None
                 and owner.word_surface_for_row(row)})
    width=getattr(getattr(owner.similarity_codebook,'mereology',None),'percept_event_width',None)
    heat=owner.relevance_weights(batch=B)
    best=root.new_full((B,limit),-torch.inf)
    selected=torch.full((B,limit),-1,dtype=torch.long,device=root.device)
    chunks=0
    for start in range(0,len(rows),chunk_size):
        ids=selected.new_tensor(rows[start:start+chunk_size])[None].expand(B,-1)
        codes=owner.similarity_codebook.lookup_rows(ids,streams=True).detach()
        weights=torch.ones_like(ids,dtype=root.dtype) if heat is None else heat.gather(1,ids)
        scores=readback_scores(root,codes,weights,meaning_start=width)
        candidates=torch.cat((selected,ids),1)
        scores=torch.cat((best,scores),1)
        best,indices=scores.topk(limit,dim=1,sorted=True)
        selected=candidates.gather(1,indices)
        chunks+=1
    valid=selected>=0
    codes=owner.similarity_codebook.lookup_rows(selected.clamp_min(0),streams=True).detach()
    codes=torch.where(valid[...,None],codes,0.)
    weights=torch.ones_like(best) if heat is None else heat.gather(1,selected.clamp_min(0))
    weights=torch.where(valid,weights,0.)
    surfaces=[[owner.word_surface_for_row(row) if row>=0 else None for row in batch]
              for batch in selected.cpu().tolist()]
    nbytes=max(1,max((len(raw or b'') for batch in surfaces for raw in batch),default=0)+1)
    byte=torch.zeros(B,limit,nbytes,dtype=torch.long,device=root.device)
    byte_valid=torch.zeros_like(byte,dtype=torch.bool)
    for b,batch in enumerate(surfaces):
        for j,raw in enumerate(batch):
            if raw:
                byte[b,j,:len(raw)]=byte.new_tensor(list(raw)); byte_valid[b,j,:len(raw)]=True
    report=dict(vocabulary=len(rows),chunks=chunks,chunk_size=chunk_size,limit=limit,
                rows=selected.cpu().tolist())
    return PrimedSymbols(selected,codes,weights,torch.zeros_like(valid),byte,byte_valid,
                         meaning_start=width),report


def walk_bank(model,root,bank):
    """Stage the lexical shortlist at the eager entrance to one compiled walk."""
    from dataclasses import replace
    import time
    started=time.monotonic()
    lexical,report=lexical_candidates(model._concept_owner(),root,
        limit=model.reconstruction_basis_limit)
    report['elapsed_seconds']=time.monotonic()-started
    model._last_lexical_retrieval=report
    if bank is None:
        result=lexical
    else:
        # Keep structural candidates; lexical availability is independent of
        # current heat. Duplicate identities are harmless to exact STOP.
        width=max(bank.bytes.shape[-1],lexical.bytes.shape[-1])
        def padded(value):
            return torch.nn.functional.pad(value,(0,width-value.shape[-1]))
        result=replace(bank,rows=torch.cat((bank.rows,lexical.rows),1),
            codes=torch.cat((bank.codes,lexical.codes),1),weights=torch.cat((bank.weights,lexical.weights),1),
            own=torch.cat((bank.own,lexical.own),1),bytes=torch.cat((padded(bank.bytes),padded(lexical.bytes)),1),
            byte_valid=torch.cat((padded(bank.byte_valid),padded(lexical.byte_valid)),1),
            forms=None if bank.forms is None else torch.cat((bank.forms,lexical.codes),1))
    from FormCentroids import candidates
    return replace(result,form_wholes=candidates(model.wholeSpace,result))


class TypeFamilies(NamedTuple):
    """A, B and native property wholes, in their existing conceptual chart."""
    codes: torch.Tensor
    valid: torch.Tensor
    families: torch.Tensor


def preserve_operator_activations(function):
    """Keep a host-side stored read from replacing live diagnostic state.

    Native numerical kernels publish detached activations during eager
    composition. Their values are useful to the owning forward, but inverse
    candidate evaluation must not publish them as another observed event.
    """
    @wraps(function)
    def read(language, *args, **kwargs):
        saved = {}
        for op in (*language._generate_binary_ops, *language._generate_unary_ops):
            for module in (op.modules() if isinstance(op, torch.nn.Module) else (op,)):
                if 'activation' in getattr(module, '__dict__', {}):
                    saved[id(module)] = (module, module.activation)
        try:
            return function(language, *args, **kwargs)
        finally:
            for module, activation in saved.values():
                module.activation = activation
    return read


@torch.no_grad()
def determiner_expansions(language, parent, order, basis, orders, *, binding=None):
    """Realize a lower-order referent through a higher-order description.

    A determiner need not change the point: increasing the description's
    order is progress in this inverse. Native order and binding facts supply
    that context; neither a recorded operation nor its probability is read.
    Marker words are ranked by the existing compose grammar on each pair.
    A bare point, without an order mismatch, licenses no such expansion.
    """
    rules = getattr(language, '_generate_binary_rules', ())
    compose = getattr(language, '_compose_binary_rules', ())
    layer = getattr(getattr(language, 'language_layer', None), 'operation_layer', None)
    if layer is None or not len(basis):
        return {}
    distance = (basis - parent).norm(dim=-1)
    descriptions = (distance <= 1e-4 * max(1., float(parent.norm()))) & (orders == order + 1)
    if not bool(descriptions.any()):
        return {}
    result = {}
    for index, rule in enumerate(rules):
        mode = getattr(rule, 'determiner_mode', None)
        if mode not in ('mint', 'bind') or (binding is not None and mode != binding):
            continue
        key = language._generate_rule_key(rule, 2)
        matches = [i for i, candidate in enumerate(compose)
                   if language._generate_rule_key(candidate, 2) == key]
        if len(matches) != 1:
            continue
        operation = matches[0]
        op = language._generate_binary_ops[index]
        description = basis[descriptions.nonzero()[0, 0]]
        # This is a bounded reverse application of the ordinary learned
        # word/operator association, with no names or vocabulary table.
        pairs = torch.stack((basis, description.expand_as(basis)), 1)
        folded = op.compose(pairs[:, 0], pairs[:, 1])
        _, scores = layer.chooser.score_binary(
            pairs[..., :layer.d_model], folded[:, None, None, :layer.d_model],
            layer.stop_anchor, layer.reduce_anchor[operation:operation+1],
            op_indices=torch.tensor([operation], device=basis.device))
        scores = scores.reshape(-1)
        finite = (torch.isfinite(scores) & torch.isfinite(basis).all(-1) & basis.ne(0).any(-1)
                  & (orders >= 0)
                  & ((folded-parent).norm(dim=-1) <= 1e-4 * max(1., float(parent.norm()))))
        if bool(finite.any()):
            marker = int(scores.masked_fill(~finite, -torch.inf).argmax())
            result[index] = (basis[marker], description, int(orders[marker]), order + 1)
    return result


class LiveConstituents:
    """An invocation-only, STM-sized inverse snapshot of composed wholes.

    Values are detached forward activations. No source positions, actions,
    child pointers or original words survive in this unordered candidate set.
    Its lifetime ends with the sentence trial; it never enters LTM.
    """
    def __init__(self, words, word_valid, capacity):
        self.words, self.word_valid = words.detach(), word_valid
        self.codes = words.new_zeros(words.shape[0], int(capacity), words.shape[-1])
        self.valid = torch.zeros(self.codes.shape[:2], device=words.device, dtype=torch.bool)

    def snapshot(self):
        return self.codes.detach().clone(), self.valid.clone()

    @torch.no_grad()
    def restore(self, snapshot, rows):
        codes,valid=snapshot
        if codes.shape!=self.codes.shape or valid.shape!=self.valid.shape:
            raise ValueError('inverse candidates must retain the constructed STM capacity')
        self.codes.copy_(torch.where(rows[:,None,None],codes,self.codes))
        self.valid.copy_(torch.where(rows[:,None],valid,self.valid))

    @torch.no_grad()
    def observe(self, state, rows, support):
        import STMChunks
        values,depth = STMChunks.stack_view(state)[:2]
        for b in rows.nonzero().flatten().tolist():
            for slot, value in enumerate(values[b, :int(depth[b])].detach()):
                if support is not None and int(support[b, slot].count_nonzero()) < 2:
                    continue
                if (not bool(value.any()) or not bool(torch.isfinite(value).all())
                        or bool((self.word_valid[b] & self.words[b].eq(value).all(-1)).any())
                        or bool((self.valid[b] & self.codes[b].eq(value).all(-1)).any())):
                    continue
                # Recency determines eviction only. Pair search is invariant
                # to this order except genuinely indistinguishable inverse ties.
                self.codes[b] = torch.cat((self.codes[b, 1:], value[None]), 0)
                self.valid[b] = torch.cat((self.valid[b, 1:], self.valid.new_ones(1)), 0)


@torch.no_grad()
def primed_type_families(space, heat, *, limit):
    """Snapshot existing types by family, without minting or reading episodes.

    Dictionary membership is learned. There is no word/POS table: A supplies
    objects, B supplies changes in A's coordinates, and other higher-order
    concepts supply property wholes. Cold and absent columns stay unavailable.
    Each family has the existing reconstruction candidate bound.
    """
    if heat.ndim == 1:
        heat = heat[None]
    B, _ = heat.shape
    book = space.similarity_codebook
    width = book.W.shape[-1]
    rows = heat.new_full((B, 3 * int(limit)), -1, dtype=torch.long)
    components = getattr(space, 'components', None)
    nouns = set(() if components is None else components.individual_ids)
    verbs = set(() if components is None else components.verbs.ids)
    for b in range(B):
        grouped = [[], [], []]
        for row in (heat[b] > 1).nonzero().flatten().tolist():
            identity = space.concept_id_at_row(row)
            if identity is None:
                continue
            family = 0 if identity in nouns else 1 if identity in verbs else 2
            if family == 2 and space._row_order(row) <= 0:
                continue
            grouped[family].append(row)
        for family, candidates in enumerate(grouped):
            candidates.sort(key=lambda row: (-float(heat[b, row]), row))
            candidates = candidates[:int(limit)]
            if candidates:
                rows[b, family * limit:family * limit + len(candidates)] = rows.new_tensor(candidates)
    valid = rows >= 0
    if bool(valid.any()):
        codes = book.lookup_rows(rows.clamp_min(0),streams=True).detach()
        codes = torch.where(valid[..., None], codes, 0.)
    else:
        codes = heat.new_zeros(B, 3 * int(limit), width)
    families = torch.arange(3, device=heat.device).repeat_interleave(int(limit))
    return TypeFamilies(codes, valid, families)


def reconstruction_coverage(emitted, expected):
    """Post-read support audit; counts never enter the inverse's decisions.

    A projection can perfectly preserve a head while losing its modifier.
    Numerical recomposition alone cannot certify that reading as complete.
    """
    missing, excess = (expected - emitted).clamp_min(0), (emitted - expected).clamp_min(0)
    return missing, excess, (missing + excess) == 0


def inverse_basis(parent, basis, basis_valid, priming, constituents, constituent_valid):
    if constituents is not None:
        if basis is None or basis_valid is None or constituent_valid is None:
            raise ValueError('structural decoding requires explicit word and constituent masks')
        priming = torch.cat((torch.ones_like(basis_valid, dtype=parent.dtype)
                             if priming is None else priming,
                             constituent_valid.to(parent)), 1)
        basis = torch.cat((basis, constituents.detach()), 1)
        basis_valid = torch.cat((basis_valid, constituent_valid), 1)
    return basis, basis_valid, priming


@preserve_operator_activations
def selected_inverse(language, parent, choice, live, *, basis, basis_valid, priming=None,
                     limit=16, case_bank=None, constituents=None, constituent_valid=None,
                     require_symbols=True, whole_bank=None):
    """Pull back the hard choice without retaining every rejected inverse graph."""
    basis, basis_valid, priming = inverse_basis(
        parent, basis, basis_valid, priming, constituents, constituent_valid)
    binary, unary = language._generate_binary_ops, language._generate_unary_ops
    use_binary = live & (choice < len(binary))
    use_unary = live & (choice >= len(binary)) & (choice < len(binary) + len(unary))
    left, right = parent, parent
    if bool(use_binary.any()):
        a, b = language.reverse_binary_step(parent, choice, use_binary, ops=binary,
            basis=basis if require_symbols else None,
            basis_valid=basis_valid if require_symbols else None,
            basis_priming=priming, candidate_limit=limit,
            free=require_symbols, case_bank=case_bank, whole_bank=whole_bank)
        left = torch.where(use_binary[:, None], a, left)
        right = torch.where(use_binary[:, None], b, right)
    if bool(use_unary.any()):
        local = choice - len(binary)
        for index in local[use_unary].unique(sorted=True).tolist():
            selected = use_unary & (local == index)
            a = language.generate_unary_step(parent, torch.zeros_like(local), selected,
                whole_bank=whole_bank, case_bank=case_bank, ops=(unary[index],))
            left = torch.where(selected[:, None], a, left)
    return left, right


class InverseMenuRead:
    """One eager CPU walk's bounded cache of exact, detached inverse menus.

    Parameters and dictionary snapshots stay fixed inside a walk. Repeated
    values can reuse their menu while the selected inverse builds its fresh
    pullback each time. The first differentiable menu still supplies every
    operator's zero-gradient edge. Capacity is the existing stack capacity;
    nothing survives the walk or enters a model checkpoint.
    """
    def __init__(self, language, capacity):
        self.language, self.capacity = language, int(capacity)
        self.entries, self.next = [], 0
        self.modules = {}
        for op in (*language._generate_binary_ops, *language._generate_unary_ops):
            for module in (op.modules() if isinstance(op, torch.nn.Module) else (op,)):
                self.modules[id(module)] = module

    def read(self, parent, live, **options):
        if not torch.is_grad_enabled():
            for source, active, result, activations in self.entries:
                if torch.equal(parent, source) and torch.equal(live, active):
                    for module, value in activations:
                        module.activation = value
                    return result
        result = inverse_menu(self.language, parent, live, **options)
        activations = tuple((module, module.activation) for module in self.modules.values()
                            if 'activation' in getattr(module, '__dict__', {}))
        entry = (parent.detach().clone(), live.clone(),
                 type(result)(*(value.detach() for value in result)), activations)
        if len(self.entries) < self.capacity:
            self.entries.append(entry)
        else:
            self.entries[self.next] = entry
        self.next = (self.next + 1) % self.capacity
        return result


def inverse_menu(language, parent, live, *, basis, basis_valid, priming=None,
                 limit=16, case_bank=None, constituents=None,
                 constituent_valid=None, constituent_families=None, require_symbols=True,
                 terminal_valid=None, whole_bank=None):
    from Language import LanguageSpace
    words, word_valid = basis, basis_valid
    terminals = word_valid if terminal_valid is None else word_valid & terminal_valid
    basis, basis_valid, priming = inverse_basis(
        parent, basis, basis_valid, priming, constituents, constituent_valid)
    # Families describe candidate types. Every type is admitted on either
    # side; recomposition and the learned generate policy decide orientation.
    binary, unary = language._generate_binary_ops, language._generate_unary_ops
    lefts, rights, missing = [], [], []
    for op in binary:
        if basis is None and require_symbols:
            left, right, unavailable = parent, parent, live
        else:
            left, right, unavailable = language.reverse_binary_step(
                parent, torch.zeros_like(live, dtype=torch.long), live,
                # This menu entry already owns one operator. Passing the
                # whole catalogue makes a compiled menu trace R dispatches
                # for each of R entries, even though its index is fixed.
                ops=(op,), basis=basis if require_symbols else None,
                basis_valid=basis_valid if require_symbols else None,
                basis_priming=priming, candidate_limit=limit,
                free=require_symbols, return_status=True, case_bank=case_bank, whole_bank=whole_bank)
        lefts.append(left); rights.append(right); missing.append(unavailable)
    for op in unary:
        value, unavailable = language.generate_unary_step(
            parent, torch.zeros_like(live, dtype=torch.long), live, return_status=True,
            whole_bank=whole_bank, case_bank=case_bank, ops=(op,))
        unavailable = unavailable | ~value.detach().ne(parent.detach()).any(-1)
        lefts.append(value); rights.append(parent); missing.append(unavailable)
    lefts.append(parent); rights.append(parent); missing.append(torch.zeros_like(live))
    available = ~torch.stack(missing, 1)
    # Priming can name an unspelled structural whole. Keep it in the
    # forward search while restricting only STOP to lexical realizations.
    options = ({} if constituents is None and terminal_valid is None else
               dict(terminal_basis=words, terminal_valid=terminals))
    if require_symbols:
        return LanguageSpace.decoder_eligibility(parent,lefts,rights,available,
            binary,basis,basis_valid,case_bank=case_bank,whole_bank=whole_bank,
            details=True,unary_ops=unary,**options)
    # A caller without symbolic guards still receives explicit residuals.
    layout=next((getattr(getattr(op,'gl',op),'meaning_layout',None)
        for op in (*binary,*unary) if getattr(getattr(op,'gl',op),'meaning_layout',None)),None)
    width=None if layout is None else layout[0]
    folded=[native_recompose(op,parent,lefts[i],rights[i],form_width=width,
        whole_bank=whole_bank,case_bank=case_bank) for i,op in enumerate(binary)]
    exact_pairs=[]
    for value in folded:
        f,m=relative_residual(parent,value,width)
        exact_pairs.append((f<=1e-8)&(m<=1e-8).all(-1))
    exact_pairs=torch.stack(exact_pairs,1) if exact_pairs else available[:,:0]
    # Nonsymbolic numeric walks retain their declared inverse-domain flags;
    # numeric values are not membership coordinates in the cube.
    proposals=[available[:,i] for i in range(len(binary))]
    menu=eligibility_menu(parent,lefts,rights,available,binary,unary,
        proposals,folded,exact_pairs,available[:,-1],parent,available[:,-1],width,
        whole_bank=whole_bank,case_bank=case_bank,cube_domain=False)
    return menu._replace(admissible=torch.stack(proposals,1)&available)
