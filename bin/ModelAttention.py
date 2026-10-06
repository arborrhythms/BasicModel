"""Eager staging for the tensor bracket body; no dictionaries inside the loop."""
import torch
import math
from torch.nn import functional as F
from Attention import narrow_words, read_code_field, BracketKeys


def canonical_native_events(model, ids, positions, brackets):
    """Contract observed contiguous parts using retained native ancestry.

    Older byte stems can still expose a group's children after its formation
    was admitted. Definition literals already use the formed identity. Give
    the current field that same canonical tiling, preserving original outer
    edges and never splitting a coarse percept or crossing a word boundary.
    """
    native = getattr(model.perceptualSpace, 'percept_store', None)
    if native is None or not native.part_groups:
        return ids, positions
    result, spans = ids.clone(), positions.clone()
    values, extents = ids.detach().cpu().tolist(), positions.detach().cpu().tolist()
    for b, words in enumerate(brackets.detach().cpu().tolist()):
        for lo, hi in words:
            selected = [i for i, (a, z) in enumerate(extents[b])
                        if lo <= a < z <= hi and values[b][i] >= 0]
            if not selected or any(extents[b][i][1] != extents[b][j][0]
                                   for i, j in zip(selected, selected[1:])):
                continue
            original = [values[b][i] for i in selected]
            canonical = native.canonical_parts(original)
            if canonical == original:
                continue
            leaves = native._canonical_group_cache[0]
            boundaries = {0: extents[b][selected[0]][0]}
            count = 0
            for i in selected:
                count += len(leaves.get(values[b][i], (values[b][i],)))
                boundaries[count] = extents[b][i][1]
            edges, cursor = [], 0
            for pid in canonical:
                end = cursor + len(leaves.get(pid, (pid,)))
                if cursor not in boundaries or end not in boundaries:
                    break
                edges.append((boundaries[cursor], boundaries[end]))
                cursor = end
            if len(edges) != len(canonical) or len(canonical) > len(selected):
                continue
            result[b, selected] = -1
            spans[b, selected] = 0
            used = selected[:len(canonical)]
            result[b, used] = result.new_tensor(canonical)
            spans[b, used] = spans.new_tensor(edges)
    return result, spans


def native_word_poles(model, spans, forms, known):
    """Read the current native definition at each word's own extent.

    A name lookup identifies the row to inspect, but does not stand in for
    evidence. The field reads only this input's parts and whole properties.
    No unknown word is allocated while deciding whether to descend.
    """
    owner = model._concept_owner()
    fi = getattr(model.perceptualSpace, '_forward_input', None) or {}
    ids = fi.get('native_indices', fi.get('indices'))
    positions = fi.get('native_part_spans', fi.get('part_spans'))
    raw = getattr(model, '_staged_concepts_in', None)
    poles = owner.similarity_codebook.getW().new_zeros(*known.shape, 2)
    if not all(torch.is_tensor(value) for value in (ids, positions, raw)):
        return poles  # no current native witness, hence neither
    if 'native_indices' not in fi:
        ids, positions = canonical_native_events(model, ids, positions, spans)
        fi['native_indices'], fi['native_part_spans'] = ids, positions
    if not bool(known.any()):
        return poles
    # The native field's concept axis can be large. Its reading axis needs
    # only extents containing known words, not the padded word capacity.
    # Keep positions unchanged and restore the silent suffix afterwards.
    width = int(torch.where(known,
        torch.arange(known.shape[1], device=known.device)[None] + 1, 0).max())
    live_spans = spans[:, :width]
    if positions.shape[1]:
        event_width = int(torch.where(positions[..., 1] > positions[..., 0],
            torch.arange(positions.shape[1], device=positions.device)[None] + 1, 0).max())
        ids, positions = ids[:, :event_width], positions[:, :event_width]
    ws = model.wholeSpaces[0]
    whole_spans, _ = ws.concept_evidence_layout(raw, int(ws.inputShape[0]))
    field = owner.cs_read_memberships((ids, positions,
        getattr(ws.subspace.what, 'primitive_properties', None), raw, whole_spans), live_spans)
    native = owner._cs_field_concept_ids[:field.shape[0]]
    native = native[:, None].expand(-1, len(forms)) if native.ndim == 1 else native
    objects = [[owner.definitions.objects(word) if (word := owner.definitions.word(form=form))
                is not None else () for form in row[:width]] for row in forms]
    alternatives = max(1, max((len(values) for row in objects for values in row), default=0))
    wanted = native.new_tensor([[list(values)+[-1]*(alternatives-len(values)) for values in row]
                               for row in objects])
    matches = (native.T[:, :, None, None] == wanted[:, None]).any(-1)
    matches = matches & (native.T >= 0)[:, :, None]
    result = torch.where(matches[..., None], field.permute(1, 0, 2, 3), 0.).amax(1).detach()
    return F.pad(result, (0, 0, 0, known.shape[1]-width))


def percept_reconstruction_score(model, keys, forms, identities, live):
    """Loss-side inverse of the input's native percept snapshot.

    Use the decoder's byte/end-of-word likelihood, including its null
    candidate. The immutable bank precedes both walks and includes omitted
    percepts, so dropping an input cannot shrink its reconstruction target.
    Neither targets nor this scorer are passed to the chooser. Concept
    admission and grammar follow only after the percept walk is selected.
    """
    B, W, _ = keys.shape
    surfaces = [[form.encode('latin1', 'replace') for form in row] for row in forms]
    width = max(1, max((len(value) for row in surfaces for value in row), default=0))
    raw = torch.tensor([[list(value)+[0]*(width-len(value)) for value in row]
                        for row in surfaces], device=keys.device, dtype=torch.long)
    valid = raw.ne(0) & live[..., None]
    indices = torch.arange(W, device=keys.device)
    repeated = ((identities[:, :, None] == identities[:, None, :])
        & (indices[None, :, None] > indices[None, None, :]) & live[:, None]).any(-1)
    bank_valid = valid & ~repeated[..., None]
    def score(reading):
        with torch.no_grad():
            costs = torch.stack([model._byte_word_cost(reading.values[:, word],
                raw.new_tensor(word), keys, raw, bank_valid, raw, valid, True)
                for word in range(W)], -1)
            return (costs*live).sum(-1)/live.sum(-1).clamp_min(1)/math.log(256.)
    return score


def stage_input(model):
    """Read the open bracket, then select bounded percepts before admission.

    The native word dictionary is only consulted here. The tensor body sees
    identities as equality keys, never as numerical features. Repeated novel
    surfaces share one descent witness and one subsequent native definition.
    """
    isp=model.inputSpace
    slab=getattr(isp,'_ar_embedded_N',None)
    active=getattr(isp,'_word_active_mask',None)
    model._attention_words=None
    model._attention_score_term=None
    model._last_attention_score_function=None
    model._word_expectation=None
    model._word_expectation_input=None
    from QueryWork import QueryWorkBudget
    pending = getattr(model, '_pending_attention_meters', ())
    model._pending_attention_meters = ()
    model._attention_meters = pending
    model._word_surprise=None
    model._attention_forms=None
    if not torch.is_tensor(slab) or not torch.is_tensor(active):return
    B,W,D=slab.shape
    if len(model._attention_meters) != B:
        model._attention_meters=tuple(QueryWorkBudget(model.attention_budget) for _ in range(B))
    if W == 0:return
    fi=getattr(model.perceptualSpace,'_forward_input',None) or {}
    texts=fi.get('word_texts') or ()
    if not any(row for row in texts):return  # a numeric field has no text brackets
    offsets=getattr(isp,'_ar_word_part_offsets',None)
    spans=fi.get('part_spans')
    if torch.is_tensor(offsets) and offsets.shape[:2] == (B,W):
        starts=offsets[:,:,0].detach().cpu().tolist()
    else:
        # A whole-slab stem's part axis is not its word axis. Recover the
        # native word occurrence positions from the staged surface itself;
        # never index byte-part spans with a word ordinal.
        raw=getattr(model,'_staged_concepts_in',None)
        raw_rows=(raw.detach().cpu().reshape(B,-1).long().tolist()
                  if torch.is_tensor(raw) else [[] for _ in range(B)])
        starts=[]
        for b in range(B):
            surface=bytes(raw_rows[b]);end=0;row=[]
            for text in (texts[b] if b<len(texts) else ()):
                value=str(text).encode('latin1','replace')
                found=surface.find(value,end) if value else end
                lo=end if found<0 else found
                row.append(lo);end=lo+len(value)
            starts.append((row+[0]*W)[:W])
    owner=model._concept_owner()
    definitions=owner.definitions
    extents=[];known=[];identities=[];forms=[];vocabulary={}
    for b in range(B):
        row=[];seen=[];ids=[];names=[];end=0
        for w in range(W):
            text=str(texts[b][w]) if b<len(texts) and w<len(texts[b]) else ''
            raw=text.encode('latin1','replace')
            leading=len(raw)-len(raw.lstrip());surface=raw.strip()
            lo=max(int(starts[b][w]),end)+leading
            hi=lo+len(surface)
            end=hi
            form=surface.decode('latin1')
            row.append((lo,hi) if surface else (0,0))
            seen.append(bool(surface) and definitions.word(form=form) is not None)
            if form and form not in vocabulary:vocabulary[form]=len(vocabulary)
            ids.append(vocabulary.get(form,-1));names.append(form)
        extents.append(row);known.append(seen);identities.append(ids);forms.append(names)
    spans=torch.tensor(extents,device=slab.device,dtype=torch.long)
    live=active & (spans[...,1]>spans[...,0])
    spans=torch.where(live[...,None],spans,0)
    ids=torch.tensor(identities,device=slab.device,dtype=torch.long)
    known=torch.tensor(known,device=slab.device,dtype=torch.bool)&live
    chooser=model._stm_reducer()
    dim=int(chooser.d_model)
    keys=F.pad(slab.detach(),(0,max(0,dim-D)))[...,:dim]
    # Open awareness is a read, with neither an optimizer step nor an EMA
    # write. Narrowing's learned choice is outside this no-gradient read.
    with torch.no_grad():
        poles=native_word_poles(model,spans,forms,known)
        model._attention_native_poles=poles
        model._open_read=read_code_field(keys,live)
        model._last_gist=(model._open_read*live[...,None]).sum((0,1))/live.sum().clamp_min(1)
    codebook=getattr(owner,'similarity_codebook',None)
    rows=codebook.getW() if codebook is not None else None
    boosts=owner.priming_weights()
    live_width = max(1, int(torch.where(live,
        torch.arange(W, device=live.device)[None] + 1, 0).max()))
    prior=BracketKeys._codebook_retrieval_prior(keys[:, :live_width],rows,model._last_gist[None],boosts)
    if prior is not None: prior=F.pad(prior,(0,W-live_width))
    spent=spans.new_tensor([meter.spent for meter in model._attention_meters])
    def read(**trial):
        result = narrow_words(chooser,keys[:, :live_width],spans[:, :live_width],
            known[:, :live_width],budget=model.attention_budget,
            prior=None if prior is None else prior[:, :live_width],
            identities=ids[:, :live_width],poles=poles[:, :live_width],spent=spent,**trial)
        # Eager attention need not multiply its field reductions by inactive
        # word storage. Restore the configured carrier before the tensor body.
        return result._replace(values=F.pad(result.values,(0,0,0,W-live_width)),
            accepted=F.pad(result.accepted,(0,W-live_width)),
            descended=F.pad(result.descended,(0,W-live_width)))
    model._last_attention_comparison = None
    if model.training and torch.is_grad_enabled():
        from WalkTrials import narrowing_pair, observe_comparison, attention_score_function
        reading, audit = narrowing_pair(read,
            percept_reconstruction_score(model, keys, forms, ids, live))
        model._last_attention_comparison = audit
        model._attention_score_term, model._last_attention_score_function = attention_score_function(audit)
        model._attention_forms=(forms,ids,keys,live)
        observe_comparison(model, 'attention.input', audit, active=live.any(-1))
    else:
        reading=read()
    model._attention_words=reading
    # Native forward values only. The paired-cost term is the chooser's sole
    # attention credit, consumed once by reconstruction at the owner step.
    score=(reading.values*keys).sum(-1)/keys.square().sum(-1).clamp_min(1e-12)
    model._attention_credit=score
    model._attention_spans=spans
    model._attention_forms=(forms,ids,keys,live)
    model._attention_spent=reading.table.spent.detach().clone()
    for b, meter in enumerate(model._attention_meters):
        meter.require('bracket',int(reading.table.spent[b])-meter.spent)
    # The top-down handoff reads exactly the selected typed table.
    table=reading.table
    last=(table.done.long()*torch.arange(1,table.done.shape[1]+1,device=slab.device)).argmax(-1)
    selected=table.intervals[torch.arange(B,device=slab.device),last]
    raw=getattr(model,'_staged_concepts_in',None)
    width=max(1,int(raw.shape[-1])) if torch.is_tensor(raw) else max(1,int(spans[...,1].max()))
    scope=selected.to(slab)/width
    # A primitive field with no glossed symbol stays open. Padding and an
    # empty candidate catalogue must not turn its scope into [0,0].
    model.conceptualSpace._passback_scope_where=torch.where(table.done.any(-1)[:,None],
        scope,torch.tensor([0.,1.],device=slab.device,dtype=slab.dtype))
    model.conceptualSpace._passback_scope_space=table.space.gather(1,last[:,None])[:,0]


def stage_expectation(model):
    """One word expectation per accepted percept, shared by both compose trials."""
    staged=getattr(model,'_attention_forms',None)
    owner=getattr(model.symbolSpace,'expectation',None)
    if staged is None or owner is None or 'word' not in owner.enabled_levels:return
    forms,identities,keys,live=staged
    isp=model.inputSpace
    codes=getattr(isp,'_ar_word_object_atoms',None)
    if not torch.is_tensor(codes):codes=getattr(isp,'_ar_grammar_object_atoms',None)
    if not torch.is_tensor(codes) or codes.shape != keys.shape:codes=keys
    codes=codes.detach()
    B,W,D=codes.shape
    # One candidate per identity, from the input's native candidate bank.
    C=max(1,int(identities.max())+1)
    flat_ids=identities.reshape(-1)
    flat_codes=codes.reshape(-1,D)
    matches=(torch.arange(C,device=codes.device)[:,None]==flat_ids[None])&live.reshape(1,-1)
    first=matches.long().argmax(-1)
    bank=flat_codes[first][None].expand(B,-1,-1).detach()
    valid=matches.any(-1)[None].expand(B,-1)
    accepted=model._attention_words.accepted & live
    targets=torch.where(accepted,identities,-1)
    model._word_expectation_input=(codes,bank,valid,targets)
    expected=owner.expect('word',codes,bank,valid,targets,
        teacher_forcing=bool(model.training),gain=model.word_expectation_gain,active=accepted)
    model._word_expectation=expected
    model._word_surprise=expected.surprise
    # The word column has its observation and negative image together. The
    # answer-owned reader consumes surprise; grammar retains factual evidence.
    model.conceptualSpace._word_column=expected.surprise
    model._word_expectation_mask=accepted
