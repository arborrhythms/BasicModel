"""Eager staging for the tensor bracket body; no dictionaries inside the loop."""
import torch
import math
from torch.nn import functional as F
from Attention import narrow_words, read_code_field, BracketKeys


POLE_CONSUMERS = ('_attention_sentence_payload', '_pushed_word_slab:poles',
                  'commit_word_reference_slab:per_word',
                  'commit_word_reference_slab:whole_slab')


def read_poles(model, consumer):
    """Declared handoff seam, also the observer's complete consumer census."""
    if consumer not in POLE_CONSUMERS:
        raise ValueError(f'undeclared attention pole consumer: {consumer}')
    pair = getattr(model, '_attention_poles', None)
    return None if pair is None else pair.detach()


def reference_evidence(model, presences, consumer):
    """Reference provenance is the walk's pair, never a scalar reconstruction."""
    pair = read_poles(model, consumer)
    if pair is None:
        from Interpret import positive_evidence
        return positive_evidence(presences.squeeze(-1)).detach()
    return pair


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
    model._attention_poles=None
    model._attention_grammar_mask=None
    model._attention_read=None
    model._attention_greedy=None
    model._attention_pole_representation=False
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
    # Zero is the ordinary-composition control: no bracket walk or thought
    # read is performed. The stem still supplies every word to the VP path.
    if model.attention_budget == 0:return
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
    model._attention_pole_representation=(dim == 2 and any(
        getattr(module,"representation",None) == "poles" for module in model.languageSpace.modules()))
    keys=F.pad(slab.detach(),(0,max(0,dim-D)))[...,:dim]
    # Open awareness is a read, with neither an optimizer step nor an EMA
    # write. Narrowing's learned choice is outside this no-gradient read.
    with torch.no_grad():
        model._open_read=read_code_field(keys,live)
        model._last_gist=(model._open_read*live[...,None]).sum(1)/live.sum(1).clamp_min(1)[:,None]
        poles=native_word_poles(model,spans,forms,known)
        model._attention_native_poles=poles
    codebook=getattr(owner,'similarity_codebook',None)
    rows=codebook.getW() if codebook is not None else None
    boosts=owner.priming_weights(batch=B)
    live_width = max(1, int(torch.where(live,
        torch.arange(W, device=live.device)[None] + 1, 0).max()))
    prior=BracketKeys._codebook_retrieval_prior(keys[:, :live_width],rows,model._last_gist,boosts)
    if prior is not None: prior=F.pad(prior,(0,W-live_width))
    spent=spans.new_tensor([meter.spent for meter in model._attention_meters])
    def read(*, sentence=None, spent_override=None, **trial):
        reading_spans=spans[:, :live_width]
        if sentence is not None:
            here=model.inputSpace._packed_sentence_ids[:, :live_width] == sentence
            reading_spans=torch.where(here[...,None],reading_spans,0)
        result = narrow_words(chooser,keys[:, :live_width],reading_spans,
            known[:, :live_width],budget=model.attention_budget,
            prior=None if prior is None else prior[:, :live_width],
            identities=ids[:, :live_width],poles=poles[:, :live_width],
            spent=spent if spent_override is None else spent_override,**trial)
        # Eager attention need not multiply its field reductions by inactive
        # word storage. Restore the configured carrier before the tensor body.
        return result._replace(values=F.pad(result.values,(0,0,0,W-live_width)),
            accepted=F.pad(result.accepted,(0,W-live_width)),
            descended=F.pad(result.descended,(0,W-live_width)),
            poles=F.pad(result.poles,(0,0,0,W-live_width)),
            pole_changes=F.pad(result.pole_changes,(0,W-live_width)))
    model._attention_read=read
    reading=read()
    model._attention_greedy=reading
    model._attention_words=reading
    model._attention_spans=spans
    model._attention_forms=(forms,ids,keys,live)
    model._attention_spent=reading.table.spent.detach().clone()
    if not getattr(model,'_sentence_ends',False):
        for b, meter in enumerate(model._attention_meters):
            meter.require('bracket',int(reading.table.spent[b])-meter.spent)
    handoff(model,reading)


def handoff(model, reading):
    """Publish the trial's scope and detached word evidence, never its values."""
    model._attention_words=reading
    base=getattr(model,"_attention_grammar_mask",None)
    model.inputSpace._ar_grammar_leaf_mask=(reading.accepted if base is None else base & reading.accepted)
    native=model._attention_native_poles
    # A successful native descent supplies a positive identification witness.
    native=torch.where(reading.descended[...,None],
        torch.stack((torch.ones_like(reading.accepted),torch.zeros_like(reading.accepted)),-1).to(native),native)
    model._attention_poles=torch.where(reading.pole_changes[...,None],reading.poles,native).detach()
    slab=model.inputSpace._ar_embedded_N
    spans=model._attention_spans
    B=slab.shape[0]
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
