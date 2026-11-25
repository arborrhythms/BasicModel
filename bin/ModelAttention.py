"""Eager staging of a sentence's reading; no dictionaries inside the loop."""
import torch
import math
from torch.nn import functional as F
from Attention import SentenceReading, read_code_field
from WordSurfaces import occupied_width


POLE_CONSUMERS = ('_attention_sentence_payload', '_pushed_word_slab:poles',
                  'commit_word_reference_slab:per_word',
                  'commit_word_reference_slab:whole_slab', 'SymbolField.read',
                  'WordExpectation.word_distribution')


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
    if native is None or not native.part_groups or not ids.shape[1]:
        return ids, positions
    # Initialize from admitted ancestry only. Identity spelling children do
    # not rewrite membership literals, so they need no occurrence readback.
    native.canonical_parts(())
    leaves, _ = native._canonical_group_cache
    if not leaves:
        return ids, positions
    groups = sorted(leaves)
    width = max(map(len, leaves.values()))
    addresses = ids.new_tensor(groups)
    table = ids.new_tensor([list(leaves[part]) + [-1] * (width - len(leaves[part])) for part in groups])
    sizes = ids.new_tensor([len(leaves[part]) for part in groups])
    at = torch.searchsorted(addresses, ids.contiguous()).clamp_max(len(groups) - 1)
    formed = addresses[at] == ids
    lengths = torch.where(formed, sizes[at], 1)
    expanded = torch.where(formed[..., None], table[at], F.pad(ids[..., None], (0, width - 1), value=-1))
    offsets = torch.arange(width, device=ids.device)
    B, E = ids.shape
    capacity = E * width
    slots = torch.arange(E, device=ids.device)
    places = torch.arange(capacity, device=ids.device).expand(B, -1)
    sentinel = torch.iinfo(ids.dtype).min
    # Longer matches win; equal-length alternatives retain inventory order.
    choices = sorted(leaves.items(), key=lambda pair: (len(pair[1]), -pair[0]))
    result, spans = ids.clone(), positions.clone()
    for bracket in brackets.unbind(1):
        lo, hi = bracket.unbind(-1)
        selected = ((ids >= 0) & (positions[..., 0] >= lo[:, None])
                    & (positions[..., 1] <= hi[:, None]) & (positions[..., 1] > positions[..., 0]))
        count = selected.sum(1)
        event_order = (~selected).int().argsort(1, stable=True)
        ordered = positions.gather(1, event_order[..., None].expand(-1, -1, 2))
        contiguous = ((slots[1:] >= count[:, None]) | (ordered[:, :-1, 1] == ordered[:, 1:, 0])).all(1)
        occupied = selected[..., None] & (offsets < lengths[..., None])
        order = (~occupied).flatten(1).int().argsort(1, stable=True)
        atoms = expanded.flatten(1).gather(1, order)
        live = occupied.flatten(1).gather(1, order)
        extent = torch.where(selected, lengths, 0).cumsum(1)
        total = extent[:, -1:]
        stride, canonical = torch.ones_like(atoms), atoms
        for part, group in choices:
            size = len(group)
            stop = capacity - size + 1
            matches = (atoms.unfold(1, size, 1) == ids.new_tensor(group)).all(-1)
            matches = F.pad(matches & (places[:, :stop] + size <= total), (0, size - 1))
            stride = torch.where(matches, size, stride)
            canonical = torch.where(matches, part, canonical)

        ends = (places + stride).clamp_max(capacity)
        jump = F.pad(ends, (0, 1), value=capacity)
        reached = F.pad(places == 0, (0, 1))
        for _ in range(capacity.bit_length()):
            reached = reached | torch.zeros_like(jump).scatter_add(1, jump, reached.long()).bool()
            jump = jump.gather(1, jump)
        starts = reached[:, :capacity] & live
        kept = starts.sum(1)
        boundaries = ids.new_full((B, capacity + 1), sentinel)
        begin = extent - torch.where(selected, lengths, 0)
        boundaries = boundaries.scatter_reduce(1, begin,
            torch.where(selected, positions[..., 0], sentinel), 'amax')
        boundaries = boundaries.scatter_reduce(1, extent,
            torch.where(selected, positions[..., 1], sentinel), 'amax')
        aligned = (~starts | ((boundaries[:, :capacity] != sentinel)
                             & (boundaries.gather(1, ends) != sentinel))).all(1)
        update = (count > 0) & contiguous & aligned & (kept <= count)
        take = torch.where(starts, places, capacity).sort(1).values[:, :E].clamp_max(capacity - 1)
        used = slots < kept[:, None]
        values = torch.where(used, canonical.gather(1, take), -1)
        edges = torch.stack((boundaries.gather(1, take), boundaries.gather(1, ends.gather(1, take))), -1)
        edges = torch.where(used[..., None], edges, 0)
        values = torch.empty_like(ids).scatter(1, event_order, values)
        edges = torch.empty_like(positions).scatter(1, event_order[..., None].expand(-1, -1, 2), edges)
        result = torch.where(update[:, None] & selected, values, result)
        spans = torch.where((update[:, None] & selected)[..., None], edges, spans)
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
    if not all(torch.is_tensor(value) for value in (ids, positions, raw)):
        return owner.similarity_codebook.W.new_zeros(*known.shape, 2)
    if 'native_indices' not in fi:
        ids, positions = canonical_native_events(model, ids, positions, spans)
        fi['native_indices'], fi['native_part_spans'] = ids, positions
    # The native field's concept axis can be large. Its reading axis needs
    # only extents containing known words, not the padded word capacity.
    # Keep positions unchanged and restore the silent suffix afterwards.
    width, event_width = torch.stack((occupied_width(known),
        occupied_width(positions[..., 1] > positions[..., 0]))).tolist()
    if not width:
        return owner.similarity_codebook.W.new_zeros(*known.shape, 2)
    live_spans = spans[:, :width]
    ids, positions = ids[:, :event_width], positions[:, :event_width]
    ws = model.wholeSpace
    whole_spans, _ = ws.concept_evidence_layout(raw, int(ws.inputShape[0]))
    field = owner.cs_read_memberships((ids, positions,
        getattr(ws.subspace.what, 'primitive_properties', None), raw, whole_spans), live_spans)
    # A concept is one signed value; a word is read both where its basic
    # percepts are.
    perception = getattr(owner, '_percept_field', None)
    if perception is not None:
        both = owner.order_zero_both_field(perception)
        field = torch.maximum(field, both[..., None].to(field))
    native = owner._cs_field_concept_ids[:field.shape[0]]
    native = native[:, None].expand(-1, len(forms)) if native.ndim == 1 else native
    objects = [[tuple(dict.fromkeys((word, *owner.definitions.objects(word))))
                if (word := owner.interpret.word_for_form(form)) is not None else ()
                for form in row[:width]] for row in forms]
    alternatives = max(1, max((len(values) for row in objects for values in row), default=0))
    wanted = native.new_tensor([[list(values)+[-1]*(alternatives-len(values)) for values in row]
                               for row in objects])
    matches = (native.T[:, :, None, None] == wanted[:, None]).any(-1)
    matches = matches & (native.T >= 0)[:, :, None]
    result = torch.where(matches[..., None], field.permute(1, 0, 2, 3), 0.).amax(1).detach()
    return F.pad(result, (0, 0, 0, known.shape[1]-width))


def stage_input(model):
    """Prepare the bracket; its sentence, or a non-sentence read, owns admission.

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
    model._attention_word_concepts=None
    model._attention_open=None
    model._attention_greedy=None
    model._attention_pole_representation=False
    model._word_expectation=None
    model._word_expectation_input=None
    model._word_expectation_mask=None
    model.conceptualSpace._word_column=None
    from QueryWork import QueryWorkBudget
    pending = getattr(model, '_pending_attention_meters', ())
    model._pending_attention_meters = ()
    model._attention_meters = pending
    model._attention_forms=None
    if not torch.is_tensor(slab) or not torch.is_tensor(active):return
    stage_percept_expectation(model, active)
    B,W,D=slab.shape
    if len(model._attention_meters) != B:
        model._attention_meters=tuple(QueryWorkBudget(model.attention_budget) for _ in range(B))
    packed = getattr(isp, '_packed_sentence_slot_mask', None)
    if packed is not None:
        rows = getattr(isp, '_packed_sentence_rows', None)
        counts = map(len, rows) if rows is not None else packed.sum(1).tolist()
        for meter, count in zip(model._attention_meters, counts, strict=True):
            meter.size_for_presentation(model.attention_budget, count)
    # Zero is the ordinary-composition control: no bracket walk or thought
    # read is performed. The stem still supplies every word to the VP path.
    if W == 0:return
    fi=getattr(model.perceptualSpace,'_forward_input',None) or {}
    texts=fi.get('word_texts') or ()
    if not any(row for row in texts) and not any(
            value is not None for value in (getattr(isp, '_scalar_values', None) or ())):
        return  # An absent text/scalar field has no bracket to examine.
    intervals=fi.get('word_intervals') or ()
    owner=model._concept_owner()
    from WordSurfaces import is_lexical_form
    native_ids=fi.get('native_indices',fi.get('indices'))
    native_spans=fi.get('native_part_spans',fi.get('part_spans'))
    whole_space=getattr(model,'wholeSpace',None)
    # Only a model whose word step reads its words nominates them here.
    reads_words=bool(getattr(model,'_aligned_serial_word_mode',lambda:False)())

    native_spans_available = torch.is_tensor(fi.get('sentence_bytes')) and torch.is_tensor(fi.get('unit_spans'))
    if native_spans_available:
        from WordSurfaces import reading_forms, trimmed_spans
        byte_surface, units = fi['sentence_bytes'], fi['unit_spans']
        trimmed = trimmed_spans(byte_surface, units)
        spans, ids = reading_forms(byte_surface, units, active, trimmed=trimmed)
    native_rows = property_bits = None

    def found(b, w, form, lo, hi):
        """The word of a form, counting this reading of it first: the first
        reader of a form in a reading is where MTM nominates it (rule 20),
        so that attention reads it as known in that reading."""
        word=owner.interpret.word_for_form(form)
        if (word is not None or not reads_words or not is_lexical_form(form) or whole_space is None
                or not torch.is_tensor(native_ids) or not torch.is_tensor(native_spans)):
            return word
        nonlocal native_rows, property_bits
        if native_rows is None:
            native_rows = torch.cat((native_ids[..., None], native_spans.to(native_ids)), -1).tolist()
        parts = [pid for pid, start, stop in native_rows[b] if pid >= 0 and lo <= start < stop <= hi]
        if not parts:
            return None
        if native_spans_available:
            if property_bits is None:
                from Spaces import _analysis_property_signature
                from WordSurfaces import reading_property_signatures
                lut, discard = _analysis_property_signature(whole_space, device=units.device)
                property_bits = reading_property_signatures(byte_surface, trimmed, lut, discard).tolist()
            bits = property_bits[b][w]
            wholes = [row for row in range(min(int(whole_space.nVectors), 63)) if bits & (1 << row)]
        else:
            wholes = sorted(set(int(v) for v in whole_space.property_rows_for_bytes(form)))
        return owner.interpret.lookup_word(parts, wholes, form=form, word_reading=True, reading=owner.reading(b))
    extents=[];known=[];identities=[];forms=[];concepts=[]
    for b in range(B):
        row=[];seen=[];row_ids=[];names=[];words=[]
        vocabulary = {} if not native_spans_available else None
        for w in range(W):
            text=str(texts[b][w]) if b<len(texts) and w<len(texts[b]) else ''
            raw=text.encode('latin1','replace')
            leading=len(raw)-len(raw.lstrip());surface=raw.strip()
            if raw:
                start, stop = intervals[b][w]
                lo=int(start)+leading
                hi=int(stop)-(len(raw)-len(raw.rstrip()))
            else:
                lo=hi=0
            form=surface.decode('latin1')
            if not native_spans_available:
                row.append((lo,hi) if surface else (0,0))
            word=found(b,w,form,lo,hi) if surface else None
            seen.append(word is not None);words.append(-1 if word is None else int(word))
            if not native_spans_available:
                if form and form not in vocabulary:
                    vocabulary[form] = w
                row_ids.append(vocabulary.get(form, -1))
            names.append(form)
        if not native_spans_available:
            extents.append(row)
            identities.append(row_ids)
        known.append(seen);forms.append(names);concepts.append(words)
    if not native_spans_available:
        spans = torch.tensor(extents, device=slab.device, dtype=torch.long)
        ids = torch.tensor(identities, device=slab.device, dtype=torch.long)
    live=active & (spans[...,1]>spans[...,0])
    spans=torch.where(live[...,None],spans,0)
    model._attention_spans=spans
    if model.attention_budget == 0:return
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
    live_width = max(1, int(occupied_width(live)))
    concepts=torch.tensor(concepts,device=slab.device,dtype=torch.long)
    model._attention_word_concepts=torch.where(known,concepts,-1)
    from Memory import ShortTermMemory
    stm = getattr(model.conceptualSpace, 'stm', model.conceptualSpace)
    history = ShortTermMemory.reserve_look_history(stm, keys, B, model.attention_budget)
    def opening(*, sentence=None, spent_override=None, exploit=None, departure=None, replay_until=None):
        reading_spans=spans[:, :live_width]
        if sentence is not None:
            here=model.inputSpace._packed_sentence_ids[:, :live_width] == sentence
            reading_spans=torch.where(here[...,None],reading_spans,0)
        spent = (spans.new_tensor([meter.spent for meter in model._attention_meters])
                 if spent_override is None else spent_override.to(spans))
        limit=spans.new_tensor([meter.spent + meter.remaining for meter in model._attention_meters])-spent
        return SentenceReading(chooser, keys[:, :live_width], reading_spans, known[:, :live_width],
            poles[:, :live_width], limit.clamp(0, model.attention_budget), W, history,
            exploit=exploit, departure=departure, replay_until=replay_until)
    model._attention_spans=spans
    model._attention_forms=(forms,ids,keys,live)
    phases = getattr(model, 'grammar_phases', None)
    if phases is not None and not phases.vector:
        return
    model._attention_open=opening
    if getattr(model, '_sentence_ends', False):
        # The sentence owns its reading: its looks are run with its
        # composition, one word arriving at each.
        return
    # A read with no sentence takes no look and interprets no word.
    reading=opening().reading()
    model._attention_greedy=reading
    handoff(model,reading)


def handoff(model, reading):
    """Publish the reading's words and their detached evidence, never its values."""
    model._attention_words=reading
    base=getattr(model,"_attention_grammar_mask",None)
    model.inputSpace._ar_grammar_leaf_mask=(reading.accepted if base is None else base & reading.accepted)
    native=model._attention_native_poles
    # A word read whole with no word concept is a positive witness of itself.
    native=torch.where(reading.descended[...,None],
        torch.stack((torch.ones_like(reading.accepted),torch.zeros_like(reading.accepted)),-1).to(native),native)
    model._attention_poles=native.detach()


def stage_expectation(model, *, prefix=None):
    """Stage the admitted arrivals once, shared by this sentence's paths."""
    staged=getattr(model,'_attention_forms',None)
    owner=getattr(model.symbolSpace,'expectation',None)
    if (staged is None or owner is None or 'word' not in owner.enabled_levels
            or getattr(model, '_attention_words', None) is None):
        return
    forms,identities,keys,live=staged
    isp=model.inputSpace
    codes=getattr(isp,'_ar_word_object_atoms',None)
    if not torch.is_tensor(codes):codes=getattr(isp,'_ar_grammar_object_atoms',None)
    if not torch.is_tensor(codes) or codes.shape != keys.shape:codes=keys
    codes=codes.detach()
    bank = codes
    valid = live & (identities == torch.arange(codes.shape[1], device=codes.device))
    accepted=model._attention_words.accepted & live
    lanes=read_poles(model, 'WordExpectation.word_distribution')
    if prefix is not None:
        # Earlier kept arrivals remain the causal word prefix of a packed
        # row. Future sentences have not been admitted by a walk yet.
        earlier=prefix.accepted & live
        accepted=accepted | earlier
        lanes=torch.where(earlier[...,None],prefix.poles,lanes)
    targets=torch.where(accepted,identities,-1)
    stage_symbol_bank(model, owner)
    model._word_expectation_input=(codes,bank,valid,targets)
    expected=owner.expect('word',codes,bank,valid,targets,
        gain=model.word_expectation_gain,active=accepted,
        epsilon=model.expectation_epsilon,saturation=model.expectation_saturation,
        lanes=lanes)
    model._word_expectation=expected
    # The reader receives the attenuated whole; novelty has separate lanes.
    model.conceptualSpace._word_column=expected.content
    model._word_expectation_mask=accepted


def stage_symbol_bank(model, owner):
    """Snapshot the native shared index, including every represented order."""
    space = model._concept_owner()
    rows = sorted({int(row): int(identity) for (_, identity), row in space._csw_rows.items()}.items())
    if not rows:
        return
    basis = space.similarity_codebook
    ids = torch.tensor([identity for _, identity in rows], device=basis.W.device)
    indices = ids.new_tensor([row for row, _ in rows])
    keys = basis.lookup_rows(indices).detach()
    orders = ids.new_tensor([space._concept_source_order(identity) for _, identity in rows])
    owner.set_symbol_bank(keys, identities=ids, orders=orders, addresses=indices)


def stage_percept_expectation(model, active):
    """Allocate this brick's immutable arrival ledger; do not observe yet."""
    owner = model.percept_expectation
    shape = (*active.shape, owner.width)
    model._percept_expected = owner.context_mean.new_zeros(shape)
    model._percept_arrival_loss = owner.context_mean.new_zeros(active.shape)
    model._percept_prior = None


def forecast_percept(model, active, *, history=None):
    """Eager serial boundary, before the current word's percept is computed."""
    owner, expectation = model._concept_owner(), model.percept_expectation
    rows = sorted({row for cid in owner.interpret.lexical_identities()
                   if (row := owner._concept_row_of(cid)) is not None})
    def bank():
        reference = owner.similarity_codebook.W
        for lo in range(0,len(rows),256):
            ids = torch.tensor(rows[lo:lo+256],device=reference.device,dtype=torch.long)
            codes = owner.similarity_codebook.lookup_rows(ids, streams=True)[..., :expectation.width]
            yield ids, codes.detach().expand(len(active), -1, -1)
    from Expectation import ActivationPredictor
    # The completed order-zero state belongs to stream slots. A previous
    # batch's carrier has its own batch extent and cannot be threaded here.
    indices = torch.tensor(sorted(row for row in owner._row_to_concept()
        if owner._row_order(row) == 0), device=active.device, dtype=torch.long)
    weights = owner._observed_knowing[:len(active), indices].detach().amax(-1)
    codes = owner.similarity_codebook.lookup_rows(indices).detach()[..., :expectation.width]
    values = codes[None] * weights[..., None]
    completed = ActivationPredictor.context_from(values, weights > 0)
    return expectation.forecast(bank,len(active),completed=completed,history=history)


class NativeReading:
    """Percept forecasts, observations and their gradients for one speculative walk."""

    def __init__(self, owner, opening, expected, loss):
        self.owner = owner
        self.history = opening.fork()
        self.expected = expected.clone()
        self.loss = loss.clone()
        self.arrivals = {}

    def publish(self, model):
        model._percept_expected = self.expected
        model._percept_arrival_loss = self.loss

    @staticmethod
    def select(model, paths, winners):
        """Select the forecasts and their live losses for the retained rows."""
        from SentenceCompose import select_rows
        if all(path is paths[0] for path in paths):
            paths[0].publish(model)
            return
        def choose(values):
            value = values[0]
            for index, other in enumerate(values[1:], 1):
                value = select_rows(value, other, winners == index)
            return value
        model._percept_expected = choose([path.expected for path in paths])
        model._percept_arrival_loss = choose([path.loss for path in paths])

    @staticmethod
    @torch.no_grad()
    def commit(paths, winners):
        """Advance the existing durable predictor only after a successful close."""
        owner = paths[0].owner
        groups = {}
        for index, path in enumerate(paths):
            # Output-only departures can share a reading. Combine their row
            # masks without reading the selected path numbers back to Python.
            rows = winners == index
            groups[path] = groups[path] | rows if path in groups else rows
        # The persistent owner receives only winning rows, in the same
        # causal arrival order as one live walk. Candidate keys are execution
        # metadata; whether an arrival is retained is a device predicate.
        for key in sorted(set().union(*(path.arrivals for path in groups))):
            active = torch.zeros_like(winners, dtype=torch.bool)
            expected = torch.zeros_like(owner.last_expected[:len(active)])
            observed = torch.zeros_like(expected)
            for path, won in groups.items():
                if key not in path.arrivals:
                    continue
                observation, forecast, seen = path.arrivals[key]
                rows = won & seen
                active = active | rows
                expected = torch.where(rows[:, None], forecast[0].detach(), expected)
                observed = torch.where(rows[:, None], observation, observed)
                # Each path can have different provisional native keys.
                admitted = forecast[1]
                owner.known[admitted] = owner.known[admitted] | rows.any()
            previous = getattr(owner, 'last_arrival', None)
            owner.arrive((expected, winners[:0]), observed, active, sentence=True)
            if previous is not None:
                # A rejected suffix must not erase the last retained arrival.
                # Context, known keys and the update count already use active.
                owner.last_arrival['local_loss'] = torch.where(
                    active.any(), owner.last_arrival['local_loss'], previous['local_loss'])


def observe_percept(model, forecast, payload, sub, index, active, *, path=None, turn=None):
    """Score this arrival against its own pre-observation forecast."""
    expectation = model.percept_expectation
    width = expectation.width
    native = sub[1][:,int(index)]
    observed = native[...,:width]
    if path is None:
        arrival = expectation.arrive(forecast, observed, active, sentence=True)
        local = expectation.last_arrival['local_loss']
    else:
        arrival, local = expectation.preview(path.history, forecast, observed, active)
        path.arrivals[turn, index] = (observed.detach(), (forecast[0].detach(), forecast[1]), active)
    # Different pages can reach one source word on different looks. A
    # later arrival updates only the streams that actually observed it.
    prior = model._percept_expected[:, index]
    loss = model._percept_arrival_loss[:, index]
    model._percept_expected[:, index] = torch.where(active[:, None], arrival.expected, prior)
    model._percept_arrival_loss[:, index] = torch.where(active, local, loss)
    # Identity-valued forward chart, retaining the ordinary source pullback.
    # The two poles of conceptual evidence are unchanged by this P interface.
    from Expectation import recombine_percept
    values = list(payload)
    for offset in (0,6):
        value = values[offset]
        prior = arrival.expected[:,None] if value.ndim == 3 else arrival.expected
        form = recombine_percept(value[...,:width],prior).content
        values[offset] = torch.cat((form,value[...,width:]),-1)
    return tuple(values)
