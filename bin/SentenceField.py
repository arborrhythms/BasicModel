"""The live parallel field shared by sentence trials and the common decoder."""
import torch
from torch.nn import functional as F
from Understanding import ConceptualField, InputReconstruction


def inventory(owner, field):
    ids = field.concept_ids
    if ids is getattr(owner, "_cs_field_concept_ids", None):
        return torch.where(ids >= 0, owner._cs_field_rows, -1)
    return owner._concept_allocator.layer(0).concept_rows(ids)


def publish(model, field):
    owner, carrier = model._concept_owner(), model.conceptualSpace.subspace
    addresses = inventory(owner, field)
    carrier._concept_activations = field.evidence
    carrier._concept_ids = field.concept_ids
    carrier._concept_inventory_rows = addresses
    carrier._concept_where, carrier._concept_when = field.where, field.when
    carrier._concept_code_owner = next(i for i, space in enumerate(model.conceptualSpaces) if space is owner)
    carrier._concept_codes = owner._field_codes(owner.similarity_codebook.getW(), addresses).detach()


def descended(model, field):
    """Give the common decoder only the order-zero face of knowing."""
    owner = model._concept_owner()
    addresses = inventory(owner, field)
    evidence = owner.cs_reverse_presence(field.evidence, inventory_rows=addresses)
    zero = (addresses >= 0) & (addresses < owner._order_caps()[0])
    zero = zero[:, None, None, None] if zero.ndim == 1 else zero[..., None, None]
    return ConceptualField(field.concept_ids, evidence * zero, field.where, field.when, None, field.wires)


def knowing(model):
    carrier = model.conceptualSpace.subspace
    return ConceptualField(carrier._concept_ids, carrier._concept_activations,
        carrier._concept_where, carrier._concept_when, None)


def open_field(model, sid, a0):
    owner = model._concept_owner()
    field = ConceptualField(owner._cs_field_concept_ids,
        F.pad(a0, (0, 0, 0, 0, 0, 0, 0, sum(owner._field_caps()) - len(a0))),
        owner._cs_field_where, owner._cs_field_when, None)
    from ScalarPercepts import join
    field = join(model,sid,field)
    from IntuitionField import open_field as intuit
    field = intuit(model, field)
    wires = getattr(owner, '_cs_field_wires', None)
    if wires is not None:
        # A reading the bank has no occurrence for (a joined scalar) has none.
        from dataclasses import replace
        field = replace(field, wires=F.pad(wires, (0, 0, 0, field.where.shape[1] - wires.shape[1])))
    model._sentence_open_fields[int(sid)] = field
    publish(model, field)
    return field


def sentence_generated(model, record, *, slots=None, occupied=None, places=None, **walk_options):
    """Unfold this sentence's idea through the shared generate walk: its
    untransformed end slots, or ``slots`` ``[B, K, D]`` (newest first) with
    ``occupied`` ``[B, K]`` where given. ``places``: by row, the extent of
    each leaf the walk puts out, in order, where its occurrence gives it
    (``None`` for a row with none)."""
    from Generative import walk_bank
    from Output import PreparedGeneration
    if slots is None:
        positions = torch.arange(record.end_slots.shape[1], device=record.root.device)[None]
        occupied = positions < record.end_depth[:, None]
        slots = torch.where(occupied[..., None], record.end_slots, 0.)
        slots = torch.where(((record.end_depth == 1)[:, None] & (positions == 0))[..., None],
                            record.root[:, None], slots)
    bank = walk_bank(model, record.end_slots, record.primed)
    prepared = PreparedGeneration((), slots, bank, None, torch.is_grad_enabled())
    return model.generate(slots, structure=record, occupied=occupied, infix_rows=None,
                          walk_options=walk_options, prepared=prepared,
                          when=record.open_field.when[:, :1], places=places)


def walk_action(language, kind, operation):
    """The walk's action for a recorded step, by its rule's name; ``None``
    where the walk has no such rule (a bind, a rule the walk lacks)."""
    if kind == 1:
        name = language._compose_binary_rules[operation].method_name
        names = [rule.method_name for rule in language._generate_binary_rules]
        return names.index(name) if name in names else None
    name = language._compose_unary_rules[operation].method_name
    names = [rule.method_name for rule in language._generate_unary_rules]
    return len(language._generate_binary_ops) + names.index(name) if name in names else None


def taken_steps(model, record, programs):
    """What the walk is made to take to put the sentence back.

    Where the trial retains a typed head (sigma-pi §19.146), follow it to
    the located leaves. Each word is read by the concept its row refers to and at
    the place its row holds (:func:`said_words`). The walk is given them as
    its slots, each said by a STOP with that concept's code as the leaf, and
    takes no other step. A head naming one occurrence supplies one slot.

    Elsewhere the recorded composition is the walk's actions (§19.135, 1):
    each root of the reading's forest that a rule composed is a slot of the
    walk's stack (newest first, as the end slots are); the walk pops its top
    and takes the inverse of the parent's recorded rule (its action by the
    rule's name, :func:`walk_action`), its right child above its left, and
    STOP at a leaf. A ``bind`` passes its parent to its one child with no
    action. A recorded rule the walk has no rule of the same name for is
    taken as STOP at that parent, and counted.

    Returns the slots ``[B, K, D]``, ``occupied`` ``[B, K]``, the actions
    ``[B, T]`` (``-1`` past a row's last), a count of what was met, the
    operands given (``values`` ``[B, T, 2, D]``, ``given`` ``[B, T]``) and,
    by row, each said word's ``(start, reference)`` in the order the walk
    puts them out (``None`` for a row that follows no edge)."""
    from DerivationReconstruction import recorded_index
    language = model.languageSpace
    stop = len(language._generate_binary_ops) + len(language._generate_unary_ops)
    B, K, D = record.end_slots.shape
    T = int(model._walk_budget())
    met = dict(rows=0, roots_composed=0, roots_alone=0, steps=0, binds=0, lacking=0, over_budget=0,
               unfinished=0, lacking_rules={}, rows_followed=0, words_followed=0, words_unnamed=0)
    said = [said_words(model, programs[b] if b < len(programs) else None, b, met) for b in range(B)]
    K = max([K] + [len(words) for words in said if words])
    slots = record.end_slots.new_zeros(B, K, D)
    occupied = torch.zeros(B, K, dtype=torch.bool, device=slots.device)
    actions = torch.full((B, T), -1, dtype=torch.long, device=slots.device)
    operands = record.end_slots.new_zeros(B, T, 2, D)
    followed = torch.zeros(B, T, dtype=torch.bool, device=slots.device)
    places = [None] * B
    for b in range(B):
        program = programs[b] if b < len(programs) else None
        depth = int(record.end_depth[b])
        if program is None or depth <= 0:
            met['unfinished'] += depth < 0
            continue
        words = said[b]
        if words is not None:
            if not words:
                continue
            met['rows'] += 1
            n = min(len(words), T)
            met['over_budget'] += len(words) > T
            # The walk's top is the newest slot; it pops the last word first
            # and puts its words out in the reverse of its pops.
            for t in range(n):
                code = words[n - 1 - t][2][:D].to(slots)
                slots[b, t, :len(code)] = code
                occupied[b, t] = True
                actions[b, t] = stop
                followed[b, t] = True
                operands[b, t, :, :len(code)] = code
            places[b] = [(start, ('sym', concept)) for concept, start, _ in words[:n]]
            continue
        reading = recorded_index(language, program)
        roots = reading.roots
        if depth != len(roots):
            raise ValueError('the recorded derivation must have the actual forest depth')
        values = (record.root[b:b + 1] if depth == 1 else record.end_slots[b, :depth])
        # Slot j holds forest[depth - 1 - j]; keep the composed roots in order.
        kept = [j for j in range(depth) if reading.steps[roots[depth - 1 - j]][0] != 0]
        met['roots_composed'] += len(kept)
        met['roots_alone'] += depth - len(kept)
        if not kept:
            continue
        met['rows'] += 1
        for k, j in enumerate(kept):
            slots[b, k] = values[j]
            occupied[b, k] = True
        # The walk's stack, bottom to top: the oldest kept root at the bottom.
        stack = [roots[depth - 1 - j] for j in reversed(kept)]
        taken = []
        while stack:
            node = stack.pop()
            kind, operation, _ = reading.steps[node]
            if kind == 0:
                taken.append(stop)
                continue
            rules = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
            name = rules[operation].method_name
            if kind == 1 and name == 'bind':
                met['binds'] += 1
                stack.append(reading.operands[node][0])
                continue
            action = walk_action(language, kind, operation)
            if action is None:
                met['lacking'] += 1
                met['lacking_rules'][name] = met['lacking_rules'].get(name, 0) + 1
                taken.append(stop)
                continue
            met['steps'] += 1
            taken.append(action)
            stack.extend(reading.operands[node])
        if len(taken) > T:
            met['over_budget'] += 1
            taken = taken[:T]
        actions[b, :len(taken)] = torch.tensor(taken, device=actions.device)
    return slots, occupied, actions, met, (operands, followed), places


def said_words(model, program, stream, met):
    """The words the trial's STM puts back for a reading, by following its
    head there (sigma-pi §19.146): the word rows under it in order, each as ``(concept, start, code)``: the
    concept its row refers to; the start of the place its row holds, read
    back from the row's ``.where`` band by the model's where wires (``None``
    where the row claims no place); and the
    concept's code, looked up as the answer looks up a perceived word
    (:func:`Output.perceived_meanings`). A row that names no concept is not
    said, and counted. ``None`` where the STM holds no head for the reading."""
    tree = getattr(program, 'tree', None)
    if tree is None:
        return None
    store = tree.ideas
    try:
        head = tree.trace.head
    except ValueError:
        return None
    if head is None:
        return None
    from Memory import OCCURRENCE
    from FieldReverse import drawn
    from architecture import WHERE_WIRES
    met['rows_followed'] += 1
    met['roots_alone' if head[0] == OCCURRENCE else 'roots_composed'] += 1
    owner = model._concept_owner()
    registry = model.where_registry
    lo, hi = registry.slices['text']
    wires = registry.grid[lo:hi]
    start = store.when_start
    words = []
    for row, concept, address in owner.mtm.read_occurrences(store, head):
        if address is None:
            met['words_unnamed'] += 1
            continue
        band = store.slots[row, start - WHERE_WIRES:start].detach().to(wires)
        # A row with no .where claim has no place: it is said in turn.
        place = (int(drawn(-(wires - band).square().sum(-1))) if bool(band.ne(0).any())
                 else None)
        rows = torch.tensor([address], device=store.slots.device)
        with torch.no_grad():
            code = owner.similarity_codebook.lookup_rows(rows, stream=stream)
            code = owner.interpret.lexical_atoms(rows, code, stream=stream)[0]
        words.append((concept, place, code))
        met['words_followed'] += 1
    return words


def put_back(model, field, *, record=None, programs=()):
    """Inspect the inverse of a recorded grammatical derivation. Return ``(generation, reverse, met)``; the generation and ``met`` are
    ``None`` in a model that reads."""
    from OutputOccurrences import realize
    from dataclasses import replace
    language = model.languageSpace
    reads = not (getattr(language, '_compose_binary_rules', ()) or getattr(language, '_compose_unary_rules', ()))
    if record is None or reads:
        return None, model.reverse(model.conceptualSpace.subspace.carrier_like(), field=field), None
    slots, occupied, actions, met, given, places = taken_steps(model, record, programs)
    # The reading's constituents are kept of the way up: not candidates here.
    idea = replace(record, constituents=None, constituent_valid=None, constituent_families=None)
    generation = sentence_generated(model, idea, slots=slots, occupied=occupied, places=places,
                                    kept_actions=actions, kept_taken=True, kept_operands=given)
    return generation, realize(model, generation), met


def reconstruction(model, field):
    """Invert opening evidence through the common perceptual inverse."""
    return model.reverse(model.conceptualSpace.subspace.carrier_like(), field=field)


def selected_field(first, second, wins):
    """Publish each stream's kept generated field, with one union of its rows."""
    return select_fields((first, second), wins.long())


def select_fields(fields, winners, *, indices=None):
    """Align candidate identities once, then gather all retained streams."""
    from SentenceCompose import select_streams
    if not torch.is_tensor(winners):
        winners = torch.as_tensor(winners, device=fields[0].evidence.device, dtype=torch.long)
    compiling = torch.compiler.is_compiling()
    # These two host schema choices preserve the compact public field shape.
    # All numerical retention below is independent of the number of streams.
    if indices is None and not compiling:
        indices = winners.tolist()
    checked = range(len(fields)) if compiling or indices is None else indices
    width = max(fields[index].evidence.shape[2] for index in checked)
    placed = all(fields[index].wires is not None for index in checked)
    wire_width = next(field.wires.shape[-1] for field in fields if field.wires is not None) if placed else 0
    batch, count = len(winners), max(len(field.concept_ids) for field in fields)
    sentinel = torch.iinfo(torch.long).min
    if compiling and fields[0].concept_ids.ndim == 1 and all(
            f.concept_ids is fields[0].concept_ids or (f.concept_ids.shape[0] == 0 and fields[0].concept_ids.shape[0] == 0)
            for f in fields):
        values, where, when, wires = [], [], [], []
        for field in fields:
            value = field.evidence.permute(1, 0, 2, 3)[:, :, :width]
            values.append(F.pad(value, (0, 0, 0, width - value.shape[2])))
            for source, target in ((field.where, where), (field.when, when)):
                target.append(F.pad(source[:, :width], (0, 0, 0, max(0, width - source.shape[1]))))
            if placed:
                source = field.wires
                wires.append(field.where.new_zeros(batch, width, wire_width) if source is None else
                    F.pad(source[:, :width], (0, 0, 0, max(0, width - source.shape[1]))))
        evidence = select_streams(values, winners).transpose(0, 1)
        return ConceptualField(fields[0].concept_ids, evidence, select_streams(where, winners),
                               select_streams(when, winners), None,
                               select_streams(wires, winners) if placed else None)
    names, values, where, when, wires = [], [], [], [], []
    for field in fields:
        ids = field.concept_ids
        ids = ids[None].expand(batch, -1) if ids.ndim == 1 else ids.T
        names.append(F.pad(ids, (0, count - ids.shape[1]), value=sentinel))
        value = field.evidence.permute(1, 0, 2, 3)[:, :, :width]
        values.append(F.pad(value, (0, 0, 0, width - value.shape[2], 0, count - value.shape[1])))
        for source, target in ((field.where, where), (field.when, when)):
            target.append(F.pad(source[:, :width], (0, 0, 0, max(0, width - source.shape[1]))))
        if placed:
            source = field.wires
            wires.append(field.where.new_zeros(batch, width, wire_width) if source is None else
                F.pad(source[:, :width], (0, 0, 0, max(0, width - source.shape[1]))))
    selected_names = select_streams(names, winners)
    ids = selected_names[selected_names != sentinel].unique(sorted=True)
    positions = torch.searchsorted(ids, selected_names.contiguous())
    positions = torch.where(selected_names == sentinel, len(ids), positions)
    # Repeated identities keep the last seat, with one extra sink for padding.
    seats = positions.new_full((batch, len(ids) + 1), count)
    seats = seats.scatter_reduce(1, positions,
        torch.arange(count, device=positions.device)[None].expand(batch, -1), 'amax', include_self=False)[:, :-1]
    selected = select_streams(values, winners)
    selected = torch.cat((selected, selected.new_zeros(batch, 1, width, 2)), 1)
    evidence = selected.gather(1, seats[..., None, None].expand(-1, -1, width, 2)).transpose(0, 1)
    return ConceptualField(ids, evidence, select_streams(where, winners), select_streams(when, winners), None,
                           select_streams(wires, winners) if placed else None)


def native_target(model, field):
    """Observed per-byte native events are targets, never decoder inputs."""
    ps = model.perceptualSpace
    from Omission import target as omission_target
    targets = omission_target(model)
    if targets is not None and 'native_values' in targets:
        values, valid = targets['native_values'], targets['native_valid']
        positions = torch.arange(values.shape[1], device=values.device).expand(len(values), -1)
        owned = ((positions[:, :, None] >= field.where[:, None, :, 0]) &
                 (positions[:, :, None] < field.where[:, None, :, 1])).any(-1)
        return ps._radix_code_events(values.clone(), positions).detach() * (valid & owned)[..., None]
    native = ps._forward_input if targets is None else targets['native']
    ids, spans = native['native_indices'], native['native_part_spans']
    if 'byte_width' in native:
        # The element stem already staged one physical byte row per position.
        # The original target owns these indices even after attention moves.
        width = max(1, native['byte_width'])
        byte_ids = ids[:, :width]
    else:
        # A promoted native part can span several bytes on the parallel path.
        # The target still has the same per-byte layout as the common decoder.
        width = max(1, int(spans[..., 1].max()) if spans.shape[1] else 0)
        byte_ids = ids.new_full((len(ids), width), -1)
        for batch, (parts, extents) in enumerate(zip(ids.tolist(), spans.tolist())):
            for part, (begin, end) in zip(parts, extents):
                if part < 0 or end <= begin:
                    continue
                raw = ps.percept_store.bytes_for(part)
                if len(raw) != end - begin:
                    raise ValueError('observed native part disagrees with its byte bracket')
                rows = [ps.percept_store.get_id(bytes([byte])) for byte in raw]
                if any(row is None for row in rows):
                    raise ValueError('observed native part has an unadmitted byte')
                byte_ids[batch, begin:end] = ids.new_tensor(rows)
    positions = torch.arange(width, device=ids.device).expand(len(ids), -1)
    event = ps._radix_part_events(byte_ids.clamp_min(0), positions).detach()
    owned = ((positions[:, :, None] >= field.where[:, None, :, 0]) &
             (positions[:, :, None] < field.where[:, None, :, 1]) &
             (field.where[:, None, :, 1] > field.where[:, None, :, 0])).any(-1)
    return event * ((byte_ids >= 0) & owned)[..., None]


def word_percepts(reverse, field, width):
    """A reporting/forecast view of decoded native rows at field brackets."""
    event = reverse._reconstruction_percepts
    positions = torch.arange(event.shape[1], device=event.device)
    # Bound the temporary [stream, word, byte, wire] mask application on
    # long pages. Each maximum still covers the complete original byte axis.
    chunk = max(1, (1 << 22) // max(1, event.numel()))
    values = []
    for bounds in field.where.split(chunk, dim=1):
        begin, end = bounds.unbind(-1)
        mask = (positions >= begin[..., None]) & (positions < end[..., None])
        values.append(torch.where(mask[..., None], event[:, None], 0.).amax(2))
    result = torch.cat(values, 1)
    return F.pad(result, (0, width - result.shape[-1])) if result.shape[-1] < width else result[..., :width]


def charge(model, errors, name, carrier, field, active, *, silent=None, hidden=True, text_width=None):
    """A reconstruction's error (:meth:`Layers.Error.pyramid`, the spatial
    pyramid match) on the content of the bytes put back against that of the
    input's bytes in its readings, the hidden ones left out where
    ``hidden`` (a bool, or ``[B]`` by row: a reconstruction that never saw
    them). The content is read from the bytes on both sides alike
    (:func:`FieldReverse.content`, §19.135, 4): at a place, the ``.what``
    of the byte there and of each form of the percept store present in the
    bytes there; the forms are those present in the input's bytes
    (:func:`FieldReverse.forms_in`). Returns the input's bytes' ``.what``
    at each place of its readings, hidden bytes included."""
    from FieldReverse import byte_rows, content, sentence_forms
    from Omission import target as omission_target
    targets = omission_target(model)
    raw = (model._staged_concepts_in if targets is None else targets['raw'])[:, 0].long()
    full_width = raw.shape[1]
    if text_width is not None:
        raw = raw[:, :text_width]
    positions = torch.arange(raw.shape[1], device=raw.device)
    scope = ((positions[None, None] >= field.where[..., :1]) &
             (positions[None, None] < field.where[..., 1:])).any(1)
    radix = model.perceptualSpace.percept_store
    rows = byte_rows(radix).to(raw.device)
    amounts = getattr(carrier, '_reconstruction_amounts', carrier._radix_activity.amounts)
    codes = (radix._basis.lookup_rows(rows.clamp_min(0)) * (rows >= 0)[:, None]).to(amounts)    # [256, what]
    wanted = codes[raw.clamp(0, 255)] * scope[..., None].to(codes)
    asked = scope
    hidden = torch.as_tensor(hidden, device=raw.device).expand(len(raw))
    bytes_hidden = None if targets is None else targets.get('bytes')
    if bytes_hidden is not None:
        bytes_hidden = bytes_hidden & hidden[:, None]
        out = F.pad(bytes_hidden, (0, max(0, scope.shape[1] - bytes_hidden.shape[1])))[:, :scope.shape[1]]
        asked = scope & ~out
        seen = F.pad(bytes_hidden, (0, max(0, amounts.shape[1] - bytes_hidden.shape[1])))[:, :amounts.shape[1]]
        amounts = amounts.masked_fill(seen[..., None], 0)
    if silent is not None:
        # Silence puts back NULL at every place.
        amounts = torch.where(silent.to(amounts.device)[:, None, None], 0., amounts)
    rows_of, spellings, found, cover = sentence_forms(radix, raw.clamp(0, 255), asked, with_cover=True)
    form_codes = torch.where(found[:, None], radix._basis.lookup_rows(rows_of).to(amounts), 0.)
    with torch.no_grad():
        target = wanted * asked[..., None]
        if rows_of.numel():
            target = target + cover.to(form_codes) @ form_codes
    # Keep the spatial pyramid's original levels and the target's public shape.
    if raw.shape[1] < full_width:
        padding = (0, 0, 0, full_width - raw.shape[1])
        target = F.pad(target, padding)
        wanted = F.pad(wanted, padding)
    errors.pyramid(name + '.pyramid', content(amounts, codes, spellings, form_codes),
                   target, mask=active,
                   weight=model.loss.reconstruction_scale, category='reconstruction')
    return wanted


def _reconstruct_core(recon_percepts, field_where, cost, raw, decoded,
                      word_values, word_valid, prior, bytes_hidden, sid_tensor, depths_width):
    """Compute reconstruction error tensors without mutating Python registries."""
    B, W, D = word_values.shape
    positions = torch.arange(recon_percepts.shape[1], device=recon_percepts.device)
    chunk = max(1, (1 << 22) // max(1, recon_percepts.numel()))
    chunks = []
    for bounds in field_where.split(chunk, dim=1):
        begin, end = bounds.unbind(-1)
        mask = (positions >= begin[..., None]) & (positions < end[..., None])
        chunks.append(torch.where(mask[..., None], recon_percepts[:, None], 0.).amax(2))
    ideas = torch.cat(chunks, 1)
    ideas = (F.pad(ideas, (0, D - ideas.shape[-1])) if ideas.shape[-1] < D else ideas[..., :D])[:, :W]
    length = min(raw.shape[-1], decoded.shape[-1])
    dec_pos = torch.arange(length, device=raw.device)
    scope = ((dec_pos[None, None] >= field_where[..., :1]) &
             (dec_pos[None, None] < field_where[..., 1:])).any(1)
    byte_error = ((raw[:, :length] != decoded[:, :length]) & scope).sum(-1) / scope.sum(-1).clamp_min(1)
    missing = (decoded[:, :length].eq(0) & scope).sum(-1)
    residual = None
    residual_debug = None
    if prior is not None:
        from Expectation import percept_reconstruction
        correction, target_residual, rebuilt, squared = percept_reconstruction(
            ideas[..., :prior.shape[-1]], word_values[..., :prior.shape[-1]], prior)
        residual_debug = (
            correction.detach(), target_residual.detach(),
            rebuilt.detach(), prior.detach(),
            word_values[..., :prior.shape[-1]].detach(), squared.detach(),
            (rebuilt - word_values[..., :prior.shape[-1]].detach()).square().detach(),
        )
        valid = word_valid
        if bytes_hidden is not None:
            hid_pos = torch.arange(bytes_hidden.shape[1], device=bytes_hidden.device)
            omitted = ((hid_pos[None, None] >= field_where[..., :1]) &
                       (hid_pos[None, None] < field_where[..., 1:]) & bytes_hidden[:, None]).any(-1)
            valid = valid & ~omitted[:, :W]
        residual = (squared * valid[..., None]).sum((1, 2)) / (valid.sum(-1) * prior.shape[-1]).clamp_min(1)
    truncated = missing > 0
    sentence_costs = ideas.new_zeros(B, depths_width).scatter(
        1, sid_tensor.reshape(1, 1).expand(B, 1), cost[:, None])
    return ideas, ideas.new_zeros(B), cost, truncated, sentence_costs, byte_error, missing, residual, residual_debug


def reconstruct(model, record, errors, active, second, *, sentence=None, _core_fn=None):
    """Score the shared opening inverse at this sentence's observed locations."""
    field = record.open_field
    model._sentence_order_zero_field = field
    def rows(prefix):
        return sum(term['weight'] * term.get('multiplier', 1.) * errors._value(term)
                   for name, term in errors._terms.items() if name.startswith(prefix + '.'))
    cost = rows('reconstruction_order_zero')
    from Omission import target as omission_target
    targets = omission_target(model)
    raw = (model._staged_concepts_in if targets is None else targets['raw'])[:, 0]
    decoded = second.materialize()[:, 0]
    prior = record.percept_expected
    bytes_hidden = None if targets is None else targets.get('bytes')
    sid_tensor = (record.sentence.to(device=record.word_values.device, dtype=torch.long)
                  if torch.is_tensor(record.sentence)
                  else torch.tensor(int(record.sentence), device=record.word_values.device, dtype=torch.long))
    fn = _core_fn if _core_fn is not None else _reconstruct_core
    ideas, zeros_b, cost, truncated, sentence_costs, byte_error, missing, residual, residual_debug = fn(
        second._reconstruction_percepts, field.where, cost, raw, decoded,
        record.word_values, record.word_valid, prior, bytes_hidden, sid_tensor, record.depths.shape[1])
    scope_sum = ((torch.arange(min(raw.shape[-1], decoded.shape[-1]), device=raw.device)[None, None] >= field.where[..., :1])
                 & (torch.arange(min(raw.shape[-1], decoded.shape[-1]), device=raw.device)[None, None] < field.where[..., 1:])).any(1).sum(-1)
    model._last_decoder_byte_identity = byte_error.detach()
    errors.error('reconstruction.free_bytes', byte_error.detach(), 1., mask=active,
        weight=model.loss.reconstruction_scale, category='reconstruction', trained=False)
    errors.error('reconstruction.coverage', missing.to(cost).detach(), scope_sum, mask=active,
        weight=model.loss.reconstruction_scale, category='reconstruction', trained=False)
    if prior is not None:
        corr, targ, rebuilt, exp, obs, rloss, floss = residual_debug
        model._last_percept_reconstruction = (dict(
            correction=corr, target=targ, rebuilt=rebuilt, expected=exp,
            observed=obs, residual_loss=rloss, full_loss=floss),)
        errors.error('reconstruction.free_percept', residual, 1., mask=active,
            weight=model.loss.reconstruction_scale, category='reconstruction')
    sid = int(sentence) if sentence is not None else int(record.sentence)
    result = (ideas, zeros_b, cost, truncated, sentence_costs)
    model._input_reconstructions[sid] = InputReconstruction(ideas, second.materialize(), result[1],
        cost, truncated, sentence_costs, second._reconstruction_percepts)
    return result


def completed(model):
    values = tuple(model._input_reconstructions.values())
    if not values:
        raise RuntimeError('input reconstruction requires a completed parallel field')
    width = max(value.event.shape[-1] for value in values)
    event = torch.stack([F.pad(value.event, (0, width - value.event.shape[-1])) for value in values]).amax(0)
    percepts = torch.stack([F.pad(value.percepts, (0, 0, 0, width - value.percepts.shape[1])) for value in values]).sum(0)
    return InputReconstruction(torch.stack([v.ideas for v in values]).sum(0), event,
        torch.stack([v.idea_cost for v in values]).sum(0), torch.stack([v.byte_cost for v in values]).sum(0),
        torch.stack([v.truncated for v in values]).any(0), torch.stack([v.sentence_costs for v in values]).sum(0), percepts)


_RECONSTRUCTION_FIELDS = ('ideas', 'event', 'idea_cost', 'byte_cost', 'truncated', 'sentence_costs', 'percepts')


def select_reconstructions(readings, winners):
    """Gather all retained reconstructions, padding their public byte extents."""
    from SentenceCompose import select_padded
    values = {}
    for name in _RECONSTRUCTION_FIELDS:
        columns = [getattr(reading, name) for reading in readings]
        if all(column is None for column in columns):
            values[name] = None
            continue
        if any(column is None for column in columns):
            raise ValueError('sentence reconstructions must share their tensor fields')
        values[name] = select_padded(columns, winners)
    return InputReconstruction(**values)


def selected(first, second, wins):
    """Keep each stream's input result from its selected sentence trial."""
    def choose(a, b):
        if a is None and b is None:
            return None
        if a.shape != b.shape:
            # Two trials may paste over buffers of different extents.
            shape = [max(x, y) for x, y in zip(a.shape, b.shape)]
            pad = lambda t: F.pad(t, [n for d in reversed(range(t.ndim)) for n in (0, shape[d] - t.shape[d])])
            a, b = pad(a), pad(b)
        return torch.where(wins.reshape((-1,) + (1,) * (a.ndim - 1)), b, a)
    return InputReconstruction(**{name: choose(getattr(first, name), getattr(second, name))
                                 for name in _RECONSTRUCTION_FIELDS})
