"""The live parallel field shared by sentence trials and the common decoder."""
import torch
from torch.nn import functional as F
from Understanding import ConceptualField, InputReconstruction


def inventory(owner, field):
    ids = field.concept_ids
    return torch.tensor([(-1 if int(cid) < 0 else owner._concept_row_of(int(cid)))
                         for cid in ids.reshape(-1).tolist()],
                        dtype=torch.long, device=ids.device).reshape(ids.shape)


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
    zero = torch.tensor([int(row) >= 0 and owner._row_order(int(row)) == 0
                         for row in addresses.reshape(-1).tolist()],
                        device=addresses.device).reshape(addresses.shape)
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
    from DerivationReconstruction import recorded_forest
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
        forest = recorded_forest(language, program)
        if depth != len(forest):
            raise ValueError('the recorded derivation must have the actual forest depth')
        values = (record.root[b:b + 1] if depth == 1 else record.end_slots[b, :depth])
        # Slot j holds forest[depth - 1 - j]; keep the composed roots in order.
        kept = [j for j in range(depth) if forest[depth - 1 - j][0] != 0]
        met['roots_composed'] += len(kept)
        met['roots_alone'] += depth - len(kept)
        if not kept:
            continue
        met['rows'] += 1
        for k, j in enumerate(kept):
            slots[b, k] = values[j]
            occupied[b, k] = True
        # The walk's stack, bottom to top: the oldest kept root at the bottom.
        stack = [forest[depth - 1 - j] for j in reversed(kept)]
        taken = []
        while stack:
            node = stack.pop()
            if node[0] == 0:
                taken.append(stop)
                continue
            kind, operation = node[0], node[1]
            rules = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
            name = rules[operation].method_name
            if kind == 1 and name == 'bind':
                met['binds'] += 1
                stack.append(node[3])
                continue
            action = walk_action(language, kind, operation)
            if action is None:
                met['lacking'] += 1
                met['lacking_rules'][name] = met['lacking_rules'].get(name, 0) + 1
                taken.append(stop)
                continue
            met['steps'] += 1
            taken.append(action)
            stack.extend(node[3:])
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
    parts = store.parts
    try:
        head = tree.trace.head
    except ValueError:
        return None
    if head is None:
        return None
    from Memory import OCCURRENCE, SYMBOL
    from FieldReverse import drawn
    met['rows_followed'] += 1
    met['roots_alone' if head[0] == OCCURRENCE else 'roots_composed'] += 1
    owner = model._concept_owner()
    registry = model.where_registry
    lo, hi = registry.slices['text']
    wires = registry.code[lo:hi]
    start = store.when_start
    words = []
    for kind, name in parts.leaves(head):
        if kind == SYMBOL:
            continue  # An operator of the parse has no spoken occurrence of its own.
        if kind != OCCURRENCE:
            raise ValueError('a reading leaf must name an occurrence')
        row = store.index_of_row(name)
        reference = store.referent_of(row)
        if reference is None or reference[0] != 'sym':
            met['words_unnamed'] += 1
            continue
        concept = int(reference[1])
        address = owner._concept_row_of(concept)
        if address is None:
            met['words_unnamed'] += 1
            continue
        band = store.slots[row, start - 4:start].detach().to(wires)
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


def policy_charge(model, generation, errors, active):
    """The walk's policy taught by reconstruction (§19.134; Claude's first
    choice, provisional on Alec): at each step the walk was made to take,
    the policy's probability of that step (STOP at a leaf) on the menu the
    walk read there (the trace's features and parents), charged as its
    cross entropy over that of the uniform choice among the legal actions.
    A step taken off the menu has no probability and is not charged; it is
    counted."""
    from Generative import WALK_FEATURE_WIDTH
    from ObjectiveOwnership import policy_objective
    language = model.languageSpace
    actions, features, tops = generation.trace[0], generation.trace[-2], generation.trace[-1]
    if language.generate_policy is None or actions.shape[1] == 0:
        return None
    B, T = actions.shape
    taken = actions >= 0
    action = actions.clamp_min(0)
    legal = features[..., WALK_FEATURE_WIDTH - 1].bool()                          # [B, T, A]
    held = taken & legal.gather(-1, action[..., None])[..., 0]
    flat = lambda value: value.reshape(B * T, *value.shape[2:])
    logits = language.generate_policy_logits(flat(tops).detach(),
                                             flat(features[..., :WALK_FEATURE_WIDTH - 1]).detach())
    logits = logits.masked_fill(~flat(legal), -torch.inf)
    _, logits = language.choose_generate(torch.where(flat(legal).any(-1, keepdim=True), logits, 0.),
                                         return_logits=True)
    chosen = torch.log_softmax(logits, -1).gather(1, flat(action)[:, None])[:, 0].reshape(B, T)
    cost = torch.where(held, -chosen, 0.).sum(1)
    uniform = torch.where(held, legal.sum(-1).clamp_min(1).to(cost).log(), 0.).sum(1)
    model._last_walk_policy = dict(charged=int(held.sum()), unheld=int((taken & ~held).sum()))
    errors.error('reconstruction.walk_policy', cost, uniform, mask=active, category='reconstruction',
                 objective=policy_objective(model, 'generate', 'reconstruction'), side='R')
    return cost


def put_back(model, field, *, record=None, programs=()):
    """What :func:`reconstruction` puts back, with nothing kept on the
    model: ``(generation, reverse, met)``; the generation and ``met`` are
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


def reconstruction(model, field, *, record=None, programs=(), errors=None, active=None):
    """The one reconstruction (§19.134, §19.135): from the top the model
    has back down to the percepts. With a record (a model that
    understands), its upper stretch is the output walk from the idea made
    to take the recorded steps (:func:`taken_steps`), each step the walk's
    own inverse of its parent alone; nothing is read from the journal's
    frames or from the open field, and a concept outside every composition
    is not put back. The walk's policy is charged the steps it was made to
    take (:func:`policy_charge`). The lower stretch is the perceptual
    inverse of the walk's leaves. With no record, or a grammar of no rule
    (a model that reads), the reconstruction is the perceptual inverse of
    the basic-level concepts of ``field``. Returns the generation (``None``
    with no record) and what is put back."""
    generation, reverse, met = put_back(model, field, record=record, programs=programs)
    if generation is None:
        return None, reverse
    model._sentence_generation = generation
    model._sentence_taken = met
    model._sentence_walk_residuals.append(generation.trace[3])
    model._last_decoder_comparison = None
    if errors is not None:
        policy_charge(model, generation, errors, active)
    return generation, reverse


def selected_field(first, second, wins):
    """Publish each stream's kept generated field, with one union of its rows."""
    return select_fields((first, second), wins.long())


def select_fields(fields, winners):
    """Select streams by concept identity, including differently sized reviews."""
    picked = [fields[int(index)] for index in winners]
    ids = sorted({int(cid) for b, field in enumerate(picked) for cid in
        (field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:, b])})
    width = max(field.evidence.shape[2] for field in picked)
    evidence, where, when, wires = [], [], [], []
    placed = all(field.wires is not None for field in picked)
    for b, field in enumerate(picked):
        column = field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:, b]
        lookup = {int(cid): i for i, cid in enumerate(column)}
        zero = field.evidence.new_zeros(1, field.evidence.shape[2], 2)
        values = torch.cat((field.evidence[:, b], zero), 0)
        indices = column.new_tensor([lookup.get(cid, len(column)) for cid in ids])
        evidence.append(F.pad(values[indices], (0, 0, 0, width - values.shape[1])))
        where.append(F.pad(field.where[b], (0, 0, 0, width - field.where.shape[1])))
        when.append(F.pad(field.when[b], (0, 0, 0, width - field.when.shape[1])))
        if placed:
            wires.append(F.pad(field.wires[b], (0, 0, 0, width - field.wires.shape[1])))
    return ConceptualField(fields[0].concept_ids.new_tensor(ids), torch.stack(evidence, 1),
                           torch.stack(where), torch.stack(when), None,
                           torch.stack(wires) if placed else None)


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
    values = []
    for word in range(field.where.shape[1]):
        begin, end = field.where[:, word].unbind(-1)
        mask = (positions[None] >= begin[:, None]) & (positions[None] < end[:, None])
        values.append(torch.where(mask[..., None], event, 0.).amax(1))
    result = torch.stack(values, 1)
    return F.pad(result, (0, width - result.shape[-1])) if result.shape[-1] < width else result[..., :width]


def charge(model, errors, name, carrier, field, active, *, silent=None, hidden=True):
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
    from FieldReverse import byte_rows, content, forms_in
    from Omission import target as omission_target
    targets = omission_target(model)
    raw = (model._staged_concepts_in if targets is None else targets['raw'])[:, 0].long()
    positions = torch.arange(raw.shape[1], device=raw.device)
    scope = ((positions[None, None] >= field.where[..., :1]) &
             (positions[None, None] < field.where[..., 1:])).any(1)
    radix = model.perceptualSpace.percept_store
    rows = byte_rows(radix).to(raw.device)
    amounts = carrier._radix_activity.amounts
    codes = (radix._basis.lookup_rows(rows.clamp_min(0)) * (rows >= 0)[:, None]).to(amounts)    # [256, what]
    wanted = codes[raw.clamp(0, 255)] * scope[..., None].to(codes)
    asked = scope
    hidden = torch.as_tensor(hidden, device=raw.device).expand(len(raw))
    bytes_hidden = None if targets is None or not bool(hidden.any()) else targets.get('bytes')
    if bytes_hidden is not None:
        bytes_hidden = bytes_hidden & hidden[:, None]
        out = F.pad(bytes_hidden, (0, max(0, scope.shape[1] - bytes_hidden.shape[1])))[:, :scope.shape[1]]
        asked = scope & ~out
        seen = F.pad(bytes_hidden, (0, max(0, amounts.shape[1] - bytes_hidden.shape[1])))[:, :amounts.shape[1]]
        amounts = amounts.masked_fill(seen[..., None], 0)
    if silent is not None:
        # Silence puts back NULL at every place.
        amounts = torch.where(silent.to(amounts.device)[:, None, None], 0., amounts)
    rows_of, spellings = forms_in(radix, raw.clamp(0, 255), asked)
    form_codes = radix._basis.lookup_rows(rows_of).to(amounts)
    read = F.one_hot(raw.clamp(0, 255), 256).to(amounts) * asked[..., None].to(amounts)
    errors.pyramid(name + '.pyramid', content(amounts, codes, spellings, form_codes),
                   content(read, codes, spellings, form_codes).detach(), mask=active,
                   weight=model.loss.reconstruction_scale, category='reconstruction')
    return wanted


def reconstruct(model, record, errors, active, *, programs=(), silent=None):
    """The sentence's one reconstruction (:func:`reconstruction`) and its
    error. A silent sentence (``silent`` ``[B]``, rule S -> NULL) puts back
    NULL at every place, and that is scored against the sentence."""
    field = record.open_field
    generation, second = reconstruction(model, field, record=record, programs=programs,
                                        errors=errors, active=active)
    if silent is not None and bool(silent.any()):
        from Interpret import SilenceLayer
        put = second._reconstruction_percepts
        gate = silent.to(put.device).reshape(-1, *([1] * (put.ndim - 1)))
        second._reconstruction_percepts = torch.where(gate, SilenceLayer().reverse(put), put)
    model._sentence_order_zero_field = generation.field
    publish(model, generation.field)
    # Reconstruction from the idea is asked for every byte, the hidden ones too.
    model._sentence_target = charge(model, errors, 'reconstruction_order_zero', second, field, active,
        hidden=False, silent=None if silent is None or not bool(silent.any()) else silent)
    B, W, D = record.word_values.shape
    def rows(prefix):
        return sum(term['weight'] * term.get('multiplier', 1.) * errors._value(term)
                   for name, term in errors._terms.items() if name.startswith(prefix + '.'))
    cost = rows('reconstruction_order_zero')
    # Placeless modalities participate in the field, not in the word ledger.
    ideas = word_percepts(second, field, D)[:, :W]
    from Omission import target as omission_target
    targets = omission_target(model)
    raw = (model._staged_concepts_in if targets is None else targets['raw'])[:, 0]
    decoded = second.materialize()[:, 0]
    length = min(raw.shape[-1], decoded.shape[-1])
    positions = torch.arange(length, device=raw.device)
    scope = ((positions[None, None] >= field.where[..., :1]) &
             (positions[None, None] < field.where[..., 1:])).any(1)
    byte_error = ((raw[:, :length] != decoded[:, :length]) & scope).sum(-1) / scope.sum(-1).clamp_min(1)
    count = record.word_valid.sum(-1)
    model._last_decoder_byte_identity = byte_error.detach()
    errors.error('reconstruction.free_bytes', byte_error.detach(), 1., mask=active,
        weight=model.loss.reconstruction_scale, category='reconstruction', trained=False)
    missing = (decoded[:, :length].eq(0) & scope).sum(-1)
    errors.error('reconstruction.coverage', missing.to(cost).detach(), scope.sum(-1), mask=active,
        weight=model.loss.reconstruction_scale, category='reconstruction', trained=False)
    prior = record.percept_expected
    if prior is not None:
        from Expectation import percept_reconstruction
        correction, target_residual, rebuilt, squared = percept_reconstruction(ideas[..., :prior.shape[-1]],
            record.word_values[..., :prior.shape[-1]], prior)
        model._last_percept_reconstruction = (dict(
            correction=correction.detach(), target=target_residual.detach(),
            rebuilt=rebuilt.detach(), expected=prior.detach(),
            observed=record.word_values[..., :prior.shape[-1]].detach(),
            residual_loss=squared.detach(),
            full_loss=(rebuilt - record.word_values[..., :prior.shape[-1]].detach()).square().detach()),)
        valid = record.word_valid
        hidden = None if targets is None else targets.get('bytes')
        if hidden is not None:
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            omitted = ((positions[None, None] >= field.where[..., :1]) &
                       (positions[None, None] < field.where[..., 1:]) & hidden[:, None]).any(-1)
            valid = valid & ~omitted[:, :W]
        residual = (squared * valid[..., None]).sum((1, 2)) / (valid.sum(-1) * prior.shape[-1]).clamp_min(1)
        errors.error('reconstruction.free_percept', residual, 1., mask=active,
            weight=model.loss.reconstruction_scale, category='reconstruction')
    truncated = missing > 0
    model._sentence_order_zero_reverse = second
    trace = generation.trace
    model._last_decoder_trace = (generation.words, generation.counts, generation.truncated,
        trace[3], trace[0], trace[4], trace[-2], trace[-1])
    sid = int(record.sentence)
    sentence_costs = ideas.new_zeros(B, record.depths.shape[1])
    sentence_costs[:, sid] = cost
    result = (ideas, ideas.new_zeros(B), cost, truncated, sentence_costs)
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


def selected(first, second, wins):
    """Keep each stream's input result from its selected sentence trial."""
    from dataclasses import fields
    def choose(a, b):
        if a is None and b is None:
            return None
        if a.shape != b.shape:
            # Two trials may paste over buffers of different extents.
            shape = [max(x, y) for x, y in zip(a.shape, b.shape)]
            pad = lambda t: F.pad(t, [n for d in reversed(range(t.ndim)) for n in (0, shape[d] - t.shape[d])])
            a, b = pad(a), pad(b)
        return torch.where(wins.reshape((-1,) + (1,) * (a.ndim - 1)), b, a)
    return InputReconstruction(**{item.name: choose(getattr(first, item.name), getattr(second, item.name))
                                 for item in fields(InputReconstruction)})
