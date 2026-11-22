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


def sentence_generated(model, record, **walk_options):
    """Unfold this sentence's untransformed end slots through shared generate."""
    from Generative import walk_bank
    from Output import PreparedGeneration
    positions = torch.arange(record.end_slots.shape[1], device=record.root.device)[None]
    occupied = positions < record.end_depth[:, None]
    slots = torch.where(occupied[..., None], record.end_slots, 0.)
    slots = torch.where(((record.end_depth == 1)[:, None] & (positions == 0))[..., None],
                        record.root[:, None], slots)
    bank = walk_bank(model, record.end_slots, record.primed)
    prepared = PreparedGeneration((), slots, bank, None, torch.is_grad_enabled())
    return model.generate(slots, structure=record, occupied=occupied, infix_rows=None,
                          walk_options=walk_options, prepared=prepared,
                          when=record.open_field.when[:, :1])


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


def undone(model, record, programs):
    """The idea taken apart (§19.134, 1): the recorded forest undone from
    its roots, each step by its rule's inverse
    (:func:`DerivationReconstruction.reconstruct`), in the reverse of the
    order of composition, down to the leaves. Returns, per row, the nodes in
    the walk's order (``('step', kind, operation, action, parent)`` and
    ``('leaf', position, source, value)``) and the leaves ``[B, W, D]``."""
    from DerivationReconstruction import reconstruct as undo
    language = model.languageSpace
    steps = []
    recovered, _, _ = undo(language, record, programs,
                           lambda b: model._sentence_leaf_positions(record, b), steps=steps)
    nodes = [[] for _ in range(len(record.root))]
    for entry in steps:
        if entry[0] == 'step':
            _, b, kind, operation, parent = entry
            nodes[b].append(('step', kind, operation, walk_action(language, kind, operation), parent))
        else:
            _, b, position, source = entry
            nodes[b].append(('leaf', position, source, recovered[b, position]))
    return nodes, recovered


def said(model, record, nodes, like):
    """The leaves of :func:`undone` said by the walk's writer
    (:func:`OutputOccurrences.generated`), each at its own reading's place
    (the open field's ``.where`` and wires at the leaf's word). The trace
    is the walk's: the undone steps as its actions, leaves as STOP, in its
    order (roots newest first, the right child first); a step the walk has
    no action for (a bind) is passed through."""
    from Generative import walk_bank, WALK_FEATURE_WIDTH
    from OutputOccurrences import generated
    language = model.languageSpace
    B, D = len(nodes), like.shape[-1]
    stop = len(language._generate_binary_ops) + len(language._generate_unary_ops)
    field = record.open_field
    # The leaves in the input's order (the trace keeps the walk's order; the
    # extension read from it is not this reconstruction's).
    leaves = [sorted((node for node in row if node[0] == 'leaf'), key=lambda node: node[1]) for row in nodes]
    W = max([len(row) for row in leaves] + [1])
    words = like.new_zeros(B, W, D)
    where = field.where.new_zeros(B, W, 2)
    wires = None if field.wires is None else field.wires.new_zeros(B, W, field.wires.shape[-1])
    for b, row in enumerate(leaves):
        for w, (_, position, _, value) in enumerate(row):
            words[b, w] = value
            where[b, w] = field.where[b, position]
            if wires is not None:
                wires[b, w] = field.wires[b, position]
    counts = torch.tensor([len(row) for row in leaves], device=like.device)
    # A binary step the walk has no rule for leaves its row as leaves alone.
    whole = [all(node[0] == 'leaf' or node[3] is not None or
                 node[1] == 2 or
                 language._compose_binary_rules[node[2]].method_name == 'bind' for node in row) for row in nodes]
    taken = [[node for node in row if node[0] == 'leaf' or node[3] is not None] if ok else
             [node for node in row if node[0] == 'leaf'] for row, ok in zip(nodes, whole)]
    T = max([len(row) for row in taken] + [1])
    actions = torch.full((B, T), -1, dtype=torch.long, device=like.device)
    for b, row in enumerate(taken):
        for t, node in enumerate(row):
            actions[b, t] = stop if node[0] == 'leaf' else node[3]
    depths = torch.tensor([int(record.end_depth[b]) if ok and any(node[0] == 'step' for node in row) else
                           len(leaves[b]) for b, (row, ok) in enumerate(zip(taken, whole))],
                          device=like.device).clamp_min(0)
    cw = int(getattr(language, '_generate_policy_width', D))
    flag = torch.ones(B, dtype=torch.bool, device=like.device)
    trace = (actions, torch.zeros_like(actions, dtype=torch.bool), ~flag, like.new_zeros(B, 3), flag, None,
             like.new_zeros(B, T, stop + 1, WALK_FEATURE_WIDTH), like.new_zeros(B, T, cw))
    bank = walk_bank(model, record.end_slots, record.primed)
    generation = generated(model, words, counts, ~flag, bank, trace, when=field.when[:, :1],
                           depths=depths, places=(where, wires))
    return generation, taken


def policy_charge(model, record, generation, taken, errors, active):
    """The walk's policy taught by reconstruction (§19.134; Claude's first
    choice, provisional on Alec): at each undone node, the policy's
    probability of the recorded step (STOP at a leaf), on the menu the walk
    reads there (:func:`Generative.inverse_menu`), charged as its cross
    entropy over that of the uniform choice among the legal actions. A
    recorded step not legal on that menu is not charged, and counted."""
    from Generative import inverse_menu, propose_actions, in_cube
    from FormCentroids import prepare_inverse
    from ObjectiveOwnership import policy_objective
    language = model.languageSpace
    T = max([len(row) for row in taken] + [0])
    if not T or language.generate_policy is None:
        return None
    bank = generation.bank
    require = (not model._aligned_serial_word_mode()
               or (bank.meaning_start is not None and bank.meaning_start < bank.codes.shape[-1]))
    whole = prepare_inverse(bank.form_wholes)
    with torch.no_grad():
        proposed = propose_actions(language, bank.codes.detach(), bank.valid, groups=bank.verb_groups,
            terminal_valid=bank.terminal_valid, constituents=record.constituents,
            constituent_valid=record.constituent_valid, limit=model.reconstruction_basis_limit,
            whole_bank=whole) if require else None
    D = bank.codes.shape[-1]
    cost = bank.codes.new_zeros(len(taken))
    uniform = torch.zeros_like(cost)
    stop = len(language._generate_binary_ops) + len(language._generate_unary_ops)
    unheld = charged = 0
    for t in range(T):
        live = torch.tensor([t < len(row) for row in taken], device=cost.device)
        parent = torch.stack([(row[t][4] if row[t][0] == 'step' else row[t][3]).reshape(-1)[:D].detach()
                              if t < len(row) else cost.new_zeros(D) for row in taken])
        action = torch.tensor([(stop if row[t][0] == 'leaf' else row[t][3]) if t < len(row) else stop
                               for row in taken], device=cost.device)
        # The walk reads its menu only at a value in the cube (each lane in
        # [0, 1]); a recorded node outside it is not charged, and counted.
        outside = live & ~in_cube(parent)
        unheld += int(outside.sum())
        live = live & ~outside
        parent = torch.where(live[:, None], parent, 0.)
        with torch.no_grad():
            menu = inverse_menu(language, parent, live, basis=bank.codes.detach(), basis_valid=bank.valid,
                priming=bank.weights, limit=model.reconstruction_basis_limit,
                case_bank=None if bank.case_bank is None else bank.case_bank.detached(),
                constituents=record.constituents, constituent_valid=record.constituent_valid,
                constituent_families=record.constituent_families, require_symbols=require,
                terminal_valid=bank.terminal_valid, whole_bank=whole, action_candidates=proposed)
        legal = menu.admissible
        logits = language.generate_policy_logits(parent, menu.features).masked_fill(~legal, -torch.inf)
        _, logits = language.choose_generate(torch.where(legal.any(-1, keepdim=True), logits, 0.),
                                             return_logits=True)
        held = live & legal.gather(1, action[:, None])[:, 0]
        unheld += int((live & ~held).sum())
        charged += int(held.sum())
        chosen = torch.log_softmax(logits, -1).gather(1, action[:, None])[:, 0]
        cost = cost + torch.where(held, -chosen, 0.)
        uniform = uniform + torch.where(held, legal.sum(-1).clamp_min(1).to(cost).log(), 0.)
    model._last_walk_policy = dict(charged=charged, unheld=unheld)
    errors.error('reconstruction.walk_policy', cost, uniform, mask=active, category='reconstruction',
                 objective=policy_objective(model, 'generate', 'reconstruction'), side='R')
    return cost


def reconstruction(model, field, *, record=None, programs=(), errors=None, active=None):
    """The one reconstruction (§19.134): from the top the model has back
    down to the percepts. Its upper stretch is the record's composition
    undone (:func:`undone`), the leaves said at their own places
    (:func:`said`), and the walk's policy charged the recorded steps
    (:func:`policy_charge`); its lower stretch is the perceptual inverse.
    The upper stretch is empty with no record (a model with no symbols) and
    in a row with no rule applied: there the leaves are the basic-level
    concepts of ``field`` themselves, and the reconstruction is the
    perceptual inverse alone. Returns the generation (``None`` with no
    record) and what is put back."""
    from OutputOccurrences import realize
    from dataclasses import replace
    if record is None:
        return None, model.reverse(model.conceptualSpace.subspace.carrier_like(), field=field)
    nodes, recovered = undone(model, record, programs)
    generation, taken = said(model, record, nodes, recovered)
    bare = torch.tensor([not any(node[0] == 'step' for node in row) for row in nodes],
                        device=record.root.device)
    if bool(bare.any()):
        pick = lambda first: select_fields((first, field), bare.long())
        generation = replace(generation, field=pick(generation.field),
                             realization_field=pick(generation.realization_field),
                             reconstruction_field=pick(generation.reconstruction_field))
    model._sentence_bare = bare
    reverse = (model.reverse(model.conceptualSpace.subspace.carrier_like(), field=field)
               if bool(bare.all()) else realize(model, generation))
    model._sentence_generation = generation
    model._sentence_undone = nodes
    model._sentence_walk_residuals.append(generation.trace[3])
    model._last_decoder_comparison = None
    if errors is not None:
        policy_charge(model, record, generation, taken, errors, active)
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
    pyramid match) on the ``.what`` of the bytes put back against the
    input's bytes in its readings, the hidden ones left out where
    ``hidden`` (a bool, or ``[B]`` by row: a reconstruction that never saw
    them). Returns the input's
    ``.what`` at each place of its readings, hidden bytes included."""
    from FieldReverse import byte_rows
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
    asked = wanted
    hidden = torch.as_tensor(hidden, device=raw.device).expand(len(raw))
    bytes_hidden = None if targets is None or not bool(hidden.any()) else targets.get('bytes')
    if bytes_hidden is not None:
        bytes_hidden = bytes_hidden & hidden[:, None]
        out = F.pad(bytes_hidden, (0, max(0, scope.shape[1] - bytes_hidden.shape[1])))[:, :scope.shape[1]]
        asked = wanted.masked_fill(out[..., None], 0)
        seen = F.pad(bytes_hidden, (0, max(0, amounts.shape[1] - bytes_hidden.shape[1])))[:, :amounts.shape[1]]
        amounts = amounts.masked_fill(seen[..., None], 0)
    if silent is not None:
        # Silence puts back NULL at every place.
        amounts = torch.where(silent.to(amounts.device)[:, None, None], 0., amounts)
    errors.pyramid(name + '.pyramid', amounts @ codes, asked, mask=active,
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
    # Reconstruction from the idea is asked for every byte, the hidden ones
    # too; one from the open field (no rule applied) never saw them.
    model._sentence_target = charge(model, errors, 'reconstruction_order_zero', second, field, active,
        hidden=model._sentence_bare, silent=None if silent is None or not bool(silent.any()) else silent)
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
