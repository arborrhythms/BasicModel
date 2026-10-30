"""Sentence-open familiarity and kept-close promotion in the shared MTM pool."""
from dataclasses import replace
from functools import wraps

import torch


def kept_close(method):
    """Batch class locations after every active stream's placement work."""
    @wraps(method)
    def boundary(model, state, sid, active, *args, **kwargs):
        owner = model._concept_owner()
        previous = getattr(owner, '_coverage_deferred', False)
        streams = getattr(owner, '_mtm_active_streams', None)
        owner._coverage_deferred = True
        owner._mtm_active_streams = set(active.nonzero().flatten().tolist())
        try:
            return method(model, state, sid, active, *args, **kwargs)
        finally:
            owner._coverage_deferred = previous
            owner._mtm_active_streams = streams
    return boundary


def centroid(owner, support):
    rows = list(support)
    codes = owner.similarity_codebook.lookup_rows(rows)
    weights = codes.new_tensor(list(support.values()))
    return (codes * weights[:, None]).sum(0) / weights.sum().clamp_min(1e-12)


def _clock(owner, batch):
    return owner.mtm.closes.get(batch, 0)+1


def witnessed_support(owner, cid):
    """Read sparse durable witnesses; MTM can be empty at an epoch boundary."""
    alloc = owner._concept_allocator
    records = alloc.store_of(cid).constituents(cid)
    saved = [ref for role, ref in records if role == 'candidate_support']
    if saved:
        return {row: weight for symbol, weight in saved
                if (row := owner._csw_row_of(symbol)) is not None}
    members = class_members(owner, cid)
    for role, ref in records:
        if role == 'part' and isinstance(ref, tuple) and ref[:1] == ('field',):
            row = owner._csw_row_of(ref[1])
            if row is not None:
                members[row] = max(members.get(row, 0.), ref[2])
    return members


@torch.no_grad()
def witnessed_match(owner, order, support, *, literal_key=None):
    """One bounded vector comparison at a time, never candidates by inventory."""
    alloc = owner._concept_allocator
    value = centroid(owner, support)
    for row in owner._witnessed_rows():
        if owner._row_order(row) != order:
            continue
        cid = owner.concept_id_at_row(row)
        records = alloc.store_of(cid).constituents(cid)
        literals = [value for role, value in records if role == 'candidate_literal']
        if literal_key is not None and literal_key in literals:
            return cid
        if literals or literal_key is not None:
            # Located conjunctions include their native literal witnesses.
            # Equal field rows alone cannot identify different conjunctions.
            continue
        members = witnessed_support(owner, cid)
        if set(members) != set(support):
            continue
        code = owner.similarity_codebook.lookup_rows([row])[0]
        if float(torch.nn.functional.cosine_similarity(value, code, dim=0)) >= owner.concept_match_cos:
            return cid
    return None


def remember_candidate(owner, cid, support, literal_key=None):
    store = owner._concept_allocator.store_of(cid)
    store.set_role_constituents(cid, 'candidate_support',
        [(owner.concept_id_at_row(row), weight) for row, weight in support.items()])
    if literal_key is not None:
        store.set_role_constituents(cid, 'candidate_literal', [literal_key])


def remember_context(owner, focal, support):
    """A reused row still witnesses its focal row's observed context."""
    if focal is None:
        return
    cid = owner.concept_id_at_row(focal)
    if cid is not None:
        owner._concept_allocator.store_of(cid).set_role_constituents(cid, 'familiar_context',
            [(owner.concept_id_at_row(row), weight) for row, weight in support.items()])


def use_witnessed(owner, cid, support, batch):
    """A durable match is an access; new members extend coverage by maximum."""
    store = owner._concept_allocator.store_of(cid)
    accesses = next((value for role, value in store.constituents(cid) if role == 'familiar_access'), 0)
    store.set_role_constituents(cid, 'familiar_access', [accesses + 1])
    previous = witnessed_support(owner, cid)
    merged = {row: max(weight, previous.get(row, 0.)) for row, weight in support.items()}
    merged.update({row: weight for row, weight in previous.items() if row not in merged})
    if merged != previous and owner._csw_row_of(cid) not in merged:
        cover_members(owner, cid, merged)
        remember_candidate(owner, cid, merged)
    owner.__dict__.setdefault('_field_activated_references', {}).setdefault(batch, set()).add(cid)
    return cid


@torch.no_grad()
def observe_candidate(owner, order, support, *, batch=0, pair=(1., 0.), literal=None, literal_key=None, focal=None):
    pool = owner.mtm
    support = pool._support(support)
    if not support:
        return None
    remember_context(owner, focal, support)
    # A native conjunction also has ordered percept support; a lone field
    # row is trivial only when there is no such additional witness.
    if literal is None and len(support) == 1 and next(iter(support)) in owner._witnessed_rows():
        use_witnessed(owner, owner.concept_id_at_row(next(iter(support))), support, batch)
        return None
    existing = witnessed_match(owner, order, support,
        literal_key=literal if literal_key is None else literal_key)
    if existing is not None:
        use_witnessed(owner, existing, support, batch)
        return None
    value = centroid(owner, support)
    key = pool.match_candidate(order, value)
    if literal is not None:
        # Located native conjunctions retain their literal identity. A pooled
        # both reading must not identify two different ordered percepts.
        key = next((key for key, entry in pool.entries.items()
                    if entry.order == order and entry.refinement.get('literal_key', entry.refinement.get('literal'))
                    == (literal if literal_key is None else literal_key)), None)
    elif key is not None and pool.entries[key].refinement.get('literal') is not None:
        key = None
    when = _clock(owner, batch)
    if key is None:
        key = pool.open_candidate(order=order, when=when, value=value, support=support)
    elif (batch, when) not in [item[:2] for item in pool.entries[key].readings]:
        pool.update_candidate(key, when=when, value=value, support=support)
        entry = pool.entries[key]
        pool.entries[key] = entry.with_actual(value=centroid(owner, entry.support))
    entry = pool.entries[key]
    refinement = dict(entry.refinement)
    if literal is not None:
        refinement['literal'] = literal
        refinement['literal_key'] = literal if literal_key is None else literal_key
    if focal is not None:
        contexts = dict(refinement.get('contexts', {}))
        contexts[int(focal)] = dict(support)
        refinement['contexts'] = dict(list(contexts.items())[-pool.field_width:])
    pool.entries[key] = replace(entry, refinement=refinement)
    pool.record_reading(key, batch, pair)
    entry = pool.entries[key]
    error = min(max((item[pole] for item in entry.readings), default=0.) for pole in (2,3))
    if refinement.get('clock') != (batch, when):
        best = refinement.get('best', float('inf'))
        refinement.update(best=min(best, error) if error else float('inf'),
            stale=0 if not error or error < best else refinement.get('stale', 0)+1,
            clock=(batch, when))
    refinement['both'] = bool(error)
    pool.entries[key] = replace(entry, refinement=refinement)
    owner.__dict__.setdefault('_field_activated_references', {}).setdefault(batch, set()).add(key)
    return key


@torch.no_grad()
def observe_field(owner, field=None, *, active_streams=None):
    """Observe one open field; no inventory identity is allocated here."""
    from ConceptEvidence import symbols
    if (not owner._promotion_enabled or getattr(owner, '_online_learning_frozen', False)):
        return {}
    field = getattr(owner, '_promo_last_acts', None) if field is None else field
    if field is None:
        return {}
    if not bool(torch.isfinite(field).all()):
        raise RuntimeError('open attention field contains NaN/Inf')
    B = field.shape[1]
    if active_streams is None:
        active_streams = torch.ones(B, dtype=torch.bool)
    if tuple(active_streams.shape) != (B,):
        raise ValueError('open field activity differs from its streams')
    owner._refinement_field = field.detach()
    owner.refine_over_collected(field=field)
    addresses = owner._field_inventory_rows(field.device)
    readings = symbols(field).detach()
    activated = {}
    witnessed = set(owner._witnessed_rows())
    # Snapshot familiar contexts before this observation; a newly opened
    # candidate cannot supply its own alternative in the same field.
    contexts = [(focal, context) for entry in owner.mtm.entries.values()
                for focal, context in entry.refinement.get('contexts', {}).items()]
    for focal in witnessed:
        cid = owner.concept_id_at_row(focal)
        saved = owner._concept_allocator.store_of(cid).constituents(cid)
        context = {row: weight for role, value in saved if role == 'familiar_context'
                   for symbol, weight in (value,) if (row := owner._csw_row_of(symbol)) is not None}
        if context:
            contexts.append((focal, context))
    for batch in range(B):
        if not bool(active_streams[batch]):
            continue
        rows = addresses[:, batch] if addresses.ndim == 2 else addresses
        paired = {int(row): readings[slot, batch] for slot, row in enumerate(rows)
                  if int(row) >= 0 and float(readings[slot, batch].amax()) > 0}
        activated[batch] = tuple(row for row, pair in paired.items() if float(pair[0]) > 0)
        used = {row: float(pair.amax()) for row, pair in paired.items()
                if row in witnessed and float(pair.amax()) > owner.concept_use_floor}
        for focal in used:
            context = {row: weight for row, weight in used.items() if row != focal}
            if not context:
                continue
            order = owner._row_order(focal)
            value = centroid(owner, context)
            for alternative, previous in contexts:
                if alternative in used or owner._row_order(alternative) != order:
                    continue
                cosine = torch.nn.functional.cosine_similarity(value, centroid(owner, previous), dim=0)
                if float(cosine) >= owner.concept_match_cos:
                    dense = field.new_zeros(owner.nVectors)
                    dense[list(context)] = dense.new_tensor(list(context.values()))
                    owner.maybe_raise_order(focal, alternative, dense, batch=batch, evidence=None)
                    break
            observe_candidate(owner, order, context, batch=batch,
                              pair=tuple(float(x) for x in paired[focal]), focal=focal)
        if owner.conceptual_pi:
            observe_native_cases(owner, batch, used)
        # MTM's leading entries are read through their live member codes;
        # similarity to this field is use, not an invented positive witness.
        if used:
            current = centroid(owner, used)
            for key, entry in tuple(owner.mtm.entries.items()):
                cosine = torch.nn.functional.cosine_similarity(entry.value.to(current), current, dim=0)
                if float(cosine) > owner.concept_use_floor:
                    owner.__dict__.setdefault('_field_activated_references', {}).setdefault(batch, set()).add(key)
    return activated


@torch.no_grad()
def observe_native_cases(owner, batch, support):
    model = getattr(owner, '_model', None)
    ps = getattr(model, 'perceptualSpace', None)
    native = getattr(ps, 'percept_store', None)
    presented = getattr(ps, '_forward_input', None)
    perception = getattr(owner, '_percept_field', None)
    if native is None or not isinstance(presented, dict) or perception is None:
        return
    ids = presented.get('native_indices', presented.get('indices'))
    spans = presented.get('native_part_spans', presented.get('part_spans'))
    brackets = perception.where[batch].to(spans)
    # Read each complete located run, including its unfused first witness.
    # Native recurrence may replace the run by one PS id on presentation two;
    # its byte identity keeps the same candidate and its first observation.
    for bracket in brackets.tolist():
        start, end = bracket
        if end <= start:
            continue
        choices = sorted([(int(span[0]), -int(span[1]), int(pid))
            for pid, span in zip(ids[batch].tolist(), spans[batch].tolist())
            if pid >= 0 and start <= span[0] < span[1] <= end])
        cursor, group = start, []
        for left, negative_right, pid in choices:
            if left == cursor:
                group.append(pid)
                cursor = -negative_right
        if cursor != end or not group:
            continue
        pieces = [native.bytes_for(pid) for pid in group]
        if not all(pieces):
            continue
        raw = b''.join(pieces)
        if len(raw) < 2:
            continue
        if owner.interpret.word_for_form(raw) is not None:
            continue
        if support:
            observe_candidate(owner, 0, support, batch=batch,
                literal=('ps', tuple(group)), literal_key=raw)


def open_sentence(model, sid, present):
    """Every present word bracket, independent of the grammar's attention."""
    owner, isp = model._concept_owner(), model.inputSpace
    owner._field_activated_references = {}
    if not hasattr(model, '_attention_spans'):
        return {}  # Metadata-only callers have no native percept field.
    ps, ws = model.perceptualSpace, model.wholeSpace
    mask = isp._word_active_mask & (isp._packed_sentence_ids == int(sid)) & present[:, None]
    spans = model._attention_spans
    if spans.shape != (*mask.shape, 2):
        raise ValueError('open attention requires the native word brackets')
    brackets = torch.where(mask[..., None], spans, 0)
    native, raw = ps._forward_input, model._staged_concepts_in
    whole_spans = getattr(ws, '_staged_word_property_spans', None)
    whole_spans = (ws.concept_evidence_layout(raw, ws.inputShape[0])[0] if whole_spans is None
                   else whole_spans.reshape(raw.shape[0], -1, 2))
    field = owner.cs_read_memberships((native.get('native_indices', native['indices']),
        native.get('native_part_spans', native['part_spans']),
        ws.subspace.what.primitive_properties, raw, whole_spans), brackets)
    owner._cs_last_a0 = field.detach()
    _, acts = owner.cs_forward_content(field, None)
    model._definition_fold_field = acts.detach()
    from ConceptEvidence import symbols
    readings = symbols(acts)
    addresses = owner._field_inventory_rows(acts.device)
    seen = {}
    for b in range(len(present)):
        rows = addresses[:, b] if addresses.ndim == 2 else addresses
        seen[b] = rows[(rows >= 0) & (readings[:, b, 0] > 0)].tolist() if bool(present[b]) else []
    if model.training:
        owner.observe_open_field(acts, active_streams=present)
    model._last_open_attention = dict(sentence=int(sid), brackets=brackets.detach().clone(),
        field_rows=addresses.detach().clone(), seen=seen)
    return seen


def member_row(owner, kind, identity):
    from LTM1 import SYMBOL, OCCURRENCE
    if kind == SYMBOL:
        return owner._csw_row_of(identity)
    if kind != OCCURRENCE:
        return None
    model = getattr(owner, '_model', None)
    store = getattr(getattr(model, 'symbolSpace', None), 'ltm_store', None)
    if store is None:
        store = owner._closed_clause_store() if hasattr(owner, '_closed_clause_store') else None
    if store is None:
        return None
    key = owner.mtm.token_of(('ltm', store.address_domain, int(identity)))
    if key is not None:
        symbol = owner.mtm.entries[key].symbol
        return None if symbol is None else owner._csw_row_of(symbol)
    row = store.index_of_row(identity)
    if row is None:
        return None
    seat = int(store.recurrence_readings[row])
    symbol = int(store.recurrence_symbols[seat]) if seat >= 0 else None
    return None if symbol is None or symbol <= 0 else owner._csw_row_of(symbol)


def class_members(owner, cid):
    from LTM1 import SYMBOL
    parts = owner._concept_allocator.parts
    members = {}
    for kind, identity in parts.parts((SYMBOL, cid)):
        row = member_row(owner, kind, identity)
        edge = parts.find((kind, identity), (SYMBOL, cid))
        weight = max(0., float(parts.truth_plus[edge]-parts.truth_minus[edge]))
        if row is not None and weight:
            members[row] = max(weight, members.get(row, 0.))
    return members


@torch.no_grad()
def cover_members(owner, cid, members):
    """The same sigma, location and form initialization for every raise."""
    from MemoryConsolidation import _fold
    members = {int(row): float(weight) for row, weight in members.items() if weight > 0}
    alloc = owner._concept_allocator
    alloc.symbol_types.setdefault(cid, set()).add('concept')
    order = alloc.order_of(cid)
    destination = owner._csw_concept_row(order, cid)
    if destination is None:
        raise RuntimeError('class coverage exceeds the constructed inventory')
    owner._populate_concept_weights(cid, witness=(members, ()))
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    if derived is not None:
        derived.placement_derived[destination] = True
        derived.placement_valid[destination] = False
        owner._priming_edge_revision = None
        if not getattr(owner, '_coverage_deferred', False):
            derived.derive_class_placements()
    value = _fold(owner, None, list(members), members=members)
    if value is not None:
        width = len(value) if derived is None else derived.percept_width
        owner.similarity_codebook.W[destination, :width].copy_(value[:width])
        if derived is not None:
            derived._form_valid[destination] = False
    alloc.layer().witnessed[destination] = True
    return destination


@torch.no_grad()
def commit_read_words(owner, reading, *, batch, document, position, clause=None, written=None):
    """Retain identified word mentions once, at the kept reading's close.

    A grammar's participant selection does not erase a word that was read.
    The observed lexical order supplies this occurrence's witness; promotion
    still belongs to the common update after every active stream closes.
    """
    if reading is None or reading.word_ids is None:
        return {}
    from Meaning import ConceptualMeaning
    from Occurrence import native_content_key
    pool, store = owner.mtm, owner._closed_clause_store()
    occurrences = {}
    retained, pending = {}, [clause]
    while pending:
        node = pending.pop()
        if node is None:
            continue
        pending.extend((*node.children, *node.companions))
        term = node.composition
        location = (written or {}).get(id(node))
        if term is not None and len(term.leaves) == 1 and isinstance(location, int):
            retained[term.leaves[0]] = location
    for index, word in enumerate(reading.word_ids.tolist()):
        if word <= 0 or owner.definitions.objects(word):
            continue
        name = pool.names.get(('name', word))
        key = name.token if name is not None else next((name.token for name in pool.names.values()
            if name.reading == word and name.token in pool.entries), None)
        if key is None or not pool.entries[key].pending or pool.entries[key].order is not None:
            continue
        value = reading.leaves[index]
        meaning = replace(ConceptualMeaning.from_description(value), bindings={
            '_identity': key, '_forward_references': ((0,word),),
            '_open_references': (('referent',0),)})
        order = owner._concept_source_order(word)
        row = retained.get(index)
        if (row is None or int(store.order[row]) != order
                or not torch.equal(store.slots[row], value.to(store.slots))):
            # This is a retained unknown referent, not an asserted answer.
            # Use the existing open-question storage contract and evidence.
            row = store.append_meaning(meaning, kind='question', evidence=(0., 0.),
                order=order, stream=batch, when=False,
                document_key=('reading', document, word, index), sentence_index=position,
                content_key=native_content_key(meaning))
        occurrence = store.occurrence_of(row)
        occurrences[index] = occurrence
        pool.bind(key, occurrence, kind=None, value=value, reading=word,
                  when=pool.closes.get(batch, 0)+1,
                  where=None if reading.symbol_where is None else reading.symbol_where[index])
        owner.__dict__.setdefault('_field_activated_references', {}).setdefault(batch, set()).add(occurrence)
    return occurrences


@torch.no_grad()
def promote_identity(owner, key, *, word=None):
    """A kept definition or recurrent mention supplies an object's witness."""
    pool, alloc = owner.mtm, owner._concept_allocator
    entry = pool.entries[key]
    if entry.symbol is not None:
        return entry.symbol
    if not entry.occurrences:
        raise RuntimeError('identity promotion requires a kept mention')
    if entry.native is not None:
        pool.promote(key, entry.native)
        return entry.native
    origin = entry.occurrences[0]
    store = owner._closed_clause_store()
    index = None if store is None else store.location(origin)
    member_order = -1 if index is None else owner._concept_source_order(origin[-1])
    if member_order < 0:
        record = None if index is None else store.raw_meaning_of(index)
        raise RuntimeError(f'identity promotion has no witnessed order: occurrence={origin!r}, record={record!r}')
    owner._preflight_concept_allocation(1, context='MTM identity promotion')
    cid = alloc.new_concept()
    # An identity's retained occurrence supplies its order. No grammar choice
    # or caller order is supplied to this operation.
    from LTM1 import OCCURRENCE
    alloc.store_of(cid).add_constituent(cid, 'part', ('occ', origin[-1]))
    alloc.reference_orders[cid] = member_order + 1
    alloc.settle(cid)
    row = owner._csw_concept_row(alloc.order_of(cid), cid)
    if row is None:
        raise RuntimeError('identity promotion exceeds the constructed inventory')
    alloc.individuals.add(cid)
    alloc.symbol_types.setdefault(cid, set()).update(('concept', 'object'))
    from MemoryConsolidation import _fold
    actual = set(entry.occurrences) | {event.occurrence for event in entry.events if event.role == 0}
    members = [store.location(address) for address in actual]
    members = [row for row in members if isinstance(row, int) and int(store.order[row]) == 0]
    form = _fold(owner, store, members)
    value = entry.value if form is None else form
    owner.similarity_codebook.W[row].copy_(value.to(owner.similarity_codebook.W))
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    if derived is not None:
        derived.remember_content(row, entry.value)
        from Occurrence import sentence_key
        streams = pool.accesses.get(key, {})
        batch = next(reversed(streams), 0) or 0
        context, close_key = getattr(owner, '_mtm_placement_contexts', {}).get(batch,
            (owner.placement_context(batch), sentence_key([str(key).encode()])))
        derived.place_rows([row], context, close_key)
    pool.promote(key, cid)
    words = {pool.reading_of(address) for address in entry.occurrences} - {None}
    if word is not None:
        words.add(word)
    for name in words:
        owner.interpret.define(name, cid)
    owner._priming_edge_revision = None
    return cid


@torch.no_grad()
def define_pending(owner, left, right, *, word=None):
    """Consume a kept equality/IS witness once its subject token is retained."""
    from LTM1 import SYMBOL, OCCURRENCE
    from ConceptIndex import index_part_row, unbound_label
    store = owner._closed_clause_store()
    reference = left[1] if left[0] == SYMBOL else (
        ('ltm', store.address_domain, left[1]) if left[0] == OCCURRENCE and store is not None else None)
    key = owner.mtm.token_of(reference)
    if key is None and word is not None:
        key = owner.mtm.token_of(('name', int(word)))
    if key is None or not owner.mtm.entries[key].pending or owner.mtm.entries[key].order is not None:
        return False
    # An unresolved or order-incompatible shared side is still a premise,
    # not a defining class witness. Use the origin's witnessed order before
    # allocating; index_part_row must never be asked to invent compatibility.
    alloc = owner._concept_allocator
    entry = owner.mtm.entries[key]
    if not entry.occurrences:
        return False  # The kept reading will retain its mentions before finish_definitions.
    member_order = owner._concept_source_order(entry.occurrences[0][-1]) + 1
    shared_order = owner._concept_source_order(right[1])
    existing_class = right[0] == SYMBOL and ('part-base', right[1]) in alloc.relate_idx
    if not (unbound_label(owner, right) or member_order == shared_order
            or existing_class and member_order == shared_order-1):
        return False
    cid = promote_identity(owner, key, word=word)
    if left != right:
        parent = index_part_row(owner, (SYMBOL, cid), right)
        if unbound_label(owner, right):
            owner.interpret.define(right[1], parent)
    return True


def finish_definitions(owner):
    pending = owner.__dict__.pop('_kept_definition_witnesses', ())
    for left, right, word in pending:
        define_pending(owner, left, right, word=word)


@torch.no_grad()
def promote_candidates(owner):
    pool, alloc = owner.mtm, owner._concept_allocator
    if getattr(owner, '_online_learning_frozen', False):
        return []
    result = []
    for key, entry in tuple(pool.entries.items()):
        if entry.symbol is not None:
            continue
        active = getattr(owner, '_mtm_active_streams', None)
        if active is not None and not any(stream in active and clock == pool.closes.get(stream)
                                         for stream, clock in pool.accesses.get(key, {}).items()):
            continue
        if entry.order is not None:
            existing = witnessed_match(owner, entry.order, entry.support,
                literal_key=entry.refinement.get('literal_key', entry.refinement.get('literal')))
            if existing is not None:
                use_witnessed(owner, existing, entry.support, next(iter(pool.accesses.get(key, {})), 0))
                pool.promote(key, existing)
                result.append(existing)
                continue
            if (entry.refinement.get('literal') is None and len(entry.support) == 1
                    and next(iter(entry.support)) in owner._witnessed_rows()):
                pool._detach(key)
                continue
        readings = [item for item in entry.readings
                    if pool.closes.get(item[0], 0)-item[1] <= pool.horizon]
        if entry.order is None:
            mentions = entry.refinement.get('mentions', {})
            # Several closing adapters can reference the same read site.
            # Recurrence counts distinct kept readings, not those aliases.
            count = (len({(stream, clock) for stream, clock in mentions.values()
                          if pool.closes.get(stream, 0)-clock <= pool.horizon})
                     if pool.closes else len(entry.occurrences))
        else:
            count = len(readings) if readings else entry.count
        if count < owner.concept_mint_count:
            continue
        if entry.order is None:
            result.append(promote_identity(owner, key))
            continue
        lanes = tuple(max((item[pole] for item in readings), default=0.) for pole in (2, 3))
        if not owner._promotion_enabled or min(lanes) > 0:
            continue
        support = {row: weight for row, weight in entry.support.items() if weight >= owner.concept_part_floor}
        if not support:
            continue
        entry = replace(entry, support=support)
        pool.entries[key] = entry
        owner._preflight_concept_allocation(1, context='MTM candidate promotion')
        cid = alloc.new_concept()
        literal = entry.refinement.get('literal')
        if entry.order == 0:
            # A literal conjunction stays in the perceptual order. Its field
            # support is initialization evidence, not a higher-order member.
            alloc.identity[cid] = (None, None)
            if literal is not None:
                for pid in literal[1]:
                    alloc.add(cid, 'part', pid)
            # Field conjunction members are same-order form witnesses.
            # Their records keep that distinction from a raised sigma class.
            scale = max(entry.support.values())
            for member, weight in entry.support.items():
                symbol = owner.concept_id_at_row(member)
                if symbol is not None:
                    alloc.store_of(cid).add_constituent(cid, 'part', ('field', symbol, weight/scale))
        else:
            from LTM1 import SYMBOL
            scale = max(1., max(entry.support.values()))
            for member, weight in entry.support.items():
                symbol = owner.concept_id_at_row(member)
                if symbol is not None:
                    alloc.parts.add((SYMBOL, symbol), (SYMBOL, cid), truth_plus=weight/scale)
            alloc.settle(cid)
        row = owner._csw_concept_row(alloc.order_of(cid), cid)
        if row is None:
            raise RuntimeError('candidate promotion exceeds the constructed inventory')
        pool.promote(key, cid)
        alloc.raised.add(cid)
        alloc.relate_idx['part-base', cid] = ('candidate', key[1])
        cover_members(owner, cid, entry.support)
        remember_candidate(owner, cid, entry.support,
            entry.refinement.get('literal_key', literal))
        if literal is not None:
            owner.add_concept_feature(row, literal[0], literal[1], 1.)
        result.append(cid)
    owner.interpret.admit_discovered()
    return result
