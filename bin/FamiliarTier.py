"""Sentence-open familiarity and kept-close promotion in the shared MTM pool."""
from dataclasses import replace
from functools import wraps

import torch
from Memory import NULL_CONCEPT


def kept_close(method):
    """Scope promotion to active streams and defer the kept word views."""
    @wraps(method)
    def boundary(model, state, sid, active, *args, **kwargs):
        owner = model._concept_owner()
        streams = getattr(owner, '_mtm_active_streams', None)
        owner._mtm_active_streams = set(active.nonzero().flatten().tolist())
        try:
            from ConceptTrust import defer_word_views
            with defer_word_views(owner):
                result = method(model, state, sid, active, *args, **kwargs)
                from ScalarPercepts import commit
                commit(model, sid, active)
                return result
        finally:
            owner._mtm_active_streams = streams
    return boundary


def is_place(owner, row):
    """Whether the concept at ``row`` is a place's (it reads a row of the
    first Pi with entries on the where wires)."""
    from Spaces import _concept_alloc_of
    cid = owner.concept_id_at_row(int(row))
    return cid is not None and 'place' in _concept_alloc_of(owner).symbol_types.get(cid, ())


def support_order(owner, support, candidate_order=None):
    """The highest member order, whose next order the candidate can become."""
    if candidate_order == 0:
        return 0  # A percept's area contains only perceptual members.
    orders = {owner._row_order(row) for row in support}
    return max(orders, default=0)


def centroid(owner, support, *, order=None):
    from FormCentroids import convex_mean
    order = support_order(owner, support) if order is None else order
    rows = [row for row in support if owner._row_order(row) == order]
    if not rows:
        return owner.similarity_codebook.W.new_zeros(owner.similarity_codebook.W.shape[-1])
    codes = owner.similarity_codebook.lookup_rows(rows)
    weights = codes.new_tensor([support[row] for row in rows])
    return convex_mean(weights, codes)[0]


def shared_part(owner, value, other):
    """The lesser of the two coverages of ``value`` and ``other`` by shared
    size (``ConceptualMatch.match``, read with the owner's codebook): how
    much each holds of the other, so a likeness of direction that differs in
    size is not the same collection."""
    from ConceptualMatch import match
    covered, covers = match(value, other.to(value), codebook=owner.similarity_codebook)
    return min(float(covered), float(covers))


def _clock(owner, batch):
    model = owner._model
    from Occurrence import sentence_tick
    return sentence_tick(model, model._open_sentence_slot)


def witnessed_support(owner, cid):
    """Read sparse durable witnesses; MTM can be empty at an epoch boundary."""
    alloc = owner._concept_allocator
    records = alloc.store_of(cid).constituents(cid)
    saved = [ref for role, ref in records if role == 'candidate_support']
    if saved:
        return {row: weight for symbol, weight in saved
                if (row := owner._concept_row_of(symbol)) is not None}
    members = class_members(owner, cid)
    for role, ref in records:
        if role == 'part' and isinstance(ref, tuple) and ref[:1] == ('field',):
            row = owner._concept_row_of(ref[1])
            if row is not None:
                members[row] = max(members.get(row, 0.), ref[2])
    return members


@torch.no_grad()
def witnessed_match(owner, order, support, *, literal_key=None):
    """One bounded vector comparison at a time, never candidates by inventory."""
    alloc = owner._concept_allocator
    source_order = support_order(owner, support, order)
    value = centroid(owner, support, order=source_order)
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
        # The class's chart position is learned from its context, not the
        # centroid of its members. Familiarity compares the two witnesses.
        code = centroid(owner, members, order=source_order)
        mass, previous_mass = sum(support.values()), sum(members.values())
        same = all(support[row] / mass == members[row] / previous_mass for row in support)
        if same or shared_part(owner, value, code) >= owner.concept_match_part:
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


def _ground_candidate_members(owner, cid, support):
    """A current field witnesses its actual members in this epoch's LTM1."""
    from Memory import SYMBOL
    bank = owner._concept_allocator.parts
    scale = max(1., max(support.values()))
    for row, weight in support.items():
        symbol = owner.concept_id_at_row(row)
        if symbol is not None and symbol != cid and bank.find(symbol, cid) is None:
            bank.add((SYMBOL, symbol), (SYMBOL, cid), truth_plus=weight/scale)


def use_witnessed(owner, cid, support, batch):
    """A durable match is an access; new members extend coverage by maximum."""
    store = owner._concept_allocator.store_of(cid)
    accesses = next((value for role, value in store.constituents(cid) if role == 'familiar_access'), 0)
    store.set_role_constituents(cid, 'familiar_access', [accesses + 1])
    previous = witnessed_support(owner, cid)
    if (cid in owner._concept_allocator.raised and owner._concept_source_order(cid) > 0
            and owner._concept_row_of(cid) not in support
            and any(role == 'candidate_support' for role, _ in store.constituents(cid))):
        _ground_candidate_members(owner, cid, support)
    merged = {row: max(weight, previous.get(row, 0.)) for row, weight in support.items()}
    merged.update({row: weight for row, weight in previous.items() if row not in merged})
    if merged != previous and owner._concept_row_of(cid) not in merged:
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
    source_order = support_order(owner, support, order)
    value = centroid(owner, support, order=source_order)
    key = pool.match_candidate(order, value, value_order=source_order, support=support)
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
    owner.refine_over_collected()
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
                   for symbol, weight in (value,) if (row := owner._concept_row_of(symbol)) is not None}
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
        # A place's concept is read at one wherever its place is, so its
        # co-occurrence tells nothing of what stands there: it is neither a
        # focal row nor a context of a conjunction by co-occurrence.
        offered = {row: weight for row, weight in used.items() if not is_place(owner, row)}
        for focal in offered:
            order = owner._row_order(focal)
            context = {row: weight for row, weight in offered.items() if row != focal}
            if not context:
                continue
            context_order = support_order(owner, context, order)
            value = centroid(owner, context, order=context_order)
            for alternative, previous in contexts:
                if (alternative in used or owner._row_order(alternative) != order
                        or support_order(owner, previous, order) != context_order):
                    continue
                if shared_part(owner, value, centroid(owner, previous, order=context_order)) >= owner.concept_match_part:
                    dense = field.new_zeros(owner.nVectors)
                    dense[list(context)] = dense.new_tensor(list(context.values()))
                    owner.maybe_raise_order(focal, alternative, dense, batch=batch, evidence=None)
                    break
            observe_candidate(owner, order, context, batch=batch,
                              pair=tuple(float(x) for x in paired[focal]), focal=focal)
        observe_native_cases(owner, batch, used)
        # MTM's leading entries are read through their live member codes;
        # their share of this field's size is use, not an invented positive
        # witness.
        if used:
            for key, entry in tuple(owner.mtm.entries.items()):
                if entry.order is None:
                    continue
                order = support_order(owner, entry.support, entry.order)
                current = centroid(owner, used, order=order)
                if shared_part(owner, current, owner.mtm.value_of(key)) > owner.concept_use_floor:
                    owner.__dict__.setdefault('_field_activated_references', {}).setdefault(batch, set()).add(key)
    return activated


@torch.no_grad()
def observe_intersection(owner, lanes, *, batch=0):
    """Recurrent concept lanes, proposed together at one reading."""
    lanes = tuple(sorted(set((int(row), int(pole)) for row, pole in lanes)))
    if len(lanes) < 2:
        return None
    if any(pole not in (0, 1) or owner._row_order(row) != 0 for row, pole in lanes):
        raise ValueError('an intersection requires order-zero concept lanes')
    return observe_candidate(owner, 0, {row: 1. for row, _ in lanes}, batch=batch,
                             literal=('field', lanes), literal_key=('field', lanes))


@torch.no_grad()
def observe_native_cases(owner, batch, support):
    support = {row: weight for row, weight in support.items() if owner._row_order(row) == 0}
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


def observed_row_lists(rows):
    """Export each stream's observed addresses for the corpus report."""
    return {b: [seat for seat in stream if seat >= 0] for b, stream in enumerate(rows.tolist())}


def open_sentence(model, sid, present, *, device_rows=False):
    """Every present word bracket, independent of the grammar's attention."""
    owner, isp = model._concept_owner(), model.inputSpace
    owner._field_activated_references = {}
    if not hasattr(model, '_attention_spans'):
        return None if device_rows else {}  # Metadata-only callers have no native percept field.
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
    bank = owner.cs_lookup((native.get('native_indices', native['indices']),
        native.get('native_part_spans', native['part_spans']),
        ws.subspace.what.primitive_properties, raw, whole_spans), brackets)
    # The sentence's one lookup, held for its looks.
    model.__dict__.setdefault('_sentence_banks', {})[int(sid)] = bank
    model._sentence_bank = bank
    field = owner.cs_read_bank(bank)
    from SentenceField import open_field
    owner._cs_last_a0 = field.detach()
    acts = owner.cs_forward_content(field)
    if getattr(owner, '_mereology_raise', False) and owner._percept_field is not None:
        owner._refinement_field = owner.refinement_field(owner._percept_field)
    opened = open_field(model, sid, acts)
    model._definition_fold_field = acts.detach()
    from ConceptEvidence import symbols
    readings = symbols(acts)
    addresses = owner._field_inventory_rows(acts.device)
    rows = addresses[:, None] if addresses.ndim == 1 else addresses
    seen = torch.where(present[None] & (readings[..., 0] > 0), rows, -1).T
    if model.training:
        owner.observe_open_field(acts, active_streams=present)
    model._last_open_attention = dict(sentence=int(sid), brackets=brackets.detach().clone(),
        field_rows=addresses.detach().clone(), seen=seen)
    return seen if device_rows else observed_row_lists(seen)


def member_row(owner, kind, identity):
    from Memory import SYMBOL, OCCURRENCE
    if kind == SYMBOL:
        return owner._concept_row_of(identity)
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
        return None if symbol is None else owner._concept_row_of(symbol)
    row = store.index_of_row(identity)
    if row is None:
        return None
    reference = store.referent_of(row)
    return owner._concept_row_of(reference[1]) if reference is not None and reference[0] == 'sym' else None


def class_members(owner, cid):
    from Memory import SYMBOL
    if cid == NULL_CONCEPT:
        return {}
    parts = owner._concept_allocator.parts
    members = {}
    for kind, identity in parts.parts((SYMBOL, cid)):
        if not owner._concept_allocator.class_member((kind, identity), (SYMBOL, cid)):
            continue
        row = member_row(owner, kind, identity)
        edge = parts.find((kind, identity), (SYMBOL, cid))
        weight = max(0., float(parts.truth_plus[edge]-parts.truth_minus[edge]))
        if row is not None and weight:
            members[row] = max(weight, members.get(row, 0.))
    return members


@torch.no_grad()
def cover_members(owner, cid, members, *, conjunctive=False):
    """The same definition fold for every raise; location is learned from context."""
    from MemoryConsolidation import _fold
    if cid == NULL_CONCEPT:
        return
    members = {int(row): float(weight) for row, weight in members.items() if weight > 0}
    alloc = owner._concept_allocator
    alloc.symbol_types.setdefault(cid, set()).add('concept')
    order = alloc.order_of(cid)
    destination = owner._csw_concept_row(order, cid)
    if destination is None:
        raise RuntimeError('class coverage exceeds the constructed inventory')
    if not conjunctive:
        owner._populate_concept_weights(cid, witness=(members, ()))
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    owner._priming_edge_revision = None
    value = (owner.similarity_codebook.lookup_rows(list(members)).amax(0) if conjunctive
             else _fold(owner, None, list(members), members=members))
    if value is not None:
        value = value.clone()
        # A whole-form chart's order-zero row is perceptual: its first
        # location is the support-weighted area, without conceptual members.
        # A conceptual row already has its seed; the fold is its extension.
        # An existing position is never reset here.
        if derived is None and order == 0 and not bool(owner.similarity_codebook.trust.known[destination]):
            percept = centroid(owner, members, order=0)
            if bool(percept.norm() > 0):
                owner.similarity_codebook.W[destination].copy_(percept)
        if derived is not None:
            derived._form_valid[destination] = False
        owner.similarity_codebook.trust.remember(cid, value)
    alloc.layer().witnessed[destination] = True
    return destination


@torch.no_grad()
def commit_read_words(owner, reading, *, batch, document, position, clause=None, written=None, link_choices=None):
    """Retain identified word mentions once, at the kept reading's close.

    A grammar's participant selection does not erase a word that was read.
    The observed lexical order supplies this occurrence's witness; promotion
    still belongs to the common update after every active stream closes.
    """
    if reading is None or reading.word_ids is None:
        return
    pool, store = owner.mtm, owner._closed_clause_store()
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
        basic = owner.basic_percepts
        point = None
        if word > 0 and word in basic.words:
            from torch.nn.functional import pad
            point = basic.code(basic.words[word])
            point = pad(point, (0, owner.similarity_codebook.W.shape[-1]-point.numel()))
        if word > 0:
            # A kept lexical occurrence contributes independently of whether
            # its interpretation already has an object.
            from ConceptTrust import observe
            # The leaf already denotes its object. A lexical witness is its
            # word's basic percept, without a placed meaning or radius.
            observe(owner, word, point)
        if word > 0 and word in owner.basic_percepts.words:
            from MemoryLink import Occurrence
            percept = basic.words[word]
            root = (written or {}).get('root_occurrence')
            location = retained.get(index)
            address = store.occurrence_of(location) if location is not None else root
            if address is None:
                raise ValueError('a kept basic percept requires its sentence occurrence')
            selected = Ellipsis if link_choices is None else link_choices.get(('percept', index), Ellipsis)
            owner.link(Occurrence(address, concept=('bp', percept), kind='percept', order=-1, selected=selected), {},
                value=point, when=pool.closes.get(batch, 0)+1, stream=batch,
                where=None if reading.symbol_where is None else reading.symbol_where[index])


def resolve_known_reference(owner, key, order, *, stream=None):
    """Finish a pending name's old path when a kept use reads its definition."""
    pool = owner.mtm
    entry = pool.entries[key]
    if not entry.pending or entry.symbol is not None or not entry.occurrences:
        return
    word = pool.reading_of(key)
    if word is None:
        word = pool.reading_of(entry.occurrences[0])
    if word not in owner._concept_allocator.named_identities:
        return
    if (order == entry.refinement.get('member_order', order) + 1
            and owner._concept_source_order(owner.interpret.referent(word, batch=stream)) == order):
        promote_identity(owner, key, word=word, stream=stream)


@torch.no_grad()
def promote_identity(owner, key, *, word=None, trust=None, evidence=None, stream=None):
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
    member_order = entry.refinement.get('member_order',
        -1 if index is None else owner._concept_source_order(origin))
    reading = pool.reading_of(origin)
    if 'word' in alloc.symbol_types.get(reading, ()):
        denotation = owner.interpret.referent(reading, batch=stream)
        if owner._concept_source_order(denotation) == member_order:
            reading = denotation
    constituent = (('sym', reading) if reading is not None else
        ('rel' if isinstance(index, tuple) else 'occ', origin[-1]))
    if member_order < 0:
        record = None if index is None else store.raw_meaning_of(index)
        raise RuntimeError(f'identity promotion has no witnessed order: occurrence={origin!r}, record={record!r}')
    # A lexical alias introduced by a bind is an observation of the token.
    # Naming belongs to the token's introducing word or the kept definition.
    introducing_word = pool.reading_of(key) if entry.kind != NULL_CONCEPT else None
    if 'word' not in alloc.symbol_types.get(introducing_word, ()):
        introducing_word = None
    words = {name for name in (introducing_word, word) if name is not None}
    if entry.pending:
        remembered = {cid for name in words for cid in owner.interpret.remembered_objects(name)
                      if cid in alloc.individuals and alloc.order_of(cid) == member_order}
        if len(remembered) > 1:
            raise ValueError(f'ambiguous pending identity re-entry: {sorted(remembered)}')
        if remembered:
            # A weak definition may open a pending name after an epoch, but
            # its remembered individual is already the member being read.
            # Re-enter it instead of minting and naming a higher-order class.
            cid, = remembered
            pool.promote(key, cid)
            for name in words:
                owner.interpret.define(name, cid, trust=trust, evidence=evidence)
            return cid
    # A name of an existing member is not automatically the name of the
    # higher-order collection formed from repeated uses of that member.
    words = {name for name in words if not any(alloc.order_of(cid) == member_order
        for cid in (*owner.interpret.denotations(name), *owner.interpret.remembered_objects(name)))}
    known = set()
    for name in words:
        active = owner.interpret.denotations(name)
        candidates = (*active, *owner.interpret.remembered_objects(name))
        known.update(cid for cid in candidates if cid not in alloc.retired
                     and alloc.order_of(cid) == member_order + 1)
    if len(known) > 1:
        forms = [owner.interpret.lexical_form(name) for name in sorted(words)]
        raise ValueError(f'ambiguous identity promotion for words {sorted(words)} {forms}: denotations {sorted(known)}')
    if known:
        cid, = known
        if cid in alloc.individuals:
            alloc.store_of(cid).add_constituent(cid, 'part', constituent)
        pool.promote(key, cid)
        for name in words:
            owner.interpret.define(name, cid, trust=trust, evidence=evidence)
        return cid
    # Check for an existing fit first: a concept of this order that reads
    # the constituent's symbol and nothing else.
    fit = (owner.concept_fit((constituent[1],), member_order + 1)
           if constituent[0] == 'sym' else None)
    if fit is not None:
        pool.promote(key, fit)
        for name in words:
            owner.interpret.define(name, fit, trust=trust, evidence=evidence)
        return fit
    from ConceptPlacement import promotion_seat, identity_start
    if not promotion_seat(owner, member_order + 1, context='MTM identity promotion'):
        return None
    owner._preflight_concept_allocation(1, context='MTM identity promotion')
    cid = alloc.new_concept()
    # An identity's retained occurrence supplies its order. No grammar choice
    # or caller order is supplied to this operation.
    from Memory import OCCURRENCE
    alloc.store_of(cid).add_constituent(cid, 'part', constituent)
    alloc.settle(cid)
    row = owner._csw_concept_row(alloc.order_of(cid), cid)
    if row is None:
        raise RuntimeError('identity promotion exceeds the constructed inventory')
    alloc.individuals.add(cid)
    alloc.symbol_types.setdefault(cid, set()).update(('concept', 'object'))
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    current = pool.value_of(key)
    # The actual event fold is retained as extension evidence on a whole-
    # value chart. A derived chart's fold contains only form, so its concept
    # extension is the MTM reading instead. Neither becomes a position.
    from MemoryConsolidation import _fold
    actual = dict.fromkeys(event.occurrence for event in entry.events if event.role == 0)
    locations = (store.location(address) for address in actual)
    members = [index for index in locations if isinstance(index, int)]
    folded = _fold(owner, store, members) if derived is None else None
    extension = current if folded is None else folded
    identity_start(owner, cid, words)
    owner.similarity_codebook.trust.remember(cid, extension)
    from ConceptTrust import observe
    for address in entry.occurrences:
        location = store.location(address)
        if isinstance(location, int):
            observe(owner, cid)
    if derived is not None:
        derived.remember_content(row, current)
    pool.promote(key, cid)
    # The nomination makes the symbol above the concept's order, which
    # brings this concept.
    owner.nominate_symbol(key, cid)
    for name in words:
        owner.interpret.define(name, cid, trust=trust, evidence=evidence)
    owner._priming_edge_revision = None
    return cid


@torch.no_grad()
def define_pending(owner, left, right, *, word=None, other_word=None, null_referent=False):
    """Consume a kept equality/IS witness once its subject token is retained."""
    from Memory import SYMBOL, OCCURRENCE
    from ConceptIndex import unbound_label
    parts = owner._concept_allocator.parts
    relation = parts.find(left, right)
    if relation is None:
        raise ValueError('definition requires its kept relation')
    store = owner._closed_clause_store()
    reference = left[1] if left[0] == SYMBOL else (
        ('ltm', store.address_domain, left[1]) if left[0] == OCCURRENCE and store is not None else None)
    key = owner.mtm.token_of(reference)
    if null_referent:
        if word is not None or key is None or owner.mtm.entries[key].kind != NULL_CONCEPT:
            raise ValueError(f'null-kind definition has no open token at {left!r}')
        return _resolve_null_token(owner, key, left, right, other_word)
    if key is not None and owner.mtm.entries[key].kind == NULL_CONCEPT:
        return _resolve_null_token(owner, key, left, right, other_word)
    if key is None or not owner.mtm.entries[key].pending or owner.mtm.entries[key].order is not None:
        return False
    # An unresolved or order-incompatible shared side is still a premise,
    # not a defining class witness. Use the origin's witnessed order before
    # allocating; index_part_row must never be asked to invent compatibility.
    alloc = owner._concept_allocator
    entry = owner.mtm.entries[key]
    if not entry.occurrences:
        return False  # The kept reading will retain its mentions before finish_definitions.
    member_order = owner._concept_source_order(entry.occurrences[0]) + 1
    shared_order = owner._concept_source_order(right)
    existing_class = right[0] == SYMBOL and ('part-base', right[1]) in alloc.relate_idx
    if not (unbound_label(owner, right) or member_order == shared_order
            or existing_class and member_order == shared_order-1):
        return False
    cid = promote_identity(owner, key, word=word, evidence=(float(parts.truth_plus[relation]), float(parts.truth_minus[relation])))
    if cid is None:
        return False
    if left != right:
        from PartClosing import PartClosing
        PartClosing(owner, store).observe_members((SYMBOL, cid), right, (left, right), (word, None))
    return True


def _resolve_null_token(owner, key, left, right, other_word):
    """Resolve an occurrence from evidence, without naming its pointing word."""
    from ConceptIndex import unbound_label
    from Memory import OCCURRENCE, SYMBOL
    from ChainRead import envelope
    pool, alloc, store = owner.mtm, owner._concept_allocator, owner._closed_clause_store()
    entry = pool.entries[key]
    if left[0] != OCCURRENCE or envelope(store,left) not in entry.occurrences:
        raise ValueError(f'perceptual token has no kept occurrence at {left!r}')
    order = owner._concept_source_order(left) + 1
    label = right[1] if right[0] == SYMBOL and 'word' in alloc.symbol_types.get(right[1], ()) else other_word
    target = right[1] if right[0] == SYMBOL and right[1] != label else None
    if label not in (None,-1,0):
        denoted = owner.interpret.denotations(label)
        if len(denoted) == 1:
            target = denoted[0]
    if target is None and right[0] == OCCURRENCE:
        other = pool.entries.get(pool.token_of(envelope(store,right)))
        if other is not None:
            target = other.symbol or other.native
            if label in (None,-1,0):
                label = pool.reading_of(envelope(store,right))
    if label not in (None,-1,0) and not owner.interpret.denotations(label):
        remembered = tuple(obj for obj in owner.interpret.remembered_objects(label)
                           if alloc.order_of(obj) == order)
        if len(remembered) == 1 and target in (None, remembered[0]):
            # A sign can strengthen a remembered kind before that kind is a
            # denotation. Keep the occurrence pending; do not mint or add rows.
            target = remembered[0]
            refinement = dict(entry.refinement)
            views = dict(refinement.get('sign_views', {}))
            seen = tuple(views.get(target, ()))
            from ConceptTrust import observe
            for address in entry.occurrences:
                if address not in seen:
                    observe(owner, target)
                    seen += (address,)
            views[target] = seen
            refinement['sign_views'] = views
            refinement['pending_sign_definition'] = (left, right, label)
            entry = replace(entry, pending=True, refinement=refinement)
            pool.entries[key] = entry
            if owner.interpret.denotations(label) != (target,):
                return True
    if target is None and label not in (None,-1,0) and unbound_label(owner,(SYMBOL,label)):
        row = alloc.parts.find(left, right)
        target = promote_identity(owner,key,word=label,evidence=(float(alloc.parts.truth_plus[row]), float(alloc.parts.truth_minus[row])))
    elif (target is not None and alloc.order_of(target) == order and
          (target in alloc.individuals or 'object' in alloc.symbol_types.get(target, ()))):
        pool.promote(key,target)
    elif target is not None and alloc.order_of(target) >= 0:
        pool.entries[key] = replace(entry,kind=target)
        return True
    else:
        return False
    if target is None:
        return False
    refinement = dict(pool.entries[key].refinement)
    refinement.pop('pending_sign_definition', None)
    pool.entries[key] = replace(pool.entries[key],kind=target,refinement=refinement)
    row = alloc.parts.find(left,right)
    for address in entry.occurrences:
        member = (OCCURRENCE, address[-1])
        if owner._concept_source_order(address) + 1 != alloc.order_of(target):
            raise ValueError(f'witnessed member {address!r} disagrees with the order of {target}')
        if (member, (SYMBOL,target)) != (left,right):
            from MemoryPaths import membership
            membership(owner, address, target, evidence=(float(alloc.parts.truth_plus[row]),
                float(alloc.parts.truth_minus[row])))
    alloc.settle(target)
    return True


def _finish_pending_signs(owner):
    joined = []
    for key, entry in tuple(owner.mtm.entries.items()):
        pending = entry.refinement.get('pending_sign_definition')
        if entry.kind != NULL_CONCEPT or not entry.pending or pending is None:
            continue
        left, right, label = pending
        denoted = owner.interpret.denotations(label)
        if len(denoted) == 1:
            _resolve_null_token(owner, key, left, right, label)
            joined.append(denoted[0])
    return joined


def finish_definitions(owner):
    pending = owner.__dict__.pop('_kept_definition_witnesses', ())
    # Shared views can reach one kept relation through more than one site.
    # Resolve that witness once: its first use may already promote the token.
    for left, right, word, other_word, null_referent in dict.fromkeys(pending):
        define_pending(owner, left, right, word=word, other_word=other_word, null_referent=null_referent)
    _finish_pending_signs(owner)


@torch.no_grad()
def promote_candidates(owner):
    pool, alloc = owner.mtm, owner._concept_allocator
    if getattr(owner, '_online_learning_frozen', False):
        return []
    result = _finish_pending_signs(owner)
    for key, entry in tuple(pool.entries.items()):
        literal = entry.refinement.get('literal')
        if 'defining_word' in entry.refinement:
            continue  # The named percept-collection writer counts these views.
        if entry.kind == NULL_CONCEPT:
            continue
        if entry.symbol is not None or 'promoted_percept' in entry.refinement:
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
            count = (len({(stream, clock) for address, (stream, clock) in mentions.items()
                          if (address in entry.occurrences
                              or address in entry.refinement.get('chain_mentions', ()))
                          and pool.closes.get(stream, 0)-clock <= pool.horizon})
                     if pool.closes else len(entry.occurrences))
        else:
            count = len(readings) if readings else entry.count
        if count < owner.concept_mint_count:
            continue
        if entry.refinement.get('member_kind') == 'percept':
            rows = {concept[1] for address, concept in entry.refinement['members']}
            basic = owner.basic_percepts
            parts = tuple(dict.fromkeys(part for row in sorted(rows) for part in basic.occurrences[row][0]))
            wholes = set.intersection(*(set(basic.occurrences[row][1]) for row in rows))
            percept = basic.reserve(parts, tuple(sorted(wholes)))
            owner.basic_percepts.promote(percept)
            pool._release_id(entry)
            pool.entries[key] = replace(entry, transient=None,
                refinement=dict(entry.refinement, promoted_percept=percept))
            continue
        if entry.order is None:
            identity = promote_identity(owner, key)
            if identity is not None:
                result.append(identity)
            continue
        lanes = tuple(max((item[pole] for item in readings), default=0.) for pole in (2, 3))
        if not owner._promotion_enabled or min(lanes) > 0:
            continue
        support = {row: weight for row, weight in entry.support.items() if weight >= owner.concept_part_floor}
        if not support:
            continue
        entry = replace(entry, support=support)
        pool.entries[key] = entry
        if entry.order > 0:
            orders = {symbol: alloc.order_of(symbol) for row in support
                      if (symbol := owner.concept_id_at_row(row)) is not None}
            if not orders or min(orders.values()) < 0:
                raise ValueError(f'MTM candidate {key!r} has unwitnessed member orders {orders!r}')
            order = 1 + max(orders.values())
            if order != entry.order:
                # The open field is addressed by its focal row. A class from
                # this same support can sit one order above its highest member.
                existing = witnessed_match(owner, order, support)
                if existing is not None:
                    use_witnessed(owner, existing, support, next(iter(pool.accesses.get(key, {})), 0))
                    pool.promote(key, existing)
                    result.append(existing)
                    continue
        else:
            order = 0
        if order > 0:
            # Check for an existing fit first: a concept of this order whose
            # entries read the same symbols as the candidate's members.
            members = [cid for row in support if (cid := owner.concept_id_at_row(row)) is not None]
            fit = owner.concept_fit(members, order) if len(members) == len(support) else None
            if fit is not None:
                pool.promote(key, fit)
                result.append(fit)
                continue
        from ConceptPlacement import promotion_seat
        if not promotion_seat(owner, order, context='MTM candidate promotion'):
            continue
        owner._preflight_concept_allocation(1, context='MTM candidate promotion')
        cid = alloc.new_concept()
        literal = entry.refinement.get('literal')
        if entry.order == 0:
            # A literal conjunction stays in the perceptual order. Its field
            # support is initialization evidence, not a higher-order member.
            alloc.identity[cid] = (None, None)
            if literal is not None and literal[0] != 'field':
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
            _ground_candidate_members(owner, cid, entry.support)
        # Keep the order the actual parts derive when their historical
        # grounding is withdrawn at the next epoch boundary.
        derived = alloc.settle(cid)
        if derived != order:
            raise ValueError(f'MTM candidate {key!r} derives order {derived}, expected {order}')
        row = owner._csw_concept_row(derived, cid)
        if row is None:
            raise RuntimeError('candidate promotion exceeds the constructed inventory')
        pool.promote(key, cid)
        if derived > 0:
            # Above order zero the nomination makes a symbol, which brings
            # this concept; at order zero it is the basic percept's row.
            owner.nominate_symbol(key, cid)
        alloc.raised.add(cid)
        alloc.relate_idx['part-base', cid] = ('candidate', key[1])
        cover_members(owner, cid, entry.support, conjunctive=literal is not None and literal[0] == 'field')
        remember_candidate(owner, cid, entry.support,
            entry.refinement.get('literal_key', literal))
        if literal is not None:
            if literal[0] == 'field':
                # A conjunction of order-zero concepts is a row of the first
                # Pi over their basic percepts. A member read on its against
                # lane, or a member with several percepts, is not one row's
                # member at order zero; no row is written for it.
                percepts = [owner.concept_percepts(member) for member, _ in literal[1]]
                if all(not pole for _, pole in literal[1]) and all(len(p) == 1 for p in percepts):
                    percept = owner.basic_percepts.reserve_conjunction([p for p, in percepts])
                    owner.basic_percepts.promote(percept)
                    # Its members now have a holder: their stored against
                    # lanes are written again.
                    owner.basic_percepts.follow(p for p, in percepts)
                    owner.bring_concept(percept, row)
            elif literal[0] == 'ps':
                # The existing field-case curriculum witnesses an entire
                # native run. Its parts/wholes become one basic percept at
                # recurrence; no raw part columns enter its concept row.
                raw = entry.refinement['literal_key']
                wholes = owner._model.wholeSpace.property_rows_for_bytes(raw)
                percept = owner.basic_percepts.reserve(literal[1], wholes)
                owner.basic_percepts.promote(percept)
                owner.bring_concept(percept, row)
            else:
                owner.add_concept_feature(row, literal[0], literal[1], 1.)
        result.append(cid)
    return result
