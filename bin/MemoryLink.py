"""One kept-close linking rule for reference, event and state occurrences."""
from dataclasses import dataclass, replace

import torch

from ConceptualMatch import match
from MemberRead import share_percept_reads


@dataclass(frozen=True)
class Occurrence:
    address: tuple
    concept: int | tuple | None = None
    referent: int | None = None
    kind: str = 'reference'
    order: int | None = None
    native: int | None = None
    pending: bool = False
    selected: object = Ellipsis
    retain: tuple = ()


def member_order(owner, site, *, stream=None):
    """Use the concept being read, not its containing clause's grammar order."""
    if site.order is not None:
        return site.order
    concept = site.concept
    if concept is None and site.address[:1] == ('bp',):
        return -1
    if isinstance(concept, tuple) and concept[:1] == ('bp',):
        return -1
    if concept is not None:
        if 'word' in owner._concept_allocator.symbol_types.get(concept, ()):
            concept = owner.interpret.referent(concept, batch=stream)
        return owner._concept_source_order(concept)
    return owner._concept_source_order(site.address)


def candidate_eligible(owner, identity, record, *, kind, order, stream, allow_promotion=True):
    """Apply the link's collection contract before a departure can select it."""
    if isinstance(identity, int) and record.get('token') is None:
        return owner._concept_source_order(identity) == order
    pool = owner.mtm
    token = pool.token_of(record.get('token') or identity)
    entry = pool.entries.get(token)
    if entry is None:
        if isinstance(identity, int):
            return owner._concept_source_order(identity) == order
        if isinstance(identity, tuple) and identity[:1] in (('ltm',), ('sym',)):
            return owner._concept_source_order(identity) == order
        return record.get('member_order', order) == order
    if not pool.in_window(token, stream) or entry.refinement.get('member_kind', 'reference') != kind:
        return False
    return (entry.refinement.get('member_order', order) == order or
        (allow_promotion and kind == 'reference' and entry.symbol is not None
         and owner._concept_source_order(entry.symbol) == order))


def chunk_candidates(owner, kind, order, stream):
    """Read unpooled retained members through STM0/LTM0's common reader."""
    from Memory import STMChunks as chunks
    stm = getattr(owner, 'stm', None)
    if stm is None or stream is None:
        return {}
    state, result = stm.state, {}
    if not 0 <= stream < state[0].shape[0]:
        raise ValueError('link stream exceeds the constructed STM batch')
    if kind == ('state' if order == 0 else 'reference'):
        wanted = chunks.OCCURRENCE
    elif kind == 'event':
        wanted = chunks.RELATION
    else:
        return result
    store = owner._closed_clause_store()
    # Non-reference collections can share members, so token_of deliberately
    # does not map those members to one owner. They are nevertheless pooled:
    # offering a bare member would judge a different value from the collection
    # append_path reopens at commit. The pool menu supplies their full values.
    pooled = {address for entry in owner.mtm.entries.values()
              if entry.refinement.get('member_kind', 'reference') == kind
              and entry.refinement.get('member_order') == order
              for address in entry.occurrences}
    refs = state[chunks.REFS][stream, :, 0]
    selected = (state[chunks.READ][stream] > 0) & (state[chunks.ORDERS][stream] == order) & (refs[:, 0] == wanted)
    indices = selected.nonzero().flatten()
    for index, member in torch.stack((indices, refs[indices, 1]), 1).tolist():
        address = ('ltm', store.address_domain, member) if wanted == chunks.OCCURRENCE else ('sym', member)
        if address in pooled or owner.mtm.token_of(address) is not None:
            continue
        result[address] = dict(value=owner.mtm.read_member(address, live=True),
            role=0, position=(index, 0), in_stm=True)
    return result


@share_percept_reads
def menu(owner, occurrence, candidates, *, value, where, when, stream, projected=None):
    pool, resolver = owner.mtm, owner.components.reference_close
    site = occurrence if isinstance(occurrence, Occurrence) else Occurrence(occurrence)
    order = member_order(owner, site, stream=stream)
    coordinates = (lambda point: point.detach()) if site.kind == 'percept' else resolver.coordinates
    query = coordinates(value)
    candidates = {**chunk_candidates(owner, site.kind, order, stream), **candidates}
    # The pool and recent STM chunks are the same menu, with no preferred store.
    for key, entry in pool.entries.items():
        if entry.order is not None or not pool.in_window(key, stream):
            continue
        if (entry.refinement.get('member_kind', 'reference') != site.kind or
                entry.refinement.get('member_order', order) != order):
            continue
        candidates.setdefault(key, dict(value=pool.value_of(key, live_candidate=True),
            token=key, role=0, position=(sum(entry.when - other.when > 0
                for other in pool.entries.values()), 0)))
    offered, by_width = [], {}
    for key, record in candidates.items():
        if not candidate_eligible(owner, key, record, kind=site.kind, order=order, stream=stream):
            continue
        if projected is not None and (site.kind, key) in projected:
            record = dict(record, value=projected[site.kind, key])
        coordinates(record['value'])  # Validate every offered carrier before filtering.
        content = record['value'].detach().to(query)
        by_width.setdefault(content.numel(), []).append(len(offered))
        offered.append((key, dict(record, content=content)))
    # Older LTM occurrences can have narrower carriers. Preserve each actual
    # reduction width and the original order in the policy's candidate menu.
    coverages = [None] * len(offered)
    for indices in by_width.values():
        values = torch.stack([offered[i][1]['content'] for i in indices])
        inside, contains = match(value.detach().to(query), values,
                                 codebook=None if site.kind == 'percept' else owner.similarity_codebook)
        for i, pair in zip(indices, torch.stack((inside, contains), -1).unbind(), strict=True):
            coverages[i] = pair
    passed = (torch.stack(coverages) == 1).any(-1).tolist() if offered else ()
    legal = {key: record for (key, record), keep in zip(offered, passed, strict=True) if keep}
    matched = [pair for pair, keep in zip(coverages, passed, strict=True) if keep]
    choices, features, logits = resolver.score_order(value, query, value, 0, legal,
        stream=stream, where=where, when=when, percept=site.kind == 'percept', member_order=order,
        matched=matched)
    return choices, features, logits


def coverage(own, covered):
    """Uncovered content and its uninformed baseline, with live reads."""
    width = min(own.shape[-1], covered.shape[-1])
    mass = own.sum()
    return mass - torch.minimum(own[..., :width], covered[..., :width]).sum(), mass


def preview_cost(resolver, plans, like):
    """The same link judge supplies the departure's untrained comparison."""
    rows = []
    for plan in plans:
        numerator, baseline = like.new_zeros(()), like.new_zeros(())
        if plan is not None:
            for item in (*plan['items'], *plan.get('arrivals', ())):
                own = item['content']
                selected = item['selected']
                record = item.get('record', plan['individuals'].get(selected))
                covered = own if (selected is None or item.get('own_object') or item.get('compose_dummy')
                    or item['mode'] == 'mint') else resolver.coordinates(
                        record['value'] if 'value' in record else record['content'])
                error, mass = coverage(own, covered)
                numerator, baseline = numerator+error, baseline+mass
            for own, covered in plan.get('link_coverage', ()):
                error, mass = coverage(own, covered)
                numerator, baseline = numerator+error, baseline+mass
        rows.append(numerator / torch.where(baseline > 0, baseline, 1.))
    return torch.stack(rows) if rows else like.new_zeros(0)


def link(owner, occurrence, candidates, *, value, where, when, stream):
    pool, resolver = owner.mtm, owner.components.reference_close
    site = occurrence if isinstance(occurrence, Occurrence) else Occurrence(occurrence)
    order = member_order(owner, site, stream=stream)
    if site.selected is Ellipsis:
        choices, features, logits = menu(owner, site, candidates, value=value,
            stream=stream, where=where, when=when)
        index = resolver.select(choices, logits)
        selected = None if index is None else choices[index]
    else:
        # Reference sites use the sentence trial's already-scored decision.
        selected = site.selected
    target = pool.token_of(selected) if selected is not None else None
    if target is not None and site.kind == 'reference':
        from FamiliarTier import resolve_known_reference
        resolve_known_reference(owner, target, order, stream=stream)
        entry = pool.entries[target]
        native = entry.symbol
        if native is None and not entry.occurrences:
            # An expired pending path has no member to promote. A definition
            # learned since that path opened still supplies a native referent.
            word = pool.reading_of(target)
            if word in owner._concept_allocator.named_identities:
                native = owner.interpret.referent(word, batch=stream)
        if (native is not None and entry.refinement.get('member_order', order) != order
                and owner._concept_source_order(native) == order):
            # After promotion a new mention reads the promoted concept, one
            # order above the original members. Keep that lower-order path
            # intact and re-enter the concept as a single native member.
            site = replace(site, native=native)
            target = None
    if target is None and site.native is not None:
        target = next((key for key, entry in pool.entries.items()
                       if entry.native == site.native and pool.in_window(key, stream)
                       and entry.refinement.get('member_kind', site.kind) == site.kind
                       and entry.refinement.get('member_order', order) == order), None)
    if target is None and site.native is None and isinstance(selected, tuple) and selected[:1] in (('ltm',), ('sym',)):
        # Choosing a retained STM member starts the path with that member,
        # followed by this close's occurrence. The trial menu did not mutate it.
        candidate = pool.read_member(selected)
        if site.kind == 'reference':
            native = resolver._derivation_symbol(selected)
            native = native if native in owner._concept_allocator.individuals else None
            target = pool.open(selected, kind=native or site.referent, native=native,
                when=when, value=candidate, pending=site.pending and native is None)
        else:
            target = pool.append_path(None, selected, kind=site.kind, when=when,
                where=where, stream=stream, order=order)
        pool.record_link(target, selected, stream, kind=site.kind, order=order)
    with torch.enable_grad():
        covered_before = None if target is None else pool.value_of(target, live_candidate=True)
    if target is not None:
        metadata = pool.entries[target].refinement
        if metadata.get('member_kind', site.kind) != site.kind or metadata.get('member_order', order) != order:
            raise ValueError('a link cannot mix collection kinds or orders')
        # Prepared reference decisions can be justified by an elided role's
        # witnessed description rather than the whole written clause. That
        # menu already checked containment; every other caller checks here too.
        if site.kind != 'reference':
            a, b = (match(value, covered_before) if site.kind == 'percept' else
                    match(value.detach().to(owner.similarity_codebook.W),
                          covered_before.detach().to(owner.similarity_codebook.W),
                          codebook=owner.similarity_codebook))
            from Meaning import require_sentence_tensor
            require_sentence_tensor((a == 1) | (b == 1), 'a link requires containment with its collection')
    if site.kind != 'reference':
        target = pool.append_path(target, site.address, kind=site.kind, when=when,
            where=where, stream=stream, order=order, concept=site.concept)
    elif target is None:
        target = pool.open(site.address, kind=site.referent, when=when, value=value,
            reading=site.concept, native=site.native, pending=site.pending,
            where=where, retain=site.retain)
    else:
        target = pool.bind(target, site.address, kind=site.referent, when=when,
            value=value, reading=site.concept, where=where)
    if site.concept is not None and site.kind != 'reference':
        entry = pool.entries[target]
        refinement = dict(entry.refinement)
        refinement['members'] = tuple(dict.fromkeys((*refinement.get('members', ()), (site.address, site.concept))))
        pool.entries[target] = replace(entry, refinement=refinement)
    pool.record_link(target, site.address, stream, kind=site.kind, order=order)
    pool.record_reading(target, stream, (1., 0.))
    model = getattr(owner, '_model', None)
    errors = getattr(model, '_sentence_close_errors', None)
    if errors is not None:
        with torch.enable_grad():
            actual = pool.value_of(target, live_candidate=True) if covered_before is None else covered_before
            # Content coordinates exclude form and location without detaching.
            derived = getattr(owner.similarity_codebook, 'mereology', None)
            start = 0 if derived is None or site.kind == 'percept' else derived.percept_event_width
            own, covered = value[..., start:], actual[..., start:]
            error, mass = coverage(own, covered)
            errors.error('memory.link_coverage', error, mass, row=stream if errors.row_mask is not None else None,
                category='reconstruction', objective='reconstruction', side='R')
    if site.kind == 'state':
        witness_participants(pool, target, site.address, stream, when)
    return target


def witness_participants(pool, collection, address, stream, when):
    """Consecutive links of one state path also witness its stable participants."""
    from dataclasses import replace
    entry = pool.entries[collection]
    clock = (stream, pool.closes.get(stream, 0)+1)
    current = {key for key, token in pool.entries.items()
               if key != collection and token.refinement.get('member_kind', 'reference') == 'reference'
               and clock in token.refinement.get('event_links', {}).values()}
    previous = set(entry.refinement.get('participants', ()))
    for key in current | previous:
        if key not in pool.entries:
            continue
        token = pool.entries[key]
        refinement = dict(token.refinement)
        paths = dict(refinement.get('chain_paths', {}))
        paths[collection] = (*paths.get(collection, ()), address) if key in current else ()
        refinement['chain_paths'] = paths
        refinement['chain_mentions'] = tuple(dict.fromkeys(member for path in paths.values() for member in path))
        pool.entries[key] = replace(token, refinement=refinement)
        if key in current:
            pool.record_link(key, address, stream)
    entry = pool.entries[collection]
    pool.entries[collection] = replace(entry, refinement=dict(entry.refinement, participants=tuple(current)))


def state_eligible(clause):
    from ThoughtReferences import has_open_slots
    return (clause is not None and clause.point is not None and clause.order == 0
        and not has_open_slots(clause.meaning)
        and not dict(clause.meaning.bindings).get('_modal')
        and not (clause.identity or {}).get('alternative', False)
        and not dict(clause.meaning.bindings).get('_question'))


def preview(resolver, plan, clause, reading, *, stream, when):
    """The ordinary sentence departure includes every proposed link menu."""
    from ThoughtReferences import has_open_slots
    owner = resolver.components.owner
    proposed = []
    state = clause
    if clause is not None:
        _, arrivals = resolver._arrival_constituents(clause)
        state, _ = resolver._prepare(clause, plan, arrivals, preview=True)
    if state_eligible(state):
        # Grammar order describes the operation's result. A retained state
        # can carry a higher-order referent; use the occurrence writer's
        # order for both its trial menu and the eventual collection.
        from PartClosing import PartClosing
        order = PartClosing(owner, resolver.store).occurrence_order(state)
        if order < 0:
            raise ValueError('a state link requires a witnessed occurrence order')
        proposed.append((('state', 0), state.point, state.where, order))
    pending = [] if state is None else [state]
    from PartClosing import PartClosing
    closing = PartClosing(owner, resolver.store)
    while pending:
        node = pending.pop()
        record = node.identity
        if record is not None and record.get('event') is not None and not record.get('alternative'):
            proposed.append((('event', record['action']), record['result'], node.where,
                             closing.event_order(node, context=state)))
        pending.extend((*node.children, *node.companions))
    if reading is not None and reading.word_ids is not None:
        for index, word in enumerate(reading.word_ids.tolist()):
            if word in owner.basic_percepts.words:
                from torch.nn.functional import pad
                point = owner.basic_percepts.code(owner.basic_percepts.words[word])
                point = pad(point, (0, owner.similarity_codebook.W.shape[-1]-point.numel()))
                proposed.append((('percept', index), point,
                    None if reading.symbol_where is None else reading.symbol_where[index], -1))
    plan['link_choices'] = {}
    plan['link_coverage'] = []
    projected = {}
    for key, value, where, order in proposed:
        if value is None: continue
        choices, features, logits = menu(owner, Occurrence((), kind=key[0], order=order), {},
            value=value, where=where, when=when, stream=stream, projected=projected)
        index, decision = resolver.choose(plan, choices, features, logits)
        decision['link'] = key
        selected = None if index is None else choices[index]
        plan['link_choices'][key] = selected
        coordinates = (lambda point: point) if key[0] == 'percept' else resolver.coordinates
        projected_key = (key[0], selected)
        covered = (value if selected is None else projected[projected_key] if projected_key in projected else
            owner.mtm.value_of(selected) if owner.mtm.token_of(selected) is not None else
            owner.mtm.read_member(selected))
        plan['link_coverage'].append((coordinates(value), coordinates(covered)))
        if selected is not None:
            # These collections read the maximum of their members. Later
            # choices in this close must see the earlier prepared additions,
            # without writing those additions before the path is kept.
            projected[projected_key] = torch.maximum(covered, value)
