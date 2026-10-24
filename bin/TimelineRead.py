"""Bounded event reads from a grammatical subject, predicate and order.

Timeline membership lasts with MTM. Surviving ideas can still be read through
their retained Part arguments; they do not reconstruct expired timelines.
"""
from dataclasses import replace

from ThoughtReferences import bindings


def request(clause, written, store):
    record = clause.identity
    if (not record or not (record.get('timeline_order') or record.get('case_value') is not None)
            or record.get('operation') != 'lift' or not record.get('event')):
        return None
    subject, verb = record['event'][1], record['event'][3]
    if not subject:
        return None
    owner = store._concept_owner()
    resolver = owner.components.reference_close
    reference = None
    pending, seen = [clause], set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if (node.reference_request is not None and node.identity is not None
                and node.identity['handle'] == subject):
            reference = written.get(('occurrence', id(node)))
            if reference is not None:
                break
        pending.extend((*node.children, *node.companions))
    if reference is None:
        tag, address = record['origins'][subject-1, :2].tolist()
        if tag in (1, 2):
            reference = int(address) if tag == 1 else ('ltm', store.address_domain, int(address))
    if reference is None:
        return None
    return reference, resolver._event_symbol(int(verb)), int(clause.order)


def phase_request(meaning, value):
    """Keep requested content in a meaning, with only its handle in metadata."""
    from Meaning import ConceptualMeaning
    data = bindings(meaning)
    data['_query_case'] = ('constituent', len(meaning.constituents))
    return replace(meaning, bindings=data, constituents=(*meaning.constituents,
        ConceptualMeaning.from_description(value)))


def case_coverage(owner, wanted, held):
    """Content entailment of a phase, without consulting occurrence time.

    An unselected event is unknown about its phase. Each evidence lane must
    cover the complete requested content; a proper fragment proves neither
    completion nor its negation.
    """
    import torch
    from IdentityRows import coordinates, meet_ratios
    if held is None:
        return 0., 0.
    a, b = (coordinates(owner, value) for value in (wanted, held))
    layout = getattr(owner.similarity_codebook, 'mereology', None)
    start = 0 if layout is None else layout.percept_event_width
    a, b = (value[start:].unflatten(-1, (2, -1)) for value in (a, b))
    _, covered = meet_ratios(a, b)
    complete = covered >= 1-8*torch.finfo(covered.dtype).eps
    return tuple(float(value) for value in complete)


def read(store, description, *, max_records, work, fallback):
    """Return one bounded set per timeline, or retained addressed ideas."""
    data = bindings(description)
    query = data.get('_query_event')
    if query is None:
        return None
    reference, verb, order = query
    phase = data.get('_query_case')
    if phase is not None:
        if (len(phase) != 2 or phase[0] != 'constituent' or type(phase[1]) is not int
                or not 0 <= phase[1] < len(description.constituents)):
            raise ValueError('a phase query requires its content constituent')
        phase = description.constituents[phase[1]].roles[0]
    owner = store._concept_owner()
    pool = owner.mtm
    key = pool.token_of(reference)
    entry = pool.entries.get(key)
    excluded = {*data.get('_query_exclude', ()), data.get('_query_occurrence')}
    scanned, incomplete, cache = 0, [], {}
    if entry is not None:
        groups = []
        for timeline in (entry.timelines if order > 0 else entry.timelines[:1]):
            records = []
            for event in timeline.events:
                # Charge a history read before inspecting its correspondence.
                marker = (event.occurrence, event.role)
                if marker not in cache:
                    if scanned >= max_records or not work.consume('record'):
                        incomplete.append('timeline_read_limit')
                        break
                    scanned += 1
                    row = store.location(event.occurrence)
                    cache[marker] = None if row is None else store.row(row)
                row = cache[marker]
                if (row is not None and event.role == 0 and event.verb == verb
                        and event.occurrence not in excluded and row['kind'] not in ('estimate', 'question', 'cover')):
                    if phase is not None:
                        coverage = case_coverage(owner, phase, event.case_value)
                        evidence = tuple(value*weight for value, weight in zip(row['evidence'], coverage))
                        if not any(evidence):
                            continue
                        row = dict(row, evidence=evidence)
                    records.append(dict(row, match=1.))
            groups.append(tuple(records))
        return dict(groups=tuple(groups), records_scanned=scanned,
                    incomplete=tuple(dict.fromkeys(incomplete)), retained=False, order=order)

    # Expiry removes branch organization, not ideas. The old derivation
    # reader supplies subject/predicate correspondence until its removal round.
    if phase is not None:
        return dict(groups=((),), records_scanned=scanned,
                    incomplete=('phase_history_expired',), retained=True, order=order)
    from PartRead import argument_reader
    operation, _ = argument_reader(store, incomplete)
    roots = set()
    if isinstance(reference, int):
        roots.add(reference)
    else:
        index = store.location(reference)
        if index is not None:
            root = store.raw_meaning_of(index).role_refs[0]
            if root and root[0] == 'sym':
                roots.add(root[1])
    clean = replace(description, bindings={k:v for k,v in data.items() if not k.startswith('_')})
    found = fallback(clean)
    scanned = found['records_scanned']
    incomplete.extend(found['incomplete'])
    candidates = list(found['value'])
    # A retained subject can have no content postings (an anonymous or
    # wholly changed state). Follow its native reference under the same
    # record bound instead of treating a nearest vector as its event.
    if not candidates:
        for index in range(len(store)):
            if scanned >= max_records or not work.consume('record'):
                incomplete.append('event_read_limit')
                break
            scanned += 1
            candidates.append(dict(store.row(index), index=index))
    visited = {}
    def pair(symbol):
        nonlocal scanned
        if symbol in visited:
            return visited[symbol]
        if scanned >= max_records or not work.consume('record'):
            incomplete.append('event_read_limit')
            return None, None
        scanned += 1
        args = tuple(store.parts.refs[symbol].tolist()) if store.parts.contains(symbol) else ()
        result = (args, operation(args) if args else None)
        visited[symbol] = result
        return result
    def event_of(symbol):
        seen = set()
        while symbol not in seen:
            seen.add(symbol)
            args, rule = pair(symbol)
            if not args or rule is None:
                return None
            if rule.clause_form == 'S':
                predicate, followed = args[1], set()
                while predicate not in followed:
                    followed.add(predicate)
                    children, head = pair(predicate)
                    if not children or head is None or not head.head_role:
                        break
                    predicate = children[head.head_role-1]
                return args[0], owner.components.reference_close._event_symbol(predicate)
            if not rule.head_role:
                return None
            symbol = args[rule.head_role-1]
        return None
    records = []
    for row in candidates:
        if row['occurrence'] in excluded or row['kind'] in ('estimate', 'question', 'cover'):
            continue
        raw = store.raw_meaning_of(row['index']).role_refs[0]
        if raw and raw[0] == 'sym':
            event = event_of(raw[1])
            if event is not None and event[0] in roots and event[1] == verb:
                records.append(dict(row, match=1.))
    return dict(groups=(tuple(records),), records_scanned=scanned,
                incomplete=tuple(dict.fromkeys(incomplete)), retained=True, order=order)


def evidence(found, description, *, every=False):
    """Bilateral some/every; an unwitnessed timeline contributes unknown."""
    groups, frames = [], []
    for records in found['groups']:
        positive = negative = 0.
        for row in records:
            poles = row['evidence']
            if row['meaning'].polarity != description.polarity:
                poles = poles[::-1]
            positive, negative = max(positive, poles[0]), max(negative, poles[1])
            frames.append(row)
        groups.append((positive, negative))
    if not groups:
        groups = [(0., 0.)]
    positive = (min if every else max)(p for p,n in groups)
    negative = (max if every else min)(n for p,n in groups)
    if every and (found['incomplete'] or found['retained']):
        positive = 0.  # an expired or unread branch cannot certify every
    frames = tuple({row['occurrence']: row for row in frames}.values())
    return dict(value=frames, frames=frames, result_kind='set', meaning=description,
                support_true=positive, support_false=negative,
                witnesses=tuple(row['occurrence'] for row in frames),
                records_scanned=found['records_scanned'], incomplete=found['incomplete'])
