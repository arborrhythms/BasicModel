"""Rebuildable reads of the native Part derivation.

No reading in this module unfolds a concept value or retains working identity.
Cycles are legal; a traversal visits each native symbol at most once.
"""
def leaves(parts, root):
    """The terminal symbols reachable through ordered composing pairs."""
    pending, visited, result = [int(root)], set(), []
    while pending:
        symbol = pending.pop()
        if symbol in visited:
            continue
        visited.add(symbol)
        if parts.contains(symbol):
            pending.extend(reversed(parts.refs[symbol].tolist()))
        else:
            result.append(symbol)
    return tuple(result)


def word_rows(owner, parts, root):
    """Read lexical seats from the retained arguments and definition edges."""
    from Spaces import _concept_alloc_of
    layer = _concept_alloc_of(owner).layer()
    # A native perceptual witness also supplies a form, including fixtures
    # and non-text inputs which do not have a surface spelling.
    witnessed = {row for row, column in layer.features._index
                 if (column // 2) % 2 == 0}
    result = set()
    for symbol in leaves(parts, root):
        # A lexical object names its word through the common Part store.
        # An objectless word is already its own native lexical symbol.
        row = owner._csw_row_of(symbol)
        if row is not None and (row in witnessed or owner.word_surface_for_row(row) is not None):
            result.add(int(row))
    return result


def occurrence_roots(owner, store, b=0):
    """Transient context rows, addressed by their native occurrence or symbol.

    LTM1's existing reference column points to the derivation of an ended
    concept. A complete relative close is itself an LTM2 root. Definitions
    and interior composing nodes are not extra sentence observations.
    """
    from Occurrence import sentence_key
    parts = store.parts
    recency = store.recency_weights(b)
    records, covered = [], set()
    for row in range(len(store)):
        if int(store.role_mask[row].sum()) != 1 or int(store.record_kind[row]) == store.KINDS.index('estimate'):
            continue
        meaning = store.raw_meaning_of(row)
        reference = meaning.role_refs[0]
        if reference is None or reference[0] != 'sym':
            continue
        root = int(reference[1])
        covered.add(root)
        records.append(dict(root=root, address=store.occurrence_of(row),
            key=store.content_key(row), witnesses=int(store.witness_count[row]),
            weight=recency[row]))
    owner.definitions.word_ids  # refresh the rebuildable definition index
    definitions = set(owner.definitions._rows.values())
    nodes = set(parts.indices().tolist()) - definitions
    inner = {argument for node in nodes for argument in parts.refs[node].tolist()
             if argument in nodes}
    roots = sorted(nodes - inner - covered)
    for position, root in enumerate(roots):
        words = []
        for symbol in leaves(parts, root):
            row = owner._csw_row_of(symbol)
            form = None if row is None else owner.word_surface_for_row(row)
            if form is not None:
                words.append(form.encode() if isinstance(form, str) else form)
        # A lexical sentence key is independent of its numerical reading.
        # Nonlexical roots cannot bootstrap a word context.
        if not words:
            continue
        records.append(dict(root=root, address=('sym', root), key=sentence_key(words),
            witnesses=int(parts.access_count[root]),
            weight=store.slots.new_tensor(1. / (len(roots) - position))))
    return records


def occurrence_codes(records, pairs, ones, like):
    from MeaningCodes import identity_code, defined_code
    return {row: defined_code(identity_code(record['key'], pairs, ones, like=like),
                              record['witnesses'])
            for row, record in enumerate(records)}


def expression_components(store, roots, *, max_records, work, scanned=0):
    """Read semantic operands from admitted occupations and ordered pairs.

    The view is local to this bounded read. A head attachment exposes its
    head's view; a clause keeps its subject beside that head. No producing
    operation or working identity is added to the retained Part rows.
    """
    from collections import Counter
    from Meaning import ConceptualMeaning
    from PartsOfSpeech import operation_name
    from Spaces import _concept_alloc_of
    owner = store._concept_owner()
    allocator = _concept_alloc_of(owner)
    ref = store.__dict__.get('_occupation_owner')
    occupations = None if ref is None else ref()
    rules = store.__dict__.get('_part_read_rules', ())
    components, active, representatives, incomplete = {}, set(), {}, []

    def counts(symbol):
        direct = Counter() if occupations is None else Counter(occupations.counts.get(symbol, {}))
        words = Counter()
        if occupations is not None:
            for word in owner.definitions.words(symbol):
                words.update(occupations.counts.get(word, {}))
        return direct, words

    def operation(arguments):
        evidence = [counts(symbol) for symbol in arguments]
        # A marker contributes surface form, even when the admitted
        # numerical derivation combined it with an unheaded operation.
        # Its objectless argument type owns the semantic head read.
        if marker(arguments[0], set()):
            surface = next((rule for rule in rules if rule.method_name == 'surface'), None)
            if surface is not None:
                return surface
        for fallback in (False, True):
            matches = {}
            for rule in rules:
                if len(rule.rhs_symbols) != 2:
                    continue
                name = operation_name(rule)
                if rule.method_name == 'surface' and not marker(arguments[0], set()):
                    continue
                roles = (2, 1) if rule.method_name == 'whole' else (1, 2)
                if all((direct + words if fallback else direct)[f'{name}_argin{role}'] > 0
                       for (direct, words), role in zip(evidence, roles)):
                    reading = (int(rule.head_role or 0), rule.clause_form, rule.relation_kind)
                    matches.setdefault(reading, rule)
            if len(matches) == 1:
                return next(iter(matches.values()))
            if len(matches) > 1:
                incomplete.append('ambiguous_argument_occupations')
                return None
        return None

    def marker(symbol, seen):
        if symbol in seen:
            return False
        if store.parts.contains(symbol):
            return all(marker(child, seen | {symbol})
                       for child in store.parts.refs[symbol].tolist())
        direct, words = counts(symbol)
        # Read this argument's type. A later definition of an object named
        # by the same word does not turn the retained marker into that object.
        return ('object' not in allocator.symbol_types.get(symbol, ())
                and (direct + words)['surface_argin1'] > 0)

    def read(symbol):
        nonlocal scanned
        address = ('sym', int(symbol))
        if address in representatives:
            return representatives[address]
        if address in active:
            incomplete.append('component_cycle')
            return address
        if scanned >= max_records or not work.consume('record'):
            incomplete.append('component_read_limit')
            return address
        scanned += 1
        active.add(address)
        if not store.parts.contains(symbol):
            point = store._concept_point(symbol)
            if point is not None:
                components[address] = dict(kind='concept', operands=(),
                    meaning=ConceptualMeaning.from_description(store._fit(point)))
            representatives[address] = address
            active.remove(address)
            return address
        row = store.part_row(symbol)
        arguments = tuple(row['refs'].tolist())
        children = tuple(read(child) for child in arguments)
        rule = operation(arguments)
        kind = 'part' if rule is None else rule.method_name
        head = 0 if rule is None else int(rule.head_role or 0)
        if rule is not None and rule.method_name == 'whole' and head:
            head = 3 - head
        # A situation owns both semantic operands. Heads only strip the
        # surrounding attachments; they cannot discard the clause's subject.
        clause = rule is not None and (rule.clause_form == 'S' or rule.relation_kind is not None)
        if head and not clause and children[head-1] in components:
            representative = children[head-1]
            components[address] = dict(components[representative])
            components[address]['reference'] = representative
            # An identity read must inspect each named head before descending
            # further. Flattening is useful for a structural view, but can
            # skip a composite that has itself been bound to a word.
            components[address]['head_reference'] = ('sym', arguments[head-1])
        else:
            representative = address
            components[address] = dict(kind=kind, operands=children, meaning=row['meaning'])
        representatives[address] = representative
        active.remove(address)
        return representative

    for reference in roots:
        if reference[0] == 'sym':
            read(reference[1])
    return components, scanned, tuple(dict.fromkeys(incomplete))
