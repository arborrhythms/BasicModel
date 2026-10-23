"""Participant positions in a kept derivation, before reference choice.

The operation catalogue supplies the heads. A noun phrase follows its head;
an embedded clause exposes its participants instead. Surface operands and
predicate heads never become objects merely because they have a word row.
"""


def participant_terms(roots, rules=()):
    """Return the eligible compositions and the compositions actually inspected.

    This is an eager reading of the admitted operation journal. The same
    head declarations underlie the occupation-based read of retained pairs.
    """
    from Language import LowerLayer, LiftLayer, VerbLayer, SurfaceLayer
    heads = {name: cls.head_role for name, cls in (
        ('lower', LowerLayer), ('lift', LiftLayer), ('verb', VerbLayer),
        ('surface', SurfaceLayer))}
    catalogue = {rule.method_name: rule for rule in rules}
    eligible, inspected, seen = {}, {}, set()

    pending = list(reversed(tuple(roots)))
    while pending:
        term, noun = pending.pop()
        if term is None or (id(term), noun) in seen:
            continue
        seen.add((id(term), noun))
        inspected[id(term)] = term
        rule = catalogue.get(term.operation)
        name = term.operation
        head = int(term.head_role or getattr(rule, 'head_role', 0) or heads.get(name, 0))
        children = term.children
        lower = (name in ('lower', 'lower_unary', 'definite', 'definite_unary')
                 or getattr(rule, 'determiner_mode', None) is not None)
        clause = (name in ('lift', 'equal', 'part', 'whole')
                  or getattr(rule, 'clause_form', None) == 'S'
                  or getattr(rule, 'relation_kind', None) is not None)
        if noun or lower or name == 'bind':
            eligible[id(term)] = term
        if not children:
            continue
        if lower:
            positions = {(head or len(children))-1: True}
        elif name == 'bind':
            # Binding is scope-transparent; it does not redetermine a VP
            # as a noun phrase and replace its declared participant roles.
            positions = {(head or len(children))-1: noun}
        elif name in ('equal', 'part', 'whole'):
            positions = {i: True for i in range(len(children))}
        elif name == 'lift' and len(children) == 2:
            positions = {i: i+1 != head for i in range(len(children))}
        elif noun and head and not clause:
            positions = {head-1: True}
        elif name == 'verb' and len(children) == 2:
            positions = {i: i+1 != head for i in range(len(children))}
        elif head:
            positions = {head-1: noun}
        else:
            positions = {i: noun for i in range(len(children))}
        # Inspect every subtree for its own declared positions. A non-head
        # operand is not itself a noun, but can contain a prior determination.
        pending.extend((child, positions.get(i, False))
                       for i, child in reversed(tuple(enumerate(children))))
    return eligible, inspected

def clause_participants(clause, rules=()):
    """Read both the operation journal and already factored semantic roles.

    Direct semantic clauses need no invented operation. Their explicit
    participant roles are the corresponding position record.
    """
    nodes, seen = [], set()
    def collect(node):
        if node is None or id(node) in seen:
            return
        seen.add(id(node)); nodes.append(node)
        for child in (*node.children, *node.companions):
            collect(child)
    collect(clause)
    roots = []
    owned = {id(child) for node in nodes for child in (*node.children, *node.companions)}
    for node in nodes:
        if id(node) not in owned:
            roots.append((node.composition, False))
    eligible, inspected = participant_terms(roots, rules)
    positions = set()
    for node in nodes:
        # With a composition the actual parse owns all positions. Semantic
        # role records supply positions only when that journal is absent.
        if node.composition is None:
            for role in (0, 2):
                ref = node.refs[role]
                if isinstance(ref, tuple) and ref[:1] == ('clause',):
                    child = node.children[ref[1]]
                    positions.add(id(child))
                    extra, read = participant_terms(((child.composition, True),), rules)
                    eligible.update(extra); inspected.update(read)
        elif id(node.composition) in eligible:
            positions.add(id(node))
    # An explicitly supplied standalone reference request is itself a
    # participant record. The journal's leaf request is provisional and must
    # instead be licensed by an enclosing operation.
    for node in nodes:
        request = node.reference_request or {}
        if (node is clause and not request.get('leaf')
                and (request or node.order == 1)
                and (node.composition is None or not node.composition.children
                     or node.relation is None and node.order == 1 and node.composition.head_role
                     and node.composition.operation not in ('lift', 'verb', 'equal', 'part', 'whole')
                     and not any(rule.method_name == node.composition.operation
                         and (getattr(rule, 'clause_form', None) in ('S', 'VP')
                              or getattr(rule, 'relation_kind', None) is not None) for rule in rules))):
            positions.add(id(node))
            extra, read = participant_terms(((node.composition, True),), rules)
            eligible.update(extra); inspected.update(read)
    return nodes, positions, eligible


def nonparticipants(store, roots, *, noun=False):
    """Known marker/predicate positions in the frames' retained derivations.

    Unknown occupations do not invent a classification. A root object is
    already accessible in its own right; the rule concerns its constituents.
    """
    from ClauseRow import Composition
    from PartRead import argument_reader
    operation, _ = argument_reader(store)
    terms, active, classified = {}, set(), set()
    def read(symbol):
        if symbol in terms:
            return terms[symbol]
        if symbol in active:
            return Composition(None, reference=symbol)
        active.add(symbol)
        children, rule = (), None
        if store.parts.contains(symbol):
            arguments = tuple(store.parts.refs[symbol].tolist())
            rule = operation(arguments)
            children = tuple(read(child) for child in arguments)
            if rule is not None:
                classified.add(symbol)
            if rule is not None and rule.method_name == 'whole':
                children = children[::-1]
        terms[symbol] = Composition(None, children, reference=symbol,
            operation=None if rule is None else rule.method_name,
            head_role=0 if rule is None else int(rule.head_role or 0))
        active.remove(symbol)
        return terms[symbol]
    trees = [read(root) for root in roots]
    if not classified:
        return set()
    eligible, _ = participant_terms(((tree, noun) for tree in trees),
                                   store.__dict__.get('_part_read_rules', ()))
    # Only a classified ancestor supplies the role of a constituent. Unknown
    # root pairs (including lexical definitions) are not negative evidence.
    excluded = set()
    def collect(term):
        for child in term.children:
            if id(child) not in eligible:
                excluded.add(child.reference)
            collect(child)
    for symbol in classified:
        collect(terms[symbol])
    return excluded
