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
        if lower or name == 'bind':
            positions = {(head or len(children))-1: True}
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
