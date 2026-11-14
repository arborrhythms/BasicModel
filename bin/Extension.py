"""Concept sets carried by the native compose and generate operators.

The input face reads an occurrence's existing activations. Neither a set
operation nor a determiner searches a codebook for a similar vector.
"""
from dataclasses import dataclass
import torch
from FieldLogic import reduce


@dataclass(frozen=True)
class ConceptSet:
    ids: torch.Tensor
    values: torch.Tensor
    missing: bool = False

    def __post_init__(self):
        if self.ids.ndim != 1 or self.values.shape != (len(self.ids), 2):
            raise ValueError('an extension requires concept references and two lanes')

    def aligned(self, ids):
        if not len(self.ids):
            return self.values.new_zeros(len(ids), 2)
        order = self.ids.argsort()
        source = self.ids[order]
        positions = torch.searchsorted(source, ids).clamp_max(len(source)-1)
        return self.values[order[positions]] * source[positions].eq(ids)[:, None]

    def withheld(self):
        return ConceptSet(self.ids, torch.zeros_like(self.values))


@dataclass(frozen=True)
class Context:
    owner: object = None
    name: str | None = None
    content: int | None = None
    identity: tuple | None = None
    trust: float | tuple = 0.
    open: bool = False
    rows: bool = False


NARROW = frozenset(('intersection', 'compound', 'compound_unary', 'adverb', 'verb',
                    'conjunction', 'pi', 'min'))
JOIN = frozenset(('union', 'disjunction', 'sigma', 'max'))
DETERMINERS = frozenset(('definite', 'definite_unary', 'lower', 'lower_unary', 'universal', 'some'))
CONTENT = frozenset(('preposition', 'preposition_unary', 'surface', 'surface_unary', 'sign',
    'interpret', 'bind', 'lift', 'transpose', 'generic', 'ask', 'part', 'whole', 'equal',
    'implies', 'gloss', 'divide', 'descend', 'sum', 'product', 'synthesize',
    'isPart', 'isEqual', 'assertPart'))


def combine(operands, operation):
    present = tuple(item for item in operands if not item.missing)
    if not present:
        first = operands[0]
        return ConceptSet(first.ids,torch.zeros_like(first.values),True)
    operands = present
    ids = torch.cat([item.ids for item in operands]).unique(sorted=True)
    values = torch.stack([item.aligned(ids) for item in operands], 1)
    return ConceptSet(ids, reduce(values, operation=operation, dim=1))


def _stored_values(owner, ids, values):
    """Read declared members once across the whole batch and reading axis."""
    from ConceptHierarchy import members
    active = values.detach().reshape(len(ids), -1).any(1) if len(ids) else ids.bool()
    positions = (active & (ids >= 0)).nonzero().flatten()
    sources, targets = [], []
    for position, identity in zip(positions.tolist(), ids[positions].tolist()):
        children = members(owner, identity)
        for child in children or (identity,):
            sources.append(position)
            targets.append(child)
    targets, inverse = ids.new_tensor(targets).unique(sorted=True, return_inverse=True)
    result = values.new_zeros((len(targets), *values.shape[1:]))
    if sources:
        gathered = values[ids.new_tensor(sources)]
        index = inverse.reshape(-1, *([1] * (values.ndim-1))).expand_as(gathered)
        result = result.scatter_reduce(0, index, gathered, reduce='amax', include_self=False)
    return targets, result


def stored(owner, source):
    """An extension is the declared members, or the concept itself if atomic."""
    return ConceptSet(*_stored_values(owner, source.ids, source.values))


def stored_field(owner, field):
    """Keep each emitted terminal's locations while reading its members."""
    from Understanding import ConceptualField
    ids, evidence = _stored_values(owner, field.concept_ids, field.evidence)
    return ConceptualField(ids, evidence, field.where, field.when, None)


def interpreted(owner, source):
    """Read the same declared word-to-object links as the native interpretation.

    A mandatory symbol shift has already interpreted its numerical leaf;
    an optional interpretation performs this read at its explicit operator.
    Both transform existing memberships without identifying any vector.
    """
    contributions = {}
    words = owner.definitions.word_ids
    for identity,value in zip(source.ids.tolist(),source.values):
        if identity < 0:
            continue
        objects = owner.interpret.denotations(identity) if identity in words else ()
        for reference in objects or (identity,):
            contributions.setdefault(reference,[]).append(value)
    ids = source.ids.new_tensor(sorted(contributions))
    values = (torch.stack([torch.stack(contributions[int(cid)]).amax(0) for cid in ids])
              if len(ids) else source.values.new_zeros(0,2))
    return ConceptSet(ids,values,source.missing)


def content_index(operator, operands, context):
    if context.content is not None:
        return min(context.content, len(operands)-1)
    semantic = getattr(operator, 'semantic_operand', None)
    head = getattr(operator, 'head_role', 0)
    return min(semantic if semantic is not None else max(0, head-1), len(operands)-1)


def compose(operator, operands, context=None):
    context = Context() if context is None else context
    name = context.name or operator.rule_name
    if not operands:
        raise ValueError('compose_extend requires its declared operands')
    if name in NARROW | JOIN:
        return combine(operands, 'intersection' if name in NARROW else 'union')
    source = operands[content_index(operator, operands, context)]
    if name == 'interpret':
        owner = context.owner if context.owner is not None else operator.owner
        return stored(owner,interpreted(owner,source))
    if name in ('not', 'non'):
        return ConceptSet(source.ids, reduce(source.values, operation=name))
    if name == 'ask' and context.open:
        return ConceptSet(source.ids,torch.zeros_like(source.values),True)
    if name == 'null':
        return source.withheld()
    if name in DETERMINERS:
        if context.rows:
            if name == 'universal':
                return source
            count = int((source.values[:,0] > 0).sum())
            number = max(2,count//2) if name == 'some' and count >= 3 else 0 if name == 'some' else min(1,count)
            # Rows arrive in chronological order; equal relevance chooses
            # the newer row, as the content recall does.
            order = source.values[:,0].flip(0).argsort(descending=True,stable=True)
            keep = torch.zeros_like(source.values[:,0])
            keep[len(order)-1-order[:number]] = 1
            return ConceptSet(source.ids,source.values*keep[:,None])
        if context.owner is None:
            raise ValueError('a determiner extension requires its concept owner')
        source = stored(context.owner, source)
        values = source.values
        present = values.amax(-1) > 0
        if name in ('definite', 'definite_unary'):
            identity = context.identity
            mask = (source.ids == identity[1]) if identity is not None and identity[0] == 'sym' else ~present
            trust = context.trust if isinstance(context.trust,tuple) else (context.trust,0.)
            values = values.new_tensor(trust).expand_as(values) * (mask & present)[:,None]
        elif name in ('lower', 'lower_unary'):
            values = torch.stack((present.to(values) / present.sum().clamp_min(1),
                                  torch.zeros_like(values[:, 1])), -1)
        elif name == 'some':
            # No proper plural subset exists for a kind with fewer than three
            # members. Such a kind therefore supplies no answer to this form.
            count = int(present.sum())
            keep = torch.zeros_like(present)
            if count >= 3:
                ranked = values.amax(-1).argsort(descending=True, stable=True)
                keep[ranked[:max(2, count//2)]] = True
            values = values * keep[:, None]
        return ConceptSet(source.ids, values)
    if name in CONTENT:
        return source
    raise ValueError(f'{name} has no extension contract')


def generate(operator, parent, operands, context=None):
    """Return each operand's part of the parent's already composed extension."""
    context = Context() if context is None else context
    name = context.name or operator.rule_name
    if name == 'not':
        return (ConceptSet(parent.ids, parent.values.flip(-1)),)
    if name in ('non', 'null'):
        return tuple(operand.withheld() for operand in operands)
    if name in CONTENT:
        content = content_index(operator, operands, context)
        return tuple(parent if i == content and not (name == 'ask' and context.open)
                     else operand.withheld() for i, operand in enumerate(operands))
    if name in NARROW | JOIN | DETERMINERS:
        return tuple(ConceptSet(parent.ids, torch.minimum(parent.values, operand.aligned(parent.ids)))
                     for operand in operands)
    raise ValueError(f'{name} has no inverse extension contract')


@dataclass(frozen=True)
class Node:
    field: ConceptSet
    children: tuple = ()
    operator: object = None
    context: Context = Context()
    reference: tuple | None = None

    def operand(self, path):
        """Read an operand only under the same declared operators."""
        current, value = self, self.field
        for name, position in path:
            if current.context.name != name or position >= len(current.children):
                return None
            values = current.operator.generate_extend(value,
                *(child.field for child in current.children), context=current.context)
            current, value = current.children[position], values[position]
        return value

    def open_paths(self, prefix=()):
        if self.context.open:
            return (prefix,)
        return tuple(path for i, child in enumerate(self.children)
                     for path in child.open_paths((*prefix, (self.context.name, i))))


def as_field(sets, *, where, when):
    from Understanding import ConceptualField
    ids = torch.cat([value.ids for value in sets]).unique(sorted=True)
    evidence = torch.stack([value.aligned(ids) for value in sets], 1)[:, :, None]
    # A set has one extent. The realization's separate extents still locate
    # words in the generated surface; these bounds describe the whole idea.
    def extent(bounds):
        valid = bounds[..., 1] > bounds[..., 0]
        maximum = (torch.finfo(bounds.dtype) if bounds.is_floating_point() else torch.iinfo(bounds.dtype)).max
        lo = bounds[..., 0].masked_fill(~valid, maximum).amin(1)
        hi = bounds[..., 1].amax(1)
        return torch.stack((torch.where(valid.any(1),lo,0),hi),-1)[:,None]
    return ConceptualField(ids, evidence, extent(where), extent(when), None)


def joined(forest, empty):
    return combine(tuple(node.field for node in forest), 'union') if forest else empty


def held_context(owner, rule, reference=None, *, open=False):
    """Determiners use the identity already selected by compose, if MTM holds it."""
    name = getattr(rule, 'surface_name', None) or rule.method_name
    identity, trust = None, 0.
    if name in ('definite', 'definite_unary') and reference is not None:
        mtm = owner.mtm
        entries = [entry for entry in mtm.entries.values() if not entry.pending
                   and (reference in entry.occurrences or reference in
                        (('sym',entry.symbol),('sym',entry.native))) ]
        if entries:
            entry = max(entries, key=lambda value: value.when)
            symbol = entry.symbol if entry.symbol is not None else entry.native
            identity = ('sym',symbol) if symbol is not None else reference
            trust = tuple(max((reading[2+lane] for reading in entry.readings),default=0.) for lane in range(2))
    semantic = getattr(rule, 'semantic_operand', None)
    content = semantic if semantic is not None else max(0, getattr(rule, 'head_role', 0)-1)
    return Context(owner, name, content, identity, trust, open)


def reading(model, program, field, row, positions):
    """Extend the pinned STM tree once in its actual compose order."""
    owner, language = model._concept_owner(), model.languageSpace
    ids = field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:,row]
    valid = ids >= 0
    ids = ids[valid]
    values = field.evidence[valid,row]
    stack = []
    dispatch = language._tree_layer(2)
    for at, (kind, local, word) in enumerate(program.actions.tolist()):
        if kind < 0:
            break
        if kind == 0:
            reference = int(program.concept_ids[word])
            source = ConceptSet(ids,values[:,int(positions[word])])
            if program.interpret_forced:
                source = stored(owner,interpreted(owner,source))
            node=Node(source,reference=('sym',reference) if reference > 0 else None)
            stack.append(node)
            if program.tree is not None:
                program.tree.install_extension(node,word=word)
            continue
        if kind == 3:
            # A transposition rewires the STM pair; it is not an application
            # node in that pair's kept tree (SentenceStrata.Tree.forest).
            stack[-2:] = reversed(stack[-2:])
            continue
        rules = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
        rule = rules[local]
        arity = 2 if kind == 1 and rule.method_name != 'bind' else 1
        offset = max(0,word) if kind == 2 else 0
        if offset:
            stack.append(stack.pop(-1-offset))
        children = tuple(stack[-arity:]); del stack[-arity:]
        wrapped = (dispatch.ops if kind == 1 else dispatch.unary_ops)[local]
        operator = getattr(wrapped,'gl',wrapped)
        reference = None
        if program.tree is not None:
            selected = program.tree.composition(action=at).reference
            if selected not in (-1,0):
                reference = owner._closed_clause_store().semantic_reference(selected)
        context = held_context(owner,rule,reference,open=rule.method_name == 'ask')
        value = operator.compose_extend(*(child.field for child in children),context=context)
        node=Node(value,children,operator,context,reference)
        stack.append(node)
        if program.tree is not None:
            program.tree.install_extension(node,action=at)
        if offset:
            stack.insert(len(stack)-1-offset,stack.pop())
    return tuple(stack)


def with_extensions(clause, roots=None):
    """Carry each kept constituent's operated field to its ordinary close."""
    from dataclasses import replace
    if clause is None:
        return None
    extension=getattr(clause.composition,'extension',()) if roots is None else roots
    return replace(clause,meaning=replace(clause.meaning,extension=extension),
        children=tuple(with_extensions(child) for child in clause.children),
        companions=tuple(with_extensions(child) for child in clause.companions))


def understand(model, record, programs):
    forest = tuple(() if program is None else reading(model, program, record.open_field, row,
        model._sentence_leaf_positions(record,row)) for row,program in enumerate(programs))
    empty = ConceptSet(record.open_field.concept_ids.new_empty(0), record.open_field.evidence.new_empty(0,2))
    field = as_field(tuple(joined(roots,empty) for roots in forest),
                     where=record.open_field.where,when=record.open_field.when)
    return forest,field


def generated(model, actions, counts, depths, leaf_field, references):
    """Read the tree of this generate walk, never a compose tree or new walk."""
    owner, language = model._concept_owner(), model.languageSpace
    rules = (*language._generate_binary_rules,*language._generate_unary_rules)
    ops = (*language._generate_binary_ops,*language._generate_unary_ops)
    binary, stop = len(language._generate_binary_ops), len(ops)
    extended = stored_field(owner, leaf_field)
    forests, sets = [], []
    for b, row in enumerate(actions.tolist()):
        roots = [dict() for _ in range(int(depths[b]))]
        stack, nodes = list(roots), []
        emitted = 0
        ids = leaf_field.concept_ids
        empty = ConceptSet(ids.new_empty(0),leaf_field.evidence.new_empty(0,2))
        for choice in row:
            if choice < 0:
                break
            current = stack.pop()
            nodes.append(current)
            if choice == stop:
                position = int(counts[b])-1-emitted; emitted += 1
                if not 0 <= position < leaf_field.evidence.shape[2]:
                    current['node'] = Node(empty)
                    continue
                reference = references[b][position] if references else None
                from OutputOccurrences import object_identity
                identity = object_identity(owner,reference) if reference is not None else None
                if reference is None:
                    source = ConceptSet(extended.concept_ids, extended.evidence[:, b, position])
                else:
                    matched = ids == identity if identity is not None else ids < 0
                    source = stored(owner, ConceptSet(ids[matched], leaf_field.evidence[matched,b,position]))
                current['node'] = Node(source,reference=reference)
            else:
                arity = 2 if choice < binary else 1
                children = [dict() for _ in range(arity)]
                current.update(choice=choice,children=children)
                stack.extend(children)
        for node in reversed(nodes):
            if 'node' in node:
                continue
            children = tuple(child.get('node',Node(empty)) for child in node['children'])
            choice = node['choice']; operator = getattr(ops[choice],'gl',ops[choice])
            context = held_context(owner,rules[choice])
            value = operator.compose_extend(*(child.field for child in children),context=context)
            node['node'] = Node(value,children,operator,context)
        forest = tuple(root.get('node',Node(empty)) for root in roots)
        forests.append(forest); sets.append(joined(forest,empty))
    return tuple(forests), as_field(sets,where=leaf_field.where,when=leaf_field.when)
