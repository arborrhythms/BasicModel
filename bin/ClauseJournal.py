"""Temporary grammar metadata and numerical values of an open reading.

Nothing from this journal is stored beside a completed row. The numerical
operators have already run in the word brick; finishing only assigns roles
and native references to their actual end-state contents.
"""
from dataclasses import replace

import torch

from ClauseRow import Clause, ClausePredicate, predicate_point, predicate_relation
from Meaning import ConceptualMeaning


def finish_clause(language, program, *, meaning=None, depth=1, registry=None):
    """Finish an open reading from the values its operations already produced.

    The action record supplies grammar metadata only. Numerical values come
    from the live operation journal, including the operands before fusion.
    This function never runs a compose operator, and its reading record is
    not part of the returned field or the row written from it.
    """
    if program is None:
        raise ValueError('cannot finish a reading that is not open')
    from Language import TheGrammar
    refs = program.reference_ids if program.reference_ids is not None else program.concept_ids
    values = list(program.leaves.unbind())
    frames = program.operation_values
    if frames is None:
        raise ValueError('the open reading has no numerical operation journal')
    stack = []
    def metadata(words):
        # These are input identification poles, never the source's authority.
        evidence = program.leaf_evidence
        if evidence is None:
            pair = (0., 0.)
        else:
            selected = evidence[list(words)]
            known = selected > 0
            minimum = torch.where(known, selected, torch.inf).amin(0)
            pair = tuple(torch.where(known.any(0), minimum, 0.).detach().tolist())
        orders = program.leaf_orders
        if program.reference_orders is not None:
            orders = (program.reference_orders if orders is None else
                      torch.where(program.reference_orders >= 0, program.reference_orders, orders))
        order = 0 if orders is None else max(0, int(orders[list(words)].max()))
        return dict(order=order, evidence=pair)
    # Each node owns the exact grammar action and its live operands.
    for action, (kind, local, word) in enumerate(program.actions.detach().cpu().tolist()):
        if kind < 0:
            break
        if kind == 0:
            has_point = program.reference_relations is None or not bool(
                program.reference_relations[word])
            stack.append(dict(value=values[word], ref=int(refs[word]), rule=None,
                              children=(), leaves=(word,), has_point=has_point))
            continue
        arity = 2 if kind == 1 else 1
        catalog = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
        if kind not in (1, 2) or not 0 <= local < len(catalog) or len(stack) < arity:
            raise ValueError('invalid clause derivation')
        operands = tuple(stack[-arity:])
        del stack[-arity:]
        relation_rule = bool(getattr(catalog[local], 'relation_kind', None))
        selected_refs = (None if program.operation_refs is None else
                         list(program.operation_refs[action].tolist()))
        for role, (operand, actual) in enumerate(zip(operands, frames[action, :arity])):
            operand['value'] = actual
            if program.operation_relations is not None:
                operand['has_point'] = not bool(program.operation_relations[action, role])
            if program.operation_refs is not None:
                selected = operand
                while selected['children']:
                    headed = getattr(selected['rule'], 'head_role', 0)
                    if not headed:
                        break
                    selected = selected['children'][headed-1]
                reference = int(program.operation_refs[action, role])
                # A trial's newly requested singleton still had its source
                # address. Resolve that alias to the winning host's admitted
                # identity; an already selected earlier occurrence is unchanged.
                pending_order = False
                if not selected['children'] and program.reference_orders is not None:
                    leaf = selected['leaves'][0]
                    pending_order = (int(program.reference_orders[leaf]) >= 0
                                     and reference == int(program.concept_ids[leaf]))
                    if pending_order:
                        reference = int(refs[leaf])
                if reference > 0 or pending_order:
                    selected['ref'] = reference
                    selected_refs[role] = reference
        stack.append(dict(value=frames[action, 2], local=local, ref=-1,
                          rule=catalog[local], children=operands,
                          operand_refs=None if selected_refs is None else tuple(selected_refs),
                          leaves=tuple(i for child in operands for i in child['leaves']),
                          has_point=not relation_rule and all(child['has_point'] for child in operands)))
    if len(stack) not in (1, 3):
        raise ValueError('a completed field must have one or three slots')

    def subject_word_id(node):
        leaf = head(node)['leaves'][0]
        if program.word_ids is not None:
            return int(program.word_ids[leaf])
        if registry is None:
            return -1
        obj = registry.space.concept_id_at_row(int(program.word_rows[leaf]))
        return (-1 if obj is None else registry.space.word_concept_of_object(obj) or -1)

    def value(node):
        return node['value']

    def bands(node):
        result = {}
        for key in ('where', 'when'):
            source = getattr(program, 'symbol_' + key)
            if source is not None:
                # These are quadrature ladders, not four interval endpoints.
                # The field is stamped once; its bracket starts at the first
                # owned leaf and its end is content-terminated (item 9b).
                result[key] = source[node['leaves'][0]].clone()
        return result

    relative = {'part', 'whole', 'equal', 'implies', 'operator'}
    absolute_starts = set(TheGrammar.ws_absolute_starts)

    def head(node, *, clause=False):
        while node['children']:
            if clause and getattr(node['rule'], 'clause_form', None) == 'VP':
                break
            role = getattr(node['rule'], 'head_role', 0)
            if not role:
                break
            node = node['children'][role - 1]
        return node

    def generic(node):
        while True:
            requests = dict(getattr(node['rule'], 'reference_kinds', ()))
            if any(mode in ('generic', 'kind') for mode in requests.values()):
                return True
            if any(mode in ('particular', 'name', 'pronoun') for mode in requests.values()):
                return False
            selected = head(node)
            if selected is node:
                return False
            node = selected

    def concept(node):
        selected = head(node)
        if selected['ref'] > 0:
            return selected['ref']
        # A composite phrase has no concept-inventory identity. When a
        # relative row needs its address, operand() writes its ended point.
        return -1

    def operation_concept(node, relation=None):
        """The selected grammar predicate lives only in its row occurrences."""
        identity = relation if relation is not None else getattr(
            node['rule'], 'predicate_identity', None)
        if identity is None:
            raise ValueError('a clause predicate requires a declared identity')
        point = predicate_point(identity, program.leaves)
        return ClausePredicate(point, identity)

    def is_clause(node):
        node = head(node, clause=True)
        while len(node['children']) == 1 and getattr(node['rule'], 'scope_transparent', False):
            node = head(node['children'][0], clause=True)
        return (getattr(node['rule'], 'relation_kind', None) is not None or
                getattr(node['rule'], 'clause_form', None) in ('S', 'implies') or
                getattr(node['rule'], 'lhs', None) in absolute_starts or
                (len(node['children']) == 2 and not node['has_point']
                 and getattr(node['rule'], 'clause_form', None) != 'VP'))

    def recover(root, *, top=False):
        pair = metadata(root['leaves'])['evidence']
        polarity = pair[0] >= pair[1]
        excluded = False
        mode = 'assertive'
        node = head(root, clause=True)
        while len(node['children']) == 1:
            if getattr(node['rule'], 'polarity_effect', None) == 'invert':
                polarity = not polarity
            if getattr(node['rule'], 'polarity_effect', None) == 'exclude':
                excluded = True
            mode = getattr(node['rule'], 'meaning_mode', None) or mode
            node = head(node['children'][0], clause=True)
        operator = getattr(node['rule'], 'relation_kind', None)
        children = []
        owned = set()
        factored_refs = None
        role_nodes = (node, None, None)

        def operand(item, *, sentence=False):
            reference = concept(item) if not is_clause(item) else None
            # An unindexed numerical NP (e.g. an embedding input) still
            # has a field. If a relative S needs its address, close that NP
            # as an unasserted one-slot S before the parent refers to it.
            # Do not invent a concept ID or infer one from its vector.
            needs_row = (relation is not None or not root['has_point']) and reference == -1
            if sentence or is_clause(item) or needs_row:
                child = recover(item)
                ref = ('clause', len(children))
                children.append(child)
                owned.add(id(item))
                return child.point, ref
            return value(item) if item['has_point'] else None, reference

        relation = operator if operator in relative else None
        operands = node['children']
        if (len(operands) == 2 and getattr(node['rule'], 'clause_form', None) == 'S'
                and generic(operands[0])):
            relation = 'part'
        if len(operands) == 2:
            left, left_ref = operand(operands[0], sentence=relation == 'implies')
            right, right_ref = None, -1
            predicate = None
            predicate_ref = -1
            if relation is not None:
                predicate_ref = operation_concept(node, relation)
                predicate = predicate_ref.point
            if predicate is None:
                vp = head(operands[1], clause=True)
                if len(vp['children']) == 2 and getattr(vp['rule'], 'clause_form', None) == 'VP':
                    verb = head(vp['children'][0])
                    predicate, predicate_ref = value(verb), concept(verb)
                    right, right_ref = operand(vp['children'][1])
                    role_nodes = operands[0], verb, head(vp['children'][1])
                else:
                    predicate, predicate_ref = value(vp), concept(vp)
                    right, right_ref = None, -1
                    role_nodes = operands[0], vp, None
            else:
                right, right_ref = operand(operands[1], sentence=relation == 'implies')
                role_nodes = operands[0], None, operands[1]
            if (any(child.relation is not None for child in children) or not root['has_point']) and relation is None:
                relation = 'operator'
                if right_ref == -1:
                    # A structural verb can itself take a completed truth.
                    # Keep both operands and the selected operation's atom.
                    right, right_ref = operand(operands[1])
                    predicate_ref = operation_concept(node)
                    predicate = predicate_ref.point
                    role_nodes = operands[0], None, operands[1]
            if relation is not None and predicate_ref == -1:
                # A lexical predicate over an ended relation can be an
                # unindexed phrase too. Give its point an unasserted LTM
                # occurrence, exactly as for either referenced operand.
                predicate, predicate_ref = operand(role_nodes[1])
            zero = torch.zeros_like(program.leaves[0])
            roles = torch.stack((zero if left is None else left,
                                 predicate.to(zero), zero if right is None else right))
            mask = torch.tensor([True, True, right is not None or right_ref != -1],
                                device=roles.device)
            described = ConceptualMeaning(roles, mask, polarity=polarity, mode=mode)
            references = ((left_ref, predicate_ref, right_ref) if relation is not None else
                          (left_ref, concept(operands[1]), -1))
            factored_refs = left_ref, predicate_ref, right_ref
            if relation is None and node.get('operand_refs') is not None:
                # Unknown composite addresses stay unknown. The temporary
                # factored target cannot invent a durable operand reference.
                references = tuple(ref if ref > 0 else -1 for ref in node['operand_refs']) + (-1,)
        else:
            described = replace(ConceptualMeaning.from_description(value(root)),
                                mode=mode, polarity=polarity)
            references = (node['ref'], -1, -1)
        equality = relation == 'equal'
        if equality:
            relation = 'part'
        if relation == 'whole':
            relation = 'part'
            references = references[2], references[1], references[0]
            factored_refs = references
            role_nodes = role_nodes[2], role_nodes[1], role_nodes[0]
            described = replace(described, roles=described.roles[[2, 1, 0]],
                                role_mask=described.role_mask[[2, 1, 0]])

        def retain_completed(part):
            pending = [part]
            while pending:
                part = pending.pop()
                if id(part) in owned:
                    continue
                selected = head(part)
                while len(selected['children']) == 1:
                    selected = head(selected['children'][0])
                if selected is not node and is_clause(part):
                    # A projection changes the enclosing NP's head, not the
                    # ownership of S events completed in its derivation.
                    children.append(recover(part))
                    owned.add(id(part))
                    continue
                pending.extend(reversed(part['children']))
        retain_completed(root)
        evidence = metadata(root['leaves'])
        if excluded:
            support = evidence['evidence']
            evidence['evidence'] = (0., support[1]) if polarity else (support[0], 0.)
        field = Clause(described, point=None if relation else (
            program.end_state[0] if top else value(root)), relation=relation,
            refs=references, children=tuple(children),
            subject_word_id=subject_word_id(role_nodes[0]),
            factored_refs=factored_refs,
            **evidence,
            eternal=relation is None and node['rule'] is None and node['ref'] > 0
            and program.symbol_when is None,
            **bands(root))
        if equality:
            converse = replace(field, refs=(field.refs[2], field.refs[1], field.refs[0]),
                               meaning=replace(field.meaning, roles=field.meaning.roles[[2, 1, 0]],
                                               role_mask=field.meaning.role_mask[[2, 1, 0]]),
                               factored_refs=None if field.factored_refs is None else field.factored_refs[::-1])
            field = replace(field, companions=(converse,))
        return field

    if len(stack) == 3:
        children, references = [], []
        roles = []
        for slot, node in enumerate(stack):
            node = head(node)
            if is_clause(node) or concept(node) == -1:
                references.append(('clause', len(children)))
                child = recover(node)
                children.append(child)
                roles.append(torch.zeros_like(
                    program.leaves[0]) if child.point is None else child.point)
            else:
                references.append(concept(node))
                roles.append(value(node))
        operation = None
        if type(references[1]) is int:
            selected = predicate_relation(references[1])
            if selected != 'operator':
                # A first predicate occurrence carries its value into the
                # writer, just as a selected binary grammar operation does.
                # It has no earlier inventory or LTM row to read.
                operation = selected
                predicate = operation_concept(stack[1], selected)
                references[1] = predicate
                roles[1] = predicate.point
        if operation is None and registry is not None:
            probe = meaning
            if probe is None:
                probe = ConceptualMeaning(torch.stack(roles), torch.ones(3, dtype=torch.bool),
                                          role_refs=tuple(('sym', ref) if type(ref) is int and ref > 0 else None for ref in references))
            if probe.role_refs[1] is not None:
                try:
                    operation = registry.signature_for(
                        replace(probe, mode='interrogative')).operation.semantic_id
                except ValueError:
                    pass
        if operation not in relative:
            if not any(child.relation is not None for child in children) and all(node['has_point'] for node in stack):
                raise ValueError('three slots have no selected relative clause')
            operation = 'operator'
        meaning = ConceptualMeaning(torch.stack(roles), torch.ones(3, dtype=torch.bool, device=program.leaves.device),
                                    mode="assertive" if meaning is None else meaning.mode,
                                    polarity=True if meaning is None else meaning.polarity)
        if operation == 'whole':
            operation = 'part'
            references = references[2], references[1], references[0]
        return Clause(meaning, relation=operation, refs=tuple(references), children=tuple(children),
                      **metadata(tuple(range(len(values)))),
                      **bands(dict(leaves=tuple(range(len(values))))))
    return recover(stack[0], top=True)
