"""Temporary grammar metadata and numerical values of an open reading.

Nothing from this journal is stored beside a completed row. The numerical
operators have already run in the word brick; finishing only assigns roles
and native references to their actual end-state contents.
"""

from dataclasses import replace

import torch

from ClauseRow import Clause, Composition, ClausePredicate, predicate_point, predicate_relation
from Meaning import ConceptualMeaning


def word_columns(program, refs, orders):
    """Read the fixed word addresses once for this formula's two consumers."""
    columns = (refs, program.concept_ids, program.word_ids,
               getattr(program, 'reference_relations', None), getattr(program, 'identity_leaves', None),
               orders, getattr(program, 'reference_orders', None),
               getattr(program, 'staged_object_rows', None), getattr(program, 'staged_object_ids', None))
    values = iter(torch.stack([column.reshape(-1) for column in columns if column is not None]).tolist())
    return tuple(next(values) if column is not None else None for column in columns)


def finish_clause(language, program, *, meaning=None, depth=1, registry=None):
    """Finish the read's existing journal without changing its numerical work."""
    reader = _finish_clause(language, program, meaning=meaning, depth=depth, registry=registry)
    while True:
        try:
            next(reader)
        except StopIteration as ended:
            return ended.value


class ComprehensionClose:
    """Consume each action's operands while comprehension holds them.

    The input program is the word formula. This reader never installs a
    numerical value or closing metadata in that formula's pinned tree.
    """
    def __init__(self, language, program, numerics, registry):
        self.reader = _finish_clause(language, program, registry=registry, numerics=numerics)
        self.action = next(self.reader)

    def consume(self, action):
        if self.action != action:
            raise ValueError('comprehension and its closing read different actions')
        self.action = next(self.reader)

    def finish(self):
        if self.action is not None:
            raise ValueError('comprehension closed before consuming the formula')
        try:
            next(self.reader)
        except StopIteration as ended:
            return ended.value
        raise RuntimeError('a completed comprehension did not return its clause')


def _finish_clause(language, program, *, meaning=None, depth=1, registry=None, numerics=None):
    """Finish an open reading from the values its operations already produced.

    The action record supplies grammar metadata only. Numerical values come
    from the live operation journal, including the operands before fusion.
    This function never runs a compose operator, and its reading record is
    not part of the returned field or the row written from it.
    """
    if program is None:
        raise ValueError('cannot finish a reading that is not open')
    from Language import TheGrammar
    from ReferenceContext import binding_spec, operation_modes, denoted_order, OrderReference
    from Interpret import leaf_readings
    interpretations = leaf_readings(language, program, len(program.leaves),
                                    forced=program.interpret_forced)
    def number(name):
        return getattr(program, name) if numerics is None else numerics.closing_value(name)
    orders = number('leaf_orders')
    if numerics is None and program.reference_orders is not None:
        orders = (program.reference_orders if orders is None else
                  torch.where(program.reference_orders >= 0, program.reference_orders, orders))
    # The carried address identifies the referent; the leaf keeps its own
    # captured content. Site eligibility comes from its reading and position.
    if numerics is None:
        refs = program.reference_ids if program.reference_ids is not None else program.concept_ids
        columns = word_columns(program, refs, orders)
    else:
        refs = numerics.native_ids
        columns = numerics.word_columns
    opening_refs, concepts, word_ids, relations, handles, initial_orders, requested_orders = columns[:7]
    native_ids = set(opening_refs) - {-1, 0}
    store_ref = getattr(getattr(registry, 'space', None), '_clause_store_ref', None)
    store = store_ref() if callable(store_ref) else None
    if store is not None:
        from Occurrence import operand_name
        native_ids.update(operand_name(store.row_name(row)) for row in range(len(store)))
    def native(reference):
        return (reference > 0 or reference in native_ids or
                (store is not None and store.slot_of_reference(reference) is not None))
    def reference_has_point(reference, default):
        if store is not None and store.location(reference) is not None:
            return store.point_of_row(reference) is not None
        return default
    def forward_name(leaf):
        # A word keeps its name when another occurrence supplies its object.
        # Pending and arriving references must use the same lexical key,
        # independently of the current word-to-object interpretation.
        if word_ids is not None and word_ids[leaf] > 0:
            return word_ids[leaf]
        concept = concepts[leaf]
        return concept if concept > 0 else None
    def forward_fields(leaf):
        name = forward_name(leaf)
        return {} if name is None else {'forward_identity': name}
    values = list(program.leaves.unbind()) if numerics is None else numerics.leaves
    frames = number('operation_values')
    if frames is None:
        raise ValueError('the open reading has no numerical operation journal')
    # The grammar still consumes integral event flags. Read their existing
    # columns together. Live references and denoted order come directly
    # from the operator that just chose them.
    events = (program.identity_actions, program.operation_relations,
              number('operation_refs'), program.timeline_actions)
    recorded = [value for value in events if value is not None]
    if recorded:
        rows = torch.cat(recorded, 1).tolist()
        columns, offset = [], 0
        for value in events:
            if value is None:
                columns.append(None)
            else:
                stop = offset + value.shape[1]
                columns.append([row[offset:stop] for row in rows])
                offset = stop
        identity_actions, relation_actions, reference_actions, timeline_actions = columns
    else:
        identity_actions = relation_actions = reference_actions = timeline_actions = None
    origins = (program.identity_origins[:, :2].tolist()
               if identity_actions and program.identity_origins is not None else None)
    stack, introductions = [], {}
    local_uses = set()
    if program.identity_actions is not None:
        # Subject, object and selected bind read the same handle table. Carry
        # their actual introducing shift or grammar action into this close.
        for _, subject, obj, _, bound, _ in identity_actions:
            local_uses.update(h for h in (subject, obj, bound)
                              if h > 0 and origins[h - 1][0] in (3, 4))
    def region(node):
        """Read the declared meaning without replaying numerical operators."""
        if not node['has_point']:
            return None
        while node['children']:
            semantic = getattr(node['rule'], 'semantic_operand', None)
            if semantic is None:
                break
            node = node['children'][semantic]
            if not node['has_point']:
                return None
        return node['value']
    def input_evidence(words):
        # These are input identification poles, never the source's authority.
        evidence = program.leaf_evidence
        if evidence is None:
            return (0., 0.)
        selected = evidence[list(words)]
        known = selected > 0
        minimum = torch.where(known, selected, torch.inf).amin(0)
        return tuple(torch.where(known.any(0), minimum, 0.).detach().tolist())

    def metadata(words):
        # A closed field has a percept point even when a leaf concept is
        # unwitnessed; only this field schema coerces its absent order to 0.
        order = 0 if orders is None else max(0, int(orders[list(words)].max()))
        return dict(order=order, evidence=input_evidence(words))
    from ParseForest import native, visits
    parse = native(program)
    nodes = {}
    # Each node owns the chooser's ordered operand references.
    for action, (kind, local, word) in visits(program):
        if kind < 0:
            break
        yield action
        if parse is not None:
            stack = [nodes[child] for child in parse.operands[action]]
        if kind == 0:
            if numerics is None:
                reference = opening_refs[word]
                order = -1 if initial_orders is None else initial_orders[word]
            else:
                # Comprehension has just pushed this operand. Its live identity
                # and order already include this shift's binding choice.
                operand = numerics.stack[-1]
                reference = numerics.native_leaf_reference(operand.reference, concepts[word])
                order = operand.order
            has_point = reference_has_point(reference, relations is None or not relations[word])
            # This closing can outlive its trial lease. Snapshot the one
            # selected row for all its readers before another trial reuses it.
            point = values[word] if numerics is None else values[word].clone()
            stack.append(dict(value=point, composed_value=point, ref=reference, rule=None,
                              order=order,
                              children=(), leaves=(word,), has_point=has_point,
                              interpretation='no_op' if interpretations[word] == 'no_op' else 'symbol'))
            handle = 0 if handles is None else handles[word]
            addressed = handle in local_uses and origins[handle - 1][0] == 4
            owner = getattr(registry, 'space', None)
            # An objectless lexical reference remains a binding request at
            # order zero. Its physical word row does not stand in for the
            # order of an object which has not been resolved yet.
            unbound = owner is not None and owner.interpret.no_object(reference)
            if stack[-1]['order'] == 1 or addressed or unbound:
                stack[-1]['close_request'] = dict(mode='bind', leaves=(word,),
                    cue=point, content=point,
                    **forward_fields(word), leaf=True)
                if unbound:
                    stack[-1]['close_request']['interpret_leaf'] = True
                if addressed:
                    stack[-1]['close_request']['addressed_introduction'] = True
            if program.identity_leaves is not None:
                stack[-1]['identity_handle'] = handle if stack[-1]['order'] == 1 or addressed or unbound else 0
                if stack[-1]['identity_handle']:
                    introductions.setdefault(stack[-1]['identity_handle'], stack[-1])
            if interpretations[word] == 'no_op':
                stack[-1].update(order=0, has_point=True)
                stack[-1]['close_request'] = dict(mode='mint', no_op_leaf=True, leaves=(word,),
                    cue=None, content=point, percept=point)
            nodes[action] = stack[-1]
            continue
        if kind == 3:
            stack[-2:] = reversed(stack[-2:])
            continue
        unary_position = max(0, word) if kind == 2 and parse is None else 0
        if unary_position:
            stack.append(stack.pop(-1-unary_position))
        arity = 2 if kind == 1 else 1
        catalog = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
        if kind == 1 and 0 <= local < len(catalog) and catalog[local].method_name == 'bind':
            arity = 1  # the right field operand is off the parse stack
        if kind not in (1, 2) or not 0 <= local < len(catalog) or len(stack) < arity:
            raise ValueError('invalid clause derivation')
        operands = tuple(stack[-arity:])
        del stack[-arity:]
        selected_refs = None if reference_actions is None else reference_actions[action].copy()
        if numerics is None:
            frame = frames[action]
        else:
            frame, operand_refs, denoted = numerics.current_operation
            selected_refs[:len(operand_refs)] = operand_refs
        relation_rule = bool(getattr(catalog[local], 'relation_kind', None))
        for role, (operand, actual) in enumerate(zip(operands, frame[:arity])):
            operand['value'] = actual
            if program.operation_relations is not None:
                operand['has_point'] = not bool(relation_actions[action][role])
            if selected_refs is not None:
                selected = operand
                while selected['children']:
                    headed = getattr(selected['rule'], 'head_role', 0)
                    if not headed:
                        break
                    selected = selected['children'][headed-1]
                reference = selected_refs[role]
                rule = catalog[local]
                mode = operation_modes(rule).get('argin'+str(role+1))
                binding_order, binding_mode = binding_spec(
                    rule, 'argin'+str(role+1), operand['order'], implicit=True)
                binding = (binding_order is not None and binding_order >= 0
                           and binding_mode not in ('kind', 'generic'))
                chosen_antecedent = (native(reference) and reference != selected['ref'])
                if program.identity_actions is not None:
                    # Implicit non-individual operands already used their
                    # typed concept in compose. Keep that reference beside
                    # the journal's point without reviving formation choices.
                    if binding and (binding_order != 1 or getattr(rule, 'order_delta', 0) == -1) and chosen_antecedent:
                        selected['ref'] = reference
                    continue
                if (getattr(program, 'close_references', False) and binding_order == 1
                        and binding and not chosen_antecedent):
                    # An enclosing operation cannot re-bind a deferred
                    # individual. Only its mint/bind/pronoun operator supplies
                    # the request; the close decides it with the predicate.
                    continue
                if (getattr(program, 'binding_choices', False) and binding
                        and program.operation_probabilities is not None):
                    formation = dict(choice='open' if reference == 0 else 'mint' if reference == -1 else 'bind',
                        probability=float(program.operation_probabilities[action]), reference=reference)
                    selected['formation'] = operand['formation'] = formation
                if reference == 0 or (reference == -1 and mode in ('bind', 'pronoun')):
                    selected['ref'] = -1
                    selected['open_reference'] = True
                    operand['open_reference'] = True
                    selected.update(forward_fields(selected['leaves'][0]))
                    request = selected.get('close_request', {})
                    if request.get('leaf') and not request.get('addressed_introduction'):
                        # A selected open formation already owns the role.
                        # Keep its cue for thought without adding a second NP
                        # closing decision or a second resolver cost.
                        selected['reference_site'] = selected.pop('close_request')
                elif reference == -1 and getattr(program, 'binding_choices', False):
                    if binding and not operand.get('open_reference'):
                        selected['mint_reference'] = operand['mint_reference'] = True
                        selected['mint_order'] = operand['mint_order'] = binding_order
                        # The operation's explicit formation owns this site.
                        # A provisional word request cannot override its mint.
                        for item in (selected, operand):
                            provisional = item.get('close_request', {})
                        if provisional.get('leaf') and not provisional.get('addressed_introduction'):
                                item.pop('close_request')
                # A trial's newly requested singleton still had its source
                # address. Resolve that alias to the winning host's admitted
                # identity; an already selected earlier occurrence is unchanged.
                pending_order = False
                if not selected['children'] and program.reference_orders is not None:
                    leaf = selected['leaves'][0]
                    pending_order = (requested_orders[leaf] >= 0
                                     and reference == concepts[leaf]
                                     and int(refs[leaf]) not in (-1, 0))
                    if pending_order:
                        reference = int(refs[leaf])
                if native(reference) or pending_order:
                    selected['ref'] = reference
                    # A later bond can resolve an earlier open operand. The
                    # selected native reference, not a stale journal flag,
                    # determines whether the closing is still a question.
                    for item in (selected, operand):
                        item.pop('open_reference', None)
                        item.pop('forward_identity', None)
                        item.pop('mint_reference', None)
                        request = item.get('close_request', {})
                        if (request.get('leaf') and not request.get('addressed_introduction')
                                and (reference != request['forward_identity']
                                     or item.get('formation', {}).get('choice') == 'bind')):
                            item.pop('close_request')
                    selected_refs[role] = reference
                    # A headed VP may retain a point-valued head while one
                    # of its operands is a relation. Transparent enclosing
                    # operations carry that relative field in their journal;
                    # looking up its head must not flatten the whole field.
                    operand['has_point'] = (operand['has_point']
                        and reference_has_point(reference, True))
        parent = dict(action=action, value=frame[2], composed_value=frame[2], local=local, ref=-1,
                          order=denoted_order(catalog[local], tuple(child['order'] for child in operands),
                              tuple(OrderReference(child['ref'], child.get('timeline', False)) for child in operands), None),
                          rule=catalog[local], children=operands,
                          operand_refs=None if selected_refs is None else tuple(selected_refs),
                          leaves=tuple(i for child in operands for i in child['leaves']),
                          has_point=not relation_rule and (all(child['has_point'] for child in operands)
                              or getattr(catalog[local], 'clause_form', None) in ('S', 'VP')))
        if program.identity_actions is not None:
            record = identity_actions[action]
            parent['identity_handle'] = record[0]
            parent['identity_event'] = tuple(record)
            parent['identity_action'] = action
        if timeline_actions is not None:
            stamp, alternative, origin, suppressed = timeline_actions[action]
            if numerics is not None:
                stamp = denoted
            parent.update(order=stamp, timeline=bool(alternative), event_origin=origin-1,
                          suppressed_event=bool(suppressed), timeline_order=True)
        if getattr(program, 'close_references', False):
            rule = catalog[local]
            for role in getattr(rule, 'reference_roles', ()):
                index = int(role[5:]) - 1
                order, mode = binding_spec(rule, role, operands[index]['order'])
                if order == 1 and mode in ('mint', 'bind', 'pronoun'):
                    description = operands[index]
                    cue = operands[1-index] if arity == 2 else None
                    content = region(description)
                    if content is not None:
                        parent['close_request'] = dict(mode=mode, leaves=description['leaves'],
                            cue=None if cue is None else region(cue), content=content)
        if catalog[local].method_name == 'interpret':
            parent['interpretation'] = 'symbol'
            parent['ref'] = operands[0]['ref'] if selected_refs is None else selected_refs[0]
            owner = getattr(registry, 'space', None)
            if owner is not None:
                parent['order'] = owner._concept_source_order(parent['ref'])
                if owner.interpret.no_object(parent['ref']):
                    parent['close_request'] = dict(mode='bind', interpret_leaf=True, leaf=True,
                        **forward_fields(parent['leaves'][0]), leaves=parent['leaves'],
                        cue=parent['value'], content=parent['value'])
        if catalog[local].method_name == 'sign':
            leaf, = operands
            while leaf['children'] and leaf['rule'].method_name == 'interpret':
                leaf, = leaf['children']
            if leaf['children'] or not leaf['leaves']:
                raise ValueError('sign requires an interpreted leaf')
            leaf_request = operands[0].pop('close_request', {})
            if leaf is not operands[0]:
                leaf_request.update(leaf.pop('close_request', {}))
            parent.update(order=0, interpretation='sign')
            percept = (leaf['value'] if word_ids is None or word_ids[leaf['leaves'][0]] < 0
                       else None)
            content = torch.zeros_like(leaf['value']) if percept is None else percept
            parent['close_request'] = dict(mode='mint', sign_leaf=True, leaves=leaf['leaves'],
                cue=None, content=content, percept=percept)
            if (leaf_request.get('addressed_introduction') and
                    parent.get('identity_handle') == leaf.get('identity_handle')):
                parent['close_request']['addressed_introduction'] = True
                introductions[parent['identity_handle']] = parent
        if program.identity_actions is not None and catalog[local].method_name == 'bind':
            # The chooser already made this choice while the reading was live.
            # Retain the own description and the pre-join record independently.
            source = operands[0]
            request = dict(source.get('close_request', {}))
            request.pop('introduction', None)
            request.pop('addressed_introduction', None)
            request.update(mode='bind', leaves=source['leaves'],
                content=request.get('content', frame[0]),
                cue=request.get('cue', frame[0]),
                compose_bind=identity_actions[action][4],
                record=frame[1])
            parent['close_request'] = request
        if program.identity_actions is not None:
            handle, subject, _, _, selected, _ = parent['identity_event']
            if handle > 0 and not subject and selected <= 0:
                tag, source = origins[handle - 1]
                addressed = handle in local_uses and tag == 3 and source == action
                if tag == 3 and (addressed or parent['order'] == 1 or catalog[local].method_name == 'bind'):
                    if addressed:
                        introductions[handle] = parent
                    else:
                        introductions.setdefault(handle, parent)
                    if introductions[handle] is parent and (addressed or not parent['has_point']):
                        request = dict(parent.get('close_request', {}))
                        if not parent['has_point'] or not request:
                            rule = catalog[local]
                            description = operands[(getattr(rule, 'head_role', 0) or 1) - 1]
                            request.update(mode=getattr(rule, 'determiner_mode', None) or 'bind',
                                leaves=description['leaves'], content=frame[2], compose_bind=0)
                            if not parent['has_point']:
                                request.update(introduction=True, determined=True)
                        if addressed:
                            request['addressed_introduction'] = True
                        parent['close_request'] = request
        stack.append(parent)
        nodes[action] = parent
        if unary_position:
            stack.insert(len(stack)-1-unary_position, stack.pop())
    if parse is not None:
        stack = [nodes[root] for root in parse.roots]
    yield None
    if len(stack) not in (1, 3):
        raise ValueError('a completed field must have one or three slots')

    def subject_word_id(node):
        leaf = head(node)['leaves'][0]
        if word_ids is not None:
            return word_ids[leaf]
        if registry is None:
            return -1
        obj = registry.space.concept_id_at_row(int(program.word_rows[leaf]))
        return (-1 if obj is None else registry.space.word_concept_of_object(obj) or -1)

    def value(node):
        return node['value']

    def identity_head(node):
        """The outer event attachment owns scope; semantic roles stay at S."""
        while node['children']:
            event = node.get('identity_event')
            if event is not None and event[1] and not node.get('suppressed_event'):
                return node
            role = getattr(node['rule'], 'head_role', 0)
            if not role and len(node['children']) == 1 and getattr(node['rule'], 'scope_transparent', False):
                role = 1
            if not role:
                break
            node = node['children'][role-1]
        return node

    action_nodes = {}
    pending = list(stack)
    while pending:
        node = pending.pop()
        if node.get('identity_action') is not None:
            action_nodes[node['identity_action']] = node
        pending.extend(node['children'])

    def identity_record(node):
        if program.identity_actions is None:
            return None
        action = node.get('identity_action')
        semantic = action_nodes.get(node.get('event_origin'), node)
        basis = None
        if action is not None and getattr(semantic['rule'], 'clause_form', None) == 'S':
            vp = semantic['children'][1]
            while vp['children'] and getattr(vp['rule'], 'clause_form', None) != 'VP':
                head_role = getattr(vp['rule'], 'head_role', 0)
                if not head_role:
                    break
                vp = vp['children'][head_role-1]
            if getattr(vp.get('rule'), 'clause_form', None) == 'VP':
                basis = vp['children'][1]['value']
        modifiers = []
        def attached(current):
            if not current['children']:
                return
            rule = current['rule']
            if rule.method_name == 'modifier':
                attached(current['children'][0])
                modifiers.append(frames[current['identity_action'],1].clone())
            elif current is semantic and getattr(rule, 'clause_form', None) == 'S':
                attached(current['children'][1])
            elif getattr(rule, 'head_role', 0) and current is not semantic:
                attached(current['children'][rule.head_role-1])
        if node.get('timeline'):
            attached(node)
        source_action = semantic.get('identity_action', action)
        return dict(handle=node.get('identity_handle', 0),
            event=node.get('identity_event'), action=action,
            event_order=source_action,
            introduction=introductions.get(node.get('identity_handle', 0)) is node,
            result_composition=None if action is None else composition(
                node['children'][1] if getattr(node['rule'], 'relation_kind', None) == 'equal' else node),
            result=None if action is None else frames[action,
                1 if getattr(node['rule'], 'relation_kind', None) == 'equal' else 2].clone(),
            before=None if source_action is None else frames[source_action, 0].clone(), basis=basis,
            verb_meaning=None if source_action is None else frames[source_action, 1].clone(),
            operation=None if semantic['rule'] is None else semantic['rule'].method_name,
            alternative=node.get('timeline', False), modifiers=tuple(modifiers),
            timeline_order=number('timeline_actions') is not None,
            timeline_values=number('timeline_values'),
            case_value=(number('case_values')[action] if action is not None and number('case_selected') is not None
                       and bool(number('case_selected')[action]) else None),
            leaf_count=len(node['leaves']),
            origins=program.identity_origins, ids=program.identity_ids, values=number('identity_values'))

    composition_cache = {}
    def composition(node):
        pending = [(node, False)]
        while pending:
            current, ready = pending.pop()
            key = id(current)
            if key in composition_cache:
                continue
            if not ready:
                pending.append((current, True))
                pending.extend((child, False) for child in reversed(current['children']))
                continue
            rule = current['rule']
            from PartsOfSpeech import operation_name
            reference = int(current.get('forward_identity', current['ref'])
                if current['ref'] in (-1, 0) else current['ref'])
            event = current.get('identity_event')
            if event is not None and rule.method_name == 'bind' and event[4] > 0:
                # This is the chooser's selected identity, already used by
                # the numerical journal. Its lexical child is only the cue.
                origin = origins[event[0]-1]
                if int(origin[0]) in (1, 2):
                    reference = int(origin[1])
            composition_cache[key] = Composition(
                point=current['composed_value'],
                children=tuple(composition_cache[id(child)] for child in current['children']),
                reference=reference,
                operation=None if rule is None else rule.method_name,
                head_role=0 if rule is None else getattr(rule, 'head_role', 0),
                # The occurrence itself is an order-0 percept if its source
                # concept remains unknown; no allocator record is assigned.
                order=max(0, int(current.get('mint_order', current['order']))),
                leaves=current['leaves'],
                word_id=subject_word_id(current) if not current['children'] or
                    getattr(rule, 'method_name', None) == 'interpret' else -1,
                formal_name=None if rule is None else operation_name(rule),
                interpretation=current.get('interpretation'))
            if numerics is not None:
                composition_cache[key] = numerics.composition(composition_cache[key],
                    action=current.get('action'), word=current['leaves'][0] if rule is None else None)
            elif program.tree is not None:
                composition_cache[key] = program.tree.install_composition(composition_cache[key],
                    action=current.get('action'),
                    word=current['leaves'][0] if rule is None else None)
        return composition_cache[id(node)]

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

    relative = {'part', 'whole', 'equal', 'implies'}
    absolute_starts = set(TheGrammar.ws_absolute_starts)

    def head(node, *, clause=False):
        while node['children']:
            if (getattr(node['rule'], 'clause_form', None) == 'S'
                    or getattr(node['rule'], 'relation_kind', None) is not None):
                break  # a clause's declared head does not erase its subject
            if clause and getattr(node['rule'], 'clause_form', None) == 'VP':
                # A headed modifier of an already completed sentence keeps
                # that sentence's region. Only a lexical VP is the boundary
                # at which its verb and object must be recovered separately.
                content = head(node['children'][0])
                if (getattr(content['rule'], 'clause_form', None) != 'S'
                        and getattr(content['rule'], 'relation_kind', None) is None):
                    break
            role = getattr(node['rule'], 'head_role', 0)
            if not role:
                break
            node = node['children'][role - 1]
        return node

    def generic(node):
        while True:
            determiner = getattr(node['rule'], 'determiner_mode', None)
            if determiner is not None:
                return determiner == 'kind'
            requests = operation_modes(node['rule'])
            if any(mode in ('generic', 'kind') for mode in requests.values()):
                return True
            if any(mode in ('particular', 'name', 'pronoun') for mode in requests.values()):
                return False
            selected = head(node)
            if selected is node:
                return False
            node = selected

    def concept(node):
        if getattr(node['rule'], 'determiner_mode', None) == 'bind':
            chosen = node.get('operand_refs')
            if chosen is None or chosen[1] in (-1, 0):
                return -1
            return chosen[1]
        selected = head(node)
        if selected['ref'] not in (-1, 0):
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

    from ReferenceParticipants import participant_terms
    noun_terms, _ = participant_terms(((composition(node), len(stack) == 3 and slot in (0, 2))
        for slot, node in enumerate(stack)),
        (*language._compose_binary_rules, *language._compose_unary_rules))

    def node_order(item):
        # The completed-field schema describes an unknown source as a
        # percept at zero. This does not allocate or change its concept order.
        return max(0, int(item['order']))

    def recover(root, *, top=False):
        # Ask belongs to the kept head, including the lexical early returns.
        # An ask inside an operand does not erase its enclosing assertion.
        current, asked = root, False
        while True:
            rule = current['rule']
            asked = asked or getattr(rule, 'method_name', None) == 'ask'
            if not current['children'] or getattr(rule, 'clause_form', None) == 'S' or getattr(rule, 'relation_kind', None) is not None:
                break
            role = getattr(rule, 'head_role', 0)
            if not role and len(current['children']) == 1:
                role = 1
            if not role:
                break
            current = current['children'][role - 1]
        result = recover_content(root, top=top)
        if asked:
            from ThoughtReferences import bindings, with_slots
            result = replace(result, evidence=(0., 0.), meaning=with_slots(result.meaning,
                bindings(result.meaning).get('_open_references', ()), pair=(0., 0.),
                source_evidence=(0., 0.)))
        return result

    def recover_content(root, *, top=False):
        # Syntactic assertion polarity is a grammar property. Identification
        # evidence remains a pair, including both and neither, through storage.
        polarity = True
        excluded = False
        def close_member(item, *, referent=True):
            own_request = item.get('close_request', {})
            if not item['has_point'] and not (own_request.get('introduction') or own_request.get('addressed_introduction')
                    or 'compose_bind' in own_request and (
                        own_request['compose_bind'] > 0 or own_request.get('determined'))):
                # A dummy bind over a relative field preserves that field.
                # Only an actual selected object or a retained determination
                # supplies a point-valued reference request around it.
                return None
            current = item
            request = current.get('close_request')
            while request is None and current['children']:
                rule = current['rule']
                if getattr(rule, 'clause_form', None) in ('S', 'VP'):
                    break
                role = getattr(rule, 'head_role', 0)
                if not role and len(current['children']) == 1 and getattr(rule, 'scope_transparent', False):
                    role = 1
                if not role:
                    break
                current = current['children'][role - 1]
                request = current.get('close_request')
            if request is None or (item['order'] != 1
                    and 'compose_bind' not in request and not request.get('sign_leaf')
                    and not request.get('no_op_leaf') and not request.get('interpret_leaf')
                    and not request.get('addressed_introduction')):
                return None
            if program.identity_actions is not None and not (request.get('sign_leaf') or request.get('no_op_leaf')):
                request = dict(request, compose_bind=request.get('compose_bind', 0))
            if (request.get('leaf') and not request.get('addressed_introduction')
                    and (not referent or id(composition(item)) not in noun_terms)):
                # A predicate's form is an operator operand, not an
                # participant. A provisional marker likewise
                # cannot become an occurrence object merely by ending a
                # field. Keep the kept derivation's raw identity/content.
                return None
            word = head(item)
            leaf_request = word.get('close_request', {})
            forward = word.get('forward_identity', leaf_request.get('forward_identity'))
            if forward is not None and request['mode'] != 'mint':
                request = dict(request, forward_identity=forward)
                if leaf_request.get('leaf'):
                    # A unary bind still owns the same word site. Keep its
                    # name and content when wrapping the leaf.
                    request.update(content=leaf_request['content'],
                                   cue=leaf_request.get('cue'))
            point = request['content'] if current is item or request.get('leaf') else region(item)
            if point is None:
                return None
            relative = request.get('introduction', False)
            description = item['children'][(getattr(item['rule'], 'head_role', 0) or 1)-1] if relative else None
            return Clause(ConceptualMeaning.from_description(point, sentence_kind="idea"), point=point,
                subject_word_id=-1 if request.get('determined') else subject_word_id(item), order=node_order(item),
                children=(recover(description),) if relative else (),
                evidence=input_evidence(item['leaves']), **bands(item),
                reference_request=request, composition=composition(item), identity=identity_record(item))
        standalone = close_member(root)
        if standalone is not None:
            return standalone
        node = head(root, clause=True)
        while len(node['children']) == 1:
            if getattr(node['rule'], 'polarity_effect', None) == 'invert':
                polarity = not polarity
            if getattr(node['rule'], 'polarity_effect', None) == 'exclude':
                excluded = True
            node = head(node['children'][0], clause=True)
        if not node['children'] and not node['has_point'] and store is not None:
            index = store.location(node['ref'])
            retained = None if index is None else store.meaning_of(index)
            if (isinstance(index, tuple) and retained is not None
                    and int(retained.role_mask.sum()) == 3):
                row = store.row(index)
                relation = 'implies' if 'conditional' in row.get('readings', ()) else 'part'
                refs = row['refs'].tolist()
                refs = (refs[0], ClausePredicate(predicate_point(relation, retained.roles[1]), relation), refs[1])
                return Clause(replace(retained, polarity=polarity), relation=relation,
                    refs=refs, subject_word_id=subject_word_id(node),
                    **metadata(root['leaves']), **bands(root))
        operator = getattr(node['rule'], 'relation_kind', None)
        children = []
        owned = set()
        factored_refs = None
        role_nodes = (node, None, None)

        def determined_order(item):
            description = item['children'][item['rule'].head_role - 1]
            if requested_orders is not None:
                selected = max((requested_orders[i] for i in description['leaves']), default=-1)
                if selected >= 0:
                    return selected
            return max(0, item['order'])

        def member(item, *, referent=True):
            ended = close_member(item, referent=referent)
            if ended is not None:
                return ended
            mode = getattr(item['rule'], 'determiner_mode', None)
            if mode != 'mint' and not item.get('mint_reference'):
                return None
            description = (item['children'][item['rule'].head_role - 1]
                           if mode == 'mint' else head(item))
            meta = dict(evidence=input_evidence(description['leaves']),
                        order=determined_order(item) if mode == 'mint' else item['mint_order'])
            formed = ConceptualMeaning.from_description(value(item), sentence_kind="idea")
            formation = item.get('formation') or head(item).get('formation')
            if formation is not None:
                formed = replace(formed, bindings={'_formation_records': (dict(formation, role=0),)})
            return Clause(formed, point=value(item), composition=composition(item), minted=True,
                          subject_word_id=subject_word_id(description), identity=identity_record(item), **bands(item), **meta)

        def operand(item, *, sentence=False, referent=True):
            ended = close_member(item, referent=referent)
            if ended is not None:
                ref = ('clause', len(children))
                children.append(ended)
                owned.add(id(item))
                return ended.point, ref
            if item.get('open_reference') or head(item).get('open_reference'):
                return value(item), -1
            if sentence or is_clause(item):
                # A composed statement owns its constituent bindings. A
                # later mint of the enclosing operand cannot flatten away
                # a pending variable (or its already completed children).
                child = recover(item)
                ref = ('clause', len(children))
                children.append(child)
                owned.add(id(item))
                return child.point, ref
            fresh = member(item, referent=referent)
            if fresh is not None:
                ref = ('clause', len(children))
                children.append(fresh)
                owned.add(id(item))
                return fresh.point, ref
            reference = concept(item) if not is_clause(item) else None
            # An unindexed numerical NP (e.g. an embedding input) still
            # has a field. If a relative S needs its address, close that NP
            # as an unasserted one-slot S before the parent refers to it.
            # Do not invent a concept ID or infer one from its vector.
            needs_row = (relation is not None or not root['has_point']) and reference == -1
            if needs_row:
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
            left, left_ref = operand(operands[0], sentence=relation == 'implies',
                                    referent=getattr(node['rule'], 'clause_form', None) != 'VP')
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
            if relation is not None and predicate_ref == -1:
                # A lexical predicate over an ended relation can be an
                # unindexed phrase too. Give its point an unasserted LTM
                # occurrence, exactly as for either referenced operand.
                predicate, predicate_ref = operand(role_nodes[1])
            zero = torch.zeros_like(program.leaves[0])
            if predicate is None:
                # A relative child supplies its address and has no fused point.
                if not (isinstance(predicate_ref, tuple) and predicate_ref[0] == 'clause'
                        and children[predicate_ref[1]].relation is not None):
                    raise ValueError(f'clause predicate has neither a point nor a relative child: {predicate_ref!r}')
            roles = torch.stack((zero if left is None else left,
                                 zero if predicate is None else predicate.to(zero),
                                 zero if right is None else right))
            occupied = (True, True, right is not None or right_ref != -1)
            mask = torch.ones(3, dtype=torch.bool, device=roles.device)
            mask[2] = occupied[2]
            described = ConceptualMeaning(roles, mask, polarity=polarity,
                                          sentence_kind='relation' if relation else 'idea')
            references = ((left_ref, predicate_ref, right_ref) if relation is not None else
                          (left_ref, concept(operands[1]), -1))
            factored_refs = left_ref, predicate_ref, right_ref
            if relation is None and node.get('operand_refs') is not None:
                # Unknown composite addresses stay unknown. The temporary
                # factored target cannot invent a durable operand reference.
                retained = []
                for position, (previous, ref, item) in enumerate(zip(
                        references, node['operand_refs'], operands)):
                    requested, _ = binding_spec(node['rule'], 'argin'+str(position+1),
                                                 item['order'], implicit=True)
                    literal = not item['children'] and requested is None
                    # A non-reference operand has no binding choice in the
                    # journal. Its null choice must not erase a known raw
                    # identity, such as the predicate of an ended VP.
                    retained.append(previous if isinstance(previous, tuple) or literal
                                    else ref if ref not in (-1, 0) else -1)
                references = (*retained, -1)
            # The predicate comes from the kept structure. Equality's
            # grammar predicate is a copula, so its complement supplies the
            # content; a lexical VP contributes its actual head instead.
            context_predicate = (right if operator == 'equal' else
                                 predicate if operator is None else None)
            contextual = []
            for child in children:
                request = child.reference_request
                if request is not None:
                    index = children.index(child)
                    role = 2 if right_ref == ('clause', index) else 0
                    child = replace(child, reference_request=dict(request, role=role,
                        predicate=context_predicate))
                contextual.append(child)
            children = contextual
        else:
            point = value(root)
            described = ConceptualMeaning.from_description(point, polarity=polarity, sentence_kind="idea")
            depth = 1 if point.ndim == 1 else point.shape[0]
            occupied = tuple(role < depth for role in range(3))
            references = (node['ref'], -1, -1)
        equality = relation == 'equal'
        if equality:
            relation = 'part'
            from ThoughtReferences import bindings
            data = bindings(described)
            data['_equality'] = True
            described = replace(described, bindings=data)
        if relation == 'whole':
            relation = 'part'
            references = references[2], references[1], references[0]
            factored_refs = references
            role_nodes = role_nodes[2], role_nodes[1], role_nodes[0]
            occupied = occupied[::-1]
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
                if part.get('suppressed_event'):
                    pending.extend(reversed(part['children']))
                    continue
                if selected is not node and is_clause(part):
                    # A projection changes the enclosing NP's head, not the
                    # ownership of S events completed in its derivation.
                    children.append(recover(part))
                    owned.add(id(part))
                    continue
                pending.extend(reversed(part['children']))
        retain_completed(root)
        # non retains the selected modifier identity beside its cleared
        # evidence. The inverse and next prior can distinguish it from absence.
        withheld, pending = [], [root]
        while pending:
            part = pending.pop()
            if getattr(part['rule'], 'polarity_effect', None) == 'exclude' and part['children']:
                operand = part['children'][0]
                withheld.append((int(operand['ref']), tuple(operand['value'].detach().cpu().tolist())))
            pending.extend(part['children'])
        if withheld:
            from ThoughtReferences import bindings
            described = replace(described, bindings=dict(bindings(described), _withheld_symbols=tuple(withheld)))
        evidence = dict(order=node_order(root), evidence=input_evidence(root['leaves']))
        formations = tuple(dict(formation, role=role) for role, item in enumerate(role_nodes)
            if item is not None and (formation := item.get('formation') or head(item).get('formation')) is not None)
        if formations:
            from ThoughtReferences import bindings
            data = bindings(described)
            data['_formation_records'] = formations
            described = replace(described, bindings=data)
        from ThoughtReferences import slot_bindings, with_slots
        opened = tuple(('relation' if role == 1 else 'referent', role)
                       for role, item in enumerate(role_nodes) if item is not None
                       and (item.get('open_reference') or head(item).get('open_reference')))
        resolver = getattr(getattr(getattr(registry, 'space', None), 'components', None),
                           'reference_close', None)
        if resolver is not None and occupied[1]:
            opened = tuple(dict.fromkeys((*opened, *(('referent', role) for role in
                resolver.expected_roles(described.roles[1]) if not occupied[role]))))
        if opened:
            data = slot_bindings(described, opened, pair=evidence['evidence'])
            data['_forward_references'] = tuple((role, name) for role, name in ((role,
                head(role_nodes[role]).get('forward_identity',
                    forward_name(head(role_nodes[role])['leaves'][0])))
                for kind, role in opened if kind == 'referent' and role_nodes[role] is not None)
                if name is not None)
            sites = dict(data.get('_reference_sites', ()))
            for kind, role in opened:
                if kind != 'referent' or role_nodes[role] is None:
                    continue
                request = head(role_nodes[role]).get('reference_site')
                if request is not None:
                    sites[role] = dict(role=role, leaves=request['leaves'],
                        content=request['content'].detach().cpu().tolist(),
                        cue=request['cue'].detach().cpu().tolist(),
                        predicate=described.roles[1].detach().cpu().tolist()
                            if occupied[1] else None)
            if sites:
                data['_reference_sites'] = tuple(sites.items())
            described = replace(described, bindings=data)
            if relation is None and sum(occupied) > 1:
                # Pending roles live in MTM; the durable entry is the fused idea.
                references = factored_refs or references
                if (references[1] == -1 and occupied[1]
                        and ('relation', 1) not in opened):
                    # An observed predicate can lack a dictionary identity.
                    # Ending the pending row must retain that observed point,
                    # not turn it into an additional unknown relation.
                    refs_with_predicate = list(references)
                    refs_with_predicate[1] = ('clause', len(children))
                    predicate = described.roles[1]
                    children.append(Clause(ConceptualMeaning.from_description(predicate, sentence_kind="idea"),
                        point=predicate, order=0))
                    references = tuple(refs_with_predicate)
                    factored_refs = references
                missing = tuple(('relation' if role == 1 else 'referent', role)
                                for role in range(3) if not occupied[role])
                described = with_slots(replace(described,
                    role_mask=torch.ones_like(described.role_mask)), (*opened, *missing),
                    pair=evidence['evidence'])
        if excluded:
            support = evidence['evidence']
            evidence['evidence'] = (0., 0.)
        field = Clause(described, point=None if relation else (
            number('end_state')[0] if top else value(root)), relation=relation,
            refs=references, children=tuple(children),
            subject_word_id=subject_word_id(role_nodes[0]),
            factored_refs=factored_refs, composition=composition(root),
            identity=identity_record(identity_head(root)),
            **evidence,
            **bands(root))
        return field

    def retain_introductions(result):
        if program.identity_actions is None:
            return result
        ordered, seen = [], set()
        def postorder(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for child in node['children']:
                postorder(child)
            ordered.append(node)
        for node in stack:
            postorder(node)
        owners, events, required = set(), set(), set()
        def collect(clause):
            if clause.identity and (event := clause.identity.get('event')) is not None and event[1]:
                events.add(clause.identity['action'])
            request = clause.reference_request or {}
            if clause.identity and request:
                required.add(clause.identity['handle'])
                if request.get('compose_bind', 0) <= 0 and clause.identity.get('introduction', True):
                    owners.add(clause.identity['handle'])
            for child in (*clause.children, *clause.companions):
                collect(child)
        collect(result)
        # The verb is a symbol, not an overlay handle.
        required.update(int(event[index]) for node in ordered
                    if (event := node.get('identity_event')) is not None
                    for index in (1, 2, 4) if int(event[index]) > 0)
        extra = []
        for node in ordered:
            handle, request = node.get('identity_handle', 0), node.get('close_request', {})
            if (handle in required and handle not in owners and request
                    and request.get('compose_bind', 0) <= 0
                    and (request.get('addressed_introduction') or
                         ((node['order'] == 1 or 'compose_bind' in request) and id(composition(node)) in noun_terms))):
                ended = recover(node)
                if ended.reference_request is not None:
                    extra.append(ended)
                    collect(ended)
            event = node.get('identity_event')
            if (event is not None and event[1] and not node.get('suppressed_event')
                    and node['identity_action'] not in events):
                ended = recover(node)
                extra.append(ended)
                collect(ended)
        return replace(result, children=(*result.children, *extra)) if extra else result

    def retain_percepts(result):
        # A no-op or sign owns its occurrence even in a predicate position.
        # The word's syntax cannot remove this selected leaf decision.
        retained, pending = set(), [result]
        while pending:
            node = pending.pop()
            request = node.reference_request or {}
            if request.get('sign_leaf') or request.get('no_op_leaf'):
                retained.update(request['leaves'])
            pending.extend((*node.children, *node.companions))
        extra, pending = [], list(stack)
        while pending:
            node = pending.pop()
            request = node.get('close_request', {})
            if ((request.get('sign_leaf') or request.get('no_op_leaf'))
                    and not retained.intersection(request['leaves'])):
                extra.append(recover(node))
                retained.update(request['leaves'])
            pending.extend(node['children'])
        return replace(result, children=(*result.children, *extra)) if extra else result

    if len(stack) == 3:
        children, references = [], []
        roles = []
        for slot, node in enumerate(stack):
            # A three-slot field still owns its selected determinations and
            # binds. Stripping to the lexical head here can orphan the very
            # introduction that a later operand's recorded bind names.
            child = recover(node) if slot in (0, 2) else None
            if (is_clause(node) or concept(node) == -1
                    or child is not None and child.reference_request is not None):
                references.append(('clause', len(children)))
                child = recover(node) if child is None else child
                children.append(child)
                roles.append(torch.zeros_like(
                    program.leaves[0]) if child.point is None else child.point)
            else:
                references.append(concept(node))
                roles.append(value(node))
        operation = None
        if type(references[1]) is int:
            selected = predicate_relation(references[1])
            if selected is not None:
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
                                          role_refs=tuple(('sym', ref) if type(ref) is int and ref not in (-1, 0) else None for ref in references))
            if probe.role_refs[1] is not None:
                try:
                    operation = registry.signature_for(
                        probe).operation.semantic_id
                except ValueError:
                    pass
        if operation not in relative:
            raise ValueError('three slots have no selected relative clause')
        meaning = ConceptualMeaning(torch.stack(roles), torch.ones(3, dtype=torch.bool, device=program.leaves.device),
                                    polarity=True if meaning is None else meaning.polarity, sentence_kind="relation")
        if operation == 'whole':
            operation = 'part'
            references = references[2], references[1], references[0]
        return retain_percepts(retain_introductions(Clause(meaning, relation=operation, refs=tuple(references), children=tuple(children),
                      **metadata(tuple(range(len(values)))),
                      **bands(dict(leaves=tuple(range(len(values))))))))
    try:
        return retain_percepts(retain_introductions(recover(stack[0], top=True)))
    finally:
        # The nested readers refer to one another through their cells, a
        # cycle that would hold the reading (and its sentence tree) until
        # the collector runs. Clearing the cells releases it on return.
        bands = composition = concept = forward_name = generic = head = None
        identity_head = identity_record = is_clause = metadata = node_order = None
        number = operation_concept = recover = recover_content = region = None
        subject_word_id = value = None
