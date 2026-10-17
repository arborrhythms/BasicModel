"""Read-only stage 4e observer and unconditional held-out gates."""
from collections import Counter
from contextlib import contextmanager


def reading(language, program, source_positions, meaning_slice):
    """Brackets and stamp transitions from the kept postfix journal alone."""
    from ReferenceContext import result_order, operation_modes
    if program is None:
        return dict(error='no_kept_program', brackets=[], steps=[], lowerings=[], leaf_positions=[])
    stack, steps, brackets, lowerings = [], [], [], []
    positions = list(source_positions)
    orders = ([-1]*len(positions) if program.leaf_orders is None
              else program.leaf_orders.detach().cpu().tolist())
    start, end = meaning_slice
    for index, (kind, operation, leaf) in enumerate(program.actions.detach().cpu().tolist()):
        if kind < 0:
            break
        if kind == 0:
            stack.append(dict(leaves=[positions[leaf]], order=orders[leaf]))
            steps.append(dict(kind='leaf', source=positions[leaf], order=orders[leaf],
                meaning_norm=float(program.leaves[leaf, start:end].detach().norm())))
            continue
        arity = 2 if kind == 1 else 1
        rules = language._compose_binary_rules if arity == 2 else language._compose_unary_rules
        if len(stack) < arity:
            raise ValueError('kept constituent journal underflows')
        operands = stack[-arity:]
        del stack[-arity:]
        rule = rules[operation]
        span = sum((node['leaves'] for node in operands), [])
        stamp = result_order(rule, tuple(node['order'] for node in operands))
        frame = program.operation_values[index, :, start:end].detach()
        name = getattr(rule, 'surface_name', None) or rule.method_name
        record = dict(kind='binary' if arity == 2 else 'unary', method=name,
            leaves=span, input_orders=[node['order'] for node in operands], order=stamp,
            meaning_norms=frame.norm(dim=-1).tolist())
        steps.append(record)
        if arity == 2 and len(span) > 1:
            brackets.append(span)
        for role, mode in operation_modes(rule).items():
            if mode not in ('mint', 'bind') or getattr(rule, 'order_delta', 0) != -1:
                continue
            operand = operands[int(role[5:])-1]
            lowerings.append(dict(span=span, description=operand['leaves'],
                before=operand['order'], after=stamp, method=name,
                meaning_norm=float(frame[2].norm())))
        stack.append(dict(leaves=span, order=stamp))
    return dict(error=None, brackets=sorted(brackets), steps=steps, lowerings=lowerings,
        leaf_positions=positions, leaf_orders=orders, roots=[node['leaves'] for node in stack],
        program=[step['method'] for step in steps if 'method' in step])


@contextmanager
def observe(model):
    """Tap admitted journals and actual writes, never annotate a model call."""
    from ReferenceClose import constituents
    records, leaves = {}, {}
    original = model._sentence_observation
    store = model.symbolSpace.ltm_store
    write = store.write_clause
    current_sid = [None]
    model._reference_close_audit = {}

    def observation(state, sid, active, **kwargs):
        value = original(state, sid, active, **kwargs)
        if not kwargs.get('admit'):
            return value
        current_sid[0] = int(sid)
        positions = model._derivation_program(int(sid))[0]
        chart = model._concept_owner().components._ready()
        for b, program in enumerate(value['entries']):
            if not bool(active[b]):
                continue
            source = (model.inputSpace._word_active_mask[b]
                & (model.inputSpace._packed_sentence_ids[b] == sid)).nonzero().flatten().tolist()
            relative = [source.index(p) if p in source else -1
                        for p in positions[b].tolist() if p >= 0]
            leaves[int(sid), b] = relative
            report = reading(model.languageSpace, program, relative,
                             (chart.meaning_start, chart.content_width))
            clause = value['clauses'][b]
            report['closed_meaning_norm'] = float(clause.point[chart.meaning_start:chart.content_width].detach().norm()) if clause.point is not None else None
            report['requests'] = [dict(mode=node.reference_request['mode'], order=node.order,
                description=[relative[i] for i in node.reference_request['leaves']],
                meaning_norm=float(node.point[chart.meaning_start:chart.content_width].detach().norm()))
                for node in constituents(clause)]
            report.update(occurrences=[], individual_writes=[])
            records[int(sid), b] = report
        return value

    def writing(clause, *args, **kwargs):
        row = write(clause, *args, **kwargs)
        key = current_sid[0], kwargs.get('stream', -1)
        if key not in records:
            return row
        written = kwargs.get('written_rows') or {}
        visited = set()
        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            index = written.get(id(node))
            # A sentence wrapper can point to the NP it owns. It is not a
            # second occurrence; the referenced NP is counted at its own row.
            head = node.refs[0]
            child = node.children[head[1]] if isinstance(head, tuple) else None
            owns_head = child is not None and child.order == 1 and child.subject_word_id == node.subject_word_id
            if index is not None and node.order == 1 and not owns_head:
                records[key]['individual_writes'].append(int(store.row_ids[index]))
            for child in (*node.children, *node.companions):
                visit(child)
        visit(clause)
        return row

    model._sentence_observation, store.write_clause = observation, writing
    try:
        yield records
    finally:
        model._sentence_observation, store.write_clause = original, write
        for key, audit in model._reference_close_audit.items():
            if key not in records:
                continue
            for item in audit:
                index = store.index_of_row(item['occurrence'])
                records[key]['occurrences'].append(dict(
                    description=[leaves[key][i] for i in item['leaves']],
                    address=item['occurrence'], written=index is not None,
                    order=None if index is None else int(store.order[index]),
                    reference=item['reference'], missing_reason=item['missing_reason']))


def grade(label, observed):
    """Missing readings and wrong spans remain in every relevant denominator."""
    observed = observed or dict(error='missing_close', brackets=[], lowerings=[],
                               occurrences=[], individual_writes=[], leaf_positions=[])
    complete = sorted(observed['leaf_positions']) == list(range(len(label['words'])))
    exact = (complete and sorted(sorted(span) for span in observed['brackets']) == label['brackets'])
    np_results = []
    for np in label['nps']:
        matches = [item for item in observed['occurrences'] if sorted(item['description']) == np['description']]
        reductions = [item for item in observed['lowerings'] if sorted(item['span']) == np['span']]
        addresses = [item['address'] for item in matches if item['written'] and item['order'] == 1]
        np_results.append(dict(span=np['span'], description=np['description'], occurrences=addresses,
            one_occurrence=len(addresses) == 1,
            duplicate=max(0, len(addresses)-1),
            description_orders=[item['before'] for item in reductions],
            reduction=(len(reductions) == 1 and reductions[0]['after'] == 1 and len(addresses) == 1)))
    used = [address for row in np_results for address in row['occurrences']]
    # One address reused for two different NPs also fails the occurrence gate.
    collisions = {address for address, count in Counter(used).items() if count > 1}
    for row in np_results:
        row['one_occurrence'] &= not bool(set(row['occurrences']) & collisions)
        row['reduction'] &= row['one_occurrence']
    writes = observed['individual_writes']
    duplicate_writes = sum(count-1 for count in Counter(writes).values())
    unexpected = [item for item in observed['occurrences'] if not any(
        sorted(item['description']) == np['description'] for np in label['nps'])]
    return dict(id=label['id'], family=label['family'], brackets=exact, nps=np_results,
        duplicate_writes=max(duplicate_writes, sum(row['duplicate'] for row in np_results)),
        occurrence_collisions=len(collisions), unexpected_occurrences=len(unexpected),
        unlowered_clean=complete and not writes and not observed['occurrences'] if label['unlowered'] else None,
        observed=observed)


def summarize(rows):
    main = [row for row in rows if row['family'] != 'description']
    nps = [np for row in rows for np in row['nps']]
    controls = [row for row in rows if row['unlowered_clean'] is not None]
    brackets = sum(row['brackets'] for row in main)
    occurrences = sum(row['one_occurrence'] for row in nps)
    duplicates = sum(row['duplicate_writes'] for row in rows)
    collisions = sum(row['occurrence_collisions'] for row in rows)
    unexpected = sum(row['unexpected_occurrences'] for row in rows)
    reductions = sum(row['reduction'] for row in nps)
    unlowered = sum(row['unlowered_clean'] for row in controls)
    gate = dict(brackets=dict(correct=brackets, total=len(main), threshold=.9),
        occurrences=dict(correct=occurrences, total=len(nps), threshold=.95,
            duplicate_writes=duplicates, collisions=collisions, unexpected=unexpected),
        reductions=dict(correct=reductions, total=len(nps), threshold=1.,
            requirement='Every lowered NP yields one order-1 occurrence; description stamp is reported'),
        unlowered=dict(correct=unlowered, total=len(controls), threshold=1.))
    for name, values in gate.items():
        values['passed'] = bool(values['total'] and values['correct']/values['total'] >= values['threshold'])
        if name == 'occurrences':
            values['passed'] &= duplicates == collisions == unexpected == 0
    return dict(gates=gate, passed=all(value['passed'] for value in gate.values()),
        groups={family: dict(sentences=len(group), brackets=sum(row['brackets'] for row in group),
            nps=sum(len(row['nps']) for row in group),
            occurrences=sum(np['one_occurrence'] for row in group for np in row['nps']),
            reductions=sum(np['reduction'] for row in group for np in row['nps']))
            for family in sorted({row['family'] for row in rows})
            for group in [[row for row in rows if row['family'] == family]]})
