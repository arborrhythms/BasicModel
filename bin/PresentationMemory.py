"""Rewrite document covers and discard epoch experience, retaining concepts."""
import copy

import torch


# Returned readings and diagnostic records can own the same occurrence handles
# as the active discourse. None is a parameter or a persistent concept record.
_EMPTY_DICTS = (
    '_document_covers', '_sentence_fields', '_sentence_understandings',
    '_what_pending', '_what_root_symbols', '_what_referents',
    '_reference_close_audit', '_costed_thought_outputs', '_thought_ordinals',
)
_EMPTY_LISTS = (
    '_what_policy_records', '_what_episode_steps', '_verb_teacher_records',
    '_modal_teacher_records', '_sentence_credit_audits', '_sentence_trial_costs',
    '_sentence_winners', '_sentence_reconstructions',
)
_EMPTY_TUPLES = ('_thought_document_ends', '_last_what_answers',
                 '_last_what_actual', '_active_what_questions', '_sentence_answer_questions',
                 '_last_closing_thoughts')
_ABSENT = (
    '_last_understanding', '_last_sentence_understanding', '_sentence_field',
    '_sentence_symbol_field', '_last_sentence_field', '_last_field_cost',
    '_sentence_primed_bank', '_sentence_inverse_candidates', '_last_selected_thought',
    '_last_thought_comparison', '_last_answer_construction', '_last_output_comparison',
    '_last_output_walk_trace', '_last_decoder_comparison', '_last_decoder_trace',
    '_answer_attention_obs', '_thought_walk', '_compose_fork',
    '_sentence_answer_cost', '_sentence_answer_predictions', '_sentence_answer_raw_cost',
    '_last_compose_score_function', '_sentence_journal_columns', '_query_ready_rows',
    '_sentence_supplied_answers', '_sentence_comparison_registry',
    '_sentence_reader_observation', '_last_sentence_credit', '_last_what_thinking',
    '_recall_brackets',
)


def begin(model):
    """Begin a document presentation; its cover is rewritten on first read."""
    if getattr(model, '_sentence_run_active', False) or getattr(model, '_open_sentence_slot', None) is not None:
        raise RuntimeError('a presentation cannot begin inside a live sentence')
    model._document_covers = {}


def _rows(store, indices):
    """Snapshot only the occupied LTM0 rows and their provenance."""
    extras = store.semantic_extras()
    indices = list(indices)
    return dict(columns={name: getattr(store, name)[indices].detach().cpu().clone()
                         for name in store.row_columns},
                records=[copy.deepcopy(extras['records'][i]) for i in indices],
                document_sources=dict(extras['document_sources']))


def initial_observation(store, index):
    """Access and surprise belong to the initial row, not this history's reads."""
    start = getattr(store, '_epoch_rows', None)
    if start is None:
        return int(store.access_count[index]), float(store.surprise[index])
    address = int(store.address_keys[index])
    matches = (start['columns']['address_keys'] == address).nonzero(as_tuple=True)[0]
    if len(matches):
        slot = int(matches.item())
        return int(start['columns']['access_count'][slot]), float(start['columns']['surprise'][slot])
    initial = start.get('initial_observations', {}).get(address)
    if initial is None:
        raise ValueError(f'LTM0 occurrence {address} lacks its creation observation')
    return initial


def import_checkpoint_observations(store, *, envelope_version):
    """Import v1/v2 observations; v3 envelopes must already contain them."""
    if envelope_version not in (1, 2, 3):
        raise ValueError(f'unsupported creation-observation envelope version {envelope_version}')
    legacy = envelope_version < 3
    start = getattr(store, '_epoch_rows', None)
    if start is not None:
        known = set(start['columns']['address_keys'].tolist())
        observations = start.setdefault('initial_observations', {})
        if legacy:
            # An older supply's saved row precedes later live access. Import
            # that observation before filling genuinely unsnapshotted rows.
            for state in start.get('supplied_rows', {}).values():
                columns = state['columns']
                for index, address in enumerate(columns['address_keys'].tolist()):
                    observations.setdefault(address, (int(columns['access_count'][index]),
                                                       float(columns['surprise'][index])))
        for index,address in enumerate(store.address_keys[:len(store)].tolist()):
            if address not in known and observations.get(address) is None:
                if not legacy:
                    raise ValueError(f'LTM0 occurrence {address} lacks its creation observation in current checkpoint')
                observations[address] = int(store.access_count[index]), float(store.surprise[index])
    bank = store.parts
    if bank.epoch_start is not None:
        observations = bank.epoch_start.setdefault('_initial_observations', {})
        for row in bank.indices().tolist():
            if not int(bank.epoch_start['ref_kinds'][row,0]) and observations.get(row) is None:
                if not legacy:
                    raise ValueError(f'LTM1 relation {row} lacks its creation observation in current checkpoint')
                observations[row] = int(bank.access_count[row]), float(bank.surprise[row])


def _row_states(state):
    return {int(address): dict(columns={name: values[i:i+1].clone()
                                       for name, values in state['columns'].items()},
                              records=[copy.deepcopy(state['records'][i])],
                              document_sources=dict(state['document_sources']))
            for i, address in enumerate(state['columns']['address_keys'].tolist())}


def _relation_state(bank, row):
    return {name: getattr(bank, name)[row].detach().cpu().clone() for name in bank.columns}


def _supply_records(store):
    return store.__dict__.setdefault('_supplies', {})


def _occurrence_state(store, index):
    state = _rows(store, [index])
    count, surprise = initial_observation(store, index)
    state['columns']['access_count'][0] = count
    state['columns']['surprise'][0] = surprise
    return state


def _initial_relation(bank, row):
    state = _relation_state(bank, row)
    start = bank.epoch_start
    if start is not None:
        if int(start['ref_kinds'][row, 0]):
            observation = int(start['access_count'][row]), float(start['surprise'][row])
        else:
            observation = start.get('_initial_observations', {}).get(row)
        if observation is None:
            raise ValueError(f'LTM1 relation {row} lacks its creation observation')
        state['access_count'].fill_(observation[0])
        state['surprise'].fill_(observation[1])
    return state


def _ensure_base(store, *, first=False):
    start, bank = getattr(store, '_epoch_rows', None), store.parts
    if start is None or bank.epoch_start is None:
        return False
    # The preceding checkpoint envelope stored explicit occurrence supplies
    # separately, without a source ledger. Import the saved pair, never the
    # live row's possibly history-updated pair.
    for state in start.pop('supplied_rows', {}).values():
        for address,saved in _row_states(state).items():
            document = bytes(saved['columns']['document_keys'][0].tolist()).hex()
            origin = saved['document_sources'].get(document)
            if origin not in (store.ORIGIN_USER,store.ORIGIN_PROVISIONED):
                raise ValueError('legacy supplied occurrence lacks its source')
            _supply_records(store).setdefault((origin,'occ',address),dict(origin=origin,
                root=('occ',address),asserted=tuple(float(saved['columns'][name][0]) for name in ('truth_plus','truth_minus')),
                occurrences={address:saved},relations={},before_relations={}))
    if '_base' not in start:
        start['_base'] = dict(columns={k:v.clone() for k,v in start['columns'].items()},
                              records=copy.deepcopy(start['records']),
                              document_sources=dict(start['document_sources']))
        if first:
            owned = {address for source in _supply_records(store).values() for address in source['occurrences']}
            base = _row_states(start['_base'])
            for address in owned:
                base.pop(address, None)
            start['_base'] = _join_rows(store, base)
    if '_base' not in bank.epoch_start:
        base = {name: bank.epoch_start[name].clone() for name in bank.state_dict()}
        if first:
            previous = {}
            for source in _supply_records(store).values():
                for row, value in source['before_relations'].items():
                    previous.setdefault(row, value)
            for row, value in previous.items():
                for name in bank.columns:
                    base[name][row] = (-1 if name in ('refs','surprise','operator','grammar_operation') else 0) if value is None else value[name]
            base['count'].fill_(int(base['ref_kinds'][:, 0].ne(0).sum()))
        bank.epoch_start['_base'] = base
    return True


def _join_rows(store, rows):
    values = list(rows.values())
    documents = {bytes(state['columns']['document_keys'][0].tolist()).hex() for state in values}
    return dict(columns={name: (torch.cat([state['columns'][name] for state in values])
                                if values else getattr(store,name)[:0].detach().cpu().clone())
                         for name in store.row_columns},
                records=[copy.deepcopy(state['records'][0]) for state in values],
                document_sources={key:origin for state in values for key,origin in state['document_sources'].items() if key in documents})


def _references(value):
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (tuple,list)):
            if len(item) == 2 and item[0] == 'sym' and type(item[1]) is int:
                yield tuple(item)
            elif len(item) == 3 and item[0] == 'ltm':
                yield tuple(item)
            else:
                pending.extend(item)


@torch.no_grad()
def refresh_supplies(store, *, first=False):
    """Build both retained memories from setup plus independently held supplies."""
    if not _ensure_base(store, first=first):
        return
    start, bank = store._epoch_rows, store.parts
    base = start['_base']
    rows = _row_states(base)
    relation_base = bank.epoch_start['_base']
    relations = {name:value.clone() for name,value in relation_base.items()}
    sources = _supply_records(store)
    for source in sources.values():
        for address, saved in source['occurrences'].items():
            if address not in rows:
                rows[address] = copy.deepcopy(saved)
                rows[address]['columns']['truth_plus'].zero_()
                rows[address]['columns']['truth_minus'].zero_()
            else:
                for name,value in saved['columns'].items():
                    if name not in ('truth_plus','truth_minus','access_count','surprise'):
                        rows[address]['columns'][name] = value.clone()
            if source['root'] == ('occ', address):
                for name, value in zip(('truth_plus','truth_minus'), source['asserted']):
                    rows[address]['columns'][name][0] = max(float(rows[address]['columns'][name][0]), value)
            # A new supply can replace its source text; observation counts are
            # still the epoch-start/creation counts in the saved row.
            rows[address]['records'] = copy.deepcopy(saved['records'])
            rows[address]['document_sources'].update(saved['document_sources'])
        for row, saved in source['relations'].items():
            if not int(relations['ref_kinds'][row, 0]):
                for name, value in saved.items():
                    relations[name][row].copy_(value)
                relations['truth_plus'][row] = relations['truth_minus'][row] = 0.
            if source['root'] == ('rel', row):
                for name, value in zip(('truth_plus','truth_minus'), source['asserted']):
                    relations[name][row] = max(float(relations[name][row]), value)
    # Close over both memories before either is restored. A raw symbol naming
    # an existing relation is an address dependency, never new authority.
    pending = [('occ', address) for address in rows]
    pending += [('rel', row) for row in relations['ref_kinds'][:,0].ne(0).nonzero().flatten().tolist()]
    seen = set()
    while pending:
        kind, key = pending.pop()
        if (kind,key) in seen:
            continue
        seen.add((kind,key))
        if kind == 'occ':
            if key not in rows:
                index = store.index_of_row(key)
                if index is None:
                    raise ValueError(f'retained occurrence {key} is unavailable')
                rows[key] = _occurrence_state(store,index)
                rows[key]['columns']['truth_plus'].zero_()
                rows[key]['columns']['truth_minus'].zero_()
            for ref in _references(rows[key]['records'][0]):
                if ref[:2] == ('ltm', store.address_domain):
                    pending.append(('occ',ref[2]))
                elif ref[0] == 'sym' and 0 < ref[1] < bank.capacity and (
                        bank.contains(ref[1]) or int(relations['ref_kinds'][ref[1],0])):
                    pending.append(('rel',ref[1]))
        else:
            if not int(relations['ref_kinds'][key,0]):
                if not bank.contains(key):
                    raise ValueError(f'retained relation {key} is unavailable')
                for name,value in _initial_relation(bank,key).items():
                    relations[name][key].copy_(value)
                relations['truth_plus'][key] = relations['truth_minus'][key] = 0.
            for tag,value in zip(relations['ref_kinds'][key].tolist(),relations['refs'][key].tolist()):
                if tag == bank.OCCURRENCE:
                    pending.append(('occ',value))
                elif tag == bank.RELATION:
                    pending.append(('rel',value))
    # A closure-only dependency may outlive the supply that retained it.
    # Preserve its saved observation before replacing either start snapshot;
    # future supplies must not substitute the history's live counters.
    observations = dict(start.get('initial_observations', {}))
    for index, address in enumerate(start['columns']['address_keys'].tolist()):
        if address not in rows and store.index_of_row(address) is not None:
            observations[address] = (int(start['columns']['access_count'][index]),
                                     float(start['columns']['surprise'][index]))
    relation_start = bank.epoch_start
    relation_observations = dict(relation_start.get('_initial_observations', {}))
    dropped = relation_start['ref_kinds'][:, 0].ne(0) & relations['ref_kinds'][:, 0].eq(0)
    for row in dropped.nonzero().flatten().tolist():
        if bank.contains(row):
            relation_observations[row] = (int(relation_start['access_count'][row]),
                                          float(relation_start['surprise'][row]))
    updated = _join_rows(store, rows)
    updated['_base'] = base
    updated['initial_observations'] = observations
    store._epoch_rows = updated
    relations['count'].fill_(int(relations['ref_kinds'][:,0].ne(0).sum()))
    ordered = bank.ordered_rows(relations['refs'], relations['ref_kinds'])
    relations['relation_order'].fill_(-1)
    if ordered:
        relations['relation_order'][:len(ordered)] = relations['relation_order'].new_tensor(ordered)
    relations['_base'] = relation_base
    relations['_initial_observations'] = relation_observations
    bank.epoch_start = relations


@torch.no_grad()
def supplied_closing(store, root, *, origin, asserted, occurrences, relations, before_relations):
    """Retain the closing's writes; only its outer root is a supply assertion."""
    sources = _supply_records(store)
    key = (int(origin), *root)
    previous = sources.get(key)
    prior_pairs = (0.,0.) if previous is None else previous['asserted']
    record = dict(origin=int(origin), root=tuple(root), asserted=tuple(max(a,float(b)) for a,b in zip(prior_pairs,asserted)),
                  occurrences={} if previous is None else previous['occurrences'],
                  relations={} if previous is None else previous['relations'],
                  before_relations={} if previous is None else previous['before_relations'])
    already = {row for value in sources.values() for row in value['relations']}
    for address in occurrences:
        index = store.index_of_row(address)
        if index is None:
            raise ValueError(f'supply occurrence {address} is unavailable')
        record['occurrences'][address] = _occurrence_state(store,index)
    for row in relations:
        record['relations'][row] = _initial_relation(store.parts,row)
        if row not in already:
            record['before_relations'][row] = before_relations.get(row)
    sources[key] = record
    refresh_supplies(store)


@torch.no_grad()
def withdraw_supplies(store, *, origin=None, removed=(), preserve=()):
    """Release supply provenance with its source, before any epoch rollback."""
    removed, preserve = set(removed), set(preserve)
    start = getattr(store, '_epoch_rows', None)
    ready = start is not None and _ensure_base(store)
    sources = _supply_records(store)
    for key, source in list(sources.items()):
        root = source['root']
        selected = (origin is not None and source['origin'] == origin and
                    not (root[0] == 'occ' and root[1] in preserve)) or (root[0] == 'occ' and root[1] in removed)
        if selected:
            del sources[key]
    if not ready:
        return
    base = _row_states(start['_base'])
    for address,state in list(base.items()):
        document = bytes(state['columns']['document_keys'][0].tolist()).hex()
        if address in removed or (origin is not None and address not in preserve and
                                  state['document_sources'].get(document,0) == origin):
            del base[address]
    start['_base'] = _join_rows(store,base)
    # Rebuild the saved copies together; live observation counters are not
    # restored here. The epoch boundary alone publishes those saved values.
    refresh_supplies(store)


@torch.no_grad()
def supplied_occurrence(store, index, *, asserted):
    """Snapshot the supply now; later evidence or access belongs to the history."""
    import math
    if asserted is None or len(asserted) != 2 or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in asserted):
        raise ValueError('a supplied occurrence requires its explicit asserted pair')
    address = int(store.address_keys[index])
    supplied_closing(store, ('occ',address), origin=store.source_document(index), asserted=asserted,
                     occurrences=(address,), relations=(), before_relations={})


def _begin_rows(store, required):
    start = getattr(store, '_epoch_rows', None)
    if start is None:
        store._epoch_rows = _rows(store, range(len(store)))
        return set()
    before_order = {address: index for index, address in enumerate(store.address_keys[:len(store)].tolist())}
    before = set(before_order)
    start_order = {address: index for index, address in enumerate(start['columns']['address_keys'].tolist())}
    kept = [start]
    if not set(required) <= set(start_order):
        raise ValueError('retained relations have unavailable occurrence endpoints')
    count = sum(len(state['records']) for state in kept)
    if count > store.capacity:
        raise OverflowError('epoch history and given rows exceed the constructed LTM0 capacity')
    supplies = copy.deepcopy(_supply_records(store))
    store.reset(keep_learned_readings=True)
    extras = store.semantic_extras()
    extras['supplies'] = supplies
    records, sources = [], {}
    rows = [(address, state, index) for state in kept
            for index, address in enumerate(state['columns']['address_keys'].tolist())]
    baseline_order = {address:index for index,address in enumerate(
        start.get('_base', start)['columns']['address_keys'].tolist())}
    rows.sort(key=lambda row: (0,baseline_order[row[0]]) if row[0] in baseline_order
              else (1,before_order.get(row[0],start_order[row[0]])))
    for state in kept:
        sources.update(state['document_sources'])
    for offset, (_, state, index) in enumerate(rows):
        for name, values in state['columns'].items():
            getattr(store, name)[offset].copy_(values[index])
        records.append(state['records'][index])
    # Recurrence membership is learned again from the retained readings.
    # The seat assignments were cleared together by reset().
    store.recurrence_readings.fill_(-1)
    store.count.fill_(count)
    store._replace_document_sources(sources)
    # Learned recurrence prototypes and document-cover symbols belong to the
    # concept, and their clocks need not be rewound to the start of the epoch.
    store.load_semantic_extras(dict(extras, records=records, document_sources=sources))
    store.rebuild_addresses()
    restored = _rows(store, range(count))
    restored['_base'] = start.get('_base', copy.deepcopy(restored))
    store._epoch_rows = restored
    return before - set(store.address_keys[:len(store)].tolist())


def _owners(model):
    owners = list(getattr(model, 'conceptualSpaces', ()) or ())
    owner = getattr(model, 'conceptualSpace', None)
    if owner is not None and all(owner is not item for item in owners):
        owners.append(owner)
    return owners


def validate_checkpoint(model, extras, *, state=None):
    """An earlier history cannot be inferred from a checkpoint's last rows."""
    from Layers import LTM0
    stores = [name for name, module in model.named_modules() if isinstance(module, LTM0)]
    if not isinstance(extras, dict):
        if state is not None and any(int(state.get(name + '.count', 0)) for name in stores):
            raise ValueError('checkpoint with LTM0 history lacks its epoch start')
        return
    for entry in (extras.get('conceptual_spaces') or {}).values():
        allocator = entry.get('allocator') or {}
        if allocator.get('word_obj_meta') or allocator.get('interpretations'):
            raise ValueError('retired META definition artifacts are not LTM2 inputs')
    rows = extras.get('epoch_rows')
    if not isinstance(rows, dict):
        raise ValueError('structural checkpoint lacks the LTM0 epoch start')
    for entry in (extras.get('conceptual_spaces') or {}).values():
        allocator = entry.get('allocator')
        if not allocator:
            continue
        if 'epoch_relations' not in allocator:
            raise ValueError('structural checkpoint lacks the LTM1 epoch start')
        if allocator['epoch_relations'][0] is not None:
            if 'epoch_mtm' not in entry:
                raise ValueError('structural checkpoint lacks the MTM epoch start')
            if any(rows.get(name) is None for name in stores):
                raise ValueError('structural checkpoint lacks a store epoch start')


@torch.no_grad()
def capture_start(model):
    """Remember intentional setup before evaluation can add a second history."""
    store = getattr(getattr(model, 'symbolSpace', None), 'ltm_store', None)
    first = store is not None and getattr(store, '_epoch_rows', None) is None
    if first:
        store._epoch_rows = _rows(store, range(len(store)))
    for item in _owners(model):
        alloc = getattr(item, '_concept_allocator', None)
        if alloc is None:
            continue
        if not hasattr(item, '_epoch_mtm'):
            item._epoch_mtm = item.mtm.state()
        if alloc.parts.epoch_start is None:
            alloc.parts.begin_epoch()
        if not hasattr(alloc, '_epoch_singletons'):
            alloc._epoch_singletons = (copy.deepcopy(alloc.singleton_members), copy.deepcopy(alloc.singleton_rows))
    if store is not None and store.parts.epoch_start is None:
        store.parts.begin_epoch()
    if store is not None:
        refresh_supplies(store, first=first)


@torch.no_grad()
def epoch(model):
    """Discard epoch experience after backward, preserving learned concepts."""
    symbol = getattr(model, 'symbolSpace', None)
    store = getattr(symbol, 'ltm_store', None)
    owners = _owners(model)
    # Save each concept's witnessed order before its historical grounds go.
    # The allocator also owns concept-to-concept records that survive the reset.
    witnesses, derived_orders, witnessed_allocators = [], [], set()
    for item in owners:
        alloc = getattr(item, '_concept_allocator', None)
        if alloc is None or id(alloc) in witnessed_allocators:
            continue
        witnessed_allocators.add(id(alloc))
        for cid in alloc.placement:
            if (alloc.parts.epoch_start is not None and cid not in alloc.reference_orders
                    and alloc.parts.parts((alloc.parts.SYMBOL, cid))):
                # Older learned classes may have a derived order but no
                # recorded stamp. Read it while their Part rows still exist.
                order, structural = alloc._part_order(cid, {cid})
                if structural and order >= 0:
                    derived_orders.append((alloc, cid, order))
            records = [(role, ref) for role, ref in alloc.store_of(cid).constituents(cid)
                       if isinstance(ref, tuple) and len(ref) == 2 and ref[0] in ('occ', 'rel')]
            if records:
                order = alloc.order_of(cid)
                if order < 0 and cid in alloc.individuals:
                    raise RuntimeError('a promoted identity has no witnessed order at the epoch boundary')
                witnesses.append((alloc, cid, order, records))
    begin(model)
    percept = getattr(model, 'percept_expectation', None)
    if percept is not None:
        percept.clear_context()
    for item in owners:
        if not hasattr(item, '_epoch_mtm'):
            item._epoch_mtm = item.mtm.state()
    removed, removed_parts = set(), set()
    banks = {id(alloc.parts): alloc.parts for item in owners
             if (alloc := getattr(item, '_concept_allocator', None)) is not None}
    if store is not None:
        banks[id(store.parts)] = store.parts
        first = getattr(store, '_epoch_rows', None) is None
        if first:
            store._epoch_rows = _rows(store, range(len(store)))
        if store.parts.epoch_start is None:
            store.parts.begin_epoch()
        # A forgotten supply cannot regain authority merely because its
        # earlier snapshot still exists. Relation-only supplies are checked
        # here as well as occurrence compaction's explicit revocation.
        for key, source in list(_supply_records(store).items()):
            kind, root = source['root']
            if (kind == 'occ' and store.index_of_row(root) is None) or (kind == 'rel' and not store.parts.contains(root)):
                del store._supplies[key]
        refresh_supplies(store, first=first)
    withdrawn = {key: bank.begin_epoch() for key, bank in banks.items()}
    if store is not None:
        removed_parts = withdrawn[id(store.parts)]
        bank = store.parts
        required = set(bank.refs[bank.ref_kinds == bank.OCCURRENCE].tolist())
        removed = _begin_rows(store, required)
        for name in ('forget_passes', 'forget_sentences', 'forget_last_sentence',
                     'forget_cutoff', 'forget_pending'):
            getattr(store, name).zero_()
        for name in ('_slot_index_source', '_slot_index', '_forward_index_source',
                     '_forward_postings', '_forward_by_address', 'last_forgetting_report',
                     'last_part_forgetting_report'):
            store.__dict__.pop(name, None)
        store.rebuild_addresses()
    # Withdrawing the old LTM0 history also forgets its MTM occurrences.
    # Restore intentional setup only after its occurrence rows are back.
    for item in owners:
        item.mtm.restore(item._epoch_mtm, device=item.similarity_codebook.W.device)
    for alloc, cid, order, records in witnesses:
        expired = [(role, ref) for role, ref in records
                   if ref[1] in (removed if ref[0] == 'occ' else withdrawn[id(alloc.parts)])]
        if expired:
            if order >= 0:
                alloc.reference_orders.setdefault(cid, order)
            for role, ref in expired:
                alloc.store_of(cid).remove_constituent(cid, role, ref)
            alloc.settle(cid)
    for alloc, cid, order in derived_orders:
        if alloc.order_of(cid) != order:
            alloc.reference_orders.setdefault(cid, order)
            alloc.settle(cid)
    seen = set()
    for item in owners:
        alloc = getattr(item, '_concept_allocator', None)
        if alloc is not None and id(alloc) not in seen:
            seen.add(id(alloc))
            removed_relations = withdrawn[id(alloc.parts)]
            from LTM1 import OCCURRENCE, RELATION
            expired_refs = {(RELATION, row) for row in removed_relations}
            expired_refs.update((OCCURRENCE, address) for address in removed)
            alloc.singletons.difference_update(expired_refs)
            alloc.definitions.rebuild()
            interpretation = item.__dict__.get('_interpret_operator')
            if interpretation is not None:
                for word, description in alloc.word_descriptions.items():
                    if not alloc.definitions.objects(word):
                        interpretation._pending.setdefault(word, description)
            if not hasattr(alloc, '_epoch_singletons'):
                alloc._epoch_singletons = (copy.deepcopy(alloc.singleton_members),
                                           copy.deepcopy(alloc.singleton_rows))
            else:
                members, rows = alloc._epoch_singletons
                alloc.singleton_members.clear()
                alloc.singleton_rows.clear()
                alloc.singleton_rows.update(copy.deepcopy(rows))
                for key, (member, token, value) in members.items():
                    alloc.remember_singleton(key, member, token, value)
            alloc.testimony_seen.clear()
            alloc.testimony_current.clear()
            alloc.relate_idx = {key: value for key, value in alloc.relate_idx.items()
                                if value not in expired_refs
                                and not (key[:1] == ('part-kind',) and key[1] in expired_refs)}
        components = getattr(item, 'components', None)
        if components is not None:
            components._innovations.clear()
            components._chart = None
            components.verbs._pending.clear()
            components.verbs._witnesses.clear()
            resolver = components.reference_close
            resolver.__dict__.pop('_mentions', None)
            resolver.last_audit = ()
            for name in ('last_verb_teacher', 'last_modal_teacher'):
                resolver.__dict__.pop(name, None)
        # Knowing's eager carrier can retain an occurrence after its thought
        # history has gone. Sentence STM itself is initialized by the driver.
        carrier = getattr(item, 'subspace', None)
        if carrier is not None:
            object.__setattr__(carrier, '_thought_occurrence', None)
    for name in _EMPTY_DICTS:
        setattr(model, name, {})
    for name in _EMPTY_LISTS:
        setattr(model, name, [])
    for name in _EMPTY_TUPLES:
        setattr(model, name, ())
    for name in _ABSENT:
        setattr(model, name, None)
    for name in ('_forget_document_boundary', '_forget_all_streams_closed', '_forget_training'):
        model.__dict__.pop(name, None)
    model._clear_recall_history()
    for name in ('expectation', 'what_memory'):
        memory = getattr(symbol, name, None)
        if memory is not None:
            memory.reset()
            if name == 'what_memory':
                memory.__dict__.pop('_address_sources', None)
            else:
                memory.set_symbol_bank(memory._bank_keys[:0], identities=memory._bank_ids[:0],
                    orders=memory._bank_orders[:0], addresses=memory._bank_addresses[:0])
    language = getattr(model, 'languageSpace', None)
    if language is not None:
        for name in ('verb_action', 'modifier_action'):
            action = getattr(language, name)()
            if action is not None:
                action.groups.witnesses.clear()
                action.groups.reports.clear()
        for name in ('_reference_bank', '_compose_primed_bank', '_last_derivation',
                     '_what_context', '_what_ltm_context'):
            setattr(language, name, None)
    truth = getattr(symbol, 'truth_layer', None)
    if truth is not None:
        truth.sync_from_ltm()
    return dict(removed=len(removed), relations_removed=len(removed_parts),
                retained=0 if store is None else len(store))
