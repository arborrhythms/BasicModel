"""Located ideas from attention, committed as one sequence in the ordinary pair store."""
import torch
from Memory import OCCURRENCE, SYMBOL, RELATION


def brackets(model, percepts):
    """Name lexical occurrences from complete, visible native percepts. MTM owns nomination."""
    from WordSurfaces import is_lexical_form
    ps, owner = model.perceptualSpace, model._concept_owner()
    ids, spans = percepts[:2]
    intervals = ps._forward_input.get('word_intervals') or ()
    words = []
    for b, limits in enumerate(intervals):
        parts = [(pid, lo, hi) for pid, (lo, hi) in zip(ids[b].tolist(), spans[b].tolist(), strict=True)
                 if pid >= 0 and lo < hi]
        row = []
        for start, stop in limits:
            members = [(pid, lo, hi) for pid, lo, hi in parts if start <= lo < hi <= stop]
            if not members or members[0][1] != start or members[-1][2] != stop:
                continue
            if any(left[2] != right[1] for left, right in zip(members, members[1:])):
                continue
            native = [pid for pid, _, _ in members]
            form = b''.join(ps.percept_store.bytes_for(pid) for pid in native)
            if not is_lexical_form(form):
                continue
            word = owner.interpret.word_for_form(form)
            if word is None:
                word = owner.interpret.lookup_word(native, model.wholeSpace.property_rows_for_bytes(form),
                    form=form, word_reading=True, reading=owner.reading(b))
            row.append((start, stop, -1 if word is None else word))
        words.append(row)
    width = max(map(len, words), default=0)
    if not width:
        return None
    table = ids.new_tensor([row + [(0, 0, -1)] * (width - len(row)) for row in words])
    return table[..., :2], table[..., 2]


def commit(stores, heads, durable):
    """Copy each occupied idea once and translate its ordered STM1 links into LTM1."""
    sizes = [len(store) for store in stores]
    pairs = sum(len(store.parts) for store in stores)
    owner = durable.owner()
    if len(durable) + sum(sizes) > durable.capacity or (pairs and
            (len(durable.parts) + pairs >= durable.parts.capacity or
             owner is not None and owner._concept_allocator.next_id + pairs > durable.parts.capacity)):
        raise OverflowError('reading exceeds remaining long-term idea or relation capacity')
    committed = []
    for store, head in zip(stores, heads, strict=True):
        translated = {}
        for row in range(len(store)):
            name = store.row_name(row)
            target = durable.append_idea(store.slots[row], trust=float(store.trust[row]),
                referent=store.referent_of(row),
                entry=(store.episodic_concepts[row], store.episodic_values[row]))
            translated[OCCURRENCE, name] = OCCURRENCE, durable.row_name(target)
        # Handles need not be issued in tree order. Copy only relations under
        # this reading's head, translating children before their parent.
        pending = [] if head is None else [head]
        while pending:
            ref = pending[-1]
            if ref in translated:
                pending.pop()
                continue
            kind, name = ref
            if kind == SYMBOL:
                translated[ref] = ref
                continue
            if kind != RELATION:
                raise ValueError('reading head names an unknown occurrence')
            children = store.parts.references(name)
            missing = [child for child in children if child not in translated]
            if missing:
                pending.extend(missing)
                continue
            left, right = children
            translated[ref] = RELATION, durable.parts.add(
                translated[left], translated[right], truth_plus=float(store.parts.truth_plus[name]),
                truth_minus=float(store.parts.truth_minus[name]))
        committed.append(translated.get(head))
    return tuple(committed)


def read(model, bank, concepts):
    """Write the narrowed looks to memory and return the caller its open field."""
    owner = model._concept_owner()
    # Each look binds its own packed rows. Keep their addresses and evidence
    # together when returning to the parallel input that triggered the read.
    names = ('_percept_field', '_cs_last_a0', '_cs_order0_raw', '_cs_field_rows',
             '_cs_field_concept_ids', '_cs_field_caps', '_cs_field_where', '_cs_field_when',
             '_cs_field_wires', '_cs_level_acts', '_cs_level_rows', '_cs_wave_qe')
    opened = {name: vars(owner)[name] for name in names if name in vars(owner)}
    try:
        return _read(model, bank, concepts)
    finally:
        for name in names:
            if name in opened:
                object.__setattr__(owner, name, opened[name])
            else:
                vars(owner).pop(name, None)


def _read(model, bank, concepts):
    """Narrow until the request stops, STM fills or nothing is found."""
    from Places import readings
    from Attention import occurrence_probabilities, select_occurrence
    from LookForecast import OccurrenceCredit
    from SentenceLooks import attend, narrow, lower
    from dataclasses import replace
    from EpisodicMemory import select, unkept, uncertain
    from SentenceField import inventory
    from RequestRead import at_rows, spread_once
    owner, reader = model._concept_owner(), model.request_read
    B = len(concepts)
    strata = model.conceptualSpace.stm.strata
    stores = [strata.idea_stores[b * strata.trials] for b in range(B)]
    heads = [None] * B
    for store in stores:
        store.expire()
        store.to(bank.table)
    current = owner._cs_open_field
    def located(field, index=None):
        return at_rows(inventory(owner, field), field, reader.rows, B, index)
    opened = located(current)
    memory, own = torch.zeros_like(opened), bank.activity.clone()
    left = bank.where[..., 1] > bank.where[..., 0]
    known = concepts >= 0
    _, _, percept_codes = readings(bank)
    observation = model.conceptualSpace.stm.model_state
    observation.begin_looks(bank.where, left.any(-1))
    credit = OccurrenceCredit(bank) if model.training and torch.is_grad_enabled() else None
    excluded, ambiguous = unkept(owner), uncertain(owner)
    for _ in range(min(model.attention_budget, model.conceptualSpace.stm.capacity)):
        units = reader.units(reader.given(opened, memory, left.any(-1)),
                             observation.request_state(model.where_registry))
        active = left.any(-1) & (reader.stopping(units).detach() <= 0)
        if not active.any():
            break
        field, _, scores = narrow(model, bank, attend(model, units), own, concepts,
                                  known, left & active[:, None])
        probabilities = occurrence_probabilities(scores)
        index = select_occurrence(probabilities, sample=model.training)
        index = torch.where(active, index, -1)
        observation.observe(bank.where, index)
        found = index >= 0
        if not found.any():
            break
        if credit is not None:
            credit.observe(probabilities, index)
        batch = torch.arange(B, device=index.device)
        evidence = field.evidence[:, batch, index.clamp_min(0)].unsqueeze(2)
        episode = select(replace(field, evidence=evidence), stores[0].episodic_capacity,
                         found, unkept=excluded, uncertain=ambiguous)
        for b in found.nonzero().flatten().tolist():
            w, store = int(index[b]), stores[b]
            cid = int(concepts[b, w])
            code = percept_codes[b, w]
            if cid >= 0:
                row = owner._concept_row_of(cid)
                code = owner.similarity_codebook.lookup_rows(index.new_tensor([row]), stream=b)[0]
            point = store._fit(code).clone()
            point[store.when_start-4:store.when_start] = field.wires[b, w]
            point[store.when_start:store.when_start+4] = bank.when_band[b, w]
            row = store.append_idea(point, referent=None if cid < 0 else ('sym', cid),
                trust=float(scores[b, w].detach().clamp(0, 1)), entry=(episode[0][b], episode[1][b]))
            heads[b] = store.parts.append_occurrence(heads[b], store.row_name(row))
            left[b, w] = False
        memory = torch.where(found[:, None, None], spread_once(owner, located(field, index)), memory)
        own = lower(own, bank, index, found)
    if credit is not None:
        credit.register(model.errors, model.inputSpace.data.paced_observations)
    durable = model.symbolSpace.ltm_store
    committed = commit(stores, heads, durable)
    for b, head in enumerate(committed):
        if head is not None:
            owner.mtm.close(durable, stream=b)
    return committed
