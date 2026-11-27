"""Construction-bounded symbol arrays with a leading MTM field."""
import torch
from architecture import WHERE_WIRES
from torch import nn
from torch.nn import functional as F

WORD_FIELDS = ('concept_ids', 'concept_rows', 'concept_orders', 'object_ids',
               'object_rows', 'object_orders', 'symbol_where', 'symbol_when')
LOOKUP_FIELDS = ('lookup_rows', 'lookup_atoms', 'lookup_sentence_ids')


def word_view(source,name):
    """Word-aligned view; plain metadata fixtures need no runtime field owner."""
    entries = getattr(source,'symbol_entries',None)
    return entries.read(name,words=True) if entries is not None else getattr(source,'_ar_word_'+name,None)


def lookup_view(source,name):
    """The lexical inverse bank excludes entries with no independent spelling."""
    entries = getattr(source,'symbol_entries',None)
    return entries.read(name,words=True) if entries is not None else getattr(source,'_ar_concept_'+name,None)


class SymbolEntries(nn.Module):
    def __init__(self, streams, words, prepend, width):
        super().__init__()
        self.streams, self.words, self.prepend = int(streams), int(words), int(prepend)
        if min(self.streams, self.words, self.prepend) < 1:
            raise ValueError('symbol field requires positive construction bounds')
        self.extents = {}
        # B=nStreams; base=mtmPrepend (default 64); W is the largest
        # constructed serial-word bucket. The existing inverse bank contains
        # two W blocks (word and object), so its extent is base+2W.
        for name in WORD_FIELDS:
            band = name in ('symbol_where', 'symbol_when')
            wires = WHERE_WIRES if name == 'symbol_where' else 4
            shape = (self.streams, self.prepend+self.words, *((wires,) if band else ()))
            self.register_buffer(name, torch.full(shape, 0. if band else -1,
                dtype=torch.float32 if band else torch.long), persistent=False)
        for name in LOOKUP_FIELDS:
            atoms = name == 'lookup_atoms'
            shape = (self.streams, self.prepend+2*self.words, *((int(width),) if atoms else ()))
            self.register_buffer(name, torch.full(shape, 0. if atoms else -1,
                dtype=torch.float32 if atoms else torch.long), persistent=False)
        self.register_buffer('timelines', torch.zeros(self.streams, self.prepend, dtype=torch.long), persistent=False)
        reconcile_cap = max(128, 2 * self.prepend)
        self.register_buffer('_reconcile_keys', torch.zeros(reconcile_cap, 2, dtype=torch.long), persistent=False)
        self.register_buffer('_reconcile_vals', torch.zeros(reconcile_cap, 2, dtype=torch.long), persistent=False)
        self.register_buffer('_host_valid', torch.zeros(self.streams, self.prepend, dtype=torch.bool), persistent=False)
        self.register_buffer('_host_ids', torch.full((self.streams, self.prepend), -1, dtype=torch.long), persistent=False)
        self.register_buffer('_host_rows', torch.full((self.streams, self.prepend), -1, dtype=torch.long), persistent=False)
        self.register_buffer('_host_orders', torch.full((self.streams, self.prepend), -1, dtype=torch.long), persistent=False)
        self.register_buffer('_host_timelines', torch.zeros(self.streams, self.prepend, dtype=torch.long), persistent=False)
        self.register_buffer('_host_refs', torch.zeros(self.streams, self.prepend, 2, 2, dtype=torch.long), persistent=False)
        self.register_buffer('_host_wheres', torch.zeros(self.streams, self.prepend, WHERE_WIRES, dtype=torch.float32), persistent=False)
        self.register_buffer('_host_values', torch.zeros(self.streams, self.prepend, int(width), dtype=torch.float32), persistent=False)
        self.register_buffer('_host_needs_stamp', torch.zeros(self.streams, self.prepend, dtype=torch.bool), persistent=False)

    def stage_words(self, name, value):
        target = getattr(self, name)
        if torch.compiler.is_compiling():
            fill = 0 if target.is_floating_point() else -1
            if value is None:
                setattr(self, name, torch.full_like(target, fill))
                self.extents.pop(name, None)
                return
            batch, count = value.shape[:2]
            if batch > self.streams or count+self.prepend > target.shape[1] or value.shape[2:] != target.shape[2:]:
                raise ValueError('symbol field exceeds its constructed base + W extent')
            if target.device != value.device or (target.is_floating_point() and target.dtype != value.dtype):
                target = target.to(value)
            pad = (0, 0) * (target.ndim - 2) + (self.prepend, target.shape[1] - (self.prepend + count), 0, target.shape[0] - batch)
            staged = F.pad(value.to(dtype=target.dtype, device=target.device), pad, value=fill)
            setattr(self, name, staged)
            self.extents[name] = (batch, count)
            return
        if target.requires_grad:
            # An earlier sentence can still own a gather of this bank. A
            # refreshed bank keeps its fixed contiguous shape, but cannot
            # overwrite storage saved by that sentence's backward pass.
            target = torch.empty_like(target)
            setattr(self, name, target)
        fill = 0 if target.is_floating_point() else -1
        if value is None:
            target.detach_()
            target.fill_(fill)
            self.extents.pop(name, None)
            return
        batch, count = value.shape[:2]
        if batch > self.streams or count+self.prepend > target.shape[1] or value.shape[2:] != target.shape[2:]:
            raise ValueError('symbol field exceeds its constructed base + W extent')
        if target.device != value.device or (target.is_floating_point() and target.dtype != value.dtype):
            target = target.to(value)
            setattr(self, name, target)
        target.detach_()
        target.fill_(fill)
        target[:batch, self.prepend:self.prepend+count].copy_(value)
        self.extents[name] = (batch,count)

    def read(self, name, *, words=False):
        extent = self.extents.get(name)
        if extent is None:
            return None
        batch, count = extent
        return getattr(self, name)[:batch, self.prepend if words else 0:self.prepend+count]


def refresh_words(model, sentence):
    """Resolve this sentence's staged names after the preceding kept close.

    Word perception can stage a whole packed batch. Its temporary object
    addresses must not outlive promotion or reuse of an MTM seat. Read the
    lexical identity and current token instead of retaining an alias table.
    """
    source, owner = model.inputSpace, model._concept_owner()
    words, references = word_view(source, 'concept_ids'), word_view(source, 'object_ids')
    if not (owner.interpret.mandatory and torch.is_tensor(words)
            and torch.is_tensor(references)):
        return
    rows, orders = word_view(source, 'object_rows'), word_view(source, 'object_orders')
    grammar_rows = getattr(source, '_ar_grammar_object_rows', None)
    mixing = torch.is_tensor(grammar_rows)
    active = (source._packed_sentence_ids == sentence) & (words > 0) & (references > 0)
    identities = owner.interpret.sentence_referents(torch.where(active, words, -1))
    changed = active & (identities != references)
    if not torch.compiler.is_compiling() and not bool(changed.any()):
        return
    from Meaning import require_sentence_tensor
    store = owner._concept_allocator.layer(0)
    resolved = store.concept_rows(identities)
    seated = resolved >= 0
    selected = torch.where(seated, resolved, store.concept_rows(words))
    require_sentence_tensor((~changed | (selected >= 0)).all(), 'unseated lexical reference')
    ends = selected.new_tensor(owner._order_caps()).cumsum(0)
    levels = torch.searchsorted(ends, selected.contiguous(), right=True)
    references.copy_(torch.where(changed, identities, references))
    if torch.is_tensor(rows):
        rows.copy_(torch.where(changed, selected, rows))
        orders.copy_(torch.where(changed, levels, orders))
    if mixing:
        grammar_rows.copy_(torch.where(changed, selected, grammar_rows))
        levels_out = source._ar_grammar_object_orders
        levels_out.copy_(torch.where(changed, levels, levels_out))
        names = source._ar_grammar_object_ids
        names.copy_(torch.where(changed, torch.where(seated, identities, words), names))

    def codes(selected, word_ids=None):
        values = owner.similarity_codebook.lookup_rows(selected.clamp_min(0), streams=True)
        values = owner.interpret.lexical_atoms(selected, values, word_ids=word_ids)
        return torch.where(selected[..., None] >= 0, values, 0.)

    if mixing:
        source._ar_grammar_object_atoms = codes(grammar_rows, words)
    elif torch.is_tensor(rows):
        source._ar_word_object_atoms = codes(rows, words)
    bank = lookup_view(source, 'lookup_rows')
    if torch.is_tensor(bank):
        # Aligned perception's bank is WORD followed by OBJECT. Mixing's
        # word-only inverse bank does not change when its referent changes.
        offset = bank.shape[1] // 2
        if not mixing:
            objects = bank[:, offset:offset + words.shape[1]]
            objects.copy_(torch.where(changed, selected, objects))
        lexical = torch.full_like(bank, -1)
        lexical[:, :words.shape[1]] = words
        if not mixing:
            lexical[:, offset:offset+words.shape[1]] = words
        source.symbol_entries.stage_words('lookup_atoms', codes(bank, lexical))


def symbolize(owner, store, key, token, timeline=0):
    """The name/mention form beside the selected timeline's working meaning."""
    from IdentityRows import point
    if token.order is not None:
        return owner.mtm.value_of(key, timeline=timeline, live_candidate=True)
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    # Legacy numerical fixtures have no separate form/meaning chart; their
    # entire point is content. The paired model supplies both explicit widths.
    paired = derived is not None and derived.meaning_pairs > 0
    width = derived.percept_event_width if paired else 0
    form_width = derived.percept_width if paired else 0
    form = None
    for occurrence in reversed(token.occurrences):
        reading = owner.mtm.reading_of(occurrence)
        if reading is not None:
            name = point(owner, reading)
            if name is not None:
                form = name[:form_width]
                break
    if form is None:
        forms = [store.slots[row, :form_width] for occurrence in token.occurrences
                 if isinstance((row := store.location(occurrence)), int)]
        form = torch.stack(forms).mean(0) if forms else owner.similarity_codebook.W.new_zeros(form_width)
    value = owner.mtm.value_of(key, timeline=timeline).detach().clone()
    value[:form_width] = form.detach().to(value)
    value[form_width:width] = 0
    from VerbLearning import reread
    return reread(owner, token, timeline, value)


def chunk_token(pool, store, state, batch, index):
    """Resolve a seat through its retained occurrence before a recycled id.

    A published identity uses its second reference for the pool origin. Recent
    entries already carry an occurrence in their first reference. No epoch or
    additional identity/value slab is needed to reject a reused transient id.
    """
    from Memory import STMChunks as chunks
    refs = state[chunks.REFS][batch, index]
    identity = state[chunks.IDENTITY][batch, index].reshape(1)
    # The MTM pool takes Python keys. Read the seat once for both tagged
    # references and its fallback identity.
    return token_from_references(pool, store, *torch.cat((refs.flatten(), identity)).tolist())


def token_from_references(pool, store, first_kind, first, second_kind, second, identity):
    from Memory import STMChunks as chunks
    if second_kind == chunks.IDENTITY_REF:
        key = ('candidate', second)
        return key if key in pool.entries else None
    for kind, address in ((second_kind, second), (first_kind, first)):
        if kind == chunks.OCCURRENCE:
            return pool.token_of(('ltm', store.address_domain, address))
    return pool.token_of(identity) if identity >= 0 else None


def reconcile(state, pool, store):
    """Retire expired field identities and remap promoted seats at staging."""
    from Memory import STMChunks as chunks
    refs = state[chunks.REFS]
    first, second = refs.unbind(-2)
    candidate = second[..., 0] == chunks.IDENTITY_REF
    occurrence = (second[..., 0] == chunks.OCCURRENCE) | (first[..., 0] == chunks.OCCURRENCE)
    eligible = candidate | occurrence
    addresses = torch.where(candidate | (second[..., 0] == chunks.OCCURRENCE), second[..., 1], first[..., 1])

    # Persistent names and event aliases are owned by MTM. Index them once
    # in token_of's precedence order, then resolve every live seat on device.
    aliases = {}
    for key, entry in pool.entries.items():
        for timeline in entry.timelines:
            for event in timeline.events:
                if event.role == 0:
                    aliases.setdefault(event.occurrence, key)
    aliases.update((origin, name.token) for origin, name in pool.names.items())
    aliases.update((key, key) for key in pool.entries)
    names, records = [], []
    for name, key in aliases.items():
        if not isinstance(name, tuple):
            continue
        if len(name) == 2 and name[0] == 'candidate' and name == key:
            tag, address = chunks.IDENTITY_REF, name[1]
        elif len(name) == 3 and name[:2] == ('ltm', store.address_domain):
            tag, address = chunks.OCCURRENCE, name[2]
        else:
            continue
        entry = pool.entries.get(key)
        if entry is None or entry.refinement.get('member_kind', 'reference') != 'reference':
            continue
        names.append((tag, address))
        records.append((chunks.IDENTITY_REF, entry.transient) if entry.symbol is None
                       else (chunks.SYMBOL, entry.symbol))
    selected = torch.zeros_like(first)
    if names:
        ids = refs.new_tensor(names)
        tags = torch.where(candidate, chunks.IDENTITY_REF, chunks.OCCURRENCE)
        matches = (ids[:, 0] == tags[..., None]) & (ids[:, 1] == addresses[..., None])
        values = refs.new_tensor(records)[matches.int().argmax(-1)]
        selected = torch.where(matches.any(-1, keepdim=True), values, 0)
    live = selected[..., 0] != 0
    write_ref = eligible & ((first[..., 0] == chunks.SYMBOL) | (first[..., 0] == chunks.IDENTITY_REF))
    result = [value.clone() for value in state]
    result[chunks.REFS] = torch.stack((torch.where(write_ref[..., None], selected, first), second), -2)
    result[chunks.IDENTITY] = torch.where(eligible, torch.where(live, selected[..., 1], -1),
                                          state[chunks.IDENTITY])
    return tuple(result)


def _reconcile_tensor(state, entries):
    """Compiled fullgraph tensor reconciliation over pre-staged MTM alias tables."""
    from Memory import STMChunks as chunks
    refs = state[chunks.REFS]
    first, second = refs.unbind(-2)
    candidate = second[..., 0] == chunks.IDENTITY_REF
    occurrence = (second[..., 0] == chunks.OCCURRENCE) | (first[..., 0] == chunks.OCCURRENCE)
    eligible = candidate | occurrence
    addresses = torch.where(candidate | (second[..., 0] == chunks.OCCURRENCE), second[..., 1], first[..., 1])
    keys = entries._reconcile_keys
    vals = entries._reconcile_vals
    tags = torch.where(candidate, chunks.IDENTITY_REF, chunks.OCCURRENCE)
    matches = (keys[:, 0] == tags[..., None]) & (keys[:, 1] == addresses[..., None]) & (keys[:, 0] != 0)
    values = vals[matches.long().argmax(-1)]
    selected = torch.where(matches.any(-1, keepdim=True), values, 0)
    live = selected[..., 0] != 0
    write_ref = eligible & ((first[..., 0] == chunks.SYMBOL) | (first[..., 0] == chunks.IDENTITY_REF))
    result = list(state)
    result[chunks.REFS] = torch.stack((torch.where(write_ref[..., None], selected, first), second), -2)
    result[chunks.IDENTITY] = torch.where(eligible, torch.where(live, selected[..., 1], -1),
                                          state[chunks.IDENTITY])
    return tuple(result)


def prepare_host_stage(model, batch_size):
    """Snapshot MTM alias and prepended-symbol tables onto pre-allocated tensors on host."""
    owner, isp = model._concept_owner(), model.inputSpace
    entries = getattr(isp, 'symbol_entries', None)
    if entries is None or owner is None or not hasattr(model, 'symbolSpace'):
        return
    store = model.symbolSpace.ltm_store
    pool = owner.mtm
    from Memory import STMChunks as chunks
    device = entries.concept_ids.device
    # 1. Alias table for reconcile()
    aliases = {}
    for key, entry in pool.entries.items():
        for timeline in entry.timelines:
            for event in timeline.events:
                if event.role == 0:
                    aliases.setdefault(event.occurrence, key)
    aliases.update((origin, name.token) for origin, name in pool.names.items())
    aliases.update((key, key) for key in pool.entries)
    names_list, records_list = [], []
    for name, key in aliases.items():
        if not isinstance(name, tuple):
            continue
        if len(name) == 2 and name[0] == 'candidate' and name == key:
            tag, address = chunks.IDENTITY_REF, name[1]
        elif len(name) == 3 and name[:2] == ('ltm', store.address_domain):
            tag, address = chunks.OCCURRENCE, name[2]
        else:
            continue
        entry = pool.entries.get(key)
        if entry is None or entry.refinement.get('member_kind', 'reference') != 'reference':
            continue
        names_list.append((tag, address))
        records_list.append((chunks.IDENTITY_REF, entry.transient) if entry.symbol is None
                            else (chunks.SYMBOL, entry.symbol))
    needed_aliases = len(names_list)
    if needed_aliases > len(entries._reconcile_keys):
        cap = max(needed_aliases, 2 * len(entries._reconcile_keys))
        entries._reconcile_keys = entries._reconcile_keys.new_zeros(cap, 2)
        entries._reconcile_vals = entries._reconcile_vals.new_zeros(cap, 2)
    entries._reconcile_keys.zero_()
    entries._reconcile_vals.zero_()
    if needed_aliases:
        entries._reconcile_keys[:needed_aliases].copy_(
            torch.tensor(names_list, dtype=torch.long, device=device))
        entries._reconcile_vals[:needed_aliases].copy_(
            torch.tensor(records_list, dtype=torch.long, device=device))

    # 2. Per-stream prepended MTM candidate choices
    entries._host_valid.zero_()
    entries._host_ids.fill_(-1)
    entries._host_rows.fill_(-1)
    entries._host_orders.fill_(-1)
    entries._host_timelines.zero_()
    entries._host_refs.zero_()
    entries._host_wheres.zero_()
    entries._host_values.zero_()
    entries._host_needs_stamp.zero_()
    if not pool.entries:
        return
    B = min(int(batch_size), entries.streams)
    P = entries.prepend
    for b in range(B):
        keys = {
            key for key, entry in pool.entries.items()
            if (entry.order is None
                and entry.refinement.get('member_kind', 'reference') == 'reference'
                and pool.in_window(key, b)
                and any(isinstance(store.location(occ), int) for occ in entry.occurrences))
        }
        keys.update(
            key for key, token in pool.entries.items()
            if token.order is not None and (not pool.accesses.get(key) or b in pool.accesses[key])
        )
        if not keys:
            continue
        ordered_keys = sorted(
            (key for key in pool.entries if key in keys),
            key=lambda key: (
                max((event.when for timeline in pool.entries[key].timelines
                     for event in timeline.events), default=pool.entries[key].when),
                pool.entries[key].when),
            reverse=True,
        )
        choices = [(key, tl) for key in ordered_keys for tl in range(len(pool.entries[key].timelines))][:P]
        if not choices:
            continue
        ids_b, rows_b, orders_b, tls_b, refs_b, wheres_b, values_b, stamp_b = [], [], [], [], [], [], [], []
        for key, timeline in choices:
            token = pool.entries[key]
            identity = token.symbol if token.symbol is not None else token.transient
            if identity is None:
                raise RuntimeError('a prepended identity has neither a transient id nor a symbol')
            row = owner._concept_row_of(identity) if token.symbol is not None else None
            row_id = -1 if row is None else row
            order = (token.order if token.order is not None else owner._row_order(row) if row is not None
                     else token.refinement['member_order'] if 'member_order' in token.refinement
                     else owner._concept_source_order(token.occurrences[0]))
            value = symbolize(owner, store, key, token, timeline).detach().to(entries._host_values)
            where = token.where.detach().to(entries._host_wheres)
            ref = [[chunks.SYMBOL if token.symbol is not None else chunks.IDENTITY_REF, int(identity)],
                   [chunks.IDENTITY_REF if token.order is not None else chunks.OCCURRENCE, int(key[-1])]]
            ids_b.append(int(identity))
            rows_b.append(int(row_id))
            orders_b.append(int(order))
            tls_b.append(int(timeline))
            refs_b.append(ref)
            wheres_b.append(where)
            values_b.append(value)
            stamp_b.append(bool(store.when_start is not None and token.order is None))
        k = len(choices)
        entries._host_valid[b, :k] = True
        entries._host_ids[b, :k] = entries._host_ids.new_tensor(ids_b)
        entries._host_rows[b, :k] = entries._host_rows.new_tensor(rows_b)
        entries._host_orders[b, :k] = entries._host_orders.new_tensor(orders_b)
        entries._host_timelines[b, :k] = entries._host_timelines.new_tensor(tls_b)
        entries._host_refs[b, :k] = entries._host_refs.new_tensor(refs_b)
        entries._host_wheres[b, :k] = torch.stack(wheres_b, 0)
        entries._host_values[b, :k] = torch.stack(values_b, 0)
        entries._host_needs_stamp[b, :k] = entries._host_needs_stamp.new_tensor(stamp_b)


def _stage_tensor(model, state, present):
    """Compiled fullgraph tensor staging of prepended MTM symbols."""
    from Memory import STMChunks as chunks
    isp = model.inputSpace
    entries = isp.symbol_entries
    store = model.symbolSpace.ltm_store
    state = _reconcile_tensor(state, entries)
    stamp = model.present_code().to(state[0])
    B = state[0].shape[0]
    P = entries.prepend
    device = state[0].device
    valid = entries._host_valid[:B] & present[:, None]
    ids = entries._host_ids[:B]
    rows = entries._host_rows[:B]
    orders = entries._host_orders[:B]
    timelines = entries._host_timelines[:B]
    refs = entries._host_refs[:B]
    wheres = entries._host_wheres[:B].to(state[0])
    values = entries._host_values[:B].to(state[0])
    needs_stamp = entries._host_needs_stamp[:B]
    if store.when_start is not None:
        ws = int(store.when_start)
        stamped_vals = values.clone()
        stamped_vals[..., ws:ws + 4] = torch.where(
            needs_stamp[..., None], stamp.expand(B, P, 4), stamped_vals[..., ws:ws + 4])
        if ws >= WHERE_WIRES:
            stamped_vals[..., ws - WHERE_WIRES:ws] = torch.where(
                needs_stamp[..., None], wheres, stamped_vals[..., ws - WHERE_WIRES:ws])
        values = stamped_vals

    # Vectorized write of leading P columns on SymbolEntries
    def _write_prefix(tensor, prefix_vals, fill_val):
        full_prefix = tensor.new_full((entries.streams, P, *tensor.shape[2:]), fill_val)
        masked = torch.where(valid.reshape(B, P, *([1] * (tensor.ndim - 2))),
                             prefix_vals.to(tensor), fill_val)
        full_prefix = torch.cat((masked, full_prefix[B:]), dim=0) if B < entries.streams else masked
        return torch.cat((full_prefix, tensor[:, P:]), dim=1)

    for name, src in (('concept_ids', ids), ('object_ids', ids),
                      ('concept_rows', rows), ('object_rows', rows),
                      ('concept_orders', orders), ('object_orders', orders),
                      ('lookup_rows', rows)):
        setattr(entries, name, _write_prefix(getattr(entries, name), src, -1))
    entries.symbol_where = _write_prefix(entries.symbol_where, wheres, 0.)
    entries.symbol_when = _write_prefix(entries.symbol_when, stamp.expand(B, P, 4), 0.)
    entries.lookup_atoms = _write_prefix(entries.lookup_atoms, values, 0.)
    sid_fill = torch.full_like(ids, int(model._open_sentence_slot))
    entries.lookup_sentence_ids = _write_prefix(entries.lookup_sentence_ids, sid_fill, -1)
    tl_full = torch.where(valid, timelines, 0)
    if B < entries.streams:
        tl_full = torch.cat((tl_full, entries.timelines[B:]), dim=0)
    entries.timelines = tl_full

    staged0 = torch.zeros_like(state[chunks.READ], dtype=torch.bool)
    stamp_b = stamp.expand(B, 4)
    col_positions = torch.arange(state[0].shape[1], device=device)[None]

    def cond(c, staged, *st):
        return (c < P) & valid.index_select(1, c.clamp_max(P - 1).unsqueeze(0)).squeeze(1).any()

    def body(c, staged, *st):
        col = c.clamp_max(P - 1).unsqueeze(0)
        gate = valid.index_select(1, col).squeeze(1)
        val_c = values.index_select(1, col).squeeze(1)
        where_c = wheres.index_select(1, col).squeeze(1)
        id_c = ids.index_select(1, col).squeeze(1)
        tl_c = timelines.index_select(1, col).squeeze(1)
        order_c = orders.index_select(1, col).squeeze(1)
        row_c = rows.index_select(1, col).squeeze(1)
        ref_c = refs.index_select(1, col).squeeze(1)
        match = ((st[chunks.IDENTITY] == id_c[:, None])
                 & (st[chunks.TIMELINE] == tl_c[:, None])
                 & (st[chunks.STACK] < 0)
                 & ((st[chunks.REFS][:, :, 0, 0] == chunks.SYMBOL)
                    | (st[chunks.REFS][:, :, 0, 0] == chunks.IDENTITY_REF)))
        has_match = match.any(-1)
        match_idx = match.long().argmax(-1)
        gate_ex = gate & has_match
        st_ex = list(st)
        for field, member in ((chunks.BUFFER, val_c), (chunks.WHERE, where_c),
                              (chunks.WHEN, stamp_b), (chunks.REFS, ref_c),
                              (chunks.ORDERS, order_c), (chunks.ROWS, row_c)):
            st_ex[field] = chunks.write(st_ex[field], match_idx, member, gate_ex)
        st_ex = chunks.read(tuple(st_ex), match_idx, gate_ex)
        gate_dep = gate & ~has_match
        st_dep, dep_idx = chunks.deposit(
            st_ex, val_c, gate_dep,
            order=order_c, row=row_c, refs=ref_c, identity=id_c,
            timeline=tl_c, where=where_c, when=stamp_b)
        idx = torch.where(gate_ex, match_idx, torch.where(gate_dep, dep_idx, -1))
        staged = staged | ((col_positions == idx[:, None]) & (idx[:, None] >= 0) & gate[:, None])
        return (c + 1, staged, *st_dep)

    _, staged, *state = torch.while_loop(cond, body, (torch.zeros((), dtype=torch.long, device=device), staged0, *state))
    indices = col_positions.expand_as(staged)
    return chunks.read(tuple(state), indices, staged)


def stage(model, state, present):
    """Host boundary: prepend live identities, then deposit them off stack."""
    if torch.compiler.is_compiling():
        return _stage_tensor(model, state, present)
    from Memory import STMChunks as chunks
    owner, isp = model._concept_owner(), model.inputSpace
    entries = isp.symbol_entries
    store, pool = model.symbolSpace.ltm_store, owner.mtm
    state = reconcile(state,pool,store)
    resolver = owner.components.reference_close
    stamp = model.present_code().to(state[0])
    for field in (*WORD_FIELDS, *LOOKUP_FIELDS):
        value = getattr(entries, field)
        if value.requires_grad:
            value = value.clone()
            setattr(entries, field, value)
        value[:, :entries.prepend].fill_(0 if value.is_floating_point() else -1)
    entries.timelines.zero_()
    staged = torch.zeros_like(state[chunks.READ], dtype=torch.bool)
    for b in range(state[0].shape[0]):
        if not bool(present[b]):
            continue
        records = resolver.accessible(tuple(value[b:b+1] for value in state), stream=b, include_pool=True)
        keys = {record.get('token') for _, record in records.values()} - {None}
        keys.update(key for key, token in pool.entries.items() if token.order is not None
                    and (not pool.accesses.get(key) or b in pool.accesses[key]))
        # Equal-time arrivals retain admission order. Sorting the set itself
        # would make bind-menu positions and training depend on Python's hash.
        keys = sorted((key for key in pool.entries if key in keys), key=lambda key: (
            max((event.when for timeline in pool.entries[key].timelines
                 for event in timeline.events), default=pool.entries[key].when),
            pool.entries[key].when), reverse=True)
        choices = [(key, timeline) for key in keys for timeline in range(len(pool.entries[key].timelines))][:entries.prepend]
        if not choices:
            continue
        batch_ids, batch_rows, batch_orders = [], [], []
        batch_values, batch_wheres = [], []
        gate = torch.arange(len(present), device=state[0].device) == b
        for column, (key, timeline) in enumerate(choices):
            token = pool.entries[key]
            identity = token.symbol if token.symbol is not None else token.transient
            if identity is None:
                raise RuntimeError('a prepended identity has neither a transient id nor a symbol')
            row = owner._concept_row_of(identity) if token.symbol is not None else None
            row_id = -1 if row is None else row
            order = (token.order if token.order is not None else owner._row_order(row) if row is not None
                     else token.refinement['member_order'] if 'member_order' in token.refinement
                     else owner._concept_source_order(token.occurrences[0]))
            value = symbolize(owner, store, key, token, timeline).to(state[0])
            where = token.where.to(value)
            if store.when_start is not None and token.order is None:
                value[store.when_start:store.when_start+4] = stamp
                if store.when_start >= WHERE_WIRES:
                    value[store.when_start-WHERE_WIRES:store.when_start] = where
            batch_ids.append(identity)
            batch_rows.append(row_id)
            batch_orders.append(order)
            batch_values.append(value)
            batch_wheres.append(where)
            refs = state[chunks.REFS].new_zeros(len(present), 2, 2)
            refs[:, 0, 0] = chunks.SYMBOL if token.symbol is not None else chunks.IDENTITY_REF
            refs[:, 0, 1] = identity
            refs[:, 1, 0] = chunks.IDENTITY_REF if token.order is not None else chunks.OCCURRENCE
            refs[:, 1, 1] = key[-1]
            # Copying the working value from MTM is an arrival/read. Merely
            # enumerating the subsequent bind menu does not read this seat.
            match = ((state[chunks.IDENTITY][b] == identity) & (state[chunks.TIMELINE][b] == timeline)
                     & (state[chunks.STACK][b] < 0)
                     & ((state[chunks.REFS][b, :, 0, 0] == chunks.SYMBOL)
                        | (state[chunks.REFS][b, :, 0, 0] == chunks.IDENTITY_REF)))
            if bool(match.any()):
                index = torch.full((len(present),), int(match.long().argmax()), device=state[0].device)
                result = list(state)
                for field, member in ((chunks.BUFFER, value), (chunks.WHERE, where), (chunks.WHEN, stamp), (chunks.REFS, refs[b]),
                        (chunks.ORDERS, state[chunks.ORDERS].new_tensor(order)),
                        (chunks.ROWS, state[chunks.ROWS].new_tensor(row_id))):
                    result[field] = chunks.write(result[field], index, member.expand(len(present), *member.shape), gate)
                state = tuple(result)
                state = chunks.read(state, index, gate)
            else:
                state, index = chunks.deposit(state, value.expand(len(present), -1), gate,
                    order=order, row=row_id, refs=refs, identity=identity, timeline=timeline,
                    where=where, when=stamp)
            if int(index[b]) >= 0:
                staged[b, index[b]] = True
        width_k = len(choices)
        ids_t = entries.concept_ids.new_tensor(batch_ids)
        rows_t = entries.concept_rows.new_tensor(batch_rows)
        orders_t = entries.concept_orders.new_tensor(batch_orders)
        entries.concept_ids[b, :width_k] = ids_t
        entries.object_ids[b, :width_k] = ids_t
        entries.concept_rows[b, :width_k] = rows_t
        entries.object_rows[b, :width_k] = rows_t
        entries.concept_orders[b, :width_k] = orders_t
        entries.object_orders[b, :width_k] = orders_t
        entries.lookup_rows[b, :width_k] = rows_t
        entries.symbol_where[b, :width_k] = torch.stack(batch_wheres, 0)
        entries.symbol_when[b, :width_k] = stamp
        entries.lookup_atoms[b, :width_k] = torch.stack(batch_values, 0)
        entries.lookup_sentence_ids[b, :width_k] = model._open_sentence_slot
        entries.timelines[b, :width_k] = entries.timelines.new_tensor([tl for _, tl in choices])
    # The leading field arrives at one boundary. Its order in the array is
    # event order, not an artificial sequence of attention reads.
    indices = torch.arange(state[0].shape[1],device=state[0].device)[None].expand_as(staged)
    state = chunks.read(state,indices,staged)
    return state
