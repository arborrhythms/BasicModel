"""Bounded content and percept-time reads, without a lexical membership index."""
import math
import torch
from torch.nn import functional as F
from Occurrence import AddressIndex


class EntryRead:
    def rebuild_addresses(self):
        self._index_occurrences = AddressIndex({int(key): row for row, key in
                                              enumerate(self.address_keys[:len(self)].tolist())})

    def entry_when(self, index):
        from RowTime import read_stamp
        return read_stamp(self, index)

    @staticmethod
    def _scope_contains(query, stored):
        stored = dict(stored)
        for key, wanted in dict(query).items():
            actual = stored.get(key)
            if key in ('where', 'when') and isinstance(wanted, tuple) and len(wanted) == 2:
                if not isinstance(actual, tuple) or len(actual) != 2 or not wanted[0] <= actual[0] <= actual[1] <= wanted[1]:
                    return False
            elif actual != wanted:
                return False
        return True

    @torch.no_grad()
    def cued_rows(self, cue, *, primed=(), references=(), retrieved=(), max_candidates=None,
                  max_records=32, work=None, stream=None, role_mask=None):
        """Score one fixed window; charge only the returned, ranked records."""
        max_candidates = self.ltm_window if max_candidates is None else max_candidates
        for name, value in (('candidate', max_candidates), ('record', max_records)):
            if type(value) is not int or value < 0:
                raise ValueError(name + ' limit must be a nonnegative integer')
        if max_candidates > self.ltm_window:
            raise OverflowError('candidate limit exceeds constructed ltmWindow')
        from Memory import SYMBOL
        explicit = [self.location(ref) for ref in (*references, *retrieved)]
        for ref in cue.role_refs:
            if ref is not None and ref[0] == 'sym' and type(ref[1]) is int and ref[1] > 0:
                explicit.extend(('sym', row) for row in self.parts.rows_referencing(SYMBOL, ref[1]))
        candidates = list(dict.fromkeys(row for row in explicit if row is not None
            and (not isinstance(row, int) or self.kind_of(row) != 'episode')))
        present = set(candidates)
        # LTM0 compaction preserves physical row order; re-witnessing keeps
        # the original stamp. Appends occupy the next seat after survivors.
        candidates.extend(row for row in reversed(range(len(self)))
                          if row not in present and self.kind_of(row) != 'episode')
        relations = self.parts.indices()
        if len(relations):
            # Relation handles can reuse freed seats, so their stamps, not
            # handle magnitude, determine recency. Unstamped ties use address.
            from RowTime import encoder
            times = encoder().decode_index(self.parts.percept[relations, 4:]).tolist()
            ordered = sorted(zip(times, relations.tolist()), reverse=True)
            candidates.extend(('sym', row) for _, row in ordered if ('sym', row) not in present)
        incomplete = ['ltm_window'] if len(candidates) > max_candidates else []
        candidates = candidates[:max_candidates]
        scanned = len(candidates)
        if not scanned:
            return dict(value=(), records_scanned=0, incomplete=tuple(incomplete))
        occupied = (cue.role_mask if role_mask is None else role_mask).to(self.slots.device)
        roles, masks, available = self._cue_roles, self._cue_role_mask, self._cue_available
        roles.zero_(); masks.zero_(); available.zero_()
        # Occurrence kind and question evidence are tensor-valued filters.
        occurrence_slots = [slot for slot, index in enumerate(candidates) if isinstance(index, int)]
        occurrence_rows = [index for index in candidates if isinstance(index, int)]
        if occurrence_rows:
            rows = torch.tensor(occurrence_rows, device=self.slots.device)
            kinds = self.kind_codes(rows)
            available[occurrence_slots] = ((kinds != self.KINDS.index('estimate')) &
                (kinds != self.KINDS.index('cover')) &
                (kinds != self.KINDS.index('episode')) &
                ((kinds != self.KINDS.index('question')) | self.truth_plus[rows].ne(0) | self.truth_minus[rows].ne(0)))
        records = []
        for slot, index in enumerate(candidates):
            record = self.row(index)
            records.append(record)
            meaning = record['meaning']
            if isinstance(index, tuple):
                available[slot] = (record['kind'] not in ('estimate', 'cover', 'episode') and
                    (record['kind'] != 'question' or any(record['evidence'])))
            if meaning is None:
                if bool(available[slot]):
                    incomplete.append('unavailable_metadata')
                available[slot] = False
                continue
            roles[slot].copy_(meaning.roles)
            masks[slot].copy_(meaning.role_mask)
        wanted = cue.roles.to(roles).clone()
        if self.when_start is not None:
            wanted[:, self.when_start:self.when_start+4] = 0
            roles[:, :, self.when_start:self.when_start+4] = 0
        available &= ~(occupied[None] & ~masks).any(-1)
        # Exactly one batched product, one batch for each semantic role.
        similarities = torch.bmm(
            F.normalize(roles, dim=-1, eps=1e-8).transpose(0, 1),
            F.normalize(wanted, dim=-1, eps=1e-8).unsqueeze(-1)).squeeze(-1).transpose(0, 1)
        if bool(occupied.any()):
            matches = similarities.clamp(0, 1)[:, occupied].mean(-1)
        else:
            matches = torch.ones_like(similarities[:, 0])
        ranked = sorted(range(scanned), key=lambda slot: (-float(matches[slot]),
            (1, candidates[slot][1]) if isinstance(candidates[slot], tuple) else (0, candidates[slot])))
        constraints = {key: value for key, value in cue.bindings if not key.startswith('_')}
        found = []
        for slot in ranked:
            if len(found) >= max_records:
                break
            if not bool(available[slot]):
                continue
            record = records[slot]
            meaning = record['meaning']
            if not self._scope_contains(cue.scope, meaning.scope) or not self._scope_contains(constraints, meaning.bindings):
                continue
            if work is not None and not work.consume('record'):
                incomplete.append('work_budget')
                break
            found.append(dict(record, index=candidates[slot], match=float(matches[slot])))
        return dict(value=tuple(found), records_scanned=scanned, incomplete=tuple(dict.fromkeys(incomplete)))

    def recall_time(self, when, *, max_records, work, window=1):
        """Nearest sentence stamps in a bounded clock window; every read is metered."""
        from RowTime import encoder
        when = torch.as_tensor(when, device=self.slots.device, dtype=self.slots.dtype)
        if when.shape != (4,) or not bool(torch.isfinite(when).all()) or not bool(when.ne(0).any()):
            raise ValueError('time recall requires a nonzero finite percept time')
        if not math.isfinite(float(window)) or window < 0:
            raise ValueError('time recall window must be bounded and nonnegative')
        target = int(encoder().decode_index(when))
        candidates = [*range(len(self)), *(('sym', row) for row in self.parts.indices().tolist())]
        found, scanned, incomplete = [], 0, []
        for index in candidates:
            if scanned >= max_records or not work.consume('record'):
                incomplete.append('time_read_limit'); break
            scanned += 1
            stamp = self.entry_when(index)
            if not bool(stamp.ne(0).any()):
                continue
            if isinstance(index, int) and self.kind_of(index) in ('cover', 'estimate'):
                continue
            distance = abs(int(encoder().decode_index(stamp))-target)
            if distance <= window:
                found.append(dict(self.row(index), index=index, time_distance=distance, match=1. / (1+distance)))
        nearest = min((row['time_distance'] for row in found), default=None)
        return dict(value=tuple(row for row in found if row['time_distance'] == nearest),
                    records_scanned=scanned, incomplete=tuple(incomplete))

    def recall_identities(self, frames, *, work, stream=0):
        """Re-enter chain identities in MTM; the returned head is the STM read."""
        owner_ref = getattr(self, '_concept_owner', None)
        if not callable(owner_ref):
            return ()
        owner = owner_ref()
        from Memory import OCCURRENCE, RELATION
        from ChainRead import envelope
        from RowTime import encoder
        missing = []
        for frame in frames:
            if frame['kind'] == 'episode':
                continue
            pending, seen = [frame['occurrence']], set()
            tick = int(encoder().decode_index(frame['when']))
            while pending:
                reference = pending.pop()
                if reference in seen:
                    continue
                seen.add(reference)
                if not work.consume('record'):
                    missing.append('identity_recall_limit'); break
                if reference[0] == 'sym' and self.parts.contains(reference[1]):
                    pending.extend(envelope(self, item) for item in self.parts.references(reference[1]))
                    continue
                if reference[0] != 'ltm':
                    continue
                row = self.location(reference)
                if row is None:
                    missing.append('unavailable_identity'); continue
                data = dict(self._semantic_rows.get(int(self.address_keys[row]), {}).get('bindings', ()))
                identity = data.get('_identity')
                if identity is not None and identity[0] == 'ltm':
                    pending.append(identity)
                if owner.mtm.token_of(reference) is None:
                    kind = identity[1] if identity is not None and identity[0] == 'sym' else None
                    where = self.slots[row,self.when_start-4:self.when_start] if self.when_start is not None and self.when_start >= 4 else None
                    native = kind if kind in owner.components.individual_ids else None
                    pool = owner.mtm
                    existing = next((key for key, entry in pool.entries.items()
                                     if native is not None and entry.native == native), None)
                    if existing is None:
                        existing = pool.open(reference, kind=kind, when=tick, where=where,
                            native=native, value=self.slots[row], reading=kind,
                            pending=kind is None and bool(data.get('_open_references') or data.get('_forward_references')))
                    else:
                        pool.bind(existing, reference, kind=kind, when=tick, where=where,
                            value=self.slots[row], reading=kind)
                owner.mtm.record_link(reference, reference, stream, kind='reference', order=self.order_of_row(row))
        return tuple(dict.fromkeys(missing))

    def recall_chunks(self, frames, *, work, stream=0):
        """Load returned heads into the same eight chunks, retaining their stamp."""
        missing = list(self.recall_identities(frames,work=work,stream=stream))
        owner_ref = getattr(self,'_concept_owner',None)
        owner = owner_ref() if callable(owner_ref) else None
        stm = getattr(owner,'stm',None)
        if stm is None:
            return tuple(missing)
        from Memory import STMChunks as chunks
        from ChainRead import head_idea
        state = stm.state
        if not 0 <= stream < state[0].shape[0]:
            raise ValueError('query stream exceeds the constructed STM batch')
        gate = torch.arange(state[0].shape[0],device=state[0].device) == stream
        for frame in frames:
            if frame['kind'] == 'episode':
                continue
            occurrence = frame['occurrence']
            location = self.location(occurrence)
            relation = isinstance(location,tuple)
            point = head_idea(self,location[1]) if relation else frame['point']
            if point is None:
                point = frame['meaning'].roles[0]
            point = point.detach().to(state[0]).clone()
            if self.when_start is not None:
                point[self.when_start:self.when_start+4] = frame['when'].to(point)
            refs = state[chunks.REFS].new_zeros(state[0].shape[0],2,2)
            refs[:,0,0], refs[:,0,1] = (chunks.RELATION if relation else chunks.OCCURRENCE),occurrence[-1]
            token = owner.mtm.token_of(occurrence)
            entry = None if token is None else owner.mtm.entries[token]
            identity = -1 if entry is None else entry.symbol if entry.symbol is not None else entry.transient
            where = point.new_zeros(4) if self.when_start is None or self.when_start < 4 else point[self.when_start-4:self.when_start]
            state,index = chunks.deposit(state,point.expand(state[0].shape[0],-1),gate,
                refs=refs,order=frame['order'],identity=identity,where=where,when=frame['when'].to(point),
                truth=frame['evidence'])
            if int(index[stream]) < 0:
                missing.append('stm_all_chunks_on_stack')
                break
        stm.set_state(state)
        return tuple(dict.fromkeys(missing))
