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

    def cued_rows(self, cue, *, primed=(), references=(), retrieved=(), max_candidates=32,
                  work=None, stream=None, role_mask=None):
        if type(max_candidates) is not int or max_candidates < 0:
            raise ValueError('candidate limit must be a nonnegative integer')
        from LTM1 import SYMBOL
        explicit = [self.location(ref) for ref in (*references, *retrieved)]
        for ref in cue.role_refs:
            if ref is not None and ref[0] == 'sym' and type(ref[1]) is int and ref[1] > 0:
                explicit.extend(('sym', row) for row in self.parts.rows_referencing(SYMBOL, ref[1]))
        candidates = list(dict.fromkeys(row for row in explicit if row is not None))
        candidates.extend(row for row in range(len(self)) if row not in candidates)
        candidates.extend(('sym', row) for row in self.parts.indices().tolist() if ('sym', row) not in candidates)
        occupied = cue.role_mask if role_mask is None else role_mask
        found, incomplete, scanned = [], [], 0
        for index in candidates:
            if scanned >= max_candidates or (work is not None and not work.consume('record')):
                incomplete.append('candidate_limit' if scanned >= max_candidates else 'work_budget')
                break
            scanned += 1
            record = self.row(index)
            if record['kind'] in ('estimate', 'cover') or (record['kind'] == 'question' and not any(record['evidence'])):
                continue
            meaning = record['meaning']
            if meaning is None:
                incomplete.append('unavailable_metadata'); continue
            # Query controls and proof records are not content constraints.
            constraints = {key: value for key, value in cue.bindings if not key.startswith('_')}
            if not self._scope_contains(cue.scope, meaning.scope) or not self._scope_contains(constraints, meaning.bindings):
                continue
            mask = occupied.to(meaning.role_mask.device)
            if bool((mask & ~meaning.role_mask).any()):
                continue
            wanted, held = cue.roles.to(meaning.roles), meaning.roles
            if self.when_start is not None:
                wanted, held = wanted.clone(), held.clone()
                wanted[:, self.when_start:self.when_start+4] = 0
                held[:, self.when_start:self.when_start+4] = 0
            similarities = F.cosine_similarity(wanted, held, dim=-1)
            match = float(similarities[mask].clamp(0, 1).mean()) if bool(mask.any()) else 1.
            found.append(dict(record, index=index, match=match))
        # Preserve the reader's address-order tie break after removing leaf
        # postings. Explicit cues affect which candidates fit the work budget,
        # not the ranking of equally matching retained descriptions.
        found.sort(key=lambda row: (-row['match'],
            (1, row['index'][1]) if isinstance(row['index'], tuple) else (0, row['index'])))
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
            if isinstance(index, int) and self.KINDS[int(self.record_kind[index])] in ('cover', 'estimate'):
                continue
            distance = abs(int(encoder().decode_index(stamp))-target)
            if distance <= window:
                found.append(dict(self.row(index), index=index, time_distance=distance, match=1. / (1+distance)))
        nearest = min((row['time_distance'] for row in found), default=None)
        return dict(value=tuple(row for row in found if row['time_distance'] == nearest),
                    records_scanned=scanned, incomplete=tuple(incomplete))

    def recall_identities(self, frames, *, work):
        """Re-enter chain identities in MTM; the returned head is the STM read."""
        owner_ref = getattr(self, '_concept_owner', None)
        if not callable(owner_ref):
            return ()
        owner = owner_ref()
        from LTM1 import OCCURRENCE, RELATION
        from ChainRead import envelope
        from RowTime import encoder
        missing = []
        for frame in frames:
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
                    owner.mtm.open(reference, kind=kind, when=tick, where=where,
                        native=kind if kind in owner.components.individual_ids else None,
                        value=self.slots[row], reading=kind,
                        pending=kind is None and bool(data.get('_open_references') or data.get('_forward_references')))
        return tuple(dict.fromkeys(missing))

    def recall_chunks(self, frames, *, work, stream=0):
        """Load returned heads into the same eight chunks, retaining their stamp."""
        missing = list(self.recall_identities(frames,work=work))
        owner_ref = getattr(self,'_concept_owner',None)
        owner = owner_ref() if callable(owner_ref) else None
        stm = getattr(owner,'stm',None)
        if stm is None:
            return tuple(missing)
        import STMChunks as chunks
        from ChainRead import head_idea
        state = stm.state
        if not 0 <= stream < state[0].shape[0]:
            raise ValueError('query stream exceeds the constructed STM batch')
        gate = torch.arange(state[0].shape[0],device=state[0].device) == stream
        for frame in frames:
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
