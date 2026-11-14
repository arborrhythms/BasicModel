"""Bounded reads of explicitly addressed or extension-recalled records."""
import math
import torch


class EntryRead:
    def entry_time(self, index):
        from RowTime import read_time
        return read_time(self, index)

    @torch.no_grad()
    def cued_rows(self, cue, *, primed=(), references=(), retrieved=(), max_candidates=None,
                  max_records=32, work=None, stream=None, role_mask=None, episodic_frames=()):
        """Read explicitly addressed and recalled rows, preserving their relevance."""
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
        recalled = {frame['index']: frame for frame in episodic_frames}
        candidates = list(dict.fromkeys(row for row in (*explicit, *recalled) if row is not None))
        incomplete = ['ltm_window'] if len(candidates) > max_candidates else []
        candidates = candidates[:max_candidates]
        scanned = len(candidates)
        if not scanned:
            return dict(value=(), records_scanned=0, incomplete=tuple(incomplete))
        found, read = [], 0
        for index in candidates:
            if len(found) >= max_records:
                break
            if index not in recalled and work is not None and not work.consume('record'):
                incomplete.append('work_budget')
                break
            record = recalled[index] if index in recalled else self.row(index)
            read += int(index not in recalled)
            if index not in recalled and isinstance(index, int):
                from MemoryActivity import accessed
                accessed(self, index)
            meaning = record['meaning']
            if meaning is None:
                continue
            # Content relevance belongs to the question's operator result.
            # An address permits a read; it supplies no resemblance score.
            found.append(dict(record,index=index,match=record.get('match',0.)))
        found.sort(key=lambda record:record['match'],reverse=True)
        return dict(value=tuple(found),records_scanned=read,
                    incomplete=tuple(dict.fromkeys(incomplete)))

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
            pending, seen = [frame['occurrence']], set()
            tick = int(encoder().decode_index(frame['time']))
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
                data = dict(self.raw_meaning_of(row).bindings)
                identity = self.referent_of(row)
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
        """Load returned heads into the same eight chunks, retaining their time."""
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
            occurrence = frame['occurrence']
            location = self.location(occurrence)
            relation = isinstance(location,tuple)
            point = head_idea(self,location[1]) if relation else frame['point']
            if point is None:
                point = frame['meaning'].roles[0]
            point = point.detach().to(state[0]).clone()
            if self.when_start is not None:
                point[self.when_start:self.when_start+4] = frame['time'].to(point)
            refs = state[chunks.REFS].new_zeros(state[0].shape[0],2,2)
            refs[:,0,0], refs[:,0,1] = (chunks.RELATION if relation else chunks.OCCURRENCE),occurrence[-1]
            token = owner.mtm.token_of(occurrence)
            entry = None if token is None else owner.mtm.entries[token]
            identity = -1 if entry is None else entry.symbol if entry.symbol is not None else entry.transient
            where = point.new_zeros(4) if self.when_start is None or self.when_start < 4 else point[self.when_start-4:self.when_start]
            state,index = chunks.deposit(state,point.expand(state[0].shape[0],-1),gate,
                refs=refs,order=frame['order'],identity=identity,where=where,when=frame['time'].to(point),
                truth=frame['evidence'])
            if int(index[stream]) < 0:
                missing.append('stm_all_chunks_on_stack')
                break
        stm.set_state(state)
        return tuple(dict.fromkeys(missing))
