"""Individual membership at a kept close, with access-bounded revision.

Operations decide mint versus bind. This owner decides only *which* accessible
individual explains a binding. Its inputs are native concept rows and accessible LTM occurrences.
The owner's MTM retains token membership between kept closes.
Trial plans are read-only. Occurrence points are immutable under revision.
"""
from dataclasses import replace
import hashlib

import torch
from Memory import NULL_CONCEPT
from torch import nn
from IdentityRows import meet_ratios


def constituents(clause):
    seen, result = set(), []
    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        if node.reference_request is not None and (node.order == 1 or 'compose_bind' in node.reference_request or 'reading_identity' in node.reference_request or node.reference_request.get('interpret_leaf') or (node.reference_request.get('sign_leaf') or node.reference_request.get('no_op_leaf'))):
            result.append(node)
        for child in (*node.children, *node.companions):
            visit(child)
    if clause is not None:
        visit(clause)
    return result


class ReferenceClose(nn.Module):
    def __init__(self, components):
        super().__init__()
        object.__setattr__(self, 'components', components)
        # G0: added defining parts enlarge the native code support. Request
        # coverage therefore ties underspecified re-mentions at initialization;
        # candidate coverage remains a separately trainable content feature.
        self.policy = nn.Parameter(torch.tensor([0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]))
        self.last_audit = ()

    @property
    def store(self):
        return self.components.owner._closed_clause_store()

    def address(self, reference):
        if isinstance(reference, tuple):
            if reference[:1] in (('candidate',), ('name',)):
                pool = self.components.owner.mtm
                entry = pool.entries.get(pool.token_of(reference))
                return None if entry is None else entry.symbol
            if len(reference) == 2 and reference[0] == 'sym':
                return int(reference[1])
            if len(reference) != 3 or reference[:2] != ('ltm', self.store.address_domain):
                return None
            return int(reference[2])
        return reference

    def origin_reference(self, tag, address):
        """Decode a kept journal origin without aliasing pool and LTM keys."""
        if tag == 1:
            return int(address)
        if tag == 2:
            return ('ltm', self.store.address_domain, int(address))
        if tag == 5:
            return ('candidate', int(address))
        if tag == 6:
            return ('name', int(address))
        return None

    def coordinates(self, point):
        from IdentityRows import coordinates
        return coordinates(self.components.owner, point)

    def predicate_coordinates(self, point):
        from IdentityRows import coordinates
        return coordinates(self.components.owner, point)

    @staticmethod
    def _predicate_key(point):
        return hashlib.sha256(point.detach().float().cpu().contiguous().numpy().tobytes()).digest()

    def _participants(self):
        """Role observations owned by the retained identities' event histories."""
        from IdentityRows import point
        owner = self.components.owner
        rows = {}
        for token, entry in owner.mtm.entries.items():
            for event in entry.events:
                value = point(owner, self._event_symbol(event.verb))
                if value is not None:
                    rows.setdefault(self._predicate_key(self.predicate_coordinates(value)), []).append(
                        (event.occurrence, event.role, token))
        return {key: tuple(dict.fromkeys(value)) for key, value in rows.items()}

    def _event_symbol(self, reference):
        """A captured lexical verb reads its declared object, never a word alias."""
        from IdentityRows import point
        if point(self.components.owner, reference) is not None:
            return reference
        objects = self.components.owner.definitions.objects(reference)
        return objects[0] if len(objects) == 1 else reference

    def _derivation_symbol(self, reference):
        """Read a token's retained kind/fold without making it an individual."""
        if reference is None:
            return None
        if isinstance(reference, tuple) and reference[:1] in (('candidate',), ('name',)):
            return self.address(reference)
        if isinstance(reference, int) and (self.store is None or self.store.index_of_row(reference) is None):
            return None if self.components.owner.interpret.no_object(reference) else reference
        if isinstance(reference, tuple) and reference[:1] == ('sym',):
            return None if self.components.owner.interpret.no_object(reference[1]) else reference[1]
        location = self.store.location(reference) if self.store is not None else None
        if isinstance(location, int):
            mtm = self.components.owner.mtm
            token = mtm.token_of(self.store.occurrence_of(location))
            if token is not None and mtm.entries[token].symbol is not None and not mtm.entries[token].pending:
                return mtm.entries[token].symbol
            value = self.store.raw_meaning_of(location).role_refs[0]
            if value is not None and value[0] == 'sym':
                return None if self.components.owner.interpret.no_object(value[1]) else self.components.owner.__dict__.get('_recurring_readings', {}).get(value[1], value[1])
            mtm = self.components.owner.mtm
            token = mtm.token_of(self.store.occurrence_of(location))
            if token is None:
                identity = dict(self.store.raw_meaning_of(location).bindings).get('_identity')
                token = mtm.token_of(identity)
            if token is not None:
                kind = mtm.entries[token].kind
                return None if kind == NULL_CONCEPT else kind
        return None

    def token_value(self, reference):
        """Read the identity's current value, never a union of past states."""
        return self.components.owner.mtm.value_of(reference)

    def is_forward_placeholder(self, reference, name):
        """A forward name and its own objects do not supply its missing referent.

        A selected arrival can resolve the kind without replacing the token's
        original naming symbol. That resolved kind takes precedence here.
        Anonymous recursive identities have no forward name and remain legal.
        """
        if name is None:
            return False
        owner = self.components.owner
        pool = owner.mtm
        entry = pool.entries.get(pool.token_of(reference))
        if entry is None:
            referents = (self._derivation_symbol(reference),)
        elif entry.kind not in (None, NULL_CONCEPT):
            referents = (entry.kind,)
        else:
            referents = tuple(value for value in (entry.native, entry.symbol)
                              if value is not None)
        own = {name, *owner.definitions.objects(name),
               *owner.interpret.remembered_objects(name)}
        return bool(referents) and all(value in own for value in referents)

    def history_value(self, reference):
        """Diagnostic union of historical mentions; no live reader uses this."""
        mtm = self.components.owner.mtm
        token = mtm.token_of(reference)
        if token is None:
            return None
        entry = mtm.entries[token]
        value = self.store.point_of_row(token[-1])
        if value is None:
            return None
        value = value.detach().clone()
        location = self.store.location(token)
        if entry.kind not in (None, NULL_CONCEPT) and self.store.raw_meaning_of(location).role_refs[0] is None:
            kind = self.store._concept_point(entry.kind)
            if kind is not None:
                size = min(value.numel(), kind.numel())
                value[:size] = torch.maximum(value[:size], kind[:size].to(value))
        for occurrence in entry.occurrences[1:]:
            point = self.store.point_of_row(occurrence[-1])
            if point is not None:
                size = min(value.numel(), point.numel())
                value[:size] = torch.maximum(value[:size], point[:size].to(value))
        return value

    def prediction(self, identity, predicate, role):
        target = self.components.owner.mtm.token_of(identity)
        if target is None:
            return predicate.sum()*0
        rows = [(edge, target) for edge, r, target in
                self._participants().get(self._predicate_key(predicate), ()) if r == role]
        total = len(rows)
        count = sum(candidate == target for _, candidate in rows)
        return predicate.new_tensor(count/total) if total else predicate.sum()*0

    def expected_roles(self, predicate):
        rows = self._participants().get(self._predicate_key(self.predicate_coordinates(predicate)), ())
        return tuple(role for role in (0, 2) if sum(r == role
            for _, r, _ in rows) >= self.components.verbs.recurrence)

    def expected_content(self, predicate, role):
        """An elided role reads what its retained participant witnesses describe."""
        value = self.predicate_coordinates(predicate)
        if role not in self.expected_roles(value):
            return torch.zeros_like(value)
        pool = self.components.owner.mtm
        records = self._participants().get(self._predicate_key(value), ())
        tokens = tuple(dict.fromkeys(token for _, observed_role, token in records if observed_role == role))
        descriptions = []
        for token in tokens:
            entry = pool.entries.get(token)
            if entry is not None and entry.occurrences:
                reference = entry.occurrences[-1]
                descriptions.append(self.coordinates(pool.read_member(reference, pool.reading_of(reference))))
        return torch.stack(descriptions).amax(0) if descriptions else torch.zeros_like(value)

    def prime(self, signature, stream):
        if stream is None:
            return signature.sum()*0
        model = getattr(self.components.owner, '_model', None)
        get_reducer = getattr(model, '_stm_reducer', None)
        reducer = None if get_reducer is None else get_reducer()
        value = getattr(reducer, '_expectation_prime', None)
        if value is None or stream is None:
            return signature.sum()*0
        value = value[int(stream)].detach().to(signature).reshape(-1)
        width = min(value.numel(), signature.numel())
        return (signature[:width]*value[:width]).sum()/signature[:width].sum().clamp_min(1e-12)

    def accessible(self, chunks, *, stream=None, include_pool=True):
        """Read identities from shared chunks, optionally adding the live MTM pool."""
        from Memory import STMChunks
        if len(chunks) != len(STMChunks.FIELDS) or not all(torch.is_tensor(value) for value in chunks):
            raise ValueError('reference access requires the shared chunk tuple')
        store, pool = self.store, self.components.owner.mtm
        from PrependedSymbols import chunk_token
        from Spaces import _concept_alloc_of
        alloc = _concept_alloc_of(self.components.owner)
        records, tokens = {}, []
        for b in range(chunks[0].shape[0]):
            for position in range(chunks[0].shape[1]):
                identity = int(chunks[STMChunks.IDENTITY][b,position])
                token = chunk_token(pool,store,chunks,b,position)
                if token is not None:
                    tokens.append(token)
                elif identity >= 0 and identity in alloc.individuals and identity not in alloc.retired:
                    records[identity] = ('sym',identity), dict(role=0,
                        position=(position,0),identity=identity,value=chunks[0][b,position])
                for tag, reference in chunks[STMChunks.REFS][b,position].tolist():
                    if tag == STMChunks.OCCURRENCE:
                        row = store.index_of_row(reference)
                        if row is not None:
                            tokens.extend(key for key, entry in pool.entries.items()
                                if key == store.occurrence_of(row) or any(event.role == 0 and
                                    event.occurrence == store.occurrence_of(row) for timeline in entry.timelines
                                    for event in timeline.events))
        if include_pool:
            for key in pool.entries:
                row = store.location(key)
                if (stream is not None and isinstance(row,int) and bool(store.stream[row].any())
                        and (stream >= store.stream.shape[1] or not bool(store.stream[row,stream]))):
                    continue
                tokens.append(key)
        # A retained, unpooled idea can start a collection on its next link.
        from MemoryLink import chunk_candidates
        for order in set(int(value) for value in chunks[STMChunks.ORDERS].reshape(-1).tolist() if int(value) > 0):
            for address, record in chunk_candidates(self.components.owner, 'reference', order, stream).items():
                records.setdefault(address[-1], (store.location(address), dict(record, identity=address)))
        names = {}
        for address, name in pool.names.items():
            names.setdefault(name.token, []).append((address, name.reading))
        for position, key in enumerate(dict.fromkeys(tokens)):
            entry = pool.entries[key]
            if (entry.order is not None or entry.refinement.get('member_kind', 'reference') != 'reference'
                    or not pool.in_window(key, stream)):
                continue
            records.pop(entry.native,None)
            # Chunks determine accessibility; the held token supplies its
            # bounded mention history. Reading that history never admits an
            # unpooled LTM occurrence or replaces the current working value.
            from RowTime import encoder
            uses = [(occurrence,store.location(occurrence)) for occurrence in entry.occurrences]
            uses = [(occurrence,row) for occurrence,row in uses if isinstance(row,int)]
            times = [int(encoder().decode_index(store.entry_when(row))) for _,row in uses]
            value = pool.value_of(key, references=names.get(key, ()))
            for use,(occurrence,location) in enumerate(uses):
                following = times[use+1] if use+1 < len(times) else float('inf')
                events = [event for event in entry.events if times[use] <= event.when < following]
                event = events[-1] if events else None
                records[occurrence[-1]] = location, dict(role=0 if event is None else event.role,
                    position=(position,use), identity=entry.native or key,token=key,
                    value=value, mention_value=store.point_of_row(occurrence[-1]),
                    predicate=None if event is None else event.verb)
        return records

    def context_records(self, frames):
        """Read the predictor/thought's own address context, independently of STM."""
        from Spaces import _concept_alloc_of
        store, records = self.store, {}
        if store is None:
            return records
        alloc, mtm = _concept_alloc_of(self.components.owner), self.components.owner.mtm
        for position, frame in enumerate(frames):
            for identity in frame.symbol_ids:
                if identity in alloc.individuals and identity not in alloc.retired:
                    records[identity] = ('sym', identity), dict(role=0,
                        position=(position, 0), identity=identity)
            activations = list(frame.activations)
            location = store.location(frame.row_id)
            if (isinstance(location, int) and (mtm.token_of(store.occurrence_of(location)) is not None
                                                or store.order_of_row(location) == 1)
                    and not any(item[0] == store.occurrence_of(location) for item in activations)):
                activations.append((store.occurrence_of(location), None, 0))
            for token, entry in (() if frame.activations else mtm.entries.items()):
                for event in entry.events:
                    if event.occurrence[-1] == frame.row_id:
                        activations.append((token, entry.native or entry.symbol, event.role))
            for occurrence, identity, role in activations:
                index = store.location(occurrence)
                token = mtm.token_of(occurrence)
                if index is None:
                    if identity is None or identity not in alloc.individuals:
                        continue
                    index = ('sym', identity)
                predicate = None
                if token is not None:
                    events = [event for event in mtm.entries[token].events
                              if event.role == role and event.occurrence[-1] == frame.row_id]
                    predicate = events[-1].verb if events else None
                records.pop(identity, None)
                records[occurrence[-1]] = index, dict(role=role,
                    position=(position, len(records)), identity=(mtm.entries[token].native if token is not None else None) or token or identity or occurrence,
                    predicate=predicate)
        return dict(tuple(records.items())[-self.components.source_capacity:])

    def occurrences(self, row):
        frame = __import__('ReferenceContext').SituationFrame.from_row(self.store.row(row))
        return tuple(('sym', context['identity']) if isinstance(context['identity'], int)
                     else context['identity'] or self.store.occurrence_of(location)
                     for location, context in self.context_records((frame,)).values())

    def logits(self, features, cue=None):
        """Score the retained alternatives for the ordinary departure choice."""
        return features.detach() @ self.policy.clone()

    def cue(self, value):
        # A form identifies the symbol whose learned conceptual region is
        # read. Its perceptual coordinates never enter a content comparison.
        from IdentityRows import carrier
        return carrier(self.components.owner, value)

    def exclusion_pairs(self):
        """Coordinate exclusions witnessed by eight uses on each side."""
        from itertools import combinations
        observed = []
        for row in range(len(self.store)):
            if self.store.order_of_row(row) != 1:
                # A kept mention is still an observation of an individual when
                # its stored referent is the concrete description it came from.
                meaning = self.store.raw_meaning_of(row)
                if meaning is None or not dict(meaning.bindings).get('_reference_sites'):
                    continue
            value = self.coordinates(self.store.slots[row])
            observed.append((set((value >= .3).nonzero().flatten().tolist()),
                             int(self.store.access_count[row])))
        counts = {}
        for members, count in observed:
            for coordinate in members:
                counts[coordinate] = counts.get(coordinate, 0) + count
        return tuple((x, y) for x, y in combinations(sorted(
            coordinate for coordinate, count in counts.items() if count >= 8), 2)
            if not any(x in members and y in members for members, _ in observed))

    def exclusion_query(self):
        """Read observed exclusion once, for grouping or a bind's candidates."""
        pairs = self.exclusion_pairs()
        def support(value):
            return set((value >= .3).nonzero().flatten().tolist())
        def contradicts(a, b):
            a, b = support(a), support(b)
            if not a or not b or a <= b or b <= a:
                return False
            return any((x in a-b and y in b-a) or (y in a-b and x in b-a)
                       for x, y in pairs)
        return contradicts

    def individuals(self, records):
        """Compute the current identities from their codes and accessible uses.

        The result owns working vectors only. Neither a bank row, an occurrence
        nor the activation/address index is changed by this interpretation.
        """
        from IdentityRows import point
        from MemoryLink import Occurrence, member_order
        owner, pool = self.components.owner, self.components.owner.mtm
        def contracts(identity, members):
            result = set()
            for member in members:
                entry = pool.entries.get(pool.token_of(member['token']))
                metadata = {} if entry is None else entry.refinement
                order = metadata.get('member_order')
                if order is None:
                    order = member_order(owner, Occurrence(identity,
                        concept=identity if isinstance(identity, int) else None))
                result.add((metadata.get('member_kind', 'reference'), order))
            return result
        def baseline(identity):
            if isinstance(identity, int):
                return point(self.components.owner, identity)
            if identity in self.components.owner.mtm.entries:
                index = self.store.location(identity)
                return self.store.point_of_row(identity[-1]) if isinstance(index, int) else self.token_value(identity)
            return None
        groups = {}
        for address, (location, original) in records.items():
            context = dict(original)
            identity = context.get('identity')
            if identity is None:
                identity = self.store.occurrence_of(location)
            native = baseline(identity)
            observed = context.get('mention_value',context.get('value'))
            if observed is None:
                observed = self.store.point_of_row(address) if isinstance(location, int) else native
            if observed is None:
                continue
            context['content'] = self.coordinates(observed)
            context['location'] = location
            context['address'] = address
            context['native'] = native
            # A native symbol may be present in two independent streams. Its
            # working state is reached through this frame's occurrence only.
            context['token'] = context.get('token') or (self.components.owner.mtm.token_of(self.store.occurrence_of(location))
                                if isinstance(location, int) else None)
            groups.setdefault(identity, []).append(context)
        def predicate(member):
            reference = member.get('predicate')
            value = None if reference is None else self.store._concept_point(reference)
            return None if value is None else self._predicate_key(self.predicate_coordinates(value))
        # Exclusion is a query over retained uses. No learned cooccurrence
        # table or word class supplies it, and no result is written to LTM.
        contradicts = self.exclusion_query()
        for identity, members in tuple(groups.items()):
            anchors = []
            for member in reversed(members):
                if not anchors or any(contradicts(member['content'], anchor['content'])
                                      for anchor in anchors):
                    anchors.append(member)
            if len(anchors) < 2:
                continue
            # A split names another already present occurrence. It does not
            # mint a symbol or edit an older binding to create that identity.
            identities = [identity, *(self.store.occurrence_of(anchor['location'])
                                       for anchor in anchors[1:])]
            partitions = {key: [] for key in identities}
            for member in members:
                scores = [meet_ratios(member['content'], anchor['content'])[1] +
                          int(predicate(member) == predicate(anchor)) for anchor in anchors]
                selected = identities[int(torch.stack(scores).argmax())]
                partitions[selected].append(member)
            groups.pop(identity)
            groups.update({key: members for key, members in partitions.items() if members})
        # Reinterpretation is local to these accessible uses. Shared later
        # distinguishing content can reconcile two initial names. Mere common
        # kind content at mint time supplies no such evidence.
        keys = list(groups)
        for offset, left in enumerate(keys):
            if left not in groups:
                continue
            for right in keys[offset+1:]:
                if right not in groups:
                    continue
                a, b = groups[left], groups[right]
                # Shared descriptive content cannot reconcile collections of
                # different kinds or orders. A trial must read the same
                # collection contract that its selected identity will commit.
                if len(contracts(left, a) | contracts(right, b)) != 1:
                    continue
                native_a, native_b = baseline(left), baseline(right)
                if native_a is None or native_b is None:
                    continue
                width = min(native_a.numel(), native_b.numel())
                initial = torch.maximum(self.coordinates(native_a)[:width],
                                        self.coordinates(native_b)[:width])
                def new_support(member):
                    value = member['content'][:width]
                    return bool(((value >= .3) & (initial[:value.numel()] < .3)).any())
                newer_a, newer_b = [x for x in a if new_support(x)], [x for x in b if new_support(x)]
                common = any(any(bool(coverage == 1) for coverage in
                    meet_ratios(x['content'], y['content']))
                    and predicate(x) == predicate(y) for x in newer_a for y in newer_b)
                if common and not any(contradicts(x['content'], y['content']) for x in a for y in b):
                    groups[left].extend(groups.pop(right))
        result = {}
        for identity, members in groups.items():
            native = baseline(identity)
            current = [self.token_value(x['token']) for x in members if x['token'] is not None]
            if current:
                # Revision may group several tokens on read. Join their current
                # states; a historical occurrence must not resurrect old state.
                value = torch.stack(current).amax(0)
                latest = max(members, key=lambda x: x['position'])
                result[identity] = dict(content=self.coordinates(value), value=value,
                    role=latest['role'], position=latest['position'],
                    token=latest['token'],
                    occurrences=tuple(x['address'] for x in members))
                continue
            width = native.numel() if native is not None else max(x['content'].numel() for x in members)
            value = native.detach().clone() if native is not None else members[-1]['content'].new_zeros(width)
            for member in members:
                observed = member['content'].to(value)
                size = min(width, observed.numel())
                value[:size] = torch.maximum(value[:size], observed[:size])
            latest = max(members, key=lambda x: x['position'])
            result[identity] = dict(content=self.coordinates(value), value=value,
                role=latest['role'], position=latest['position'],
                occurrences=tuple(x['address'] for x in members))
        # The promoted concept is one higher-order member, distinct from
        # its retained lower-order path. Refinement can choose either at
        # the appropriate order without merging those two collections.
        pool = self.components.owner.mtm
        for candidate in tuple(result.values()):
            entry = pool.entries.get(pool.token_of(candidate.get('token')))
            if entry is not None and entry.symbol is not None:
                value = point(self.components.owner, entry.symbol)
                result.setdefault(entry.symbol, dict(value=value, content=self.coordinates(value),
                    role=candidate['role'], position=candidate['position'], occurrences=()))
        return result

    def introduction_order(self, node, read_order, *, keep_reference):
        """Read the same named concept or occurrence identity as the close."""
        request = node.reference_request or {}
        if request.get('sign_leaf') or request.get('no_op_leaf'):
            # PartClosing writes this surface-only leaf as a concrete
            # occurrence; its spelling's denotation is not being read.
            return 0
        if read_order is not None:
            return read_order
        from PartClosing import PartClosing
        from ClauseRow import Composition
        from dataclasses import fields
        from Memory import SYMBOL
        from MemoryLink import Occurrence, member_order
        from ChainRead import envelope
        owner = self.components.owner
        closing = PartClosing(owner, self.store)
        term = closing.clause_term(node)
        # Match prepare's new introduction: no selected alias, and only an
        # own-object request preserves its existing root reference.
        term = Composition(**dict({field.name:getattr(term, field.name) for field in fields(Composition)},
            reference=term.reference if keep_reference else -1, order=node.order, binding=None))
        record = closing._plan_term(term)
        reference = closing.occurrence_reference(record)
        marks = dict(node.meaning.bindings)
        if reference is not None and not (reference[0] == SYMBOL and owner.interpret.no_object(reference[1])):
            marks.setdefault('_identity', envelope(self.store, reference))
        identity = marks.get('_identity')
        if identity and identity[0] in ('ltm', 'sym', 'name'):
            return member_order(owner, Occurrence(identity,
                concept=identity[1] if identity[0] in ('sym', 'name') else None))
        names = tuple(marks.get('_forward_references', ()))
        if len(names) == 1 and names[0][0] == 0:
            return member_order(owner, Occurrence((), concept=names[0][1]))
        # LTM0.order_of_row reads a newly written idea without a native
        # identity or lexical witness as a concrete order-zero occurrence.
        return 0

    def score(self, content, predicate, cue, role, individuals, *, stream=None, asserted=(), where=None, when=None, coordinate_read=None, member_order=None):
        """Meet content and form, native expectation, then recency and roles."""
        from IdentityRows import form_region
        coordinate_read = self.coordinates if coordinate_read is None else coordinate_read
        content = coordinate_read(content)
        region = form_region(self.components.owner,cue)
        contradicts = self.exclusion_query()
        constraints = (content, *(self.coordinates(value) for value in asserted))
        choices, features = [], []
        for identity,candidate in individuals.items():
            if member_order is not None:
                from MemoryLink import candidate_eligible
                if not candidate_eligible(self.components.owner, identity, candidate,
                        kind='reference', order=member_order, stream=stream, allow_promotion=False):
                    continue
            value = coordinate_read(candidate['content']).to(content)
            if any(contradicts(own, value) for own in constraints):
                continue
            candidate_coverage,request_coverage = meet_ratios(content,value)
            if not bool((candidate_coverage == 1) | (request_coverage == 1)):
                continue
            form = (content.sum()*0 if region is None else meet_ratios(region.to(value),value)[0])
            recency = sum(other['position'] > candidate['position'] for other in individuals.values())
            features.append(torch.stack((candidate_coverage,request_coverage,
                self.prediction(candidate.get('token') or candidate.get('kind', identity),predicate,role),self.prime(value,stream),form,
                content.new_tensor(-recency/max(1,len(individuals))),
                content.new_tensor(candidate['role']/2),content.new_tensor(role/2),
                *self.continuity(candidate, content, stream=stream, where=where, coordinate_read=coordinate_read))))
            choices.append(identity)
        phi = torch.stack(features).to(self.policy) if features else self.policy.new_empty((0,12))
        return tuple(choices),phi,self.logits(phi)

    def continuity(self, candidate, value, *, stream, where, coordinate_read):
        pool = self.components.owner.mtm
        token = pool.token_of(candidate.get('token'))
        entry = pool.entries.get(token)
        zeros = value.new_zeros(())
        if entry is None:
            return zeros, zeros, zeros, value.new_tensor(float(candidate.get('in_stm', False)))
        mentions = entry.refinement.get('mentions', {})
        latest = max((clock for member_stream, clock in mentions.values() if member_stream == stream),
                     default=pool.closes.get(stream, 0)+1)
        gap = value.new_tensor((pool.closes.get(stream, 0)+1-latest)/pool.horizon)
        overlap = zeros if where is None else torch.minimum(where.to(value), entry.where.to(value)).sum()
        tail = entry.occurrences[-1] if entry.occurrences else token
        concept = next((concept for address, concept in reversed(entry.refinement.get('members', ())) if address == tail), pool.reading_of(tail))
        last = coordinate_read(pool.read_member(tail, concept)).to(value)
        difference = (value-last).square().mean()
        stm = getattr(self.components.owner, 'stm', None)
        present = False if stm is None else bool((stm.state[7] == entry.transient).any()) if entry.transient is not None else False
        return overlap, gap, difference, value.new_tensor(float(present))

    @staticmethod
    def choice_cost(features, logits, selected):
        credit = 6*features[:,1]+features[:,2]+features[:,3]
        target = credit.detach().softmax(0)
        return ((1-features[selected,1]).clamp_min(0)-features[selected,2]
                -(target*logits.log_softmax(0)).sum())

    def select(self, choices, logits, *, forward_identity=None, individuals=None):
        """The ordinary resolver choice; fixtures may force this operation seam."""
        if not choices or not bool((logits.detach() > 0).any()):
            return None
        return int(logits.detach().argmax())

    def choose(self, plan, choices, features, logits, *, forward_identity=None, individuals=None):
        """The zero-logit dummy wins ties; a departure takes the next option.

        Candidate indices at the fixture seam retain their existing meaning.
        The recorded menu includes the dummy at index zero, including when
        there are no accessible candidates. Preview owns this live graph.
        """
        menu = torch.cat((logits.new_zeros(1), logits))
        greedy = self.select(choices, logits, forward_identity=forward_identity,
                             individuals=individuals)
        greedy = 0 if greedy is None else greedy + 1
        ranked = menu.detach().argsort(descending=True, stable=True).tolist()
        alternate = next((index for index in ranked if index != greedy), greedy)
        departed = bool(choices) and len(plan['decisions']) == plan['departure']
        selected = alternate if departed else greedy
        probabilities = menu.softmax(0)
        decision = dict(candidates=(None, *choices), features=features,
            logits=menu, probabilities=probabilities, greedy=greedy,
            selected=selected, next=alternate, departed=departed,
            forward_identity=forward_identity)
        plan['decisions'].append(decision)
        return (None if selected == 0 else selected-1), decision

    @staticmethod
    def has_word(node):
        """A noun occurrence has a lexical carrier; an elided role has none."""
        request = node.reference_request or {}
        if request.get('determined') or request.get('forward_identity') is not None or node.subject_word_id >= 0:
            return True
        # Lowering itself introduces the noun token. An elided role can
        # retain a cue's composition without having a word of its own.
        return node.composition is not None and node.composition.formal_name == 'lower'

    def bind_site(self, meaning, role, frames, *, stream=None, decisions=None):
        """The thought and closing faces read the same native candidates."""
        from ThoughtReferences import bindings, fill, open_slots
        if ('referent', role) not in open_slots(meaning):
            raise ValueError('bind requires one open referent site')
        site = dict(dict(bindings(meaning).get('_reference_sites', ())).get(role, ()))
        content = self.coordinates(meaning.roles[role] if site.get('content') is None else
                                   meaning.roles.new_tensor(site['content']))
        predicate = site.get('predicate')
        predicate = self.predicate_coordinates(meaning.roles[1] if predicate is None else
                                    meaning.roles.new_tensor(predicate))
        if not site.get('leaves') and not bool(content.any()):
            content = self.expected_content(predicate, site.get('role', role))
        cue = self.cue(None if site.get('cue') is None else meaning.roles.new_tensor(site['cue']))
        records = self.context_records(frames)
        query_occurrence = bindings(meaning).get('_query_occurrence')
        binding_occurrence = site.get('occurrence', query_occurrence)
        # A shared identity can participate in a self-referential relation.
        # Bind reads its finite content; bounded readers own graph traversal.
        individuals = self.individuals(records)
        choices, features, logits = self.score(content, predicate, cue,
            site.get('role', role), individuals, stream=stream)
        base = dict(binding_role=role, reverse_leaves=tuple(site.get('leaves', ())),
                    binding_candidates=choices, witnesses=(),
                    binding_occurrence=binding_occurrence)
        plan = dict(meaning=meaning, role=role, records=records, individuals=individuals,
            candidates=choices, features=features, cue=cue, base=base,
            forward_identity=dict(bindings(meaning).get('_forward_references', ())).get(role))
        result = self.finish_thought_bind(plan, logits=logits)
        if decisions is not None:
            decisions.append(plan)
        return result

    def finish_thought_bind(self, plan, *, departure=False, logits=None):
        """Select from a held, already paid menu; only episode advantage trains it."""
        from ThoughtReferences import fill, open_slots, with_slots
        meaning, role = plan['meaning'], plan['role']
        choices, features, cue = plan['candidates'], plan['features'], plan['cue']
        individuals, records, base = plan['individuals'], plan['records'], plan['base']
        if logits is None:
            logits = self.logits(features, cue)
        index, decision = self.choose(dict(decisions=[], departure=0 if departure else -1),
            choices, features, logits, forward_identity=plan['forward_identity'], individuals=individuals)
        plan['decision'] = decision
        if index is None:
            return dict(base, meaning=meaning, value=meaning, support_true=0., support_false=0.,
                        reference_cost=1., incomplete=('no_accessible_individual',))
        identity = choices[index]
        probability = float(decision['probabilities'][index+1].detach())
        pool = self.components.owner.mtm
        entry = pool.entries.get(pool.token_of(identity))
        if ((entry is not None and (entry.pending or entry.kind == NULL_CONCEPT))
                or self.is_forward_placeholder(identity, plan['forward_identity'])):
            # Selecting an unresolved identity is still a legal choice. Its
            # name does not supply the missing referent merely by being read.
            return dict(base, meaning=meaning, value=meaning, support_true=0., support_false=0.,
                        reference_cost=float(self.choice_cost(features, logits, index).detach()),
                        incomplete=('pending_antecedent',))
        point = individuals[identity]['value'].detach().to(meaning.roles)
        witnesses = tuple(('sym', context['identity']) if isinstance(context.get('identity'), int)
                          else self.store.occurrence_of(location)
                          for location, context in records.values()
                          if (context.get('identity') == identity or
                              (isinstance(identity, tuple) and location ==
                               self.store._index_occurrences.get(identity)))
                          and (not isinstance(location, int) or
                               identity not in self.components.owner.mtm.entries or
                               self.store.occurrence_of(location) in
                               self.components.owner.mtm.names))
        evidence = dict(base, reference=identity if isinstance(identity,tuple) else ('sym', identity), binding_value=point,
            witnesses=witnesses,
            support_true=probability, support_false=1-probability,
            reference_cost=float(self.choice_cost(features, logits, index).detach()),
            incomplete=())
        result = fill(meaning, evidence, witnesses=witnesses,
                      operation='bind', slots=(('referent', role),))
        result = with_slots(result, open_slots(result), pair=(probability, 1-probability))
        return dict(evidence, value=result, meaning=result)

    def answer_meaning(self, plan, meaning):
        """The supplied-referent judge observes this trial's selected binding.

        This view writes nothing. Arrival menus deliberately stay outside
        answer supervision. An unlinked object supplies no answer identity.
        """
        from ThoughtReferences import bindings, fill, open_slots
        roles = {role for kind, role in (*open_slots(meaning),
            *bindings(meaning).get('_bound_roles', ())) if kind == 'referent'}
        result = meaning
        for item in plan['items']:
            identity = item['selected']
            answered = roles.intersection(item['root_roles'])
            if not answered or identity is None or item['mode'] == 'mint':
                continue
            if isinstance(identity, tuple) and identity[:1] == ('new',):
                continue
            if isinstance(identity, tuple) and self._derivation_symbol(identity) is None:
                continue
            # A compose bind carries the actual selected working value. Its
            # token need not belong to the later individual-choice menu:
            # familiar candidates are deliberately excluded from that menu.
            candidate = item.get('record')
            if candidate is None:
                candidate = plan['individuals'][identity]
            reference = ('sym', identity) if isinstance(identity, int) else identity
            if isinstance(identity, tuple) and identity[:1] == ('candidate',):
                reference = ('sym', self._derivation_symbol(identity))
            result = fill(result, dict(reference=reference,
                binding_value=candidate['value'], support_true=1., support_false=0.),
                operation='bind', slots=tuple(('referent', role) for role in sorted(answered)))
        return result

    def resolved(self, row, role, reference, errors):
        """Release the old diagnostic menu after the actual referent arrives."""
        mentions = self.__dict__.get('_mentions', {})
        occurrence = self.store.occurrence_of(row)
        remaining = tuple(record for record in mentions.get(occurrence, ()) if record[0] != role)
        if remaining:
            mentions[occurrence] = remaining
        else:
            mentions.pop(occurrence, None)

    def sites(self, clause, *, pending_names=()):
        """Keep explicit open formations as one site, owned by their word.

        The journal need not invent a second NP request. Its retained forward
        name supplies the anonymous constituent only at the closing boundary.
        """
        from ClauseRow import Clause
        from Meaning import ConceptualMeaning
        from ThoughtReferences import bindings, open_slots
        from ReferenceParticipants import clause_participants
        _, participants, terms = clause_participants(clause,
            self.store.__dict__.get('_part_read_rules', ()))
        def source_leaf(node, name, held):
            leaves = set(held.get('leaves', ()))
            pending = [node.composition]
            while pending:
                term = pending.pop()
                if term is None:
                    continue
                if (not term.children and id(term) in terms and term.order == 1
                        and term.word_id == name
                        and (not leaves or leaves.intersection(term.leaves))):
                    return term
                pending.extend(reversed(term.children))
            return None
        result, seen, nodes = [], set(), []
        def visit(node):
            if node is None or id(node) in seen:
                return
            seen.add(id(node))
            nodes.append(node)
            if node.reference_request is not None and (node.order == 1 or 'compose_bind' in node.reference_request or 'reading_identity' in node.reference_request or node.reference_request.get('interpret_leaf') or (node.reference_request.get('sign_leaf') or node.reference_request.get('no_op_leaf'))):
                if (node.reference_request.get('addressed_introduction')
                        or node.reference_request.get('sign_leaf') or node.reference_request.get('no_op_leaf')
                        or not self.has_word(node) or id(node) in participants):
                    result.append(dict(node=node, parent=None, site_role=None))
            else:
                data = bindings(node.meaning)
                requests = dict(data.get('_reference_sites', ()))
                for role, name in data.get('_forward_references', ()):
                    if ('referent', role) not in open_slots(node.meaning):
                        continue
                    ref = node.refs[role]
                    if isinstance(ref, tuple) and ref[:1] == ('clause',):
                        child = node.children[ref[1]]
                        if (child.reference_request or {}).get('forward_identity') == name:
                            continue
                    held = dict(requests.get(role, ()))
                    term = source_leaf(node, name, held)
                    if role not in (0, 2) or term is None:
                        continue
                    def value(key, fallback):
                        return fallback if held.get(key) is None else term.point.new_tensor(held[key])
                    request = dict(mode='bind', forward_identity=name, role=role,
                        leaves=term.leaves, content=term.point,
                        cue=value('cue', term.point),
                        predicate=value('predicate', node.meaning.roles[1]))
                    site = Clause(ConceptualMeaning.from_description(term.point), point=term.point,
                        order=term.order, evidence=node.evidence, subject_word_id=name,
                        reference_request=request, composition=term)
                    result.append(dict(node=site, parent=node, site_role=role))
            for child in (*node.children, *node.companions):
                visit(child)
        visit(clause)
        if (clause is not None and clause.order == 1 and clause.reference_request is None
                and clause.relation is None and clause.point is not None
                and id(clause) in participants and clause.composition is not None
                and clause.composition.children):
            term = clause.composition
            head = term
            while head.children and head.head_role:
                head = head.children[head.head_role-1]
            request = dict(mode='bind', determined=True, role=0, leaves=term.leaves,
                content=clause.point, cue=head.point, predicate=torch.zeros_like(clause.point))
            result.append(dict(node=replace(clause, reference_request=request),
                source_node=clause, parent=None, site_role=None))
        # A lexical concept cache does not replace a familiar word's token
        # choice. Re-mentions of accessible pending names still call bind at
        # the closing boundary, before their derivation is admitted.
        covered = {site['node'].reference_request.get('forward_identity') for site in result}
        names, visited_terms = set(pending_names) - covered, set()
        def lexical(term):
            if term is None or id(term) in visited_terms:
                return
            visited_terms.add(id(term))
            if (not term.children and id(term) in terms and term.order == 1
                    and term.word_id in names):
                request = dict(mode='bind', forward_identity=term.word_id,
                    role=0, leaves=term.leaves, cue=term.point, content=term.point,
                    predicate=torch.zeros_like(term.point))
                node = Clause(ConceptualMeaning.from_description(term.point), point=term.point,
                    order=term.order,
                    evidence=clause.evidence, subject_word_id=term.word_id,
                    composition=term, reference_request=request)
                result.append(dict(node=node, parent=None, site_role=None, lexical=True))
            for child in term.children:
                lexical(child)
        for node in nodes:
            lexical(node.composition)
        # Request roles belong to their immediate clause. Only direct root
        # attachments can supply the root answer; nested role numbers cannot.
        parents = {}
        for parent in nodes:
            for role, reference in enumerate(parent.refs):
                if isinstance(reference, tuple) and reference[:1] == ('clause',):
                    parents.setdefault(id(parent.children[reference[1]]), []).append((parent, role))
        for site in result:
            node = site.get('source_node', site['node'])
            positions = ([(site['parent'], site['site_role'])] if site['parent'] is not None
                         else list(parents.get(id(node), ())))
            roots = [0] if node is clause else []
            if site.get('lexical'):
                for original in nodes:
                    if original.composition is node.composition:
                        positions.extend(parents.get(id(original), ()))
                        if original is clause:
                            roots.append(0)
            roots.extend(role for parent, role in positions if parent is clause)
            site['root_roles'] = tuple(sorted(set(roots)))
            site['asserted'] = tuple(parent.meaning.roles[2-role] for parent, role in positions
                if role in (0, 2) and site['node'].reference_request.get('forward_identity') is not None
                and (parent.relation == 'equal' or bindings(parent.meaning).get('_equality')))
        # Scoring and admission must enumerate the same sites. Introductions
        # precede their uses even when a later wrapper is the reading's root.
        result.sort(key=lambda site: (site['node'].identity or {}).get('action')
                    if (site['node'].identity or {}).get('action') is not None else -1)
        return result

    def pending_names(self, clause, individuals, stream):
        """Names whose chosen token is in this close's accessible frames."""
        from ThoughtReferences import bindings
        names, _ = self._arrival_constituents(clause)
        result, store, mtm = set(), self.store, self.components.owner.mtm
        for address in store._forward_addresses(names) if store is not None and names else ():
            index = store.location(address)
            if not isinstance(index, int) or (stream is not None and bool(store.stream[index].any())
                    and (stream >= store.stream.shape[1] or not bool(store.stream[index, stream]))):
                continue
            value = store.raw_meaning_of(index)
            for role, name in bindings(value).get('_forward_references', ()):
                reference = value.role_refs[role]
                occurrence = (reference if reference is not None and reference[0] == 'ltm'
                              else store.occurrence_of(index))
                if name in names and mtm.token_of(occurrence) in individuals:
                    result.add(name)
        return tuple(sorted(result))

    def preview(self, clause, chunks, *, document=None, position=None, stream=None, departure=-1, reading=None):
        """One decision per constituent, reading the retained chunk field."""
        records = self.accessible(chunks,stream=stream,include_pool=True)
        individuals = self.individuals(records)
        pending_names = self.pending_names(clause, individuals, stream)
        items, cost = [], self.policy.new_zeros(())
        plan = dict(items=items, records=records, pending_names=pending_names,
                    decisions=[], departure=int(departure), stream=stream)
        handles, introductions, local_bindings = {}, {}, {}
        for site in self.sites(clause, pending_names=pending_names):
            node = site['node']
            request = node.reference_request
            from MemoryLink import Occurrence, member_order
            concept = request.get('forward_identity')
            if concept is None and node.subject_word_id >= 0 and len(request['leaves']) == 1:
                concept = node.subject_word_id
            # An unnamed bind reads its resulting idea: the selected native
            # referent supplies its order. A named member already has a read
            # concept, whose order must agree before the choice is offered.
            order = (None if concept is None else
                member_order(self.components.owner, Occurrence((), concept=concept)))
            mode, role = request['mode'], request.get('role', 0)
            content = self.coordinates(node.point)
            predicate = self.predicate_coordinates(request.get('predicate'))
            cue = self.cue(request.get('cue'))
            if not self.has_word(node) and not bool(content.any()):
                content = self.expected_content(predicate, role)
            menu = individuals
            if introductions and node.composition is not None:
                # A trial-local introduction is constructed from its operands.
                # It cannot serve as the referent of one of its own dependencies.
                menu = {key: record for key, record in individuals.items()
                        if key not in introductions or not self._term_depends_on(
                            introductions[key], node.composition, local_bindings)}
            if (request.get('sign_leaf') or request.get('no_op_leaf')):
                choices, phi = (), self.policy.new_empty((0, 12))
            else:
                choices, phi, logits = self.score(content, predicate, cue, role, menu,
                    stream=stream, asserted=site['asserted'], member_order=order)
            selected, probabilities = None, None
            if mode == 'mint':
                # A trial-local handle is not a durable identity. It allows
                # later constituents of this same close to bind the new one.
                selected = ('new', len(items))
                signature = content.clone()
                latest = max((other['position'] for other in individuals.values()), default=(-1, -1))
                individuals[selected] = dict(content=signature, role=role,
                    position=(latest[0]+1, len(items)),
                    member_order=self.introduction_order(node, order, keep_reference=False))
                # A redundant mint has a cost; contradictory content does
                # not. This credits the operation chooser without assigning
                # a determiner word to either operation.
                if choices:
                    cost = cost + phi[:, 1].amax().clamp(0, 1)
            elif 'reading_identity' in request:
                handle = request['reading_identity']
                tag, address = node.identity['origins'][handle-1, :2].tolist()
                selected = self.origin_reference(tag, address)
                if self.components.owner.mtm.token_of(selected) is None:
                    raise RuntimeError('a read word lost its pending identity')
                choices, phi = (), None
            elif 'compose_bind' in request:
                handle = request['compose_bind']
                if handle <= 0 and node.identity is not None:
                    # Dummy preserves an already determined own object; the
                    # introducing occurrence remains its owner.
                    selected = handles.get(node.identity['handle'])
                if handle > 0:
                    if node.identity is None:
                        raise RuntimeError('a recorded bind lost its identity transaction')
                    tag, address = node.identity['origins'][handle - 1, :2].tolist()
                    selected = self.origin_reference(tag, address) if tag in (1, 2, 5, 6) else handles.get(handle)
                    if selected is None:
                        raise RuntimeError('a compose bind names an unwritten introduction')
                # The global operation chooser already owns this decision and
                # its R + A credit. Never choose again at the closing boundary.
                choices, phi = (), None
            else:
                index, decision = self.choose(plan, choices, phi, logits,
                    forward_identity=request.get('forward_identity'), individuals=menu)
                probabilities = decision['probabilities']
                selected = None if index is None else choices[index]
                # Retained for diagnosis only; these local costs do not
                # enter R, A, the keep or a training objective.
                cost = cost + (1 if index is None else self.choice_cost(phi, logits, index))
            own_object = mode != 'mint' and selected is None and (self.has_word(node)
                or ('compose_bind' in request and request['compose_bind'] <= 0))
            anonymous = own_object and request.get('forward_identity') is not None
            if own_object:
                selected = ('new', len(items))
                latest = max((other['position'] for other in individuals.values()), default=(-1, -1))
                individuals[selected] = dict(content=content, value=content,
                    role=role, position=(latest[0]+1, len(items)),
                    member_order=self.introduction_order(node, order, keep_reference=True))
            if node.identity is not None and node.identity['handle'] > 0:
                handles.setdefault(node.identity['handle'], selected)
            if selected in introductions and selected not in menu:
                raise ValueError(f'bind to {selected} depends on its own site {request["leaves"]}')
            if node.composition is not None:
                local_bindings[id(node.composition)] = introductions.get(selected, ())
            if mode == 'mint' or own_object:
                # prepare publishes an immutable copy of the introducing term.
                # Its children remain dependencies. Its root is a new object,
                # even when two native roles carry the same original term.
                introductions[selected] = () if node.composition is None else node.composition.children
            selected_order = order
            if selected is not None:
                candidate = individuals.get(selected, {})
                pool = self.components.owner.mtm
                entry = pool.entries.get(pool.token_of(candidate.get('token') or selected))
                if isinstance(selected, int) and candidate.get('token') is None:
                    selected_order = member_order(self.components.owner,
                        Occurrence((), concept=selected))
                else:
                    selected_order = (entry.refinement.get('member_order') if entry is not None
                                      else candidate.get('member_order'))
                # A recorded compose bind already selected its referent. A
                # later lexical definition must not replace that member's
                # order with the order of the word naming the use.
                local = isinstance(selected, tuple) and selected[:1] == ('new',)
                if selected_order is None and not local:
                    selected_order = member_order(self.components.owner,
                        Occurrence(selected, concept=selected if isinstance(selected, int) else None))
                if selected_order is None:
                    raise RuntimeError('a selected introduction has no member order')
            items.append(dict(**site, mode=mode, content=content, predicate=predicate,
                role=role, selected=selected, candidates=tuple(choices),
                features=phi if choices else None, cue=cue,
                probabilities=probabilities, leaves=request['leaves'], anonymous=anonymous,
                own_object=own_object, member_order=selected_order,
                compose_dummy='compose_bind' in request and request['compose_bind'] <= 0))
            if 'record' in request and selected is not None and not own_object:
                items[-1]['record'] = dict(value=request['record'])
            elif 'reading_identity' in request:
                token = self.components.owner.mtm.token_of(selected)
                items[-1]['record'] = dict(value=self.components.owner.mtm.value_of(token).detach().clone())
        plan['cost'] = cost
        plan['individuals'] = individuals
        self._arrival_plan(clause, plan, individuals, stream=stream)
        from MemoryLink import preview as preview_links
        preview_links(self, plan, clause, reading, stream=stream, when=position)
        return plan

    @staticmethod
    def _term_depends_on(operands, site, local_bindings):
        pending, seen = list(operands), set()
        while pending:
            term = pending.pop()
            if term is site:
                return True
            if term is not None and id(term) not in seen:
                seen.add(id(term))
                pending.extend(term.children)
                if id(term) in local_bindings:
                    pending.extend(local_bindings[id(term)])
                elif term.binding is not None:
                    pending.append(term.binding)
        return False

    def _arrival_constituents(self, clause):
        """Only participant constituents and their lexical copies are objects."""
        from ClauseRow import Clause, Composition
        from Meaning import ConceptualMeaning
        from ThoughtReferences import bindings
        from ReferenceParticipants import clause_participants
        nodes, participants, terms = clause_participants(clause,
            self.store.__dict__.get('_part_read_rules', ()))
        names = {site['node'].reference_request['forward_identity']
                 for site in self.sites(clause)
                 if site['node'].reference_request.get('forward_identity') is not None}
        candidates = [node for node in nodes if id(node) in participants
                      and node.point is not None and node.reference_request is None]
        represented = {id(node.composition) for node in nodes}
        for term in terms.values():
            if term.word_id >= 0:
                names.add(term.word_id)
            if not term.children and id(term) not in represented:
                candidates.append(Clause(ConceptualMeaning.from_description(term.point),
                    point=term.point, composition=term, order=term.order,
                    subject_word_id=term.word_id))
            if not term.children and term.word_id >= 0 and term.reference != term.word_id:
                # A chosen operand can differ from its identified lexical
                # constituent. Both belong to this arrival's candidate set.
                lexical = self.store._concept_point(term.word_id)
                if lexical is not None:
                    candidates.append(Clause(ConceptualMeaning.from_description(lexical),
                        point=lexical, composition=Composition(lexical, reference=term.word_id,
                            word_id=term.word_id, leaves=term.leaves),
                        subject_word_id=term.word_id))
        return names, candidates

    def _arrival_plan(self, clause, plan, individuals, *, stream):
        """Plan each pending site's bind with the same resolver and cost.

        Arrival handles are temporary constituent addresses. Only prepare and
        the kept admission materialize selected constituents; this read neither
        opens a pool entry nor changes an older row.
        """
        from ThoughtReferences import bindings
        names, candidates = self._arrival_constituents(clause)
        sites = {}
        mtm, store = self.components.owner.mtm, self.store
        asserted = {}
        for item in plan['items']:
            name = item['node'].reference_request.get('forward_identity')
            if name is not None:
                asserted.setdefault(name, []).extend(item['asserted'])
        # A posting locates a site; it does not bring its token into focus.
        # Use the occurrence records before the read-time split/merge pass,
        # which may represent an accessible token under another group key.
        accessible_tokens = {mtm.token_of(store.occurrence_of(location))
                             for location, _ in plan['records'].values()
                             if isinstance(location, int)}
        for address in sorted(store._forward_addresses(names)) if store is not None and names else ():
            location = store.location(address)
            if not isinstance(location, int):
                continue
            if (stream is not None and bool(store.stream[location].any())
                    and (stream >= store.stream.shape[1] or not bool(store.stream[location, stream]))):
                continue
            value = store.raw_meaning_of(location)
            data = bindings(value)
            requests = dict(data.get('_reference_sites', ()))
            for role, name in data.get('_forward_references', ()):
                if name not in names:
                    continue
                reference = value.role_refs[role]
                occurrence = reference if reference is not None and reference[0] == 'ltm' else store.occurrence_of(location)
                token = mtm.token_of(occurrence)
                if token != occurrence or token not in accessible_tokens or occurrence in sites:
                    continue
                request = dict(requests.get(role, ()))
                sites[occurrence] = (token, occurrence, dict(forward_identity=name, role=request.get('role', role),
                    content=value.roles[role] if request.get('content') is None else value.roles.new_tensor(request['content']),
                    predicate=value.roles[1] if request.get('predicate') is None else value.roles.new_tensor(request['predicate']),
                    cue=None if request.get('cue') is None else value.roles.new_tensor(request['cue'])))
        for number, item in enumerate(plan['items']):
            request = item['node'].reference_request
            if request.get('forward_identity') is None:
                continue
            selected = item['selected']
            if item.get('anonymous') and isinstance(selected, tuple) and selected[:1] == ('new',):
                sites[('site', number)] = selected, ('site', number), request
        plan['arrivals'], plan['arrival_count'] = [], len(candidates)
        if not sites:
            return
        # A constituent already selected by an ordinary site is represented
        # by that member in the menu. Its lexical occurrence must not also
        # appear as an independent arrival with a different native order.
        planned_members = {id(item['node'].composition) for item in plan['items']
                           if item['node'].composition is not None and item['selected'] in individuals}
        claimed_arrivals = {number for number, node in enumerate(candidates)
                            if id(node.composition) in planned_members}
        # Ordinary sites can share these terms and change their identities.
        # Score the same prepared view that the kept occurrence will write,
        # without changing the pinned STM metadata during this trial read.
        _, candidates = self._prepare(clause, plan, candidates, preview=True)
        from PartClosing import PartClosing
        menu = dict(individuals)
        latest = max((other['position'] for other in menu.values()), default=(-1, -1))
        for number, node in enumerate(candidates):
            if number in claimed_arrivals:
                continue
            kind = None if node.composition is None else self._derivation_symbol(node.composition.reference)
            menu[('arrival', number)] = dict(content=self.coordinates(node.point), value=node.point,
                role=0, position=(latest[0]+1, number), kind=kind,
                word_id=node.subject_word_id, member_order=PartClosing(
                    self.components.owner, store).occurrence_order(node))
        for site, (token, occurrence, request) in sites.items():
            content = request.get('content')
            if content is None:
                content = self.policy.new_zeros(self.store.slots.shape[-1])
            predicate = self.predicate_coordinates(request.get('predicate'))
            cue = self.cue(request.get('cue'))
            from MemoryLink import Occurrence, member_order
            entry = mtm.entries.get(mtm.token_of(token))
            order = (entry.refinement.get('member_order') if entry is not None else
                individuals[token].get('member_order'))
            if order is None:
                order = member_order(self.components.owner,
                    Occurrence((), concept=request['forward_identity']))
            choices, phi, logits = self.score(content, predicate, cue, request.get('role', 0),
                menu, stream=stream, asserted=asserted.get(request['forward_identity'], ()),
                member_order=order)
            index, decision = self.choose(plan, choices, phi, logits,
                forward_identity=request['forward_identity'], individuals=menu)
            selected = None if index is None else choices[index]
            candidate_occurrence = None
            if isinstance(selected, int):
                for address in reversed(menu[selected].get('occurrences', ())):
                    location = store.location(address)
                    if isinstance(location, int):
                        candidate_occurrence = store.occurrence_of(location)
                        break
            plan['cost'] = plan['cost'] + (1 if index is None else self.choice_cost(phi, logits, index))
            plan['arrivals'].append(dict(site=site, site_occurrence=occurrence,
                token=token, selected=selected, candidates=choices,
                content=content, record=None if selected is None else menu[selected], mode='bind',
                features=phi, cue=cue, role=request.get('role', 0),
                forward_identity=request['forward_identity'],
                candidate_occurrence=candidate_occurrence,
                probabilities=decision['probabilities']))

    @torch.no_grad()
    def prepare(self, clause, plan):
        """Attach decisions to the temporary derivation, before any admission."""
        from ClauseRow import Clause, Composition
        from Meaning import ConceptualMeaning
        from ThoughtReferences import bindings, with_slots
        # Validate the scored arrivals before attaching decisions. Composition
        # views write their metadata into the shared STM occurrence, so doing
        # this after rewrite would enumerate the newly bound view instead.
        _, arrival_nodes = self._arrival_constituents(clause)
        if len(arrival_nodes) != plan.get('arrival_count', len(arrival_nodes)):
            raise RuntimeError('admission changed arrival constituents')
        return self._prepare(clause, plan, arrival_nodes)[0]

    @torch.no_grad()
    def _prepare(self, clause, plan, arrival_nodes, *, preview=False):
        """One rewrite for trial reads and the kept pinned composition."""
        from dataclasses import fields
        from ClauseRow import Composition
        from ThoughtReferences import bindings, with_slots
        from SentenceStrata import replace_composition

        def replace_term(term, **changes):
            if preview:
                # A temporary value view must not mutate CompositionView's
                # shared STM occurrence before all choices have been made.
                return Composition(**dict({field.name: getattr(term, field.name)
                    for field in fields(Composition)}, **changes))
            return replace_composition(term, **changes)

        if not plan['items'] and not plan.get('arrivals'):
            if not preview:
                plan['prepared_root'] = clause
            return clause, arrival_nodes
        replacements, terms = {}, {}
        minted, anonymous = {}, set()
        attached = {}
        for item in plan['items']:
            node, selected = item['node'], item['selected']
            reference = None if isinstance(selected, tuple) and selected[:1] == ('new',) else selected
            if not preview:
                item['reference'] = reference
            request = node.reference_request
            token = self.components.owner.mtm.token_of(reference)
            unresolved = (selected in anonymous or (token is not None
                          and self.components.owner.mtm.entries[token].pending))
            if unresolved and request.get('forward_identity') is None and node.subject_word_id >= 0:
                request = dict(request, forward_identity=node.subject_word_id)
            alias = minted.get(selected) if isinstance(selected, tuple) else None
            kind = self._derivation_symbol(reference)
            resolved_bind = (request.get('compose_bind', 0) > 0 and not unresolved
                             and (kind is not None or alias is not None or token is not None))
            site = dict(role=item['role'], leaves=item['leaves'],
                cue=None if request.get('cue') is None else request['cue'].detach().cpu().tolist(),
                content=None if request.get('content') is None else request['content'].detach().cpu().tolist(),
                predicate=None if request.get('predicate') is None else request['predicate'].detach().cpu().tolist())
            meaning = replace(node.meaning, bindings=dict(bindings(node.meaning), _reference_sites=((0, site),)))
            named = request.get('forward_identity') is not None and not resolved_bind
            if resolved_bind:
                # The chooser has already supplied this resolved identity.
                # Its lexical name is not an open reference. Today's Part
                # writer still consumes the retained kind/alias of that bind.
                data = bindings(meaning)
                data['_forward_references'] = tuple(item for item in
                    data.get('_forward_references', ()) if item[0] != 0)
                meaning = with_slots(replace(meaning, bindings=data),
                    (slot for slot in data.get('_open_references', ()) if slot[1] != 0))
            if selected is None or named or unresolved or (request.get('sign_leaf') or request.get('no_op_leaf')):
                meaning = with_slots(meaning, (('referent', 0),), pair=node.evidence)
                if request.get('forward_identity') is not None:
                    meaning = replace(meaning, bindings=dict(bindings(meaning),
                        _forward_references=((0, request['forward_identity']),)))
            stable = isinstance(reference, int) or (isinstance(reference, tuple)
                and reference[:1] in (('ltm',), ('sym',), ('name',)))
            selected_name = isinstance(reference, tuple) and reference[:1] == ('name',)
            if stable and (not named or selected_name):
                # The writer gets the selected native reference from the
                # composition. Remove earlier identity metadata without making
                # the chosen LTM address part of this occurrence's content key.
                data = dict(bindings(meaning))
                data.pop('_identity', None)
                meaning = replace(meaning, bindings=data)
            term = node.composition or Composition(node.point, order=node.order)
            # Two sites can name the same shared occurrence. Its earlier
            # admission already owns the binding; referring to that view
            # again must not turn the occurrence into its own operand.
            if alias is term:
                alias = term.binding
            # Keep the selected occurrence identity even when it has a kind;
            # the kind's symbol is not the identity that composition used.
            from MemoryLink import retained_reference
            # A lexical name reads its admitted symbol (or its word while
            # pending). Only a compose bind selected a particular retained
            # member whose occurrence must survive later denotation changes.
            native = self.address(retained_reference(self.components.owner, reference)
                                  if request.get('compose_bind') else reference)
            source = (term.reference if item.get('own_object') and not item.get('anonymous')
                      else -1)
            updated = replace_term(term, reference=source if native is None else native,
                              order=node.order, binding=alias)
            if item['mode'] == 'mint' or item.get('own_object'):
                minted[selected] = updated
            if item.get('anonymous'):
                anonymous.add(selected)
            if node.composition is not None:
                terms[id(node.composition)] = updated
            replacements[id(item.get('source_node', node))] = replace(node, meaning=meaning, composition=updated,
                refs=(-1 if native is None or unresolved or
                      (named and not isinstance(reference, int)) else native, -1, -1))
            if item.get('parent') is not None:
                attached.setdefault(id(item['parent']), []).append((item['site_role'], node))
        composed = {}
        def rewrite_term(term):
            if term is None:
                return None
            if id(term) not in composed:
                value = terms.get(id(term), term)
                # Bindings can already name the updated term. Both its old
                # and updated address must resolve to the same closing view.
                if id(value) not in composed:
                    composed[id(value)] = replace_term(value,
                        children=tuple(rewrite_term(child) for child in value.children),
                        binding=rewrite_term(value.binding))
                composed[id(term)] = composed[id(value)]
            return composed[id(term)]
        rebuilt = {}
        def rewrite(node):
            if id(node) not in rebuilt:
                value = replacements.get(id(node), node)
                children, refs = list(value.children), list(value.refs)
                for role, site in attached.get(id(node), ()):
                    refs[role] = ('clause', len(children))
                    children.append(site)
                children = tuple(rewrite(c) for c in children)
                # An explicit compose bind owns a child occurrence. If that
                # child is a direct role, its pending name belongs to the
                # enclosing relation's same role as well. Preserve the
                # predicate/complement when thought asks for that referent.
                meaning = value.meaning
                data = bindings(meaning)
                forwards = dict(data.get('_forward_references', ()))
                sites = dict(data.get('_reference_sites', ()))
                opened = list(data.get('_open_references', ()))
                for role, ref in enumerate(refs):
                    if (value.relation is None or role not in (0, 2)
                            or not bool(meaning.role_mask[role])
                            or not (isinstance(ref, tuple) and ref[:1] == ('clause',))):
                        continue
                    child = children[ref[1]].meaning
                    if int(child.role_mask.sum()) != 1 or not bool(child.role_mask[0]):
                        continue
                    marks = bindings(child)
                    name = dict(marks.get('_forward_references', ())).get(0)
                    if name is None:
                        continue
                    forwards[role] = name
                    opened.append(('referent', role))
                    site = dict(marks.get('_reference_sites', ())).get(0)
                    if site is not None:
                        sites[role] = dict(site, role=role)
                if forwards:
                    data.update(_forward_references=tuple(forwards.items()),
                                _reference_sites=tuple(sites.items()))
                    meaning = with_slots(replace(meaning, bindings=data), opened)
                rebuilt[id(node)] = replace(value,
                    meaning=meaning,
                    composition=rewrite_term(node.composition) if node.composition is not None else value.composition,
                    refs=tuple(refs), children=children,
                    companions=tuple(rewrite(c) for c in value.companions))
            return rebuilt[id(node)]
        result = rewrite(clause)
        prepared_candidates = tuple(rewrite(node) for node in arrival_nodes)
        arrivals = {}
        extra = [rewrite(item['node']) for item in plan['items'] if item.get('lexical')]
        for item in plan.get('arrivals', ()):
            selected = item['selected']
            if isinstance(selected, tuple) and selected[:1] == ('arrival',) and selected not in arrivals:
                arrivals[selected] = prepared_candidates[selected[1]]
                extra.append(arrivals[selected])
        previous_root = result
        if extra:
            result = replace(result, companions=(*result.companions, *extra))
        if not preview:
            plan['arrival_written'] = {key: result if node is previous_root else node
                                       for key, node in arrivals.items()}
            for item in plan['items']:
                node = rebuilt[id(item.get('source_node', item['node']))]
                item['written_node'] = result if node is previous_root else node
            plan['prepared_root'] = result
        return result, prepared_candidates

    @torch.no_grad()
    def _bind_with_capture(self, antecedent, occurrence, *, stream, document, position, **kwargs):
        """Retain an invalid identity's evidence at its rejecting bind boundary."""
        pool = self.components.owner.mtm
        try:
            from MemoryLink import Occurrence
            owner = self.components.owner
            reading = kwargs.pop('reading', None)
            if reading is None:
                reading = pool.reading_of(occurrence)
            value = kwargs.pop('value', None)
            if value is None:
                value = pool.read_member(occurrence, reading)
            site = Occurrence(occurrence, concept=reading,
                referent=kwargs.pop('kind'), selected=antecedent, order=kwargs.pop('order', None))
            return owner.link(site, {}, value=value, where=kwargs.pop('where', None),
                when=kwargs.pop('when'), stream=stream)
        except ValueError as error:
            if str(error) != 'an identity requires one finite working value':
                raise
            from Memory import MediumTermMemory
            trace = error.__traceback__
            while trace is not None and trace.tb_frame.f_code is not MediumTermMemory._value.__code__:
                trace = trace.tb_next
            if trace is None:
                raise
            rejected = trace.tb_frame.f_locals['value']
            from pathlib import Path
            from uuid import uuid4
            from CorpusCheckpoint import detached
            model = getattr(self.components.owner, '_model', None)
            directory = Path(getattr(model, '_receipt_directory', 'output'))
            directory.mkdir(parents=True, exist_ok=True)
            token = pool.token_of(antecedent)
            entry = pool.entries.get(token)
            capture = dict(version=1, error=str(error),
                working_value=detached(rejected), antecedent=antecedent,
                occurrence=occurrence, token=token,
                members=() if entry is None else entry.occurrences,
                pool=pool.state(), stream=stream, document=document, position=position,
                arguments=detached(kwargs))
            path = directory / f'mtm-value-rejection-{uuid4().hex}.pt'
            torch.save(capture, path)
            self.last_identity_rejection = str(path)
            raise

    @torch.no_grad()
    def commit(self, plan, row, written, *, document, position, training,
               verb_action=None, errors=None, modifier_action=None, reading=None):
        """Record training inputs and the kept working decision, never rewrite LTM."""
        kept_reading = reading
        from collections import OrderedDict
        from IdentityRows import point
        from Spaces import _concept_alloc_of
        alloc = _concept_alloc_of(self.components.owner)
        mtm = self.components.owner.mtm
        previous = dict(mtm.entries) if verb_action is not None else {}
        observed = set()
        from RowTime import encoder, stamp
        when = int(encoder().decode_index(stamp(self.store)))
        mentions = self.__dict__.setdefault('_mentions', OrderedDict())
        allocated, audit, uses, sites, native_tokens = {}, [], [], {}, {}
        closed_mentions, refinements = set(), []
        mention_occurrences = plan['mention_occurrences'] = {}
        stream = plan.get('stream')
        def open_reference(occurrence, *, kind, when, value, reading=None, native=None,
                           pending=False, where=None, retain=(), order=None):
            from MemoryLink import Occurrence
            return self.components.owner.link(Occurrence(occurrence, concept=reading,
                referent=kind, native=native, pending=pending, selected=None, retain=retain, order=order),
                {}, value=value, where=where, when=when, stream=stream)
        def bind(antecedent, occurrence, **kwargs):
            return self._bind_with_capture(antecedent, occurrence, stream=stream,
                document=document, position=position, **kwargs)
        def plan_use(reference, occurrence, kind, value, reading, where, item):
            order = item['member_order']
            if order is None:
                from MemoryLink import Occurrence, member_order
                entry = mtm.entries.get(mtm.token_of(reference))
                order = None if entry is None else entry.refinement.get('member_order')
                if order is None:
                    order = member_order(self.components.owner,
                        Occurrence(reference, concept=reference if isinstance(reference, int) else None))
                item['member_order'] = order
            uses.append((reference, occurrence, kind, value, reading, where, order))
        from ChainRead import envelope
        from Memory import OCCURRENCE
        retained = tuple(envelope(self.store, side)
            for left, right, _, _, null in getattr(self.components.owner, '_kept_definition_witnesses', ())
            if null for side in (left, right) if side[0] == OCCURRENCE)
        for item in plan['items']:
            if item['mode'] == 'bind':
                mtm.keep_access(item['reference'], stream)
        for number, item in enumerate(plan['items']):
            node = item['written_node']
            location = written.get(id(node), row if node is plan['prepared_root'] else None)
            if location is None:
                raise RuntimeError('kept reference constituent has no closing address')
            occurrence = written.get(('mention', id(node)))
            if occurrence is None and node is plan['prepared_root']:
                occurrence = written.get('root_mention')
            if occurrence is None:
                occurrence = self.store.occurrence_of(location)
            if occurrence[:1] == ('sym',):
                raise ValueError(f'description {item["leaves"]!r} has no own occurrence: {occurrence!r}')
            mention_occurrences[id(node)] = occurrence
            sites[('site', number)] = occurrence
            reference = item['reference']
            if item['mode'] == 'mint' or item.get('own_object'):
                symbol = written.get(('symbol', id(node)))
                reference = symbol if symbol in alloc.individuals else occurrence
                allocated[('new', number)] = reference
            elif isinstance(item['selected'], tuple) and item['selected'][:1] == ('new',):
                reference = allocated.get(item['selected'])
            item['reference'] = reference
            reading = (node.reference_request or {}).get('forward_identity')
            if reading is None and node.subject_word_id >= 0 and len(item['leaves']) == 1:
                reading = node.subject_word_id
            where = node.where
            if kept_reading is not None and kept_reading.symbol_where is not None and item['leaves']:
                where = kept_reading.symbol_where[item['leaves'][-1]]
            # Binding a second timeline records the alias and its bracket,
            # while that timeline's event publishes its own value below.
            joined_value = None if node.identity is not None and node.identity.get('alternative') else node.point
            if (node.reference_request or {}).get('sign_leaf') or (node.reference_request or {}).get('no_op_leaf'):
                # The chooser selected a leaf, not an antecedent. The envelope
                # gives this mention its own time and place until ordinary MTM expiry.
                content = (node.reference_request or {}).get('percept')
                value = torch.zeros_like(node.point) if content is None else content
                open_reference(occurrence, kind=NULL_CONCEPT, when=when, value=value, where=where, retain=retained)
                closed_mentions.add(occurrence)
            elif isinstance(reference, tuple) and reference[:1] == ('candidate',):
                # A selected familiar concept is a use, not an identity
                # mention or another field observation. Its centroid/count
                # are owned by the open pass and its mint by the update.
                closed_mentions.add(reference)
            elif isinstance(reference, tuple) and reference[:1] in (('ltm',), ('name',)):
                from RowTime import encoder, stamp
                kind = self._derivation_symbol(occurrence)
                token = mtm.token_of(reference)
                if reference[:1] == ('name',) and (token is None or mtm.entries[token].pending):
                    kind = None
                elif kind is None and item['mode'] != 'mint' and not item.get('anonymous'):
                    kind = self._derivation_symbol(reference)
                if item['mode'] == 'mint' or item.get('own_object'):
                    open_reference(occurrence, kind=None if item.get('anonymous') else kind,
                             when=when, value=node.point, reading=reading,
                             pending=bool(item.get('anonymous')), where=where, retain=retained,
                             order=item['member_order'])
                    closed_mentions.add(occurrence)
                else:
                    plan_use(reference, occurrence, kind, joined_value, reading, where, item)
            elif isinstance(reference, int) and reference in alloc.individuals:
                from ConceptTrust import observe
                observe(self.components.owner, reference, node.point)
                kind = self._derivation_symbol(occurrence)
                token = plan['individuals'].get(reference, {}).get('token')
                if token is None:
                    token = open_reference(occurrence, kind=reference, when=when, reading=reading,
                             native=reference, value=torch.maximum(point(self.components.owner, reference), node.point),
                             where=where, retain=retained, order=item['member_order'])
                    closed_mentions.add(occurrence)
                else:
                    plan_use(token, occurrence, kind, joined_value, reading, where, item)
                native_tokens[reference] = token
            if occurrence in closed_mentions:
                mtm.keep_access(occurrence, stream)
            if item['features'] is not None:
                held = dict(candidates=tuple(allocated.get(identity, identity) for identity in item['candidates']),
                    features=item['features'].detach().cpu().tolist(), cue=item['cue'].detach().cpu().tolist())
                mentions[occurrence] = ((item['role'], held),)
                mentions.move_to_end(occurrence)
            audit.append(dict(leaves=item['leaves'], occurrence=occurrence[-1], reference=reference,
                operation=item['mode'], candidates=tuple(
                    allocated.get(identity, identity) for identity in item['candidates']),
                missing_reason=None if reference is not None else 'no_accessible_individual'))
        for item in plan.get('arrivals', ()):
            token = sites.get(item['site_occurrence'], item['site_occurrence'])
            item['site_occurrence'] = token
            # A joined site's earlier identity is settled for this round.
            # Planning and committing both guard it: another kept refinement
            # in this close may have joined it since the plan was made.
            if mtm.token_of(token) != token:
                item['reference'] = mtm.token_of(token)
                continue
            selected = item.get('candidate_occurrence') or allocated.get(item['selected'], item['selected'])
            native = allocated.get(item['selected'], item['selected'])
            native_token = (native_tokens.get(native) or plan['individuals'].get(native, {}).get('token')
                            if isinstance(native, int) else None)
            if native_token is not None and native_token != mtm.token_of(token):
                # Native individuals now have working entries too. A pending
                # singleton joins that accessible value rather than creating
                # a second, independently mutable value for the same choice.
                if mtm.entries[token].occurrences == (token,):
                    mtm.merge(token, native_token, when=when)
                    if isinstance(selected, int):
                        selected = native_token
                    elif mtm.token_of(selected) is None:
                        bind(native_token, selected, kind=self._derivation_symbol(selected),
                            value=self.store.point_of_row(selected[-1]), when=when)
                else:
                    selected = None
            node = plan.get('arrival_written', {}).get(selected)
            if node is not None:
                location = written.get(id(node), row if node is plan['prepared_root'] else None)
                selected = None if location is None else self.store.occurrence_of(location)
            if isinstance(selected, int) and selected in alloc.individuals and mtm.token_of(token) is not None:
                # A native identity needs no synthetic occurrence holding its
                # working vector. Refine through the existing introduction.
                from RowTime import encoder, stamp
                bind(token, token, kind=selected,
                         when=when, value=point(self.components.owner, selected))
                selected = token
            if (isinstance(selected, tuple) and selected[:1] == ('ltm',)
                    and selected != token and mtm.token_of(token) is not None):
                from RowTime import encoder, stamp
                when = int(encoder().decode_index(stamp(self.store)))
                if mtm.token_of(selected) not in (None, mtm.token_of(token)):
                    if token in mtm.entries and mtm.entries[token].occurrences == (token,):
                        mtm.merge(token, selected, when=when)
                    else:
                        selected = None
                elif mtm.token_of(selected) == mtm.token_of(token):
                    # Re-observing an existing member is no new use.
                    pass
                else:
                    bind(token, selected, kind=self._derivation_symbol(selected), when=when,
                             value=self.store.point_of_row(selected[-1]))
            item['reference'] = selected if selected is not None else token
            if selected is not None:
                refinements.append(selected)
            audit.append(dict(operation='bind', refinement=True, occurrence=token[-1],
                reference=item['reference'], selected=item['selected'], candidates=item['candidates'],
                missing_reason=('no_selected_candidate' if selected is None else
                                None if mtm.token_of(token) is not None else 'no_accessible_individual')))
        # Refinement may join a pending token to another antecedent. Attach
        # this close's uses to the owner of their chosen occurrence afterwards.
        for reference, occurrence, kind, value, reading, where, order in uses:
            token = bind(reference, occurrence, kind=kind, value=value, reading=reading, when=when, where=where, order=order)
            if verb_action is not None and value is not None and bool(self.coordinates(value).ne(0).any()):
                observed.add(token)
            closed_mentions.add(occurrence)
        events = self.link_events(plan, written, when=when, closed_mentions=closed_mentions,
                                    observed=observed, verb_action=verb_action)
        from VerbLearning import observe
        self.last_verb_observation = observe(verb_action, mtm, previous, observed, training=training, errors=errors, row=stream)
        from ModalLearning import observe as observe_modal
        self.last_modal_observation = observe_modal(modifier_action, mtm, events,
            training=training, errors=errors, row=stream)
        # Several kept closes can share one model-present tick. Actual
        # mention/refinement/event references, not equal timestamps, refresh
        # the pool's independent kept-close access counter.
        state_link = plan.get('state_link')
        if state_link is not None:
            occurrence, value, where = state_link
            self.components.owner.link(occurrence, {}, value=value, where=where, when=when, stream=stream)
        from FamiliarTier import finish_definitions
        finish_definitions(self.components.owner)
        mtm.close(self.store, stream=plan.get('stream'),
            references=(*closed_mentions, *refinements, *events,
                *getattr(self.components.owner, '_field_activated_references', {}).get(plan.get('stream'), ())))
        from MemoryConsolidation import consolidate
        owner = self.components.owner
        derived = getattr(owner.similarity_codebook, 'mereology', None)
        batch = int(plan.get('stream') or 0)
        context = None if derived is None else derived.observe_close(kept_reading, batch)
        key = None if derived is None or kept_reading is None else derived.reading_key(kept_reading)
        owner.__dict__.setdefault('_mtm_placement_contexts', {})[batch] = (context, key)
        consolidate(owner, self.store, context=context, key=key, batch=batch)
        while len(mentions) > self.components.source_capacity:
            mentions.popitem(last=False)
        self.last_audit = tuple(audit)

    def committed_chunks(self, state, plan, written, *, stream):
        """Remap fork-local reservations and publish the kept pool values once.

        Parallel streams may reserve the same free transient seat in separate
        trials. The close assigns actual pool seats; no trial mutates the pool.
        The journal's compact handle map identifies those admissions exactly.
        """
        from Memory import STMChunks as chunks
        from PrependedSymbols import symbolize, chunk_token
        owner, pool = self.components.owner, self.components.owner.mtm
        mapping = {}
        for item in plan['items']:
            node = item['written_node']
            record = node.identity
            if record is None or record.get('ids') is None or not record['handle']:
                continue
            occurrence = plan.get('mention_occurrences', {}).get(id(node))
            if occurrence is None:
                occurrence = written.get(('occurrence', id(node)), written.get('root_occurrence')
                                         if node is plan['prepared_root'] else None)
                if occurrence is None:
                    continue
            token = pool.token_of(occurrence)
            if token is not None:
                mapping[int(record['ids'][record['handle']-1])] = token
        result = [value.clone() for value in state]
        for index, identity in enumerate(state[chunks.IDENTITY][stream].tolist()):
            if identity < 0:
                continue
            refs = state[chunks.REFS][stream,index]
            token = (chunk_token(pool,self.store,state,stream,index)
                     if bool((refs[:,0] == chunks.OCCURRENCE).any()) or int(refs[1,0]) == chunks.IDENTITY_REF
                     else mapping.get(identity,pool.token_of(identity)))
            if token is None or pool.entries[token].refinement.get('member_kind', 'reference') in ('state', 'event', 'percept'):
                result[chunks.IDENTITY][stream,index] = -1
                result[chunks.REFS][stream,index,1] = 0
                continue
            entry = pool.entries[token]
            assigned = entry.symbol if entry.symbol is not None else entry.transient
            result[chunks.IDENTITY][stream,index] = assigned
            refs = result[chunks.REFS][stream,index]
            if int(refs[0,0]) == chunks.IDENTITY_REF and int(refs[0,1]) == identity:
                refs[0] = refs.new_tensor((chunks.SYMBOL if entry.symbol is not None else chunks.IDENTITY_REF, assigned))
            refs[1] = refs.new_tensor((chunks.IDENTITY_REF if entry.order is not None or token[0] == 'name' else chunks.OCCURRENCE,
                                      -token[1]-1 if token[0] == 'name' else token[-1]))
            timeline = int(state[chunks.TIMELINE][stream,index])
            if timeline >= len(entry.timelines):
                # Conceiving a branch does not publish it. A recent top can
                # retain its written question, never this trial-only identity.
                if int(state[chunks.STACK][stream,index]) >= 0:
                    result[chunks.IDENTITY][stream,index] = -1
                    result[chunks.TIMELINE][stream,index] = 0
                    refs[1] = 0
                else:
                    mask = torch.zeros_like(result[chunks.STACK],dtype=torch.bool)
                    mask[stream,index] = True
                    result = list(chunks.clear(tuple(result),mask))
                continue
            if int(state[chunks.STACK][stream,index]) < 0:
                result[0][stream,index] = symbolize(owner,self.store,token,entry,timeline).to(result[0])
                result[chunks.WHERE][stream,index] = entry.where.to(result[chunks.WHERE])
                if self.store.when_start is not None and entry.order is None:
                    result[0][stream,index,self.store.when_start:self.store.when_start+4] = result[chunks.WHEN][stream,index]
                    if self.store.when_start >= 4:
                        result[0][stream,index,self.store.when_start-4:self.store.when_start] = result[chunks.WHERE][stream,index]
        return tuple(result)

    def link_events(self, plan, written, *, when, closed_mentions=(), observed=None, verb_action=None):
        """Read the kept role correspondence, never infer roles from a value."""
        from ClauseRow import ClausePredicate
        if plan.get('prepared_root') is None or tuple(plan['prepared_root'].evidence) == (0., 0.):
            return set()
        mtm = self.components.owner.mtm
        identities, introduced, referenced = {}, set(), set()
        new_events = {}
        def occurrence_of(node):
            return written.get(('occurrence', id(node)), written.get('root_occurrence')
                               if node is plan['prepared_root'] else None)
        for item in plan['items']:
            node = item['written_node']
            # The reading close can attach a durable referent to its located
            # word occurrence. Event roles must use that actual mention,
            # while event addresses themselves remain the written rows.
            occurrence = plan.get('mention_occurrences', {}).get(id(node), occurrence_of(node))
            if node.identity is not None and occurrence in closed_mentions:
                introduced.add(node.identity['handle'])
            token = mtm.token_of(occurrence)
            if token is None:
                continue
            if node.identity is not None and node.identity['handle']:
                identities[node.identity['handle']] = token
            if node.identity is not None:
                # Captured programs publish their completed events below.
                # The direct-clause adapter must not add a containing event.
                continue
            parent = item.get('parent') or plan['prepared_root']
            if parent is node:
                # A standalone mention is not an event of a predicate merely
                # because its own reading also supplies the reference cue.
                continue
            event = occurrence_of(parent) or occurrence
            verb = None
            if parent.factored_refs is not None:
                verb = parent.factored_refs[1]
                if isinstance(verb, ClausePredicate):
                    verb = verb.identity
            if type(verb) is not int or verb in (-1, 0):
                # Direct clause fixtures carry the same correspondence in
                # their temporary composition instead of a captured journal.
                pending = [parent.composition] if parent.composition is not None else []
                while pending:
                    term = pending.pop()
                    if (not term.children and type(term.reference) is int
                            and term.reference not in (-1, 0)
                            and torch.equal(self.predicate_coordinates(term.point), item['predicate'])):
                        verb = term.reference
                        break
                    pending.extend(reversed(term.children))
            if type(verb) is int and verb not in (-1, 0):
                mtm.event(token, event, verb=self._event_symbol(verb), role=item['role'], when=when)
                referenced.add(token)
                new_events[event] = Ellipsis
        pending, nodes, seen = [plan['prepared_root']], [], set()
        while pending:
            node = pending.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if node.identity is not None and node.identity.get('event') is not None:
                nodes.append(node)
            pending.extend((*node.children, *node.companions))
        for node in sorted(nodes, key=lambda node: node.identity.get('event_order', node.identity['action'])):
            record = node.identity
            _, subject, obj, verb, _, changed = record['event']
            if not subject:
                continue
            event = written.get(('event_occurrence', id(node)))
            if event is None and node is plan['prepared_root']:
                event = written.get('root_event_occurrence')
            if event is None:
                event = occurrence_of(node)
            if event is None:
                raise RuntimeError('a kept identity event has no occurrence')
            if node.factored_refs is not None:
                predicate = node.factored_refs[1]
                if isinstance(predicate, ClausePredicate):
                    verb = predicate.identity
            else:
                # Use the actual carried reading when the capture's predicate
                # names a lexical word without its own conceptual seat.
                terms = [node.composition] if node.composition is not None else []
                while terms:
                    term = terms.pop()
                    if not term.children and term.word_id == verb and term.reference > 0:
                        verb = term.reference
                        break
                    terms.extend(reversed(term.children))
            if int(verb) <= 0:
                # A structural action can change a participant without a
                # lexical predicate. Its a-priori operation concept is the
                # event's identity; an absent word is not a verb id.
                verb = self.components.owner._concept_allocator.operator_concepts[record['operation']]
            verb = self._event_symbol(int(verb))
            for handle, role in ((subject, 0), (obj, 2)):
                if not handle:
                    continue
                token = identities.get(handle)
                if token is None:
                    tag, address = record['origins'][handle-1, :2].tolist()
                    reference = self.origin_reference(tag, address)
                    token = mtm.token_of(reference)
                if token is None:
                    # Capacity can retire an earlier mention during this
                    # same close. Its event history ends with its token.
                    # A determination that never closed is still an error.
                    if int(record['origins'][handle-1, 0]) == 3 and handle not in introduced:
                        raise RuntimeError('a kept identity event lost its determined introduction')
                    continue
                if mtm.entries[token].order is not None:
                    referenced.add(token)
                    continue
                alternative = record.get('alternative', False)
                timeline = (mtm.alternative(token, when=when,
                            before=record.get('before') if role == 0 else None)
                            if alternative else 0)
                is_action = role == 0 and record.get('operation') == 'lift'
                basis = record.get('basis')
                if is_action and basis is None and verb_action is not None:
                    basis = verb_action.groups.assignments.get(int(verb), -1)
                result = record.get('result')
                mtm.event(token, event, verb=int(verb), role=role, when=when, timeline=timeline,
                    before=record.get('before') if role == 0 and changed else None,
                    verb_meaning=record.get('verb_meaning') if is_action else None,
                    basis=basis if is_action else None,
                    case_value=record.get('case_value') if role == 0 else None,
                    modifiers=record.get('modifiers', ()) if role == 0 and alternative else ())
                referenced.add(token)
                if not alternative and role == 0:
                    new_events[event] = plan.get('link_choices', {}).get(('event', record['action']), Ellipsis)
                if role == 0 and changed:
                    values = record.get('timeline_values') if alternative else record['values']
                    if values is None:
                        raise RuntimeError('a kept alternative lost its numerical state')
                    if not alternative and observed is not None and record.get('operation') in ('part', 'whole', 'equal'):
                        observed.add(token)
        # Event paths use the same link rule as mentions and kept states.
        from MemoryLink import Occurrence
        owner = self.components.owner
        for token in tuple(referenced):
            entry = mtm.entries[token]
            for branch in entry.timelines:
                for event in branch.events:
                    if event.occurrence not in new_events or event.role != 0:
                        continue
                    owner.link(Occurrence(event.occurrence, kind='event', selected=new_events[event.occurrence]), {},
                        value=mtm.event_value(event), where=None, when=when, stream=plan.get('stream'))
        return referenced

    def get_extra_state(self):
        from IndependentComponents import _pack_state
        return _pack_state(dict(version=3))

    def set_extra_state(self, state):
        from IndependentComponents import _unpack_state
        if _unpack_state(state).get('version') != 3:
            raise ValueError('reference checkpoint requires the native-row migration')
        if self.store is not None:
            self.store._participant_cache = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if not any(key.startswith(prefix) for key in state_dict):
            for key, value in self.state_dict().items():
                state_dict[prefix + key] = value
        key = prefix+'policy'
        if key in state_dict and state_dict[key].shape == (8,):
            state_dict[key] = torch.cat((state_dict[key], state_dict[key].new_zeros(4)))
            self.policy_migration = dict(before=8, after=12, added_weights=[0., 0., 0., 0.])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
