"""Individual membership at a kept close, with access-bounded revision.

Operations decide mint versus bind. This owner decides only *which* accessible
individual explains a binding. Its coordinates are the existing event chart
and A columns; it has no vocabulary, POS list, kind inventory or text parser.
Trial plans are read-only. Occurrence points are immutable under revision.
"""
from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F


def constituents(clause):
    seen, result = set(), []
    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        if node.reference_request is not None and node.order == 1:
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
        # Membership and prediction start usable; salience and surface cues
        # start neutral and are credited by the same closing evidence.
        self.policy = nn.Parameter(torch.tensor([1., 1., 0., 0., 0.]))
        derived = getattr(getattr(components.owner, 'similarity_codebook', None), 'mereology', None)
        form_width = int(components.owner.nWhat) if derived is None else int(derived.percept_event_width)
        self.cue_policy = nn.Parameter(torch.zeros(form_width, 3))
        self.seen, self.cooccurrence, self.associations, self.predicates = {}, {}, {}, {}
        self.total = 0.
        self.last_audit = ()

    @property
    def store(self):
        return self.components.owner._closed_clause_store()

    def address(self, reference):
        if isinstance(reference, tuple):
            if len(reference) != 3 or reference[:2] != ('ltm', self.store.address_domain):
                return None
            return int(reference[2])
        return reference

    def coordinates(self, point):
        chart = self.components._ready()
        if point is None:
            return chart.basis.new_zeros(2 * self.components.capacity)
        return chart.expand(chart.project(point.detach()))

    def _evidence(self, value):
        # A's native coordinates are correlations with the event chart. Read
        # coefficients through that same chart's Gram solve, so crosstalk is
        # not treated as an observed property. This creates no second learned
        # dictionary and leaves the sparse objective's coordinate system intact.
        chart = self.components._ready()
        gram = getattr(chart, '_reference_gram', None)
        if gram is None:
            gram = chart.basis @ chart.basis.T
            chart._reference_gram = gram
            chart._reference_gram_inverse = torch.linalg.pinv(gram.float()).to(gram)
        count = len(chart.rows)
        if not count:
            return torch.zeros_like(value)
        projected = value.index_select(-1, chart.features).reshape(2, count)
        weights = projected @ chart._reference_gram_inverse.to(value)
        weights = weights / weights.abs().amax().clamp_min(1e-8)
        return chart.expand(weights.flatten().clamp(0, 1))

    def _coordinates(self, evidence):
        chart = self.components._ready()
        count = len(chart.rows)
        if not count:
            return torch.zeros_like(evidence)
        weights = evidence.index_select(-1, chart.features).reshape(2, count)
        gram = chart.basis @ chart.basis.T
        return chart.expand((weights @ gram.to(evidence)).flatten())

    def absorb(self, signature, content, predicate):
        evidence = torch.maximum(self._evidence(signature), self._evidence(content))
        return self._coordinates((evidence + .5 * self._evidence(predicate)).clamp_max(1))

    def contradicts(self, left, right):
        return any(self.seen.get(a, 0) >= 8 and self.seen.get(b, 0) >= 8 and
            self.cooccurrence.get((min(a, b), max(a, b)), 0) == 0
            for a in self.support(left) for b in self.support(right))

    def support(self, value):
        value = self._evidence(value)
        return {int(i): float(value[i]) for i in (value >= .3).nonzero().flatten()}

    def explanation(self, content, signature):
        present = self.support(content)
        if not present:
            return signature.sum() * 0
        evidence = self._evidence(signature)
        values = []
        for atom, strength in present.items():
            excluded = [other for other in self.seen
                if self.seen.get(atom, 0) >= 8 and self.seen[other] >= 8
                and self.cooccurrence.get((min(atom, other), max(atom, other)), 0) == 0]
            contrary = evidence[excluded].amax() if excluded else evidence.sum() * 0
            values.append(strength * (evidence[atom] - contrary))
        return torch.stack(values).sum() / sum(present.values())

    def prediction(self, signature, predicate):
        atoms, heads = self.support(signature), self.support(predicate)
        result = self.explanation(predicate, signature)
        if not heads or not atoms:
            return result
        vocabulary = max(1, len(self.predicates))
        for head, strength in heads.items():
            base = (self.predicates.get(head, 0.) + 1.) / (self.total + vocabulary)
            for atom, weight in atoms.items():
                count = self.associations.get((atom, head), 0.)
                if count:
                    probability = (count + 10 * base) / (self.seen.get(atom, 0.) + 10)
                    result = result + strength * weight * math.log(max(probability / base, 1e-8)) / sum(atoms.values())
        return result

    def accessible(self, frames):
        """Read only supplied STM/cued frames and their owned NP occurrences."""
        store = self.store
        records = {}
        if store is None:
            return records
        for frame in frames:
            row = store.index_of_row(frame.row_id)
            if row is None:
                continue
            from ThoughtReferences import bindings
            for reference in self.occurrences(row):
                address = self.address(reference)
                if address is None:
                    continue
                index = store.index_of_row(address)
                if index is None or int(store.order[index]) != 1:
                    continue
                data = bindings(store.meaning_of(index)).get('_individual_context')
                identity = int(store.refs[index, 0])
                if data is not None and identity in self.components.nouns.ids:
                    records[int(address)] = (index, dict(data))
        # Access is bounded by the frames, not a scan through all A columns.
        return records

    def occurrences(self, row):
        from ThoughtReferences import bindings
        meaning = self.store.meaning_of(row)
        if meaning is None:
            return ()
        data = bindings(meaning)
        owned = data.get('_individual_occurrences', ())
        return ((*owned, self.store.occurrence_of(row))
                if '_individual_context' in data else owned)

    def preview(self, clause, frames, *, document=None, position=None):
        """One decision per constituent; no allocation, writes or statistics."""
        if position is not None and frames:
            from ReferenceContext import exclude_replaced_occurrences
            ids = torch.tensor([frame.row_id for frame in frames], device=self.policy.device)
            allowed = exclude_replaced_occurrences(self.store, ids, torch.ones_like(ids, dtype=torch.bool),
                                                  document, position)
            frames = tuple(frame for frame, keep in zip(frames, allowed.tolist()) if keep)
        records = self.accessible(frames)
        individuals = {}
        for address, (row, context) in records.items():
            identity = int(self.store.refs[row, 0])
            position = context['position']
            position = position if isinstance(position, tuple) else (position, -1)
            individuals[identity] = dict(signature=self.components.nouns.directions[str(identity)],
                role=context['role'], position=position)
        items, cost = [], self.policy.sum() * 0
        for node in constituents(clause):
            request = node.reference_request
            mode, role = request['mode'], request.get('role', 0)
            content = self.coordinates(node.point)
            predicate = self.coordinates(request.get('predicate'))
            cue = request.get('cue')
            cue = (self.cue_policy.new_zeros(self.cue_policy.shape[0]) if cue is None else
                   F.normalize(cue[:self.cue_policy.shape[0]].detach().to(self.policy), dim=0))
            ids = list(individuals)
            choices, features = [], []
            for identity in ids:
                candidate = individuals[identity]
                signature = candidate['signature'].to(content)
                explained = self.explanation(content, signature)
                predicted = self.prediction(signature, predicate)
                recency = sum(other['position'] > candidate['position'] for other in individuals.values())
                features.append(torch.stack((explained, predicted,
                    content.new_tensor(-recency / max(1, len(ids))),
                    content.new_tensor(candidate['role'] / 2), content.new_tensor(role / 2))))
                choices.append(identity)
            selected, probabilities = None, None
            if mode == 'mint':
                # A trial-local handle is not a durable identity. It allows
                # later constituents of this same close to bind the new one.
                selected = ('new', len(items))
                signature = self.absorb(torch.zeros_like(content), content, predicate)
                latest = max((other['position'] for other in individuals.values()), default=(-1, -1))
                individuals[selected] = dict(signature=signature, role=role, position=(latest[0]+1, len(items)))
                # A redundant mint has a cost; contradictory content does
                # not. This credits the operation chooser without assigning
                # a determiner word to either operation.
                if features:
                    cost = cost + torch.stack(features)[:, 0].amax().clamp(0, 1)
            elif features:
                phi = torch.stack(features).to(self.policy)
                # Both trial graphs precede either optimizer step. Own the
                # parameter values used by this graph, as the shared chooser
                # does, while retaining their gradient connection.
                logits = phi @ self.policy.clone() + (phi[:, 2:] * (cue @ self.cue_policy.clone())).sum(-1)
                probabilities = logits.softmax(0)
                index = int(logits.detach().argmax())
                selected = choices[index]
                credit = 6 * phi[:, 0] + phi[:, 1]
                target = credit.detach().softmax(0)
                # The evidence cost trains the operation chooser; the
                # posterior cross entropy trains learned salience and cues.
                cost = cost + (1 - phi[index, 0]).clamp_min(0) - phi[index, 1]
                cost = cost - (target * logits.log_softmax(0)).sum()
            else:
                cost = cost + 1
            items.append(dict(node=node, mode=mode, content=content, predicate=predicate,
                role=role, selected=selected, candidates=tuple(choices),
                probabilities=probabilities, leaves=request['leaves']))
        return dict(items=items, cost=cost, records=records)

    @torch.no_grad()
    def _mint(self, signature):
        dictionary = self.components.nouns
        identity = self.components._allocate(signature)
        if identity is None:
            return None
        key = str(identity)
        dictionary.directions[key] = nn.Parameter(signature.detach().clone())
        dictionary.relevance[key] = nn.Parameter(signature.new_tensor(1.))
        dictionary._witnesses[identity] = set()
        dictionary.admissions.append(identity)
        dictionary._next_id = max(dictionary._next_id, identity + 1)
        model = getattr(self.components.owner, '_model', None)
        if model is not None:
            model.__dict__.setdefault('_fresh_synthesis_params', []).extend(
                (dictionary.directions[key], dictionary.relevance[key]))
        return identity

    @torch.no_grad()
    def prepare(self, clause, plan):
        """Prepare kept occurrence rows; allocation follows their successful write."""
        replacements = {}
        for index, item in enumerate(plan['items']):
            reference = item['selected']
            if isinstance(reference, tuple):
                reference = None
            item['reference'] = reference
            node = item['node']
            predicate = node.reference_request.get('predicate')
            children = node.children
            if predicate is not None:
                from ClauseRow import Clause
                from Meaning import ConceptualMeaning
                predicate_node = Clause(ConceptualMeaning.from_description(predicate),
                                        point=predicate, order=0)
                item['predicate_node'] = predicate_node
                children = (*children, predicate_node)
            replacements[id(node)] = replace(node, children=children,
                refs=(-1 if reference is None else reference, -1, -1))
        def rewrite(node):
            value = replacements.get(id(node), node)
            result = replace(value, children=tuple(rewrite(c) for c in value.children),
                             companions=tuple(rewrite(c) for c in value.companions))
            if id(node) in replacements:
                replacements[id(node)] = result
            for item in plan['items']:
                if item.get('predicate_node') is node:
                    item['predicate_node'] = result
            return result
        result = rewrite(clause)
        for item in plan['items']:
            item['written_node'] = replacements[id(item['node'])]
        plan['prepared_root'] = result
        return result

    def _predicate(self, context):
        address = self.address(context.get('predicate'))
        return self.coordinates(None if address is None else self.store.point_of_row(address))

    @torch.no_grad()
    def _learn(self, content, predicate):
        atoms, heads = self.support(content), self.support(predicate)
        for atom in atoms:
            self.seen[atom] = self.seen.get(atom, 0.) + 1
            for other in atoms:
                if atom <= other:
                    key = atom, other
                    self.cooccurrence[key] = self.cooccurrence.get(key, 0.) + 1
            for head in heads:
                key = atom, head
                self.associations[key] = self.associations.get(key, 0.) + 1
        for head in heads:
            self.predicates[head] = self.predicates.get(head, 0.) + 1
            self.total += 1

    def _metadata(self, row, key, value):
        from ThoughtReferences import bindings
        store = self.store
        address = int(store.address_keys[row])
        data = bindings(store.meaning_of(row))
        data[key] = value
        metadata = dict(store._semantic_rows[address])
        metadata['bindings'] = replace(store.meaning_of(row), bindings=data).bindings
        store._semantic_rows[address] = metadata
        store.metadata_required[row] = True
        store._update_semantic_fingerprint(row)

    @torch.no_grad()
    def commit(self, plan, row, written, *, document, position, training):
        store, dictionary = self.store, self.components.nouns
        from Occurrence import document_digest
        document = document_digest(document).hex()
        addresses, audit, allocated = [], [], {}
        for item_number, item in enumerate(plan['items']):
            index = written.get(id(item['written_node']))
            if index is None and item['written_node'] is plan['prepared_root']:
                index = row
            if index is None:
                raise RuntimeError('kept reference constituent was not written')
            address = int(store.row_ids[index])
            reference = item['reference']
            if item['mode'] == 'mint':
                reference = self._mint(self.absorb(torch.zeros_like(item['content']), item['content'], item['predicate']))
                allocated[('new', item_number)] = reference
            elif isinstance(item['selected'], tuple):
                reference = allocated.get(item['selected'])
            item['reference'] = reference
            store.refs[index, 0] = -1 if reference is None else reference
            # Context is semantic membership evidence, not a derivation. The
            # NP's actual content remains exclusively in its occurrence row.
            predicate_row = written.get(id(item.get('predicate_node')))
            context = dict(role=item['role'], position=(int(position), max(item['leaves'], default=-1)), document=document,
                predicate=None if predicate_row is None else store.occurrence_of(predicate_row), mint=item['mode'] == 'mint',
                initial=reference)
            self._metadata(index, '_individual_context', context)
            plan['records'][address] = index, context
            addresses.append(store.occurrence_of(index))
            if reference is not None:
                key = str(reference)
                old = dictionary.directions[key]
                dictionary.directions[key].copy_(self.absorb(old, item['content'].to(old), item['predicate'].to(old)))
                dictionary._witnesses[reference].add((address, int(store.witness_count[index])))
            if training:
                self._learn(item['content'], item['predicate'])
            audit.append(dict(leaves=item['leaves'], occurrence=address, reference=reference,
                operation=item['mode'], candidates=item['candidates'],
                missing_reason=(None if reference is not None else
                    'individual_capacity_exhausted' if item['mode'] == 'mint' or isinstance(item['selected'], tuple)
                    else 'no_accessible_individual')))
        self._metadata(row, '_individual_occurrences', tuple(addresses))
        self.revise(plan['records'], document=document)
        self.last_audit = tuple(audit)

    @torch.no_grad()
    def revise(self, records, *, document):
        """Re-tie accessible occurrences when later content contradicts a group.

        A split is an identity revision, not an implicit mint by a binder.
        Earlier ambiguous occurrences follow the predicate evidence of the
        resulting groups. No coordinates of a stored occurrence are edited.
        """
        store, dictionary = self.store, self.components.nouns
        current = [(address, row, context, self.coordinates(store.point_of_row(address)))
            for address, (row, context) in records.items()
            if context['document'] == document and int(store.refs[row, 0]) in dictionary.ids]
        groups = {}
        for record in current:
            groups.setdefault(int(store.refs[record[1], 0]), []).append(record)
        for identity, members in list(groups.items()):
            anchors = []
            for record in reversed(members):
                if not anchors or any(self.contradicts(record[3], anchor[3]) for anchor in anchors):
                    anchors.append(record)
            if len(anchors) < 2:
                continue
            references = [identity]
            for anchor in anchors[1:]:
                fresh = self._mint(self.absorb(torch.zeros_like(anchor[3]), anchor[3], self._predicate(anchor[2])))
                if fresh is not None:
                    references.append(fresh)
            if len(references) != len(anchors):
                continue
            for address, row, context, content in members:
                predicate = self._predicate(context)
                scores = [self.explanation(content, anchor[3]) +
                    self.explanation(predicate, self._predicate(anchor[2])) for anchor in anchors]
                selected = references[int(torch.stack(scores).argmax())]
                store.refs[row, 0] = selected
                store._update_semantic_fingerprint(row)
        # Merge only on new shared, distinguishing content with predicate
        # agreement. Equal initial kind content alone cannot merge two mints.
        groups = {}
        for record in current:
            groups.setdefault(int(store.refs[record[1], 0]), []).append(record)
        ids = list(groups)
        for offset, left in enumerate(ids):
            for right in ids[offset+1:]:
                a, b = groups[left], groups[right]
                initial = [r for r in (*a, *b) if r[2]['mint']]
                initial_support = set().union(*(self.support(r[3]) for r in initial))
                later_a = [r for r in a if set(self.support(r[3])) - initial_support]
                later_b = [r for r in b if set(self.support(r[3])) - initial_support]
                if not later_a or not later_b:
                    continue
                contradiction = any(self.contradicts(x[3], y[3]) for x in a for y in b)
                corroborated = any(float(self.explanation(x[3], y[3])) >= .9 and
                    float(self.explanation(self._predicate(x[2]), self._predicate(y[2]))) >= .9
                    for x in later_a for y in later_b)
                if corroborated and not contradiction:
                    for _, row, _, _ in b:
                        store.refs[row, 0] = left
                        store._update_semantic_fingerprint(row)
        # The column is the current signature of its accessible membership.
        # Reassignment moves witnesses; old occurrence content never changes.
        touched = {int(store.refs[row, 0]) for _, row, _, _ in current} | set(groups)
        for address, row, _, _ in current:
            for witnesses in dictionary._witnesses.values():
                witnesses.difference_update(w for w in tuple(witnesses) if w[0] == address)
            identity = int(store.refs[row, 0])
            dictionary._witnesses.setdefault(identity, set()).add((address, int(store.witness_count[row])))
        for identity in touched:
            # Retained historical members still contribute to this column's
            # signature. Only the bounded records above can be re-tied.
            from ThoughtReferences import bindings
            members = []
            for address, _ in dictionary._witnesses.get(identity, ()):
                row = store.index_of_row(address)
                if row is not None:
                    context = bindings(store.meaning_of(row)).get('_individual_context')
                    if context is not None:
                        members.append((address, row, dict(context), self.coordinates(store.point_of_row(address))))
            signature = torch.zeros_like(dictionary.directions[str(identity)])
            for address, row, context, content in members:
                signature = self.absorb(signature, content.to(signature), self._predicate(context).to(signature))
            dictionary.directions[str(identity)].copy_(signature)

    def get_extra_state(self):
        from IndependentComponents import _pack_state
        return _pack_state(dict(seen=self.seen, cooccurrence=self.cooccurrence,
            associations=self.associations, predicates=self.predicates, total=self.total))

    def set_extra_state(self, state):
        from IndependentComponents import _unpack_state
        values = _unpack_state(state)
        for name in ('seen', 'cooccurrence', 'associations', 'predicates', 'total'):
            setattr(self, name, values[name])

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if not any(key.startswith(prefix) for key in state_dict):
            for key, value in self.state_dict().items():
                state_dict[prefix + key] = value
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
