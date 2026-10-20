"""Individual membership at a kept close, with access-bounded revision.

Operations decide mint versus bind. This owner decides only *which* accessible
individual explains a binding. Its inputs are native concept rows and accessible LTM occurrences.
It has no identity dictionary, event chart or text parser.
Trial plans are read-only. Occurrence points are immutable under revision.
"""
from dataclasses import replace
import hashlib

import torch
from torch import nn
from IdentityRows import meet_ratios


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
        # G0: added defining parts enlarge the native code support. Request
        # coverage therefore ties underspecified re-mentions at initialization;
        # candidate coverage remains a separately trainable content feature.
        self.policy = nn.Parameter(torch.tensor([0., 1., 1., 1., 0., 0., 0., 0.]))
        self._participant_cache = None
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
        from IdentityRows import coordinates
        return coordinates(self.components.owner, point)

    def predicate_coordinates(self, point):
        from IdentityRows import coordinates
        # A predicate may carry stance. It is an exact cache key, never a
        # region in the identity meet or a column of a second scorer.
        return coordinates(self.components.owner, point, region=False)

    @staticmethod
    def absorb(signature, content, predicate=None):
        return torch.maximum(signature, content)

    @staticmethod
    def support(value):
        return {int(i): float(value[i]) for i in (value >= .3).nonzero().flatten()}

    def contradicts(self, left, right):
        """Read retained evidence for exclusion; no learned count table."""
        from ThoughtReferences import bindings
        a, b = set(self.support(left)), set(self.support(right))
        if not a or not b or a <= b or b <= a:
            return False
        observed = []
        for row in range(len(self.store)):
            if '_individual_context' in bindings(self.store.meaning_of(row)):
                point = self.store.point_of_row(int(self.store.row_ids[row]))
                if point is not None:
                    observed.append((set(self.support(self.coordinates(point))),
                                     int(self.store.witness_count[row])))
        return any(sum(n for members,n in observed if x in members) >= 8
                   and sum(n for members,n in observed if y in members) >= 8
                   and not any(x in members and y in members for members,_ in observed)
                   for x in a-b for y in b-a)

    @staticmethod
    def _predicate_key(point):
        return hashlib.sha256(point.detach().float().cpu().contiguous().numpy().tobytes()).digest()

    def _participants(self):
        """Rebuildable address cache over the retained identity columns."""
        from ThoughtReferences import bindings
        store = self.store
        if store is None:
            return {}
        version = (id(store),len(store),store.refs._version,store.slots._version,
                   store.semantic_fingerprint._version,store.witness_count._version)
        if self._participant_cache is None or self._participant_cache[0] != version:
            rows = {}
            for row in range(len(store)):
                context = bindings(store.meaning_of(row)).get('_individual_context')
                if context is None:
                    continue
                context = dict(context)
                predicate = self._predicate(context)
                if bool(predicate.any()) and int(store.refs[row,0]) not in (-1,0):
                    rows.setdefault(self._predicate_key(predicate),[]).append((row,context['role']))
            self._participant_cache = version, {key:tuple(value) for key,value in rows.items()}
        return self._participant_cache[1]

    def prediction(self, identity, predicate, role):
        if isinstance(identity, tuple) and identity[:1] == ('new',):
            return predicate.sum()*0
        rows = [(row,r) for row,r in self._participants().get(self._predicate_key(predicate),())
                if r == role]
        total = sum(int(self.store.witness_count[row]) for row,_ in rows)
        if not total:
            return predicate.sum()*0
        target = identity[-1] if isinstance(identity,tuple) else identity
        count = sum(int(self.store.witness_count[row]) for row,_ in rows
                    if int(self.store.refs[row,0]) == target)
        return predicate.new_tensor(count/total)

    def expected_roles(self, predicate):
        rows = self._participants().get(self._predicate_key(self.predicate_coordinates(predicate)),())
        return tuple(role for role in (0,2) if sum(int(self.store.witness_count[row])
            for row,r in rows if r == role) >= self.components.verbs.recurrence)

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
        if value.numel() != signature.numel():
            raise ValueError('expectation prime must use native concept coordinates')
        return (signature*value).sum()/signature.sum().clamp_min(1e-12)

    def accessible(self, frames):
        """Read only supplied recent/cued row addresses and their NP rows."""
        from ThoughtReferences import bindings, open_slots
        store, records = self.store, {}
        if store is None:
            return records
        for frame in frames:
            row = store.index_of_row(frame.row_id)
            if row is None:
                continue
            for reference in self.occurrences(row):
                address = self.address(reference)
                index = None if address is None else store.index_of_row(address)
                if index is None or int(store.order[index]) != 1:
                    continue
                meaning = store.meaning_of(index)
                context = bindings(meaning).get('_individual_context')
                point = store.point_of_row(address)
                if point is None or (('referent',0) in open_slots(meaning)
                                     and not bool(point.any())):
                    continue
                if context is None:
                    context = dict(role=0,position=(int(store.sentence_index[index]),index),
                                   predicate=None,mint=False,initial=None)
                records[int(address)] = index,dict(context)
        return records

    def occurrences(self, row):
        from ThoughtReferences import bindings
        meaning = self.store.meaning_of(row)
        if meaning is None:
            return ()
        data = bindings(meaning)
        owned = data.get('_individual_occurrences', ())
        return ((*owned, self.store.occurrence_of(row))
                if '_individual_context' in data or int(self.store.order[row]) == 1 else owned)

    def logits(self, features, cue=None, *, supervised=False):
        """One factored score; known antecedents train content and form."""
        policy = self.policy.clone()
        if supervised:
            train = torch.tensor([True,True,False,False,True,False,False,False],device=policy.device)
            policy = torch.where(train,policy,policy.detach())
        return features @ policy

    def cue(self, value):
        return self.coordinates(value)

    def individuals(self, records):
        from IdentityRows import point
        result = {}
        for address,(row,context) in records.items():
            identity = int(self.store.refs[row,0])
            if identity in self.components.individual_ids:
                content = point(self.components.owner,identity)
            else:
                identity = self.store.occurrence_of(row)
                content = self.store.point_of_row(address)
            position = context['position']
            result[identity] = dict(content=self.coordinates(content),value=content.detach(),role=context['role'],
                position=position if isinstance(position,tuple) else (position,-1))
        return dict(tuple(result.items())[-self.components.source_capacity:])

    def score(self, content, predicate, cue, role, individuals, *, stream=None):
        """Meet content and form, native expectation, then recency and roles."""
        from IdentityRows import form_region
        region = form_region(self.components.owner,cue)
        choices, features = [], []
        for identity,candidate in individuals.items():
            value = candidate['content'].to(content)
            candidate_coverage,request_coverage = meet_ratios(content,value)
            form = (content.sum()*0 if region is None else meet_ratios(region.to(value),value)[0])
            recency = sum(other['position'] > candidate['position'] for other in individuals.values())
            features.append(torch.stack((candidate_coverage,request_coverage,
                self.prediction(identity,predicate,role),self.prime(value,stream),form,
                content.new_tensor(-recency/max(1,len(individuals))),
                content.new_tensor(candidate['role']/2),content.new_tensor(role/2))))
            choices.append(identity)
        phi = torch.stack(features).to(self.policy) if features else self.policy.new_empty((0,8))
        return tuple(choices),phi,self.logits(phi)

    @staticmethod
    def choice_cost(features, logits, selected):
        credit = 6*features[:,1]+features[:,2]+features[:,3]
        target = credit.detach().softmax(0)
        return ((1-features[selected,1]).clamp_min(0)-features[selected,2]
                -(target*logits.log_softmax(0)).sum())

    def acyclic_candidates(self, individuals, occurrences):
        """An occurrence cannot be filled by itself or a containing row.

        Read only the store's typed role edges for this structural check.
        Binding/scoping metadata are retention roots, not constituent edges;
        native concept identities cannot create an occurrence cycle.
        """
        store = self.store
        targets = {ref for ref in occurrences if ref in store._index_occurrences}
        if not targets:
            return individuals
        eligible = {}
        for identity, candidate in individuals.items():
            pending, seen = [identity], set()
            while pending:
                reference = pending.pop()
                if reference in targets:
                    break
                if reference in seen or reference not in store._index_occurrences:
                    continue
                seen.add(reference)
                context = store._semantic_rows.get(reference[2], {})
                pending.extend(ref for _, ref in store._role_occurrence_references(
                    context.get('role_refs', ())))
            else:
                eligible[identity] = candidate
        return eligible

    def bind_site(self, meaning, role, frames, *, stream=None):
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
        cue = self.cue(None if site.get('cue') is None else meaning.roles.new_tensor(site['cue']))
        records = self.accessible(frames)
        query_occurrence = bindings(meaning).get('_query_occurrence')
        binding_occurrence = site.get('occurrence', query_occurrence)
        individuals = self.acyclic_candidates(self.individuals(records),
                                             (binding_occurrence, query_occurrence))
        choices, features, logits = self.score(content, predicate, cue,
            site.get('role', role), individuals, stream=stream)
        base = dict(binding_role=role, reverse_leaves=tuple(site.get('leaves', ())),
                    binding_candidates=choices, witnesses=(),
                    binding_occurrence=binding_occurrence)
        if not choices:
            return dict(base, meaning=meaning, value=meaning, support_true=0., support_false=0.,
                        reference_cost=1., incomplete=('no_accessible_individual',))
        index = int(logits.detach().argmax())
        identity = choices[index]
        probability = float(logits.softmax(0)[index].detach())
        point = individuals[identity]['value'].detach().to(meaning.roles)
        witnesses = tuple(self.store.occurrence_of(row) for row, _ in records.values()
                          if (self.store.occurrence_of(row) == identity
                              or int(self.store.refs[row, 0]) == identity))
        evidence = dict(base, reference=identity if isinstance(identity,tuple) else ('sym', identity), binding_value=point,
            witnesses=witnesses,
            support_true=probability, support_false=1-probability,
            reference_cost=float(self.choice_cost(features, logits, index).detach()),
            reference_training=dict(features=features.detach().cpu().tolist(),
                cue=cue.detach().cpu().tolist(), selected=index),
            incomplete=())
        result = fill(meaning, evidence, witnesses=witnesses,
                      operation='bind', slots=(('referent', role),))
        from ThoughtReferences import with_slots
        result = with_slots(result, open_slots(result), pair=(probability, 1-probability))
        return dict(evidence, value=result, meaning=result)

    @torch.no_grad()
    def commit_bindings(self, result):
        """Re-tie kept NP occurrences, preserving their observed content."""
        from ThoughtReferences import bindings, fill
        store = self.store
        for record in result.records:
            value = record.result
            if record.kind != 'thought' or value is None or value.semantic_id != 'bind':
                continue
            evidence = value.evidence
            reference = evidence.get('reference')
            row = store._index_occurrences.get(evidence.get('binding_occurrence'))
            if row is None or reference is None:
                continue
            original = store.meaning_of(row)
            if '_individual_context' not in bindings(original):
                continue
            identity = int(reference[-1])
            previous = int(store.refs[row,0])
            updated = fill(original, dict(reference=reference,binding_value=original.roles[0],
                support_true=value.support_true,support_false=value.support_false),
                witnesses=evidence.get('witnesses',()),operation='bind',slots=(('referent',0),))
            store.upsert_resolved(row,updated)
            store.refs[row,0] = identity
            store._update_semantic_fingerprint(row)
            from IdentityRows import refold
            for changed in {previous,identity}.intersection(self.components.individual_ids):
                refold(self.components.owner,changed)

    def train_thought(self, result, errors):
        """Only the kept episode trains the resolver's existing evidence cost."""
        losses = []
        for record in result.records:
            value = record.result
            if record.kind != 'thought' or value is None or value.semantic_id != 'bind':
                continue
            held = value.evidence.get('reference_training')
            if held is None:
                continue
            features = self.policy.new_tensor(held['features'])
            cue = self.policy.new_tensor(held['cue'])
            losses.append(self.choice_cost(features, self.logits(features, cue), held['selected']))
        if losses:
            errors.add('expectation.reference', torch.stack(losses).sum(),
                       category='expectation', weight=self.components.weight)

    def mention_loss(self, record, correct):
        """Negative log marginal mass of all known correct antecedents.

        No selected argmax becomes a label. If the known answer was outside
        the retained menu there is no supervised update to that menu.
        """
        candidates = record['candidates']
        gold = torch.tensor([identity in correct for identity in candidates],
                            device=self.policy.device, dtype=torch.bool)
        if not bool(gold.any()):
            return None
        features = torch.as_tensor(record['features'], device=self.policy.device,
                                   dtype=self.policy.dtype).detach()
        cue = torch.as_tensor(record['cue'], device=self.policy.device,
                             dtype=self.policy.dtype).detach()
        logits = self.logits(features, cue, supervised=True)
        return logits.logsumexp(0) - logits[gold].logsumexp(0)

    def answer_loss(self, plan, meaning, word):
        """A supplied answer labels only the reference roles it answers."""
        from ThoughtReferences import bindings
        data = bindings(meaning)
        roles = {role for kind, role in (*data.get('_open_references', ()),
                 *data.get('_bound_roles', ())) if kind == 'referent'}
        correct = set(self.components.owner.word_concepts(word))
        losses = []
        for item in plan['items']:
            if item['role'] not in roles or item['features'] is None:
                continue
            item['correct'] = correct.intersection(item['candidates'])
            losses.append(self.mention_loss(item, correct))
        losses = [value for value in losses if value is not None]
        return torch.stack(losses).sum() if losses else None

    def resolved(self, row, role, reference, errors):
        """A later fill trains the recorded menu for this exact occurrence."""
        from ThoughtReferences import bindings
        records = bindings(self.store.meaning_of(row)).get('_reference_mentions', ())
        correct = {reference}
        target = self.store.index_of_row(reference)
        if target is not None:
            correct.add(int(self.store.refs[target, 0]))
        losses = []
        with torch.enable_grad():
            for held_role, record in records:
                if held_role == role:
                    loss = self.mention_loss(dict(record), correct)
                    if loss is not None:
                        losses.append(loss)
                        from IdentityRows import expose_form, point
                        value = point(self.components.owner, reference)
                        if value is not None:
                            expose_form(self.components.owner,
                                self.policy.new_tensor(dict(record)['cue']), value)
            if losses:
                errors.add('expectation.reference_mentions', torch.stack(losses).sum(),
                           category='expectation', weight=self.components.weight)

    def preview(self, clause, frames, *, document=None, position=None, stream=None):
        """One decision per constituent; no allocation, writes or statistics."""
        if position is not None and frames:
            from ReferenceContext import exclude_replaced_occurrences
            ids = torch.tensor([frame.row_id for frame in frames], device=self.policy.device)
            allowed = exclude_replaced_occurrences(self.store, ids, torch.ones_like(ids, dtype=torch.bool),
                                                  document, position)
            frames = tuple(frame for frame, keep in zip(frames, allowed.tolist()) if keep)
        records = self.accessible(frames)
        individuals = self.individuals(records)
        items, cost = [], self.policy.new_zeros(())
        for node in constituents(clause):
            request = node.reference_request
            mode, role = request['mode'], request.get('role', 0)
            content = self.coordinates(node.point)
            predicate = self.predicate_coordinates(request.get('predicate'))
            cue = self.cue(request.get('cue'))
            choices, phi, logits = self.score(content, predicate, cue, role, individuals, stream=stream)
            selected, probabilities = None, None
            if mode == 'mint':
                # A trial-local handle is not a durable identity. It allows
                # later constituents of this same close to bind the new one.
                selected = ('new', len(items))
                signature = self.absorb(torch.zeros_like(content), content, predicate)
                latest = max((other['position'] for other in individuals.values()), default=(-1, -1))
                individuals[selected] = dict(content=signature, role=role, position=(latest[0]+1, len(items)))
                # A redundant mint has a cost; contradictory content does
                # not. This credits the operation chooser without assigning
                # a determiner word to either operation.
                if choices:
                    cost = cost + phi[:, 1].amax().clamp(0, 1)
            elif choices:
                # Both trial graphs precede either optimizer step. Own the
                # parameter values used by this graph, as the shared chooser
                # does, while retaining their gradient connection.
                probabilities = logits.softmax(0)
                index = int(logits.detach().argmax())
                selected = choices[index]
                # The evidence cost trains the operation chooser; the
                # posterior cross entropy trains learned salience and cues.
                cost = cost + self.choice_cost(phi, logits, index)
            else:
                cost = cost + 1
            items.append(dict(node=node, mode=mode, content=content, predicate=predicate,
                role=role, selected=selected, candidates=tuple(choices),
                features=phi if choices else None, cue=cue,
                probabilities=probabilities, leaves=request['leaves']))
        return dict(items=items, cost=cost, records=records)

    @torch.no_grad()
    def _mint(self, signature):
        from IdentityRows import mint
        return mint(self.components.owner,signature)

    @torch.no_grad()
    def prepare(self, clause, plan):
        """Prepare kept occurrence rows; allocation follows their successful write."""
        if not plan['items']:
            plan['prepared_root'] = clause
            return clause
        replacements = {}
        for index, item in enumerate(plan['items']):
            reference = item['selected']
            if isinstance(reference, tuple) and reference[:1] == ('new',):
                reference = None
            item['reference'] = reference
            node = item['node']
            from ThoughtReferences import bindings, with_slots
            request = node.reference_request
            site = dict(role=item['role'], leaves=item['leaves'],
                cue=None if request.get('cue') is None else request['cue'].detach().cpu().tolist(),
                content=None if request.get('content') is None else request['content'].detach().cpu().tolist(),
                predicate=None if request.get('predicate') is None else request['predicate'].detach().cpu().tolist())
            meaning = replace(node.meaning, bindings=dict(bindings(node.meaning),
                _reference_sites=((0, site),)))
            if item['selected'] is None:
                meaning = with_slots(meaning, (('referent', 0),), pair=node.evidence)
                if request.get('forward_identity') is not None:
                    meaning = replace(meaning, bindings=dict(bindings(meaning),
                        _forward_references=((0, request['forward_identity']),)))
            predicate = node.reference_request.get('predicate')
            children = node.children
            if predicate is not None:
                from ClauseRow import Clause
                from Meaning import ConceptualMeaning
                predicate_node = Clause(ConceptualMeaning.from_description(predicate),
                                        point=predicate, order=0)
                item['predicate_node'] = predicate_node
                children = (*children, predicate_node)
            replacements[id(node)] = replace(node, meaning=meaning, children=children,
                refs=(-1 if reference is None else self.address(reference), -1, -1))
        rebuilt = {}
        def rewrite(node):
            if id(node) in rebuilt:
                return rebuilt[id(node)]
            value = replacements.get(id(node), node)
            result = replace(value, children=tuple(rewrite(c) for c in value.children),
                             companions=tuple(rewrite(c) for c in value.companions))
            rebuilt[id(node)] = result
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
        return self.predicate_coordinates(None if address is None else self.store.point_of_row(address))

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
        if not plan['items']:
            self.last_audit = ()
            return
        store = self.store
        from Occurrence import document_digest
        document = document_digest(document).hex()
        addresses, audit, allocated, mentions = [], [], {}, []
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
            elif isinstance(item['selected'], tuple) and item['selected'][:1] == ('new',):
                reference = allocated.get(item['selected'])
            item['reference'] = reference
            store.refs[index, 0] = (-1 if reference is None else
                                   reference[-1] if isinstance(reference, tuple) else reference)
            # Context is semantic membership evidence, not a derivation. The
            # NP's actual content remains exclusively in its occurrence row.
            predicate_row = written.get(id(item.get('predicate_node')))
            context = dict(role=item['role'], position=(int(position), max(item['leaves'], default=-1)), document=document,
                predicate=None if predicate_row is None else store.occurrence_of(predicate_row), mint=item['mode'] == 'mint',
                initial=reference)
            self._metadata(index, '_individual_context', context)
            from ThoughtReferences import bindings
            site = dict(dict(bindings(store.meaning_of(index))['_reference_sites'])[0])
            site['occurrence'] = store.occurrence_of(index)
            self._metadata(index, '_reference_sites', ((0, site),))
            if index != row and reference is None:
                sites = dict(bindings(store.meaning_of(row)).get('_reference_sites', ()))
                sites[item['role']] = site
                self._metadata(row, '_reference_sites', tuple(sites.items()))
            if item['features'] is not None:
                # Retain detached sufficient inputs on the occurrence, not a
                # live graph or a separate pending-reference cache.
                record = dict(candidates=tuple(allocated.get(identity, identity)
                    for identity in item['candidates']),
                    features=item['features'].detach().cpu().tolist(),
                    cue=item['cue'].detach().cpu().tolist())
                mentions.append((item['role'], record))
                self._metadata(index, '_reference_mentions', ((0, record),))
            plan['records'][address] = index, context
            addresses.append(store.occurrence_of(index))
            if reference in self.components.individual_ids:
                from IdentityRows import absorb
                absorb(self.components.owner,reference,item['content'])
            if training:
                from IdentityRows import expose_form, point
                # A supplied label teaches exposure only at the kept write.
                # Otherwise the selected occurrence supplies its own evidence.
                for target in item.get('correct', (reference,)):
                    value = (point(self.components.owner,target) if isinstance(target,int)
                             else store.point_of_row(self.address(target))
                             if target is not None else None)
                    if value is not None:
                        expose_form(self.components.owner,item['cue'],value)
            audit.append(dict(leaves=item['leaves'], occurrence=address, reference=reference,
                operation=item['mode'], candidates=item['candidates'],
                missing_reason=(None if reference is not None else
                    'individual_capacity_exhausted' if item['mode'] == 'mint' or isinstance(item['selected'], tuple)
                    else 'no_accessible_individual')))
        self._metadata(row, '_individual_occurrences', tuple(addresses))
        if mentions:
            self._metadata(row, '_reference_mentions', tuple(mentions))
        self.revise(plan['records'])
        self.last_audit = tuple(audit)

    @torch.no_grad()
    def revise(self, records):
        """Re-tie accessible occurrences when later content contradicts a group.

        A split is an identity revision, not an implicit mint by a binder.
        Earlier ambiguous occurrences follow the predicate evidence of the
        resulting groups. No coordinates of a stored occurrence are edited.
        """
        store = self.store
        changed = set()
        current = [(address, row, context, self.coordinates(store.point_of_row(address)))
            for address, (row, context) in records.items()
            if int(store.refs[row, 0]) in self.components.individual_ids]
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
                scores = [meet_ratios(content, anchor[3])[1] +
                    int(self._predicate_key(predicate) == self._predicate_key(self._predicate(anchor[2])))
                    for anchor in anchors]
                selected = references[int(torch.stack(scores).argmax())]
                previous = int(store.refs[row, 0])
                if previous != selected:
                    changed.update((previous, selected))
                store.refs[row, 0] = selected
                store._update_semantic_fingerprint(row)
        # Merge only on new shared, distinguishing content with predicate
        # agreement. Equal initial kind content alone cannot merge two mints.
        groups = {}
        for record in current:
            groups.setdefault(int(store.refs[record[1], 0]), []).append(record)
        ids = list(groups)
        for offset, left in enumerate(ids):
            if left not in groups:
                continue
            for right in ids[offset+1:]:
                if right not in groups:
                    continue
                a, b = groups[left], groups[right]
                initial = [r for r in (*a, *b) if r[2]['mint']]
                initial_support = set().union(*(self.support(r[3]) for r in initial))
                later_a = [r for r in a if set(self.support(r[3])) - initial_support]
                later_b = [r for r in b if set(self.support(r[3])) - initial_support]
                if not later_a or not later_b:
                    continue
                contradiction = any(self.contradicts(x[3], y[3]) for x in a for y in b)
                corroborated = any(float(meet_ratios(x[3], y[3])[1]) >= .9 and
                    self._predicate_key(self._predicate(x[2])) == self._predicate_key(self._predicate(y[2]))
                    for x in later_a for y in later_b)
                if corroborated and not contradiction:
                    changed.update((left, right))
                    for _, row, _, _ in b:
                        store.refs[row, 0] = left
                        store._update_semantic_fingerprint(row)
                    groups[left].extend(groups.pop(right))
        # Re-fold all retained members for each identity touched by revision.
        # A forgetting event does not call this path or subtract lost content.
        from IdentityRows import refold
        for identity in changed:
            refold(self.components.owner,identity)

    def get_extra_state(self):
        from IndependentComponents import _pack_state
        return _pack_state(dict(version=3))

    def set_extra_state(self, state):
        from IndependentComponents import _unpack_state
        if _unpack_state(state).get('version') != 3:
            raise ValueError('reference checkpoint requires the native-row migration')
        self._participant_cache = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if not any(key.startswith(prefix) for key in state_dict):
            for key, value in self.state_dict().items():
                state_dict[prefix + key] = value
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
