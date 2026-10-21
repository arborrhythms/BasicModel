"""Individual membership at a kept close, with access-bounded revision.

Operations decide mint versus bind. This owner decides only *which* accessible
individual explains a binding. Its inputs are native concept rows and accessible LTM occurrences.
The owner's MTM retains token membership between kept closes.
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
        self.last_audit = ()

    @property
    def store(self):
        return self.components.owner._closed_clause_store()

    def address(self, reference):
        if isinstance(reference, tuple):
            if len(reference) == 2 and reference[0] == 'sym':
                return int(reference[1])
            if len(reference) != 3 or reference[:2] != ('ltm', self.store.address_domain):
                return None
            return int(reference[2])
        return reference

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
        """Derived participant cache over the one Part store, never count tables."""
        from Spaces import _concept_alloc_of
        owner = self.components.owner
        alloc = _concept_alloc_of(owner)
        parts = alloc.parts
        version = (parts.revision, owner.similarity_codebook.W._version)
        cache = getattr(parts, '_participant_cache', None)
        if cache is None or cache[0] != version:
            rows = {}
            def add(predicate, edge, role, identity):
                from IdentityRows import point
                value = point(owner, predicate)
                if value is not None:
                    rows.setdefault(self._predicate_key(self.predicate_coordinates(value)), []).append(
                        (edge, role, identity))
            for edge in parts.indices().tolist():
                part, whole = parts.refs[edge].tolist()
                if 'state' in alloc.symbol_types.get(edge, ()):
                    add(whole, edge, 2, part)
                    add(edge, edge, 2, part)
                if parts.contains(whole) and 'state' in alloc.symbol_types.get(whole, ()):
                    predicate = int(parts.refs[whole, 1])
                    add(predicate, edge, 0, part)
                    add(whole, edge, 0, part)
                elif 'situation' in alloc.symbol_types.get(edge, ()):
                    add(whole, edge, 0, part)
            cache = version, {key: tuple(dict.fromkeys(value)) for key, value in rows.items()}
            parts._participant_cache = cache
        return cache[1]

    def _derivation_symbol(self, reference):
        """Read a token's retained kind/fold without making it an individual."""
        if reference is None:
            return None
        if isinstance(reference, int) and (self.store is None or self.store.index_of_row(reference) is None):
            return reference
        if isinstance(reference, tuple) and reference[:1] == ('sym',):
            return reference[1]
        location = self.store.location(reference) if self.store is not None else None
        if isinstance(location, int):
            value = self.store.raw_meaning_of(location).role_refs[0]
            if value is not None and value[0] == 'sym':
                return value[1]
            mtm = self.components.owner.mtm
            token = mtm.token_of(self.store.occurrence_of(location))
            if token is not None:
                return mtm.entries[token].kind
        return None

    def token_value(self, reference):
        """Compute an anonymous introduction's content from its retained uses."""
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
        if entry.kind is not None and self.store.raw_meaning_of(location).role_refs[0] is None:
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
        target = self._derivation_symbol(identity)
        if target is None:
            return predicate.sum()*0
        rows = [(edge, target) for edge, r, target in
                self._participants().get(self._predicate_key(predicate), ()) if r == role]
        from Spaces import _concept_alloc_of
        parts = _concept_alloc_of(self.components.owner).parts
        total = sum(int(parts.access_count[edge]) for edge, _ in rows)
        count = sum(int(parts.access_count[edge]) for edge, candidate in rows if candidate == target)
        return predicate.new_tensor(count/total) if total else predicate.sum()*0

    def expected_roles(self, predicate):
        rows = self._participants().get(self._predicate_key(self.predicate_coordinates(predicate)), ())
        from Spaces import _concept_alloc_of
        parts = _concept_alloc_of(self.components.owner).parts
        return tuple(role for role in (0, 2) if sum(int(parts.access_count[edge])
            for edge, r, _ in rows if r == role) >= self.components.verbs.recurrence)

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

    def accessible(self, frames):
        """Query the recent addresses and the Part edges at their activation times."""
        from Spaces import _concept_alloc_of
        store, records = self.store, {}
        if store is None:
            return records
        alloc = _concept_alloc_of(self.components.owner)
        parts = alloc.parts
        active = parts.indices()
        mtm = self.components.owner.mtm
        for position, frame in enumerate(frames):
            for identity in frame.symbol_ids:
                if identity in alloc.individuals and identity not in alloc.retired:
                    records[identity] = ('sym', identity), dict(role=0,
                        position=(position, 0), identity=identity)
            location = store.location(frame.row_id)
            if location is None:
                continue
            row = store.row(location)
            if isinstance(location, int) and int(row['order']) == 1:
                occurrence = store.occurrence_of(location)
                records[row['row_id']] = location, dict(role=0, position=(position, 0),
                    identity=mtm.token_of(occurrence) or occurrence)
            at_time = active[(parts.when[active] == row['when'].to(parts.when)).all(-1)]
            # A timestamp is not a sentence identity. Traverse this frame's
            # retained derivation before consulting any contemporaneous edge.
            roots = ([location[1]] if isinstance(location, tuple) else
                     [ref[1] for ref in row['meaning'].role_refs
                      if ref is not None and ref[0] == 'sym' and parts.contains(ref[1])])
            if roots:
                pending, reachable = list(roots), set()
                while pending:
                    edge = pending.pop()
                    if edge in reachable or not parts.contains(edge):
                        continue
                    reachable.add(edge)
                    pending.extend(parts.refs[edge].tolist())
                at_time = active.new_tensor(sorted(reachable))
            edges = list(at_time.tolist())
            if isinstance(location, tuple):
                edges.append(location[1])
            seen = set()
            while edges:
                edge = edges.pop()
                if edge in seen or not parts.contains(edge):
                    continue
                seen.add(edge)
                for role, identity in zip((0, 2), parts.refs[edge].tolist()):
                    if identity in alloc.individuals and identity not in alloc.retired:
                        records[identity] = ('sym', identity), dict(role=role,
                            position=(position, len(seen)), identity=identity)
                    if parts.contains(identity):
                        edges.append(identity)
            for occurrence, identity, role in frame.activations:
                index = store.location(occurrence)
                if not isinstance(index, int):
                    continue
                argument = self._derivation_symbol(identity if identity is not None else occurrence)
                predicate = None
                for edge in at_time.tolist():
                    part, whole = parts.refs[edge].tolist()
                    if role == 2 and 'state' in alloc.symbol_types.get(edge, ()):
                        if part == argument:
                            predicate = whole
                            break
                    if role == 0 and part == argument:
                        predicate = int(parts.refs[whole, 1]) if parts.contains(whole) and (
                            'state' in alloc.symbol_types.get(whole, ())) else whole
                        break
                records.pop(identity, None)
                records[int(occurrence[-1])] = index, dict(role=role, position=(position, index),
                    identity=(mtm.token_of(occurrence) or occurrence) if identity is None else identity,
                    predicate=predicate)
        return dict(tuple(records.items())[-self.components.source_capacity:])

    def occurrences(self, row):
        frame = __import__('ReferenceContext').SituationFrame.from_row(self.store.row(row))
        return tuple(('sym', context['identity']) if isinstance(context['identity'], int)
                     else context['identity'] or self.store.occurrence_of(location)
                     for location, context in self.accessible((frame,)).values())

    def logits(self, features, cue=None, *, supervised=False):
        """One factored score; known antecedents train content and form."""
        policy = self.policy.clone()
        if supervised:
            train = torch.tensor([True,True,False,False,True,False,False,False],device=policy.device)
            policy = torch.where(train,policy,policy.detach())
        return features @ policy

    def cue(self, value):
        # A form identifies the symbol whose learned conceptual region is
        # read. Its perceptual coordinates never enter a content comparison.
        from IdentityRows import carrier
        return carrier(self.components.owner, value)

    def individuals(self, records):
        """Compute the current identities from their codes and accessible uses.

        The result owns working vectors only. Neither a bank row, an occurrence
        nor the activation/address index is changed by this interpretation.
        """
        from IdentityRows import point
        def baseline(identity):
            if isinstance(identity, int):
                return point(self.components.owner, identity)
            if identity in self.components.owner.mtm.entries:
                index = self.store.location(identity)
                if isinstance(index, int) and self.store.raw_meaning_of(index).role_refs[0] is None:
                    return self.token_value(identity)
                return self.store.point_of_row(identity[-1])
            return None
        groups = {}
        for address, (location, original) in records.items():
            context = dict(original)
            identity = context.get('identity')
            if identity is None:
                identity = self.store.occurrence_of(location)
            native = baseline(identity)
            observed = (self.store.point_of_row(address) if isinstance(location, int) else native)
            if observed is None:
                continue
            context['content'] = self.coordinates(observed)
            context['location'] = location
            context['address'] = address
            context['native'] = native
            groups.setdefault(identity, []).append(context)
        def predicate(member):
            reference = member.get('predicate')
            value = None if reference is None else self.store._concept_point(reference)
            return None if value is None else self._predicate_key(self.predicate_coordinates(value))
        def support(value):
            return set((value >= .3).nonzero().flatten().tolist())
        # Exclusion is a query over retained uses. No learned cooccurrence
        # table or word class supplies it, and no result is written to LTM.
        observed = []
        for row in range(len(self.store)):
            if int(self.store.order[row]) != 1 or int(self.store.role_mask[row].sum()) != 1:
                continue
            value = self.coordinates(self.store.slots[row, 0])
            observed.append((support(value), int(self.store.witness_count[row])))
        def contradicts(a, b):
            a, b = support(a), support(b)
            if not a or not b or a <= b or b <= a:
                return False
            return any(sum(n for members, n in observed if x in members) >= 8
                       and sum(n for members, n in observed if y in members) >= 8
                       and not any(x in members and y in members for members, _ in observed)
                       for x in a-b for y in b-a)
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
                common = any(float(meet_ratios(x['content'], y['content'])[1]) >= .9
                    and predicate(x) == predicate(y) for x in newer_a for y in newer_b)
                if common and not any(contradicts(x['content'], y['content']) for x in a for y in b):
                    groups[left].extend(groups.pop(right))
        result = {}
        for identity, members in groups.items():
            native = baseline(identity)
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
        return result

    def score(self, content, predicate, cue, role, individuals, *, stream=None):
        """Meet content and form, native expectation, then recency and roles."""
        from IdentityRows import form_region
        content = self.coordinates(content)
        region = form_region(self.components.owner,cue)
        choices, features = [], []
        for identity,candidate in individuals.items():
            value = self.coordinates(candidate['content']).to(content)
            candidate_coverage,request_coverage = meet_ratios(content,value)
            form = (content.sum()*0 if region is None else meet_ratios(region.to(value),value)[0])
            recency = sum(other['position'] > candidate['position'] for other in individuals.values())
            features.append(torch.stack((candidate_coverage,request_coverage,
                self.prediction(candidate.get('kind', identity),predicate,role),self.prime(value,stream),form,
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

    def select(self, choices, logits, *, forward_identity=None, individuals=None):
        """The ordinary resolver choice; fixtures may force this operation seam."""
        return int(logits.detach().argmax()) if choices else None

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
        # A shared identity can participate in a self-referential relation.
        # Bind reads its finite content; bounded readers own graph traversal.
        individuals = self.individuals(records)
        choices, features, logits = self.score(content, predicate, cue,
            site.get('role', role), individuals, stream=stream)
        base = dict(binding_role=role, reverse_leaves=tuple(site.get('leaves', ())),
                    binding_candidates=choices, witnesses=(),
                    binding_occurrence=binding_occurrence)
        if not choices:
            return dict(base, meaning=meaning, value=meaning, support_true=0., support_false=0.,
                        reference_cost=1., incomplete=('no_accessible_individual',))
        index = self.select(choices, logits, forward_identity=dict(
            bindings(meaning).get('_forward_references', ())).get(role), individuals=individuals)
        if index is None:
            return dict(base, meaning=meaning, value=meaning, support_true=0., support_false=0.,
                        reference_cost=1., incomplete=('no_accessible_individual',))
        identity = choices[index]
        probability = float(logits.softmax(0)[index].detach())
        point = individuals[identity]['value'].detach().to(meaning.roles)
        witnesses = tuple(('sym', context['identity']) if isinstance(context.get('identity'), int)
                          else self.store.occurrence_of(location)
                          for location, context in records.values()
                          if context.get('identity') == identity or
                          (isinstance(identity, tuple) and location ==
                           self.store._index_occurrences.get(identity)))
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
        mentions = self.__dict__.get('_mentions', {})
        occurrence = self.store.occurrence_of(row)
        records = mentions.get(occurrence, ())
        correct = {reference}
        if isinstance(reference, tuple) and reference[:1] == ('sym',):
            correct.add(reference[1])
        losses = []
        with torch.enable_grad():
            for held_role, record in records:
                if held_role == role:
                    loss = self.mention_loss(dict(record), correct)
                    if loss is not None:
                        losses.append(loss)
                        from IdentityRows import expose_form, point
                        native = reference[1] if isinstance(reference, tuple) and reference[0] == 'sym' else reference
                        value = point(self.components.owner, native) if isinstance(native, int) else None
                        if value is not None:
                            expose_form(self.components.owner,
                                self.policy.new_tensor(dict(record)['cue']), value)
            if losses:
                errors.add('expectation.reference_mentions', torch.stack(losses).sum(),
                           category='expectation', weight=self.components.weight)
                remaining = tuple(record for record in records if record[0] != role)
                if remaining:
                    mentions[occurrence] = remaining
                else:
                    mentions.pop(occurrence, None)

    def sites(self, clause, *, pending_names=()):
        """Keep explicit open formations as one site, owned by their word.

        The journal need not invent a second NP request. Its retained forward
        name supplies the anonymous constituent only at the closing boundary.
        """
        from ClauseRow import Clause, Composition
        from Meaning import ConceptualMeaning
        from ThoughtReferences import bindings, open_slots
        result, seen, nodes = [], set(), []
        def visit(node):
            if node is None or id(node) in seen:
                return
            seen.add(id(node))
            nodes.append(node)
            if node.reference_request is not None and node.order == 1:
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
                    zero = torch.zeros_like(node.meaning.roles[role])
                    def value(key, fallback):
                        return fallback if held.get(key) is None else zero.new_tensor(held[key])
                    request = dict(mode='pronoun', forward_identity=name, role=role,
                        leaves=tuple(held.get('leaves', ())), content=zero,
                        cue=value('cue', node.meaning.roles[role]),
                        predicate=value('predicate', node.meaning.roles[1]))
                    site = Clause(ConceptualMeaning.from_description(zero), point=zero,
                        order=1, evidence=node.evidence, subject_word_id=name,
                        reference_request=request, composition=Composition(zero,
                            word_id=name, leaves=request['leaves'], order=1))
                    result.append(dict(node=site, parent=node, site_role=role))
            for child in (*node.children, *node.companions):
                visit(child)
        visit(clause)
        # A lexical concept cache does not replace a familiar word's token
        # choice. Re-mentions of accessible pending names still call bind at
        # the closing boundary, before their derivation is admitted.
        covered = {site['node'].reference_request.get('forward_identity') for site in result}
        names, terms = set(pending_names) - covered, set()
        def lexical(term):
            if term is None or id(term) in terms:
                return
            terms.add(id(term))
            if not term.children and term.word_id in names:
                zero = torch.zeros_like(term.point)
                request = dict(mode='pronoun', forward_identity=term.word_id,
                    role=0, leaves=term.leaves, cue=term.point, content=zero, predicate=zero)
                node = Clause(ConceptualMeaning.from_description(zero), point=zero, order=1,
                    evidence=clause.evidence, subject_word_id=term.word_id,
                    composition=term, reference_request=request)
                result.append(dict(node=node, parent=None, site_role=None, lexical=True))
            for child in term.children:
                lexical(child)
        for node in nodes:
            lexical(node.composition)
        return result

    def pending_names(self, clause, individuals, stream):
        """Names whose chosen token is in this close's accessible frames."""
        from ThoughtReferences import bindings
        names, _ = self._arrival_constituents(clause)
        result, store, mtm = set(), self.store, self.components.owner.mtm
        for address in store._forward_addresses(names) if store is not None and names else ():
            index = store.location(address)
            if not isinstance(index, int) or (stream is not None and bool(store.stream[index].any())
                                             and not bool(store.stream[index, stream])):
                continue
            value = store.raw_meaning_of(index)
            for role, name in bindings(value).get('_forward_references', ()):
                reference = value.role_refs[role]
                occurrence = (reference if reference is not None and reference[0] == 'ltm'
                              else store.occurrence_of(index))
                if name in names and mtm.token_of(occurrence) in individuals:
                    result.add(name)
        return tuple(sorted(result))

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
        pending_names = self.pending_names(clause, individuals, stream)
        items, cost = [], self.policy.new_zeros(())
        for site in self.sites(clause, pending_names=pending_names):
            node = site['node']
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
                signature = content.clone()
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
                index = self.select(choices, logits,
                    forward_identity=request.get('forward_identity'), individuals=individuals)
                selected = None if index is None else choices[index]
                # The evidence cost trains the operation chooser; the
                # posterior cross entropy trains learned salience and cues.
                cost = cost + (1 if index is None else self.choice_cost(phi, logits, index))
            else:
                cost = cost + 1
            anonymous = mode != 'mint' and selected is None and request.get('forward_identity') is not None
            if anonymous:
                selected = ('new', len(items))
                latest = max((other['position'] for other in individuals.values()), default=(-1, -1))
                individuals[selected] = dict(content=content*0, value=content*0,
                    role=role, position=(latest[0]+1, len(items)))
            items.append(dict(**site, mode=mode, content=content, predicate=predicate,
                role=role, selected=selected, candidates=tuple(choices),
                features=phi if choices else None, cue=cue,
                probabilities=probabilities, leaves=request['leaves'], anonymous=anonymous))
        plan = dict(items=items, cost=cost, records=records, pending_names=pending_names)
        self._arrival_plan(clause, plan, individuals, stream=stream)
        return plan

    def _arrival_constituents(self, clause):
        """The kept derivation's own fields and lexical constituents, locally."""
        from ClauseRow import Clause, Composition
        from Meaning import ConceptualMeaning
        from ThoughtReferences import bindings
        nodes, names, seen = [], set(), set()
        def visit(node):
            if node is None or id(node) in seen:
                return
            seen.add(id(node)); nodes.append(node)
            names.update(name for _, name in bindings(node.meaning).get('_forward_references', ()))
            request = node.reference_request or {}
            if request.get('forward_identity') is not None:
                names.add(request['forward_identity'])
            for child in (*node.children, *node.companions):
                visit(child)
        visit(clause)
        candidates = [node for node in nodes if node.point is not None and node.reference_request is None]
        represented = {id(node.composition) for node in nodes}
        terms = set()
        def visit_term(term):
            if term is None or id(term) in terms:
                return
            terms.add(id(term))
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
            for child in term.children:
                visit_term(child)
        for node in nodes:
            visit_term(node.composition)
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
            if stream is not None and bool(store.stream[location].any()) and not bool(store.stream[location, stream]):
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
                if token is None or token not in accessible_tokens or occurrence in sites:
                    continue
                request = dict(requests.get(role, ()))
                sites[occurrence] = (token, dict(forward_identity=name, role=request.get('role', role),
                    content=value.roles[role] if request.get('content') is None else value.roles.new_tensor(request['content']),
                    predicate=value.roles[1] if request.get('predicate') is None else value.roles.new_tensor(request['predicate']),
                    cue=None if request.get('cue') is None else value.roles.new_tensor(request['cue'])))
        for number, item in enumerate(plan['items']):
            request = item['node'].reference_request
            if request.get('forward_identity') is None:
                continue
            selected = item['selected']
            token = selected if isinstance(selected, tuple) and selected[:1] == ('new',) else mtm.token_of(selected)
            if token is not None:
                sites[('site', number)] = token, request
        plan['arrivals'], plan['arrival_count'] = [], len(candidates)
        if not sites:
            return
        menu = dict(individuals)
        latest = max((other['position'] for other in menu.values()), default=(-1, -1))
        for number, node in enumerate(candidates):
            kind = None if node.composition is None else self._derivation_symbol(node.composition.reference)
            menu[('arrival', number)] = dict(content=self.coordinates(node.point), value=node.point,
                role=0, position=(latest[0]+1, number), kind=kind,
                word_id=node.subject_word_id)
        for site, (token, request) in sites.items():
            content = request.get('content')
            if content is None:
                content = self.policy.new_zeros(self.store.slots.shape[-1])
            predicate = self.predicate_coordinates(request.get('predicate'))
            cue = self.cue(request.get('cue'))
            choices, phi, logits = self.score(content, predicate, cue, request.get('role', 0),
                                              menu, stream=stream)
            index = self.select(choices, logits, forward_identity=request['forward_identity'], individuals=menu)
            selected = None if index is None else choices[index]
            occurrence = None
            if isinstance(selected, int):
                for address in reversed(menu[selected].get('occurrences', ())):
                    location = store.location(address)
                    if isinstance(location, int):
                        occurrence = store.occurrence_of(location)
                        break
            plan['cost'] = plan['cost'] + (1 if index is None else self.choice_cost(phi, logits, index))
            plan['arrivals'].append(dict(site=site, token=token, selected=selected, candidates=choices,
                features=phi, cue=cue, role=request.get('role', 0),
                forward_identity=request['forward_identity'],
                candidate_occurrence=occurrence,
                probabilities=None if not choices else logits.softmax(0)))

    @torch.no_grad()
    def prepare(self, clause, plan):
        """Attach decisions to the temporary derivation, before any admission."""
        from ClauseRow import Clause, Composition
        from Meaning import ConceptualMeaning
        from ThoughtReferences import bindings, with_slots
        if not plan['items'] and not plan.get('arrivals'):
            plan['prepared_root'] = clause
            return clause
        replacements, terms = {}, {}
        minted, anonymous = {}, set()
        attached = {}
        for item in plan['items']:
            node, selected = item['node'], item['selected']
            reference = None if isinstance(selected, tuple) and selected[:1] == ('new',) else selected
            item['reference'] = reference
            request = node.reference_request
            token = self.components.owner.mtm.token_of(reference)
            unresolved = (selected in anonymous or (token is not None
                          and self.components.owner.mtm.entries[token].kind is None))
            if unresolved and request.get('forward_identity') is None and node.subject_word_id >= 0:
                request = dict(request, forward_identity=node.subject_word_id)
            site = dict(role=item['role'], leaves=item['leaves'],
                cue=None if request.get('cue') is None else request['cue'].detach().cpu().tolist(),
                content=None if request.get('content') is None else request['content'].detach().cpu().tolist(),
                predicate=None if request.get('predicate') is None else request['predicate'].detach().cpu().tolist())
            meaning = replace(node.meaning, bindings=dict(bindings(node.meaning), _reference_sites=((0, site),)))
            named = request.get('forward_identity') is not None
            if selected is None or named or unresolved:
                meaning = with_slots(meaning, (('referent', 0),), pair=node.evidence)
                if request.get('forward_identity') is not None:
                    meaning = replace(meaning, bindings=dict(bindings(meaning),
                        _forward_references=((0, request['forward_identity']),)))
            term = node.composition or Composition(node.point, order=node.order)
            native = self.address(reference)
            alias = minted.get(selected) if isinstance(selected, tuple) else None
            updated = replace(term, reference=-1 if native is None else native,
                              order=node.order, binding=alias)
            if item['mode'] == 'mint' or item.get('anonymous'):
                minted[selected] = updated
            if item.get('anonymous'):
                anonymous.add(selected)
            if node.composition is not None:
                terms[id(node.composition)] = updated
            replacements[id(node)] = replace(node, meaning=meaning, composition=updated,
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
                composed[id(term)] = replace(value,
                    children=tuple(rewrite_term(child) for child in value.children),
                    binding=rewrite_term(value.binding))
            return composed[id(term)]
        rebuilt = {}
        def rewrite(node):
            if id(node) not in rebuilt:
                value = replacements.get(id(node), node)
                children, refs = list(value.children), list(value.refs)
                for role, site in attached.get(id(node), ()):
                    refs[role] = ('clause', len(children))
                    children.append(site)
                rebuilt[id(node)] = replace(value,
                    composition=rewrite_term(node.composition) if node.composition is not None else value.composition,
                    refs=tuple(refs), children=tuple(rewrite(c) for c in children),
                    companions=tuple(rewrite(c) for c in value.companions))
            return rebuilt[id(node)]
        result = rewrite(clause)
        _, arrival_nodes = self._arrival_constituents(clause)
        if len(arrival_nodes) != plan.get('arrival_count', len(arrival_nodes)):
            raise RuntimeError('admission changed arrival constituents')
        arrivals = {}
        extra = [rewrite(item['node']) for item in plan['items'] if item.get('lexical')]
        for item in plan.get('arrivals', ()):
            selected = item['selected']
            if isinstance(selected, tuple) and selected[:1] == ('arrival',) and selected not in arrivals:
                node = arrival_nodes[selected[1]]
                arrivals[selected] = rebuilt.get(id(node), node)
                if id(node) not in rebuilt:
                    extra.append(node)
        previous_root = result
        if extra:
            result = replace(result, companions=(*result.companions, *extra))
        plan['arrival_written'] = {key: result if node is previous_root else node
                                   for key, node in arrivals.items()}
        for item in plan['items']:
            node = rebuilt[id(item['node'])]
            item['written_node'] = result if node is previous_root else node
        plan['prepared_root'] = result
        return result

    @torch.no_grad()
    def commit(self, plan, row, written, *, document, position, training):
        """Record training inputs and the kept working decision, never rewrite LTM."""
        from collections import OrderedDict
        from IdentityRows import expose_form, point
        from Spaces import _concept_alloc_of
        alloc = _concept_alloc_of(self.components.owner)
        mtm = self.components.owner.mtm
        mentions = self.__dict__.setdefault('_mentions', OrderedDict())
        allocated, audit, uses = {}, [], []
        for number, item in enumerate(plan['items']):
            node = item['written_node']
            location = written.get(id(node), row if node is plan['prepared_root'] else None)
            if location is None:
                raise RuntimeError('kept reference constituent has no closing address')
            occurrence = self.store.occurrence_of(location)
            reference = item['reference']
            if item.get('anonymous'):
                reference = occurrence
                allocated[('new', number)] = reference
            elif item['mode'] == 'mint':
                symbol = written.get(('symbol', id(node)))
                reference = symbol if symbol in alloc.individuals else occurrence
                allocated[('new', number)] = reference
            elif isinstance(item['selected'], tuple) and item['selected'][:1] == ('new',):
                reference = allocated.get(item['selected'])
            item['reference'] = reference
            if isinstance(reference, tuple) and reference[:1] == ('ltm',):
                from RowTime import encoder, stamp
                kind = self._derivation_symbol(occurrence)
                if kind is None and item['mode'] != 'mint' and not item.get('anonymous'):
                    kind = self._derivation_symbol(reference)
                if item['mode'] == 'mint' or item.get('anonymous'):
                    mtm.open(occurrence, kind=None if item.get('anonymous') else kind,
                             when=int(encoder().decode_index(stamp(self.store))))
                else:
                    uses.append((reference, occurrence, kind))
            if item['features'] is not None:
                held = dict(candidates=tuple(allocated.get(identity, identity) for identity in item['candidates']),
                    features=item['features'].detach().cpu().tolist(), cue=item['cue'].detach().cpu().tolist())
                mentions[occurrence] = ((item['role'], held),)
                mentions.move_to_end(occurrence)
            if training:
                for target in item.get('correct', (reference,)):
                    if isinstance(target, int):
                        value = point(self.components.owner, target)
                    else:
                        index = self.store.location(target) if target is not None else None
                        value = self.store.slots[index, 0] if isinstance(index, int) else None
                    if value is not None:
                        expose_form(self.components.owner, item['cue'], value)
            audit.append(dict(leaves=item['leaves'], occurrence=occurrence[-1], reference=reference,
                operation=item['mode'], candidates=tuple(
                    allocated.get(identity, identity) for identity in item['candidates']),
                missing_reason=None if reference is not None else 'no_accessible_individual'))
        for item in plan.get('arrivals', ()):
            token = allocated.get(item['token'], item['token'])
            selected = item.get('candidate_occurrence') or allocated.get(item['selected'], item['selected'])
            node = plan.get('arrival_written', {}).get(selected)
            if node is not None:
                location = written.get(id(node), row if node is plan['prepared_root'] else None)
                selected = None if location is None else self.store.occurrence_of(location)
            if isinstance(selected, int) and selected in alloc.individuals and mtm.token_of(token) is not None:
                # A native identity needs no synthetic occurrence holding its
                # working vector. Refine through the existing introduction.
                from RowTime import encoder, stamp
                mtm.bind(token, token, kind=selected,
                         when=int(encoder().decode_index(stamp(self.store))))
                selected = token
            if (isinstance(selected, tuple) and selected[:1] == ('ltm',)
                    and selected != token and mtm.token_of(token) is not None):
                from RowTime import encoder, stamp
                mtm.bind(token, selected, kind=self._derivation_symbol(selected),
                         when=int(encoder().decode_index(stamp(self.store))))
            item['reference'] = selected if selected is not None else token
            audit.append(dict(operation='bind', refinement=True, occurrence=token[-1],
                reference=item['reference'], selected=item['selected'], candidates=item['candidates'],
                missing_reason=('no_selected_candidate' if selected is None else
                                None if mtm.token_of(token) is not None else 'no_accessible_individual')))
        # Refinement can reassign a candidate's introducing occurrence. Attach
        # this close's uses afterwards, to the identity that now owns their
        # chosen antecedent; detaching an introduction must not lose new uses.
        for reference, occurrence, kind in uses:
            mtm.bind(reference, occurrence, kind=kind,
                     when=int(encoder().decode_index(stamp(self.store))))
        while len(mentions) > self.components.source_capacity:
            mentions.popitem(last=False)
        self.last_audit = tuple(audit)

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
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
