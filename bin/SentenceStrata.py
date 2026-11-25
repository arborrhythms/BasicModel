"""The sentence trace supplies the word tree. Semantic ideas occupy STM0 and STM1."""
from dataclasses import replace
import weakref

import torch
from torch import nn

from Memory import STM0, OCCURRENCE, SYMBOL, RELATION
from Meaning import ConceptualMeaning, require_sentence_tensor
from ReadingIndex import ReadingIndex


class IdeaStack(STM0):
    """A trial's native semantic entries, beside its immutable word tree."""
    def __init__(self, *args, action_capacity=0, word_capacity=0, **kwargs):
        super().__init__(*args, **kwargs)
        # A semantic binding can explicitly name the idea's referent. A
        # lexical row may have the same referent without that annotation.
        self.register_buffer('_identity_bound', self.row_names.new_zeros(self.capacity, dtype=torch.bool),
                             persistent=False)
        self.register_buffer('_operation_rows', self.slots.new_empty(action_capacity, 3, self.nDim),
                             persistent=False)

        self.register_buffer('_word_rows', self.slots.new_empty(word_capacity, self.nDim),
                             persistent=False)

    def word_rows(self, like):
        """Own the interpreted values written by shifts of this candidate."""
        rows = self._word_rows
        if len(like) > len(rows) or like.shape[1:] != rows.shape[1:]:
            raise OverflowError('candidate words exceed the reserved STM word rows')
        if rows.device != like.device or rows.dtype != like.dtype:
            self._word_rows = rows = like.new_empty(rows.shape)
        # Each shift writes its addressed word before comprehension reads it.
        return rows[:len(like)].detach()

    def operation_rows(self, like):
        """Start the numerical journal for this stream's leased candidate."""
        rows = self._operation_rows
        if len(like) > len(rows) or like.shape[1:] != rows.shape[1:]:
            raise OverflowError('candidate actions exceed the reserved STM operation rows')
        if rows.device != like.device or rows.dtype != like.dtype:
            self._operation_rows = rows = like.new_empty(rows.shape)
        return rows[:len(like)].detach().zero_()

    def append_closed_idea(self, point, *, identity, evidence, entry, surprise, name):
        if point.shape != (self.nDim,):
            raise ValueError('idea width differs from its store')
        require_sentence_tensor(point.isfinite().all(), 'non-finite occupied meaning role', FloatingPointError)
        reference = self._reference_value(identity)
        self._reserve_idea_name(name)
        first = len(self)
        try:
            row = self._append_point(point, evidence, reference, entry, surprise, None)
        finally:
            self.__dict__.pop('_next_row_name', None)
        self._identity_bound[first:len(self)] = identity is not None
        return row

    def raw_meaning_of(self, index):
        if isinstance(index, tuple):
            return self.part_row(index[1])['meaning']
        if not 0 <= int(index) < len(self):
            raise IndexError('unavailable idea')
        # Only compatibility readers request the owned grammatical view.
        tag, value, bound = torch.cat((self.referent[index], self._identity_bound[index, None])).tolist()
        reference = self._decode_referent(tag, value)
        metadata = self.working.get(self.row_name(index))
        if metadata is None:
            metadata = dict(role_refs=(reference, None, None), sentence_kind='idea',
                            bindings={'_identity': reference} if bound else ())
        meaning = ConceptualMeaning.from_description(self.slots[index].detach(), **metadata)
        owner = self.owner()
        return meaning if owner is None else owner.mtm.pending_meaning(self.occurrence_of(index), meaning)

    def compact_rows(self, keep):
        indices = torch.as_tensor(keep, device=self._identity_bound.device, dtype=torch.long).reshape(-1)
        super().compact_rows(indices)
        kept = self._identity_bound.index_select(0, indices.sort().values) if len(indices) else None
        self._identity_bound.zero_()
        if kept is not None:
            self._identity_bound[:len(kept)].copy_(kept)

    def prepare(self, source, tree, sites):
        self.__dict__.pop('_concept_owner', None)
        self.expire()
        self.address_domain = source.address_domain
        self._external_memory = weakref.ref(source)
        self._concept_owner = source.__dict__.get('_concept_owner')
        self.parts.handle_reserved = source.parts.contains
        self._tree_sites = sites
        self._tree_root = tree.roots[0]
        self.names_issued.fill_(tree.semantic_start)
        if not self.names_issued.is_inference():
            self._prepared_word_names = self.names_issued, self.names_issued._version, tree.semantic_start

    def name_for(self, term, *, sentence, count):
        address = self._tree_sites.get(id(term))
        if (address is None or self.index_of_row(address) is not None
                or not sentence and address <= self._tree_root < address+count
                or any(self.index_of_row(address+i) is not None for i in range(count))):
            return None
        external = self._external_memory()
        if external.index_of_row(address) is not None:
            raise ValueError('the sentence index aliases a retained idea')
        return address

    def _reference_value(self, reference):
        if (isinstance(reference, tuple) and reference[:2] == ('ltm', self.address_domain)
                and self.index_of_row(reference[2]) is None):
            external = self._external_memory()
            if external.location(reference) is not None:
                return OCCURRENCE, reference[2]
        return super()._reference_value(reference)

    def _keep_working(self, row, meaning, reference):
        # An unkept path must not deposit pending bindings in the live MTM.
        reference = self._decode_referent(*reference)
        self.working[self.row_name(row)] = replace(meaning,
            role_refs=(reference, *meaning.role_refs[1:]),
            sentence_kind=meaning.sentence_kind or 'idea').metadata()

    def append_words(self, points, concepts):
        """Write located lexical ideas in shift order, preserving the live graph."""
        count = len(concepts)
        start, end = len(self), len(self) + count
        if end > self.capacity:
            raise OverflowError('STM0 construction capacity exhausted')
        if points.shape != (count, self.nDim):
            raise ValueError('word ideas must have the width of their STM store')
        require_sentence_tensor(points.isfinite().all(), 'non-finite occupied meaning role', FloatingPointError)
        # Basic word ideas have only their native referent. raw_meaning_of
        # reads it directly from this column, as it does for copied LTM0 ideas.
        if torch.is_tensor(concepts):
            named = concepts > 0
            refs = torch.stack((torch.where(named, SYMBOL, 0), torch.where(named, concepts, -1)), -1)
        else:
            refs = self.referent.new_tensor([
                self._reference_value(None if cid is None else ('sym', cid)) for cid in concepts
            ]).reshape(count, 2)
        prepared = self.__dict__.pop('_prepared_word_names', None)
        first_name = (prepared[2] if prepared is not None and prepared[0] is self.names_issued
                      and prepared[1] == self.names_issued._version else int(self.names_issued))
        issued = range(first_name, first_name + count)
        seats = torch.arange(start, end, device=self.slots.device)
        with torch.enable_grad():
            self.slots = self.slots.index_copy(0, seats, points.to(self.slots))
        self.referent[start:end] = refs
        self.trust[start:end] = 0.
        self.activity[start:end] = self.activity_config.default
        column, _, names, index = self._occupied_names()
        column[start:end] = first_name + seats - start
        if len(names) == start and not column.is_inference():
            names.extend(issued)
            for row, name in enumerate(issued, start):
                index.setdefault(name, row)
            self._name_cache = column, column._version, names, index
        self.names_issued.fill_(first_name + count)
        self._write_count(end)
        from MemoryActivity import written_complete
        written_complete(self, range(start, end))
        return issued

    def expire(self):
        # Expiring a private bank cannot withdraw same-numbered live tokens.
        self.__dict__.pop('_prepared_word_names', None)
        owner = self.__dict__.pop('_concept_owner', None)
        try:
            super().expire()
        finally:
            if owner is not None:
                self._concept_owner = owner


class Trace:
    """A path owns the recorded points and the close's occurrence marks."""
    def __init__(self, program, index, first):
        self.first = first
        self.index = index
        self.leaves = program.leaves
        self.frames = getattr(program, 'operation_values', None)
        finite = self.leaves.isfinite().all()
        if any(kind != 0 for kind, _, _ in index.steps):
            if self.frames is None:
                raise ValueError('recorded operations require their result points')
            if self.frames.shape[0] != index.actions.shape[0]:
                raise ValueError('recorded result points must align with their actions')
            operators = index.actions[:, 0] != 0
            finite = finite & (~operators | self.frames[:, 2].isfinite().all(-1)).all()
        require_sentence_tensor(finite, 'non-finite occupied meaning role', FloatingPointError)
        self.head = None
        self.marks = {}
        self.live = True

    def point(self, action):
        kind, _, word = self.index.steps[action]
        return self.leaves[word] if kind == 0 else self.frames[action, 2]


class Tree:
    address_domain = STM0.address_domain

    def __init__(self, trace, ideas):
        self._trace = trace
        self.ideas = ideas
        # A view holds its tree; the tree does not hold its views, so the
        # tree is released, and its trial with it, when no reader holds it.
        self._views = weakref.WeakValueDictionary()
        self.semantic_start = trace.first + len(trace.index.steps)

    @property
    def trace(self):
        if not self._trace.live:
            raise ValueError('the sentence reference has expired')
        return self._trace

    @property
    def roots(self):
        trace = self.trace
        return tuple(trace.first + action for action in trace.index.roots)

    def forest(self):
        return self.trace.index.forest

    def point(self, address):
        return self.trace.point(address - self.trace.first)

    def write_edges(self, store, concepts, where, when, language):
        """Write the located leaves and reduce ordered operands to a single typed head."""
        trace = self.trace
        index = trace.index
        heads = {}
        start = store.when_start
        shifts = sorted(index.word_actions.items(), key=lambda item: item[1])
        words = torch.tensor([word for word, _ in shifts], device=trace.leaves.device, dtype=torch.long)
        points = trace.leaves.index_select(0, words).to(store.slots)
        if points.shape[-1] != store.nDim:
            points = torch.nn.functional.pad(points, (0, store.nDim - points.shape[-1]))
        if start is not None and start >= 4:
            for band, offset in ((where, start - 4), (when, start)):
                if band is not None:
                    points[:, offset:offset + 4] = band.detach().index_select(0, words.to(band.device)).to(points)
        shifted = (concepts.index_select(0, words) if torch.is_tensor(concepts)
                   else [concepts[word] for word, _ in shifts])
        names = store.append_words(points, shifted)
        for (_, at), name in zip(shifts, names, strict=True):
            heads[at] = OCCURRENCE, name
        from PartsOfSpeech import operation_name
        # An operand references an earlier pair or an actual idea name. Publish
        # all pairs to this trial's STM1 together, in the original write order.
        pairs = []
        def join(left, right):
            pairs.append((left, right))
            return RELATION, len(pairs) - 1

        def sequence(operands):
            head = None
            for operand in reversed(operands):
                head = operand if head is None else join(operand, head)
            return head

        for at, (kind, local, _) in enumerate(index.steps):
            if kind in (1, 2):
                rules = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
                operator = language.parts_of_speech.seats[operation_name(rules[local])].concept
                operands = sequence([heads[child] for child in index.operands[at]])
                heads[at] = join((SYMBOL, operator), operands)
        head = sequence([heads[root] for root in index.roots])
        handles = store.parts.write_tree_pairs(pairs)
        trace.head = (RELATION, handles[head[1]]) if head is not None and head[0] == RELATION else head
        return (concepts <= 0).sum() if torch.is_tensor(concepts) else sum(concept is None for concept in concepts)

    def view(self, address):
        view = self._views.get(address)
        if view is None:
            view = self._views[address] = CompositionView(self, address)
        return view

    def composition(self, action=None, word=None):
        trace = self.trace
        if action is None:
            action = trace.index.word_actions.get(word)
        if action is None or not 0 <= action < len(trace.index.steps):
            raise ValueError('composition has no shared occurrence')
        return self.view(trace.first + action)

    def install_composition(self, composition, *, action=None, word=None):
        view = self.composition(action, word)
        marks = view.marks
        prior = marks.get('extension', ())
        marks.update((name, getattr(composition, name)) for name in (
            'reference', 'operation', 'head_role', 'order', 'leaves', 'binding',
            'word_id', 'formal_name', 'interpretation', 'extension'))
        if not marks['extension']:
            marks['extension'] = prior
        return view

    def install_extension(self, node, *, action=None, word=None):
        self.composition(action, word).marks['extension'] = (node,)


class CompositionView:
    def __init__(self, tree, address):
        self.tree, self.address = tree, address

    @property
    def marks(self):
        return self.tree.trace.marks.setdefault(self.address, {})

    @property
    def point(self):
        return self.tree.point(self.address)

    @property
    def children(self):
        trace = self.tree.trace
        at = self.address - trace.first
        if trace.index.steps[at][0] == 3:
            raise ValueError('a transposition is not an application')
        return tuple(self.tree.view(trace.first + child) for child in trace.index.operands[at])

    def __getattr__(self, name):
        try:
            return self.marks[name]
        except KeyError:
            raise AttributeError(name) from None


def replace_composition(value, **changes):
    if not isinstance(value, CompositionView):
        return replace(value, **changes)
    children = changes.pop('children', value.children)
    if tuple(child.address for child in children) != tuple(child.address for child in value.children):
        raise ValueError('a composition view cannot rewrite the recorded topology')
    value.marks.update(changes)
    return value


class SentenceStrata(nn.Module):
    def __init__(self, streams, words, recent, width, percept_width=0, relation_capacity=8192, *, paths=2,
                 episodic_concepts=0):
        super().__init__()
        self.words, self.streams, self.recent = int(words), int(streams), int(recent)
        if self.words < 1: raise ValueError('sentence strata require a positive word bound')
        # A sentence has three operation rounds per word and one closing
        # drain of two STM capacities. Unary steps and transpositions do not
        # consume a word, so n-1 is not an operation bound.
        self.operators = 3*self.words + 2*self.recent
        # Every configured path and the admitted view coexist until the close.
        self.trials = int(paths) + 1
        if self.trials < 2:
            raise ValueError('sentence strata require an exploit and admission bank')
        # Each action can keep an idea on both lanes, a separate event on
        # both lanes, and an untrusted located mention. These are the native
        # close's five possible rows, not copies of numerical inner nodes.
        self.idea_stores = nn.ModuleList(IdeaStack(width,5*(self.words+self.operators),
            action_capacity=self.words+self.operators, word_capacity=self.words,
            percept_width=percept_width, relation_capacity=relation_capacity,
            episodic_concepts=episodic_concepts)
            for _ in range(self.trials*self.streams))
        self.recent_stores = nn.ModuleList(STM0(width, self.recent, percept_width=percept_width,
            relation_capacity=relation_capacity) for _ in range(self.streams))
        self._leases = [None] * len(self.idea_stores)
        self._traces = [None] * len(self.idea_stores)
        self._names = 0

    def capture(self, program, language, stream, *, after=0, kept_tree=None):
        self._names = max(self._names, int(after))
        indices = range(self.trials*stream, self.trials*(stream+1))
        index = next((i for i in indices if self._leases[i] is None or self._leases[i]() is None), None)
        if index is None:
            raise OverflowError('all constructed sentence trials are still retained')
        if len(program.leaves) > self.words or len(program.actions) - len(program.leaves) > self.operators:
            raise OverflowError(f'sentence exceeds construction: {len(program.leaves)}/{self.words} words, '
                                f'{len(program.actions)-len(program.leaves)}/{self.operators} operations')
        self._expire_trial(index)
        ideas = self.idea_stores[index]
        ideas.address_domain = STM0.address_domain
        record = getattr(program, 'action_index', None)
        if record is None:
            record = ReadingIndex(program.actions, len(program.leaves), language)
            object.__setattr__(program, 'action_index', record)
        # The winning tree has already resolved these operator interfaces.
        # Admission can change a referent while preserving its argument site.
        if kept_tree is None or record is not kept_tree.trace.index:
            record.validate_interfaces(language)
        trace = Trace(program, record, self._names + 1)
        tree = Tree(trace, ideas)
        self._names += len(trace.index.steps) + ideas.capacity
        self._traces[index] = trace
        self._leases[index] = weakref.ref(tree)
        object.__setattr__(program, 'actions', trace.index.actions)
        object.__setattr__(program, 'tree', tree)
        return program

    def _expire_trial(self, index):
        if self._traces[index] is not None:
            self._traces[index].live = False
        self._traces[index] = self._leases[index] = None
        self.idea_stores[index].expire()

    def trace_point(self, address):
        if isinstance(address, tuple):
            if len(address) != 3 or address[:2] != ('ltm', STM0.address_domain):
                return None
            name = address[2]
        else:
            from Occurrence import operand_time
            name = operand_time(address)
            if name is None:
                return None
        for trace in self._traces:
            if trace is not None and 0 <= name - trace.first < len(trace.index.steps):
                return trace.point(name - trace.first)
        return None

    def keep_close(self, stream, point, when, *, address=None, relations=(), source=None):
        store = self.recent_stores[stream]
        owner_ref = None if source is None else source.__dict__.get('_occurrence_store')
        owner = None if owner_ref is None else owner_ref()
        if owner is not None:
            store.address_domain = owner.address_domain
            if address is None:
                # A relation's recent head has no LTM0 row. Reserve its
                # occurrence name from the retained rows' own count of names;
                # the syntax counter can otherwise alias a later row's name.
                from RowTime import name_row
                address = name_row(owner)
        store.to(device=point.device, dtype=point.dtype)
        previous = None if address is None else store.index_of_row(address)
        if previous is not None or len(store) == store.capacity:
            store.slots = store.slots.detach().clone()
            retired = 0 if previous is None else previous
            store.compact_rows([index for index in range(len(store)) if index != retired])
        self._names += 1
        meaning = ConceptualMeaning.from_description(point.detach(), bindings={'_relations': tuple(relations)})
        row = store.append_meaning(meaning, name=self._names if address is None else address)
        if source is not None:
            selected, pending = set(), [relation for context in store.working.values()
                for relation in dict(context['bindings']).get('_relations', ())]
            while pending:
                relation = pending.pop()
                if relation in selected: continue
                if not source.contains(relation):
                    raise ValueError('recent idea references an unavailable semantic relation')
                selected.add(relation)
                pending.extend(ref for kind, ref in source.references(relation) if kind == RELATION)
            bank = store.parts
            bank.expire()
            bank._concept_owner = source._concept_owner
            source._sort()
            for relation in source.relation_order[:len(source)].tolist():
                if relation not in selected: continue
                left, right = source.references(relation)
                bank.add(left, right, symbol=relation, truth_plus=float(source.truth_plus[relation]),
                    truth_minus=float(source.truth_minus[relation]))
        self.release_trials(stream)
        return store.occurrence_of(row)

    def release_trials(self, stream):
        for index in range(self.trials*stream, self.trials*(stream+1)):
            self._expire_trial(index)

    def clear(self, stream=None):
        for b in range(self.streams) if stream is None else (stream,):
            self.recent_stores[b].expire()
            self.release_trials(b)

    def read(self, address):
        for store in (*self.recent_stores, *self.idea_stores):
            if store.location(address) is not None:
                return store.point_of_row(address)
        point = self.trace_point(address)
        if point is not None:
            return point
        raise ValueError('short-term member reference has expired')
