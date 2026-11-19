"""The sentence trace supplies the word tree. Semantic ideas occupy STM0 and STM1."""
from dataclasses import replace
import weakref

import torch
from torch import nn

from Memory import STM0, OCCURRENCE, RELATION
from Meaning import ConceptualMeaning
from ReadingIndex import ReadingIndex


class IdeaStack(STM0):
    """A trial's native semantic entries, beside its immutable word tree."""
    def prepare(self, source, tree, sites):
        self.__dict__.pop('_concept_owner', None)
        self.expire()
        self.address_domain = source.address_domain
        self._external_memory = weakref.ref(source)
        self._concept_owner = source.__dict__.get('_concept_owner')
        self.parts.handle_reserved = source.parts.contains
        self._tree_sites = sites
        self._tree_root = tree.roots[0]
        self.model_clock.fill_(tree.semantic_start)

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

    def _keep_working(self, row, meaning):
        # An unkept path must not deposit pending bindings in the live MTM.
        reference = self.referent_of(row)
        self.working[self.row_name(row)] = replace(meaning,
            role_refs=(reference, *meaning.role_refs[1:]),
            sentence_kind=meaning.sentence_kind or 'idea').metadata()

    def expire(self):
        # Expiring a private bank cannot withdraw same-numbered live tokens.
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
        if not bool(self.leaves.isfinite().all()):
            raise FloatingPointError('non-finite occupied meaning role')
        operators = index.actions[:, 0] != 0
        if any(kind != 0 for kind, _, _ in index.steps):
            if self.frames is None:
                raise ValueError('recorded operations require their result points')
            if not bool(self.frames[operators, 2].isfinite().all()):
                raise FloatingPointError('non-finite occupied meaning role')
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
        self._views = {}
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

    def view(self, address):
        if address not in self._views:
            self._views[address] = CompositionView(self, address)
        return self._views[address]

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
            percept_width=percept_width, relation_capacity=relation_capacity,
            episodic_concepts=episodic_concepts)
            for _ in range(self.trials*self.streams))
        self.recent_stores = nn.ModuleList(STM0(width, self.recent, percept_width=percept_width,
            relation_capacity=relation_capacity) for _ in range(self.streams))
        self._leases = [None] * len(self.idea_stores)
        self._traces = [None] * len(self.idea_stores)
        self._clock = 0

    def capture(self, program, language, stream, *, after=0):
        self._clock = max(self._clock, int(after))
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
        record.validate_interfaces(language)
        trace = Trace(program, record, self._clock + 1)
        tree = Tree(trace, ideas)
        self._clock += len(trace.index.steps) + ideas.capacity
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
                # occurrence name from the same clock as the retained rows;
                # the syntax counter can otherwise alias a later row's time.
                from RowTime import next_time, encoder
                address = int(encoder().decode_index(next_time(owner)))
        store.to(device=point.device, dtype=point.dtype)
        previous = None if address is None else store.index_of_row(address)
        if previous is not None or len(store) == store.capacity:
            store.slots = store.slots.detach().clone()
            retired = 0 if previous is None else previous
            store.compact_rows([index for index in range(len(store)) if index != retired])
        self._clock += 1
        meaning = replace(ConceptualMeaning.from_description(point.detach()), bindings={'_relations': tuple(relations)})
        row = store.append_meaning(meaning, name=self._clock if address is None else address)
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
        for index in range(self.trials*stream, self.trials*(stream+1)):
            self._expire_trial(index)
        return store.occurrence_of(row)

    def clear(self, stream=None):
        for b in range(self.streams) if stream is None else (stream,):
            self.recent_stores[b].expire()
            for index in range(self.trials*b, self.trials*(b+1)):
                self._expire_trial(index)

    def read(self, address):
        for store in (*self.recent_stores, *self.idea_stores):
            if store.location(address) is not None:
                return store.point_of_row(address)
        point = self.trace_point(address)
        if point is not None:
            return point
        raise ValueError('short-term member reference has expired')
