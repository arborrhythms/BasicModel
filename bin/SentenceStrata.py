"""The pinned sentence: occurrence content in STM0, ordered pairs in STM1.

Reading and Composition expose views of these stores. Numerical inverse
witnesses are allowed to outlive a forward operation; they carry no topology.
"""
from dataclasses import replace
import weakref

import torch
from torch import nn

from Memory import STM0, OCCURRENCE, RELATION
from Meaning import ConceptualMeaning
from PartsOfSpeech import operation_name


class Tree:
    def __init__(self, bank, language, roots, length):
        self.bank, self.language = bank, language
        self.generation, self.roots, self.length = bank.generation, tuple(roots), length
        self._views = {}

    @property
    def store(self):
        if self.generation != self.bank.generation:
            raise ValueError('the pinned sentence reference has expired')
        return self.bank

    def metadata(self, address):
        return dict(self.store._semantic_rows[address]['bindings'])

    def operator(self, ref):
        first, second = self.store.parts.references(ref[1])
        if first[0] == RELATION:
            operator, left = self.store.parts.references(first[1])
            return operator[1], (left, second)
        return first[1], (second,)

    def interface(self, data):
        seat = self.language.parts_of_speech.seats[data['_identity'][1]]
        matches = [(kind, index) for kind, rules in (
            (1, self.language._compose_binary_rules), (2, self.language._compose_unary_rules))
            for index, rule in enumerate(rules)
            if operation_name(rule) == seat.name and rule.canonical == data['_interface']]
        if len(matches) != 1:
            raise ValueError('the operator seat has no unique declared execution interface')
        kind, index = matches[0]
        return kind, index

    def node(self, ref):
        if ref[0] == OCCURRENCE:
            data = self.metadata(ref[1])
            return (0, data['_word'])
        operator, children = self.operator(ref)
        data = self.metadata(operator)
        kind, index = self.interface(data)
        return kind, index, data['_at'], *(self.node(child) for child in children)

    def forest(self):
        return tuple(self.node(ref) for ref in self.roots)

    def actions(self):
        # Event order belongs to occurrences. Pair references determine which
        # operands were consumed, including non-top unary steps and swaps.
        entries = {}
        for address, context in self.store._semantic_rows.items():
            data = dict(context['bindings'])
            if '_at' in data:
                entries[data['_at']] = (address, data)
        nodes = {}
        def collect(node):
            if node[0]:
                nodes[node[2]] = node
                for child in node[3:]: collect(child)
        for node in self.forest(): collect(node)
        actions, stack = [], []
        for index in range(self.length):
            if index not in entries:
                raise ValueError('a stored operation has no concept occurrence')
            address, data = entries[index]
            if '_word' in data:
                node = (0, data['_word']); stack.append(node)
                actions.append((0, -1, data['_word'])); continue
            if index not in nodes:
                kind, operation = self.interface(data)
                rule = self.language._compose_binary_rules[operation]
                if kind != 1 or len(rule.lhs.split(',')) != 2 or len(stack) < 2:
                    raise ValueError('an unconsumed operator must be a declared transposition')
                stack[-2:] = reversed(stack[-2:])
                actions.append((3, operation, -1)); continue
            node = nodes[index]
            kind, operation = node[:2]
            if kind == 2:
                position = stack.index(node[3]); word = len(stack)-1-position
                stack[position] = node
            elif len(node) == 4:
                if stack[-1] != node[3]: raise ValueError('invalid stored bound operand')
                stack[-1] = node; word = -1
            else:
                if tuple(stack[-2:]) != node[3:]: raise ValueError('invalid ordered binary operands')
                stack[-2:] = [node]; word = -1
            actions.append((kind, operation, word))
        return self.store.address_keys.new_tensor(actions).reshape(-1, 3)

    def leaves(self):
        leaves = sorted((dict(context['bindings'])['_word'], address)
            for address, context in self.store._semantic_rows.items()
            if '_word' in dict(context['bindings']))
        return torch.stack([self.store.point_of_row(address) for _, address in leaves])

    def view(self, address):
        if address not in self._views:
            self._views[address] = CompositionView(self, address)
        return self._views[address]

    def composition(self, action=None, word=None):
        for address, context in self.store._semantic_rows.items():
            data = dict(context['bindings'])
            if (action is not None and data.get('_at') == action or
                    word is not None and data.get('_word') == word):
                return self.view(address)
        raise ValueError('composition has no shared occurrence')

    def install_composition(self, composition, *, action=None, word=None):
        view = self.composition(action, word)
        context = self.store._semantic_rows[view.address]
        data = dict(context['bindings'])
        data['_composition'] = {name: getattr(composition, name) for name in (
            'reference', 'operation', 'head_role', 'order', 'leaves', 'binding',
            'word_id', 'formal_name', 'interpretation')}
        context['bindings'] = tuple(data.items())
        return view


class CompositionView:
    """Metadata and child views of an occurrence, never a second tree."""
    def __init__(self, tree, address):
        self.tree, self.address = tree, address

    @property
    def point(self):
        return self.tree.store.point_of_row(self.address)

    @property
    def children(self):
        data = self.tree.metadata(self.address)
        if '_word' in data: return ()
        pending = list(self.tree.roots)
        while pending:
            ref = pending.pop()
            if ref[0] != RELATION: continue
            operator, children = self.tree.operator(ref)
            if operator == self.address:
                return tuple(self.tree.view(child[1] if child[0] == OCCURRENCE else
                    self.tree.operator(child)[0]) for child in children)
            pending.extend(children)
        raise ValueError('operator occurrence has no application pair')

    def __getattr__(self, name):
        data = self.tree.metadata(self.address).get('_composition', {})
        if name not in data: raise AttributeError(name)
        return data[name]


def replace_composition(value, **changes):
    if not isinstance(value, CompositionView):
        return replace(value, **changes)
    children = changes.pop('children', value.children)
    if tuple(child.address for child in children) != tuple(child.address for child in value.children):
        raise ValueError('a composition view cannot rewrite the pinned topology')
    data = value.tree.metadata(value.address)
    data['_composition'] = dict(data['_composition'], **changes)
    value.tree.store._semantic_rows[value.address]['bindings'] = tuple(data.items())
    return value


class SentenceStrata(nn.Module):
    def __init__(self, streams, words, recent, width, percept_width=0, relation_capacity=8192, *, paths=2):
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
        self.banks = nn.ModuleList(STM0(width, self.words+self.operators, percept_width=percept_width,
            relation_capacity=2*self.operators+1) for _ in range(self.trials*self.streams))
        self.recent_stores = nn.ModuleList(STM0(width, self.recent, percept_width=percept_width,
            relation_capacity=relation_capacity) for _ in range(self.streams))
        self._leases = [None] * len(self.banks)
        self._clock = 0

    def capture(self, program, language, stream):
        indices = range(self.trials*stream, self.trials*(stream+1))
        index = next((i for i in indices if self._leases[i] is None or self._leases[i]() is None), None)
        if index is None:
            raise OverflowError('all constructed pinned sentence trials are still retained')
        store = self.banks[index]
        store.expire()
        # Device follows the value, without ever changing a construction size.
        store.to(device=program.leaves.device, dtype=program.leaves.dtype)
        actions = program.actions.tolist()
        operators = sum(kind in (1, 2, 3) for kind, _, _ in actions)
        if len(program.leaves) > self.words or operators > self.operators:
            raise OverflowError(f'pinned sentence exceeds construction: {len(program.leaves)}/{self.words} words, {operators}/{self.operators} operations')
        stack = []
        for at, (kind, operation, word) in enumerate(actions):
            self._clock += 1
            data = {'_at': at}
            if kind == 0:
                point = program.leaves[word]
                data.update(_word=word)
                concept = int(program.concept_ids[word])
            else:
                rules = language._compose_binary_rules if kind in (1, 3) else language._compose_unary_rules
                rule = rules[operation]
                concept = language.parts_of_speech.seats[operation_name(rule)].concept
                point = program.operation_values[at, 2]
                data['_interface'] = rule.canonical
            if concept not in (-1, 0): data['_identity'] = ('sym', concept)
            meaning = replace(ConceptualMeaning.from_description(point), bindings=data)
            row = store.append_meaning(meaning, kind='unverified', time=False,
                document_key=('pinned', stream, self._clock), sentence_index=at)
            ref = OCCURRENCE, int(store.address_keys[row])
            if kind == 0: stack.append(ref); continue
            if kind == 3:
                left, right = stack[-2:]
                first = store.parts.add(ref, left, truth_plus=1.)
                store.parts.add((RELATION, first), right, truth_plus=1.)
                stack[-2:] = right, left
                continue
            if kind == 2 or rule.method_name == 'bind':
                position = len(stack)-1-(max(0, word) if kind == 2 else 0)
                pair = store.parts.add(ref, stack[position], truth_plus=1.)
                stack[position] = RELATION, pair
            else:
                right, left = stack.pop(), stack.pop()
                first = store.parts.add(ref, left, truth_plus=1.)
                stack.append((RELATION, store.parts.add((RELATION, first), right, truth_plus=1.)))
        tree = Tree(store, language, stack, len(actions))
        self._leases[index] = weakref.ref(tree)
        object.__setattr__(program, 'tree', tree)
        object.__setattr__(program, 'actions', None)
        object.__setattr__(program, 'leaves', None)
        return program

    def keep_close(self, stream, point, when, *, address=None, relations=(), source=None):
        store = self.recent_stores[stream]
        store.to(device=point.device, dtype=point.dtype)
        previous = None if address is None else store.index_of_row(address)
        if previous is not None or len(store) == store.capacity:
            store.slots = store.slots.detach().clone()
            retired = 0 if previous is None else previous
            store.compact_rows([index for index in range(len(store)) if index != retired])
        self._clock += 1
        meaning = replace(ConceptualMeaning.from_description(point.detach()), bindings={'_relations': tuple(relations)})
        row = store.append_meaning(meaning,
            kind='unverified', time=when, address=address,
            document_key=('recent', stream, self._clock))
        if source is not None:
            selected, pending = set(), [relation for context in store._semantic_rows.values()
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
            bank._occupation_owner = source._occupation_owner
            bank.operation_catalog = source.operation_catalog
            source._sort()
            for relation in source.relation_order[:len(source)].tolist():
                if relation not in selected: continue
                left, right = source.references(relation)
                bank.add(left, right, symbol=relation, truth_plus=float(source.truth_plus[relation]),
                    truth_minus=float(source.truth_minus[relation]), operator=int(source.operator_reference[relation]),
                    operation_seat=int(source.operation_seat[relation]))
                bank.sentence_reference[relation] = source.sentence_reference[relation]
        for index in range(self.trials*stream, self.trials*(stream+1)):
            self.banks[index].expire(); self._leases[index] = None
        return store.occurrence_of(row)

    def clear(self, stream=None):
        for b in range(self.streams) if stream is None else (stream,):
            self.recent_stores[b].expire()
            for index in range(self.trials*b, self.trials*(b+1)):
                self.banks[index].expire(); self._leases[index] = None

    def read(self, address):
        for store in (*self.recent_stores, *self.banks):
            if store.location(address) is not None:
                return store.point_of_row(address[-1])
        raise ValueError('short-term member reference has expired')
