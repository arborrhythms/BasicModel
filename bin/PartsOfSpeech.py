"""Formal argument concepts learned from kept word members in conceptual space."""
from collections import Counter
import math

import torch
from torch import nn

FIXED_OPERATIONS = frozenset(('interpret', 'sign', 'bind', 'part', 'whole',
    'equal', 'generic', 'implies', 'assertPart'))


def operation_name(rule):
    return getattr(rule, 'surface_name', None) or getattr(rule, 'method_name', None)


def formal_arguments(rules):
    arguments = set()
    for rule in rules:
        name = operation_name(rule)
        if not name:
            continue
        outputs = tuple(part.strip() for part in rule.lhs.split(',') if part.strip())
        for face, symbols in (('argin', rule.rhs_symbols), ('argout', outputs)):
            arguments.update(f'{name}_{face}{i+1}' for i in range(len(symbols)))
    return tuple(sorted(arguments))


def flow_mask(rule):
    """§2b: input-role bits whose words occupy a single result.

    Heads select an operand, not a word within that operand. Symmetric
    results retain every lexical occurrence; clauses retain none.
    """
    name = rule.method_name
    if name in ('intersection', 'union', 'conjunction', 'disjunction', 'product', 'sum'):
        return 3
    if name in ('not', 'non'):
        return 1
    if (name in ('lift', 'equal', 'part', 'whole', 'implies', 'generic', 'ask',
                 'assertPart', 'isEqual', 'isPart')
            or getattr(rule, 'clause_form', None) in ('S', 'implies')):
        return 0
    head = int(getattr(rule, 'head_role', 0) or 0)
    return 1 << (head - 1) if head else 0


class PartsOfSpeech(nn.Module):
    """Declared argument concepts in the native concept hierarchy.

    Kept members teach centroids and confidence. There is no word-by-role
    frequency table, probability simplex, or exclusive category assignment.
    """
    VERSION = 4

    def __init__(self, arguments, space=None):
        super().__init__()
        self.arguments = tuple(arguments)
        if len(set(self.arguments)) != len(self.arguments):
            raise ValueError('formal argument names must be unique')
        self.index = {name: i for i, name in enumerate(self.arguments)}
        self.operations = tuple(dict.fromkeys(name.rsplit('_argin', 1)[0]
            for name in self.arguments if '_argin' in name))
        self.uses = Counter()
        self._space = None
        self.concepts = {}
        self._revision = 0
        self.frozen = False
        self.register_buffer('_value_arguments', torch.empty(0, 0), persistent=False)
        self._value_start = self._value_width = 0
        if space is not None:
            self.bind(space)

    def bind(self, space):
        import weakref
        previous = None if self._space is None else self._space()
        if previous is space:
            return
        if previous is not None:
            from Spaces import _concept_alloc_of
            if _concept_alloc_of(previous) is not _concept_alloc_of(space):
                saved = _concept_alloc_of(space).formal_arguments
                if any(saved.get(name) != cid for name, cid in self.concepts.items()):
                    raise ValueError('argument concepts cannot move between unrelated concept owners')
        self.concepts = space.prebuild_formal_arguments(self.arguments)
        self._space = weakref.ref(space)
        self._revision += 1
        self.prepare_values()

    @torch.no_grad()
    def prepare_values(self):
        """Stage the same committed argument directions used by argument_scores."""
        if self._space is None or self._space() is None or not self.arguments:
            return
        owner = self.owner
        bank = owner.similarity_codebook
        derived = getattr(bank, 'mereology', None)
        self._value_start = 0 if derived is None else derived.percept_event_width
        self._value_arguments = torch.stack([self._direction(self.concepts[name])
            for name in self.arguments]) * self.radii()[:, None]
        self._value_width = self._value_arguments.shape[-1]

    def value_scores(self, values):
        """Candidate coverage in the same normalized meaning coordinates."""
        from IdentityRows import meet_ratios
        if not self._value_width:
            return values.new_zeros(*values.shape[:-1], len(self.arguments))
        direction = torch.nn.functional.normalize(
            values[..., self._value_start:self._value_start+self._value_width], dim=-1)
        return meet_ratios(direction[..., None, :], self._value_arguments.to(values))[0]

    @property
    def owner(self):
        space = None if self._space is None else self._space()
        if space is None:
            raise ValueError('formal argument concepts require their native concept owner')
        return space

    def members(self, argument):
        if argument not in self.index:
            raise ValueError(f'undeclared formal argument: {argument}')
        from ConceptHierarchy import members
        return members(self.owner, self.concepts[argument])

    def order_of(self, word):
        return self.owner._concept_allocator.order_of(int(word))

    def _direction(self, identity):
        from ConceptHierarchy import direction
        return direction(self.owner, identity)

    def radii(self):
        bank = self.owner.similarity_codebook
        rows = [self.owner._csw_row_of(self.concepts[name]) for name in self.arguments]
        return bank.trust.lengths[rows, 0]

    def similarities(self, word):
        if not self.arguments or int(word) < 0 or self.owner._csw_row_of(int(word)) is None:
            return self.owner.similarity_codebook.W.new_zeros(len(self.arguments))
        from ConceptHierarchy import fit
        return torch.stack([fit(self.owner, int(word), self.concepts[name]) for name in self.arguments])

    def argument_scores(self, words, like):
        values = [self.similarities(word).to(like) for word in words.detach().cpu().flatten().tolist()]
        if not values:
            return like.new_zeros(*words.shape, len(self.arguments))
        return torch.stack(values).reshape(*words.shape, len(self.arguments))

    def add(self, word, argument, order=None):
        if self.frozen or int(word) < 0:
            return
        if argument not in self.index:
            raise ValueError(f'undeclared formal argument: {argument}')
        from ConceptHierarchy import observe_member
        observe_member(self.owner, self.concepts[argument], int(word))
        self._revision += 1

    @property
    def revision(self):
        owner = None if self._space is None else self._space()
        bank = None if owner is None else owner.similarity_codebook
        derived = None if bank is None else getattr(bank, 'mereology', None)
        return (id(self), self._revision, None if bank is None else (bank.trust.revision,
            bank.trust.views._version, bank.W._version if derived is None else derived.placement._version))

    def operation(self, name):
        """Definiteness is separability times evidence; tightness remains radius."""
        from ConceptHierarchy import can_observe
        from MereologicalCodes import CLASS_CONTIGUITY_COSINE
        key = self.revision
        if getattr(self, '_measurement_revision', None) != key:
            self._measurement_cache = {}
            self._measurement_revision = key
        if name in self._measurement_cache:
            return self._measurement_cache[name]
        fixed = name in FIXED_OPERATIONS
        arguments = tuple(arg for arg in self.arguments if arg.startswith(name+'_argin'))
        if not arguments and not fixed:
            raise ValueError(f'operation has no declared formal arguments: {name}')
        measurable = self._space is not None and can_observe(self.owner)
        details = {}
        for argument in arguments:
            members = self.members(argument) if self._space is not None else ()
            views = (int(self.owner.similarity_codebook.trust.views[
                self.owner._csw_row_of(self.concepts[argument])]) if self._space is not None else 0)
            separation = 0.
            if measurable and members:
                direction = self._direction(self.concepts[argument])
                competitors = [self.index[other] for other in self.arguments
                    if other != argument and self.members(other) and
                    float(direction @ self._direction(self.concepts[other])) < CLASS_CONTIGUITY_COSINE]
                scores = torch.stack([self.similarities(member) for member in members])
                own = scores[:, self.index[argument]]
                wins = ((own[:, None] > scores[:, competitors]).all(-1) if competitors
                        else torch.ones_like(own, dtype=torch.bool))
                if float(own.sum()) > 0:
                    separation = float(own[wins].sum() / own.sum())
            evidence = -math.expm1(-views/self.owner.similarity_codebook.trust.trust_views) if measurable else 0.
            details[argument] = dict(d_j=separation, views_j=views,
                definiteness_j=separation*evidence if measurable else None)
        definiteness = tuple(details[arg]['definiteness_j'] for arg in arguments)
        result = dict(uses=self.uses[name], arguments=arguments, definiteness=definiteness,
                    argument_measurements=details, measurable=measurable,
                    trust=1. if fixed or not measurable else min(definiteness))
        self._measurement_cache[name] = result
        return result

    def operation_index(self, name):
        if name not in self.operations:
            raise ValueError(f'operation absent from argument catalogue: {name}')
        return self.operations.index(name)

    def observe_use(self, name, operands):
        if self.frozen:
            return
        if name not in self.operations:
            raise ValueError(f'undeclared kept operation: {name}')
        for i, carried in enumerate(operands):
            argument = f'{name}_argin{i+1}'
            if argument not in self.index:
                raise ValueError(f'undeclared formal argument: {argument}')
            for word in carried:
                self.add(word, argument)
        self.uses[name] += 1
        self._revision += 1

    def observe_kept(self, program, binary_rules, unary_rules):
        if self.frozen or program is None:
            return set()
        observed = set()
        words = ([None] * len(program.leaves) if program.word_ids is None
                 else program.word_ids.detach().cpu().tolist())
        stack = []
        for kind, local, position in program.actions.detach().cpu().tolist():
            if kind < 0:
                break
            if kind == 0:
                if not 0 <= position < len(words):
                    raise ValueError('occupation journal has an invalid leaf')
                stack.append(() if words[position] is None else (int(words[position]),))
                continue
            if kind not in (1, 2, 3):
                raise ValueError('occupation journal has an invalid operation kind')
            rules = unary_rules if kind == 2 else binary_rules
            if not 0 <= local < len(rules):
                raise ValueError('occupation journal has an invalid operation')
            rule = rules[local]
            field_bind = kind == 1 and rule.method_name == 'bind'
            if field_bind:
                position = 0
            if kind == 2 or field_bind:
                if not 0 <= position < len(stack):
                    raise ValueError('occupation journal has an invalid unary position')
                at = len(stack) - 1 - position
                operands = [stack[at]]
            else:
                if len(stack) < 2:
                    raise ValueError('occupation journal underflows')
                operands = stack[-2:]
            name = operation_name(rule)
            self.observe_use(name, operands)
            observed.update(word for carried in operands for word in carried if word is not None and word >= 0)
            mask = flow_mask(rule)
            results = operands[::-1] if kind == 3 else [tuple(
                word for i, carried in enumerate(operands) if mask & (1 << i)
                for word in carried)]
            for i, carried in enumerate(results):
                for word in carried:
                    self.add(word, f'{name}_argout{i+1}')
            if kind == 2 or field_bind:
                stack[at] = results[0]
            else:
                stack[-2:] = results

        return observed

    def collapse(self):
        directions = [self._direction(self.concepts[name]) for name in self.arguments]
        return [dict(left=a, right=b, cosine=float(directions[i] @ directions[j]))
                for i, a in enumerate(self.arguments) for j, b in enumerate(self.arguments) if j > i]

    def get_extra_state(self):
        from IndependentComponents import _pack_state
        return _pack_state(dict(version=self.VERSION, arguments=self.arguments,
            concepts=self.concepts, operations=self.operations, uses=dict(self.uses)))

    def set_extra_state(self, value):
        from IndependentComponents import _unpack_state
        state = _unpack_state(value)
        if state['version'] != self.VERSION or tuple(state['arguments']) != self.arguments:
            raise ValueError('argument concept checkpoint schema does not match the grammar; replay its producer')
        if tuple(state['operations']) != self.operations or state['concepts'] != self.concepts:
            raise ValueError('argument concept identities differ from construction')
        if any(name not in self.operations or type(count) is not int or count < 0
               for name, count in state['uses'].items()):
            raise ValueError('invalid kept operation counts')
        self.uses = Counter(state['uses'])
        self._revision += 1


def update_flow(words, choice, binary_flow, unary_flow):
    """Carry sums of word-to-argument scores with the newest-first STM.

    The sum preserves one direct score contribution per carried occurrence;
    it is not renormalized into a new phrase class.
    """
    _, K, _ = words.shape
    result = words
    if binary_flow and K >= 2:
        mask = words.new_tensor(binary_flow, dtype=torch.long)[choice.local_op.clamp(0, len(binary_flow)-1)]
        selected = (words[:, 1:2] * ((mask & 1) != 0)[:, None, None]
                    + words[:, :1] * ((mask & 2) != 0)[:, None, None])
        folded = torch.cat((selected, words[:, 2:], torch.zeros_like(words[:, :1])), 1)[:, :K]
        result = torch.where((choice.applied & (choice.kind == 1) & (choice.stack_arity != 1 if choice.stack_arity is not None else True))[:, None, None], folded, words)
        swapped = torch.cat((words[:, :2].flip(1), words[:, 2:]), 1)
        result = torch.where((choice.applied & (choice.kind == 3))[:, None, None], swapped, result)
    if unary_flow:
        mask = words.new_tensor(unary_flow, dtype=torch.long)[choice.local_op.clamp(0, len(unary_flow)-1)]
        clear = choice.applied & (choice.kind == 2) & (mask == 0)
        position = torch.arange(K, device=words.device)[None] == choice.position[:, None]
        result = torch.where(clear[:, None, None] & position[..., None], 0., result)
    return result
