"""Supplied grammar lessons train the live grammar, never interpret input.

Trees are teacher-only targets over the completed forward's retained word
leaves. This module has no parameters, inference path, decoder or word table.
It cannot install a program on an Understanding. The parser always runs first
and its own trace remains authoritative, including when its choices are wrong.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


MODIFICATION_OPERATORS = ('intersection', 'product')


def target_operations(target):
    """A bracket may admit several operators; STOP is not a fold target."""
    return () if target == -1 else ((target,) if isinstance(target, int) else target)


def _annotation_tree(language, tree, count):
    # A declared form can share an implementation and still have different
    # semantics (mint, bind, kind). Do not collapse those lesson targets.
    rules = {arity: {(getattr(r, 'surface_name', None) or r.method_name): i for i, r in enumerate(
        language._compose_binary_rules if arity == 2 else language._compose_unary_rules)}
        for arity in (1, 2)}
    parents, order = {}, []

    def validate(node):
        if isinstance(node, int) and not isinstance(node, bool):
            order.append(node)
            return ("leaf", node)
        if not isinstance(node, (tuple, list)) or len(node) not in (2, 3):
            raise ValueError("grammar lesson nodes need a declared unary/binary form")
        arity = len(node) - 1
        modification = arity == 2 and node[0] == 'modification'
        if modification and not any(name in rules[2] for name in MODIFICATION_OPERATORS):
            raise ValueError('modification bracket has no declared modification operator')
        if not modification and node[0] not in rules[arity]:
            raise ValueError(f"grammar lesson uses undeclared {arity}-ary form {node[0]!r}")
        children = tuple(validate(child) for child in node[1:])
        result = (node[0], *children)
        if children in parents:
            raise ValueError("grammar lesson repeats a subtree")
        parents[children] = result
        return result

    root = validate(tree)
    if order != list(range(count)):
        raise ValueError("grammar lesson must retain every forward word exactly once, in order")
    return root, parents, rules


def compose_examples(language, program, tree, *, variables=()):
    """Legacy operator-only annotations, over real captured forward leaves."""
    leaves = program.leaves.detach()
    if variables:
        if (len(set(variables)) != len(variables)
                or any(type(i) is not int or not 0 <= i < len(leaves) for i in variables)):
            raise ValueError("grammar variables must name distinct captured operand leaves")
        # Explicit variable augmentation stays on the teacher side.
        leaves = leaves.clone()
        for i in variables:
            from Layers import cube_chart, cube_unchart
            substitute = cube_unchart(F.normalize(torch.randn_like(leaves[i]), dim=-1)
                                     * cube_chart(leaves[i]).norm())
            leaves[i] = torch.where(torch.rand((), device=leaves.device) < .5,
                                    substitute, leaves[i])
    root, parents, rules = _annotation_tree(language, tree, len(leaves))
    stack, binary, unary = [], [], []

    def reduce(closing=False):
        if len(stack) < 2:
            return
        left, right = stack[-2:]
        parent = parents.get((left[0], right[0]))
        if parent is not None and parent[0] == 'modification':
            raise ValueError('modification brackets require student-state lessons (waits=True)')
        window = torch.stack((left[1], right[1]))
        target = -1 if parent is None else rules[2][parent[0]]
        if closing and target < 0:
            raise ValueError("grammar lesson tree cannot be scheduled by the bounded parser")
        binary.append((window.detach(), target, bool(closing), len(stack)))
        if target >= 0:
            op = language._tree_layer(2).ops[target]
            with torch.no_grad():
                value = op(window[0], window[1])
            stack[-2:] = [(parent, value)]

    for index, leaf in enumerate(leaves):
        stack.append((("leaf", index), leaf))
        reduce()
        node, value = stack[-1]
        parent = parents.get((node,))
        target = -1 if parent is None else rules[1][parent[0]]
        unary.append((value.detach(), target))
        if target >= 0:
            with torch.no_grad():
                value = language._tree_layer(1).unary_ops[target](value)
            stack[-1] = (parent, value)
    while len(stack) > 1:
        reduce(closing=True)
    if stack[0][0] != root:
        raise ValueError("grammar lesson needs an unavailable post-closing unary step")
    return binary, unary


def detached_state(value):
    """Own a lesson observation, including bounded native reference/context data."""
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass, replace
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: detached_state(item) for key, item in value.items()}
    if is_dataclass(value):
        return replace(value, **{field.name: detached_state(getattr(value, field.name))
                                 for field in fields(value)})
    if isinstance(value, tuple):
        items = [detached_state(item) for item in value]
        return type(value)(*items) if hasattr(value, '_fields') else tuple(items)
    if isinstance(value, list):
        return [detached_state(item) for item in value]
    return value


class ComposeLessonStates:
    """Observe actual choices, never install targets in the native traversal.

    A paired fork reuses its prefix. Keep that prefix's observations and replace
    only the rows actually visited again. The current attempted mask filters
    observations from a discarded suffix when constructing the lesson.
    """
    def __init__(self):
        self.rounds = {}

    def record(self, slot, window, kwargs, route):
        if slot is None:
            raise ValueError('constituent lesson needs an addressed compose round')
        payload = detached_state(dict(window=window, kwargs=kwargs))
        for row in route['valid'].nonzero().flatten().tolist():
            key = int(slot[row] if torch.is_tensor(slot) and slot.ndim else slot)
            self.rounds[row, key] = dict(payload, row=row,
                depth=int(kwargs['load_depth'][row]),
                kind=int(route['kind'][row]), position=int(route['position'][row]), slot=key,
                observed_probabilities=route['probabilities'][row].detach().clone())

    def sentence(self, row, positions, *, width, closing_base, closing_width, attempted):
        positions = positions.tolist()
        source = {position: index for index, position in enumerate(sorted(positions))}
        stack, states = [], []
        def visit(slot):
            if slot >= attempted.shape[1] or not bool(attempted[row, slot]):
                return
            state = self.rounds.get((row, slot))
            if state is None:
                raise ValueError(f'constituent lesson missed student round {row}:{slot}')
            if state['depth'] != len(stack):
                raise ValueError('constituent lesson provenance differs from the student stack')
            states.append(dict(state, spans=tuple(stack[-2:])))
            if state['kind'] == 1:
                right, left = stack.pop(), stack.pop()
                stack.append(left + right)
            elif state['kind'] == 3:
                stack[-2:] = reversed(stack[-2:])
            elif state['kind'] == 2:
                # Unary operations preserve the source leaves, even if lossy.
                pass
        for position in positions:
            stack.append((source[position],))
            for round_index in range(3):
                visit(3*position + round_index)
        for offset in range(closing_width):
            visit(closing_base + offset)
        return states


def constituent_examples(language, program, tree):
    """Bracket targets on the student's visited states, including wrong folds.

    Membership in a gold bracket is determined by the actual operands' source
    leaves. An incorrect fold cannot be undone by a replacement teacher trace.
    """
    root, _, rules = _annotation_tree(language, tree, len(program.leaves))
    parents = {}
    def spans(node):
        if node[0] == 'leaf':
            return (node[1],)
        children = tuple(spans(child) for child in node[1:])
        if len(children) != 2:
            raise ValueError('constituent brackets require binary annotations')
        parents[children] = (tuple(rules[2][name] for name in MODIFICATION_OPERATORS if name in rules[2])
                             if node[0] == 'modification' else rules[2][node[0]])
        return sum(children, ())
    spans(root)
    states = getattr(program, 'student_states', None)
    if states is None:
        raise ValueError('constituent lessons require captured student states')
    return [dict(state, target=parents.get(state['spans'], -1)) for state in states]


def score_student_state(layer, state):
    # The original call's masks, pressure, references and context are retained.
    # A sampled departure mask changes the action, not the CE distribution.
    kwargs = dict(state['kwargs'], policy_grad=True, sample=False, masked_action=None)
    _, _, route = layer(state['window'], **kwargs)
    return route['probabilities'][state['row']]


def student_target_actions(layer, state):
    """Map a rule target to the native menu's reference-expanded actions.

    The captured window has one binary position (the two newest operands).
    The annotation supplies admissible operators, not their references.
    STOP is the final action regardless of the menu's expansion.
    """
    if state['target'] == -1:
        return [-1]
    references = state['kwargs'].get('reference_data')
    menu = (tuple(range(layer.r_reduce)) if references is None else
            references.get('binary_ops', tuple(range(layer.r_reduce))))
    targets = target_operations(state['target'])
    return [index for index, operation in enumerate(menu) if operation in targets]


def student_target_probability(layer, state, probabilities):
    return probabilities[student_target_actions(layer, state)].sum()


def compose_loss(language, programs, lessons, *, errors=None, streams=None):
    """Supervise supplied forms in the same joint compose softmax.

    Operator-only lessons retain their contract. A complete constituent
    lesson explicitly sets ``waits``: its brackets also teach STOP while
    awaiting the next word. All targets follow the student's own forward.
    """
    if len(programs) != len(lessons):
        raise ValueError("one optional grammar lesson is required per forward row")
    layer = language._tree_layer(2)
    # A supplied syntax tree has no question context. Each teacher state is
    # scored alone, independently of the last reading's context or batch size.
    context_width = getattr(layer.chooser, 'WHAT_CONTEXT_DIM', 0)
    from Layers import Error
    errors = Error() if errors is None else errors
    for b, (program, lesson) in enumerate(zip(programs, lessons)):
        if lesson is None:
            continue
        if program is None:
            raise ValueError("grammar supervision requires an actual captured forward program")
        if lesson.get('waits', False):
            for example in constituent_examples(language, program, lesson['tree']):
                probabilities = score_student_state(layer, example)
                target = example['target']
                name = 'compose.wait' if target == -1 else 'compose.choice'
                # A bad prior fold can make the annotated wait illegal at the
                # closing deadline. Count its full CE and report it explicitly;
                # never substitute an unannotated fold or alter a native mask.
                probability = student_target_probability(layer, example, probabilities)
                if not bool(probability.detach() > 0):
                    errors.add('compose.unreachable', probabilities.new_ones(()),
                               category='grammar', trained=False)
                errors.error(name, -probability.clamp_min(1e-30).log(),
                    math.log(probabilities.shape[-1]), category='grammar')
            continue
        binary, unary = compose_examples(language, program, lesson["tree"],
                                         variables=lesson.get("variables", ()))
        prime = getattr(layer, '_expectation_prime', None)
        stream = b if streams is None else streams[b]
        prime = None if prime is None else prime[stream:stream+1]
        for window, target, _closing, _depth in binary:
            if target >= 0:
                _, _, route = layer(window.detach()[None], slots=1,
                    what_ctx=window.new_zeros(1, context_width), expectation_prime=prime)
                probabilities = route['probabilities']
                errors.error('compose.choice', -probabilities[0, target].clamp_min(1e-30).log(),
                             math.log(probabilities.shape[-1]), category='grammar')
        for value, target in unary:
            if target >= 0:
                _, _, route = layer(value.detach()[None, None], slots=1,
                    what_ctx=value.new_zeros(1, context_width), expectation_prime=prime)
                probabilities = route['probabilities']
                errors.error('compose.choice', -probabilities[0, target].clamp_min(1e-30).log(),
                             math.log(probabilities.shape[-1]), category='grammar')
    return errors.total()


def generate_loss(language, programs, lessons, registry, word_code, *, errors=None, word_code_by_stream=None):
    """Learn declared generate choices from separately supplied answer trees.

    Called only after the student's own output is fixed. The input's selected
    actions are never output labels. Each optional ``generation`` annotation
    supplies a desired semantic form, source operand positions, answer text,
    and one tree for each canonical role. Leaves resolve through the ordinary
    vocabulary; target spelling is unavailable to the inference walk.
    """
    from Layers import Error
    errors = Error() if errors is None else errors
    rules = {arity: {name: (i, op) for i, (name, op) in enumerate(zip(
        getattr(language, f"_generate_{'binary' if arity == 2 else 'unary'}_names"),
        getattr(language, f"_generate_{'binary' if arity == 2 else 'unary'}_ops")))}
        for arity in (1, 2)}
    n_binary = len(language._generate_binary_ops)
    stop = n_binary + len(language._generate_unary_ops)
    values, targets, numerical = [], [], []
    for b,(program, lesson) in enumerate(zip(programs, lessons)):
        supplied = None if lesson is None else lesson.get("generation")
        if supplied is None:
            continue
        if program is None:
            raise ValueError("generation lesson requires owned input operands")
        indices = supplied["operands"]
        if len(indices) != 2 or any(type(i) is not int or not 0 <= i < len(program.leaves) for i in indices):
            raise ValueError("generation lesson requires two valid source operands")
        refs = tuple(("sym", int(program.concept_ids[i])) for i in indices)
        meaning = registry.form(supplied["form"], *refs)
        # The supplied form is canonical, so these are explicitly labelled
        # NP1/NP2 source positions, never inferred from the student's parse.
        roles = (program.leaves[indices[0]].detach(), meaning.roles[1].detach(),
                 program.leaves[indices[1]].detach())
        words = supplied["text"].split()
        codes = [word_code(word) if word_code_by_stream is None else word_code_by_stream(word,b)
                 for word in words]
        if any(code is None for code in codes):
            # An answer-only word is not admitted by reading its label. It
            # becomes teachable after ordinary input has introduced it.
            continue
        leaves = [code.detach().to(roles[0]) for code in codes]
        order = []

        def tree(node):
            if type(node) is int and 0 <= node < len(leaves):
                order.append(node)
                return (leaves[node], stop, None, ())
            if not isinstance(node, (tuple, list)) or len(node) not in (2, 3):
                raise ValueError("generation tree needs declared unary/binary nodes")
            arity = len(node) - 1
            if node[0] not in rules[arity]:
                raise ValueError(f"generation lesson uses undeclared form {node[0]!r}")
            local, op = rules[arity][node[0]]
            op = getattr(op, "gl", op)
            children = tuple(tree(child) for child in node[1:])
            with torch.no_grad():
                parent = op.compose(*(child[0] for child in children))
            action = local if arity == 2 else n_binary + local
            return (parent, action, op, children)

        roots = tuple(tree(node) for node in supplied["roots"])
        if len(roots) != 3 or order != list(range(len(words))):
            raise ValueError("generation roots must cover every output word once, in order")

        def teach(node, parent):
            _natural_parent, action, op, children = node
            parent = parent.detach()
            values.append(parent)
            targets.append(action)
            if op is None:
                return
            predicted = op.generate(parent)
            predicted = predicted if len(children) == 2 else (predicted,)
            for actual, child in zip(predicted, children):
                errors.squared('generate.operands', actual.detach(), child[0],
                               category='grammar', objective='output', trained=False)
                teach(child, child[0])

        for root, role in zip(roots, roles):
            teach(root, role)
    if not values:
        return None
    values = torch.stack(values)
    target = torch.tensor(targets, device=values.device, dtype=torch.long)
    errors.categorical('generate.choice', language.generate_policy_logits(values), target,
                       category='grammar', objective='output')
    return errors.total()
