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


def _annotation_tree(language, tree, count):
    rules = {arity: {r.method_name: i for i, r in enumerate(
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
        if node[0] not in rules[arity]:
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
            substitute = F.normalize(torch.randn_like(leaves[i]), dim=-1) * leaves[i].norm()
            leaves[i] = torch.where(torch.rand((), device=leaves.device) < .5,
                                    substitute, leaves[i])
    root, parents, rules = _annotation_tree(language, tree, len(leaves))
    stack, binary, unary = [], [], []

    def reduce(closing=False):
        if len(stack) < 2:
            return
        left, right = stack[-2:]
        parent = parents.get((left[0], right[0]))
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


def constituent_examples(language, program, tree):
    """Explicit folds AND waits on the native three-round, two-slot window.

    A bracket lesson permits several reductions after a word and waits when
    the next subtree needs more input. The final closing uses its own bound.
    Only detached teacher states are built; the student's journal is untouched.
    """
    leaves = program.leaves.detach()
    root, parents, rules = _annotation_tree(language, tree, len(leaves))
    layer = language._tree_layer(2)
    capacity = int(program.compose_capacity)
    closing_budget = int(program.compose_closing_budget)
    stack, examples = [], []

    def step(closing, remaining):
        depth = len(stack)
        if not depth:
            raise ValueError('constituent lesson has an empty compose state')
        parent = parents.get(tuple(node for node, _ in stack[-2:])) if depth >= 2 else None
        arity = 2
        if parent is None:
            parent = parents.get((stack[-1][0],))
            arity = 1
        window = leaves.new_zeros(2, leaves.shape[-1])
        window[:min(2, depth)] = torch.stack([value for _, value in stack[-2:]])
        target = -1
        if parent is not None:
            local = rules[arity][parent[0]]
            target = (local if arity == 2 else
                      layer.r_reduce + (min(2, depth)-1)*layer.r_apply + local)
        if target < 0 and (depth > (1 if closing else capacity-1)):
            raise ValueError('constituent lesson requires an illegal wait')
        examples.append(dict(window=window.detach(), target=target, depth=depth,
            closing=closing, remaining=remaining))
        if parent is None:
            return False
        with torch.no_grad():
            operands = [value for _, value in stack[-arity:]]
            value = (layer.ops[local](*operands) if arity == 2
                     else layer.unary_ops[local](*operands))
        stack[-arity:] = [(parent, value)]
        return True

    for index, leaf in enumerate(leaves):
        if len(stack) >= capacity:
            raise ValueError('constituent annotation exceeds the live STM capacity')
        stack.append((("leaf", index), leaf))
        for round_index in range(3):
            if not step(False, 3-round_index):
                break
    for round_index in range(closing_budget):
        if not step(True, closing_budget-round_index):
            break
    if len(stack) != 1 or stack[0][0] != root:
        raise ValueError('constituent annotation exceeds the live closing budget')
    return examples


def compose_loss(language, programs, lessons, *, errors=None):
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
    for program, lesson in zip(programs, lessons):
        if lesson is None:
            continue
        if program is None:
            raise ValueError("grammar supervision requires an actual captured forward program")
        if lesson.get('waits', False):
            for example in constituent_examples(language, program, lesson['tree']):
                window, depth = example['window'], example['depth']
                closing = example['closing']
                _, _, route = layer(window[None],
                    depth=torch.tensor([min(2, depth)], device=window.device),
                    slots=2 if not closing or depth == 1 else 0,
                    allowance=1 if closing else program.compose_capacity-1,
                    rounds_left=example['remaining'],
                    load_depth=torch.tensor([depth], device=window.device),
                    what_ctx=window.new_zeros(1, context_width))
                probabilities = route['probabilities']
                target = example['target']
                name = 'compose.wait' if target < 0 else 'compose.choice'
                errors.error(name, -probabilities[0, target].clamp_min(1e-30).log(),
                    math.log(probabilities.shape[-1]), category='grammar')
            continue
        binary, unary = compose_examples(language, program, lesson["tree"],
                                         variables=lesson.get("variables", ()))
        for window, target, _closing, _depth in binary:
            if target >= 0:
                _, _, route = layer(window.detach()[None], slots=1,
                    what_ctx=window.new_zeros(1, context_width))
                probabilities = route['probabilities']
                errors.error('compose.choice', -probabilities[0, target].clamp_min(1e-30).log(),
                             math.log(probabilities.shape[-1]), category='grammar')
        for value, target in unary:
            if target >= 0:
                _, _, route = layer(value.detach()[None, None], slots=1,
                    what_ctx=value.new_zeros(1, context_width))
                probabilities = route['probabilities']
                errors.error('compose.choice', -probabilities[0, target].clamp_min(1e-30).log(),
                             math.log(probabilities.shape[-1]), category='grammar')
    return errors.total()


def generate_loss(language, programs, lessons, registry, word_code, *, errors=None):
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
    for program, lesson in zip(programs, lessons):
        supplied = None if lesson is None else lesson.get("generation")
        if supplied is None:
            continue
        if program is None:
            raise ValueError("generation lesson requires owned input operands")
        indices = supplied["operands"]
        if len(indices) != 2 or any(type(i) is not int or not 0 <= i < len(program.leaves) for i in indices):
            raise ValueError("generation lesson requires two valid source operands")
        refs = tuple(("sym", int(program.concept_ids[i])) for i in indices)
        meaning = registry.form(supplied["form"], *refs, mode="assertive")
        # The supplied form is canonical, so these are explicitly labelled
        # NP1/NP2 source positions, never inferred from the student's parse.
        roles = (program.leaves[indices[0]].detach(), meaning.roles[1].detach(),
                 program.leaves[indices[1]].detach())
        words = supplied["text"].split()
        codes = [word_code(word) for word in words]
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
