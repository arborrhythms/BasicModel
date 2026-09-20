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


def compose_examples(language, program, tree, *, variables=()):
    """Teacher-forced states for one annotated tree, over real forward leaves.

    A leaf is its zero-based ordinal; a node is [declared_form, children...].
    Every word must occur exactly once in order. One post-word binary and one
    unary are allowed, followed by sentence seals, matching the live parser.
    Targets stay on this scoring side of the completed forward boundary.
    """
    leaves = program.leaves.detach()
    if variables:
        if (len(set(variables)) != len(variables)
                or any(type(i) is not int or not 0 <= i < len(leaves) for i in variables)):
            raise ValueError("grammar variables must name distinct captured operand leaves")
        # Explicitly annotated variables teach structural invariance over
        # full concept values. Half retain the actual lexical value; half
        # receive a fresh direction at the same activation magnitude. These
        # are teacher states, never a replacement for the student's program.
        leaves = leaves.clone()
        for i in variables:
            substitute = F.normalize(torch.randn_like(leaves[i]), dim=-1) * leaves[i].norm()
            leaves[i] = torch.where(torch.rand((), device=leaves.device) < .5,
                                    substitute, leaves[i])
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
    if order != list(range(len(leaves))):
        raise ValueError("grammar lesson must retain every forward word exactly once, in order")
    stack, binary, unary = [], [], []

    def reduce(seal=False):
        if len(stack) < 2:
            return
        left, right = stack[-2:]
        parent = parents.get((left[0], right[0]))
        window = torch.stack((left[1], right[1]))
        target = -1 if parent is None else rules[2][parent[0]]
        if seal and target < 0:
            raise ValueError("grammar lesson tree cannot be scheduled by the bounded parser")
        binary.append((window.detach(), target, bool(seal), len(stack)))
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
                value = language._tree_layer(1).ops[target](value)
            stack[-1] = (parent, value)
    while len(stack) > 1:
        reduce(seal=True)
    if stack[0][0] != root:
        raise ValueError("grammar lesson needs an unavailable post-seal unary step")
    return binary, unary


def compose_loss(language, programs, lessons, *, base_tau=.75, capacity=8):
    """Cross entropy for the EXISTING binary/unary grammar MLPs."""
    if len(programs) != len(lessons):
        raise ValueError("one optional grammar lesson is required per forward row")
    binary, unary = [], []
    for program, lesson in zip(programs, lessons):
        if lesson is None:
            continue
        if program is None:
            raise ValueError("grammar supervision requires an actual captured forward program")
        b, u = compose_examples(language, program, lesson["tree"],
                                variables=lesson.get("variables", ()))
        binary.extend(b)
        unary.extend(u)
    costs = []
    if binary:
        layer = language._tree_layer(2)
        values = torch.stack([entry[0] for entry in binary])
        with torch.no_grad():
            candidates = layer._stacked_reduced(values)
        copy, reduce = layer.chooser.score_binary(
            values[..., :layer.d_model], candidates[..., :layer.d_model],
            layer.copy_anchor, layer.reduce_anchor)
        # This is exactly the live bounded parser's copy/reduce confidence.
        copy = (copy.logsumexp(-1) - math.log(copy.shape[-1])).sum(-1)
        seals = torch.tensor([entry[2] for entry in binary], device=values.device)
        depth = torch.tensor([entry[3] for entry in binary], device=values.device)
        pressure = ((depth.float() - 2) / max(1, capacity - 2)).clamp(0, 1) if capacity > 2 else torch.zeros_like(copy)
        threshold = (float(base_tau) * (1 - pressure)).clamp(1e-5, 1 - 1e-5)
        seals = seals | (depth >= capacity)
        if any(entry[1] < 0 and entry[3] >= capacity for entry in binary):
            raise ValueError("grammar lesson requires more STM capacity")
        keep = copy + math.log(layer.r_reduce) + torch.logit(threshold)
        logits = torch.cat((keep[:, None], reduce[:, 0]), -1)
        logits = torch.cat((logits[:, :1].masked_fill(seals[:, None], -1e4), logits[:, 1:]), -1)
        targets = torch.tensor([entry[1] + 1 for entry in binary], device=values.device)
        costs.append(F.cross_entropy(logits, targets))
    if unary:
        layer = language._tree_layer(1)
        values = torch.stack([entry[0] for entry in unary])[:, None]
        with torch.no_grad():
            candidates = layer._stacked_applied(values)
        copy, apply = layer.chooser.score_unary(
            values[..., :layer.d_model], candidates[..., :layer.d_model],
            layer.copy_anchor, layer.apply_anchor)
        # One aggregate copy action, just as for the bounded binary decision.
        keep = copy.logsumexp(-1) - math.log(copy.shape[-1])
        logits = torch.cat((keep[:, :, None], apply), -1)[:, 0]
        targets = torch.tensor([entry[1] + 1 for entry in unary], device=values.device)
        costs.append(F.cross_entropy(logits, targets))
    return torch.stack(costs).mean() if costs else None


def generate_loss(language, programs, lessons, registry, word_code):
    """Learn declared generate choices from separately supplied answer trees.

    Called only after the student's own output is fixed. The input's selected
    actions are never output labels. Each optional ``generation`` annotation
    supplies a desired semantic form, source operand positions, answer text,
    and one tree for each canonical role. Leaves resolve through the ordinary
    vocabulary; target spelling is unavailable to the inference walk.
    """
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
                numerical.append((actual - child[0]).square().sum())
                teach(child, child[0])

        for root, role in zip(roots, roles):
            teach(root, role)
    if not values:
        return None
    values = torch.stack(values)
    target = torch.tensor(targets, device=values.device, dtype=torch.long)
    cost = F.cross_entropy(language.generate_policy_logits(values), target)
    if numerical:
        cost = cost + torch.stack(numerical).mean()
    return cost
