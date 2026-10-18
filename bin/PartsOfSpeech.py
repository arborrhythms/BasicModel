"""Distributional word types: occupations of fixed formal compose arguments.

Only admitted programs supply evidence. The grammar lesson has no entry point
here. Argument occupant cosines are measurements, never a merge operation.
"""
from collections import Counter

import torch
from torch import nn


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


def head_role(rule):
    """Only an existing head or a sole operand identifies a result's word."""
    return int(getattr(rule, 'head_role', 0) or (len(rule.rhs_symbols) == 1))


class PartsOfSpeech(nn.Module):
    def __init__(self, arguments):
        super().__init__()
        self.arguments = tuple(arguments)
        self.index = {name: i for i, name in enumerate(self.arguments)}
        self.counts = {}
        self.labels = {}
        self.frozen = False

    def distribution(self, word):
        counts = self.counts.get(int(word), {})
        total = sum(counts.values())
        if not total:
            return [1. / max(1, len(self.arguments))] * len(self.arguments)
        return [counts.get(name, 0) / total for name in self.arguments]

    def part_of_speech(self, word):
        return self.labels.get(int(word), self.arguments[0] if self.arguments else None)

    def add(self, word, argument):
        if self.frozen or int(word) < 0:
            return
        if argument not in self.index:
            raise ValueError(f'undeclared formal argument: {argument}')
        word = int(word)
        counts = self.counts.setdefault(word, Counter())
        counts[argument] += 1
        self.labels[word] = max(self.arguments, key=lambda name: counts.get(name, 0))

    def probabilities(self, words, like):
        """Eager boundary: a frozen table shared by the paired trials."""
        values = [self.distribution(word) for word in words.detach().cpu().flatten().tolist()]
        return like.new_tensor(values).reshape(*words.shape, len(self.arguments))

    def observe_kept(self, program, binary_rules, unary_rules):
        if self.frozen or program is None or program.word_ids is None:
            return
        words = program.word_ids.detach().cpu().tolist()
        stack = []
        for kind, local, position in program.actions.detach().cpu().tolist():
            if kind < 0:
                break
            if kind == 0:
                if not 0 <= position < len(words):
                    raise ValueError('occupation journal has an invalid leaf')
                stack.append(int(words[position]))
                continue
            if kind not in (1, 2, 3):
                raise ValueError('occupation journal has an invalid operation kind')
            rules = unary_rules if kind == 2 else binary_rules
            if not 0 <= local < len(rules):
                raise ValueError('occupation journal has an invalid operation')
            rule = rules[local]
            if kind == 2:
                if not 0 <= position < len(stack):
                    raise ValueError('occupation journal has an invalid unary position')
                at = len(stack) - 1 - position
                operands = [stack[at]]
            else:
                if len(stack) < 2:
                    raise ValueError('occupation journal underflows')
                operands = stack[-2:]
            name = operation_name(rule)
            for i, word in enumerate(operands):
                self.add(word, f'{name}_argin{i+1}')
            head = head_role(rule)
            results = operands[::-1] if kind == 3 else [operands[head-1] if head else -1]
            for i, word in enumerate(results):
                self.add(word, f'{name}_argout{i+1}')
            if kind == 2:
                stack[at] = results[0]
            else:
                stack[-2:] = results

    def collapse(self):
        words = sorted(self.counts)
        vectors = torch.tensor([[self.counts[word].get(arg, 0) for word in words]
                                for arg in self.arguments], dtype=torch.float64)
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
        cosines = vectors @ vectors.T
        return [dict(left=a, right=b, cosine=float(cosines[i, j]))
                for i, a in enumerate(self.arguments)
                for j, b in enumerate(self.arguments) if j > i]

    def get_extra_state(self):
        from IndependentComponents import _pack_state
        return _pack_state(dict(version=1, arguments=self.arguments,
            counts={word: dict(counts) for word, counts in self.counts.items()}))

    def set_extra_state(self, value):
        from IndependentComponents import _unpack_state
        state = _unpack_state(value)
        if state['version'] != 1 or tuple(state['arguments']) != self.arguments:
            raise ValueError('occupation argument schema does not match the grammar')
        self.counts = {int(word): Counter(counts) for word, counts in state['counts'].items()}
        self.labels = {word: max(self.arguments, key=lambda name: counts.get(name, 0))
                       for word, counts in self.counts.items()}


def update_heads(heads, choice, binary_heads):
    """Move head distributions with the newest-first STM, including transposes."""
    B, K, R = heads.shape
    if not binary_heads or K < 2:
        return heads
    role = heads.new_tensor(binary_heads, dtype=torch.long)[choice.local_op.clamp(0, len(binary_heads)-1)]
    selected = heads.gather(1, (2-role.clamp(1, 2))[:, None, None].expand(B, 1, R))
    selected = torch.where((role > 0)[:, None, None], selected, torch.zeros_like(selected))
    folded = torch.cat((selected, heads[:, 2:], torch.zeros_like(heads[:, :1])), 1)[:, :K]
    result = torch.where((choice.applied & (choice.kind == 1))[:, None, None], folded, heads)
    swapped = torch.cat((heads[:, :2].flip(1), heads[:, 2:]), 1)
    return torch.where((choice.applied & (choice.kind == 3))[:, None, None], swapped, result)
