"""Action order and operand provenance of one recorded grammar walk."""
import torch


def steps(reading):
    """The shared index for a native path, or the actions of an isolated journal."""
    index = getattr(reading, 'action_index', None)
    if index is not None:
        return index.steps
    if isinstance(reading, ReadingIndex):
        return reading.steps
    actions = reading if torch.is_tensor(reading) else reading.actions
    return actions.tolist()


class ReadingIndex:
    def __init__(self, actions, word_count, language):
        self.steps = []
        self.operands = []
        self.parents = []
        self.word_actions = {}
        stack = []
        operations = actions.tolist()
        for at, (kind, local, word) in enumerate(operations):
            self.parents.append(None)
            if kind == 0:
                if word_count is not None and (not 0 <= word < word_count or word in self.word_actions):
                    raise ValueError('a reading must shift each leaf exactly once')
                self.word_actions[word] = at
                local, children = -1, ()
                stack.append(at)
            elif kind == 3:
                if len(stack) < 2:
                    raise ValueError('transposition requires two operands')
                children = tuple(stack[-2:])
                stack[-2:] = reversed(children)
                word = -1
            else:
                if kind not in (1, 2):
                    raise ValueError('unknown sentence action')
                if kind == 2 or getattr(language._compose_binary_rules[local], 'method_name', None) == 'bind':
                    offset = max(0, word) if kind == 2 else 0
                    position = len(stack) - 1 - offset
                    if position < 0:
                        raise ValueError('operation requires an occupied operand')
                    children = (stack[position],)
                    stack[position] = at
                    word = offset if kind == 2 else -1
                else:
                    if len(stack) < 2:
                        raise ValueError('binary operation requires two operands')
                    children = tuple(stack[-2:])
                    stack[-2:] = [at]
                    word = -1
                for child in children:
                    self.parents[child] = at
            self.operands.append(children)
            self.steps.append((kind, local, word))
        if word_count is not None and len(self.word_actions) != word_count:
            raise ValueError('a reading must shift each leaf exactly once')
        self.roots = tuple(stack)
        self.actions = actions.new_tensor(self.steps).reshape(-1, 3)

    @property
    def forest(self):
        """Nested display form, built only for a caller asking to inspect the tree."""
        nodes = []
        for at, (kind, local, word) in enumerate(self.steps):
            if kind == 0:
                nodes.append((kind, word))
            elif kind == 3:
                nodes.append(None)
            else:
                nodes.append((kind, local, at, *(nodes[child] for child in self.operands[at])))
        return tuple(nodes[at] for at in self.roots)


    def validate_interfaces(self, language):
        """The selected seat must still identify one declared execution interface."""
        from PartsOfSpeech import operation_name
        selected = {(kind, local) for kind, local, _ in self.steps if kind != 0}
        if not selected:
            return
        catalogs = {1: language._compose_binary_rules, 2: language._compose_unary_rules}
        matches = {}
        for kind, catalog in catalogs.items():
            for local, rule in enumerate(catalog):
                key = operation_name(rule), rule.canonical, rule.kernel_index
                matches.setdefault(key, []).append((kind, local))
        for kind, local in selected:
            rule = catalogs[1 if kind == 3 else kind][local]
            seat = language.parts_of_speech.seats[operation_name(rule)]
            key = language.parts_of_speech.seats[seat.concept].name, rule.canonical, rule.kernel_index
            if len(matches.get(key, ())) != 1:
                raise ValueError('the operator seat has no unique declared execution interface')
            if kind == 3 and len(rule.lhs.split(',')) != 2:
                raise ValueError('an unconsumed operator must be a declared transposition')
