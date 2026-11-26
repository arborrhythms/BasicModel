"""The close's single host view of chooser-written STM pair heads.

The operation journal labels a pair's execution interface and numerical
column. Only the pair references determine its operands and roots; no
shift/reduce stack is replayed to discover a parse.
"""
from Memory import OCCURRENCE, SYMBOL, RELATION


def validate_catalog(binary, unary):
    """Validate immutable execution interfaces when the catalogs are attached."""
    from PartsOfSpeech import operation_name
    interfaces = set()
    for rules in (binary, unary):
        for rule in rules:
            key = operation_name(rule), rule.canonical, rule.kernel_index
            if key in interfaces:
                raise ValueError('the operator seat has no unique declared execution interface')
            interfaces.add(key)
    return binary, unary


class ParseForest:
    def __init__(self, actions, records, words, pairs, roots, columns, language):
        from PartsOfSpeech import operation_name
        self.actions = actions
        self.steps = [tuple(row) for row in records]
        self.columns = columns
        self.word_actions = {word: at for at, (kind, _, word) in enumerate(self.steps) if kind == 0}
        self.operands = [()] * len(self.steps)
        self.parents = [None] * len(self.steps)
        self.visits = tuple(at for at, (kind, _, _) in enumerate(self.steps) if kind in (0, 1, 2))
        sites = {(OCCURRENCE, words[word]): at for word, at in self.word_actions.items()}
        if len(self.word_actions) != len(words):
            raise ValueError('a parse must contain every read occurrence exactly once')
        catalogs = {1: language._compose_binary_rules, 2: language._compose_unary_rules}
        for at in self.visits:
            kind, local, _ = self.steps[at]
            if kind == 0:
                continue
            column = self.columns[at]
            parent = pairs[2 * column + 1]
            expected = language.parts_of_speech.seats[operation_name(catalogs[kind][local])].concept
            if tuple(parent[0]) != (SYMBOL, expected):
                raise ValueError('parse operator does not match its numerical journal')
            operand = parent[1]
            children = pairs[2 * column] if tuple(operand) == (RELATION, 2 * column) else [operand]
            self.operands[at] = tuple(sites[tuple(child)] for child in children)
            for child in self.operands[at]:
                self.parents[child] = at
            sites[RELATION, 2 * column + 1] = at
        self.roots = tuple(sites[tuple(root)] for root in roots if root[0])

    @property
    def forest(self):
        nodes = {}
        for at in self.visits:
            kind, local, word = self.steps[at]
            nodes[at] = ((kind, word) if kind == 0 else
                         (kind, local, at, *(nodes[child] for child in self.operands[at])))
        return tuple(nodes[at] for at in self.roots)

    def validate_interfaces(self, language):
        catalogs = language._compose_binary_rules, language._compose_unary_rules
        checked = getattr(language, '_parse_interfaces', None)
        if checked is None or any(a is not b for a, b in zip(checked, catalogs)):
            # Small independent fixtures can attach their own catalogs.
            validate_catalog(*catalogs)
        for kind, local, _ in self.steps:
            if kind == 3 and len(catalogs[0][local].lhs.split(',')) != 2:
                raise ValueError('an unconsumed operator must be a declared transposition')


def native(program):
    index = getattr(program, 'action_index', program)
    return index if isinstance(index, ParseForest) else None


def visits(program):
    """Native pair traversal; isolated legacy fixtures keep their old records."""
    index = native(program)
    if index is not None:
        return ((at, index.steps[at]) for at in index.visits)
    from ReadingIndex import steps
    return enumerate(steps(program))
