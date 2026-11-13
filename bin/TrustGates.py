"""Monotone operation authority inferred from declared roots and kept trees."""
import math


class TrustGates:
    def __init__(self):
        from util import TheXMLConfig
        from Language import TheGrammar
        from PartsOfSpeech import FIXED_OPERATIONS, operation_name
        TheGrammar._ensure_configured()
        data = TheXMLConfig.space('ConceptualSpace', 'gates', dict(closedTrust=.1, threshold=.75))
        if not isinstance(data, dict) or set(data) != {'closedTrust', 'threshold'}:
            raise ValueError('gates requires closedTrust and threshold; interpret/clause groups are retired')
        self.closed_trust = self.unit(data['closedTrust'], 'closedTrust')
        self.threshold = self.unit(data['threshold'], 'threshold')
        rules = tuple(TheGrammar.rules_upward)
        self.names = tuple(dict.fromkeys(operation_name(rule) for rule in rules if operation_name(rule)))
        starts = {symbol for pattern in TheGrammar.ws_start_patterns for symbol in pattern}
        roots = {operation_name(rule) for rule in rules if
                 getattr(rule, 'clause_form', None) in ('S', 'implies') or rule.lhs in starts}
        self.exempt = frozenset(roots | FIXED_OPERATIONS)
        self.open = {name: name in self.exempt for name in self.names}
        self.parents = {name: set() for name in self.names}
        self.opened_at = {name: 0 for name in self.names if self.open[name]}
        self.revision = 0

    @staticmethod
    def unit(value, name):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be finite and in [0, 1]')
        return value

    def state(self):
        return dict(open=dict(self.open), parents={name: tuple(sorted(value)) for name,value in self.parents.items()},
                    opened_at=dict(self.opened_at))

    def validate_state(self, value):
        if (not isinstance(value,dict) or set(value) != {'open','parents','opened_at'}
                or set(value['open']) != set(self.names) or set(value['parents']) != set(self.names)
                or any(type(v) is not bool for v in value['open'].values())
                or any(set(parents)-set(self.names) for parents in value['parents'].values())):
            raise ValueError('checkpoint requires per-operator gates and kept-tree dependencies')
        return value

    def restore(self, value):
        value = self.validate_state(value)
        self.open = dict(value['open'])
        self.parents = {name:set(parents) for name,parents in value['parents'].items()}
        self.opened_at = dict(value['opened_at'])
        self.revision += 1

    def observe(self, parent, descendants):
        for child in descendants:
            if child != parent and child in self.parents and parent in self.open and parent not in self.parents[child]:
                self.parents[child].add(parent)
                self.revision += 1

    def advance(self, name, measurement, *, when=None):
        if not self.open[name] and measurement is not None and measurement >= self.threshold:
            self.open[name] = True
            self.opened_at[name] = when
            self.revision += 1
            return True
        return False

    def cap(self, *, named, operation):
        name = 'interpret' if named and operation is None else operation
        return 1. if name is None or name in self.exempt or self.open.get(name,False) else self.closed_trust

    def measurements(self, owner, occupations):
        result = {}
        for name in self.names:
            if name in self.exempt:
                result[name] = 1.
                continue
            values = []
            for parent in self.parents[name]:
                if parent in self.exempt:
                    # Root exemption removes its cap, not the requirement
                    # that its arguments become definite for descendants.
                    pass
                if occupations is None or parent not in occupations.operations:
                    values = []; break
                measurement = occupations.operation(parent)
                if not measurement['measurable'] or any(v is None for v in measurement['definiteness']):
                    values = []; break
                values.extend(measurement['definiteness'])
            result[name] = min(values) if values else None
        return result
