"""Model-file authority caps and their monotone opening state."""
import math


class TrustGates:
    names = ('interpret', 'clause')

    def __init__(self):
        from util import TheXMLConfig
        data = TheXMLConfig.space('ConceptualSpace', 'gates', dict(
            closedTrust=.1, interpret=dict(start='open', fraction=.5),
            clause=dict(start='open', threshold=.75, **{'except':'interpret sign bind lift'})))
        if not isinstance(data, dict) or set(data) != {'closedTrust', *self.names}:
            raise ValueError('gates requires closedTrust, interpret and clause')
        self.closed_trust = self.unit(data['closedTrust'], 'closedTrust')
        self.thresholds = {}
        self.open = {}
        for name, key in (('interpret','fraction'), ('clause','threshold')):
            entry = data[name]
            expected = {'start',key} | ({'except'} if name == 'clause' else set())
            if not isinstance(entry, dict) or set(entry) != expected or entry['start'] not in ('open','closed'):
                raise ValueError(f'invalid {name} gate configuration')
            self.thresholds[name] = self.unit(entry[key], key)
            self.open[name] = entry['start'] == 'open'
        excluded = data['clause']['except']
        excluded = '' if excluded is None else excluded
        if not isinstance(excluded, str):
            raise ValueError('clause gate except must be operation names separated by whitespace')
        self.excluded = frozenset(excluded.split())
        self.revision = 0

    @staticmethod
    def unit(value, name):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be finite and in [0, 1]')
        return value

    @classmethod
    def validate_state(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.names) or any(type(x) is not bool for x in value.values()):
            raise ValueError('checkpoint requires both boolean gate states')
        return dict(value)

    def restore(self, value):
        self.open = self.validate_state(value)
        self.revision += 1

    def advance(self, name, measurement):
        if not self.open[name] and measurement is not None and measurement >= self.thresholds[name]:
            self.open[name] = True
            self.revision += 1
            return True
        return False

    def cap(self, *, named, operation):
        covered = named and not self.open['interpret']
        covered |= (operation is not None and operation not in self.excluded and not self.open['clause'])
        return self.closed_trust if covered else 1.

    @staticmethod
    def measurements(owner, occupations):
        from ConceptTrust import measurements
        stable = None if owner is None else measurements(owner)['stable_word_fraction']
        if occupations is None or 'lift' not in occupations.operations:
            clause = None
        else:
            measured = occupations.operation('lift')
            if not measured['measurable']:
                return dict(interpret=stable, clause=None)
            arguments = measured['definiteness']
            if len(arguments) != 2:
                raise ValueError('clause gate requires two lift argument classes')
            clause = sum(arguments)/len(arguments)
        return dict(interpret=stable, clause=clause)
