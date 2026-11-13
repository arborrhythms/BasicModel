"""Execution phases declared by the grammar, independent of model sizes."""
from dataclasses import dataclass

EXAMINATION = frozenset(('divide', 'descend', 'gloss'))
FIELD = frozenset(('intersection', 'union', 'not'))
# These are per-space substrate folds, not symbolic choices. The default
# dense numeric body still dispatches its declared pi/sigma layers.
SUBSTRATE = frozenset(('pi', 'sigma'))


@dataclass(frozen=True)
class GrammarPhases:
    examination: bool
    vector: bool
    interpret: bool
    operators: frozenset

    @property
    def per_word(self):
        return self.examination or self.vector

    @classmethod
    def configured(cls):
        from Language import TheGrammar
        TheGrammar._ensure_configured()
        names = frozenset(rule.method_name for rule in TheGrammar.rules
                          if rule.method_name)
        return cls(bool(names & EXAMINATION),
                   bool(names - EXAMINATION - FIELD - SUBSTRATE),
                   'interpret' in names, names)
