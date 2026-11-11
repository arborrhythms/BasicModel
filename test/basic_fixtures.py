"""Explicit basic-level occurrences for isolated native-literal fixtures."""


def feature(owner, row, tower, value, weight=0., *, negated=False, parts=None):
    """Declare the fixture occurrences, then exercise the native selector writer."""
    basic = owner.basic_percepts
    if tower == 'ps':
        group = tuple(value) if isinstance(value, (tuple, list)) else (int(value),)
        basic.reserve(group, ())
    elif tower == 'ws':
        if parts is None:
            raise ValueError('a property fixture must declare its identity parts')
        groups = tuple((part,) if isinstance(part, int) else tuple(part) for part in parts)
        for group in groups:
            basic.reserve(group, (int(value),))
    return owner.add_concept_feature(row, tower, value, weight, negated=negated)
