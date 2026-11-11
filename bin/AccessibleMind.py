"""The ten accessible subsystems and the grammar's enforced effect contract."""
from enum import Enum
from dataclasses import dataclass


class Subsystem(Enum):
    PERCEPT = 'perceptual knowing'
    KNOWING = 'order-zero knowing'
    SYMBOLIC = 'higher-order knowing'
    SERIAL = 'serial thinking'
    PRIMING = 'priming'
    EXPECTATION = 'expectation'
    LTM = 'long-term memory'
    BUDGET = 'budget'
    MERONYMY = 'meronymic access'
    TAXONOMY = 'taxonomic access'


S = Subsystem
PERMISSIONS = {
    'compose': (frozenset((S.PERCEPT, S.KNOWING, S.SYMBOLIC, S.SERIAL, S.PRIMING, S.MERONYMY)),
                frozenset((S.KNOWING, S.SYMBOLIC, S.SERIAL))),
    'thought': (frozenset(S) - {S.PERCEPT},
                frozenset((S.KNOWING, S.SYMBOLIC, S.SERIAL, S.EXPECTATION, S.BUDGET, S.LTM))),
    'generate': (frozenset((S.KNOWING, S.SYMBOLIC, S.SERIAL, S.BUDGET)), frozenset((S.PERCEPT, S.BUDGET))),
    'chooser': (frozenset((S.KNOWING, S.SYMBOLIC, S.SERIAL, S.EXPECTATION, S.LTM, S.BUDGET)), frozenset()),
    'closing': (frozenset((S.SERIAL, S.EXPECTATION)), frozenset((S.LTM, S.EXPECTATION))),
}


def check_access(grammar, reads, writes):
    if grammar not in PERMISSIONS:
        raise ValueError('unknown grammar effect owner')
    if any(not isinstance(item, Subsystem) for item in (*reads, *writes)):
        raise ValueError('grammar scopes must be accessible-mind Subsystem members')
    allowed_read, allowed_write = PERMISSIONS[grammar]
    if not set(reads) <= allowed_read or not set(writes) <= allowed_write:
        raise ValueError(f'{grammar} effect exceeds its subsystem permissions')


@dataclass(frozen=True)
class OperatorEffects:
    """One exhaustive capability declaration, shared by every operator face.

    Face permissions only remove capabilities. Tensor kernels calculate a
    candidate; the owning walk performs the selected effect once per round.
    """
    reads: tuple
    writes: tuple

    def __post_init__(self):
        for name in ('reads', 'writes'):
            original = tuple(getattr(self, name))
            try:
                values = tuple(value if isinstance(value, Subsystem) else
                               Subsystem[str(value).upper()] for value in original)
            except KeyError as error:
                raise ValueError(f'{name} names an unknown subsystem') from error
            if len(set(values)) != len(values):
                raise ValueError(f'{name} repeats a subsystem')
            object.__setattr__(self, name, values)

    def for_face(self, phase):
        if phase not in PERMISSIONS:
            raise ValueError('unknown grammar effect owner')
        reads, writes = PERMISSIONS[phase]
        return OperatorEffects(tuple(s for s in self.reads if s in reads),
                               tuple(s for s in self.writes if s in writes))

    def require(self, phase, *, reads=(), writes=()):
        permitted = self.for_face(phase)
        if not set(reads) <= set(permitted.reads) or not set(writes) <= set(permitted.writes):
            raise ValueError('operator attempted an undeclared or face-forbidden effect')


class EffectRound:
    """Commit guard for a selected operation's effects; never a second chooser."""
    def __init__(self, effects, phase):
        self.effects, self.phase, self.written = effects, phase, set()

    def claim(self, subsystem):
        self.effects.require(self.phase, writes=(subsystem,))
        if subsystem in self.written:
            raise ValueError('a subsystem may be written only once in an operation round')
        self.written.add(subsystem)


@dataclass(frozen=True)
class OperatorFootprint:
    """Exhaustive coordinate capabilities, subordinate to subsystem effects.

    Addresses, time and work meters are outside these three coordinate
    families. A percept can carry form and poles; meaning needs a conceptual
    carrier. A face may remove a subsystem write, never grant a coordinate.
    """
    reads: tuple
    writes: tuple

    def __post_init__(self):
        for name in ('reads', 'writes'):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values) or not set(values) <= {'form', 'meaning', 'poles'}:
                raise ValueError(f'footprint {name} requires distinct form, meaning or poles')
            object.__setattr__(self, name, values)

    def check_effects(self, effects):
        carriers = {
            'form': {S.PERCEPT, S.KNOWING, S.SYMBOLIC, S.SERIAL, S.LTM, S.EXPECTATION},
            'meaning': {S.KNOWING, S.SYMBOLIC, S.SERIAL, S.LTM, S.EXPECTATION, S.PRIMING},
            'poles': {S.PERCEPT, S.KNOWING, S.SYMBOLIC, S.SERIAL, S.LTM, S.EXPECTATION},
        }
        for name in ('reads', 'writes'):
            for component in getattr(self, name):
                if not carriers[component].intersection(getattr(effects, name)):
                    raise ValueError(f'footprint {name} {component} has no declared subsystem {name}')
        return self


def apply_thought_effect(model, result, *, row, work, effect_round=None):
    """Commit parameter-free effects to the existing parallel field.

    Serial effects remain on the one ThoughtRecord owner. Code/set/what
    effects seed dictionary-indexed knowing; native higher-order membership
    unfolds through the same conceptual decomposition that builds the pyramid.
    No LTM row or nearest-vector search is performed here.
    """
    import torch
    from Queries import _existing_row, _basis, THOUGHT_EXECUTORS
    if result is None:
        return
    descriptor = THOUGHT_EXECUTORS.get(result.semantic_id)
    if descriptor is None:
        raise ValueError('thought effect has no declared operator contract')
    kind = descriptor.effect_kind
    effect_round = effect_round or EffectRound(
        OperatorEffects(descriptor.read_scope, descriptor.write_scope), 'thought')
    if kind == 'gain':
        if 'gain' not in result.evidence:
            # A checked operation may exhaust its shared meter before it
            # produces a value. Such a result has no effect to commit.
            return
        value = float(result.evidence['gain'])
        if not 0 <= value <= 1:
            raise ValueError('expectation gain must lie in [0,1]')
        effect_round.claim(S.EXPECTATION)
        model.expectation_gain = value
        return
    if kind == 'expectation':
        discourse = getattr(getattr(model, 'symbolSpace', None), 'expectation', None)
        if discourse is not None and result.value is not None:
            # The sole predictor owns staging. Its live prior calculation is
            # numerically the same detached value retained on the thought.
            effect_round.claim(S.EXPECTATION)
            discourse.expect_next_meaning(row, record=True, refresh=True)
        return
    if getattr(model, '_anticipating_expectation_row', None) == row:
        # Anticipation can retrieve serial frames and form an estimate, but
        # cannot change the input's order-0 field or its reading attention.
        return
    if kind == 'intuition':
        effect_round.claim(S.KNOWING)
        from Intuition import write_knowing
        write_knowing(model, result.evidence, row)
        return
    if kind not in ('reference', 'frames'):
        return
    space = getattr(model, 'conceptualSpace', None)
    carrier = getattr(space, 'subspace', None)
    if carrier is None:
        return
    seeds, ideas = [], []
    if kind == 'reference':
        reference = result.evidence.get('reference')
        if reference is not None:
            seeds.append(reference)
    else:
        members = result.evidence.get('frames', ())
        for member in members:
            meaning = member.get('meaning')
            if meaning is not None:
                ideas.append(meaning)
                continue
            reference = member.get('reference')
            if reference is not None:
                seeds.append(reference)
    if not seeds and not ideas:
        return
    effect_round.claim(S.SERIAL)
    from Meaning import ConceptualMeaning
    from dataclasses import replace
    from ThoughtChunks import put
    owner = model._concept_owner()
    # Retrieved frames already carry the idea. Their occurrence references
    # are LTM addresses, not conceptual dictionary identifiers.
    for meaning in ideas:
        if not work.consume('effect_node'):
            break
        put(model.conceptualSpace.stm, meaning, row=row)
    for reference in dict.fromkeys(seeds):
        if not work.consume('effect_node'):
            break
        index = reference[1] if reference[0] == 'row' else _existing_row(owner, reference)
        value = owner.similarity_codebook.lookup_rows(
            torch.tensor(index, device=owner.similarity_codebook.W.device))
        cid = owner.concept_id_at_row(index)
        meaning = replace(ConceptualMeaning.from_description(value),
                          role_refs=(('sym', cid), None, None))
        put(model.conceptualSpace.stm, meaning, row=row)
