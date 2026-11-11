"""Primitive identities and execution interfaces, with learned concept content."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Seat:
    concept: int
    name: str
    fixed: bool
    native: object = None


class OperationSeats:
    def __init__(self, owner, names):
        from PartsOfSpeech import FIXED_OPERATIONS
        concepts = owner.prebuild_operator_concepts(names)
        self.by_name = {name: Seat(concept, name, name in FIXED_OPERATIONS)
                        for name, concept in concepts.items()}
        self.by_concept = {seat.concept: seat for seat in self.by_name.values()}

    def configure(self, rules, implementations):
        from PartsOfSpeech import operation_name
        from dataclasses import replace
        for rule, implementation in zip(rules, implementations, strict=True):
            name = operation_name(rule)
            if name is None:
                continue
            if name not in self.by_name:
                raise ValueError(f'primitive operation has no concept seat: {name}')
            seat = replace(self.by_name[name], native={**(self.by_name[name].native or {}), rule.canonical: implementation})
            self.by_name[name] = self.by_concept[seat.concept] = seat

    def __getitem__(self, name_or_concept):
        return self.by_name[name_or_concept] if isinstance(name_or_concept, str) else self.by_concept[name_or_concept]
