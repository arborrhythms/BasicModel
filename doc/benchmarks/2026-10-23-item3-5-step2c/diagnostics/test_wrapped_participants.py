"""A wrapper cannot replace the declared participant position inside a VP."""
from ClauseRow import Composition
from ReferenceParticipants import participant_terms


def test_bind_wrapper_keeps_the_verb_participant_and_excludes_its_predicate():
    predicate = Composition(None, word_id=101)
    participant = Composition(None, word_id=102)
    phrase = Composition(None, (predicate, participant), operation='verb', head_role=1)
    root = Composition(None, (phrase,), operation='bind', head_role=1)
    eligible, _ = participant_terms(((root, False),))
    assert id(participant) in eligible
    assert id(predicate) not in eligible
