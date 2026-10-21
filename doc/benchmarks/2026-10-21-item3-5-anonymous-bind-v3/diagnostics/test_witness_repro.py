"""Reproduce the occurrence witness error without changing the frozen source."""
from test_bind_thought import execute, native, occurrence, site


def test_occurrence_bind_witness_is_a_locatable_address(monkeypatch):
    f = native(monkeypatch)
    target, _ = occurrence(f, f.atoms[0], f.atoms[4])
    address = f.identities[target]
    result = execute(f, site(f, f.atoms[0]*0, predicate=f.atoms[4]))
    assert result.value.role_refs[0] == address
    assert result.evidence['witnesses']
    assert all(f.store.location(witness) is not None
               for witness in result.evidence['witnesses'])
