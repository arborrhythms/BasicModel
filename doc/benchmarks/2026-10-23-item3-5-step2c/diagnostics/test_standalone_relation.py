"""A surface head does not turn a retained relation into a standalone object."""
import torch
from ClauseRow import Clause, Composition
from Meaning import ConceptualMeaning
from test_reference_close import fixture
from test_medium_term_memory import snapshot, unchanged


def test_surface_wrapped_relation_does_not_add_a_standalone_object():
    resolver, store, atoms = fixture()
    relation = Composition(atoms[0], (Composition(atoms[1]), Composition(atoms[2])), operation='equal')
    wrapped = Composition(atoms[0], (Composition(atoms[3]), relation), operation='surface', head_role=2, order=1)
    clause = Clause(ConceptualMeaning(atoms[:3], torch.ones(3,dtype=torch.bool)),
        point=None, order=1, relation='part', composition=wrapped)
    before = snapshot(resolver)
    plan = resolver.preview(clause, ())
    assert plan['items'] == []
    unchanged(resolver, before)
