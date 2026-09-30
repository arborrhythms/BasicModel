"""TruthSet supplies plain text and provenance; grammar supplies structure."""
from pathlib import Path
import pytest
import torch
from ClauseRow import Clause
from Layers import ConceptAllocator, TernaryTruthStore
from Meaning import ConceptualMeaning


def test_truth_kind_is_rejected_by_the_schema(tmp_path):
    from util import XMLConfig
    source = Path('data/MM_query_reasoning.xml').read_text()
    import re
    source = re.sub(r' kind="[^"]*"', '', source)
    source = source.replace('<truth text=', '<truth kind="partOf" text=', 1)
    path = tmp_path / 'declared_kind.xml'
    path.write_text(source)
    with pytest.raises(ValueError, match='kind'):
        XMLConfig._parse_xml(str(path))


def test_given_truth_texts_are_plain_english():
    import xml.etree.ElementTree as ET
    query = ET.parse('data/MM_query_reasoning.xml')
    assert [row.attrib for row in query.findall('.//truth')] == [
        {'text': 'socrates is a human', 'trust': '0.9'},
        {'text': 'humans are mortal', 'trust': '0.9'}]
    for name in ('MM_qa', 'MM_ltm_consolidation_fixture',
                 'MM_ltm_consolidation_serial_fixture', 'MM_ltm_consolidation_stateful_fixture'):
        rows = ET.parse('data/' + name + '.xml').findall('.//truth')
        assert any(row.get('text') == 'a paw is part of a cat' for row in rows)
        assert all(set(row.attrib) <= {'text', 'trust'} for row in rows)


def test_supplied_trust_preserves_both_possible_grammatical_fields():
    allocator = ConceptAllocator()
    points = {allocator.new_concept(): value for value in torch.eye(4)}
    store = TernaryTruthStore(4, capacity=8)
    store.configure_clause_index(allocate=lambda point: allocator.new_concept(),
        concept_point=points.get, predicate_kind=lambda ref: 'part' if ref == 3 else 'operator')
    meaning = ConceptualMeaning(torch.eye(4)[:3], torch.ones(3, dtype=torch.bool))
    fields = [Clause(meaning, point=torch.ones(4), refs=(1, 2, -1)),
              Clause(meaning, relation='part', refs=(1, 3, 2))]
    for field in fields:
        with store.clause_assertions(trust=.8, origin=store.ORIGIN_PROVISIONED,
                                     text='given plain English') as rows:
            store.write_clause(field)
        assert len(rows) == 1
        row = rows[0]
        assert int(store.role_mask[row].sum()) == len(field.slots)
        assert store.row(row)['trust'] == pytest.approx(.8)
    assert store.ideas().tolist() == [0]
    assert store.relations(rel_type=store.REL_PARTOF).tolist() == [1]
