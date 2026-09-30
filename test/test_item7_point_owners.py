from dataclasses import replace
import torch
from Language import LanguageSpace
from test_item7_acceptance import SentenceFixture


def test_program_meaning_reads_ended_points_from_their_rows(monkeypatch):
    f = SentenceFixture(monkeypatch)
    row = f.store.write_clause(f.clause('earlier'), trust=.8)
    identity = int(f.store.row_ids[row])
    assert f.cs._csw_row_of(identity) is None
    entry = f.program(('part', 'it', 'animals'))
    refs = entry.concept_ids.clone()
    refs[0] = identity
    entry = replace(entry, reference_ids=refs)
    meaning = LanguageSpace.program_meaning(f.language, entry, f.registry)
    assert meaning is not None
    torch.testing.assert_close(meaning.roles[0], f.store.point_of_row(identity))
    torch.testing.assert_close(meaning.roles[2], f.registry._payload(('sym', int(refs[1]))))
