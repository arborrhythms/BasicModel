"""The lexical reader takes the current graded table point, including after restore."""
import torch


def test_form_and_atom_reads_follow_table_movement_without_rewriting_the_store(monkeypatch):
    import WordIdentity
    def retired_codes(*args, **kwargs):
        raise AssertionError('a configured text model read the retired parts bits')
    monkeypatch.setattr(WordIdentity, 'bit_stream', retired_codes)
    from Models import BaseModel
    from data import Data
    model, _ = BaseModel.from_config('data/XOR_grammar.xml', data=Data())
    ps = model.perceptualSpace
    ps.seat_bytes([b'aba'])
    identity = ps.percept_store.identity
    identity.admit(b'aba')
    basis, table = ps.subspace.what, ps.form_links
    word = identity.word_rows[b'aba']
    atom = identity.atoms[('part', 'a')]
    rows = torch.tensor([word, atom], device=table.code.device)
    forms = basis.fixed_form_rows[rows]
    before = basis.W[rows].detach().clone()
    with torch.no_grad():
        table.code[forms[0], :64] = .314159
        table.code[forms[1], :64] = .271828
    assert torch.equal(before, basis.W[rows])
    expected = table.code[forms, :64]
    assert torch.equal(basis.lookup_rows(rows)[:, :64], expected)
    assert torch.equal(basis.getW()[rows, :64], expected)
    assert torch.equal(identity.form(b'aba'), expected[0])
    assert torch.equal(identity.atom_code(('part', 'a')), expected[1])
    assert ((expected > 0) & (expected < 1)).all()
    model.End()
