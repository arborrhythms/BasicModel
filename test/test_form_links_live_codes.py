"""What the lexical reader takes for the parts half of a code: the set of a
form's parts, and the table of forms' point only where ``LIVE_TABLE_CODES``."""
import torch


def _aba(model):
    """``aba`` admitted: its word row and the row of its part ``a``, their
    rows of the table of forms, and the codebook's stored codes."""
    ps = model.perceptualSpace
    ps.seat_bytes([b'aba'])
    identity = ps.percept_store.identity
    identity.admit(b'aba')
    basis, table = ps.subspace.what, ps.form_links
    word = identity.word_rows[b'aba']
    atom = identity.atoms[('part', 'a')]
    rows = torch.tensor([word, atom], device=table.code.device)
    forms = basis.fixed_form_rows[rows]
    return identity, basis, table, rows, forms, basis.W[rows].detach().clone()


def _move(table, forms):
    with torch.no_grad():
        table.code[forms[0], :64] = .314159
        table.code[forms[1], :64] = .271828


def test_form_and_atom_reads_are_the_set_of_parts_and_do_not_follow_the_table():
    """As built (``WordIdentity.LIVE_TABLE_CODES`` false): the first Pi reads
    a form as the set of its parts, so a part's code is its fixed bits, a
    form's is the OR of its parts', the store holds those, and a movement of
    the table of forms changes no read."""
    from Models import BaseModel
    from WordIdentity import WordIdentity
    from data import Data
    assert WordIdentity.LIVE_TABLE_CODES is False
    model, _ = BaseModel.from_config('data/XOR_grammar.xml', data=Data())
    try:
        identity, basis, table, rows, forms, before = _aba(model)
        assert not basis.form_codes_live
        parts = identity.words[b'aba']
        bits = torch.stack([identity.atom_code(atom) for atom in parts])
        # Codes of sets: each part's bits are ones and zeros, the form their OR.
        assert ((bits == 0) | (bits == 1)).all() and bits.sum(-1).eq(identity.ones).all()
        assert torch.equal(identity.form(b'aba'), bits.amax(0))
        expected = torch.stack((identity.form(b'aba'), identity.atom_code(('part', 'a')))).to(before)
        assert torch.equal(before[:, :64], expected)
        reads = [basis.lookup_rows(rows)[:, :64].detach().clone(), basis.getW()[rows, :64].detach().clone()]
        _move(table, forms)
        assert torch.equal(before, basis.W[rows])
        assert torch.equal(basis.lookup_rows(rows)[:, :64], reads[0]) and torch.equal(reads[0], expected)
        assert torch.equal(basis.getW()[rows, :64], reads[1]) and torch.equal(reads[1], expected)
        assert torch.equal(identity.form(b'aba').to(before), expected[0])
        assert torch.equal(identity.atom_code(('part', 'a')).to(before), expected[1])
    finally:
        model.End()


def test_form_and_atom_reads_follow_table_movement_without_rewriting_the_store(monkeypatch):
    """With ``LIVE_TABLE_CODES`` set for a model built here (item 1.5,
    ``53825210c5``): the reader takes the current graded table point, and no
    read is of the parts' fixed bits."""
    import WordIdentity
    monkeypatch.setattr(WordIdentity.WordIdentity, 'LIVE_TABLE_CODES', True)
    def retired_codes(*args, **kwargs):
        raise AssertionError('a configured text model read the retired parts bits')
    monkeypatch.setattr(WordIdentity, 'bit_stream', retired_codes)
    from Models import BaseModel
    from data import Data
    model, _ = BaseModel.from_config('data/XOR_grammar.xml', data=Data())
    try:
        identity, basis, table, rows, forms, before = _aba(model)
        assert basis.form_codes_live
        _move(table, forms)
        assert torch.equal(before, basis.W[rows])
        expected = table.code[forms, :64]
        assert torch.equal(basis.lookup_rows(rows)[:, :64], expected)
        assert torch.equal(basis.getW()[rows, :64], expected)
        assert torch.equal(identity.form(b'aba'), expected[0])
        assert torch.equal(identity.atom_code(('part', 'a')), expected[1])
        assert ((expected > 0) & (expected < 1)).all()
    finally:
        model.End()
