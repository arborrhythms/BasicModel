"""The mixing reading supplies its own tied inverse bank at the eager boundary."""
from pathlib import Path
import torch


def test_mixing_reconstruction_keeps_non_ascii_target_and_candidate_bytes(tmp_path, eager_reading):
    import Models
    from test_mm_xor import _fresh_model
    root = Path(Models.__file__).resolve().parents[1]
    config = tmp_path / 'XOR_utf8_reconstruction.xml'
    config.write_text((root / 'data/XOR_grammar.xml').read_text().replace(
        '<training>', '<training><reconstructInLoop>true</reconstructInLoop>', 1))
    model, _, _ = _fresh_model(str(config))
    try:
        # Fits the unchanged six-unit XOR fixture without truncation.
        surface = 'café '
        raw = torch.tensor([list(surface.encode('utf-8'))], dtype=torch.long)
        model._lex_embed_stem(raw)
        isp = model.inputSpace
        target = [bytes(v[m].tolist()) for v, m in
                  zip(isp._ar_target_word_bytes[0], isp._ar_target_word_mask[0]) if bool(m.any())]
        assert b''.join(target) == surface.encode('utf-8'), target
        candidates = [bytes(v[m].tolist()) for v, m, row in
                      zip(isp._ar_bank_bytes[0], isp._ar_bank_valid[0],
                          isp._ar_concept_lookup_rows[0]) if int(row) >= 0]
        assert candidates == [b'caf', 'é'.encode('utf-8')], candidates
        assert b''.join(candidates) == surface.strip().encode('utf-8')
        assert model._word_symbol_concept_ids() is None
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_mixing_bank_uses_byte_preserving_word_texts():
    from types import SimpleNamespace
    from Models import BasicModel
    active = torch.tensor([[True]])
    isp = SimpleNamespace(_word_active_mask=active,
        _ar_grammar_object_rows=torch.tensor([[1]]),
        _ar_grammar_object_atoms=torch.ones(1, 1, 2),
        _packed_sentence_ids=torch.zeros(1, 1, dtype=torch.long),
        _finalize_sentence_word_layout=lambda active: None)
    observed = 'café'.encode('utf-8')
    model = SimpleNamespace(word_brackets=True, reconstruct_in_loop=True, inputSpace=isp,
        _aligned_serial_word_mode=lambda: False,
        _concept_owner=lambda: SimpleNamespace(word_surface_for_row=lambda row: observed),
        perceptualSpace=SimpleNamespace(_forward_input={'word_texts': [[observed.decode('latin1')]]}))
    BasicModel._stage_mixing_reconstruction_bank(model)
    assert bytes(isp._ar_target_word_bytes[isp._ar_target_word_mask].tolist()) == observed
    assert bytes(isp._ar_bank_bytes[isp._ar_bank_valid].tolist()) == observed


def test_mixing_reconstruction_stages_objects_surfaces_and_occurrences(tmp_path, monkeypatch, eager_reading):
    import Models
    from test_mm_xor import _fresh_model
    root = Path(Models.__file__).resolve().parents[1]
    config = tmp_path / 'XOR_grammar_reconstruction.xml'
    xml = (root / 'data/XOR_grammar.xml').read_text()
    config.write_text(xml.replace('<training>', '<training><reconstructInLoop>true</reconstructInLoop>', 1))
    model, _, _ = _fresh_model(str(config))
    try:
        raw = model.inputSpace.prepInput(['hello world', 'hello there', 'loving world', 'loving there'])
        model._lex_embed_stem(raw)
        isp = model.inputSpace
        model._validate_reconstruction_bank()
        assert model._word_symbol_concept_ids() is None
        rows = isp._ar_concept_lookup_rows
        valid = rows >= 0
        assert valid.sum(1).tolist() == [2, 2, 2, 2]
        torch.testing.assert_close(rows, isp._ar_grammar_object_rows)
        torch.testing.assert_close(isp._ar_concept_lookup_atoms, isp._ar_grammar_object_atoms)
        assert isp._ar_concept_lookup_sentence_ids[valid].tolist() == [0] * 8
        assert bool((isp._ar_concept_lookup_sentence_ids[~valid] == -1).all())
        owner = model._concept_owner()
        decoded = []
        for b in range(rows.shape[0]):
            words = []
            for w in range(rows.shape[1]):
                if valid[b, w]:
                    byte = isp._ar_bank_bytes[b, w][isp._ar_bank_valid[b, w]]
                    surface = bytes(byte.tolist())
                    assert surface == owner.word_surface_for_row(int(rows[b, w]))
                    words.append(surface.decode('utf-8'))
            decoded.append(words)
        assert decoded == [['hello', 'world'], ['hello', 'there'], ['loving', 'world'], ['loving', 'there']]
        assert bool(isp._ar_target_word_mask.any())
        assert not bool(isp._ar_bank_valid[~valid].any())
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_no_compile_reconstruction_uses_the_eager_loop_body(tmp_path, monkeypatch):
    import Models
    import util
    from test_mm_xor import _fresh_model
    root = Path(Models.__file__).resolve().parents[1]
    config = tmp_path / 'XOR_grammar_eager_reconstruction.xml'
    config.write_text((root / 'data/XOR_grammar.xml').read_text().replace(
        '<training>', '<training><reconstructInLoop>true</reconstructInLoop>', 1))
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    def forbidden_capture(*args, **kwargs):
        raise AssertionError('MODEL_COMPILE=none must not capture the reconstruction loops')
    monkeypatch.setattr(torch, 'while_loop', forbidden_capture)
    model, _, _ = _fresh_model(str(config))
    try:
        with torch.no_grad():
            model(model.inputSpace.prepInput(['hello world', 'hello there', 'loving world', 'loving there']))
        assert model._recon_completed
        assert model._recon_cost.shape == (4,)
        assert bool(torch.isfinite(model._recon_cost).all())
        assert model._word_symbol_concept_ids() is None
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_eager_reconstruction_loop_keeps_captured_values_and_gradients(monkeypatch):
    import util
    from Models import _reconstruction_while_loop
    results = []
    for backend in ('none', 'eager'):
        monkeypatch.setattr(util, 'TheCompileBackend', backend)
        x = torch.tensor([1.5, -.7], requires_grad=True)
        weight = torch.tensor([.8, 1.2], requires_grad=True)
        def condition(i, value):
            return i < 3
        def body(i, value):
            return i + 1, value * weight
        _, result = _reconstruction_while_loop(condition, body, (torch.tensor(0), x))
        gradients = torch.autograd.grad(result.sum(), (x, weight))
        torch.testing.assert_close(result, x * weight ** 3)
        torch.testing.assert_close(gradients[0], weight ** 3)
        torch.testing.assert_close(gradients[1], 3 * x * weight ** 2)
        results.append((result, gradients))
    torch.testing.assert_close(results[0], results[1])
    torch._dynamo.reset()
