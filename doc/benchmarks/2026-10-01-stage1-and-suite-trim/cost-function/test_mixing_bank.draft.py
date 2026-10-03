"""The mixing reading supplies its own tied inverse bank at the eager boundary."""
from pathlib import Path
import torch


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
