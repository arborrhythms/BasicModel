"""Every WholeSpace is a property basis under the September 28 decision."""
from pathlib import Path
import pytest
import torch


def test_configuration_without_property_element_binds_at_first_sentence(tmp_path, monkeypatch):
    from test_meronomy_ladder import _build_ladder_variant
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _build_ladder_variant(tmp_path, 'properties_by_definition', [
        ('<architecture>', '<architecture><ltmConsolidation>true</ltmConsolidation>')])
    try:
        for ws in model.wholeSpaces:
            assert not hasattr(ws, 'property_basis')
            assert ws.subspace.what.primitive_properties is not None
            assert not hasattr(ws, '_ws_pos_to_row')
        model._tensor_peer_while_eager = True
        model._install_unit_span_fn()
        model.reconstruct_in_loop = False
        model.loss.reconstruction_scale = 0.
        model(model.inputSpace.prepInput(['cat']))
        assert model._concept_owner().word_concepts('cat')
        assert model._sentence_fields[0][0] is not None
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_retired_property_element_is_a_load_error_naming_the_decision(tmp_path):
    from util import XMLConfig
    source = Path('data/MM_ladder.xml').read_text()
    source = source.replace('<propertyBasis>true</propertyBasis>', '')
    source = source.replace('<WholeSpace>', '<WholeSpace><propertyBasis>false</propertyBasis>')
    path = tmp_path / 'retired_property.xml'
    path.write_text(source)
    with pytest.raises(ValueError, match='11.4'):
        XMLConfig._parse_xml(str(path))


def test_dictionary_checkpoint_warns_and_keeps_prior_properties(tmp_path, monkeypatch):
    from test_meronomy_ladder import _build_ladder_variant
    from checkpoint_migrations import is_legacy_wholespace_dense_key
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _build_ladder_variant(tmp_path, 'old_dictionary_checkpoint', [])
    try:
        priors = [{name: value.detach().clone() for name, value in ws.subspace.what.state_dict().items()}
                  for ws in model.wholeSpaces]
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        for name, value in state.items():
            if is_legacy_wholespace_dense_key(name):
                state[name] = torch.full_like(value, 71)
        checkpoint = tmp_path / 'dictionary.ckpt'
        torch.save({'state_dict': state,
                    'vocab_extras': {'well_known_atoms': {'cat': 12},
                                     'ws_taxonomy_extras': {'taxonomy': {7: [2, 3]}}},
                    'structural_extras': {'version': 1, 'whole_spaces': {
                        '0': {'attributes': {'_word_whole_ss': {'cat': 12}}}}}}, checkpoint)
        with pytest.warns(UserWarning, match='11.4'):
            assert model.load_weights(checkpoint)
        for ws, prior in zip(model.wholeSpaces, priors):
            assert not hasattr(ws, '_word_whole_ss')
            assert not hasattr(ws, 'taxonomy')
            assert not hasattr(ws, '_ws_pos_to_row')
            for name, value in ws.subspace.what.state_dict().items():
                torch.testing.assert_close(value, prior[name], rtol=0, atol=0)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
