"""Input occurrence coordinates use the existing byte extent, not token slots."""
from pathlib import Path
import pytest
import torch

@pytest.mark.parametrize('configuration', ['MM_math','MM_add','idempotent','stream_smoke'])
def test_promoted_percepts_keep_their_original_byte_addresses(configuration, monkeypatch):
    import util, Models, Language
    from data import TheData
    root=Path(__file__).resolve().parents[1]
    path=str(root/'data'/f'{configuration}.xml')
    monkeypatch.setattr(util,'TheCompileBackend','none')
    util.init_config(path=path,defaults_path=str(root/'data/model.xml'))
    Language.TheGrammar._configured=False
    dat=dict(util.TheXMLConfig.get('architecture.data'))
    TheData.load(dat['dataset'],dat=dat)
    model,_=Models.BaseModel.from_config(path,data=TheData)
    try:
        assert model.where_registry.slices['input'][1] >= TheData.inputLength
        items,_=next(iter(TheData.data_loader(split='train',num_streams=1)))
        for _ in range(3):
            model._lex_embed_stem(model.inputSpace.prepInput(items))
            spans=model.perceptualSpace._forward_input['part_spans']
            assert int(spans.max()) <= TheData.inputLength
    finally:
        model.End();model.symbolSpace.soft_reset();torch._dynamo.reset()
