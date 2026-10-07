"""One small real MNIST batch, with and without ergodic exploration."""
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
import torch


def test_mnist_loader_names_an_unfetched_lfs_pointer(tmp_path, monkeypatch):
    import data
    monkeypatch.setattr(data.ProjectPaths, 'DATA_DIR', str(tmp_path))
    (tmp_path/'mnist_train.csv').write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 20\n')
    with pytest.raises(RuntimeError, match='Git LFS.*mnist_train.csv'):
        data.TheData.loadMNist()


@pytest.mark.parametrize('ergodic', [False, True])
def test_real_mnist_subset_trains_one_short_batch(tmp_path, monkeypatch, ergodic):
    import data, util, Models
    root=Path(__file__).resolve().parents[1]
    # Use rows from the real downloaded files; bound only this test's subset.
    read_csv=data.pd.read_csv
    monkeypatch.setattr(data.pd,'read_csv',lambda path,*a,**kw:read_csv(path,*a,nrows=32,**kw))
    monkeypatch.setattr(util,'TheCompileBackend','none')
    monkeypatch.setenv('BASIC_AUTOLOAD','false')
    monkeypatch.setenv('BASIC_AUTOSAVE','false')
    # MNIST's scalar pixels need 784 locations through the lossless binding.
    # The historical 784 -> 20 slot geometry skips that binding; this small
    # data-subset fixture keeps all pixels and the original ten-label target.
    # Production XMLs are retained for their separate geometry review.
    path=tmp_path/'ergodic-only.xml'
    config=ET.parse(root/'data/ergodic-only.xml')
    config.find('./architecture/ergodic').text=str(ergodic).lower()
    for space in ('ConceptualSpace', 'WholeSpace'):
        for field in ('nOutput', 'nVectors'):
            config.find(f'./{space}/{field}').text='784'
    config.write(path, encoding='unicode')
    util.init_config(path=str(path),defaults_path=str(root/'data/model.xml'))
    data.TheData.load('mnist')
    model,_=Models.BaseModel.from_config(str(path),data=data.TheData)
    try:
        assert model.ergodic is ergodic
        assert len(data.TheData.train_input)==32
        assert data.TheData.inputLength==784
        optimizer=model.getOptimizer(lr=.001)
        raw,target=next(iter(data.TheData.data_loader(split='train',num_streams=2)))
        batch=model.inputSpace.prepInput(raw),model.outputSpace.prepOutput(target)
        before=[p.detach().clone() for g in optimizer.param_groups for p in g['params']]
        result,_=model.runBatch(train=True,batchSize=2,split='train',optimizer=optimizer,batch_override=batch)
        assert result is not None and torch.isfinite(result.lossOut)
        after=[p for g in optimizer.param_groups for p in g['params']]
        assert any(not torch.equal(a,b) for a,b in zip(before,after))
        assert all(torch.isfinite(p).all() for p in after)
    finally:
        model.End();model.symbolSpace.soft_reset();torch._dynamo.reset()
