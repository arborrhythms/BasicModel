from pathlib import Path
import sys, importlib.util
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]


def test_audit_observes_actual_steps_and_kept_derivation(tmp_path,monkeypatch):
    output=tmp_path/'audit';output.mkdir()
    path=ROOT/'test/objective_conflicts_probe.py'
    monkeypatch.setattr(sys,'argv',[str(path),'--config','XOR_grammar','--output',str(output),'--validate-only'])
    spec=importlib.util.spec_from_file_location('observed_probe',path);module=importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit as exc:
        assert exc.code==0
    from test_mm_xor import _fresh_model
    import util
    monkeypatch.setattr(util,'TheCompileBackend','none')
    model,_,data=_fresh_model(str(ROOT/'data/XOR_grammar.xml'))
    optimizer=model.getOptimizer(lr=.01)
    try:
        raw,target=next(iter(data.data_loader(split='train',num_streams=4)))
        module.P.epoch=0
        model.runBatch(train=True,optimizer=optimizer,batchSize=4,split='train',batch_override=(model.inputSpace.prepInput(raw),model.outputSpace.prepOutput(target)))
        assert module.P.displacement_steps>0
        assert len(module.P.stability)==4
        assert list((output/'displacements').glob('*.npz'))
        assert module.P.ownership_steps>0
    finally:
        model.End()
    import shutil
    dest=Path(__file__).parent/'audit-validation'
    assert not dest.exists()
    shutil.copytree(output,dest)
