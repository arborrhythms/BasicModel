"""Observe moved benchmark decodes and already-computed free inverses."""
import hashlib,json,os
from pathlib import Path
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if item.path.name not in ('test_reconstruction_roundtrip.py','test_output_path_supervised.py'):
        yield
        return
    import recon_bench, Models, torch
    base=Path(os.environ['MOVED_OBSERVER_OUTPUT'])
    folder=base/hashlib.sha256(item.nodeid.encode()).hexdigest()[:16]
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'case.txt').write_text(item.nodeid)
    decode=recon_bench._decode_texts
    inverse=Models.BasicModel._reconstruct_sentences
    count=0
    def observed_decode(model):
        targets,decoded=decode(model)
        (folder/'decoded.json').write_text(json.dumps(dict(targets=targets,decoded=decoded),indent=2))
        return targets,decoded
    def observed_inverse(model,*args,**kwargs):
        nonlocal count
        result=inverse(model,*args,**kwargs)
        record=kwargs.get('understanding')
        if record is not None and kwargs.get('keep_ideas') and not model._sentence_training:
            bank=record.primed
            owner=model._concept_owner()
            surfaces=[[owner.word_surface_for_row(int(r)).decode('utf8') if int(r)>=0 else '' for r in rows] for rows in bank.rows.detach().cpu()]
            torch.save(dict(root=record.root.detach().cpu(),word_values=record.word_values.detach().cpu(),
                word_rows=record.word_rows.cpu(),word_valid=record.word_valid.cpu(),recovered=result[0].detach().cpu(),
                codes=bank.codes.detach().cpu(),rows=bank.rows.cpu(),weights=bank.weights.cpu(),surfaces=surfaces,
                rule_ids=record.rule_ids.cpu(),arities=record.arities.cpu(),rule_valid=record.rule_valid.cpu(),
                operand_positions=record.operand_positions.cpu(),truncated=result[3].cpu()),folder/f'inverse-{count:03}.pt')
            count+=1
        return result
    recon_bench._decode_texts=observed_decode
    Models.BasicModel._reconstruct_sentences=observed_inverse
    try:
        yield
    finally:
        recon_bench._decode_texts=decode
        Models.BasicModel._reconstruct_sentences=inverse
