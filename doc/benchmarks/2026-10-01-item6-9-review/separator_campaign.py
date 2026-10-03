"""Ten unseeded runs each: sum-only control, XOR class, XOR grammar read-back."""
from collections import Counter
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'test'),str(HERE)]
import bounded_tests as bounded
from measure import environment


def child(kind,output,config):
    from unittest.mock import patch
    import torch
    import Models
    import test_explicit_dimensions as gates
    runner=Models.ModelFactory.run
    with patch.object(Models.ModelFactory,'run',lambda ignored:runner(config)):
        model=gates._run_xor_grammar_in_process()
    data=model.inputSpace.data
    answers=torch.stack(data.reconstructed_output).reshape(-1).detach().cpu()
    targets=torch.stack(data.test_output).reshape(-1).to(answers)
    inputs=[model._bytes_to_text(t).rstrip(chr(0)) for t in data.test_input]
    y=dict(zip(inputs,answers.tolist()))
    contrast=y['hello world']+y['loving there']-y['hello there']-y['loving world']
    reconstructed=sum(Counter(a.split())==Counter((b or '').replace(chr(0),' ').split())
                      for a,b in zip(inputs,model._grammar_gate_reconstructions))
    correct=int(((answers>.5)==(targets>.5)).sum())
    mse=float((answers-targets).square().mean())
    result=dict(kind=kind,answers=answers.tolist(),targets=targets.tolist(),inputs=inputs,
        mse=mse,correct=correct,class_bar=correct==4 and mse<.05,
        read_backs=model._grammar_gate_reconstructions,unavailable=model._grammar_gate_unavailable,
        reconstructed=reconstructed,reconstruction_bar=reconstructed==4 and not any(model._grammar_gate_unavailable),
        contrast=contrast,max_deviation_from_half=float((answers-.5).abs().max()),
        half_at_float32_precision=bool(torch.allclose(answers,torch.full_like(answers,.5))),
        half_comparison='torch.allclose defaults rtol=1e-5, atol=1e-8; raw deviations retained; class bar unchanged')
    bounded.write_json(output/'measurement.json',result)


def campaign():
    out=HERE/'separator-thirty'
    out.mkdir(exist_ok=False)
    source=bounded.source_snapshot(ROOT)
    bounded.write_json(out/'source-manifest.json',dict(validated_source=source))
    old=(ROOT/'data/XOR_grammar.xml').read_text()
    rule='<rule>S = not.forward(S)</rule>\n            <rule>S = conjunction.forward(S, S)</rule>\n            <rule>S = disjunction.forward(S, S)</rule>'
    assert rule in old
    new=old.replace(rule,'<rule>S = sum.forward(S, S)</rule>')
    control=out/'XOR_grammar_sum_control.xml'
    control.write_text(new)
    (out/'control.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='data/XOR_grammar.xml',tofile='receipt/sum_control.xml')))
    bounded.write_json(out/'plan.json',dict(runs_per_kind=10,kinds=['sum','class','reconstruction'],epochs=400,
        seed=None,class_bar=dict(correct=4,mse_less_than=.05),worker_bytes=8*bounded.GIB,aggregate_bytes=24*bounded.GIB,
        max_workers=3,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    pending=[(kind,trial) for kind in ('sum','class','reconstruction') for trial in range(1,11)]
    active,done=[],[]
    start=time.monotonic()
    try:
        while pending or active:
            used=0
            for job in list(active):
                result=job['worker'].poll();used+=job['worker'].current_memory_bytes
                if result is not None:
                    bounded.write_json(job['output']/'process.json',result)
                    done.append(dict(kind=job['kind'],trial=job['trial'],process=result))
                    active.remove(job)
                    assert source==bounded.source_snapshot(ROOT)
            if used>24*bounded.GIB:raise RuntimeError('aggregate memory guard')
            while pending and len(active)<3:
                kind,trial=pending.pop(0)
                path=out/f'{kind}-{trial:02}'
                path.mkdir()
                config=control if kind=='sum' else ROOT/'data/XOR_grammar.xml'
                worker=bounded.GuardedProcess([sys.executable,str(__file__),'child',kind,str(path),str(config)],
                    cwd=ROOT,env=environment(ROOT),log_path=path/'run.log',memory_bytes=8*bounded.GIB,timeout=1800).start()
                active.append(dict(kind=kind,trial=trial,output=path,worker=worker))
            bounded.write_json(out/'progress.json',dict(done=done,pending=len(pending),seconds=time.monotonic()-start,
                active=[dict(kind=j['kind'],trial=j['trial'],pid=j['worker'].proc.pid) for j in active]))
            time.sleep(.25)
    finally:
        for j in active:j['worker'].stop(exit_code=130,reason='campaign_stopped')
    rows=[]
    for j in done:
        p=out/f"{j['kind']}-{j['trial']:02}"/'measurement.json'
        rows.append(dict(j,measurement=json.loads(p.read_text()) if p.exists() else None))
    bounded.write_json(out/'summary.json',dict(rows=rows,seconds=time.monotonic()-start,source_matched=source==bounded.source_snapshot(ROOT)))


if __name__=='__main__':
    if len(sys.argv)>1:child(sys.argv[2],Path(sys.argv[3]),sys.argv[4])
    else:campaign()
