import json,time
from pathlib import Path
import torch

def test_measure_open_read_cost_before_6_8():
    from test_mm_xor import _fresh_model, _PROJECT
    model,_,_= _fresh_model(str(Path(_PROJECT)/'data/XOR_grammar.xml'))
    model.eval()
    durations=[]; openings=[]
    original=model._sentence_prelude
    def timed(*args,**kwargs):
        start=time.perf_counter(); result=original(*args,**kwargs)
        openings.append(time.perf_counter()-start); return result
    model._sentence_prelude=timed
    with torch.no_grad():
        for _ in range(6):
            value=model.inputSpace.prepInput(['hello world','hello there','loving world','loving there'])
            start=time.perf_counter(); model.forward(value); durations.append(time.perf_counter()-start)
    Path(__file__).with_suffix('.json').write_text(json.dumps(dict(batch=4,words=8,warmup=1,measurements=5,forward_seconds=durations[1:],open_read_seconds=openings[1:],seed=None),indent=2)+'\n')
