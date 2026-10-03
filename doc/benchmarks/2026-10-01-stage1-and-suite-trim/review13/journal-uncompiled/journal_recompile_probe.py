"""Read-only capture-count diagnostic on fixed word buckets, varying sentence sizes."""
from pathlib import Path
import json, sys, tempfile, time
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import pytest,torch,util,SentenceCompose
from test_compiled_word_chunk import _tiny_canonical_model
captures=[];runs=[];journals=[]
def compile_brick(function):
    def backend(graph,inputs):
        captures.append(dict(brick=function.__name__,inputs=[str(tuple(v.shape)) if torch.is_tensor(v) else str(v) for v in inputs]))
        return graph.forward
    return torch.compile(function,backend=backend,fullgraph=True)
with tempfile.TemporaryDirectory() as td, pytest.MonkeyPatch.context() as patch:
    patch.setattr(util,'TheCompileBackend','none')
    patch.setattr(SentenceCompose,'compile_word_brick',compile_brick)
    model=_tiny_canonical_model(Path(td),patch,input_width=32,word_buckets='16',
        stm_capacity=3,chooser_depth=1,training_overrides={'reconstructInLoop':False},
        architecture_overrides={'symbolicOrder':1})
    model._chart_compose_per_word=lambda:None
    original=model._run_sentence_word_bricks
    def observe(*args,**kwargs):
        journals.append(list(args[6][22].shape))
        return original(*args,**kwargs)
    patch.setattr(model,'_run_sentence_word_bricks',observe)
    try:
        for samples in (['a b','c d'],['a b c','c d a'],['a b','c d'],['a b c','c d a']):
            begin=len(captures);start=time.monotonic()
            with torch.no_grad(): model(model.inputSpace.prepInput(samples))
            runs.append(dict(samples=samples,seconds=time.monotonic()-start,
                new_captures=[item['brick'] for item in captures[begin:]],journal=journals[-1]))
            (Path(__file__).parent/'journal-recompiles.json').write_text(json.dumps(dict(runs=runs,captures=captures),indent=2)+'\n')
    finally:
        model.End();model.symbolSpace.soft_reset();torch._dynamo.reset()
print(json.dumps(runs),flush=True)
