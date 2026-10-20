import inspect,json,os,traceback
from pathlib import Path
import pytest
import torch

@pytest.fixture(autouse=True)
def inspect_cube(monkeypatch,request):
    import Layers
    original=Layers.assert_cube
    emitted=False
    def checked(value):
        nonlocal emitted
        if torch.compiler.is_compiling():
            return original(value)
        if not emitted and not bool((torch.isfinite(value)&(value>=0)&(value<=1)).all()):
            emitted=True
            folder=Path(os.environ['CUBE_DIAGNOSTIC'])/request.node.name
            folder.mkdir(parents=True,exist_ok=True)
            tensors={'invalid':value.detach().cpu()};frames=[]
            for depth,frameinfo in enumerate(inspect.stack()[1:]):
                frame=frameinfo.frame
                item={'file':frameinfo.filename,'line':frameinfo.lineno,'function':frameinfo.function}
                values={}
                for name in ('left','right','parent','reference','point','value','event','root','x','values','ref','codes','result','stack','slots'):
                    obj=frame.f_locals.get(name)
                    if torch.is_tensor(obj) and obj.layout==torch.strided and obj.numel():
                        key=f'{depth}.{name}'
                        tensors[key]=obj.detach().cpu()
                        values[name]={'shape':list(obj.shape),'min':float(obj.detach().min()),'max':float(obj.detach().max())}
                self=frame.f_locals.get('self')
                if self is not None:item['owner']=type(self).__name__
                op=frame.f_locals.get('op')
                if op is not None:
                    item['operation']=type(getattr(op,'gl',op)).__name__
                item['values']=values;frames.append(item)
            torch.save(tensors,folder/'tensors.pt')
            (folder/'trace.json').write_text(json.dumps({'seed':torch.initial_seed(),'frames':frames},indent=2)+'\n')
        return original(value)
    monkeypatch.setattr(Layers,'assert_cube',checked)
    import Language
    monkeypatch.setattr(Language,'assert_cube',checked)
