"""One fresh, unseeded 900-epoch MM_grammar run; no stopping threshold."""
import json,sys,time,warnings
from pathlib import Path
import torch
from test_mm_xor import _fresh_model
from util import init_device

init_device('cpu')
torch.set_num_threads(1)
out=Path(sys.argv[1]); started=time.monotonic()
m,cfg,data=_fresh_model(str(Path.cwd()/'data/MM_grammar.xml'))
optimizer=torch.optim.Adam(m.parameters(),lr=.01)
criterion=torch.nn.MSELoss()
report=dict(epochs=900,seed=None,configuration='data/MM_grammar.xml',completed_epochs=0)
try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        loader=m.inputSpace.data.data_loader(split='train',num_streams=4)
        for epoch in range(900):
            texts,answers=next(iter(loader))
            inp=m.inputSpace.prepInput(texts)
            target=m.outputSpace.prepOutput(answers)
            optimizer.zero_grad()
            _,_,output,_=m.forward(inp)
            target=target.to(output.device)
            while target.dim()<output.dim():target=target.unsqueeze(-1)
            target=target.expand_as(output)
            loss=criterion(output,target)
            if not torch.isfinite(loss):raise FloatingPointError('non-finite MM_grammar loss')
            loss.backward(); optimizer.step()
            report.update(completed_epochs=epoch+1,ending_training_mse=float(loss.detach()),
                          ending_training_predictions=output.detach().reshape(-1).tolist(),
                          targets=target.detach().reshape(-1).tolist())
            if (epoch+1)%50==0:
                report['elapsed_seconds']=time.monotonic()-started
                out.write_text(json.dumps(report,indent=2)+'\n')
        m.eval()
        with torch.no_grad():
            _,_,output,_=m.forward(inp)
            report['after_900_updates_mse']=float(criterion(output,target))
            report['after_900_updates_predictions']=output.reshape(-1).tolist()
        store=m.symbolSpace.ltm_store
        report['store_owner_present']=store is not None
        report['store_rows']=0 if store is None else len(store)
        report['definition_rows']=0 if store is None else int((store.rel_type[:len(store)]==getattr(store,'REL_DEF',-1)).sum())
        report['elapsed_seconds']=time.monotonic()-started
        out.write_text(json.dumps(report,indent=2)+'\n')
except BaseException as exc:
    report['error'] = type(exc).__name__ + ': ' + str(exc)
    raise
finally:
    report['elapsed_seconds'] = time.monotonic() - started
    out.write_text(json.dumps(report, indent=2) + '\n')
    m.End();m.symbolSpace.soft_reset()
